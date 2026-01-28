// Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::avx::butterflies::AvxButterfly;
use crate::avx::butterflies::shared::{boring_avx_butterfly, boring_avx_butterfly2};
use crate::avx::mixed::{AvxStoreD, ColumnButterfly13d};
use crate::avx::rotate::AvxRotate;
use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps};
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly13d {
    direction: FftDirection,
    bf13: ColumnButterfly13d,
}

impl AvxButterfly13d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }
    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf13: ColumnButterfly13d::new(fft_direction),
        }
    }
}

boring_avx_butterfly2!(AvxButterfly13d, f64, 13);

impl AvxButterfly13d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows = [AvxStoreD::zero(); 13];
        for i in 0..13 {
            rows[i] = AvxStoreD::from_complex(chunk.index(i));
        }
        rows = self.bf13.exec(rows);
        for i in 0..13 {
            rows[i].write_lo(chunk.slice_from_mut(i..));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run2<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows = [AvxStoreD::zero(); 13];
        unsafe {
            const HI_HI: i32 = 0b0011_0001;
            const LO_LO: i32 = 0b0010_0000;
            for i in 0..6 {
                let q0 = _mm256_loadu_pd(chunk.slice_from(i * 2..).as_ptr().cast());
                let q1 = _mm256_loadu_pd(chunk.slice_from(i * 2 + 13..).as_ptr().cast());
                rows[i * 2] = AvxStoreD::raw(_mm256_permute2f128_pd::<LO_LO>(q0, q1));
                rows[i * 2 + 1] = AvxStoreD::raw(_mm256_permute2f128_pd::<HI_HI>(q0, q1));
            }

            let q0 = _mm_loadu_pd(chunk.slice_from(12..).as_ptr().cast());
            let q1 = _mm_loadu_pd(chunk.slice_from(12 + 13..).as_ptr().cast());
            rows[12] = AvxStoreD::raw(_mm256_create_pd(q0, q1));

            rows = self.bf13.exec(rows);
            for i in 0..6 {
                let r0 = rows[i * 2];
                let r1 = rows[i * 2 + 1];
                let new_row0 = AvxStoreD::raw(_mm256_permute2f128_pd::<LO_LO>(r0.v, r1.v));
                let new_row1 = AvxStoreD::raw(_mm256_permute2f128_pd::<HI_HI>(r0.v, r1.v));
                new_row0.write(chunk.slice_from_mut(i * 2..));
                new_row1.write(chunk.slice_from_mut(i * 2 + 13..));
            }

            let r0 = rows[12];
            r0.write_lo(chunk.slice_from_mut(12..));
            r0.write_hi(chunk.slice_from_mut(12 + 13..));
        }
    }
}

pub(crate) struct AvxButterfly13f {
    direction: FftDirection,
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
    twiddle3: Complex<f32>,
    twiddle4: Complex<f32>,
    twiddle5: Complex<f32>,
    twiddle6: Complex<f32>,
    rotate: AvxRotate<f32>,
}

boring_avx_butterfly!(AvxButterfly13f, f32, 13);

impl AvxButterfly13f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 13, fft_direction),
            twiddle2: compute_twiddle(2, 13, fft_direction),
            twiddle3: compute_twiddle(3, 13, fft_direction),
            twiddle4: compute_twiddle(4, 13, fft_direction),
            twiddle5: compute_twiddle(5, 13, fft_direction),
            twiddle6: compute_twiddle(6, 13, fft_direction),
            rotate: unsafe { AvxRotate::new(FftDirection::Inverse) },
        }
    }
}

impl AvxButterfly13f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        unsafe {
            let u0u1u2u3 = _mm256_loadu_ps(chunk.slice_from(0..).as_ptr().cast());
            let u4u5u6u7 = _mm256_loadu_ps(chunk.slice_from(4..).as_ptr().cast());
            let u7u8u9u10 = _mm256_loadu_ps(chunk.slice_from(7..).as_ptr().cast());
            let u9u10u11u12 = _mm256_loadu_ps(chunk.slice_from(9..).as_ptr().cast());

            let u0 = _mm256_castps256_ps128(u0u1u2u3);
            let u11u12 = _mm256_extractf128_ps::<1>(u9u10u11u12);
            let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
            let u9u10 = _mm256_extractf128_ps::<1>(u7u8u9u10);
            let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
            let u7u8 = _mm256_castps256_ps128(u7u8u9u10);
            let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);

            let q = self.kernel_f32([
                u0,
                _mm_unpackhi_ps64(u0, u0),
                u2u3,
                _mm_unpackhi_ps64(u2u3, u2u3),
                u4u5,
                _mm_unpackhi_ps64(u4u5, u4u5),
                u6u7,
                _mm_unpackhi_ps64(u6u7, u6u7),
                _mm_unpackhi_ps64(u7u8, u7u8),
                u9u10,
                _mm_unpackhi_ps64(u9u10, u9u10),
                u11u12,
                _mm_unpackhi_ps64(u11u12, u11u12),
            ]);

            let y0000 =
                _mm256_create_ps(_mm_unpacklo_ps64(q[0], q[1]), _mm_unpacklo_ps64(q[2], q[3]));
            let y0001 =
                _mm256_create_ps(_mm_unpacklo_ps64(q[4], q[5]), _mm_unpacklo_ps64(q[6], q[7]));
            let y0002 = _mm256_create_ps(
                _mm_unpacklo_ps64(q[7], q[8]),
                _mm_unpacklo_ps64(q[9], q[10]),
            );
            let y0003 = _mm256_create_ps(
                _mm_unpacklo_ps64(q[9], q[10]),
                _mm_unpacklo_ps64(q[11], q[12]),
            );

            _mm256_storeu_ps(chunk.slice_from_mut(0..).as_mut_ptr().cast(), y0000);
            _mm256_storeu_ps(chunk.slice_from_mut(4..).as_mut_ptr().cast(), y0001);
            _mm256_storeu_ps(chunk.slice_from_mut(7..).as_mut_ptr().cast(), y0002);
            _mm256_storeu_ps(chunk.slice_from_mut(9..).as_mut_ptr().cast(), y0003);
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn kernel_f32(&self, v: [__m128; 13]) -> [__m128; 13] {
        let y00 = v[0];
        let (x1p12, x1m12) = AvxButterfly::butterfly2_f32_m128(v[1], v[12]);
        let x1m12 = self.rotate.rotate_m128(x1m12);
        let y00 = _mm_add_ps(y00, x1p12);
        let (x2p11, x2m11) = AvxButterfly::butterfly2_f32_m128(v[2], v[11]);
        let x2m11 = self.rotate.rotate_m128(x2m11);
        let y00 = _mm_add_ps(y00, x2p11);
        let (x3p10, x3m10) = AvxButterfly::butterfly2_f32_m128(v[3], v[10]);
        let x3m10 = self.rotate.rotate_m128(x3m10);
        let y00 = _mm_add_ps(y00, x3p10);
        let (x4p9, x4m9) = AvxButterfly::butterfly2_f32_m128(v[4], v[9]);
        let x4m9 = self.rotate.rotate_m128(x4m9);
        let y00 = _mm_add_ps(y00, x4p9);
        let (x5p8, x5m8) = AvxButterfly::butterfly2_f32_m128(v[5], v[8]);
        let x5m8 = self.rotate.rotate_m128(x5m8);
        let y00 = _mm_add_ps(y00, x5p8);
        let (x6p7, x6m7) = AvxButterfly::butterfly2_f32_m128(v[6], v[7]);
        let x6m7 = self.rotate.rotate_m128(x6m7);
        let y00 = _mm_add_ps(y00, x6p7);

        let m0112a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle1.re), v[0]);
        let m0112a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle2.re), x2p11, m0112a);
        let m0112a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle3.re), x3p10, m0112a);
        let m0112a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle4.re), m0112a);
        let m0112a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle5.re), m0112a);
        let m0112a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle6.re), m0112a);
        let m0112b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle1.im));
        let m0112b = _mm_fmadd_ps(x2m11, _mm_set1_ps(self.twiddle2.im), m0112b);
        let m0112b = _mm_fmadd_ps(x3m10, _mm_set1_ps(self.twiddle3.im), m0112b);
        let m0112b = _mm_fmadd_ps(x4m9, _mm_set1_ps(self.twiddle4.im), m0112b);
        let m0112b = _mm_fmadd_ps(x5m8, _mm_set1_ps(self.twiddle5.im), m0112b);
        let m0112b = _mm_fmadd_ps(x6m7, _mm_set1_ps(self.twiddle6.im), m0112b);
        let (y01, y12) = AvxButterfly::butterfly2_f32_m128(m0112a, m0112b);

        let m0211a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle2.re), v[0]);
        let m0211a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle4.re), m0211a);
        let m0211a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle6.re), m0211a);
        let m0211a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle5.re), m0211a);
        let m0211a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle3.re), m0211a);
        let m0211a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle1.re), m0211a);
        let m0211b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle2.im));
        let m0211b = _mm_fmadd_ps(x2m11, _mm_set1_ps(self.twiddle4.im), m0211b);
        let m0211b = _mm_fmadd_ps(x3m10, _mm_set1_ps(self.twiddle6.im), m0211b);
        let m0211b = _mm_fnmadd_ps(x4m9, _mm_set1_ps(self.twiddle5.im), m0211b);
        let m0211b = _mm_fnmadd_ps(x5m8, _mm_set1_ps(self.twiddle3.im), m0211b);
        let m0211b = _mm_fnmadd_ps(x6m7, _mm_set1_ps(self.twiddle1.im), m0211b);
        let (y02, y11) = AvxButterfly::butterfly2_f32_m128(m0211a, m0211b);

        let m0310a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle3.re), v[0]);
        let m0310a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle6.re), m0310a);
        let m0310a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle4.re), m0310a);
        let m0310a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle1.re), m0310a);
        let m0310a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle2.re), m0310a);
        let m0310a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle5.re), m0310a);
        let m0310b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle3.im));
        let m0310b = _mm_fmadd_ps(x2m11, _mm_set1_ps(self.twiddle6.im), m0310b);
        let m0310b = _mm_fnmadd_ps(x3m10, _mm_set1_ps(self.twiddle4.im), m0310b);
        let m0310b = _mm_fnmadd_ps(x4m9, _mm_set1_ps(self.twiddle1.im), m0310b);
        let m0310b = _mm_fmadd_ps(x5m8, _mm_set1_ps(self.twiddle2.im), m0310b);
        let m0310b = _mm_fmadd_ps(x6m7, _mm_set1_ps(self.twiddle5.im), m0310b);
        let (y03, y10) = AvxButterfly::butterfly2_f32_m128(m0310a, m0310b);

        let m0409a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle4.re), v[0]);
        let m0409a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle5.re), m0409a);
        let m0409a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle1.re), m0409a);
        let m0409a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle3.re), m0409a);
        let m0409a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle6.re), m0409a);
        let m0409a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle2.re), m0409a);
        let m0409b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle4.im));
        let m0409b = _mm_fnmadd_ps(x2m11, _mm_set1_ps(self.twiddle5.im), m0409b);
        let m0409b = _mm_fnmadd_ps(x3m10, _mm_set1_ps(self.twiddle1.im), m0409b);
        let m0409b = _mm_fmadd_ps(x4m9, _mm_set1_ps(self.twiddle3.im), m0409b);
        let m0409b = _mm_fnmadd_ps(x5m8, _mm_set1_ps(self.twiddle6.im), m0409b);
        let m0409b = _mm_fnmadd_ps(x6m7, _mm_set1_ps(self.twiddle2.im), m0409b);
        let (y04, y09) = AvxButterfly::butterfly2_f32_m128(m0409a, m0409b);

        let m0508a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle5.re), v[0]);
        let m0508a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle3.re), m0508a);
        let m0508a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle2.re), m0508a);
        let m0508a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle6.re), m0508a);
        let m0508a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle1.re), m0508a);
        let m0508a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle4.re), m0508a);
        let m0508b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle5.im));
        let m0508b = _mm_fnmadd_ps(x2m11, _mm_set1_ps(self.twiddle3.im), m0508b);
        let m0508b = _mm_fmadd_ps(x3m10, _mm_set1_ps(self.twiddle2.im), m0508b);
        let m0508b = _mm_fnmadd_ps(x4m9, _mm_set1_ps(self.twiddle6.im), m0508b);
        let m0508b = _mm_fnmadd_ps(x5m8, _mm_set1_ps(self.twiddle1.im), m0508b);
        let m0508b = _mm_fmadd_ps(x6m7, _mm_set1_ps(self.twiddle4.im), m0508b);
        let (y05, y08) = AvxButterfly::butterfly2_f32_m128(m0508a, m0508b);

        let m0607a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle6.re), v[0]);
        let m0607a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle1.re), m0607a);
        let m0607a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle5.re), m0607a);
        let m0607a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle2.re), m0607a);
        let m0607a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle4.re), m0607a);
        let m0607a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle3.re), m0607a);
        let m0607b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle6.im));
        let m0607b = _mm_fnmadd_ps(x2m11, _mm_set1_ps(self.twiddle1.im), m0607b);
        let m0607b = _mm_fmadd_ps(x3m10, _mm_set1_ps(self.twiddle5.im), m0607b);
        let m0607b = _mm_fnmadd_ps(x4m9, _mm_set1_ps(self.twiddle2.im), m0607b);
        let m0607b = _mm_fmadd_ps(x5m8, _mm_set1_ps(self.twiddle4.im), m0607b);
        let m0607b = _mm_fnmadd_ps(x6m7, _mm_set1_ps(self.twiddle3.im), m0607b);
        let (y06, y07) = AvxButterfly::butterfly2_f32_m128(m0607a, m0607b);
        [
            y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12,
        ]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly13, f32, AvxButterfly13f, 13, 1e-5);
    test_avx_butterfly!(test_avx_butterfly13_f64, f64, AvxButterfly13d, 13, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly13, f32, AvxButterfly13f, 13, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly13_f64, f64, AvxButterfly13d, 13, 1e-7);
}
