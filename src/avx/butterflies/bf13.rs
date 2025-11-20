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

use crate::avx::butterflies::{
    AvxButterfly, shift_load2d, shift_load2dd, shift_load4, shift_load8, shift_store2d,
    shift_store2dd, shift_store4, shift_store8,
};
use crate::avx::rotate::AvxRotate;
use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;
use std::sync::Arc;

pub(crate) struct AvxButterfly13<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    rotate: AvxRotate<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly13<T>
where
    f64: AsPrimitive<T>,
{
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

impl AvxButterfly13<f64> {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn kernel_f64(&self, v: [__m128d; 13]) -> [__m128d; 13] {
        let y00 = v[0];
        let (x1p12, x1m12) = AvxButterfly::butterfly2_f64_m128(v[1], v[12]);
        let x1m12 = self.rotate.rotate_m128d(x1m12);
        let y00 = _mm_add_pd(y00, x1p12);
        let (x2p11, x2m11) = AvxButterfly::butterfly2_f64_m128(v[2], v[11]);
        let x2m11 = self.rotate.rotate_m128d(x2m11);
        let y00 = _mm_add_pd(y00, x2p11);
        let (x3p10, x3m10) = AvxButterfly::butterfly2_f64_m128(v[3], v[10]);
        let x3m10 = self.rotate.rotate_m128d(x3m10);
        let y00 = _mm_add_pd(y00, x3p10);
        let (x4p9, x4m9) = AvxButterfly::butterfly2_f64_m128(v[4], v[9]);
        let x4m9 = self.rotate.rotate_m128d(x4m9);
        let y00 = _mm_add_pd(y00, x4p9);
        let (x5p8, x5m8) = AvxButterfly::butterfly2_f64_m128(v[5], v[8]);
        let x5m8 = self.rotate.rotate_m128d(x5m8);
        let y00 = _mm_add_pd(y00, x5p8);
        let (x6p7, x6m7) = AvxButterfly::butterfly2_f64_m128(v[6], v[7]);
        let x6m7 = self.rotate.rotate_m128d(x6m7);
        let y00 = _mm_add_pd(y00, x6p7);

        let m0112a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle1.re), v[0]);
        let m0112a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle2.re), x2p11, m0112a);
        let m0112a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle3.re), x3p10, m0112a);
        let m0112a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle4.re), m0112a);
        let m0112a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle5.re), m0112a);
        let m0112a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle6.re), m0112a);
        let m0112b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle1.im));
        let m0112b = _mm_fmadd_pd(x2m11, _mm_set1_pd(self.twiddle2.im), m0112b);
        let m0112b = _mm_fmadd_pd(x3m10, _mm_set1_pd(self.twiddle3.im), m0112b);
        let m0112b = _mm_fmadd_pd(x4m9, _mm_set1_pd(self.twiddle4.im), m0112b);
        let m0112b = _mm_fmadd_pd(x5m8, _mm_set1_pd(self.twiddle5.im), m0112b);
        let m0112b = _mm_fmadd_pd(x6m7, _mm_set1_pd(self.twiddle6.im), m0112b);
        let (y01, y12) = AvxButterfly::butterfly2_f64_m128(m0112a, m0112b);

        let m0211a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle2.re), v[0]);
        let m0211a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle4.re), m0211a);
        let m0211a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle6.re), m0211a);
        let m0211a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle5.re), m0211a);
        let m0211a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle3.re), m0211a);
        let m0211a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle1.re), m0211a);
        let m0211b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle2.im));
        let m0211b = _mm_fmadd_pd(x2m11, _mm_set1_pd(self.twiddle4.im), m0211b);
        let m0211b = _mm_fmadd_pd(x3m10, _mm_set1_pd(self.twiddle6.im), m0211b);
        let m0211b = _mm_fnmadd_pd(x4m9, _mm_set1_pd(self.twiddle5.im), m0211b);
        let m0211b = _mm_fnmadd_pd(x5m8, _mm_set1_pd(self.twiddle3.im), m0211b);
        let m0211b = _mm_fnmadd_pd(x6m7, _mm_set1_pd(self.twiddle1.im), m0211b);
        let (y02, y11) = AvxButterfly::butterfly2_f64_m128(m0211a, m0211b);

        let m0310a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle3.re), v[0]);
        let m0310a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle6.re), m0310a);
        let m0310a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle4.re), m0310a);
        let m0310a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle1.re), m0310a);
        let m0310a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle2.re), m0310a);
        let m0310a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle5.re), m0310a);
        let m0310b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle3.im));
        let m0310b = _mm_fmadd_pd(x2m11, _mm_set1_pd(self.twiddle6.im), m0310b);
        let m0310b = _mm_fnmadd_pd(x3m10, _mm_set1_pd(self.twiddle4.im), m0310b);
        let m0310b = _mm_fnmadd_pd(x4m9, _mm_set1_pd(self.twiddle1.im), m0310b);
        let m0310b = _mm_fmadd_pd(x5m8, _mm_set1_pd(self.twiddle2.im), m0310b);
        let m0310b = _mm_fmadd_pd(x6m7, _mm_set1_pd(self.twiddle5.im), m0310b);
        let (y03, y10) = AvxButterfly::butterfly2_f64_m128(m0310a, m0310b);

        let m0409a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle4.re), v[0]);
        let m0409a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle5.re), m0409a);
        let m0409a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle1.re), m0409a);
        let m0409a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle3.re), m0409a);
        let m0409a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle6.re), m0409a);
        let m0409a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle2.re), m0409a);
        let m0409b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle4.im));
        let m0409b = _mm_fnmadd_pd(x2m11, _mm_set1_pd(self.twiddle5.im), m0409b);
        let m0409b = _mm_fnmadd_pd(x3m10, _mm_set1_pd(self.twiddle1.im), m0409b);
        let m0409b = _mm_fmadd_pd(x4m9, _mm_set1_pd(self.twiddle3.im), m0409b);
        let m0409b = _mm_fnmadd_pd(x5m8, _mm_set1_pd(self.twiddle6.im), m0409b);
        let m0409b = _mm_fnmadd_pd(x6m7, _mm_set1_pd(self.twiddle2.im), m0409b);
        let (y04, y09) = AvxButterfly::butterfly2_f64_m128(m0409a, m0409b);

        let m0508a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle5.re), v[0]);
        let m0508a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle3.re), m0508a);
        let m0508a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle2.re), m0508a);
        let m0508a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle6.re), m0508a);
        let m0508a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle1.re), m0508a);
        let m0508a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle4.re), m0508a);
        let m0508b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle5.im));
        let m0508b = _mm_fnmadd_pd(x2m11, _mm_set1_pd(self.twiddle3.im), m0508b);
        let m0508b = _mm_fmadd_pd(x3m10, _mm_set1_pd(self.twiddle2.im), m0508b);
        let m0508b = _mm_fnmadd_pd(x4m9, _mm_set1_pd(self.twiddle6.im), m0508b);
        let m0508b = _mm_fnmadd_pd(x5m8, _mm_set1_pd(self.twiddle1.im), m0508b);
        let m0508b = _mm_fmadd_pd(x6m7, _mm_set1_pd(self.twiddle4.im), m0508b);
        let (y05, y08) = AvxButterfly::butterfly2_f64_m128(m0508a, m0508b);

        let m0607a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle6.re), v[0]);
        let m0607a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle1.re), m0607a);
        let m0607a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle5.re), m0607a);
        let m0607a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle2.re), m0607a);
        let m0607a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle4.re), m0607a);
        let m0607a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle3.re), m0607a);
        let m0607b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle6.im));
        let m0607b = _mm_fnmadd_pd(x2m11, _mm_set1_pd(self.twiddle1.im), m0607b);
        let m0607b = _mm_fmadd_pd(x3m10, _mm_set1_pd(self.twiddle5.im), m0607b);
        let m0607b = _mm_fnmadd_pd(x4m9, _mm_set1_pd(self.twiddle2.im), m0607b);
        let m0607b = _mm_fmadd_pd(x5m8, _mm_set1_pd(self.twiddle4.im), m0607b);
        let m0607b = _mm_fnmadd_pd(x6m7, _mm_set1_pd(self.twiddle3.im), m0607b);
        let (y06, y07) = AvxButterfly::butterfly2_f64_m128(m0607a, m0607b);

        [
            y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12,
        ]
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn kernel_f64_m256(&self, v: [__m256d; 13]) -> [__m256d; 13] {
        let y00 = v[0];
        let (x1p12, x1m12) = AvxButterfly::butterfly2_f64(v[1], v[12]);
        let x1m12 = self.rotate.rotate_m256d(x1m12);
        let y00 = _mm256_add_pd(y00, x1p12);
        let (x2p11, x2m11) = AvxButterfly::butterfly2_f64(v[2], v[11]);
        let x2m11 = self.rotate.rotate_m256d(x2m11);
        let y00 = _mm256_add_pd(y00, x2p11);
        let (x3p10, x3m10) = AvxButterfly::butterfly2_f64(v[3], v[10]);
        let x3m10 = self.rotate.rotate_m256d(x3m10);
        let y00 = _mm256_add_pd(y00, x3p10);
        let (x4p9, x4m9) = AvxButterfly::butterfly2_f64(v[4], v[9]);
        let x4m9 = self.rotate.rotate_m256d(x4m9);
        let y00 = _mm256_add_pd(y00, x4p9);
        let (x5p8, x5m8) = AvxButterfly::butterfly2_f64(v[5], v[8]);
        let x5m8 = self.rotate.rotate_m256d(x5m8);
        let y00 = _mm256_add_pd(y00, x5p8);
        let (x6p7, x6m7) = AvxButterfly::butterfly2_f64(v[6], v[7]);
        let x6m7 = self.rotate.rotate_m256d(x6m7);
        let y00 = _mm256_add_pd(y00, x6p7);

        let m0112a = _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle1.re), v[0]);
        let m0112a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle2.re), x2p11, m0112a);
        let m0112a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle3.re), x3p10, m0112a);
        let m0112a = _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle4.re), m0112a);
        let m0112a = _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle5.re), m0112a);
        let m0112a = _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle6.re), m0112a);
        let m0112b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle1.im));
        let m0112b = _mm256_fmadd_pd(x2m11, _mm256_set1_pd(self.twiddle2.im), m0112b);
        let m0112b = _mm256_fmadd_pd(x3m10, _mm256_set1_pd(self.twiddle3.im), m0112b);
        let m0112b = _mm256_fmadd_pd(x4m9, _mm256_set1_pd(self.twiddle4.im), m0112b);
        let m0112b = _mm256_fmadd_pd(x5m8, _mm256_set1_pd(self.twiddle5.im), m0112b);
        let m0112b = _mm256_fmadd_pd(x6m7, _mm256_set1_pd(self.twiddle6.im), m0112b);
        let (y01, y12) = AvxButterfly::butterfly2_f64(m0112a, m0112b);

        let m0211a = _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle2.re), v[0]);
        let m0211a = _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle4.re), m0211a);
        let m0211a = _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle6.re), m0211a);
        let m0211a = _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle5.re), m0211a);
        let m0211a = _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle3.re), m0211a);
        let m0211a = _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle1.re), m0211a);
        let m0211b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle2.im));
        let m0211b = _mm256_fmadd_pd(x2m11, _mm256_set1_pd(self.twiddle4.im), m0211b);
        let m0211b = _mm256_fmadd_pd(x3m10, _mm256_set1_pd(self.twiddle6.im), m0211b);
        let m0211b = _mm256_fnmadd_pd(x4m9, _mm256_set1_pd(self.twiddle5.im), m0211b);
        let m0211b = _mm256_fnmadd_pd(x5m8, _mm256_set1_pd(self.twiddle3.im), m0211b);
        let m0211b = _mm256_fnmadd_pd(x6m7, _mm256_set1_pd(self.twiddle1.im), m0211b);
        let (y02, y11) = AvxButterfly::butterfly2_f64(m0211a, m0211b);

        let m0310a = _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle3.re), v[0]);
        let m0310a = _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle6.re), m0310a);
        let m0310a = _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle4.re), m0310a);
        let m0310a = _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle1.re), m0310a);
        let m0310a = _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle2.re), m0310a);
        let m0310a = _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle5.re), m0310a);
        let m0310b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle3.im));
        let m0310b = _mm256_fmadd_pd(x2m11, _mm256_set1_pd(self.twiddle6.im), m0310b);
        let m0310b = _mm256_fnmadd_pd(x3m10, _mm256_set1_pd(self.twiddle4.im), m0310b);
        let m0310b = _mm256_fnmadd_pd(x4m9, _mm256_set1_pd(self.twiddle1.im), m0310b);
        let m0310b = _mm256_fmadd_pd(x5m8, _mm256_set1_pd(self.twiddle2.im), m0310b);
        let m0310b = _mm256_fmadd_pd(x6m7, _mm256_set1_pd(self.twiddle5.im), m0310b);
        let (y03, y10) = AvxButterfly::butterfly2_f64(m0310a, m0310b);

        let m0409a = _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle4.re), v[0]);
        let m0409a = _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle5.re), m0409a);
        let m0409a = _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle1.re), m0409a);
        let m0409a = _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle3.re), m0409a);
        let m0409a = _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle6.re), m0409a);
        let m0409a = _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle2.re), m0409a);
        let m0409b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle4.im));
        let m0409b = _mm256_fnmadd_pd(x2m11, _mm256_set1_pd(self.twiddle5.im), m0409b);
        let m0409b = _mm256_fnmadd_pd(x3m10, _mm256_set1_pd(self.twiddle1.im), m0409b);
        let m0409b = _mm256_fmadd_pd(x4m9, _mm256_set1_pd(self.twiddle3.im), m0409b);
        let m0409b = _mm256_fnmadd_pd(x5m8, _mm256_set1_pd(self.twiddle6.im), m0409b);
        let m0409b = _mm256_fnmadd_pd(x6m7, _mm256_set1_pd(self.twiddle2.im), m0409b);
        let (y04, y09) = AvxButterfly::butterfly2_f64(m0409a, m0409b);

        let m0508a = _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle5.re), v[0]);
        let m0508a = _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle3.re), m0508a);
        let m0508a = _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle2.re), m0508a);
        let m0508a = _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle6.re), m0508a);
        let m0508a = _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle1.re), m0508a);
        let m0508a = _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle4.re), m0508a);
        let m0508b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle5.im));
        let m0508b = _mm256_fnmadd_pd(x2m11, _mm256_set1_pd(self.twiddle3.im), m0508b);
        let m0508b = _mm256_fmadd_pd(x3m10, _mm256_set1_pd(self.twiddle2.im), m0508b);
        let m0508b = _mm256_fnmadd_pd(x4m9, _mm256_set1_pd(self.twiddle6.im), m0508b);
        let m0508b = _mm256_fnmadd_pd(x5m8, _mm256_set1_pd(self.twiddle1.im), m0508b);
        let m0508b = _mm256_fmadd_pd(x6m7, _mm256_set1_pd(self.twiddle4.im), m0508b);
        let (y05, y08) = AvxButterfly::butterfly2_f64(m0508a, m0508b);

        let m0607a = _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle6.re), v[0]);
        let m0607a = _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle1.re), m0607a);
        let m0607a = _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle5.re), m0607a);
        let m0607a = _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle2.re), m0607a);
        let m0607a = _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle4.re), m0607a);
        let m0607a = _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle3.re), m0607a);
        let m0607b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle6.im));
        let m0607b = _mm256_fnmadd_pd(x2m11, _mm256_set1_pd(self.twiddle1.im), m0607b);
        let m0607b = _mm256_fmadd_pd(x3m10, _mm256_set1_pd(self.twiddle5.im), m0607b);
        let m0607b = _mm256_fnmadd_pd(x4m9, _mm256_set1_pd(self.twiddle2.im), m0607b);
        let m0607b = _mm256_fmadd_pd(x5m8, _mm256_set1_pd(self.twiddle4.im), m0607b);
        let m0607b = _mm256_fnmadd_pd(x6m7, _mm256_set1_pd(self.twiddle3.im), m0607b);
        let (y06, y07) = AvxButterfly::butterfly2_f64(m0607a, m0607b);

        [
            y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12,
        ]
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 13 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(26) {
                let (u0, u1) = shift_load2dd!(chunk, 13, 0);
                let (u2, u3) = shift_load2dd!(chunk, 13, 2);
                let (u4, u5) = shift_load2dd!(chunk, 13, 4);
                let (u6, u7) = shift_load2dd!(chunk, 13, 6);
                let (u8, u9) = shift_load2dd!(chunk, 13, 8);
                let (u10, u11) = shift_load2dd!(chunk, 13, 10);
                let u12 = shift_load2d!(chunk, 13, 12);

                let q =
                    self.kernel_f64_m256([u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12]);

                shift_store2dd!(chunk, 13, 0, q[0], q[1]);
                shift_store2dd!(chunk, 13, 2, q[2], q[3]);
                shift_store2dd!(chunk, 13, 4, q[4], q[5]);
                shift_store2dd!(chunk, 13, 6, q[6], q[7]);
                shift_store2dd!(chunk, 13, 8, q[8], q[9]);
                shift_store2dd!(chunk, 13, 10, q[10], q[11]);
                shift_store2d!(chunk, 13, 12, q[12]);
            }

            let rem = in_place.chunks_exact_mut(26).into_remainder();

            for chunk in rem.chunks_exact_mut(13) {
                let u0u1 = _mm256_loadu_pd(chunk.as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u12 = _mm_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());

                let q = self.kernel_f64([
                    _mm256_castpd256_pd128(u0u1),
                    _mm256_extractf128_pd::<1>(u0u1),
                    _mm256_castpd256_pd128(u2u3),
                    _mm256_extractf128_pd::<1>(u2u3),
                    _mm256_castpd256_pd128(u4u5),
                    _mm256_extractf128_pd::<1>(u4u5),
                    _mm256_castpd256_pd128(u6u7),
                    _mm256_extractf128_pd::<1>(u6u7),
                    _mm256_castpd256_pd128(u8u9),
                    _mm256_extractf128_pd::<1>(u8u9),
                    _mm256_castpd256_pd128(u10u11),
                    _mm256_extractf128_pd::<1>(u10u11),
                    u12,
                ]);

                _mm256_storeu_pd(chunk.as_mut_ptr().cast(), _mm256_create_pd(q[0], q[1]));
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(q[2], q[3]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(q[4], q[5]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(q[6], q[7]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(q[8], q[9]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_create_pd(q[10], q[11]),
                );
                _mm_storeu_pd(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), q[12]);
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_out_of_place_f64(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe {
            if src.len() % 13 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 13 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            for (dst, src) in dst.chunks_exact_mut(26).zip(src.chunks_exact(26)) {
                let (u0, u1) = shift_load2dd!(src, 13, 0);
                let (u2, u3) = shift_load2dd!(src, 13, 2);
                let (u4, u5) = shift_load2dd!(src, 13, 4);
                let (u6, u7) = shift_load2dd!(src, 13, 6);
                let (u8, u9) = shift_load2dd!(src, 13, 8);
                let (u10, u11) = shift_load2dd!(src, 13, 10);
                let u12 = shift_load2d!(src, 13, 12);

                let q =
                    self.kernel_f64_m256([u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12]);

                shift_store2dd!(dst, 13, 0, q[0], q[1]);
                shift_store2dd!(dst, 13, 2, q[2], q[3]);
                shift_store2dd!(dst, 13, 4, q[4], q[5]);
                shift_store2dd!(dst, 13, 6, q[6], q[7]);
                shift_store2dd!(dst, 13, 8, q[8], q[9]);
                shift_store2dd!(dst, 13, 10, q[10], q[11]);
                shift_store2d!(dst, 13, 12, q[12]);
            }

            let rem_dst = dst.chunks_exact_mut(26).into_remainder();
            let rem_src = src.chunks_exact(26).remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(13).zip(rem_src.chunks_exact(13)) {
                let u0u1 = _mm256_loadu_pd(src.as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(src.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(src.get_unchecked(10..).as_ptr().cast());
                let u12 = _mm_loadu_pd(src.get_unchecked(12..).as_ptr().cast());

                let q = self.kernel_f64([
                    _mm256_castpd256_pd128(u0u1),
                    _mm256_extractf128_pd::<1>(u0u1),
                    _mm256_castpd256_pd128(u2u3),
                    _mm256_extractf128_pd::<1>(u2u3),
                    _mm256_castpd256_pd128(u4u5),
                    _mm256_extractf128_pd::<1>(u4u5),
                    _mm256_castpd256_pd128(u6u7),
                    _mm256_extractf128_pd::<1>(u6u7),
                    _mm256_castpd256_pd128(u8u9),
                    _mm256_extractf128_pd::<1>(u8u9),
                    _mm256_castpd256_pd128(u10u11),
                    _mm256_extractf128_pd::<1>(u10u11),
                    u12,
                ]);

                _mm256_storeu_pd(dst.as_mut_ptr().cast(), _mm256_create_pd(q[0], q[1]));
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(q[2], q[3]),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(q[4], q[5]),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(q[6], q[7]),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(q[8], q[9]),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_create_pd(q[10], q[11]),
                );
                _mm_storeu_pd(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), q[12]);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly13<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly13<f64> {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f64> for AvxButterfly13<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        13
    }
}

impl AvxButterfly13<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 13 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(52) {
                let (u0, u1, u2, u3) = shift_load8!(chunk, 13, 0);
                let (u4, u5, u6, u7) = shift_load8!(chunk, 13, 4);
                let (_, u8, u9, u10) = shift_load8!(chunk, 13, 7);
                let (_, _, u11, u12) = shift_load8!(chunk, 13, 9);

                let q =
                    self.kernel_f32_m256([u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12]);

                shift_store8!(chunk, 13, 0, q[0], q[1], q[2], q[3]);
                shift_store8!(chunk, 13, 4, q[4], q[5], q[6], q[7]);
                shift_store8!(chunk, 13, 7, q[7], q[8], q[9], q[10]);
                shift_store8!(chunk, 13, 9, q[9], q[10], q[11], q[12]);
            }

            let rem = in_place.chunks_exact_mut(52).into_remainder();

            for chunk in rem.chunks_exact_mut(26) {
                let (u0, u1, u2, u3) = shift_load4!(chunk, 13, 0);
                let (u4, u5, u6, u7) = shift_load4!(chunk, 13, 4);
                let (_, u8, u9, u10) = shift_load4!(chunk, 13, 7);
                let (_, _, u11, u12) = shift_load4!(chunk, 13, 9);

                let q = self.kernel_f32([u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12]);

                shift_store4!(chunk, 13, 0, q[0], q[1], q[2], q[3]);
                shift_store4!(chunk, 13, 4, q[4], q[5], q[6], q[7]);
                shift_store4!(chunk, 13, 7, q[7], q[8], q[9], q[10]);
                shift_store4!(chunk, 13, 9, q[9], q[10], q[11], q[12]);
            }

            let rem = rem.chunks_exact_mut(26).into_remainder();

            for chunk in rem.chunks_exact_mut(13) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u7u8u9u10 = _mm256_loadu_ps(chunk.get_unchecked(7..).as_ptr().cast());
                let u9u10u11u12 = _mm256_loadu_ps(chunk.get_unchecked(9..).as_ptr().cast());

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

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0000);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y0001);
                _mm256_storeu_ps(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y0002);
                _mm256_storeu_ps(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y0003);
            }
        }
        Ok(())
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn kernel_f32(&self, v: [__m128; 13]) -> [__m128; 13] {
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

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn kernel_f32_m256(&self, v: [__m256; 13]) -> [__m256; 13] {
        let y00 = v[0];
        let (x1p12, x1m12) = AvxButterfly::butterfly2_f32(v[1], v[12]);
        let x1m12 = self.rotate.rotate_m256(x1m12);
        let y00 = _mm256_add_ps(y00, x1p12);
        let (x2p11, x2m11) = AvxButterfly::butterfly2_f32(v[2], v[11]);
        let x2m11 = self.rotate.rotate_m256(x2m11);
        let y00 = _mm256_add_ps(y00, x2p11);
        let (x3p10, x3m10) = AvxButterfly::butterfly2_f32(v[3], v[10]);
        let x3m10 = self.rotate.rotate_m256(x3m10);
        let y00 = _mm256_add_ps(y00, x3p10);
        let (x4p9, x4m9) = AvxButterfly::butterfly2_f32(v[4], v[9]);
        let x4m9 = self.rotate.rotate_m256(x4m9);
        let y00 = _mm256_add_ps(y00, x4p9);
        let (x5p8, x5m8) = AvxButterfly::butterfly2_f32(v[5], v[8]);
        let x5m8 = self.rotate.rotate_m256(x5m8);
        let y00 = _mm256_add_ps(y00, x5p8);
        let (x6p7, x6m7) = AvxButterfly::butterfly2_f32(v[6], v[7]);
        let x6m7 = self.rotate.rotate_m256(x6m7);
        let y00 = _mm256_add_ps(y00, x6p7);

        let m0112a = _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle1.re), v[0]);
        let m0112a = _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle2.re), x2p11, m0112a);
        let m0112a = _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle3.re), x3p10, m0112a);
        let m0112a = _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle4.re), m0112a);
        let m0112a = _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle5.re), m0112a);
        let m0112a = _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle6.re), m0112a);
        let m0112b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle1.im));
        let m0112b = _mm256_fmadd_ps(x2m11, _mm256_set1_ps(self.twiddle2.im), m0112b);
        let m0112b = _mm256_fmadd_ps(x3m10, _mm256_set1_ps(self.twiddle3.im), m0112b);
        let m0112b = _mm256_fmadd_ps(x4m9, _mm256_set1_ps(self.twiddle4.im), m0112b);
        let m0112b = _mm256_fmadd_ps(x5m8, _mm256_set1_ps(self.twiddle5.im), m0112b);
        let m0112b = _mm256_fmadd_ps(x6m7, _mm256_set1_ps(self.twiddle6.im), m0112b);
        let (y01, y12) = AvxButterfly::butterfly2_f32(m0112a, m0112b);

        let m0211a = _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle2.re), v[0]);
        let m0211a = _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle4.re), m0211a);
        let m0211a = _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle6.re), m0211a);
        let m0211a = _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle5.re), m0211a);
        let m0211a = _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle3.re), m0211a);
        let m0211a = _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle1.re), m0211a);
        let m0211b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle2.im));
        let m0211b = _mm256_fmadd_ps(x2m11, _mm256_set1_ps(self.twiddle4.im), m0211b);
        let m0211b = _mm256_fmadd_ps(x3m10, _mm256_set1_ps(self.twiddle6.im), m0211b);
        let m0211b = _mm256_fnmadd_ps(x4m9, _mm256_set1_ps(self.twiddle5.im), m0211b);
        let m0211b = _mm256_fnmadd_ps(x5m8, _mm256_set1_ps(self.twiddle3.im), m0211b);
        let m0211b = _mm256_fnmadd_ps(x6m7, _mm256_set1_ps(self.twiddle1.im), m0211b);
        let (y02, y11) = AvxButterfly::butterfly2_f32(m0211a, m0211b);

        let m0310a = _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle3.re), v[0]);
        let m0310a = _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle6.re), m0310a);
        let m0310a = _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle4.re), m0310a);
        let m0310a = _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle1.re), m0310a);
        let m0310a = _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle2.re), m0310a);
        let m0310a = _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle5.re), m0310a);
        let m0310b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle3.im));
        let m0310b = _mm256_fmadd_ps(x2m11, _mm256_set1_ps(self.twiddle6.im), m0310b);
        let m0310b = _mm256_fnmadd_ps(x3m10, _mm256_set1_ps(self.twiddle4.im), m0310b);
        let m0310b = _mm256_fnmadd_ps(x4m9, _mm256_set1_ps(self.twiddle1.im), m0310b);
        let m0310b = _mm256_fmadd_ps(x5m8, _mm256_set1_ps(self.twiddle2.im), m0310b);
        let m0310b = _mm256_fmadd_ps(x6m7, _mm256_set1_ps(self.twiddle5.im), m0310b);
        let (y03, y10) = AvxButterfly::butterfly2_f32(m0310a, m0310b);

        let m0409a = _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle4.re), v[0]);
        let m0409a = _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle5.re), m0409a);
        let m0409a = _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle1.re), m0409a);
        let m0409a = _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle3.re), m0409a);
        let m0409a = _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle6.re), m0409a);
        let m0409a = _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle2.re), m0409a);
        let m0409b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle4.im));
        let m0409b = _mm256_fnmadd_ps(x2m11, _mm256_set1_ps(self.twiddle5.im), m0409b);
        let m0409b = _mm256_fnmadd_ps(x3m10, _mm256_set1_ps(self.twiddle1.im), m0409b);
        let m0409b = _mm256_fmadd_ps(x4m9, _mm256_set1_ps(self.twiddle3.im), m0409b);
        let m0409b = _mm256_fnmadd_ps(x5m8, _mm256_set1_ps(self.twiddle6.im), m0409b);
        let m0409b = _mm256_fnmadd_ps(x6m7, _mm256_set1_ps(self.twiddle2.im), m0409b);
        let (y04, y09) = AvxButterfly::butterfly2_f32(m0409a, m0409b);

        let m0508a = _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle5.re), v[0]);
        let m0508a = _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle3.re), m0508a);
        let m0508a = _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle2.re), m0508a);
        let m0508a = _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle6.re), m0508a);
        let m0508a = _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle1.re), m0508a);
        let m0508a = _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle4.re), m0508a);
        let m0508b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle5.im));
        let m0508b = _mm256_fnmadd_ps(x2m11, _mm256_set1_ps(self.twiddle3.im), m0508b);
        let m0508b = _mm256_fmadd_ps(x3m10, _mm256_set1_ps(self.twiddle2.im), m0508b);
        let m0508b = _mm256_fnmadd_ps(x4m9, _mm256_set1_ps(self.twiddle6.im), m0508b);
        let m0508b = _mm256_fnmadd_ps(x5m8, _mm256_set1_ps(self.twiddle1.im), m0508b);
        let m0508b = _mm256_fmadd_ps(x6m7, _mm256_set1_ps(self.twiddle4.im), m0508b);
        let (y05, y08) = AvxButterfly::butterfly2_f32(m0508a, m0508b);

        let m0607a = _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle6.re), v[0]);
        let m0607a = _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle1.re), m0607a);
        let m0607a = _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle5.re), m0607a);
        let m0607a = _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle2.re), m0607a);
        let m0607a = _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle4.re), m0607a);
        let m0607a = _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle3.re), m0607a);
        let m0607b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle6.im));
        let m0607b = _mm256_fnmadd_ps(x2m11, _mm256_set1_ps(self.twiddle1.im), m0607b);
        let m0607b = _mm256_fmadd_ps(x3m10, _mm256_set1_ps(self.twiddle5.im), m0607b);
        let m0607b = _mm256_fnmadd_ps(x4m9, _mm256_set1_ps(self.twiddle2.im), m0607b);
        let m0607b = _mm256_fmadd_ps(x5m8, _mm256_set1_ps(self.twiddle4.im), m0607b);
        let m0607b = _mm256_fnmadd_ps(x6m7, _mm256_set1_ps(self.twiddle3.im), m0607b);
        let (y06, y07) = AvxButterfly::butterfly2_f32(m0607a, m0607b);
        [
            y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12,
        ]
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_out_of_place_f32(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe {
            if src.len() % 13 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 13 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            for (dst, src) in dst.chunks_exact_mut(52).zip(src.chunks_exact(52)) {
                let (u0, u1, u2, u3) = shift_load8!(src, 13, 0);
                let (u4, u5, u6, u7) = shift_load8!(src, 13, 4);
                let (_, u8, u9, u10) = shift_load8!(src, 13, 7);
                let (_, _, u11, u12) = shift_load8!(src, 13, 9);

                let q =
                    self.kernel_f32_m256([u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12]);

                shift_store8!(dst, 13, 0, q[0], q[1], q[2], q[3]);
                shift_store8!(dst, 13, 4, q[4], q[5], q[6], q[7]);
                shift_store8!(dst, 13, 7, q[7], q[8], q[9], q[10]);
                shift_store8!(dst, 13, 9, q[9], q[10], q[11], q[12]);
            }

            let rem_src = src.chunks_exact(52).remainder();
            let rem_dst = dst.chunks_exact_mut(52).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(26).zip(rem_src.chunks_exact(26)) {
                let (u0, u1, u2, u3) = shift_load4!(src, 13, 0);
                let (u4, u5, u6, u7) = shift_load4!(src, 13, 4);
                let (_, u8, u9, u10) = shift_load4!(src, 13, 7);
                let (_, _, u11, u12) = shift_load4!(src, 13, 9);

                let q = self.kernel_f32([u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12]);

                shift_store4!(dst, 13, 0, q[0], q[1], q[2], q[3]);
                shift_store4!(dst, 13, 4, q[4], q[5], q[6], q[7]);
                shift_store4!(dst, 13, 7, q[7], q[8], q[9], q[10]);
                shift_store4!(dst, 13, 9, q[9], q[10], q[11], q[12]);
            }

            let rem_src = rem_src.chunks_exact(26).remainder();
            let rem_dst = rem_dst.chunks_exact_mut(26).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(13).zip(rem_src.chunks_exact(13)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let u7u8u9u10 = _mm256_loadu_ps(src.get_unchecked(7..).as_ptr().cast());
                let u9u10u11u12 = _mm256_loadu_ps(src.get_unchecked(9..).as_ptr().cast());

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

                _mm256_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0000);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y0001);
                _mm256_storeu_ps(dst.get_unchecked_mut(7..).as_mut_ptr().cast(), y0002);
                _mm256_storeu_ps(dst.get_unchecked_mut(9..).as_mut_ptr().cast(), y0003);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly13<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly13<f32> {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for AvxButterfly13<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        13
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly13, f32, AvxButterfly13, 13, 1e-5);
    test_avx_butterfly!(test_avx_butterfly13_f64, f64, AvxButterfly13, 13, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly13, f32, AvxButterfly13, 13, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly13_f64, f64, AvxButterfly13, 13, 1e-7);
}
