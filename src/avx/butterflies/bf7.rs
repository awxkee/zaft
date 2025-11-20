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
use crate::avx::rotate::AvxRotate;
use crate::avx::util::{
    _m128s_store_f32x2, _m128s_storeh_f32x2, _mm_unpackhi_ps64, _mm_unpackhilo_ps64,
    _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps, _mm256_permute4x64_ps,
    _mm256_unpackhi_ps64, _mm256_unpacklo_ps64, shuffle,
};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;
use std::sync::Arc;

pub(crate) struct AvxButterfly7<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly7<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 7, fft_direction),
            twiddle2: compute_twiddle(2, 7, fft_direction),
            twiddle3: compute_twiddle(3, 7, fft_direction),
        }
    }
}

impl AvxButterfly7<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 7 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let rotate = AvxRotate::<f64>::new(FftDirection::Inverse);

            for chunk in in_place.chunks_exact_mut(14) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u0_2 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u1u2_2 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u3u4_2 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u5u6_2 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());

                const LO_HI: i32 = 0b0011_0000;
                const HI_LO: i32 = 0b0010_0001;
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u0 = _mm256_permute2f128_pd::<LO_HI>(u0u1, u6u0_2);
                let u1 = _mm256_permute2f128_pd::<HI_LO>(u0u1, u1u2_2);
                let u2 = _mm256_permute2f128_pd::<LO_HI>(u2u3, u1u2_2);
                let u3 = _mm256_permute2f128_pd::<HI_LO>(u2u3, u3u4_2);
                let u4 = _mm256_permute2f128_pd::<LO_HI>(u4u5, u3u4_2);
                let u5 = _mm256_permute2f128_pd::<HI_LO>(u4u5, u5u6_2);
                let u6 = _mm256_permute2f128_pd::<LO_HI>(u6u0_2, u5u6_2);

                let (x1p6, x1m6) = AvxButterfly::butterfly2_f64(u1, u6);
                let x1m6 = rotate.rotate_m256d(x1m6);
                let y00 = _mm256_add_pd(u0, x1p6);
                let (x2p5, x2m5) = AvxButterfly::butterfly2_f64(u2, u5);
                let x2m5 = rotate.rotate_m256d(x2m5);
                let y00 = _mm256_add_pd(y00, x2p5);
                let (x3p4, x3m4) = AvxButterfly::butterfly2_f64(u3, u4);
                let x3m4 = rotate.rotate_m256d(x3m4);
                let y00 = _mm256_add_pd(y00, x3p4);

                let m0106a = _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle1.re), u0);
                let m0106a = _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle2.re), m0106a);
                let m0106a = _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle3.re), m0106a);
                let m0106b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle1.im));
                let m0106b = _mm256_fmadd_pd(x2m5, _mm256_set1_pd(self.twiddle2.im), m0106b);
                let m0106b = _mm256_fmadd_pd(x3m4, _mm256_set1_pd(self.twiddle3.im), m0106b);
                let (y01, y06) = AvxButterfly::butterfly2_f64(m0106a, m0106b);

                let m0205a = _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle2.re), u0);
                let m0205a = _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle3.re), m0205a);
                let m0205a = _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle1.re), m0205a);
                let m0205b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle2.im));
                let m0205b = _mm256_fnmadd_pd(x2m5, _mm256_set1_pd(self.twiddle3.im), m0205b);
                let m0205b = _mm256_fnmadd_pd(x3m4, _mm256_set1_pd(self.twiddle1.im), m0205b);
                let (y02, y05) = AvxButterfly::butterfly2_f64(m0205a, m0205b);

                let m0304a = _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle3.re), u0);
                let m0304a = _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle1.re), m0304a);
                let m0304a = _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle2.re), m0304a);
                let m0304b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle3.im));
                let m0304b = _mm256_fnmadd_pd(x2m5, _mm256_set1_pd(self.twiddle1.im), m0304b);
                let m0304b = _mm256_fmadd_pd(x3m4, _mm256_set1_pd(self.twiddle2.im), m0304b);
                let (y03, y04) = AvxButterfly::butterfly2_f64(m0304a, m0304b);

                let y0y1 = _mm256_permute2f128_pd::<LO_LO>(y00, y01);
                let y2y3 = _mm256_permute2f128_pd::<LO_LO>(y02, y03);
                let y4y5 = _mm256_permute2f128_pd::<LO_LO>(y04, y05);
                let y6y0_2 = _mm256_permute2f128_pd::<LO_HI>(y06, y00);
                let y1y0_2 = _mm256_permute2f128_pd::<HI_HI>(y01, y02);
                let y3y4_2 = _mm256_permute2f128_pd::<HI_HI>(y03, y04);
                let y5y6_2 = _mm256_permute2f128_pd::<HI_HI>(y05, y06);

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2y3);
                _mm256_storeu_pd(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4y5);
                _mm256_storeu_pd(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y6y0_2);
                _mm256_storeu_pd(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y1y0_2);
                _mm256_storeu_pd(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y3y4_2);
                _mm256_storeu_pd(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y5y6_2);
            }

            let rem = in_place.chunks_exact_mut(14).into_remainder();

            let tw1tw2r = _mm256_setr_pd(
                self.twiddle1.re,
                self.twiddle1.re,
                self.twiddle2.re,
                self.twiddle2.re,
            );
            let tw2tw3r = _mm256_setr_pd(
                self.twiddle2.re,
                self.twiddle2.re,
                self.twiddle3.re,
                self.twiddle3.re,
            );
            let tw3tw1r = _mm256_setr_pd(
                self.twiddle3.re,
                self.twiddle3.re,
                self.twiddle1.re,
                self.twiddle1.re,
            );

            let tw1tw2i = _mm256_setr_pd(
                self.twiddle1.im,
                self.twiddle1.im,
                self.twiddle2.im,
                self.twiddle2.im,
            );
            let tw2tw3i = _mm256_setr_pd(
                self.twiddle2.im,
                self.twiddle2.im,
                -self.twiddle3.im,
                -self.twiddle3.im,
            );
            let tw3tw1i = _mm256_setr_pd(
                self.twiddle3.im,
                self.twiddle3.im,
                -self.twiddle1.im,
                -self.twiddle1.im,
            );

            for chunk in rem.chunks_exact_mut(7) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6 = _mm_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(u0u1);
                let u1 = _mm256_extractf128_pd::<1>(u0u1);
                let u2 = _mm256_castpd256_pd128(u2u3);
                let u3 = _mm256_extractf128_pd::<1>(u2u3);
                let u4 = _mm256_castpd256_pd128(u4u5);
                let u5 = _mm256_extractf128_pd::<1>(u4u5);

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let (x1p6x2p5, x1m6x2m5) = AvxButterfly::butterfly2_f64(
                    _mm256_create_pd(u1, u2),
                    _mm256_create_pd(u6, u5),
                );
                let x1m6x2m5 = rotate.rotate_m256d(x1m6x2m5);

                let x1p6 = _mm256_castpd256_pd128(x1p6x2p5);
                let x2p5 = _mm256_extractf128_pd::<1>(x1p6x2p5);
                let x1m6 = _mm256_castpd256_pd128(x1m6x2m5);
                let x2m5 = _mm256_extractf128_pd::<1>(x1m6x2m5);

                let y00 = _mm_add_pd(
                    _mm_add_pd(u0, _mm256_castpd256_pd128(x1p6x2p5)),
                    _mm256_extractf128_pd::<1>(x1p6x2p5),
                );
                let (x3p4, x3m4) = AvxButterfly::butterfly2_f64_m128(u3, u4);
                let x3m4 = rotate.rotate_m128d(x3m4);
                let y00 = _mm_add_pd(y00, x3p4);

                let x1p6d = _mm256_permute2f128_pd::<LO_LO>(x1p6x2p5, x1p6x2p5);
                let x2p5d = _mm256_permute2f128_pd::<HI_HI>(x1p6x2p5, x1p6x2p5);
                let x3p4d = _mm256_create_pd(x3p4, x3p4);

                let x1m6d = _mm256_permute2f128_pd::<LO_LO>(x1m6x2m5, x1m6x2m5);
                let x2m5d = _mm256_permute2f128_pd::<HI_HI>(x1m6x2m5, x1m6x2m5);
                let x3m4d = _mm256_create_pd(x3m4, x3m4);

                let m0106am0205a = _mm256_fmadd_pd(x1p6d, tw1tw2r, _mm256_create_pd(u0, u0));
                let m0106am0205a = _mm256_fmadd_pd(x2p5d, tw2tw3r, m0106am0205a);
                let m0106am0205a = _mm256_fmadd_pd(x3p4d, tw3tw1r, m0106am0205a);
                let m0106bm0205b = _mm256_mul_pd(x1m6d, tw1tw2i);
                let m0106bm0205b = _mm256_fmadd_pd(x2m5d, tw2tw3i, m0106bm0205b);
                let m0106bm0205b = _mm256_fmadd_pd(x3m4d, tw3tw1i, m0106bm0205b);
                let (y01y02, y06y05) = AvxButterfly::butterfly2_f64(m0106am0205a, m0106bm0205b);

                let m0304a = _mm_fmadd_pd(x1p6, _mm256_castpd256_pd128(tw3tw1r), u0);
                let m0304a = _mm_fmadd_pd(x2p5, _mm256_castpd256_pd128(tw1tw2r), m0304a);
                let m0304a = _mm_fmadd_pd(x3p4, _mm256_castpd256_pd128(tw2tw3r), m0304a);
                let m0304b = _mm_mul_pd(x1m6, _mm256_castpd256_pd128(tw3tw1i));
                let m0304b = _mm_fnmadd_pd(x2m5, _mm256_castpd256_pd128(tw1tw2i), m0304b);
                let m0304b = _mm_fmadd_pd(x3m4, _mm256_castpd256_pd128(tw2tw3i), m0304b);
                let (y03, y04) = AvxButterfly::butterfly2_f64_m128(m0304a, m0304b);

                let y3y4 = _mm256_create_pd(y03, y04);

                const HI_LO: i32 = 0b0010_0001;

                _mm_storeu_pd(chunk.as_mut_ptr().cast(), y00);
                _mm256_storeu_pd(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y01y02);
                _mm256_storeu_pd(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3y4);
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_LO>(y06y05, y06y05),
                );
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
            if src.len() % 7 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 7 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            let rotate = AvxRotate::<f64>::new(FftDirection::Inverse);

            for (dst, src) in dst.chunks_exact_mut(14).zip(src.chunks_exact(14)) {
                let u0u1 = _mm256_loadu_pd(src.as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let u6u0_2 = _mm256_loadu_pd(src.get_unchecked(6..).as_ptr().cast());
                let u1u2_2 = _mm256_loadu_pd(src.get_unchecked(8..).as_ptr().cast());
                let u3u4_2 = _mm256_loadu_pd(src.get_unchecked(10..).as_ptr().cast());
                let u5u6_2 = _mm256_loadu_pd(src.get_unchecked(12..).as_ptr().cast());

                const LO_HI: i32 = 0b0011_0000;
                const HI_LO: i32 = 0b0010_0001;
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u0 = _mm256_permute2f128_pd::<LO_HI>(u0u1, u6u0_2);
                let u1 = _mm256_permute2f128_pd::<HI_LO>(u0u1, u1u2_2);
                let u2 = _mm256_permute2f128_pd::<LO_HI>(u2u3, u1u2_2);
                let u3 = _mm256_permute2f128_pd::<HI_LO>(u2u3, u3u4_2);
                let u4 = _mm256_permute2f128_pd::<LO_HI>(u4u5, u3u4_2);
                let u5 = _mm256_permute2f128_pd::<HI_LO>(u4u5, u5u6_2);
                let u6 = _mm256_permute2f128_pd::<LO_HI>(u6u0_2, u5u6_2);

                let (x1p6, x1m6) = AvxButterfly::butterfly2_f64(u1, u6);
                let x1m6 = rotate.rotate_m256d(x1m6);
                let y00 = _mm256_add_pd(u0, x1p6);
                let (x2p5, x2m5) = AvxButterfly::butterfly2_f64(u2, u5);
                let x2m5 = rotate.rotate_m256d(x2m5);
                let y00 = _mm256_add_pd(y00, x2p5);
                let (x3p4, x3m4) = AvxButterfly::butterfly2_f64(u3, u4);
                let x3m4 = rotate.rotate_m256d(x3m4);
                let y00 = _mm256_add_pd(y00, x3p4);

                let m0106a = _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle1.re), u0);
                let m0106a = _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle2.re), m0106a);
                let m0106a = _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle3.re), m0106a);
                let m0106b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle1.im));
                let m0106b = _mm256_fmadd_pd(x2m5, _mm256_set1_pd(self.twiddle2.im), m0106b);
                let m0106b = _mm256_fmadd_pd(x3m4, _mm256_set1_pd(self.twiddle3.im), m0106b);
                let (y01, y06) = AvxButterfly::butterfly2_f64(m0106a, m0106b);

                let m0205a = _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle2.re), u0);
                let m0205a = _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle3.re), m0205a);
                let m0205a = _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle1.re), m0205a);
                let m0205b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle2.im));
                let m0205b = _mm256_fnmadd_pd(x2m5, _mm256_set1_pd(self.twiddle3.im), m0205b);
                let m0205b = _mm256_fnmadd_pd(x3m4, _mm256_set1_pd(self.twiddle1.im), m0205b);
                let (y02, y05) = AvxButterfly::butterfly2_f64(m0205a, m0205b);

                let m0304a = _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle3.re), u0);
                let m0304a = _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle1.re), m0304a);
                let m0304a = _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle2.re), m0304a);
                let m0304b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle3.im));
                let m0304b = _mm256_fnmadd_pd(x2m5, _mm256_set1_pd(self.twiddle1.im), m0304b);
                let m0304b = _mm256_fmadd_pd(x3m4, _mm256_set1_pd(self.twiddle2.im), m0304b);
                let (y03, y04) = AvxButterfly::butterfly2_f64(m0304a, m0304b);

                let y0y1 = _mm256_permute2f128_pd::<LO_LO>(y00, y01);
                let y2y3 = _mm256_permute2f128_pd::<LO_LO>(y02, y03);
                let y4y5 = _mm256_permute2f128_pd::<LO_LO>(y04, y05);
                let y6y0_2 = _mm256_permute2f128_pd::<LO_HI>(y06, y00);
                let y1y0_2 = _mm256_permute2f128_pd::<HI_HI>(y01, y02);
                let y3y4_2 = _mm256_permute2f128_pd::<HI_HI>(y03, y04);
                let y5y6_2 = _mm256_permute2f128_pd::<HI_HI>(y05, y06);

                _mm256_storeu_pd(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y2y3);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y4y5);
                _mm256_storeu_pd(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), y6y0_2);
                _mm256_storeu_pd(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), y1y0_2);
                _mm256_storeu_pd(dst.get_unchecked_mut(10..).as_mut_ptr().cast(), y3y4_2);
                _mm256_storeu_pd(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), y5y6_2);
            }

            let rem_dst = dst.chunks_exact_mut(14).into_remainder();
            let rem_src = src.chunks_exact(14).remainder();

            let tw1tw2r = _mm256_setr_pd(
                self.twiddle1.re,
                self.twiddle1.re,
                self.twiddle2.re,
                self.twiddle2.re,
            );
            let tw2tw3r = _mm256_setr_pd(
                self.twiddle2.re,
                self.twiddle2.re,
                self.twiddle3.re,
                self.twiddle3.re,
            );
            let tw3tw1r = _mm256_setr_pd(
                self.twiddle3.re,
                self.twiddle3.re,
                self.twiddle1.re,
                self.twiddle1.re,
            );

            let tw1tw2i = _mm256_setr_pd(
                self.twiddle1.im,
                self.twiddle1.im,
                self.twiddle2.im,
                self.twiddle2.im,
            );
            let tw2tw3i = _mm256_setr_pd(
                self.twiddle2.im,
                self.twiddle2.im,
                -self.twiddle3.im,
                -self.twiddle3.im,
            );
            let tw3tw1i = _mm256_setr_pd(
                self.twiddle3.im,
                self.twiddle3.im,
                -self.twiddle1.im,
                -self.twiddle1.im,
            );

            for (dst, src) in rem_dst.chunks_exact_mut(7).zip(rem_src.chunks_exact(7)) {
                let u0u1 = _mm256_loadu_pd(src.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let u6 = _mm_loadu_pd(src.get_unchecked(6..).as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(u0u1);
                let u1 = _mm256_extractf128_pd::<1>(u0u1);
                let u2 = _mm256_castpd256_pd128(u2u3);
                let u3 = _mm256_extractf128_pd::<1>(u2u3);
                let u4 = _mm256_castpd256_pd128(u4u5);
                let u5 = _mm256_extractf128_pd::<1>(u4u5);

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let (x1p6x2p5, x1m6x2m5) = AvxButterfly::butterfly2_f64(
                    _mm256_create_pd(u1, u2),
                    _mm256_create_pd(u6, u5),
                );
                let x1m6x2m5 = rotate.rotate_m256d(x1m6x2m5);

                let x1p6 = _mm256_castpd256_pd128(x1p6x2p5);
                let x2p5 = _mm256_extractf128_pd::<1>(x1p6x2p5);
                let x1m6 = _mm256_castpd256_pd128(x1m6x2m5);
                let x2m5 = _mm256_extractf128_pd::<1>(x1m6x2m5);

                let y00 = _mm_add_pd(
                    _mm_add_pd(u0, _mm256_castpd256_pd128(x1p6x2p5)),
                    _mm256_extractf128_pd::<1>(x1p6x2p5),
                );
                let (x3p4, x3m4) = AvxButterfly::butterfly2_f64_m128(u3, u4);
                let x3m4 = rotate.rotate_m128d(x3m4);
                let y00 = _mm_add_pd(y00, x3p4);

                let x1p6d = _mm256_permute2f128_pd::<LO_LO>(x1p6x2p5, x1p6x2p5);
                let x2p5d = _mm256_permute2f128_pd::<HI_HI>(x1p6x2p5, x1p6x2p5);
                let x3p4d = _mm256_create_pd(x3p4, x3p4);

                let x1m6d = _mm256_permute2f128_pd::<LO_LO>(x1m6x2m5, x1m6x2m5);
                let x2m5d = _mm256_permute2f128_pd::<HI_HI>(x1m6x2m5, x1m6x2m5);
                let x3m4d = _mm256_create_pd(x3m4, x3m4);

                let m0106am0205a = _mm256_fmadd_pd(x1p6d, tw1tw2r, _mm256_create_pd(u0, u0));
                let m0106am0205a = _mm256_fmadd_pd(x2p5d, tw2tw3r, m0106am0205a);
                let m0106am0205a = _mm256_fmadd_pd(x3p4d, tw3tw1r, m0106am0205a);
                let m0106bm0205b = _mm256_mul_pd(x1m6d, tw1tw2i);
                let m0106bm0205b = _mm256_fmadd_pd(x2m5d, tw2tw3i, m0106bm0205b);
                let m0106bm0205b = _mm256_fmadd_pd(x3m4d, tw3tw1i, m0106bm0205b);
                let (y01y02, y06y05) = AvxButterfly::butterfly2_f64(m0106am0205a, m0106bm0205b);

                let m0304a = _mm_fmadd_pd(x1p6, _mm256_castpd256_pd128(tw3tw1r), u0);
                let m0304a = _mm_fmadd_pd(x2p5, _mm256_castpd256_pd128(tw1tw2r), m0304a);
                let m0304a = _mm_fmadd_pd(x3p4, _mm256_castpd256_pd128(tw2tw3r), m0304a);
                let m0304b = _mm_mul_pd(x1m6, _mm256_castpd256_pd128(tw3tw1i));
                let m0304b = _mm_fnmadd_pd(x2m5, _mm256_castpd256_pd128(tw1tw2i), m0304b);
                let m0304b = _mm_fmadd_pd(x3m4, _mm256_castpd256_pd128(tw2tw3i), m0304b);
                let (y03, y04) = AvxButterfly::butterfly2_f64_m128(m0304a, m0304b);

                let y3y4 = _mm256_create_pd(y03, y04);

                const HI_LO: i32 = 0b0010_0001;

                _mm_storeu_pd(dst.as_mut_ptr().cast(), y00);
                _mm256_storeu_pd(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), y01y02);
                _mm256_storeu_pd(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), y3y4);
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_LO>(y06y05, y06y05),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly7<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly7<f64> {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f64> for AvxButterfly7<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        7
    }
}

impl AvxButterfly7<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 7 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let rotate = AvxRotate::<f32>::new(FftDirection::Inverse);

            let tw1r = _mm256_set1_ps(self.twiddle1.re);
            let tw2r = _mm256_set1_ps(self.twiddle2.re);
            let tw3r = _mm256_set1_ps(self.twiddle3.re);
            let tw1i = _mm256_set1_ps(self.twiddle1.im);
            let tw2i = _mm256_set1_ps(self.twiddle2.im);
            let tw3i = _mm256_set1_ps(self.twiddle3.im);

            for chunk in in_place.chunks_exact_mut(28) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u3u4u5u6 = _mm256_loadu_ps(chunk.get_unchecked(3..).as_ptr().cast());

                let u0u1u2u3_1 = _mm256_loadu_ps(chunk.get_unchecked(7..).as_ptr().cast());
                let u3u4u5u6_1 = _mm256_loadu_ps(chunk.get_unchecked(10..).as_ptr().cast());

                let u0u1u2u3_2 = _mm256_loadu_ps(chunk.get_unchecked(14..).as_ptr().cast());
                let u3u4u5u6_2 = _mm256_loadu_ps(chunk.get_unchecked(17..).as_ptr().cast());

                let u0u1u2u3_3 = _mm256_loadu_ps(chunk.get_unchecked(21..).as_ptr().cast());
                let u3u4u5u6_3 = _mm256_loadu_ps(chunk.get_unchecked(24..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u2u3_1 = _mm256_extractf128_ps::<1>(u0u1u2u3_1);
                let u2u3_2 = _mm256_extractf128_ps::<1>(u0u1u2u3_2);
                let u2u3_3 = _mm256_extractf128_ps::<1>(u0u1u2u3_3);

                let u0 = _mm256_create_ps(
                    _mm_unpacklo_ps64(
                        _mm256_castps256_ps128(u0u1u2u3),
                        _mm256_castps256_ps128(u0u1u2u3_1),
                    ),
                    _mm_unpacklo_ps64(
                        _mm256_castps256_ps128(u0u1u2u3_2),
                        _mm256_castps256_ps128(u0u1u2u3_3),
                    ),
                );
                let u1 = _mm256_create_ps(
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u0u1u2u3),
                        _mm256_castps256_ps128(u0u1u2u3_1),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u0u1u2u3_2),
                        _mm256_castps256_ps128(u0u1u2u3_3),
                    ),
                );
                let u2 = _mm256_create_ps(
                    _mm_unpacklo_ps64(u2u3, u2u3_1),
                    _mm_unpacklo_ps64(u2u3_2, u2u3_3),
                );
                let u3 = _mm256_create_ps(
                    _mm_unpackhi_ps64(u2u3, u2u3_1),
                    _mm_unpackhi_ps64(u2u3_2, u2u3_3),
                );
                let u4 = _mm256_create_ps(
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u3u4u5u6),
                        _mm256_castps256_ps128(u3u4u5u6_1),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u3u4u5u6_2),
                        _mm256_castps256_ps128(u3u4u5u6_3),
                    ),
                );
                let u5u6 = _mm256_extractf128_ps::<1>(u3u4u5u6);
                let u5u6_1 = _mm256_extractf128_ps::<1>(u3u4u5u6_1);
                let u5u6_2 = _mm256_extractf128_ps::<1>(u3u4u5u6_2);
                let u5u6_3 = _mm256_extractf128_ps::<1>(u3u4u5u6_3);
                let u5 = _mm256_create_ps(
                    _mm_unpacklo_ps64(u5u6, u5u6_1),
                    _mm_unpacklo_ps64(u5u6_2, u5u6_3),
                );
                let u6 = _mm256_create_ps(
                    _mm_unpackhi_ps64(u5u6, u5u6_1),
                    _mm_unpackhi_ps64(u5u6_2, u5u6_3),
                );

                let (x1p6, x1m6) = AvxButterfly::butterfly2_f32(u1, u6);
                let x1m6 = rotate.rotate_m256(x1m6);
                let y00 = _mm256_add_ps(u0, x1p6);
                let (x2p5, x2m5) = AvxButterfly::butterfly2_f32(u2, u5);
                let x2m5 = rotate.rotate_m256(x2m5);
                let y00 = _mm256_add_ps(y00, x2p5);
                let (x3p4, x3m4) = AvxButterfly::butterfly2_f32(u3, u4);
                let x3m4 = rotate.rotate_m256(x3m4);
                let y00 = _mm256_add_ps(y00, x3p4);

                let m0106a = _mm256_fmadd_ps(x1p6, tw1r, u0);
                let m0106a = _mm256_fmadd_ps(x2p5, tw2r, m0106a);
                let m0106a = _mm256_fmadd_ps(x3p4, tw3r, m0106a);
                let m0106b = _mm256_mul_ps(x1m6, tw1i);
                let m0106b = _mm256_fmadd_ps(x2m5, tw2i, m0106b);
                let m0106b = _mm256_fmadd_ps(x3m4, tw3i, m0106b);
                let (y01, y06) = AvxButterfly::butterfly2_f32(m0106a, m0106b);

                let m0205a = _mm256_fmadd_ps(x1p6, tw2r, u0);
                let m0205a = _mm256_fmadd_ps(x2p5, tw3r, m0205a);
                let m0205a = _mm256_fmadd_ps(x3p4, tw1r, m0205a);
                let m0205b = _mm256_mul_ps(x1m6, tw2i);
                let m0205b = _mm256_fnmadd_ps(x2m5, tw3i, m0205b);
                let m0205b = _mm256_fnmadd_ps(x3m4, tw1i, m0205b);
                let (y02, y05) = AvxButterfly::butterfly2_f32(m0205a, m0205b);

                let m0304a = _mm256_fmadd_ps(x1p6, tw3r, u0);
                let m0304a = _mm256_fmadd_ps(x2p5, tw1r, m0304a);
                let m0304a = _mm256_fmadd_ps(x3p4, tw2r, m0304a);
                let m0304b = _mm256_mul_ps(x1m6, tw3i);
                let m0304b = _mm256_fnmadd_ps(x2m5, tw1i, m0304b);
                let m0304b = _mm256_fmadd_ps(x3m4, tw2i, m0304b);
                let (y03, y04) = AvxButterfly::butterfly2_f32(m0304a, m0304b);

                let y00y01_lo = _mm256_unpacklo_ps64(y00, y01);
                let y00y01_hi = _mm256_unpackhi_ps64(y00, y01);

                let y02y03_lo = _mm256_unpacklo_ps64(y02, y03);
                let y02y03_hi = _mm256_unpackhi_ps64(y02, y03);

                let y04y05_lo = _mm256_unpacklo_ps64(y04, y05);
                let y04y05_hi = _mm256_unpackhi_ps64(y04, y05);

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let y00y01y02y03_0 = _mm256_permute2f128_ps::<LO_LO>(y00y01_lo, y02y03_lo);
                let y00y01y02y03_1 = _mm256_permute2f128_ps::<LO_LO>(y00y01_hi, y02y03_hi);
                let y00y01y02y03_2 = _mm256_permute2f128_ps::<HI_HI>(y00y01_lo, y02y03_lo);
                let y00y01y02y03_3 = _mm256_permute2f128_ps::<HI_HI>(y00y01_hi, y02y03_hi);

                _mm256_storeu_ps(chunk.as_mut_ptr().cast(), y00y01y02y03_0);
                _mm_storeu_ps(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y04y05_lo),
                );
                _m128s_store_f32x2(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y06),
                );

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(7..).as_mut_ptr().cast(),
                    y00y01y02y03_1,
                );
                _mm_storeu_ps(
                    chunk.get_unchecked_mut(11..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y04y05_hi),
                );
                _m128s_storeh_f32x2(
                    chunk.get_unchecked_mut(13..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y06),
                );

                let y06hi = _mm256_extractf128_ps::<1>(y06);

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    y00y01y02y03_2,
                );
                _mm_storeu_ps(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    _mm256_extractf128_ps::<1>(y04y05_lo),
                );
                _m128s_store_f32x2(chunk.get_unchecked_mut(20..).as_mut_ptr().cast(), y06hi);

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(21..).as_mut_ptr().cast(),
                    y00y01y02y03_3,
                );
                _mm_storeu_ps(
                    chunk.get_unchecked_mut(25..).as_mut_ptr().cast(),
                    _mm256_extractf128_ps::<1>(y04y05_hi),
                );
                _m128s_storeh_f32x2(chunk.get_unchecked_mut(27..).as_mut_ptr().cast(), y06hi);
            }

            let tw1tw2tw3r = _mm256_setr_ps(
                self.twiddle1.re,
                self.twiddle1.re,
                self.twiddle2.re,
                self.twiddle2.re,
                self.twiddle3.re,
                self.twiddle3.re,
                0.,
                0.,
            );

            let tw2tw3tw1r = _mm256_setr_ps(
                self.twiddle2.re,
                self.twiddle2.re,
                self.twiddle3.re,
                self.twiddle3.re,
                self.twiddle1.re,
                self.twiddle1.re,
                0.,
                0.,
            );

            let tw3tw1tw2r = _mm256_setr_ps(
                self.twiddle3.re,
                self.twiddle3.re,
                self.twiddle1.re,
                self.twiddle1.re,
                self.twiddle2.re,
                self.twiddle2.re,
                0.,
                0.,
            );

            let tw1tw2tw3i = _mm256_setr_ps(
                self.twiddle1.im,
                self.twiddle1.im,
                self.twiddle2.im,
                self.twiddle2.im,
                self.twiddle3.im,
                self.twiddle3.im,
                0.,
                0.,
            );

            let tw2tw3tw1i = _mm256_setr_ps(
                self.twiddle2.im,
                self.twiddle2.im,
                -self.twiddle3.im,
                -self.twiddle3.im,
                -self.twiddle1.im,
                -self.twiddle1.im,
                0.,
                0.,
            );

            let tw3tw1tw2i = _mm256_setr_ps(
                self.twiddle3.im,
                self.twiddle3.im,
                -self.twiddle1.im,
                -self.twiddle1.im,
                self.twiddle2.im,
                self.twiddle2.im,
                0.,
                0.,
            );

            let rem = in_place.chunks_exact_mut(28).into_remainder();

            for chunk in rem.chunks_exact_mut(7) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u3u4u5u6 = _mm256_loadu_ps(chunk.get_unchecked(3..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);

                let u0 = _mm256_castps256_ps128(u0u1u2u3);
                let u1 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u0, u0);
                let u2 = u2u3;
                let u3 = _mm256_castps256_ps128(u3u4u5u6);
                let u4 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u3, u3);
                let u5u6 = _mm256_extractf128_ps::<1>(u3u4u5u6);
                let u5 = u5u6;
                let u6 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u5u6, u5u6);

                let u1u2u3 = _mm256_insertf128_ps::<1>(
                    _mm256_castps128_ps256(_mm_unpacklo_ps64(u1, u2)),
                    u3,
                );
                let u6u5u4 = _mm256_insertf128_ps::<1>(
                    _mm256_castps128_ps256(_mm_unpacklo_ps64(u6, u5)),
                    u4,
                );

                let (x1p6x2p5x3p4, x1m6x2m5x3m4) = AvxButterfly::butterfly2_f32(u1u2u3, u6u5u4);
                let x1m6x2m5x3m4 = rotate.rotate_m256(x1m6x2m5x3m4);
                let x1p6x2p5 = _mm256_castps256_ps128(x1p6x2p5x3p4);
                let y00 = _mm_add_ps(
                    _mm_add_ps(
                        _mm_add_ps(u0, x1p6x2p5),
                        _mm_unpackhi_ps64(x1p6x2p5, x1p6x2p5),
                    ),
                    _mm256_extractf128_ps::<1>(x1p6x2p5x3p4),
                );

                let u00 = _mm_unpacklo_ps64(u0, u0);
                let u0000 = _mm256_create_ps(u00, u00);

                let x1p6 = _mm256_permute4x64_ps::<{ shuffle(0, 0, 0, 0) }>(x1p6x2p5x3p4);
                let x2p5 = _mm256_permute4x64_ps::<{ shuffle(1, 1, 1, 1) }>(x1p6x2p5x3p4);
                let x3p4 = _mm256_permute4x64_ps::<{ shuffle(2, 2, 2, 2) }>(x1p6x2p5x3p4);
                let x1m6 = _mm256_permute4x64_ps::<{ shuffle(0, 0, 0, 0) }>(x1m6x2m5x3m4);
                let x2m5 = _mm256_permute4x64_ps::<{ shuffle(1, 1, 1, 1) }>(x1m6x2m5x3m4);
                let x3m4 = _mm256_permute4x64_ps::<{ shuffle(2, 2, 2, 2) }>(x1m6x2m5x3m4);

                let m0106a = _mm256_fmadd_ps(x1p6, tw1tw2tw3r, u0000);
                let m0106a = _mm256_fmadd_ps(x2p5, tw2tw3tw1r, m0106a);
                let m0106a = _mm256_fmadd_ps(x3p4, tw3tw1tw2r, m0106a);
                let m0106b = _mm256_mul_ps(x1m6, tw1tw2tw3i);
                let m0106b = _mm256_fmadd_ps(x2m5, tw2tw3tw1i, m0106b);
                let m0106b = _mm256_fmadd_ps(x3m4, tw3tw1tw2i, m0106b);
                let (y01y02y03, y06y05y04) = AvxButterfly::butterfly2_f32(m0106a, m0106b);

                let y0300 = _mm256_extractf128_ps::<1>(y01y02y03);
                let y0y1y2y3 = _mm256_create_ps(
                    _mm_unpacklo_ps64(y00, _mm256_castps256_ps128(y01y02y03)),
                    _mm_unpackhilo_ps64(_mm256_castps256_ps128(y01y02y03), y0300),
                );

                let y06y05 = _mm256_castps256_ps128(y06y05y04);

                let y3y4y5y6 = _mm256_create_ps(
                    _mm_unpacklo_ps64(y0300, _mm256_extractf128_ps::<1>(y06y05y04)),
                    _mm_unpackhilo_ps64(y06y05, y06y05),
                );

                _mm256_storeu_ps(chunk.as_mut_ptr().cast(), y0y1y2y3);
                _mm256_storeu_ps(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3y4y5y6);
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_out_of_place_f32(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe {
            if src.len() % 7 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 7 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            let rotate = AvxRotate::<f32>::new(FftDirection::Inverse);

            let tw1r = _mm256_set1_ps(self.twiddle1.re);
            let tw2r = _mm256_set1_ps(self.twiddle2.re);
            let tw3r = _mm256_set1_ps(self.twiddle3.re);
            let tw1i = _mm256_set1_ps(self.twiddle1.im);
            let tw2i = _mm256_set1_ps(self.twiddle2.im);
            let tw3i = _mm256_set1_ps(self.twiddle3.im);

            for (dst, src) in dst.chunks_exact_mut(28).zip(src.chunks_exact(28)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.as_ptr().cast());
                let u3u4u5u6 = _mm256_loadu_ps(src.get_unchecked(3..).as_ptr().cast());

                let u0u1u2u3_1 = _mm256_loadu_ps(src.get_unchecked(7..).as_ptr().cast());
                let u3u4u5u6_1 = _mm256_loadu_ps(src.get_unchecked(10..).as_ptr().cast());

                let u0u1u2u3_2 = _mm256_loadu_ps(src.get_unchecked(14..).as_ptr().cast());
                let u3u4u5u6_2 = _mm256_loadu_ps(src.get_unchecked(17..).as_ptr().cast());

                let u0u1u2u3_3 = _mm256_loadu_ps(src.get_unchecked(21..).as_ptr().cast());
                let u3u4u5u6_3 = _mm256_loadu_ps(src.get_unchecked(24..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u2u3_1 = _mm256_extractf128_ps::<1>(u0u1u2u3_1);
                let u2u3_2 = _mm256_extractf128_ps::<1>(u0u1u2u3_2);
                let u2u3_3 = _mm256_extractf128_ps::<1>(u0u1u2u3_3);

                let u0 = _mm256_create_ps(
                    _mm_unpacklo_ps64(
                        _mm256_castps256_ps128(u0u1u2u3),
                        _mm256_castps256_ps128(u0u1u2u3_1),
                    ),
                    _mm_unpacklo_ps64(
                        _mm256_castps256_ps128(u0u1u2u3_2),
                        _mm256_castps256_ps128(u0u1u2u3_3),
                    ),
                );
                let u1 = _mm256_create_ps(
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u0u1u2u3),
                        _mm256_castps256_ps128(u0u1u2u3_1),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u0u1u2u3_2),
                        _mm256_castps256_ps128(u0u1u2u3_3),
                    ),
                );
                let u2 = _mm256_create_ps(
                    _mm_unpacklo_ps64(u2u3, u2u3_1),
                    _mm_unpacklo_ps64(u2u3_2, u2u3_3),
                );
                let u3 = _mm256_create_ps(
                    _mm_unpackhi_ps64(u2u3, u2u3_1),
                    _mm_unpackhi_ps64(u2u3_2, u2u3_3),
                );
                let u4 = _mm256_create_ps(
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u3u4u5u6),
                        _mm256_castps256_ps128(u3u4u5u6_1),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u3u4u5u6_2),
                        _mm256_castps256_ps128(u3u4u5u6_3),
                    ),
                );
                let u5u6 = _mm256_extractf128_ps::<1>(u3u4u5u6);
                let u5u6_1 = _mm256_extractf128_ps::<1>(u3u4u5u6_1);
                let u5u6_2 = _mm256_extractf128_ps::<1>(u3u4u5u6_2);
                let u5u6_3 = _mm256_extractf128_ps::<1>(u3u4u5u6_3);
                let u5 = _mm256_create_ps(
                    _mm_unpacklo_ps64(u5u6, u5u6_1),
                    _mm_unpacklo_ps64(u5u6_2, u5u6_3),
                );
                let u6 = _mm256_create_ps(
                    _mm_unpackhi_ps64(u5u6, u5u6_1),
                    _mm_unpackhi_ps64(u5u6_2, u5u6_3),
                );

                let (x1p6, x1m6) = AvxButterfly::butterfly2_f32(u1, u6);
                let x1m6 = rotate.rotate_m256(x1m6);
                let y00 = _mm256_add_ps(u0, x1p6);
                let (x2p5, x2m5) = AvxButterfly::butterfly2_f32(u2, u5);
                let x2m5 = rotate.rotate_m256(x2m5);
                let y00 = _mm256_add_ps(y00, x2p5);
                let (x3p4, x3m4) = AvxButterfly::butterfly2_f32(u3, u4);
                let x3m4 = rotate.rotate_m256(x3m4);
                let y00 = _mm256_add_ps(y00, x3p4);

                let m0106a = _mm256_fmadd_ps(x1p6, tw1r, u0);
                let m0106a = _mm256_fmadd_ps(x2p5, tw2r, m0106a);
                let m0106a = _mm256_fmadd_ps(x3p4, tw3r, m0106a);
                let m0106b = _mm256_mul_ps(x1m6, tw1i);
                let m0106b = _mm256_fmadd_ps(x2m5, tw2i, m0106b);
                let m0106b = _mm256_fmadd_ps(x3m4, tw3i, m0106b);
                let (y01, y06) = AvxButterfly::butterfly2_f32(m0106a, m0106b);

                let m0205a = _mm256_fmadd_ps(x1p6, tw2r, u0);
                let m0205a = _mm256_fmadd_ps(x2p5, tw3r, m0205a);
                let m0205a = _mm256_fmadd_ps(x3p4, tw1r, m0205a);
                let m0205b = _mm256_mul_ps(x1m6, tw2i);
                let m0205b = _mm256_fnmadd_ps(x2m5, tw3i, m0205b);
                let m0205b = _mm256_fnmadd_ps(x3m4, tw1i, m0205b);
                let (y02, y05) = AvxButterfly::butterfly2_f32(m0205a, m0205b);

                let m0304a = _mm256_fmadd_ps(x1p6, tw3r, u0);
                let m0304a = _mm256_fmadd_ps(x2p5, tw1r, m0304a);
                let m0304a = _mm256_fmadd_ps(x3p4, tw2r, m0304a);
                let m0304b = _mm256_mul_ps(x1m6, tw3i);
                let m0304b = _mm256_fnmadd_ps(x2m5, tw1i, m0304b);
                let m0304b = _mm256_fmadd_ps(x3m4, tw2i, m0304b);
                let (y03, y04) = AvxButterfly::butterfly2_f32(m0304a, m0304b);

                let y00y01_lo = _mm256_unpacklo_ps64(y00, y01);
                let y00y01_hi = _mm256_unpackhi_ps64(y00, y01);

                let y02y03_lo = _mm256_unpacklo_ps64(y02, y03);
                let y02y03_hi = _mm256_unpackhi_ps64(y02, y03);

                let y04y05_lo = _mm256_unpacklo_ps64(y04, y05);
                let y04y05_hi = _mm256_unpackhi_ps64(y04, y05);

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let y00y01y02y03_0 = _mm256_permute2f128_ps::<LO_LO>(y00y01_lo, y02y03_lo);
                let y00y01y02y03_1 = _mm256_permute2f128_ps::<LO_LO>(y00y01_hi, y02y03_hi);
                let y00y01y02y03_2 = _mm256_permute2f128_ps::<HI_HI>(y00y01_lo, y02y03_lo);
                let y00y01y02y03_3 = _mm256_permute2f128_ps::<HI_HI>(y00y01_hi, y02y03_hi);

                _mm256_storeu_ps(dst.as_mut_ptr().cast(), y00y01y02y03_0);
                _mm_storeu_ps(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y04y05_lo),
                );
                _m128s_store_f32x2(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y06),
                );

                _mm256_storeu_ps(
                    dst.get_unchecked_mut(7..).as_mut_ptr().cast(),
                    y00y01y02y03_1,
                );
                _mm_storeu_ps(
                    dst.get_unchecked_mut(11..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y04y05_hi),
                );
                _m128s_storeh_f32x2(
                    dst.get_unchecked_mut(13..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y06),
                );

                let y06hi = _mm256_extractf128_ps::<1>(y06);

                _mm256_storeu_ps(
                    dst.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    y00y01y02y03_2,
                );
                _mm_storeu_ps(
                    dst.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    _mm256_extractf128_ps::<1>(y04y05_lo),
                );
                _m128s_store_f32x2(dst.get_unchecked_mut(20..).as_mut_ptr().cast(), y06hi);

                _mm256_storeu_ps(
                    dst.get_unchecked_mut(21..).as_mut_ptr().cast(),
                    y00y01y02y03_3,
                );
                _mm_storeu_ps(
                    dst.get_unchecked_mut(25..).as_mut_ptr().cast(),
                    _mm256_extractf128_ps::<1>(y04y05_hi),
                );
                _m128s_storeh_f32x2(dst.get_unchecked_mut(27..).as_mut_ptr().cast(), y06hi);
            }

            let tw1tw2tw3r = _mm256_setr_ps(
                self.twiddle1.re,
                self.twiddle1.re,
                self.twiddle2.re,
                self.twiddle2.re,
                self.twiddle3.re,
                self.twiddle3.re,
                0.,
                0.,
            );

            let tw2tw3tw1r = _mm256_setr_ps(
                self.twiddle2.re,
                self.twiddle2.re,
                self.twiddle3.re,
                self.twiddle3.re,
                self.twiddle1.re,
                self.twiddle1.re,
                0.,
                0.,
            );

            let tw3tw1tw2r = _mm256_setr_ps(
                self.twiddle3.re,
                self.twiddle3.re,
                self.twiddle1.re,
                self.twiddle1.re,
                self.twiddle2.re,
                self.twiddle2.re,
                0.,
                0.,
            );

            let tw1tw2tw3i = _mm256_setr_ps(
                self.twiddle1.im,
                self.twiddle1.im,
                self.twiddle2.im,
                self.twiddle2.im,
                self.twiddle3.im,
                self.twiddle3.im,
                0.,
                0.,
            );

            let tw2tw3tw1i = _mm256_setr_ps(
                self.twiddle2.im,
                self.twiddle2.im,
                -self.twiddle3.im,
                -self.twiddle3.im,
                -self.twiddle1.im,
                -self.twiddle1.im,
                0.,
                0.,
            );

            let tw3tw1tw2i = _mm256_setr_ps(
                self.twiddle3.im,
                self.twiddle3.im,
                -self.twiddle1.im,
                -self.twiddle1.im,
                self.twiddle2.im,
                self.twiddle2.im,
                0.,
                0.,
            );

            let rem_dst = dst.chunks_exact_mut(28).into_remainder();
            let rem_src = src.chunks_exact(28).remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(7).zip(rem_src.chunks_exact(7)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.as_ptr().cast());
                let u3u4u5u6 = _mm256_loadu_ps(src.get_unchecked(3..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);

                let u0 = _mm256_castps256_ps128(u0u1u2u3);
                let u1 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u0, u0);
                let u2 = u2u3;
                let u3 = _mm256_castps256_ps128(u3u4u5u6);
                let u4 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u3, u3);
                let u5u6 = _mm256_extractf128_ps::<1>(u3u4u5u6);
                let u5 = u5u6;
                let u6 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u5u6, u5u6);

                let u1u2u3 = _mm256_insertf128_ps::<1>(
                    _mm256_castps128_ps256(_mm_unpacklo_ps64(u1, u2)),
                    u3,
                );
                let u6u5u4 = _mm256_insertf128_ps::<1>(
                    _mm256_castps128_ps256(_mm_unpacklo_ps64(u6, u5)),
                    u4,
                );

                let (x1p6x2p5x3p4, x1m6x2m5x3m4) = AvxButterfly::butterfly2_f32(u1u2u3, u6u5u4);
                let x1m6x2m5x3m4 = rotate.rotate_m256(x1m6x2m5x3m4);
                let x1p6x2p5 = _mm256_castps256_ps128(x1p6x2p5x3p4);
                let y00 = _mm_add_ps(
                    _mm_add_ps(
                        _mm_add_ps(u0, x1p6x2p5),
                        _mm_unpackhi_ps64(x1p6x2p5, x1p6x2p5),
                    ),
                    _mm256_extractf128_ps::<1>(x1p6x2p5x3p4),
                );

                let u00 = _mm_unpacklo_ps64(u0, u0);
                let u0000 = _mm256_create_ps(u00, u00);

                let x1p6 = _mm256_permute4x64_ps::<{ shuffle(0, 0, 0, 0) }>(x1p6x2p5x3p4);
                let x2p5 = _mm256_permute4x64_ps::<{ shuffle(1, 1, 1, 1) }>(x1p6x2p5x3p4);
                let x3p4 = _mm256_permute4x64_ps::<{ shuffle(2, 2, 2, 2) }>(x1p6x2p5x3p4);
                let x1m6 = _mm256_permute4x64_ps::<{ shuffle(0, 0, 0, 0) }>(x1m6x2m5x3m4);
                let x2m5 = _mm256_permute4x64_ps::<{ shuffle(1, 1, 1, 1) }>(x1m6x2m5x3m4);
                let x3m4 = _mm256_permute4x64_ps::<{ shuffle(2, 2, 2, 2) }>(x1m6x2m5x3m4);

                let m0106a = _mm256_fmadd_ps(x1p6, tw1tw2tw3r, u0000);
                let m0106a = _mm256_fmadd_ps(x2p5, tw2tw3tw1r, m0106a);
                let m0106a = _mm256_fmadd_ps(x3p4, tw3tw1tw2r, m0106a);
                let m0106b = _mm256_mul_ps(x1m6, tw1tw2tw3i);
                let m0106b = _mm256_fmadd_ps(x2m5, tw2tw3tw1i, m0106b);
                let m0106b = _mm256_fmadd_ps(x3m4, tw3tw1tw2i, m0106b);
                let (y01y02y03, y06y05y04) = AvxButterfly::butterfly2_f32(m0106a, m0106b);

                let y0300 = _mm256_extractf128_ps::<1>(y01y02y03);
                let y0y1y2y3 = _mm256_create_ps(
                    _mm_unpacklo_ps64(y00, _mm256_castps256_ps128(y01y02y03)),
                    _mm_unpackhilo_ps64(_mm256_castps256_ps128(y01y02y03), y0300),
                );

                let y06y05 = _mm256_castps256_ps128(y06y05y04);

                let y3y4y5y6 = _mm256_create_ps(
                    _mm_unpacklo_ps64(y0300, _mm256_extractf128_ps::<1>(y06y05y04)),
                    _mm_unpackhilo_ps64(y06y05, y06y05),
                );

                _mm256_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1y2y3);
                _mm256_storeu_ps(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), y3y4y5y6);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly7<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly7<f32> {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for AvxButterfly7<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        7
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly7, f32, AvxButterfly7, 7, 1e-5);
    test_avx_butterfly!(test_avx_butterfly7_f64, f64, AvxButterfly7, 7, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly7, f32, AvxButterfly7, 7, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly7_f64, f64, AvxButterfly7, 7, 1e-7);
}
