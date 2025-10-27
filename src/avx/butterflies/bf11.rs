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
use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly11<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly11<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 11, fft_direction),
            twiddle2: compute_twiddle(2, 11, fft_direction),
            twiddle3: compute_twiddle(3, 11, fft_direction),
            twiddle4: compute_twiddle(4, 11, fft_direction),
            twiddle5: compute_twiddle(5, 11, fft_direction),
        }
    }
}

impl AvxButterfly11<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 11 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let rotate = AvxRotate::<f64>::new(FftDirection::Inverse);

            for chunk in in_place.chunks_exact_mut(22) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u0_2 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u1u2_2 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());
                let u3u4_2 = _mm256_loadu_pd(chunk.get_unchecked(14..).as_ptr().cast());
                let u5u6_2 = _mm256_loadu_pd(chunk.get_unchecked(16..).as_ptr().cast());
                let u7u8_2 = _mm256_loadu_pd(chunk.get_unchecked(18..).as_ptr().cast());
                let u9u10_2 = _mm256_loadu_pd(chunk.get_unchecked(20..).as_ptr().cast());

                const LO_HI: i32 = 0b0011_0000;
                const HI_LO: i32 = 0b0010_0001;
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u0 = _mm256_permute2f128_pd::<LO_HI>(u0u1, u10u0_2);
                let y00 = u0;
                let (x1p10, x1m10) = AvxButterfly::butterfly2_f64(
                    _mm256_permute2f128_pd::<HI_LO>(u0u1, u1u2_2),
                    _mm256_permute2f128_pd::<LO_HI>(u10u0_2, u9u10_2),
                );
                let x1m10 = rotate.rotate_m256d(x1m10);
                let y00 = _mm256_add_pd(y00, x1p10);
                let (x2p9, x2m9) = AvxButterfly::butterfly2_f64(
                    _mm256_permute2f128_pd::<LO_HI>(u2u3, u1u2_2),
                    _mm256_permute2f128_pd::<HI_LO>(u8u9, u9u10_2),
                );
                let x2m9 = rotate.rotate_m256d(x2m9);
                let y00 = _mm256_add_pd(y00, x2p9);
                let (x3p8, x3m8) = AvxButterfly::butterfly2_f64(
                    _mm256_permute2f128_pd::<HI_LO>(u2u3, u3u4_2),
                    _mm256_permute2f128_pd::<LO_HI>(u8u9, u7u8_2),
                );
                let x3m8 = rotate.rotate_m256d(x3m8);
                let y00 = _mm256_add_pd(y00, x3p8);
                let (x4p7, x4m7) = AvxButterfly::butterfly2_f64(
                    _mm256_permute2f128_pd::<LO_HI>(u4u5, u3u4_2),
                    _mm256_permute2f128_pd::<HI_LO>(u6u7, u7u8_2),
                );
                let x4m7 = rotate.rotate_m256d(x4m7);
                let y00 = _mm256_add_pd(y00, x4p7);
                let (x5p6, x5m6) = AvxButterfly::butterfly2_f64(
                    _mm256_permute2f128_pd::<HI_LO>(u4u5, u5u6_2),
                    _mm256_permute2f128_pd::<LO_HI>(u6u7, u5u6_2),
                );
                let x5m6 = rotate.rotate_m256d(x5m6);
                let y00 = _mm256_add_pd(y00, x5p6);

                let m0110a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle1.re), u0);
                let m0110a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle2.re), x2p9, m0110a);
                let m0110a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle3.re), x3p8, m0110a);
                let m0110a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle4.re), x4p7, m0110a);
                let m0110a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle5.re), x5p6, m0110a);
                let m0110b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle1.im));
                let m0110b = _mm256_fmadd_pd(x2m9, _mm256_set1_pd(self.twiddle2.im), m0110b);
                let m0110b = _mm256_fmadd_pd(x3m8, _mm256_set1_pd(self.twiddle3.im), m0110b);
                let m0110b = _mm256_fmadd_pd(x4m7, _mm256_set1_pd(self.twiddle4.im), m0110b);
                let m0110b = _mm256_fmadd_pd(x5m6, _mm256_set1_pd(self.twiddle5.im), m0110b);
                let (y01, y10) = AvxButterfly::butterfly2_f64(m0110a, m0110b);

                let m0209a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle2.re), u0);
                let m0209a = _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle4.re), m0209a);
                let m0209a = _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle5.re), m0209a);
                let m0209a = _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle3.re), m0209a);
                let m0209a = _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle1.re), m0209a);
                let m0209b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle2.im));
                let m0209b = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle4.im), x2m9, m0209b);
                let m0209b = _mm256_fnmadd_pd(x3m8, _mm256_set1_pd(self.twiddle5.im), m0209b);
                let m0209b = _mm256_fnmadd_pd(x4m7, _mm256_set1_pd(self.twiddle3.im), m0209b);
                let m0209b = _mm256_fnmadd_pd(x5m6, _mm256_set1_pd(self.twiddle1.im), m0209b);
                let (y02, y09) = AvxButterfly::butterfly2_f64(m0209a, m0209b);

                let m0308a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle3.re), u0);
                let m0308a = _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle5.re), m0308a);
                let m0308a = _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle2.re), m0308a);
                let m0308a = _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle1.re), m0308a);
                let m0308a = _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle4.re), m0308a);
                let m0308b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle3.im));
                let m0308b = _mm256_fnmadd_pd(x2m9, _mm256_set1_pd(self.twiddle5.im), m0308b);
                let m0308b = _mm256_fnmadd_pd(x3m8, _mm256_set1_pd(self.twiddle2.im), m0308b);
                let m0308b = _mm256_fmadd_pd(x4m7, _mm256_set1_pd(self.twiddle1.im), m0308b);
                let m0308b = _mm256_fmadd_pd(x5m6, _mm256_set1_pd(self.twiddle4.im), m0308b);
                let (y03, y08) = AvxButterfly::butterfly2_f64(m0308a, m0308b);

                let m0407a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle4.re), u0);
                let m0407a = _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle3.re), m0407a);
                let m0407a = _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle1.re), m0407a);
                let m0407a = _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle5.re), m0407a);
                let m0407a = _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle2.re), m0407a);
                let m0407b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle4.im));
                let m0407b = _mm256_fnmadd_pd(x2m9, _mm256_set1_pd(self.twiddle3.im), m0407b);
                let m0407b = _mm256_fmadd_pd(x3m8, _mm256_set1_pd(self.twiddle1.im), m0407b);
                let m0407b = _mm256_fmadd_pd(x4m7, _mm256_set1_pd(self.twiddle5.im), m0407b);
                let m0407b = _mm256_fnmadd_pd(x5m6, _mm256_set1_pd(self.twiddle2.im), m0407b);
                let (y04, y07) = AvxButterfly::butterfly2_f64(m0407a, m0407b);

                let m0506a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle5.re), u0);
                let m0506a = _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle1.re), m0506a);
                let m0506a = _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle4.re), m0506a);
                let m0506a = _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle2.re), m0506a);
                let m0506a = _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle3.re), m0506a);
                let m0506b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle5.im));
                let m0506b = _mm256_fnmadd_pd(x2m9, _mm256_set1_pd(self.twiddle1.im), m0506b);
                let m0506b = _mm256_fmadd_pd(x3m8, _mm256_set1_pd(self.twiddle4.im), m0506b);
                let m0506b = _mm256_fnmadd_pd(x4m7, _mm256_set1_pd(self.twiddle2.im), m0506b);
                let m0506b = _mm256_fmadd_pd(x5m6, _mm256_set1_pd(self.twiddle3.im), m0506b);
                let (y05, y06) = AvxButterfly::butterfly2_f64(m0506a, m0506b);

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y00, y01),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y02, y03),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y04, y05),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y06, y07),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y08, y09),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y10, y00),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y01, y02),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y03, y04),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y05, y06),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y07, y08),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y09, y10),
                );
            }

            let rem = in_place.chunks_exact_mut(22).into_remainder();

            for chunk in rem.chunks_exact_mut(11) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10 = _mm_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(u0u1);
                let y00 = u0;
                let (x1p10, x1m10) =
                    AvxButterfly::butterfly2_f64_m128(_mm256_extractf128_pd::<1>(u0u1), u10);
                let x1m10 = rotate.rotate_m128d(x1m10);
                let y00 = _mm_add_pd(y00, x1p10);
                let (x2p9, x2m9) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u2u3),
                    _mm256_extractf128_pd::<1>(u8u9),
                );
                let x2m9 = rotate.rotate_m128d(x2m9);
                let y00 = _mm_add_pd(y00, x2p9);
                let (x3p8, x3m8) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u2u3),
                    _mm256_castpd256_pd128(u8u9),
                );
                let x3m8 = rotate.rotate_m128d(x3m8);
                let y00 = _mm_add_pd(y00, x3p8);
                let (x4p7, x4m7) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u4u5),
                    _mm256_extractf128_pd::<1>(u6u7),
                );
                let x4m7 = rotate.rotate_m128d(x4m7);
                let y00 = _mm_add_pd(y00, x4p7);
                let (x5p6, x5m6) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u4u5),
                    _mm256_castpd256_pd128(u6u7),
                );
                let x5m6 = rotate.rotate_m128d(x5m6);
                let y00 = _mm_add_pd(y00, x5p6);

                let m0110a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle1.re), u0);
                let m0110a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle2.re), x2p9, m0110a);
                let m0110a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle3.re), x3p8, m0110a);
                let m0110a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle4.re), x4p7, m0110a);
                let m0110a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle5.re), x5p6, m0110a);
                let m0110b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle1.im));
                let m0110b = _mm_fmadd_pd(x2m9, _mm_set1_pd(self.twiddle2.im), m0110b);
                let m0110b = _mm_fmadd_pd(x3m8, _mm_set1_pd(self.twiddle3.im), m0110b);
                let m0110b = _mm_fmadd_pd(x4m7, _mm_set1_pd(self.twiddle4.im), m0110b);
                let m0110b = _mm_fmadd_pd(x5m6, _mm_set1_pd(self.twiddle5.im), m0110b);
                let (y01, y10) = AvxButterfly::butterfly2_f64_m128(m0110a, m0110b);

                let m0209a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle2.re), u0);
                let m0209a = _mm_fmadd_pd(x2p9, _mm_set1_pd(self.twiddle4.re), m0209a);
                let m0209a = _mm_fmadd_pd(x3p8, _mm_set1_pd(self.twiddle5.re), m0209a);
                let m0209a = _mm_fmadd_pd(x4p7, _mm_set1_pd(self.twiddle3.re), m0209a);
                let m0209a = _mm_fmadd_pd(x5p6, _mm_set1_pd(self.twiddle1.re), m0209a);
                let m0209b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle2.im));
                let m0209b = _mm_fmadd_pd(_mm_set1_pd(self.twiddle4.im), x2m9, m0209b);
                let m0209b = _mm_fnmadd_pd(x3m8, _mm_set1_pd(self.twiddle5.im), m0209b);
                let m0209b = _mm_fnmadd_pd(x4m7, _mm_set1_pd(self.twiddle3.im), m0209b);
                let m0209b = _mm_fnmadd_pd(x5m6, _mm_set1_pd(self.twiddle1.im), m0209b);
                let (y02, y09) = AvxButterfly::butterfly2_f64_m128(m0209a, m0209b);

                let m0308a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle3.re), u0);
                let m0308a = _mm_fmadd_pd(x2p9, _mm_set1_pd(self.twiddle5.re), m0308a);
                let m0308a = _mm_fmadd_pd(x3p8, _mm_set1_pd(self.twiddle2.re), m0308a);
                let m0308a = _mm_fmadd_pd(x4p7, _mm_set1_pd(self.twiddle1.re), m0308a);
                let m0308a = _mm_fmadd_pd(x5p6, _mm_set1_pd(self.twiddle4.re), m0308a);
                let m0308b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle3.im));
                let m0308b = _mm_fnmadd_pd(x2m9, _mm_set1_pd(self.twiddle5.im), m0308b);
                let m0308b = _mm_fnmadd_pd(x3m8, _mm_set1_pd(self.twiddle2.im), m0308b);
                let m0308b = _mm_fmadd_pd(x4m7, _mm_set1_pd(self.twiddle1.im), m0308b);
                let m0308b = _mm_fmadd_pd(x5m6, _mm_set1_pd(self.twiddle4.im), m0308b);
                let (y03, y08) = AvxButterfly::butterfly2_f64_m128(m0308a, m0308b);

                let m0407a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle4.re), u0);
                let m0407a = _mm_fmadd_pd(x2p9, _mm_set1_pd(self.twiddle3.re), m0407a);
                let m0407a = _mm_fmadd_pd(x3p8, _mm_set1_pd(self.twiddle1.re), m0407a);
                let m0407a = _mm_fmadd_pd(x4p7, _mm_set1_pd(self.twiddle5.re), m0407a);
                let m0407a = _mm_fmadd_pd(x5p6, _mm_set1_pd(self.twiddle2.re), m0407a);
                let m0407b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle4.im));
                let m0407b = _mm_fnmadd_pd(x2m9, _mm_set1_pd(self.twiddle3.im), m0407b);
                let m0407b = _mm_fmadd_pd(x3m8, _mm_set1_pd(self.twiddle1.im), m0407b);
                let m0407b = _mm_fmadd_pd(x4m7, _mm_set1_pd(self.twiddle5.im), m0407b);
                let m0407b = _mm_fnmadd_pd(x5m6, _mm_set1_pd(self.twiddle2.im), m0407b);
                let (y04, y07) = AvxButterfly::butterfly2_f64_m128(m0407a, m0407b);

                let m0506a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle5.re), u0);
                let m0506a = _mm_fmadd_pd(x2p9, _mm_set1_pd(self.twiddle1.re), m0506a);
                let m0506a = _mm_fmadd_pd(x3p8, _mm_set1_pd(self.twiddle4.re), m0506a);
                let m0506a = _mm_fmadd_pd(x4p7, _mm_set1_pd(self.twiddle2.re), m0506a);
                let m0506a = _mm_fmadd_pd(x5p6, _mm_set1_pd(self.twiddle3.re), m0506a);
                let m0506b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle5.im));
                let m0506b = _mm_fnmadd_pd(x2m9, _mm_set1_pd(self.twiddle1.im), m0506b);
                let m0506b = _mm_fmadd_pd(x3m8, _mm_set1_pd(self.twiddle4.im), m0506b);
                let m0506b = _mm_fnmadd_pd(x4m7, _mm_set1_pd(self.twiddle2.im), m0506b);
                let m0506b = _mm_fmadd_pd(x5m6, _mm_set1_pd(self.twiddle3.im), m0506b);
                let (y05, y06) = AvxButterfly::butterfly2_f64_m128(m0506a, m0506b);

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm256_create_pd(y00, y01),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(y02, y03),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(y04, y05),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(y06, y07),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(y08, y09),
                );
                _mm_storeu_pd(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);
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
            if src.len() % 11 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 11 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            let rotate = AvxRotate::<f64>::new(FftDirection::Inverse);

            for (dst, src) in dst.chunks_exact_mut(22).zip(src.chunks_exact(22)) {
                let u0u1 = _mm256_loadu_pd(src.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(src.get_unchecked(8..).as_ptr().cast());
                let u10u0_2 = _mm256_loadu_pd(src.get_unchecked(10..).as_ptr().cast());
                let u1u2_2 = _mm256_loadu_pd(src.get_unchecked(12..).as_ptr().cast());
                let u3u4_2 = _mm256_loadu_pd(src.get_unchecked(14..).as_ptr().cast());
                let u5u6_2 = _mm256_loadu_pd(src.get_unchecked(16..).as_ptr().cast());
                let u7u8_2 = _mm256_loadu_pd(src.get_unchecked(18..).as_ptr().cast());
                let u9u10_2 = _mm256_loadu_pd(src.get_unchecked(20..).as_ptr().cast());

                const LO_HI: i32 = 0b0011_0000;
                const HI_LO: i32 = 0b0010_0001;
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u0 = _mm256_permute2f128_pd::<LO_HI>(u0u1, u10u0_2);
                let y00 = u0;
                let (x1p10, x1m10) = AvxButterfly::butterfly2_f64(
                    _mm256_permute2f128_pd::<HI_LO>(u0u1, u1u2_2),
                    _mm256_permute2f128_pd::<LO_HI>(u10u0_2, u9u10_2),
                );
                let x1m10 = rotate.rotate_m256d(x1m10);
                let y00 = _mm256_add_pd(y00, x1p10);
                let (x2p9, x2m9) = AvxButterfly::butterfly2_f64(
                    _mm256_permute2f128_pd::<LO_HI>(u2u3, u1u2_2),
                    _mm256_permute2f128_pd::<HI_LO>(u8u9, u9u10_2),
                );
                let x2m9 = rotate.rotate_m256d(x2m9);
                let y00 = _mm256_add_pd(y00, x2p9);
                let (x3p8, x3m8) = AvxButterfly::butterfly2_f64(
                    _mm256_permute2f128_pd::<HI_LO>(u2u3, u3u4_2),
                    _mm256_permute2f128_pd::<LO_HI>(u8u9, u7u8_2),
                );
                let x3m8 = rotate.rotate_m256d(x3m8);
                let y00 = _mm256_add_pd(y00, x3p8);
                let (x4p7, x4m7) = AvxButterfly::butterfly2_f64(
                    _mm256_permute2f128_pd::<LO_HI>(u4u5, u3u4_2),
                    _mm256_permute2f128_pd::<HI_LO>(u6u7, u7u8_2),
                );
                let x4m7 = rotate.rotate_m256d(x4m7);
                let y00 = _mm256_add_pd(y00, x4p7);
                let (x5p6, x5m6) = AvxButterfly::butterfly2_f64(
                    _mm256_permute2f128_pd::<HI_LO>(u4u5, u5u6_2),
                    _mm256_permute2f128_pd::<LO_HI>(u6u7, u5u6_2),
                );
                let x5m6 = rotate.rotate_m256d(x5m6);
                let y00 = _mm256_add_pd(y00, x5p6);

                let m0110a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle1.re), u0);
                let m0110a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle2.re), x2p9, m0110a);
                let m0110a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle3.re), x3p8, m0110a);
                let m0110a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle4.re), x4p7, m0110a);
                let m0110a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle5.re), x5p6, m0110a);
                let m0110b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle1.im));
                let m0110b = _mm256_fmadd_pd(x2m9, _mm256_set1_pd(self.twiddle2.im), m0110b);
                let m0110b = _mm256_fmadd_pd(x3m8, _mm256_set1_pd(self.twiddle3.im), m0110b);
                let m0110b = _mm256_fmadd_pd(x4m7, _mm256_set1_pd(self.twiddle4.im), m0110b);
                let m0110b = _mm256_fmadd_pd(x5m6, _mm256_set1_pd(self.twiddle5.im), m0110b);
                let (y01, y10) = AvxButterfly::butterfly2_f64(m0110a, m0110b);

                let m0209a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle2.re), u0);
                let m0209a = _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle4.re), m0209a);
                let m0209a = _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle5.re), m0209a);
                let m0209a = _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle3.re), m0209a);
                let m0209a = _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle1.re), m0209a);
                let m0209b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle2.im));
                let m0209b = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle4.im), x2m9, m0209b);
                let m0209b = _mm256_fnmadd_pd(x3m8, _mm256_set1_pd(self.twiddle5.im), m0209b);
                let m0209b = _mm256_fnmadd_pd(x4m7, _mm256_set1_pd(self.twiddle3.im), m0209b);
                let m0209b = _mm256_fnmadd_pd(x5m6, _mm256_set1_pd(self.twiddle1.im), m0209b);
                let (y02, y09) = AvxButterfly::butterfly2_f64(m0209a, m0209b);

                let m0308a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle3.re), u0);
                let m0308a = _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle5.re), m0308a);
                let m0308a = _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle2.re), m0308a);
                let m0308a = _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle1.re), m0308a);
                let m0308a = _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle4.re), m0308a);
                let m0308b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle3.im));
                let m0308b = _mm256_fnmadd_pd(x2m9, _mm256_set1_pd(self.twiddle5.im), m0308b);
                let m0308b = _mm256_fnmadd_pd(x3m8, _mm256_set1_pd(self.twiddle2.im), m0308b);
                let m0308b = _mm256_fmadd_pd(x4m7, _mm256_set1_pd(self.twiddle1.im), m0308b);
                let m0308b = _mm256_fmadd_pd(x5m6, _mm256_set1_pd(self.twiddle4.im), m0308b);
                let (y03, y08) = AvxButterfly::butterfly2_f64(m0308a, m0308b);

                let m0407a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle4.re), u0);
                let m0407a = _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle3.re), m0407a);
                let m0407a = _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle1.re), m0407a);
                let m0407a = _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle5.re), m0407a);
                let m0407a = _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle2.re), m0407a);
                let m0407b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle4.im));
                let m0407b = _mm256_fnmadd_pd(x2m9, _mm256_set1_pd(self.twiddle3.im), m0407b);
                let m0407b = _mm256_fmadd_pd(x3m8, _mm256_set1_pd(self.twiddle1.im), m0407b);
                let m0407b = _mm256_fmadd_pd(x4m7, _mm256_set1_pd(self.twiddle5.im), m0407b);
                let m0407b = _mm256_fnmadd_pd(x5m6, _mm256_set1_pd(self.twiddle2.im), m0407b);
                let (y04, y07) = AvxButterfly::butterfly2_f64(m0407a, m0407b);

                let m0506a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle5.re), u0);
                let m0506a = _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle1.re), m0506a);
                let m0506a = _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle4.re), m0506a);
                let m0506a = _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle2.re), m0506a);
                let m0506a = _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle3.re), m0506a);
                let m0506b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle5.im));
                let m0506b = _mm256_fnmadd_pd(x2m9, _mm256_set1_pd(self.twiddle1.im), m0506b);
                let m0506b = _mm256_fmadd_pd(x3m8, _mm256_set1_pd(self.twiddle4.im), m0506b);
                let m0506b = _mm256_fnmadd_pd(x4m7, _mm256_set1_pd(self.twiddle2.im), m0506b);
                let m0506b = _mm256_fmadd_pd(x5m6, _mm256_set1_pd(self.twiddle3.im), m0506b);
                let (y05, y06) = AvxButterfly::butterfly2_f64(m0506a, m0506b);

                _mm256_storeu_pd(
                    dst.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y00, y01),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y02, y03),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y04, y05),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y06, y07),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y08, y09),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y10, y00),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y01, y02),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y03, y04),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y05, y06),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y07, y08),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y09, y10),
                );
            }

            let rem_src = src.chunks_exact(22).remainder();
            let rem_dst = dst.chunks_exact_mut(22).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(11).zip(rem_src.chunks_exact(11)) {
                let u0u1 = _mm256_loadu_pd(src.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(src.get_unchecked(8..).as_ptr().cast());
                let u10 = _mm_loadu_pd(src.get_unchecked(10..).as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(u0u1);
                let y00 = u0;
                let (x1p10, x1m10) =
                    AvxButterfly::butterfly2_f64_m128(_mm256_extractf128_pd::<1>(u0u1), u10);
                let x1m10 = rotate.rotate_m128d(x1m10);
                let y00 = _mm_add_pd(y00, x1p10);
                let (x2p9, x2m9) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u2u3),
                    _mm256_extractf128_pd::<1>(u8u9),
                );
                let x2m9 = rotate.rotate_m128d(x2m9);
                let y00 = _mm_add_pd(y00, x2p9);
                let (x3p8, x3m8) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u2u3),
                    _mm256_castpd256_pd128(u8u9),
                );
                let x3m8 = rotate.rotate_m128d(x3m8);
                let y00 = _mm_add_pd(y00, x3p8);
                let (x4p7, x4m7) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u4u5),
                    _mm256_extractf128_pd::<1>(u6u7),
                );
                let x4m7 = rotate.rotate_m128d(x4m7);
                let y00 = _mm_add_pd(y00, x4p7);
                let (x5p6, x5m6) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u4u5),
                    _mm256_castpd256_pd128(u6u7),
                );
                let x5m6 = rotate.rotate_m128d(x5m6);
                let y00 = _mm_add_pd(y00, x5p6);

                let m0110a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle1.re), u0);
                let m0110a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle2.re), x2p9, m0110a);
                let m0110a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle3.re), x3p8, m0110a);
                let m0110a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle4.re), x4p7, m0110a);
                let m0110a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle5.re), x5p6, m0110a);
                let m0110b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle1.im));
                let m0110b = _mm_fmadd_pd(x2m9, _mm_set1_pd(self.twiddle2.im), m0110b);
                let m0110b = _mm_fmadd_pd(x3m8, _mm_set1_pd(self.twiddle3.im), m0110b);
                let m0110b = _mm_fmadd_pd(x4m7, _mm_set1_pd(self.twiddle4.im), m0110b);
                let m0110b = _mm_fmadd_pd(x5m6, _mm_set1_pd(self.twiddle5.im), m0110b);
                let (y01, y10) = AvxButterfly::butterfly2_f64_m128(m0110a, m0110b);

                let m0209a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle2.re), u0);
                let m0209a = _mm_fmadd_pd(x2p9, _mm_set1_pd(self.twiddle4.re), m0209a);
                let m0209a = _mm_fmadd_pd(x3p8, _mm_set1_pd(self.twiddle5.re), m0209a);
                let m0209a = _mm_fmadd_pd(x4p7, _mm_set1_pd(self.twiddle3.re), m0209a);
                let m0209a = _mm_fmadd_pd(x5p6, _mm_set1_pd(self.twiddle1.re), m0209a);
                let m0209b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle2.im));
                let m0209b = _mm_fmadd_pd(_mm_set1_pd(self.twiddle4.im), x2m9, m0209b);
                let m0209b = _mm_fnmadd_pd(x3m8, _mm_set1_pd(self.twiddle5.im), m0209b);
                let m0209b = _mm_fnmadd_pd(x4m7, _mm_set1_pd(self.twiddle3.im), m0209b);
                let m0209b = _mm_fnmadd_pd(x5m6, _mm_set1_pd(self.twiddle1.im), m0209b);
                let (y02, y09) = AvxButterfly::butterfly2_f64_m128(m0209a, m0209b);

                let m0308a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle3.re), u0);
                let m0308a = _mm_fmadd_pd(x2p9, _mm_set1_pd(self.twiddle5.re), m0308a);
                let m0308a = _mm_fmadd_pd(x3p8, _mm_set1_pd(self.twiddle2.re), m0308a);
                let m0308a = _mm_fmadd_pd(x4p7, _mm_set1_pd(self.twiddle1.re), m0308a);
                let m0308a = _mm_fmadd_pd(x5p6, _mm_set1_pd(self.twiddle4.re), m0308a);
                let m0308b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle3.im));
                let m0308b = _mm_fnmadd_pd(x2m9, _mm_set1_pd(self.twiddle5.im), m0308b);
                let m0308b = _mm_fnmadd_pd(x3m8, _mm_set1_pd(self.twiddle2.im), m0308b);
                let m0308b = _mm_fmadd_pd(x4m7, _mm_set1_pd(self.twiddle1.im), m0308b);
                let m0308b = _mm_fmadd_pd(x5m6, _mm_set1_pd(self.twiddle4.im), m0308b);
                let (y03, y08) = AvxButterfly::butterfly2_f64_m128(m0308a, m0308b);

                let m0407a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle4.re), u0);
                let m0407a = _mm_fmadd_pd(x2p9, _mm_set1_pd(self.twiddle3.re), m0407a);
                let m0407a = _mm_fmadd_pd(x3p8, _mm_set1_pd(self.twiddle1.re), m0407a);
                let m0407a = _mm_fmadd_pd(x4p7, _mm_set1_pd(self.twiddle5.re), m0407a);
                let m0407a = _mm_fmadd_pd(x5p6, _mm_set1_pd(self.twiddle2.re), m0407a);
                let m0407b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle4.im));
                let m0407b = _mm_fnmadd_pd(x2m9, _mm_set1_pd(self.twiddle3.im), m0407b);
                let m0407b = _mm_fmadd_pd(x3m8, _mm_set1_pd(self.twiddle1.im), m0407b);
                let m0407b = _mm_fmadd_pd(x4m7, _mm_set1_pd(self.twiddle5.im), m0407b);
                let m0407b = _mm_fnmadd_pd(x5m6, _mm_set1_pd(self.twiddle2.im), m0407b);
                let (y04, y07) = AvxButterfly::butterfly2_f64_m128(m0407a, m0407b);

                let m0506a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle5.re), u0);
                let m0506a = _mm_fmadd_pd(x2p9, _mm_set1_pd(self.twiddle1.re), m0506a);
                let m0506a = _mm_fmadd_pd(x3p8, _mm_set1_pd(self.twiddle4.re), m0506a);
                let m0506a = _mm_fmadd_pd(x4p7, _mm_set1_pd(self.twiddle2.re), m0506a);
                let m0506a = _mm_fmadd_pd(x5p6, _mm_set1_pd(self.twiddle3.re), m0506a);
                let m0506b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle5.im));
                let m0506b = _mm_fnmadd_pd(x2m9, _mm_set1_pd(self.twiddle1.im), m0506b);
                let m0506b = _mm_fmadd_pd(x3m8, _mm_set1_pd(self.twiddle4.im), m0506b);
                let m0506b = _mm_fnmadd_pd(x4m7, _mm_set1_pd(self.twiddle2.im), m0506b);
                let m0506b = _mm_fmadd_pd(x5m6, _mm_set1_pd(self.twiddle3.im), m0506b);
                let (y05, y06) = AvxButterfly::butterfly2_f64_m128(m0506a, m0506b);

                _mm256_storeu_pd(
                    dst.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm256_create_pd(y00, y01),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(y02, y03),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(y04, y05),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(y06, y07),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(y08, y09),
                );
                _mm_storeu_pd(dst.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly11<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly11<f64> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f64> for AvxButterfly11<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        11
    }
}

impl AvxButterfly11<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 11 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let rotate = AvxRotate::<f32>::new(FftDirection::Inverse);

            for chunk in in_place.chunks_exact_mut(11) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u7u8u9u10 = _mm256_loadu_ps(chunk.get_unchecked(7..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u9u10 = _mm256_extractf128_ps::<1>(u7u8u9u10);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u7u8 = _mm256_castps256_ps128(u7u8u9u10);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);

                let u0 = u0u1;
                let y00 = u0u1;
                let (x1p10, x1m10) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u0u1, u0u1),
                    _mm_unpackhi_ps64(u9u10, u9u10),
                );
                let x1m10 = rotate.rotate_m128(x1m10);
                let y00 = _mm_add_ps(y00, x1p10);
                let (x2p9, x2m9) = AvxButterfly::butterfly2_f32_m128(u2u3, u9u10);
                let x2m9 = rotate.rotate_m128(x2m9);
                let y00 = _mm_add_ps(y00, x2p9);
                let (x3p8, x3m8) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u2u3, u2u3),
                    _mm_unpackhi_ps64(u7u8, u7u8),
                );
                let x3m8 = rotate.rotate_m128(x3m8);
                let y00 = _mm_add_ps(y00, x3p8);
                let (x4p7, x4m7) = AvxButterfly::butterfly2_f32_m128(u4u5, u7u8);
                let x4m7 = rotate.rotate_m128(x4m7);
                let y00 = _mm_add_ps(y00, x4p7);
                let (x5p6, x5m6) =
                    AvxButterfly::butterfly2_f32_m128(_mm_unpackhi_ps64(u4u5, u4u5), u6u7);
                let x5m6 = rotate.rotate_m128(x5m6);
                let y00 = _mm_add_ps(y00, x5p6);

                let m0110a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle1.re), u0);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle2.re), x2p9, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle3.re), x3p8, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.re), x4p7, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle5.re), x5p6, m0110a);
                let m0110b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle1.im));
                let m0110b = _mm_fmadd_ps(x2m9, _mm_set1_ps(self.twiddle2.im), m0110b);
                let m0110b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle3.im), m0110b);
                let m0110b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle4.im), m0110b);
                let m0110b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle5.im), m0110b);
                let (y01, y10) = AvxButterfly::butterfly2_f32_m128(m0110a, m0110b);

                let m0209a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle2.re), u0);
                let m0209a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle4.re), m0209a);
                let m0209a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle5.re), m0209a);
                let m0209a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle3.re), m0209a);
                let m0209a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle1.re), m0209a);
                let m0209b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle2.im));
                let m0209b = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.im), x2m9, m0209b);
                let m0209b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle5.im), m0209b);
                let m0209b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle3.im), m0209b);
                let m0209b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle1.im), m0209b);
                let (y02, y09) = AvxButterfly::butterfly2_f32_m128(m0209a, m0209b);

                let m0308a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle3.re), u0);
                let m0308a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle5.re), m0308a);
                let m0308a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle2.re), m0308a);
                let m0308a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle1.re), m0308a);
                let m0308a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle4.re), m0308a);
                let m0308b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle3.im));
                let m0308b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle5.im), m0308b);
                let m0308b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle2.im), m0308b);
                let m0308b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle1.im), m0308b);
                let m0308b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle4.im), m0308b);
                let (y03, y08) = AvxButterfly::butterfly2_f32_m128(m0308a, m0308b);

                let m0407a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle4.re), u0);
                let m0407a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle3.re), m0407a);
                let m0407a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle1.re), m0407a);
                let m0407a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle5.re), m0407a);
                let m0407a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle2.re), m0407a);
                let m0407b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle4.im));
                let m0407b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle3.im), m0407b);
                let m0407b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle1.im), m0407b);
                let m0407b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle5.im), m0407b);
                let m0407b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle2.im), m0407b);
                let (y04, y07) = AvxButterfly::butterfly2_f32_m128(m0407a, m0407b);

                let m0506a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle5.re), u0);
                let m0506a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle1.re), m0506a);
                let m0506a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle4.re), m0506a);
                let m0506a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle2.re), m0506a);
                let m0506a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle3.re), m0506a);
                let m0506b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle5.im));
                let m0506b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle1.im), m0506b);
                let m0506b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle4.im), m0506b);
                let m0506b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle2.im), m0506b);
                let m0506b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle3.im), m0506b);
                let (y05, y06) = AvxButterfly::butterfly2_f32_m128(m0506a, m0506b);

                let y0000 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y00, y01), _mm_unpacklo_ps64(y02, y03));
                let y0001 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y04, y05), _mm_unpacklo_ps64(y06, y07));
                let y0002 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y07, y08), _mm_unpacklo_ps64(y09, y10));

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0000);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y0001);
                _mm256_storeu_ps(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y0002);
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
            if src.len() % 11 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 11 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            let rotate = AvxRotate::<f32>::new(FftDirection::Inverse);

            for (dst, src) in dst.chunks_exact_mut(11).zip(src.chunks_exact(11)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let u7u8u9u10 = _mm256_loadu_ps(src.get_unchecked(7..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u9u10 = _mm256_extractf128_ps::<1>(u7u8u9u10);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u7u8 = _mm256_castps256_ps128(u7u8u9u10);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);

                let u0 = u0u1;
                let y00 = u0u1;
                let (x1p10, x1m10) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u0u1, u0u1),
                    _mm_unpackhi_ps64(u9u10, u9u10),
                );
                let x1m10 = rotate.rotate_m128(x1m10);
                let y00 = _mm_add_ps(y00, x1p10);
                let (x2p9, x2m9) = AvxButterfly::butterfly2_f32_m128(u2u3, u9u10);
                let x2m9 = rotate.rotate_m128(x2m9);
                let y00 = _mm_add_ps(y00, x2p9);
                let (x3p8, x3m8) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u2u3, u2u3),
                    _mm_unpackhi_ps64(u7u8, u7u8),
                );
                let x3m8 = rotate.rotate_m128(x3m8);
                let y00 = _mm_add_ps(y00, x3p8);
                let (x4p7, x4m7) = AvxButterfly::butterfly2_f32_m128(u4u5, u7u8);
                let x4m7 = rotate.rotate_m128(x4m7);
                let y00 = _mm_add_ps(y00, x4p7);
                let (x5p6, x5m6) =
                    AvxButterfly::butterfly2_f32_m128(_mm_unpackhi_ps64(u4u5, u4u5), u6u7);
                let x5m6 = rotate.rotate_m128(x5m6);
                let y00 = _mm_add_ps(y00, x5p6);

                let m0110a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle1.re), u0);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle2.re), x2p9, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle3.re), x3p8, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.re), x4p7, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle5.re), x5p6, m0110a);
                let m0110b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle1.im));
                let m0110b = _mm_fmadd_ps(x2m9, _mm_set1_ps(self.twiddle2.im), m0110b);
                let m0110b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle3.im), m0110b);
                let m0110b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle4.im), m0110b);
                let m0110b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle5.im), m0110b);
                let (y01, y10) = AvxButterfly::butterfly2_f32_m128(m0110a, m0110b);

                let m0209a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle2.re), u0);
                let m0209a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle4.re), m0209a);
                let m0209a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle5.re), m0209a);
                let m0209a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle3.re), m0209a);
                let m0209a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle1.re), m0209a);
                let m0209b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle2.im));
                let m0209b = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.im), x2m9, m0209b);
                let m0209b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle5.im), m0209b);
                let m0209b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle3.im), m0209b);
                let m0209b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle1.im), m0209b);
                let (y02, y09) = AvxButterfly::butterfly2_f32_m128(m0209a, m0209b);

                let m0308a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle3.re), u0);
                let m0308a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle5.re), m0308a);
                let m0308a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle2.re), m0308a);
                let m0308a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle1.re), m0308a);
                let m0308a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle4.re), m0308a);
                let m0308b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle3.im));
                let m0308b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle5.im), m0308b);
                let m0308b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle2.im), m0308b);
                let m0308b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle1.im), m0308b);
                let m0308b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle4.im), m0308b);
                let (y03, y08) = AvxButterfly::butterfly2_f32_m128(m0308a, m0308b);

                let m0407a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle4.re), u0);
                let m0407a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle3.re), m0407a);
                let m0407a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle1.re), m0407a);
                let m0407a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle5.re), m0407a);
                let m0407a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle2.re), m0407a);
                let m0407b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle4.im));
                let m0407b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle3.im), m0407b);
                let m0407b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle1.im), m0407b);
                let m0407b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle5.im), m0407b);
                let m0407b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle2.im), m0407b);
                let (y04, y07) = AvxButterfly::butterfly2_f32_m128(m0407a, m0407b);

                let m0506a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle5.re), u0);
                let m0506a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle1.re), m0506a);
                let m0506a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle4.re), m0506a);
                let m0506a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle2.re), m0506a);
                let m0506a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle3.re), m0506a);
                let m0506b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle5.im));
                let m0506b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle1.im), m0506b);
                let m0506b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle4.im), m0506b);
                let m0506b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle2.im), m0506b);
                let m0506b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle3.im), m0506b);
                let (y05, y06) = AvxButterfly::butterfly2_f32_m128(m0506a, m0506b);

                let y0000 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y00, y01), _mm_unpacklo_ps64(y02, y03));
                let y0001 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y04, y05), _mm_unpacklo_ps64(y06, y07));
                let y0002 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y07, y08), _mm_unpacklo_ps64(y09, y10));

                _mm256_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0000);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y0001);
                _mm256_storeu_ps(dst.get_unchecked_mut(7..).as_mut_ptr().cast(), y0002);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly11<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly11<f32> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for AvxButterfly11<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        11
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::butterflies::Butterfly11;
    use crate::dft::Dft;
    use rand::Rng;

    #[test]
    fn test_butterfly11_f32() {
        for i in 1..4 {
            let size = 11usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly11::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly11::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly11::new(FftDirection::Forward);

            radix_forward.execute(&mut input).unwrap();
            radix_forward_ref.execute(&mut ref0).unwrap();

            input
                .iter()
                .zip(ref0.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-4,
                        "forward at {idx} a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-4,
                        "forward at {idx} a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 11f32)).collect();

            input
                .iter()
                .zip(src.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-5,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-5,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });
        }
    }

    #[test]
    fn test_butterfly11_f64() {
        for i in 1..4 {
            let size = 11usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly11::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly11::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly11::new(FftDirection::Forward);

            radix_forward.execute(&mut input).unwrap();
            radix_forward_ref.execute(&mut ref0).unwrap();

            input
                .iter()
                .zip(ref0.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "forward at {idx} a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "forward at {idx} a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 11f64)).collect();

            input
                .iter()
                .zip(src.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });
        }
    }

    #[test]
    fn test_butterfly11_out_of_place_f64() {
        for i in 1..4 {
            let size = 11usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut out_of_place = vec![Complex::<f64>::default(); size];
            let mut ref_input = input.to_vec();
            let radix_forward = AvxButterfly11::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly11::new(FftDirection::Inverse);

            let reference_dft = Dft::new(11, FftDirection::Forward).unwrap();
            reference_dft.execute(&mut ref_input).unwrap();

            radix_forward
                .execute_out_of_place(&input, &mut out_of_place)
                .unwrap();

            out_of_place
                .iter()
                .zip(ref_input.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse
                .execute_out_of_place(&out_of_place, &mut input)
                .unwrap();

            input = input.iter().map(|&x| x * (1.0 / 11f64)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly11_out_of_place_f32() {
        for i in 1..4 {
            let size = 11usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut out_of_place = vec![Complex::<f32>::default(); size];
            let mut ref_input = input.to_vec();
            let radix_forward = AvxButterfly11::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly11::new(FftDirection::Inverse);

            let reference_dft = Dft::new(11, FftDirection::Forward).unwrap();
            reference_dft.execute(&mut ref_input).unwrap();

            radix_forward
                .execute_out_of_place(&input, &mut out_of_place)
                .unwrap();

            out_of_place
                .iter()
                .zip(ref_input.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-4,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-4,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse
                .execute_out_of_place(&out_of_place, &mut input)
                .unwrap();

            input = input.iter().map(|&x| x * (1.0 / 11f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }
}
