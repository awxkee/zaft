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
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly13<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
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
        }
    }
}

impl AvxButterfly13<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 13 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let rotate = AvxRotate::<f64>::new(FftDirection::Inverse);

            for chunk in in_place.chunks_exact_mut(13) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u12 = _mm_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(u0u1);
                let y00 = _mm256_castpd256_pd128(u0u1);
                let (x1p12, x1m12) =
                    AvxButterfly::butterfly2_f64_m128(_mm256_extractf128_pd::<1>(u0u1), u12);
                let x1m12 = rotate.rotate_m128d(x1m12);
                let y00 = _mm_add_pd(y00, x1p12);
                let (x2p11, x2m11) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u2u3),
                    _mm256_extractf128_pd::<1>(u10u11),
                );
                let x2m11 = rotate.rotate_m128d(x2m11);
                let y00 = _mm_add_pd(y00, x2p11);
                let (x3p10, x3m10) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u2u3),
                    _mm256_castpd256_pd128(u10u11),
                );
                let x3m10 = rotate.rotate_m128d(x3m10);
                let y00 = _mm_add_pd(y00, x3p10);
                let (x4p9, x4m9) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u4u5),
                    _mm256_extractf128_pd::<1>(u8u9),
                );
                let x4m9 = rotate.rotate_m128d(x4m9);
                let y00 = _mm_add_pd(y00, x4p9);
                let (x5p8, x5m8) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u4u5),
                    _mm256_castpd256_pd128(u8u9),
                );
                let x5m8 = rotate.rotate_m128d(x5m8);
                let y00 = _mm_add_pd(y00, x5p8);
                let (x6p7, x6m7) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u6u7),
                    _mm256_extractf128_pd::<1>(u6u7),
                );
                let x6m7 = rotate.rotate_m128d(x6m7);
                let y00 = _mm_add_pd(y00, x6p7);

                let m0112a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle1.re), u0);
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

                let m0211a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle2.re), u0);
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

                let m0310a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle3.re), u0);
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

                let m0409a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle4.re), u0);
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

                let m0508a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle5.re), u0);
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

                let m0607a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle6.re), u0);
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
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_create_pd(y10, y11),
                );
                _mm_storeu_pd(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y12);
            }
        }
        Ok(())
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

            let rotate = AvxRotate::<f32>::new(FftDirection::Inverse);

            for chunk in in_place.chunks_exact_mut(13) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u7u8u9u10 = _mm256_loadu_ps(chunk.get_unchecked(7..).as_ptr().cast());
                let u9u10u11u12 = _mm256_loadu_ps(chunk.get_unchecked(9..).as_ptr().cast());

                let u0 = _mm256_castps256_ps128(u0u1u2u3);
                let y00 = _mm256_castps256_ps128(u0u1u2u3);
                let u11u12 = _mm256_extractf128_ps::<1>(u9u10u11u12);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u9u10 = _mm256_extractf128_ps::<1>(u7u8u9u10);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u7u8 = _mm256_castps256_ps128(u7u8u9u10);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);

                let (x1p12, x1m12) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u0, u0),
                    _mm_unpackhi_ps64(u11u12, u11u12),
                );
                let x1m12 = rotate.rotate_m128(x1m12);
                let y00 = _mm_add_ps(y00, x1p12);
                let (x2p11, x2m11) = AvxButterfly::butterfly2_f32_m128(u2u3, u11u12);
                let x2m11 = rotate.rotate_m128(x2m11);
                let y00 = _mm_add_ps(y00, x2p11);
                let (x3p10, x3m10) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u2u3, u2u3),
                    _mm_unpackhi_ps64(u9u10, u9u10),
                );
                let x3m10 = rotate.rotate_m128(x3m10);
                let y00 = _mm_add_ps(y00, x3p10);
                let (x4p9, x4m9) = AvxButterfly::butterfly2_f32_m128(u4u5, u9u10);
                let x4m9 = rotate.rotate_m128(x4m9);
                let y00 = _mm_add_ps(y00, x4p9);
                let (x5p8, x5m8) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u4u5, u4u5),
                    _mm_unpackhi_ps64(u7u8, u7u8),
                );
                let x5m8 = rotate.rotate_m128(x5m8);
                let y00 = _mm_add_ps(y00, x5p8);
                let (x6p7, x6m7) =
                    AvxButterfly::butterfly2_f32_m128(u6u7, _mm_unpackhi_ps64(u6u7, u6u7));
                let x6m7 = rotate.rotate_m128(x6m7);
                let y00 = _mm_add_ps(y00, x6p7);

                let m0112a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle1.re), u0);
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

                let m0211a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle2.re), u0);
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

                let m0310a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle3.re), u0);
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

                let m0409a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle4.re), u0);
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

                let m0508a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle5.re), u0);
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

                let m0607a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle6.re), u0);
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

                let y0000 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y00, y01), _mm_unpacklo_ps64(y02, y03));
                let y0001 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y04, y05), _mm_unpacklo_ps64(y06, y07));
                let y0002 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y07, y08), _mm_unpacklo_ps64(y09, y10));
                let y0003 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y09, y10), _mm_unpacklo_ps64(y11, y12));

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0000);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y0001);
                _mm256_storeu_ps(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y0002);
                _mm256_storeu_ps(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y0003);
            }
        }
        Ok(())
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
    use crate::butterflies::Butterfly13;
    use rand::Rng;

    #[test]
    fn test_butterfly13_f32() {
        for i in 1..3 {
            let size = 13usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly13::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly13::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly13::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 13f32)).collect();

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
    fn test_butterfly13_f64() {
        for i in 1..3 {
            let size = 13usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly13::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly13::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly13::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 13f64)).collect();

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
}
