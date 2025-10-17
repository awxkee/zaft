/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use crate::avx::butterflies::AvxButterfly;
use crate::avx::butterflies::fast_bf5::{AvxFastButterfly5d, AvxFastButterfly5f};
use crate::avx::util::{
    _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps, shuffle,
};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly10d {
    direction: FftDirection,
    bf5: AvxFastButterfly5d,
}

impl AvxButterfly10d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf5: unsafe { AvxFastButterfly5d::new(fft_direction) },
        }
    }
}

impl AvxButterfly10d {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 10 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(10) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());

                const LO_HI: i32 = 0b0011_0000;
                const HI_LO: i32 = 0b0010_0001;
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u0u5 = _mm256_permute2f128_pd::<LO_HI>(u0u1, u4u5);
                let u2u7 = _mm256_permute2f128_pd::<LO_HI>(u2u3, u6u7);
                let u4u9 = _mm256_permute2f128_pd::<LO_HI>(u4u5, u8u9);
                let u6u1 = _mm256_permute2f128_pd::<LO_HI>(u6u7, u0u1);
                let u8u3 = _mm256_permute2f128_pd::<LO_HI>(u8u9, u2u3);

                let mid0mid1 = self.bf5.exec(u0u5, u2u7, u4u9, u6u1, u8u3);

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.0),
                    _mm256_extractf128_pd::<1>(mid0mid1.0),
                );
                let (y2, y3) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.1),
                    _mm256_extractf128_pd::<1>(mid0mid1.1),
                );
                let (y4, y5) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.2),
                    _mm256_extractf128_pd::<1>(mid0mid1.2),
                );
                let (y6, y7) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.3),
                    _mm256_extractf128_pd::<1>(mid0mid1.3),
                );
                let (y8, y9) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.4),
                    _mm256_extractf128_pd::<1>(mid0mid1.4),
                );

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm256_create_pd(y0, y3),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(y4, y7),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(y8, y1),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(y2, y5),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(y6, y9),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly10d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        10
    }
}

pub(crate) struct AvxButterfly10f {
    direction: FftDirection,
    bf5: AvxFastButterfly5f,
}

impl AvxButterfly10f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf5: unsafe { AvxFastButterfly5f::new(fft_direction) },
        }
    }
}

impl AvxButterfly10f {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 10 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(10) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7u8u9 = _mm256_loadu_ps(chunk.get_unchecked(6..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u8u9 = _mm256_extractf128_ps::<1>(u6u7u8u9);

                let u0u5 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(
                    _mm256_castps256_ps128(u0u1u2u3),
                    _mm256_castps256_ps128(u4u5u6u7),
                );
                let u2u7 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(
                    u2u3,
                    _mm256_castps256_ps128(u6u7u8u9),
                );
                let u4u9 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(
                    _mm256_castps256_ps128(u4u5u6u7),
                    u8u9,
                );
                let u6u1 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(
                    _mm256_castps256_ps128(u6u7u8u9),
                    _mm256_castps256_ps128(u0u1u2u3),
                );
                let u8u3 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(u8u9, u2u3);

                let mid0mid1 = self.bf5.exec(u0u5, u2u7, u4u9, u6u1, u8u3);

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.0,
                    _mm_unpackhi_ps64(mid0mid1.0, mid0mid1.0),
                );
                let (y2, y3) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.1,
                    _mm_unpackhi_ps64(mid0mid1.1, mid0mid1.1),
                );
                let (y4, y5) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.2,
                    _mm_unpackhi_ps64(mid0mid1.2, mid0mid1.2),
                );
                let (y6, y7) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.3,
                    _mm_unpackhi_ps64(mid0mid1.3, mid0mid1.3),
                );
                let (y8, y9) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.4,
                    _mm_unpackhi_ps64(mid0mid1.4, mid0mid1.4),
                );

                let y0y3 = _mm_unpacklo_ps64(y0, y3);
                let y4y7 = _mm_unpacklo_ps64(y4, y7);
                let y8y1 = _mm_unpacklo_ps64(y8, y1);
                let y2y5 = _mm_unpacklo_ps64(y2, y5);
                let y6y9 = _mm_unpacklo_ps64(y6, y9);

                let yyyy0 = _mm256_create_ps(y0y3, y4y7);
                let yyyy1 = _mm256_create_ps(y8y1, y2y5);

                _mm256_storeu_ps(chunk.as_mut_ptr().cast(), yyyy0);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), yyyy1);
                _mm_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y6y9);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly10f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        10
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::butterflies::Butterfly10;
    use rand::Rng;

    #[test]
    fn test_butterfly10_f32() {
        for i in 1..5 {
            let size = 10usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly10f::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly10f::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly10::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 10f32)).collect();

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
    fn test_butterfly10_f64() {
        for i in 1..5 {
            let size = 10usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly10d::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly10d::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly10::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 10f64)).collect();

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
