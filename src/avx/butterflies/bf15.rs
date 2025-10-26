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

use crate::avx::butterflies::{AvxFastButterfly3, AvxFastButterfly5d, AvxFastButterfly5f};
use crate::avx::util::{
    _m128s_store_f32x2, _mm_unpackhi_ps64, _mm_unpackhilo_ps64, _mm_unpacklo_ps64,
    _mm_unpacklohi_ps64, _mm256_create_pd, _mm256_create_ps,
};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly15d {
    direction: FftDirection,
    bf5: AvxFastButterfly5d,
    bf3: AvxFastButterfly3<f64>,
}

impl AvxButterfly15d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf5: unsafe { AvxFastButterfly5d::new(fft_direction) },
            bf3: unsafe { AvxFastButterfly3::<f64>::new(fft_direction) },
        }
    }
}

impl AvxButterfly15d {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 15 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(15) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());
                let u14 = _mm_loadu_pd(chunk.get_unchecked(14..).as_ptr().cast());

                const LO_HI: i32 = 0b0011_0000;
                const HI_LO: i32 = 0b0010_0001;
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let mid0mid1 = self.bf5.exec(
                    _mm256_permute2f128_pd::<LO_HI>(u0u1, u4u5),
                    _mm256_permute2f128_pd::<HI_LO>(u2u3, u8u9),
                    _mm256_permute2f128_pd::<LO_HI>(u6u7, u10u11),
                    _mm256_permute2f128_pd::<HI_LO>(u8u9, _mm256_castpd128_pd256(u14)),
                    _mm256_permute2f128_pd::<LO_LO>(u12u13, u2u3),
                );
                let mid2 = self.bf5.exec_short(
                    _mm256_castpd256_pd128(u10u11),
                    _mm256_extractf128_pd::<1>(u12u13),
                    _mm256_extractf128_pd::<1>(u0u1),
                    _mm256_castpd256_pd128(u4u5),
                    _mm256_extractf128_pd::<1>(u6u7),
                );

                let (y0y3, y1y4, y2y5) = self.bf3.exec(
                    _mm256_permute2f128_pd::<LO_LO>(mid0mid1.0, mid0mid1.1),
                    _mm256_permute2f128_pd::<HI_HI>(mid0mid1.0, mid0mid1.1),
                    _mm256_create_pd(mid2.0, mid2.1),
                );
                let (y6y9, y7y10, y8y11) = self.bf3.exec(
                    _mm256_permute2f128_pd::<LO_LO>(mid0mid1.2, mid0mid1.3),
                    _mm256_permute2f128_pd::<HI_HI>(mid0mid1.2, mid0mid1.3),
                    _mm256_create_pd(mid2.2, mid2.3),
                );
                let (y12, y13, y14) = self.bf3.exec_m128(
                    _mm256_castpd256_pd128(mid0mid1.4),
                    _mm256_extractf128_pd::<1>(mid0mid1.4),
                    mid2.4,
                );

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y0y3, y1y4),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y8y11, y6y9),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(_mm256_castpd128_pd256(y13), y2y5),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_LO>(y0y3, y7y10),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_LO>(y8y11, _mm256_castpd128_pd256(y12)),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y1y4, y2y5),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y6y9, y7y10),
                );
                _mm_storeu_pd(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y14);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly15d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        15
    }
}

pub(crate) struct AvxButterfly15f {
    direction: FftDirection,
    bf5: AvxFastButterfly5f,
    bf3: AvxFastButterfly3<f32>,
}

impl AvxButterfly15f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf5: unsafe { AvxFastButterfly5f::new(fft_direction) },
            bf3: unsafe { AvxFastButterfly3::<f32>::new(fft_direction) },
        }
    }
}

impl AvxButterfly15f {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 15 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(15) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u11u12u13u14 = _mm256_loadu_ps(chunk.get_unchecked(11..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u8u9 = _mm256_castps256_ps128(u8u9u10u11);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);
                let u11u12 = _mm256_castps256_ps128(u11u12u13u14);
                let u13u14 = _mm256_extractf128_ps::<1>(u11u12u13u14);
                let u10u11 = _mm256_extractf128_ps::<1>(u8u9u10u11);

                let u0u5 = _mm_unpacklohi_ps64(u0u1, u4u5);

                let mid0mid1 = self.bf5.exec(
                    u0u5,
                    _mm_unpackhilo_ps64(u2u3, u8u9),
                    _mm_unpacklo_ps64(u6u7, u11u12),
                    _mm_unpackhi_ps64(u8u9, u13u14),
                    _mm_unpackhilo_ps64(u11u12, u2u3),
                );
                let mid2 = self.bf5.exec(
                    u10u11,
                    u13u14,
                    _mm_unpackhi_ps64(u0u1, u0u1),
                    u4u5,
                    _mm_unpackhi_ps64(u6u7, u6u7),
                );

                let (y0y3, y1y4, y2y5) = self.bf3.exec_m128(
                    _mm_unpacklo_ps64(mid0mid1.0, mid0mid1.1),
                    _mm_unpackhi_ps64(mid0mid1.0, mid0mid1.1),
                    _mm_unpacklo_ps64(mid2.0, mid2.1),
                );
                let (y6y9, y7y10, y8y11) = self.bf3.exec_m128(
                    _mm_unpacklo_ps64(mid0mid1.2, mid0mid1.3),
                    _mm_unpackhi_ps64(mid0mid1.2, mid0mid1.3),
                    _mm_unpacklo_ps64(mid2.2, mid2.3),
                );
                let (y12, y13, y14) = self.bf3.exec_m128(
                    mid0mid1.4,
                    _mm_unpackhi_ps64(mid0mid1.4, mid0mid1.4),
                    mid2.4,
                );

                let y0000 = _mm256_create_ps(
                    _mm_unpacklohi_ps64(y0y3, y1y4),
                    _mm_unpacklohi_ps64(y8y11, y6y9),
                );
                let y0001 = _mm256_create_ps(
                    _mm_unpacklo_ps64(y13, y2y5),
                    _mm_unpackhilo_ps64(y0y3, y7y10),
                );
                let y0002 = _mm256_create_ps(
                    _mm_unpackhilo_ps64(y8y11, y12),
                    _mm_unpacklohi_ps64(y1y4, y2y5),
                );
                let y003 = _mm_unpacklohi_ps64(y6y9, y7y10);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0000);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y0001);
                _mm256_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y0002);
                _mm_storeu_ps(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y003);
                _m128s_store_f32x2(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y14);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly15f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        15
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::butterflies::Butterfly15;
    use rand::Rng;

    #[test]
    fn test_butterfly15_f32() {
        for i in 1..4 {
            let size = 15usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly15f::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly15f::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly15::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 15f32)).collect();

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
        for i in 1..4 {
            let size = 15usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly15d::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly15d::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly15::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 15f64)).collect();

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
