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
use crate::avx::butterflies::fast_bf7::AvxFastButterfly7;
use crate::avx::util::{
    _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps, shuffle,
};
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly14<T> {
    direction: FftDirection,
    bf7: AvxFastButterfly7<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly14<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf7: unsafe { AvxFastButterfly7::new(fft_direction) },
        }
    }
}

impl AvxButterfly14<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 14 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(14) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());

                let (u0, u1) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u0u1),
                    _mm256_extractf128_pd::<1>(u6u7),
                ); // 0,7
                let (u2, u3) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u8u9),
                    _mm256_extractf128_pd::<1>(u0u1),
                ); // 8,1
                let (u4, u5) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u2u3),
                    _mm256_extractf128_pd::<1>(u8u9),
                ); // 2,9
                let (u6, u7) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u10u11),
                    _mm256_extractf128_pd::<1>(u2u3),
                ); // 10, 3
                let (u8, u9) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u4u5),
                    _mm256_extractf128_pd::<1>(u10u11),
                ); // 4, 11
                let (u10, u11) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u12u13),
                    _mm256_extractf128_pd::<1>(u4u5),
                ); // 12, 5
                let (u12, u13) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u6u7),
                    _mm256_extractf128_pd::<1>(u12u13),
                ); // 6, 13

                let (y0y7, y2y9, y4y11, y6y13, y8y1, y10y3, y12y5) = self.bf7.exec(
                    _mm256_create_pd(u0, u1),
                    _mm256_create_pd(u2, u3),
                    _mm256_create_pd(u4, u5),
                    _mm256_create_pd(u6, u7),
                    _mm256_create_pd(u8, u9),
                    _mm256_create_pd(u10, u11),
                    _mm256_create_pd(u12, u13),
                );

                const LO_HI: i32 = 0b0011_0000;

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y0y7, y8y1),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y2y9, y10y3),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y4y11, y12y5),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y6y13, y0y7),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y8y1, y2y9),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y10y3, y4y11),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_HI>(y12y5, y6y13),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly14<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        14
    }
}

impl AvxButterfly14<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 14 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(14) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u13 = _mm_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());

                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u10u11 = _mm256_extractf128_ps::<1>(u8u9u10u11);

                let (u0, u1) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(u0u1u2u3),
                    _mm_unpackhi_ps64(u6u7, u6u7),
                ); // 0,7
                let (u2, u3) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(u8u9u10u11),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u0u1u2u3),
                        _mm256_castps256_ps128(u0u1u2u3),
                    ),
                ); // 8,1
                let (u4, u5) = AvxButterfly::butterfly2_f32_m128(
                    u2u3,
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u8u9u10u11),
                        _mm256_castps256_ps128(u8u9u10u11),
                    ),
                ); // 2,9
                let (u6, u7) =
                    AvxButterfly::butterfly2_f32_m128(u10u11, _mm_unpackhi_ps64(u2u3, u2u3)); // 10, 3
                let (u8, u9) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(u4u5u6u7),
                    _mm_unpackhi_ps64(u10u11, u10u11),
                ); // 4, 11
                let (u10, u11) = AvxButterfly::butterfly2_f32_m128(
                    u12u13,
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u4u5u6u7),
                        _mm256_castps256_ps128(u4u5u6u7),
                    ),
                ); // 12, 5
                let (u12, u13) =
                    AvxButterfly::butterfly2_f32_m128(u6u7, _mm_unpackhi_ps64(u12u13, u12u13)); // 6, 13

                let (y0y7, y2y9, y4y11, y6y13, y8y1, y10y3, y12y5) = self.bf7.exec_short(
                    _mm_unpacklo_ps64(u0, u1),
                    _mm_unpacklo_ps64(u2, u3),
                    _mm_unpacklo_ps64(u4, u5),
                    _mm_unpacklo_ps64(u6, u7),
                    _mm_unpacklo_ps64(u8, u9),
                    _mm_unpacklo_ps64(u10, u11),
                    _mm_unpacklo_ps64(u12, u13),
                );

                const LO_HI: i32 = shuffle(3, 2, 1, 0);
                let y0000 = _mm256_create_ps(
                    _mm_shuffle_ps::<LO_HI>(y0y7, y8y1),
                    _mm_shuffle_ps::<LO_HI>(y2y9, y10y3),
                );
                let y0001 = _mm256_create_ps(
                    _mm_shuffle_ps::<LO_HI>(y4y11, y12y5),
                    _mm_shuffle_ps::<LO_HI>(y6y13, y0y7),
                );
                let y0002 = _mm256_create_ps(
                    _mm_shuffle_ps::<LO_HI>(y8y1, y2y9),
                    _mm_shuffle_ps::<LO_HI>(y10y3, y4y11),
                );
                let y0003 = _mm_shuffle_ps::<LO_HI>(y12y5, y6y13);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0000);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y0001);
                _mm256_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y0002);
                _mm_storeu_ps(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y0003);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly14<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        14
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::butterflies::Butterfly14;
    use rand::Rng;

    #[test]
    fn test_butterfly14_f32() {
        for i in 1..3 {
            let size = 14usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly14::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly14::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly14::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 14f32)).collect();

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
            let size = 14usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly14::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly14::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly14::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 14f64)).collect();

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
