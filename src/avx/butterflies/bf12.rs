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

use crate::avx::butterflies::fast_bf3::AvxFastButterfly3;
use crate::avx::butterflies::fast_bf4::AvxFastButterfly4;
use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps};
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;
use std::marker::PhantomData;

pub(crate) struct AvxButterfly12<T> {
    direction: FftDirection,
    phantom_data: PhantomData<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly12<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            phantom_data: PhantomData,
        }
    }
}

impl AvxButterfly12<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 12 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let bf4 = AvxFastButterfly4::<f64>::new(self.direction);
            let bf3 = AvxFastButterfly3::<f64>::new(self.direction);

            for chunk in in_place.chunks_exact_mut(12) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());

                let zu0 = _mm256_castpd256_pd128(u0u1);
                let zu1 = _mm256_extractf128_pd::<1>(u2u3);
                let zu2 = _mm256_castpd256_pd128(u6u7);
                let zu3 = _mm256_extractf128_pd::<1>(u8u9);
                let zu4 = _mm256_castpd256_pd128(u4u5);
                let zu5 = _mm256_extractf128_pd::<1>(u6u7);
                let zu6 = _mm256_castpd256_pd128(u10u11);
                let zu7 = _mm256_extractf128_pd::<1>(u0u1);
                let zu8 = _mm256_castpd256_pd128(u8u9);
                let zu9 = _mm256_extractf128_pd::<1>(u10u11);
                let zu10 = _mm256_castpd256_pd128(u2u3);
                let zu11 = _mm256_extractf128_pd::<1>(u4u5);

                let (u0, u1, u2, u3) = bf4.exec_short(zu0, zu3, zu6, zu9);
                let (u4, u5, u6, u7) = bf4.exec_short(zu4, zu7, zu10, zu1);
                let (u8, u9, u10, u11) = bf4.exec_short(zu8, zu11, zu2, zu5);

                let (v0, v4, v8) = bf3.exec_short(u0, u4, u8); // (v0, v4, v8)
                let (v9, v1, v5) = bf3.exec_short(u1, u5, u9); // (v9, v1, v5)
                let (v6, v10, v2) = bf3.exec_short(u2, u6, u10); // (v6, v10, v2)
                let (v3, v7, v11) = bf3.exec_short(u3, u7, u11); // (v3, v7, v11)

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm256_create_pd(v0, v1),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(v2, v3),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(v4, v5),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(v6, v7),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(v8, v9),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_create_pd(v10, v11),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly12<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        12
    }
}

impl AvxButterfly12<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 12 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let bf4 = AvxFastButterfly4::<f32>::new(self.direction);
            let bf3 = AvxFastButterfly3::<f32>::new(self.direction);

            for chunk in in_place.chunks_exact_mut(12) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);
                let u10u11 = _mm256_extractf128_ps::<1>(u8u9u10u11);

                let zu0 = _mm256_castps256_ps128(u0u1u2u3);
                let zu1 = _mm_unpackhi_ps64(u2u3, u2u3);
                let zu2 = u6u7;
                let zu3 = _mm_unpackhi_ps64(
                    _mm256_castps256_ps128(u8u9u10u11),
                    _mm256_castps256_ps128(u8u9u10u11),
                );
                let zu4 = _mm256_castps256_ps128(u4u5u6u7);
                let zu5 = _mm_unpackhi_ps64(u6u7, u6u7);
                let zu6 = u10u11;
                let zu7 = _mm_unpackhi_ps64(
                    _mm256_castps256_ps128(u0u1u2u3),
                    _mm256_castps256_ps128(u0u1u2u3),
                );
                let zu8 = _mm256_castps256_ps128(u8u9u10u11);
                let zu9 = _mm_unpackhi_ps64(u10u11, u10u11);
                let zu10 = u2u3;
                let zu11 = _mm_unpackhi_ps64(
                    _mm256_castps256_ps128(u4u5u6u7),
                    _mm256_castps256_ps128(u4u5u6u7),
                );

                let (u0, u1, u2, u3) = bf4.exec_short(zu0, zu3, zu6, zu9);
                let (u4, u5, u6, u7) = bf4.exec_short(zu4, zu7, zu10, zu1);
                let (u8, u9, u10, u11) = bf4.exec_short(zu8, zu11, zu2, zu5);

                let (v0, v4, v8) = bf3.exec_short(u0, u4, u8); // (v0, v4, v8)
                let (v9, v1, v5) = bf3.exec_short(u1, u5, u9); // (v9, v1, v5)
                let (v6, v10, v2) = bf3.exec_short(u2, u6, u10); // (v6, v10, v2)
                let (v3, v7, v11) = bf3.exec_short(u3, u7, u11); // (v3, v7, v11)

                let y0000 = _mm256_create_ps(_mm_unpacklo_ps64(v0, v1), _mm_unpacklo_ps64(v2, v3));
                let y0001 = _mm256_create_ps(_mm_unpacklo_ps64(v4, v5), _mm_unpacklo_ps64(v6, v7));
                let y0002 =
                    _mm256_create_ps(_mm_unpacklo_ps64(v8, v9), _mm_unpacklo_ps64(v10, v11));

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0000);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y0001);
                _mm256_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y0002);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly12<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        12
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::butterflies::Butterfly12;
    use rand::Rng;

    #[test]
    fn test_butterfly12_f32() {
        for i in 1..4 {
            let size = 12usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly12::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly12::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly12::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 12f32)).collect();

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
    fn test_butterfly12_f64() {
        for i in 1..4 {
            let size = 12usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly12::<f64>::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly12::<f64>::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly12::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 12f64)).collect();

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
