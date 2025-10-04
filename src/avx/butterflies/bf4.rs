// Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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

use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64, shuffle};
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly4<T> {
    direction: FftDirection,
    multiplier: [T; 4],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly4<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            multiplier: match fft_direction {
                FftDirection::Inverse => [-0.0.as_(), 0.0.as_(), -0.0.as_(), 0.0.as_()],
                FftDirection::Forward => [0.0.as_(), -0.0.as_(), 0.0.as_(), -0.0.as_()],
            },
        }
    }
}

impl AvxButterfly4<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let v_i_multiplier = unsafe { _mm_loadu_ps(self.multiplier.as_ptr()) };

        for chunk in in_place.chunks_exact_mut(4) {
            unsafe {
                let aaaa0 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());

                let a = _mm256_castps256_ps128(aaaa0);
                let b =
                    _mm_unpackhi_ps64(_mm256_castps256_ps128(aaaa0), _mm256_castps256_ps128(aaaa0));

                let aa0 = _mm256_extractf128_ps::<1>(aaaa0);

                let c = aa0;
                let d = _mm_unpackhi_ps64(aa0, aa0);

                let t0 = _mm_add_ps(a, c);
                let t1 = _mm_sub_ps(a, c);
                let t2 = _mm_add_ps(b, d);
                let mut t3 = _mm_sub_ps(b, d);
                const SH: i32 = shuffle(2, 3, 0, 1);
                t3 = _mm_xor_ps(_mm_shuffle_ps::<SH>(t3, t3), v_i_multiplier);

                let yy0 = _mm_unpacklo_ps64(_mm_add_ps(t0, t2), _mm_add_ps(t1, t3));
                let yy1 = _mm_unpacklo_ps64(_mm_sub_ps(t0, t2), _mm_sub_ps(t1, t3));
                let yyyy = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(yy0), yy1);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), yyyy);
            }
        }
        Ok(())
    }
}

impl AvxButterfly4<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let v_i_multiplier = unsafe { _mm_loadu_pd(self.multiplier.as_ptr()) };

        for chunk in in_place.chunks_exact_mut(4) {
            unsafe {
                let aa0 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let bb0 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());

                let a = _mm256_castpd256_pd128(aa0);
                let b = _mm256_extractf128_pd::<1>(aa0);
                let c = _mm256_castpd256_pd128(bb0);
                let d = _mm256_extractf128_pd::<1>(bb0);

                let t0 = _mm_add_pd(a, c);
                let t1 = _mm_sub_pd(a, c);
                let t2 = _mm_add_pd(b, d);
                let mut t3 = _mm_sub_pd(b, d);
                t3 = _mm_xor_pd(_mm_shuffle_pd::<0b01>(t3, t3), v_i_multiplier);

                let yy0 = _mm256_insertf128_pd::<1>(
                    _mm256_castpd128_pd256(_mm_add_pd(t0, t2)),
                    _mm_add_pd(t1, t3),
                );
                let yy1 = _mm256_insertf128_pd::<1>(
                    _mm256_castpd128_pd256(_mm_sub_pd(t0, t2)),
                    _mm_sub_pd(t1, t3),
                );

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), yy0);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), yy1);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly4<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        4
    }
}

impl FftExecutor<f64> for AvxButterfly4<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_butterfly4_f32() {
        for i in 1..6 {
            let size = 4usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly4::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly4::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 4f32)).collect();

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

    #[test]
    fn test_butterfly4_f64() {
        for i in 1..6 {
            let size = 4usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly4::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly4::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 4f64)).collect();

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
}
