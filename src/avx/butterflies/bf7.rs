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
use crate::avx::util::{_mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps, shuffle};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

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

                let (x1p6, x1m6) = AvxButterfly::butterfly2_f64_m128(u1, u6);
                let x1m6 = rotate.rotate_m128d(x1m6);
                let y00 = _mm_add_pd(u0, x1p6);
                let (x2p5, x2m5) = AvxButterfly::butterfly2_f64_m128(u2, u5);
                let x2m5 = rotate.rotate_m128d(x2m5);
                let y00 = _mm_add_pd(y00, x2p5);
                let (x3p4, x3m4) = AvxButterfly::butterfly2_f64_m128(u3, u4);
                let x3m4 = rotate.rotate_m128d(x3m4);
                let y00 = _mm_add_pd(y00, x3p4);

                let m0106a = _mm_fmadd_pd(x1p6, _mm_set1_pd(self.twiddle1.re), u0);
                let m0106a = _mm_fmadd_pd(x2p5, _mm_set1_pd(self.twiddle2.re), m0106a);
                let m0106a = _mm_fmadd_pd(x3p4, _mm_set1_pd(self.twiddle3.re), m0106a);
                let m0106b = _mm_mul_pd(x1m6, _mm_set1_pd(self.twiddle1.im));
                let m0106b = _mm_fmadd_pd(x2m5, _mm_set1_pd(self.twiddle2.im), m0106b);
                let m0106b = _mm_fmadd_pd(x3m4, _mm_set1_pd(self.twiddle3.im), m0106b);
                let (y01, y06) = AvxButterfly::butterfly2_f64_m128(m0106a, m0106b);

                let m0205a = _mm_fmadd_pd(x1p6, _mm_set1_pd(self.twiddle2.re), u0);
                let m0205a = _mm_fmadd_pd(x2p5, _mm_set1_pd(self.twiddle3.re), m0205a);
                let m0205a = _mm_fmadd_pd(x3p4, _mm_set1_pd(self.twiddle1.re), m0205a);
                let m0205b = _mm_mul_pd(x1m6, _mm_set1_pd(self.twiddle2.im));
                let m0205b = _mm_fnmadd_pd(x2m5, _mm_set1_pd(self.twiddle3.im), m0205b);
                let m0205b = _mm_fnmadd_pd(x3m4, _mm_set1_pd(self.twiddle1.im), m0205b);
                let (y02, y05) = AvxButterfly::butterfly2_f64_m128(m0205a, m0205b);

                let m0304a = _mm_fmadd_pd(x1p6, _mm_set1_pd(self.twiddle3.re), u0);
                let m0304a = _mm_fmadd_pd(x2p5, _mm_set1_pd(self.twiddle1.re), m0304a);
                let m0304a = _mm_fmadd_pd(x3p4, _mm_set1_pd(self.twiddle2.re), m0304a);
                let m0304b = _mm_mul_pd(x1m6, _mm_set1_pd(self.twiddle3.im));
                let m0304b = _mm_fnmadd_pd(x2m5, _mm_set1_pd(self.twiddle1.im), m0304b);
                let m0304b = _mm_fmadd_pd(x3m4, _mm_set1_pd(self.twiddle2.im), m0304b);
                let (y03, y04) = AvxButterfly::butterfly2_f64_m128(m0304a, m0304b);

                let y0y1 = _mm256_create_pd(y00, y01);
                let y2y3 = _mm256_create_pd(y02, y03);
                let y4y5 = _mm256_create_pd(y04, y05);

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2y3);
                _mm256_storeu_pd(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4y5);
                _mm_storeu_pd(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y06);
            }
        }
        Ok(())
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

            for chunk in in_place.chunks_exact_mut(7) {
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

                let (x1p6, x1m6) = AvxButterfly::butterfly2_f32_m128(u1, u6);
                let x1m6 = rotate.rotate_m128(x1m6);
                let y00 = _mm_add_ps(u0, x1p6);
                let (x2p5, x2m5) = AvxButterfly::butterfly2_f32_m128(u2, u5);
                let x2m5 = rotate.rotate_m128(x2m5);
                let y00 = _mm_add_ps(y00, x2p5);
                let (x3p4, x3m4) = AvxButterfly::butterfly2_f32_m128(u3, u4);
                let x3m4 = rotate.rotate_m128(x3m4);
                let y00 = _mm_add_ps(y00, x3p4);

                let m0106a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle1.re), u0);
                let m0106a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle2.re), m0106a);
                let m0106a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle3.re), m0106a);
                let m0106b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle1.im));
                let m0106b = _mm_fmadd_ps(x2m5, _mm_set1_ps(self.twiddle2.im), m0106b);
                let m0106b = _mm_fmadd_ps(x3m4, _mm_set1_ps(self.twiddle3.im), m0106b);
                let (y01, y06) = AvxButterfly::butterfly2_f32_m128(m0106a, m0106b);

                let m0205a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle2.re), u0);
                let m0205a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle3.re), m0205a);
                let m0205a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle1.re), m0205a);
                let m0205b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle2.im));
                let m0205b = _mm_fnmadd_ps(x2m5, _mm_set1_ps(self.twiddle3.im), m0205b);
                let m0205b = _mm_fnmadd_ps(x3m4, _mm_set1_ps(self.twiddle1.im), m0205b);
                let (y02, y05) = AvxButterfly::butterfly2_f32_m128(m0205a, m0205b);

                let m0304a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle3.re), u0);
                let m0304a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle1.re), m0304a);
                let m0304a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle2.re), m0304a);
                let m0304b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle3.im));
                let m0304b = _mm_fnmadd_ps(x2m5, _mm_set1_ps(self.twiddle1.im), m0304b);
                let m0304b = _mm_fmadd_ps(x3m4, _mm_set1_ps(self.twiddle2.im), m0304b);
                let (y03, y04) = AvxButterfly::butterfly2_f32_m128(m0304a, m0304b);

                let y0y1y2y3 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y00, y01), _mm_unpacklo_ps64(y02, y03));
                let y3y4y5y6 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y03, y04), _mm_unpacklo_ps64(y05, y06));

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1y2y3);
                _mm256_storeu_ps(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3y4y5y6);
            }
        }
        Ok(())
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
    use rand::Rng;

    #[test]
    fn test_butterfly7_f32() {
        for i in 1..5 {
            let size = 7usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly7::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly7::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 7f32)).collect();

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
    fn test_butterfly7_f64() {
        for i in 1..5 {
            let size = 7usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly7::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly7::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 7f64)).collect();

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
