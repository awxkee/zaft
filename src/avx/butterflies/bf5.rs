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

use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd,
    _mm256_create_ps, shuffle,
};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly5<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly5<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 5, fft_direction),
            twiddle2: compute_twiddle(2, 5, fft_direction),
        }
    }
}

impl AvxButterfly5<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 5 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let tw1_re = _mm256_set1_ps(self.twiddle1.re);
        let tw1_im = _mm256_set1_ps(self.twiddle1.im);
        let tw2_re = _mm256_set1_ps(self.twiddle2.re);
        let tw2_im = _mm256_set1_ps(self.twiddle2.im);
        let rot_sign =
            unsafe { _mm256_loadu_ps([-0.0f32, 0.0, -0.0, 0.0, -0.0f32, 0.0, -0.0, 0.0].as_ptr()) };

        unsafe {
            for chunk in in_place.chunks_exact_mut(10) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9 = _mm_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());

                let lo = _mm256_castps256_ps128(u0u1u2u3); // (u0,u1)
                let hi = _mm256_extractf128_ps::<1>(u0u1u2u3); // (u2, u3)

                let hi2 = _mm256_castps256_ps128(u4u5u6u7); // (u4,u5)
                let hi3 = _mm256_extractf128_ps::<1>(u4u5u6u7); // (u6, u7)

                let u0 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(lo, hi2); // (u0,u5)
                let u1 = _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(lo, hi3); // (u1,u6)
                let u2 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(hi, hi3); // (u2,u7)
                let u3 = _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(hi, u8u9); // (u3,u8)
                let u4 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(hi2, u8u9); // (u4,u9)

                // Radix-5 butterfly

                let x14p = _mm_add_ps(u1, u4);
                let x14n = _mm_sub_ps(u1, u4);
                let x23p = _mm_add_ps(u2, u3);
                let x23n = _mm_sub_ps(u2, u3);
                let y0 = _mm_add_ps(_mm_add_ps(u0, x14p), x23p);

                let temp_b1_1 = _mm_mul_ps(_mm256_castps256_ps128(tw1_im), x14n);
                let temp_b2_1 = _mm_mul_ps(_mm256_castps256_ps128(tw2_im), x14n);

                let temp_a1 = _mm_fmadd_ps(
                    _mm256_castps256_ps128(tw2_re),
                    x23p,
                    _mm_fmadd_ps(_mm256_castps256_ps128(tw1_re), x14p, u0),
                );
                let temp_a2 = _mm_fmadd_ps(
                    _mm256_castps256_ps128(tw1_re),
                    x23p,
                    _mm_fmadd_ps(_mm256_castps256_ps128(tw2_re), x14p, u0),
                );

                let temp_b1 = _mm_fmadd_ps(_mm256_castps256_ps128(tw2_im), x23n, temp_b1_1);
                let temp_b2 = _mm_fnmadd_ps(_mm256_castps256_ps128(tw1_im), x23n, temp_b2_1);

                const SH: i32 = shuffle(2, 3, 0, 1);
                let temp_b1_rot = _mm_xor_ps(
                    _mm_shuffle_ps::<SH>(temp_b1, temp_b1),
                    _mm256_castps256_ps128(rot_sign),
                );
                let temp_b2_rot = _mm_xor_ps(
                    _mm_shuffle_ps::<SH>(temp_b2, temp_b2),
                    _mm256_castps256_ps128(rot_sign),
                );

                let y1 = _mm_add_ps(temp_a1, temp_b1_rot);
                let y2 = _mm_add_ps(temp_a2, temp_b2_rot);
                let y3 = _mm_sub_ps(temp_a2, temp_b2_rot);
                let y4 = _mm_sub_ps(temp_a1, temp_b1_rot);

                let zu0 = _mm_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(y0, y1); // (u0,u5)
                let zu1 = _mm_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(y2, y3); // (u1,u6)
                let zu2 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(y4, y0); // (u2,u7)
                let zu3 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(y1, y2); // (u3,u8)
                let zu4 = _mm_unpackhi_ps64(y3, y4); // (u4,u9)

                let zu0u1 = _mm256_create_ps(zu0, zu1);
                let zu2u3 = _mm256_create_ps(zu2, zu3);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), zu0u1);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), zu2u3);
                _mm_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), zu4);
            }
        }

        let remainder = in_place.chunks_exact_mut(10).into_remainder();

        for chunk in remainder.chunks_exact_mut(5) {
            unsafe {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u0 = u0u1;
                let u1 = _mm_unpackhi_ps64(u0u1, u0u1);
                let u2 = u2u3;
                let u3 = _mm_unpackhi_ps64(u2u3, u2u3);
                let u4 = _m128s_load_f32x2(chunk.get_unchecked(4..).as_ptr().cast());

                // Radix-5 butterfly

                let x14p = _mm_add_ps(u1, u4);
                let x14n = _mm_sub_ps(u1, u4);
                let x23p = _mm_add_ps(u2, u3);
                let x23n = _mm_sub_ps(u2, u3);
                let y0 = _mm_add_ps(_mm_add_ps(u0, x14p), x23p);

                let temp_b1_1 = _mm_mul_ps(_mm256_castps256_ps128(tw1_im), x14n);
                let temp_b2_1 = _mm_mul_ps(_mm256_castps256_ps128(tw2_im), x14n);

                let temp_a1 = _mm_fmadd_ps(
                    _mm256_castps256_ps128(tw2_re),
                    x23p,
                    _mm_fmadd_ps(_mm256_castps256_ps128(tw1_re), x14p, u0),
                );
                let temp_a2 = _mm_fmadd_ps(
                    _mm256_castps256_ps128(tw1_re),
                    x23p,
                    _mm_fmadd_ps(_mm256_castps256_ps128(tw2_re), x14p, u0),
                );

                let temp_b1 = _mm_fmadd_ps(_mm256_castps256_ps128(tw2_im), x23n, temp_b1_1);
                let temp_b2 = _mm_fnmadd_ps(_mm256_castps256_ps128(tw1_im), x23n, temp_b2_1);

                const SH: i32 = shuffle(2, 3, 0, 1);
                let temp_b1_rot = _mm_xor_ps(
                    _mm_shuffle_ps::<SH>(temp_b1, temp_b1),
                    _mm256_castps256_ps128(rot_sign),
                );
                let temp_b2_rot = _mm_xor_ps(
                    _mm_shuffle_ps::<SH>(temp_b2, temp_b2),
                    _mm256_castps256_ps128(rot_sign),
                );

                let y1 = _mm_add_ps(temp_a1, temp_b1_rot);
                let y2 = _mm_add_ps(temp_a2, temp_b2_rot);
                let y3 = _mm_sub_ps(temp_a2, temp_b2_rot);
                let y4 = _mm_sub_ps(temp_a1, temp_b1_rot);

                let y0y1y2y3 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y0, y1), _mm_unpacklo_ps64(y2, y3));

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1y2y3);
                _m128s_store_f32x2(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly5<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        5
    }
}

impl AvxButterfly5<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 5 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let tw1_re = _mm256_set1_pd(self.twiddle1.re);
        let tw1_im = _mm256_set1_pd(self.twiddle1.im);
        let tw2_re = _mm256_set1_pd(self.twiddle2.re);
        let tw2_im = _mm256_set1_pd(self.twiddle2.im);
        let rot_sign = unsafe { _mm256_loadu_pd([-0.0f64, 0.0, -0.0, 0.0].as_ptr()) };

        for chunk in in_place.chunks_exact_mut(10) {
            unsafe {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u10 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());

                const LO_HI: i32 = 0b0011_0000;
                const HI_LO: i32 = 0b0010_0001;
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u0 = _mm256_permute2f128_pd::<LO_HI>(u0u1, u4u5);
                let u1 = _mm256_permute2f128_pd::<HI_LO>(u0u1, u6u7);
                let u2 = _mm256_permute2f128_pd::<LO_HI>(u2u3, u6u7);
                let u3 = _mm256_permute2f128_pd::<HI_LO>(u2u3, u8u10);
                let u4 = _mm256_permute2f128_pd::<LO_HI>(u4u5, u8u10);

                // Radix-5 butterfly

                let x14p = _mm256_add_pd(u1, u4);
                let x14n = _mm256_sub_pd(u1, u4);
                let x23p = _mm256_add_pd(u2, u3);
                let x23n = _mm256_sub_pd(u2, u3);
                let y0 = _mm256_add_pd(_mm256_add_pd(u0, x14p), x23p);

                let temp_b1_1 = _mm256_mul_pd(tw1_im, x14n);
                let temp_b2_1 = _mm256_mul_pd(tw2_im, x14n);

                let temp_a1 = _mm256_fmadd_pd(tw2_re, x23p, _mm256_fmadd_pd(tw1_re, x14p, u0));
                let temp_a2 = _mm256_fmadd_pd(tw1_re, x23p, _mm256_fmadd_pd(tw2_re, x14p, u0));

                let temp_b1 = _mm256_fmadd_pd(tw2_im, x23n, temp_b1_1);
                let temp_b2 = _mm256_fnmadd_pd(tw1_im, x23n, temp_b2_1);

                let temp_b1_rot = _mm256_xor_pd(_mm256_permute_pd::<0b0101>(temp_b1), rot_sign);
                let temp_b2_rot = _mm256_xor_pd(_mm256_permute_pd::<0b0101>(temp_b2), rot_sign);

                let y1 = _mm256_add_pd(temp_a1, temp_b1_rot);
                let y2 = _mm256_add_pd(temp_a2, temp_b2_rot);
                let y3 = _mm256_sub_pd(temp_a2, temp_b2_rot);
                let y4 = _mm256_sub_pd(temp_a1, temp_b1_rot);

                let u0u1 = _mm256_permute2f128_pd::<LO_LO>(y0, y1);
                let u2u3 = _mm256_permute2f128_pd::<LO_LO>(y2, y3);
                let u4u5 = _mm256_permute2f128_pd::<LO_HI>(y4, y0);
                let u6u7 = _mm256_permute2f128_pd::<HI_HI>(y1, y2);
                let u8u9 = _mm256_permute2f128_pd::<HI_HI>(y3, y4);

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), u0u1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), u2u3);
                _mm256_storeu_pd(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), u4u5);
                _mm256_storeu_pd(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), u6u7);
                _mm256_storeu_pd(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), u8u9);
            }
        }

        let rem = in_place.chunks_exact_mut(10).into_remainder();

        for chunk in rem.chunks_exact_mut(5) {
            unsafe {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u0 = _mm256_castpd256_pd128(u0u1);
                let u1 = _mm256_extractf128_pd::<1>(u0u1);
                let u2 = _mm256_castpd256_pd128(u2u3);
                let u3 = _mm256_extractf128_pd::<1>(u2u3);
                let u4 = _mm_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());

                // Radix-5 butterfly

                let x14p = _mm_add_pd(u1, u4);
                let x14n = _mm_sub_pd(u1, u4);
                let x23p = _mm_add_pd(u2, u3);
                let x23n = _mm_sub_pd(u2, u3);
                let y0 = _mm_add_pd(_mm_add_pd(u0, x14p), x23p);

                let temp_b1_1 = _mm_mul_pd(_mm256_castpd256_pd128(tw1_im), x14n);
                let temp_b2_1 = _mm_mul_pd(_mm256_castpd256_pd128(tw2_im), x14n);

                let temp_a1 = _mm_fmadd_pd(
                    _mm256_castpd256_pd128(tw2_re),
                    x23p,
                    _mm_fmadd_pd(_mm256_castpd256_pd128(tw1_re), x14p, u0),
                );
                let temp_a2 = _mm_fmadd_pd(
                    _mm256_castpd256_pd128(tw1_re),
                    x23p,
                    _mm_fmadd_pd(_mm256_castpd256_pd128(tw2_re), x14p, u0),
                );

                let temp_b1 = _mm_fmadd_pd(_mm256_castpd256_pd128(tw2_im), x23n, temp_b1_1);
                let temp_b2 = _mm_fnmadd_pd(_mm256_castpd256_pd128(tw1_im), x23n, temp_b2_1);

                let temp_b1_rot = _mm_xor_pd(
                    _mm_shuffle_pd::<0b01>(temp_b1, temp_b1),
                    _mm256_castpd256_pd128(rot_sign),
                );
                let temp_b2_rot = _mm_xor_pd(
                    _mm_shuffle_pd::<0b01>(temp_b2, temp_b2),
                    _mm256_castpd256_pd128(rot_sign),
                );

                let y1 = _mm_add_pd(temp_a1, temp_b1_rot);
                let y2 = _mm_add_pd(temp_a2, temp_b2_rot);
                let y3 = _mm_sub_pd(temp_a2, temp_b2_rot);
                let y4 = _mm_sub_pd(temp_a1, temp_b1_rot);

                let y0y1 = _mm256_create_pd(y0, y1);
                let y2y3 = _mm256_create_pd(y2, y3);

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2y3);
                _mm_storeu_pd(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly5<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    #[test]
    fn test_butterfly5_f32() {
        for i in 1..6 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly5::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly5::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 5f32)).collect();

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
    fn test_butterfly5_f64() {
        for i in 1..6 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly5::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly5::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 5f64)).collect();

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
