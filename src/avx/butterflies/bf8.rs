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

use crate::avx::butterflies::AvxButterfly;
use crate::avx::util::{
    _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps,
    _mm256s_deinterleave2_epi64, _mm256s_interleave2_epi64, shuffle,
};
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly8<T> {
    direction: FftDirection,
    multiplier: [T; 8],
    root2: T,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly8<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            multiplier: match fft_direction {
                FftDirection::Inverse => [
                    -0.0f64.as_(),
                    0.0.as_(),
                    -0.0.as_(),
                    0.0.as_(),
                    -0.0f64.as_(),
                    0.0.as_(),
                    -0.0.as_(),
                    0.0.as_(),
                ],
                FftDirection::Forward => [
                    0.0f64.as_(),
                    -0.0.as_(),
                    0.0.as_(),
                    -0.0.as_(),
                    0.0f64.as_(),
                    -0.0.as_(),
                    0.0.as_(),
                    -0.0.as_(),
                ],
            },
            root2: 0.5f64.sqrt().as_(),
        }
    }
}

impl AvxButterfly8<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 8 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let z_mul = _mm256_loadu_pd(self.multiplier.as_ptr());
            static ROT_SIGN_INVERSE: [f64; 4] = [-0.0, 0.0, -0.0, 0.0];
            static ROT_SIGN_FORWARD: [f64; 4] = [0.0, -0.0, 0.0, -0.0];
            let rot_sign = _mm256_loadu_pd(match self.direction {
                FftDirection::Inverse => ROT_SIGN_INVERSE.as_ptr(),
                FftDirection::Forward => ROT_SIGN_FORWARD.as_ptr(),
            });
            let root2 = _mm256_set1_pd(self.root2);

            for chunk in in_place.chunks_exact_mut(16) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = _mm256_loadu_pd(chunk.get_unchecked(14..).as_ptr().cast());

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u0 = _mm256_permute2f128_pd::<LO_LO>(u0u1, u8u9);
                let u1 = _mm256_permute2f128_pd::<HI_HI>(u0u1, u8u9);
                let u2 = _mm256_permute2f128_pd::<LO_LO>(u2u3, u10u11);
                let u3 = _mm256_permute2f128_pd::<HI_HI>(u2u3, u10u11);
                let u4 = _mm256_permute2f128_pd::<LO_LO>(u4u5, u12u13);
                let u5 = _mm256_permute2f128_pd::<HI_HI>(u4u5, u12u13);
                let u6 = _mm256_permute2f128_pd::<LO_LO>(u6u7, u14u15);
                let u7 = _mm256_permute2f128_pd::<HI_HI>(u6u7, u14u15);

                let (u0, u2, u4, u6) = AvxButterfly::butterfly4_f64(u0, u2, u4, u6, z_mul);
                let (u1, mut u3, mut u5, mut u7) =
                    AvxButterfly::butterfly4_f64(u1, u3, u5, u7, z_mul);

                u3 = _mm256_mul_pd(
                    _mm256_add_pd(_mm256_xor_pd(_mm256_permute_pd::<0b0101>(u3), rot_sign), u3),
                    root2,
                );
                u5 = _mm256_xor_pd(_mm256_permute_pd::<0b0101>(u5), rot_sign);
                u7 = _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_xor_pd(_mm256_permute_pd::<0b0101>(u7), rot_sign), u7),
                    root2,
                );

                let (y0, y1) = AvxButterfly::butterfly2_f64(u0, u1);
                let (y2, y3) = AvxButterfly::butterfly2_f64(u2, u3);
                let (y4, y5) = AvxButterfly::butterfly2_f64(u4, u5);
                let (y6, y7) = AvxButterfly::butterfly2_f64(u6, u7);

                let q0 = _mm256_permute2f128_pd::<LO_LO>(y0, y2);
                let q1 = _mm256_permute2f128_pd::<LO_LO>(y4, y6);
                let q2 = _mm256_permute2f128_pd::<LO_LO>(y1, y3);
                let q3 = _mm256_permute2f128_pd::<LO_LO>(y5, y7);

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), q0);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), q1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), q2);
                _mm256_storeu_pd(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), q3);

                let q4 = _mm256_permute2f128_pd::<HI_HI>(y0, y2);
                let q5 = _mm256_permute2f128_pd::<HI_HI>(y4, y6);
                let q6 = _mm256_permute2f128_pd::<HI_HI>(y1, y3);
                let q7 = _mm256_permute2f128_pd::<HI_HI>(y5, y7);

                _mm256_storeu_pd(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), q4);
                _mm256_storeu_pd(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), q5);
                _mm256_storeu_pd(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), q6);
                _mm256_storeu_pd(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), q7);
            }

            let rem = in_place.chunks_exact_mut(16).into_remainder();

            for chunk in rem.chunks_exact_mut(8) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(u0u1);
                let u1 = _mm256_extractf128_pd::<1>(u0u1);
                let u2 = _mm256_castpd256_pd128(u2u3);
                let u3 = _mm256_extractf128_pd::<1>(u2u3);
                let u4 = _mm256_castpd256_pd128(u4u5);
                let u5 = _mm256_extractf128_pd::<1>(u4u5);
                let u6 = _mm256_castpd256_pd128(u6u7);
                let u7 = _mm256_extractf128_pd::<1>(u6u7);

                let (u0, u2, u4, u6) =
                    AvxButterfly::butterfly4h_f64(u0, u2, u4, u6, _mm256_castpd256_pd128(z_mul));
                let (u1, mut u3, mut u5, mut u7) =
                    AvxButterfly::butterfly4h_f64(u1, u3, u5, u7, _mm256_castpd256_pd128(z_mul));

                u3 = _mm_mul_pd(
                    _mm_add_pd(
                        _mm_xor_pd(
                            _mm_shuffle_pd::<0b01>(u3, u3),
                            _mm256_castpd256_pd128(rot_sign),
                        ),
                        u3,
                    ),
                    _mm256_castpd256_pd128(root2),
                );
                u5 = _mm_xor_pd(
                    _mm_shuffle_pd::<0b01>(u5, u5),
                    _mm256_castpd256_pd128(rot_sign),
                );
                u7 = _mm_mul_pd(
                    _mm_sub_pd(
                        _mm_xor_pd(
                            _mm_shuffle_pd::<0b01>(u7, u7),
                            _mm256_castpd256_pd128(rot_sign),
                        ),
                        u7,
                    ),
                    _mm256_castpd256_pd128(root2),
                );

                let (y0, y1) = AvxButterfly::butterfly2_f64_m128(u0, u1);
                let (y2, y3) = AvxButterfly::butterfly2_f64_m128(u2, u3);
                let (y4, y5) = AvxButterfly::butterfly2_f64_m128(u4, u5);
                let (y6, y7) = AvxButterfly::butterfly2_f64_m128(u6, u7);

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm256_create_pd(y0, y2),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(y4, y6),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(y1, y3),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(y5, y7),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly8<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        8
    }
}

impl AvxButterfly8<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 8 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let z_mul = _mm256_loadu_ps(self.multiplier.as_ptr());
            static ROT_SIGN_INVERSE: [f32; 8] = [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0];
            static ROT_SIGN_FORWARD: [f32; 8] = [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0];
            let rot_sign = _mm256_loadu_ps(match self.direction {
                FftDirection::Inverse => ROT_SIGN_INVERSE.as_ptr(),
                FftDirection::Forward => ROT_SIGN_FORWARD.as_ptr(),
            });
            let root2 = _mm256_set1_ps(self.root2);

            for chunk in in_place.chunks_exact_mut(16) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());

                let (u0u8u1u9, u2u8u3u11) = _mm256s_interleave2_epi64(u0u1u2u3, u8u9u10u11);
                let (u4u12u5u13, u6u14u7u15) = _mm256s_interleave2_epi64(u4u5u6u7, u12u13u14u15);

                let u0 = _mm256_castps256_ps128(u0u8u1u9);
                let u1 = _mm256_extractf128_ps::<1>(u0u8u1u9);
                let u2 = _mm256_castps256_ps128(u2u8u3u11);
                let u3 = _mm256_extractf128_ps::<1>(u2u8u3u11);
                let u4 = _mm256_castps256_ps128(u4u12u5u13);
                let u5 = _mm256_extractf128_ps::<1>(u4u12u5u13);
                let u6 = _mm256_castps256_ps128(u6u14u7u15);
                let u7 = _mm256_extractf128_ps::<1>(u6u14u7u15);

                let (u0, u2, u4, u6) =
                    AvxButterfly::butterfly4h_f32(u0, u2, u4, u6, _mm256_castps256_ps128(z_mul));
                let (u1, mut u3, mut u5, mut u7) =
                    AvxButterfly::butterfly4h_f32(u1, u3, u5, u7, _mm256_castps256_ps128(z_mul));

                const SH: i32 = shuffle(2, 3, 0, 1);

                u3 = _mm_mul_ps(
                    _mm_add_ps(
                        _mm_xor_ps(
                            _mm_shuffle_ps::<SH>(u3, u3),
                            _mm256_castps256_ps128(rot_sign),
                        ),
                        u3,
                    ),
                    _mm256_castps256_ps128(root2),
                );
                u5 = _mm_xor_ps(
                    _mm_shuffle_ps::<SH>(u5, u5),
                    _mm256_castps256_ps128(rot_sign),
                );
                u7 = _mm_mul_ps(
                    _mm_sub_ps(
                        _mm_xor_ps(
                            _mm_shuffle_ps::<SH>(u7, u7),
                            _mm256_castps256_ps128(rot_sign),
                        ),
                        u7,
                    ),
                    _mm256_castps256_ps128(root2),
                );

                let (zy0, zy1) = AvxButterfly::butterfly2_f32_m128(u0, u1);
                let (zy2, zy3) = AvxButterfly::butterfly2_f32_m128(u2, u3);
                let (zy4, zy5) = AvxButterfly::butterfly2_f32_m128(u4, u5);
                let (zy6, zy7) = AvxButterfly::butterfly2_f32_m128(u6, u7);

                let y0y1 = _mm256_create_ps(zy0, zy2);
                let y2y3 = _mm256_create_ps(zy4, zy6);
                let y4y5 = _mm256_create_ps(zy1, zy3);
                let y6y7 = _mm256_create_ps(zy5, zy7);

                let (row0, row2) = _mm256s_deinterleave2_epi64(y0y1, y2y3);
                let (row1, row3) = _mm256s_deinterleave2_epi64(y4y5, y6y7);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), row0);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), row1);
                _mm256_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), row2);
                _mm256_storeu_ps(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), row3);
            }

            let rem = in_place.chunks_exact_mut(16).into_remainder();

            for chunk in rem.chunks_exact_mut(8) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());

                let u1u2 = _mm256_castps256_ps128(u0u1u2u3);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);

                let u0 = u1u2;
                let u1 = _mm_unpackhi_ps64(u1u2, u1u2);
                let u2 = u2u3;
                let u3 = _mm_unpackhi_ps64(u2u3, u2u3);
                let u4 = u4u5;
                let u5 = _mm_unpackhi_ps64(u4u5, u4u5);
                let u6 = u6u7;
                let u7 = _mm_unpackhi_ps64(u6u7, u6u7);

                let (u0, u2, u4, u6) =
                    AvxButterfly::butterfly4h_f32(u0, u2, u4, u6, _mm256_castps256_ps128(z_mul));
                let (u1, mut u3, mut u5, mut u7) =
                    AvxButterfly::butterfly4h_f32(u1, u3, u5, u7, _mm256_castps256_ps128(z_mul));

                const SH: i32 = shuffle(2, 3, 0, 1);

                u3 = _mm_mul_ps(
                    _mm_add_ps(
                        _mm_xor_ps(
                            _mm_shuffle_ps::<SH>(u3, u3),
                            _mm256_castps256_ps128(rot_sign),
                        ),
                        u3,
                    ),
                    _mm256_castps256_ps128(root2),
                );
                u5 = _mm_xor_ps(
                    _mm_shuffle_ps::<SH>(u5, u5),
                    _mm256_castps256_ps128(rot_sign),
                );
                u7 = _mm_mul_ps(
                    _mm_sub_ps(
                        _mm_xor_ps(
                            _mm_shuffle_ps::<SH>(u7, u7),
                            _mm256_castps256_ps128(rot_sign),
                        ),
                        u7,
                    ),
                    _mm256_castps256_ps128(root2),
                );

                let (zy0, zy1) = AvxButterfly::butterfly2_f32_m128(u0, u1);
                let (zy2, zy3) = AvxButterfly::butterfly2_f32_m128(u2, u3);
                let (zy4, zy5) = AvxButterfly::butterfly2_f32_m128(u4, u5);
                let (zy6, zy7) = AvxButterfly::butterfly2_f32_m128(u6, u7);

                let y0y1 = _mm_unpacklo_ps64(zy0, zy2);
                let y2y3 = _mm_unpacklo_ps64(zy4, zy6);
                let y4y5 = _mm_unpacklo_ps64(zy1, zy3);
                let y6y7 = _mm_unpacklo_ps64(zy5, zy7);

                let y0y1y2y3 = _mm256_create_ps(y0y1, y2y3);
                let y4y5y6y7 = _mm256_create_ps(y4y5, y6y7);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1y2y3);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4y5y6y7);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly8<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        8
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_butterfly8_f32() {
        for i in 1..5 {
            let size = 8usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly8::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly8::new(FftDirection::Inverse);

            let radix_forward_ref = AvxButterfly8::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 8f32)).collect();

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
    fn test_butterfly8_f64() {
        for i in 1..6 {
            let size = 8usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly8::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly8::new(FftDirection::Inverse);

            let radix_forward_ref = AvxButterfly8::new(FftDirection::Forward);

            radix_forward.execute(&mut input).unwrap();
            radix_forward_ref.execute(&mut ref0).unwrap();

            input
                .iter()
                .zip(ref0.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "forward at {idx} a_re {} != b_re {} for size {}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "forward at {idx} a_im {} != b_im {} for size {}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 8f64)).collect();

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
