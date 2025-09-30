/*
 * // Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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
use crate::neon::util::{fcma_complex_f32, fcma_complex_f64, fcmah_complex_f32};
use crate::radix3::Radix3Twiddles;
use crate::traits::FftTrigonometry;
use crate::util::{compute_twiddle, digit_reverse_indices, permute_inplace};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaRadix3<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    twiddle_re: T,
    twiddle_im: [T; 4],
    direction: FftDirection,
}

impl<T: Default + Clone + Radix3Twiddles + 'static + Copy + FftTrigonometry + Float>
    NeonFcmaRadix3<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonFcmaRadix3<T>, ZaftError> {
        assert!(
            size.is_power_of_two() || size % 3 == 0,
            "Input length must be divisible by 3"
        );

        let twiddles = T::make_twiddles(size, fft_direction)?;
        let rev = digit_reverse_indices(size, 3)?;

        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);

        Ok(NeonFcmaRadix3 {
            permutations: rev,
            execution_length: size,
            twiddles,
            twiddle_re: twiddle.re,
            twiddle_im: [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im],
            direction: fft_direction,
        })
    }
}

impl NeonFcmaRadix3<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let twiddle_re = vdupq_n_f64(self.twiddle_re);
        let twiddle_w_2 = unsafe { vld1q_f64(self.twiddle_im.as_ptr().cast()) };

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // Trit-reversal permutation
            permute_inplace(chunk, &self.permutations);

            let mut len = 3;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len <= self.execution_length {
                    let third = len / 3;
                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < third {
                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + third..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                            );
                            let u2 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast()),
                            );

                            let u3 = vld1q_f64(data.get_unchecked(j + 1..).as_ptr().cast());
                            let u4 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + third + 1..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * (j + 1)..).as_ptr().cast()),
                            );
                            let u5 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 1 + 2 * third..).as_ptr().cast()),
                                vld1q_f64(
                                    m_twiddles.get_unchecked(2 * (j + 1) + 1..).as_ptr().cast(),
                                ),
                            );

                            // Radix-3 butterfly
                            let xp0 = vaddq_f64(u1, u2);
                            let xn0 = vsubq_f64(u1, u2);
                            let sum0 = vaddq_f64(u0, xp0);

                            let xp1 = vaddq_f64(u4, u5);
                            let xn1 = vsubq_f64(u4, u5);
                            let sum1 = vaddq_f64(u3, xp1);

                            let w_01 = vfmaq_f64(u0, twiddle_re, xp0);
                            let vw_02 = vmulq_f64(twiddle_w_2, vextq_f64::<1>(xn0, xn0));

                            let w_02 = vfmaq_f64(u3, twiddle_re, xp1);
                            let vw_03 = vmulq_f64(twiddle_w_2, vextq_f64::<1>(xn1, xn1));

                            let vy0 = sum0;
                            let vy1 = vaddq_f64(w_01, vw_02);
                            let vy2 = vsubq_f64(w_01, vw_02);

                            let vy3 = sum1;
                            let vy4 = vaddq_f64(w_02, vw_03);
                            let vy5 = vsubq_f64(w_02, vw_03);

                            vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            vst1q_f64(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );
                            vst1q_f64(data.get_unchecked_mut(j + 1..).as_mut_ptr().cast(), vy3);
                            vst1q_f64(
                                data.get_unchecked_mut(j + 1 + third..).as_mut_ptr().cast(),
                                vy4,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 1 + 2 * third..)
                                    .as_mut_ptr()
                                    .cast(),
                                vy5,
                            );
                            j += 2;
                        }

                        for j in j..third {
                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + third..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                            );
                            let u2 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast()),
                            );

                            // Radix-3 butterfly
                            let xp = vaddq_f64(u1, u2);
                            let xn = vsubq_f64(u1, u2);
                            let sum = vaddq_f64(u0, xp);

                            let w_1 = vfmaq_f64(u0, twiddle_re, xp);
                            let vw_2 = vmulq_f64(twiddle_w_2, vextq_f64::<1>(xn, xn));

                            let vy0 = sum;
                            let vy1 = vaddq_f64(w_1, vw_2);
                            let vy2 = vsubq_f64(w_1, vw_2);

                            vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            vst1q_f64(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[third * 2..];
                    len *= 3;
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for NeonFcmaRadix3<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}

impl NeonFcmaRadix3<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let twiddle_re = vdupq_n_f32(self.twiddle_re);
        let twiddle_w_2 = unsafe { vld1q_f32(self.twiddle_im.as_ptr().cast()) };

        for chunk in in_place.chunks_mut(self.execution_length) {
            // Trit-reversal permutation
            permute_inplace(chunk, &self.permutations);

            let mut len = 3;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len <= self.execution_length {
                    let third = len / 3;
                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 4 < third {
                            let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(2 * (j + 1)..).as_ptr().cast());

                            let tw2 =
                                vld1q_f32(m_twiddles.get_unchecked(2 * (j + 2)..).as_ptr().cast());
                            let tw3 =
                                vld1q_f32(m_twiddles.get_unchecked(2 * (j + 3)..).as_ptr().cast());

                            let u1 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + third..).as_ptr().cast()),
                                vcombine_f32(vget_low_f32(tw0), vget_low_f32(tw1)),
                            );

                            let u2 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                vcombine_f32(vget_high_f32(tw0), vget_high_f32(tw1)),
                            );

                            let u3 = vld1q_f32(data.get_unchecked(j + 2..).as_ptr().cast());

                            let u4 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + third + 2..).as_ptr().cast()),
                                vcombine_f32(vget_low_f32(tw2), vget_low_f32(tw3)),
                            );
                            let u5 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * third + 2..).as_ptr().cast()),
                                vcombine_f32(vget_high_f32(tw2), vget_high_f32(tw3)),
                            );

                            // Radix-3 butterfly
                            let xp0 = vaddq_f32(u1, u2);
                            let xn0 = vsubq_f32(u1, u2);
                            let sum0 = vaddq_f32(u0, xp0);

                            let xp1 = vaddq_f32(u4, u5);
                            let xn1 = vsubq_f32(u4, u5);
                            let sum1 = vaddq_f32(u3, xp1);

                            let w_01 = vfmaq_f32(u0, twiddle_re, xp0);
                            let vw_02 = vmulq_f32(twiddle_w_2, vrev64q_f32(xn0));

                            let w_02 = vfmaq_f32(u3, twiddle_re, xp1);
                            let vw_03 = vmulq_f32(twiddle_w_2, vrev64q_f32(xn1));

                            let vy0 = sum0;
                            let vy1 = vaddq_f32(w_01, vw_02);
                            let vy2 = vsubq_f32(w_01, vw_02);

                            let vy3 = sum1;
                            let vy4 = vaddq_f32(w_02, vw_03);
                            let vy5 = vsubq_f32(w_02, vw_03);

                            vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            vst1q_f32(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );

                            vst1q_f32(data.get_unchecked_mut(j + 2..).as_mut_ptr().cast(), vy3);
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 + third..).as_mut_ptr().cast(),
                                vy4,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 + 2 * third..)
                                    .as_mut_ptr()
                                    .cast(),
                                vy5,
                            );

                            j += 4;
                        }

                        while j + 2 < third {
                            let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(2 * (j + 1)..).as_ptr().cast());

                            let u1 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + third..).as_ptr().cast()),
                                vcombine_f32(vget_low_f32(tw0), vget_low_f32(tw1)),
                            );
                            let u2 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                vcombine_f32(vget_high_f32(tw0), vget_high_f32(tw1)),
                            );

                            // Radix-3 butterfly
                            let xp = vaddq_f32(u1, u2);
                            let xn = vsubq_f32(u1, u2);
                            let sum = vaddq_f32(u0, xp);

                            let w_1 = vfmaq_f32(u0, twiddle_re, xp);
                            let vw_2 = vmulq_f32(twiddle_w_2, vrev64q_f32(xn));

                            let vy0 = sum;
                            let vy1 = vaddq_f32(w_1, vw_2);
                            let vy2 = vsubq_f32(w_1, vw_2);

                            vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            vst1q_f32(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );

                            j += 2;
                        }

                        for j in j..third {
                            let u0 = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw = vld1q_f32(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());

                            let u1 = fcmah_complex_f32(
                                vld1_f32(data.get_unchecked(j + third..).as_ptr().cast()),
                                vget_low_f32(tw),
                            );
                            let u2 = fcmah_complex_f32(
                                vld1_f32(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                vget_high_f32(tw),
                            );

                            // Radix-3 butterfly
                            let xp = vadd_f32(u1, u2);
                            let xn = vsub_f32(u1, u2);
                            let sum = vadd_f32(u0, xp);

                            let w_1 = vfma_f32(u0, vget_low_f32(twiddle_re), xp);
                            let vw_2 = vmul_f32(vget_low_f32(twiddle_w_2), vext_f32::<1>(xn, xn));

                            let vy0 = sum;
                            let vy1 = vadd_f32(w_1, vw_2);
                            let vy2 = vsub_f32(w_1, vw_2);

                            vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            vst1_f32(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                            vst1_f32(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[third * 2..];
                    len *= 3;
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaRadix3<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_neon_fcma_radix3() {
        for i in 1..9 {
            let size = 3usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaRadix3::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix3::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input
                .iter()
                .map(|&x| Complex::new(x.re as f64, x.im as f64) * (1.0f64 / input.len() as f64))
                .map(|x| Complex::new(x.re as f32, x.im as f32))
                .collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_neon_fcma_radix3_f64() {
        for i in 1..9 {
            let size = 3usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaRadix3::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix3::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input
                .iter()
                .map(|&x| x * (1.0f64 / input.len() as f64))
                .collect();

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
