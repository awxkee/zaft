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
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::neon::util::{create_neon_twiddles, vfcmulq_f32, vfcmulq_f64};
use crate::radix3::Radix3Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{bitreversed_transpose, compute_logarithm, compute_twiddle, is_power_of_three};
use crate::{FftDirection, FftExecutor, Zaft, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::aarch64::*;
use std::fmt::Display;

pub(crate) struct NeonRadix3<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle_re: T,
    twiddle_im: [T; 4],
    direction: FftDirection,
    base_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    base_len: usize,
}

impl<
    T: Default
        + Clone
        + Radix3Twiddles
        + 'static
        + Copy
        + FftTrigonometry
        + Float
        + Send
        + Sync
        + AlgorithmFactory<T>
        + MulAdd<T, Output = T>
        + SpectrumOpsFactory<T>
        + Display
        + TransposeFactory<T>,
> NeonRadix3<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonRadix3<T>, ZaftError> {
        assert!(
            is_power_of_three(size as u64),
            "Input length must be power of 3"
        );

        let exponent = compute_logarithm::<3>(size).unwrap_or_else(|| {
            panic!("Neon Fcma Radix3 length must be power of 3, but got {size}",)
        });

        let base_fft = match exponent {
            0 => Zaft::strategy(1, fft_direction)?,
            1 => Zaft::strategy(3, fft_direction)?,
            2 => Zaft::strategy(9, fft_direction)?,
            _ => Zaft::strategy(27, fft_direction)?,
        };

        let twiddles = create_neon_twiddles::<T, 3>(base_fft.length(), size, fft_direction)?;

        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);

        Ok(NeonRadix3 {
            execution_length: size,
            twiddles,
            twiddle_re: twiddle.re,
            twiddle_im: [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im],
            direction: fft_direction,
            base_len: base_fft.length(),
            base_fft,
        })
    }
}

impl FftExecutor<f64> for NeonRadix3<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let twiddle_re = vdupq_n_f64(self.twiddle_re);
            let twiddle_w_2 = vld1q_f64(self.twiddle_im.as_ptr().cast());

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f64>, 3>(self.base_len, chunk, &mut scratch);

                self.base_fft.execute(&mut scratch)?;

                let mut len = self.base_len;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 3;
                    let third = len / 3;

                    for data in scratch.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < third {
                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + third..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                            );
                            let u2 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast()),
                            );

                            let u3 = vld1q_f64(data.get_unchecked(j + 1..).as_ptr().cast());
                            let u4 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + third + 1..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * (j + 1)..).as_ptr().cast()),
                            );
                            let u5 = vfcmulq_f64(
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
                            let w_02 = vfmaq_f64(u3, twiddle_re, xp1);

                            let xn0_rot = vextq_f64::<1>(xn0, xn0);
                            let xn1_rot = vextq_f64::<1>(xn1, xn1);

                            let vy0 = sum0;
                            let vy1 = vfmaq_f64(w_01, twiddle_w_2, xn0_rot);
                            let vy2 = vfmsq_f64(w_01, twiddle_w_2, xn0_rot);

                            let vy3 = sum1;
                            let vy4 = vfmaq_f64(w_02, twiddle_w_2, xn1_rot);
                            let vy5 = vfmsq_f64(w_02, twiddle_w_2, xn1_rot);

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
                            let u1 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + third..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                            );
                            let u2 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast()),
                            );

                            // Radix-3 butterfly
                            let xp = vaddq_f64(u1, u2);
                            let xn = vsubq_f64(u1, u2);
                            let sum = vaddq_f64(u0, xp);

                            let w_1 = vfmaq_f64(u0, twiddle_re, xp);

                            let xn_rot = vextq_f64::<1>(xn, xn);

                            let vy0 = sum;
                            let vy1 = vfmaq_f64(w_1, twiddle_w_2, xn_rot);
                            let vy2 = vfmsq_f64(w_1, twiddle_w_2, xn_rot);

                            vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            vst1q_f64(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 2..];
                }
                chunk.copy_from_slice(&scratch);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}

impl FftExecutor<f32> for NeonRadix3<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let twiddle_re = vdupq_n_f32(self.twiddle_re);
            let twiddle_w_2 = vld1q_f32(self.twiddle_im.as_ptr().cast());

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f32>, 3>(self.base_len, chunk, &mut scratch);

                self.base_fft.execute(&mut scratch)?;

                let mut len = self.base_len;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 3;
                    let third = len / 3;

                    for data in scratch.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 4 < third {
                            let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(2 * j + 2..).as_ptr().cast());

                            let tw2 =
                                vld1q_f32(m_twiddles.get_unchecked(2 * j + 4..).as_ptr().cast());
                            let tw3 =
                                vld1q_f32(m_twiddles.get_unchecked(2 * j + 6..).as_ptr().cast());

                            let u1 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + third..).as_ptr().cast()),
                                tw0,
                            );

                            let u2 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                tw1,
                            );

                            let u3 = vld1q_f32(data.get_unchecked(j + 2..).as_ptr().cast());

                            let u4 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + third + 2..).as_ptr().cast()),
                                tw2,
                            );
                            let u5 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * third + 2..).as_ptr().cast()),
                                tw3,
                            );

                            // Radix-3 butterfly
                            let xp0 = vaddq_f32(u1, u2);
                            let xn0 = vsubq_f32(u1, u2);
                            let sum0 = vaddq_f32(u0, xp0);

                            let xp1 = vaddq_f32(u4, u5);
                            let xn1 = vsubq_f32(u4, u5);
                            let sum1 = vaddq_f32(u3, xp1);

                            let xn0_rot = vrev64q_f32(xn0);
                            let xn1_rot = vrev64q_f32(xn1);

                            let w_01 = vfmaq_f32(u0, twiddle_re, xp0);
                            let w_02 = vfmaq_f32(u3, twiddle_re, xp1);

                            let vy0 = sum0;
                            let vy1 = vfmaq_f32(w_01, twiddle_w_2, xn0_rot);
                            let vy2 = vfmsq_f32(w_01, twiddle_w_2, xn0_rot);

                            let vy3 = sum1;
                            let vy4 = vfmaq_f32(w_02, twiddle_w_2, xn1_rot);
                            let vy5 = vfmsq_f32(w_02, twiddle_w_2, xn1_rot);

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
                                vld1q_f32(m_twiddles.get_unchecked(2 * j + 2..).as_ptr().cast());

                            let u1 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + third..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                tw1,
                            );

                            // Radix-3 butterfly
                            let xp = vaddq_f32(u1, u2);
                            let xn = vsubq_f32(u1, u2);
                            let sum = vaddq_f32(u0, xp);

                            let w_1 = vfmaq_f32(u0, twiddle_re, xp);
                            let xn_rot = vrev64q_f32(xn);

                            let vy0 = sum;
                            let vy1 = vfmaq_f32(w_1, twiddle_w_2, xn_rot);
                            let vy2 = vfmsq_f32(w_1, twiddle_w_2, xn_rot);

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

                            let u1u2 = vfcmulq_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + third..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                ),
                                tw,
                            );

                            let u1 = vget_low_f32(u1u2);
                            let u2 = vget_high_f32(u1u2);

                            // Radix-3 butterfly
                            let xp = vadd_f32(u1, u2);
                            let xn = vsub_f32(u1, u2);
                            let sum = vadd_f32(u0, xp);

                            let w_1 = vfma_f32(u0, vget_low_f32(twiddle_re), xp);

                            let xn_rot = vext_f32::<1>(xn, xn);

                            let vy0 = sum;
                            let vy1 = vfma_f32(w_1, vget_low_f32(twiddle_w_2), xn_rot);
                            let vy2 = vfms_f32(w_1, vget_low_f32(twiddle_w_2), xn_rot);

                            vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            vst1_f32(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                            vst1_f32(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 2..];
                }
                chunk.copy_from_slice(&scratch);
            }
        }
        Ok(())
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
    fn test_neon_radix3() {
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
            let radix_forward = NeonRadix3::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonRadix3::new(size, FftDirection::Inverse).unwrap();
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
    fn test_neon_radix3_f64() {
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
            let radix_forward = NeonRadix3::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonRadix3::new(size, FftDirection::Inverse).unwrap();
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
