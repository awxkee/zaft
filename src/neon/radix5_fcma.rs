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
use crate::neon::radix5::neon_bitreversed_transpose_f32_radix5;
use crate::neon::util::{create_neon_twiddles, vfcmulq_fcma_f32, vfcmulq_fcma_f64};
use crate::radix5::Radix5Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{bitreversed_transpose, compute_logarithm, compute_twiddle, is_power_of_five};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::aarch64::*;
use std::fmt::Display;

pub(crate) struct NeonFcmaRadix5<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    direction: FftDirection,
    butterfly: Box<dyn CompositeFftExecutor<T> + Send + Sync>,
    butterfly_length: usize,
}

impl<
    T: Default
        + Clone
        + Radix5Twiddles
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
> NeonFcmaRadix5<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonFcmaRadix5<T>, ZaftError> {
        assert!(
            is_power_of_five(size as u64),
            "Input length must be a power of 5"
        );

        let log5 = compute_logarithm::<5>(size).unwrap();
        let butterfly = match log5 {
            0 => T::butterfly1(fft_direction)?,
            1 => T::butterfly5(fft_direction)?,
            _ => T::butterfly25(fft_direction)?,
        };

        let butterfly_length = butterfly.length();

        let twiddles = create_neon_twiddles::<T, 5>(butterfly_length, size, fft_direction)?;

        Ok(NeonFcmaRadix5 {
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 5, fft_direction),
            twiddle2: compute_twiddle(2, 5, fft_direction),
            direction: fft_direction,
            butterfly,
            butterfly_length,
        })
    }
}

impl NeonFcmaRadix5<f64> {
    #[target_feature(enable = "fcma")]
    fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let tw1_re = vdupq_n_f64(self.twiddle1.re);
            let tw1_im = vdupq_n_f64(self.twiddle1.im);
            let tw2_re = vdupq_n_f64(self.twiddle2.re);
            let tw2_im = vdupq_n_f64(self.twiddle2.im);

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f64>, 5>(
                    self.butterfly_length,
                    chunk,
                    &mut scratch,
                );

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.butterfly_length;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 5;
                    let fifth = len / 5;

                    for data in chunk.chunks_exact_mut(len) {
                        for j in 0..fifth {
                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + fifth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(4 * j..).as_ptr().cast()),
                            );
                            let u2 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * fifth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(4 * j + 1..).as_ptr().cast()),
                            );
                            let u3 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 3 * fifth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast()),
                            );
                            let u4 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 4 * fifth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(4 * j + 3..).as_ptr().cast()),
                            );

                            // Radix-5 butterfly

                            let x14p = vaddq_f64(u1, u4);
                            let x14n = vsubq_f64(u1, u4);
                            let x23p = vaddq_f64(u2, u3);
                            let x23n = vsubq_f64(u2, u3);
                            let y0 = vaddq_f64(vaddq_f64(u0, x14p), x23p);

                            let temp_b1_1 = vmulq_f64(tw1_im, x14n);
                            let temp_b2_1 = vmulq_f64(tw2_im, x14n);

                            let temp_a1 = vfmaq_f64(vfmaq_f64(u0, tw1_re, x14p), tw2_re, x23p);
                            let temp_a2 = vfmaq_f64(vfmaq_f64(u0, tw2_re, x14p), tw1_re, x23p);

                            let temp_b1 = vfmaq_f64(temp_b1_1, tw2_im, x23n);
                            let temp_b2 = vfmsq_f64(temp_b2_1, tw1_im, x23n);

                            let y1 = vcaddq_rot90_f64(temp_a1, temp_b1);
                            let y2 = vcaddq_rot90_f64(temp_a2, temp_b2);
                            let y3 = vcaddq_rot270_f64(temp_a2, temp_b2);
                            let y4 = vcaddq_rot270_f64(temp_a1, temp_b1);

                            vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            vst1q_f64(data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(), y1);
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                                y2,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                                y3,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                                y4,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 4..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for NeonFcmaRadix5<f64> {
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

impl NeonFcmaRadix5<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let tw1_re = vdupq_n_f32(self.twiddle1.re);
            let tw1_im = vdupq_n_f32(self.twiddle1.im);
            let tw2_re = vdupq_n_f32(self.twiddle2.re);
            let tw2_im = vdupq_n_f32(self.twiddle2.im);

            let tw1_tw2_im = vcombine_f32(vget_low_f32(tw1_im), vget_low_f32(tw2_im));
            let tw2_ntw1_im = vcombine_f32(vget_low_f32(tw2_im), vdup_n_f32(-self.twiddle1.im));
            let a1_a2 = vcombine_f32(vget_low_f32(tw1_re), vget_low_f32(tw2_re));
            let a1_a2_2 = vcombine_f32(vget_low_f32(tw2_re), vget_low_f32(tw1_re));

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                neon_bitreversed_transpose_f32_radix5(self.butterfly_length, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.butterfly_length;
                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 5;
                    let fifth = len / 5;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < fifth {
                            let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast());
                            let tw2 =
                                vld1q_f32(m_twiddles.get_unchecked(4 * j + 4..).as_ptr().cast());
                            let tw3 =
                                vld1q_f32(m_twiddles.get_unchecked(4 * j + 6..).as_ptr().cast());

                            let u1 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + fifth..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * fifth..).as_ptr().cast()),
                                tw1,
                            );
                            let u3 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 3 * fifth..).as_ptr().cast()),
                                tw2,
                            );
                            let u4 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 4 * fifth..).as_ptr().cast()),
                                tw3,
                            );

                            // Radix-5 butterfly

                            let x14p = vaddq_f32(u1, u4);
                            let x14n = vsubq_f32(u1, u4);
                            let x23p = vaddq_f32(u2, u3);
                            let x23n = vsubq_f32(u2, u3);
                            let y0 = vaddq_f32(vaddq_f32(u0, x14p), x23p);

                            let temp_b1_1 = vmulq_f32(tw1_im, x14n);
                            let temp_b2_1 = vmulq_f32(tw2_im, x14n);

                            let temp_a1 = vfmaq_f32(vfmaq_f32(u0, tw1_re, x14p), tw2_re, x23p);
                            let temp_a2 = vfmaq_f32(vfmaq_f32(u0, tw2_re, x14p), tw1_re, x23p);

                            let temp_b1 = vfmaq_f32(temp_b1_1, tw2_im, x23n);
                            let temp_b2 = vfmsq_f32(temp_b2_1, tw1_im, x23n);

                            let y1 = vcaddq_rot90_f32(temp_a1, temp_b1);
                            let y2 = vcaddq_rot90_f32(temp_a2, temp_b2);
                            let y3 = vcaddq_rot270_f32(temp_a2, temp_b2);
                            let y4 = vcaddq_rot270_f32(temp_a1, temp_b1);

                            vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            vst1q_f32(data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(), y1);
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                                y2,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                                y3,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                                y4,
                            );

                            j += 2;
                        }

                        for j in j..fifth {
                            let u0 = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast());

                            let u1u2 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + fifth..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 2 * fifth..).as_ptr().cast()),
                                ),
                                tw0,
                            );
                            let u3u4 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + 3 * fifth..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 4 * fifth..).as_ptr().cast()),
                                ),
                                tw1,
                            );

                            // Radix-5 butterfly

                            let u1 = vget_low_f32(u1u2);
                            let u2 = vget_high_f32(u1u2);
                            let u3 = vget_low_f32(u3u4);
                            let u4 = vget_high_f32(u3u4);

                            let x14px23p = vaddq_f32(vcombine_f32(u1, u2), vcombine_f32(u4, u3));
                            let x14nx23n = vsubq_f32(vcombine_f32(u1, u2), vcombine_f32(u4, u3));
                            let y0 = vadd_f32(
                                vadd_f32(u0, vget_low_f32(x14px23p)),
                                vget_high_f32(x14px23p),
                            );

                            let temp_b1_1_b2_1 = vmulq_f32(
                                tw1_tw2_im,
                                vcombine_f32(vget_low_f32(x14nx23n), vget_low_f32(x14nx23n)),
                            );

                            let wx23p =
                                vcombine_f32(vget_high_f32(x14px23p), vget_high_f32(x14px23p));
                            let wx23n =
                                vcombine_f32(vget_high_f32(x14nx23n), vget_high_f32(x14nx23n));

                            let temp_a1_a2 = vfmaq_f32(
                                vfmaq_f32(
                                    vcombine_f32(u0, u0),
                                    a1_a2,
                                    vcombine_f32(vget_low_f32(x14px23p), vget_low_f32(x14px23p)),
                                ),
                                a1_a2_2,
                                wx23p,
                            );

                            let temp_b1_b2 = vfmaq_f32(temp_b1_1_b2_1, tw2_ntw1_im, wx23n);

                            let y1y2 = vcaddq_rot90_f32(temp_a1_a2, temp_b1_b2);
                            let y4y3 = vcaddq_rot270_f32(temp_a1_a2, temp_b1_b2);

                            vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            vst1_f32(
                                data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(),
                                vget_low_f32(y1y2),
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                                vget_high_f32(y1y2),
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                                vget_high_f32(y4y3),
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                                vget_low_f32(y4y3),
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 4..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaRadix5<f32> {
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
    use crate::neon::test_fcma_radix;

    test_fcma_radix!(test_neon_radix5, f32, NeonFcmaRadix5, 5, 5, 1e-3);
    test_fcma_radix!(test_neon_radix5_f64, f64, NeonFcmaRadix5, 5, 5, 1e-8);
}
