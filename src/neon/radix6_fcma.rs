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
use crate::neon::butterflies::NeonButterfly;
use crate::neon::radix6::neon_bitreversed_transpose_f32_radix6;
use crate::neon::util::{
    create_neon_twiddles, vfcmul_fcma_f32, vfcmulq_fcma_f32, vfcmulq_fcma_f64,
};
use crate::radix6::Radix6Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{bitreversed_transpose, compute_logarithm, compute_twiddle, is_power_of_six};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::aarch64::*;
use std::fmt::Display;
use std::sync::Arc;

pub(crate) struct NeonFcmaRadix6<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle_re: T,
    twiddle_im: [T; 4],
    direction: FftDirection,
    butterfly: Arc<dyn CompositeFftExecutor<T> + Send + Sync>,
    butterfly_len: usize,
}

impl<
    T: Default
        + Clone
        + Radix6Twiddles
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
> NeonFcmaRadix6<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonFcmaRadix6<T>, ZaftError> {
        assert!(
            is_power_of_six(size as u64),
            "Input length must be a power of 6"
        );

        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);

        let exponent = compute_logarithm::<6>(size).unwrap_or_else(|| {
            panic!("Neon Fcma Radix6 length must be power of 6, but got {size}",)
        });

        let butterfly = match exponent {
            0 => T::butterfly1(fft_direction)?,
            1 => T::butterfly6(fft_direction)?,
            _ => T::butterfly36(fft_direction).map_or_else(|| T::butterfly6(fft_direction), Ok)?,
        };

        let butterfly_len = butterfly.length();

        let twiddles = create_neon_twiddles::<T, 6>(butterfly_len, size, fft_direction)?;

        Ok(NeonFcmaRadix6 {
            execution_length: size,
            twiddles,
            twiddle_re: twiddle.re,
            twiddle_im: [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im],
            direction: fft_direction,
            butterfly,
            butterfly_len,
        })
    }
}

impl NeonFcmaRadix6<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.execution_length) {
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
                bitreversed_transpose::<Complex<f64>, 6>(self.butterfly_len, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.butterfly_len;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 6;
                    let sixth = len / 6;

                    for data in chunk.chunks_exact_mut(len) {
                        for j in 0..sixth {
                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + sixth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(5 * j..).as_ptr().cast()),
                            );
                            let u2 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * sixth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(5 * j + 1..).as_ptr().cast()),
                            );
                            let u3 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 3 * sixth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(5 * j + 2..).as_ptr().cast()),
                            );
                            let u4 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 4 * sixth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(5 * j + 3..).as_ptr().cast()),
                            );
                            let u5 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 5 * sixth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(5 * j + 4..).as_ptr().cast()),
                            );

                            let (t0, t2, t4) = NeonButterfly::butterfly3_f64_fcma(
                                u0,
                                u2,
                                u4,
                                twiddle_re,
                                twiddle_w_2,
                            );
                            let (t1, t3, t5) = NeonButterfly::butterfly3_f64_fcma(
                                u3,
                                u5,
                                u1,
                                twiddle_re,
                                twiddle_w_2,
                            );
                            let (y0, y3) = NeonButterfly::butterfly2_f64(t0, t1);
                            let (y4, y1) = NeonButterfly::butterfly2_f64(t2, t3);
                            let (y2, y5) = NeonButterfly::butterfly2_f64(t4, t5);

                            // Store results
                            vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            vst1q_f64(data.get_unchecked_mut(j + sixth..).as_mut_ptr().cast(), y1);
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * sixth..).as_mut_ptr().cast(),
                                y2,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 3 * sixth..).as_mut_ptr().cast(),
                                y3,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 4 * sixth..).as_mut_ptr().cast(),
                                y4,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 5 * sixth..).as_mut_ptr().cast(),
                                y5,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 5..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for NeonFcmaRadix6<f64> {
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

impl NeonFcmaRadix6<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.execution_length) {
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
                neon_bitreversed_transpose_f32_radix6(self.butterfly_len, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.butterfly_len;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 6;
                    let sixth = len / 6;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < sixth {
                            let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(5 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(5 * j + 2..).as_ptr().cast());
                            let tw2 =
                                vld1q_f32(m_twiddles.get_unchecked(5 * j + 4..).as_ptr().cast());
                            let tw3 =
                                vld1q_f32(m_twiddles.get_unchecked(5 * j + 6..).as_ptr().cast());
                            let tw4 =
                                vld1q_f32(m_twiddles.get_unchecked(5 * j + 8..).as_ptr().cast());

                            let u1 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + sixth..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * sixth..).as_ptr().cast()),
                                tw1,
                            );
                            let u3 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 3 * sixth..).as_ptr().cast()),
                                tw2,
                            );
                            let u4 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 4 * sixth..).as_ptr().cast()),
                                tw3,
                            );
                            let u5 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 5 * sixth..).as_ptr().cast()),
                                tw4,
                            );

                            let (t0, t2, t4) = NeonButterfly::butterfly3_f32_fcma(
                                u0,
                                u2,
                                u4,
                                twiddle_re,
                                twiddle_w_2,
                            );
                            let (t1, t3, t5) = NeonButterfly::butterfly3_f32_fcma(
                                u3,
                                u5,
                                u1,
                                twiddle_re,
                                twiddle_w_2,
                            );
                            let (y0, y3) = NeonButterfly::butterfly2_f32(t0, t1);
                            let (y4, y1) = NeonButterfly::butterfly2_f32(t2, t3);
                            let (y2, y5) = NeonButterfly::butterfly2_f32(t4, t5);

                            // Store results
                            vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            vst1q_f32(data.get_unchecked_mut(j + sixth..).as_mut_ptr().cast(), y1);
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * sixth..).as_mut_ptr().cast(),
                                y2,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 3 * sixth..).as_mut_ptr().cast(),
                                y3,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 4 * sixth..).as_mut_ptr().cast(),
                                y4,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 5 * sixth..).as_mut_ptr().cast(),
                                y5,
                            );

                            j += 2;
                        }

                        for j in j..sixth {
                            let u0 = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(5 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(5 * j + 2..).as_ptr().cast());

                            let u1u2 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + sixth..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 2 * sixth..).as_ptr().cast()),
                                ),
                                tw0,
                            );
                            let u3u4 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + 3 * sixth..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 4 * sixth..).as_ptr().cast()),
                                ),
                                tw1,
                            );
                            let u5 = vfcmul_fcma_f32(
                                vld1_f32(data.get_unchecked(j + 5 * sixth..).as_ptr().cast()),
                                vld1_f32(m_twiddles.get_unchecked(5 * j + 4..).as_ptr().cast()),
                            );

                            let u1 = vget_low_f32(u1u2);
                            let u2 = vget_high_f32(u1u2);
                            let u3 = vget_low_f32(u3u4);
                            let u4 = vget_high_f32(u3u4);

                            let (t0, t2, t4) = NeonButterfly::butterfly3h_f32_fcma(
                                u0,
                                u2,
                                u4,
                                vget_low_f32(twiddle_re),
                                vget_low_f32(twiddle_w_2),
                            );
                            let (t1, t3, t5) = NeonButterfly::butterfly3h_f32_fcma(
                                u3,
                                u5,
                                u1,
                                vget_low_f32(twiddle_re),
                                vget_low_f32(twiddle_w_2),
                            );
                            let (y0, y3) = NeonButterfly::butterfly2h_f32(t0, t1);
                            let (y4, y1) = NeonButterfly::butterfly2h_f32(t2, t3);
                            let (y2, y5) = NeonButterfly::butterfly2h_f32(t4, t5);

                            // Store results
                            vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            vst1_f32(data.get_unchecked_mut(j + sixth..).as_mut_ptr().cast(), y1);
                            vst1_f32(
                                data.get_unchecked_mut(j + 2 * sixth..).as_mut_ptr().cast(),
                                y2,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 3 * sixth..).as_mut_ptr().cast(),
                                y3,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 4 * sixth..).as_mut_ptr().cast(),
                                y4,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 5 * sixth..).as_mut_ptr().cast(),
                                y5,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 5..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaRadix6<f32> {
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

    test_fcma_radix!(test_neon_radix6, f32, NeonFcmaRadix6, 4, 6, 1e-2);
    test_fcma_radix!(test_neon_radix6_f64, f64, NeonFcmaRadix6, 4, 6, 1e-8);
}
