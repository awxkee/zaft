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
use crate::neon::util::{create_neon_twiddles, vfcmulq_fcma_f32, vfcmulq_fcma_f64};
use crate::radix7::Radix7Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{bitreversed_transpose, compute_twiddle, is_power_of_seven};
use crate::{FftDirection, FftExecutor, Zaft, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::aarch64::*;
use std::fmt::Display;

pub(crate) struct NeonFcmaRadix7<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    direction: FftDirection,
    butterfly: Box<dyn FftExecutor<T> + Send + Sync>,
}

impl<
    T: Default
        + Clone
        + Radix7Twiddles
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
> NeonFcmaRadix7<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonFcmaRadix7<T>, ZaftError> {
        assert!(
            is_power_of_seven(size as u64),
            "Input length must be a power of 7"
        );

        let twiddles = create_neon_twiddles::<T, 7>(7, size, fft_direction)?;

        Ok(NeonFcmaRadix7 {
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 7, fft_direction),
            twiddle2: compute_twiddle(2, 7, fft_direction),
            twiddle3: compute_twiddle(3, 7, fft_direction),
            direction: fft_direction,
            butterfly: Zaft::strategy(7, fft_direction)?,
        })
    }
}

impl FftExecutor<f64> for NeonFcmaRadix7<f64> {
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

impl NeonFcmaRadix7<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }
        unsafe {
            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f64>, 7>(7, chunk, &mut scratch);

                self.butterfly.execute(&mut scratch)?;

                let mut len = 7;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 7;
                    let seventh = len / 7;

                    for data in scratch.chunks_exact_mut(len) {
                        for j in 0..seventh {
                            let tw0 = vld1q_f64(m_twiddles.get_unchecked(6 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f64(m_twiddles.get_unchecked(6 * j + 1..).as_ptr().cast());
                            let tw2 =
                                vld1q_f64(m_twiddles.get_unchecked(6 * j + 2..).as_ptr().cast());
                            let tw3 =
                                vld1q_f64(m_twiddles.get_unchecked(6 * j + 3..).as_ptr().cast());
                            let tw4 =
                                vld1q_f64(m_twiddles.get_unchecked(6 * j + 4..).as_ptr().cast());
                            let tw5 =
                                vld1q_f64(m_twiddles.get_unchecked(6 * j + 5..).as_ptr().cast());

                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + seventh..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * seventh..).as_ptr().cast()),
                                tw1,
                            );
                            let u3 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 3 * seventh..).as_ptr().cast()),
                                tw2,
                            );
                            let u4 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 4 * seventh..).as_ptr().cast()),
                                tw3,
                            );
                            let u5 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 5 * seventh..).as_ptr().cast()),
                                tw4,
                            );
                            let u6 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 6 * seventh..).as_ptr().cast()),
                                tw5,
                            );

                            // Radix-7 butterfly

                            let (x1p6, x1m6) = NeonButterfly::butterfly2_f64(u1, u6);
                            let x1m6 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m6);
                            let y00 = vaddq_f64(u0, x1p6);
                            let (x2p5, x2m5) = NeonButterfly::butterfly2_f64(u2, u5);
                            let x2m5 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m5);
                            let y00 = vaddq_f64(y00, x2p5);
                            let (x3p4, x3m4) = NeonButterfly::butterfly2_f64(u3, u4);
                            let x3m4 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m4);
                            let y00 = vaddq_f64(y00, x3p4);

                            let m0106a = vfmaq_n_f64(u0, x1p6, self.twiddle1.re);
                            let m0106a = vfmaq_n_f64(m0106a, x2p5, self.twiddle2.re);
                            let m0106a = vfmaq_n_f64(m0106a, x3p4, self.twiddle3.re);
                            let m0106b = vmulq_n_f64(x1m6, self.twiddle1.im);
                            let m0106b = vfmaq_n_f64(m0106b, x2m5, self.twiddle2.im);
                            let m0106b = vfmaq_n_f64(m0106b, x3m4, self.twiddle3.im);
                            let (y01, y06) = NeonButterfly::butterfly2_f64(m0106a, m0106b);

                            let m0205a = vfmaq_n_f64(u0, x1p6, self.twiddle2.re);
                            let m0205a = vfmaq_n_f64(m0205a, x2p5, self.twiddle3.re);
                            let m0205a = vfmaq_n_f64(m0205a, x3p4, self.twiddle1.re);
                            let m0205b = vmulq_n_f64(x1m6, self.twiddle2.im);
                            let m0205b = vfmsq_n_f64(m0205b, x2m5, self.twiddle3.im);
                            let m0205b = vfmsq_n_f64(m0205b, x3m4, self.twiddle1.im);
                            let (y02, y05) = NeonButterfly::butterfly2_f64(m0205a, m0205b);

                            let m0304a = vfmaq_n_f64(u0, x1p6, self.twiddle3.re);
                            let m0304a = vfmaq_n_f64(m0304a, x2p5, self.twiddle1.re);
                            let m0304a = vfmaq_n_f64(m0304a, x3p4, self.twiddle2.re);
                            let m0304b = vmulq_n_f64(x1m6, self.twiddle3.im);
                            let m0304b = vfmsq_n_f64(m0304b, x2m5, self.twiddle1.im);
                            let m0304b = vfmaq_n_f64(m0304b, x3m4, self.twiddle2.im);
                            let (y03, y04) = NeonButterfly::butterfly2_f64(m0304a, m0304b);

                            // Store results
                            vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            vst1q_f64(
                                data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                                y01,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 3 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 4 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 5 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 6 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 6..];
                }

                chunk.copy_from_slice(&scratch);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaRadix7<f32> {
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

impl NeonFcmaRadix7<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f32>, 7>(7, chunk, &mut scratch);

                self.butterfly.execute(&mut scratch)?;

                let mut len = 7;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 7;
                    let seventh = len / 7;

                    for data in scratch.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < seventh {
                            let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(6 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(6 * j + 2..).as_ptr().cast());
                            let tw2 =
                                vld1q_f32(m_twiddles.get_unchecked(6 * j + 4..).as_ptr().cast());
                            let tw3 =
                                vld1q_f32(m_twiddles.get_unchecked(6 * j + 6..).as_ptr().cast());
                            let tw4 =
                                vld1q_f32(m_twiddles.get_unchecked(6 * j + 8..).as_ptr().cast());
                            let tw5 =
                                vld1q_f32(m_twiddles.get_unchecked(6 * j + 10..).as_ptr().cast());

                            let u1 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + seventh..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * seventh..).as_ptr().cast()),
                                tw1,
                            );
                            let u3 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 3 * seventh..).as_ptr().cast()),
                                tw2,
                            );
                            let u4 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 4 * seventh..).as_ptr().cast()),
                                tw3,
                            );
                            let u5 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 5 * seventh..).as_ptr().cast()),
                                tw4,
                            );
                            let u6 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 6 * seventh..).as_ptr().cast()),
                                tw5,
                            );

                            // Radix-7 butterfly

                            let (x1p6, x1m6) = NeonButterfly::butterfly2_f32(u1, u6);
                            let x1m6 = vcaddq_rot90_f32(vdupq_n_f32(0.), x1m6);
                            let y00 = vaddq_f32(u0, x1p6);
                            let (x2p5, x2m5) = NeonButterfly::butterfly2_f32(u2, u5);
                            let x2m5 = vcaddq_rot90_f32(vdupq_n_f32(0.), x2m5);
                            let y00 = vaddq_f32(y00, x2p5);
                            let (x3p4, x3m4) = NeonButterfly::butterfly2_f32(u3, u4);
                            let x3m4 = vcaddq_rot90_f32(vdupq_n_f32(0.), x3m4);
                            let y00 = vaddq_f32(y00, x3p4);

                            let m0106a = vfmaq_n_f32(u0, x1p6, self.twiddle1.re);
                            let m0106a = vfmaq_n_f32(m0106a, x2p5, self.twiddle2.re);
                            let m0106a = vfmaq_n_f32(m0106a, x3p4, self.twiddle3.re);
                            let m0106b = vmulq_n_f32(x1m6, self.twiddle1.im);
                            let m0106b = vfmaq_n_f32(m0106b, x2m5, self.twiddle2.im);
                            let m0106b = vfmaq_n_f32(m0106b, x3m4, self.twiddle3.im);
                            let (y01, y06) = NeonButterfly::butterfly2_f32(m0106a, m0106b);

                            let m0205a = vfmaq_n_f32(u0, x1p6, self.twiddle2.re);
                            let m0205a = vfmaq_n_f32(m0205a, x2p5, self.twiddle3.re);
                            let m0205a = vfmaq_n_f32(m0205a, x3p4, self.twiddle1.re);
                            let m0205b = vmulq_n_f32(x1m6, self.twiddle2.im);
                            let m0205b = vfmsq_n_f32(m0205b, x2m5, self.twiddle3.im);
                            let m0205b = vfmsq_n_f32(m0205b, x3m4, self.twiddle1.im);
                            let (y02, y05) = NeonButterfly::butterfly2_f32(m0205a, m0205b);

                            let m0304a = vfmaq_n_f32(u0, x1p6, self.twiddle3.re);
                            let m0304a = vfmaq_n_f32(m0304a, x2p5, self.twiddle1.re);
                            let m0304a = vfmaq_n_f32(m0304a, x3p4, self.twiddle2.re);
                            let m0304b = vmulq_n_f32(x1m6, self.twiddle3.im);
                            let m0304b = vfmsq_n_f32(m0304b, x2m5, self.twiddle1.im);
                            let m0304b = vfmaq_n_f32(m0304b, x3m4, self.twiddle2.im);
                            let (y03, y04) = NeonButterfly::butterfly2_f32(m0304a, m0304b);

                            // Store results
                            vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            vst1q_f32(
                                data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                                y01,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 3 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 4 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 5 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 6 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );

                            j += 2;
                        }

                        for j in j..seventh {
                            let u0 = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                            let w0w1 = vld1q_f32(m_twiddles.get_unchecked(6 * j..).as_ptr().cast());
                            let w2w3 =
                                vld1q_f32(m_twiddles.get_unchecked(6 * j + 2..).as_ptr().cast());
                            let w4w5 =
                                vld1q_f32(m_twiddles.get_unchecked(6 * j + 4..).as_ptr().cast());

                            let u1u2 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + seventh..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 2 * seventh..).as_ptr().cast()),
                                ),
                                w0w1,
                            );
                            let u3u4 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + 3 * seventh..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 4 * seventh..).as_ptr().cast()),
                                ),
                                w2w3,
                            );
                            let u5u6 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + 5 * seventh..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 6 * seventh..).as_ptr().cast()),
                                ),
                                w4w5,
                            );

                            let u1 = vget_low_f32(u1u2);
                            let u2 = vget_high_f32(u1u2);
                            let u3 = vget_low_f32(u3u4);
                            let u4 = vget_high_f32(u3u4);
                            let u5 = vget_low_f32(u5u6);
                            let u6 = vget_high_f32(u5u6);

                            // Radix-7 butterfly

                            let (x1p6, x1m6) = NeonButterfly::butterfly2h_f32(u1, u6);
                            let x1m6 = vcadd_rot90_f32(vdup_n_f32(0.), x1m6);
                            let y00 = vadd_f32(u0, x1p6);
                            let (x2p5, x2m5) = NeonButterfly::butterfly2h_f32(u2, u5);
                            let x2m5 = vcadd_rot90_f32(vdup_n_f32(0.), x2m5);
                            let y00 = vadd_f32(y00, x2p5);
                            let (x3p4, x3m4) = NeonButterfly::butterfly2h_f32(u3, u4);
                            let x3m4 = vcadd_rot90_f32(vdup_n_f32(0.), x3m4);
                            let y00 = vadd_f32(y00, x3p4);

                            let m0106a = vfma_n_f32(u0, x1p6, self.twiddle1.re);
                            let m0106a = vfma_n_f32(m0106a, x2p5, self.twiddle2.re);
                            let m0106a = vfma_n_f32(m0106a, x3p4, self.twiddle3.re);
                            let m0106b = vmul_n_f32(x1m6, self.twiddle1.im);
                            let m0106b = vfma_n_f32(m0106b, x2m5, self.twiddle2.im);
                            let m0106b = vfma_n_f32(m0106b, x3m4, self.twiddle3.im);
                            let (y01, y06) = NeonButterfly::butterfly2h_f32(m0106a, m0106b);

                            let m0205a = vfma_n_f32(u0, x1p6, self.twiddle2.re);
                            let m0205a = vfma_n_f32(m0205a, x2p5, self.twiddle3.re);
                            let m0205a = vfma_n_f32(m0205a, x3p4, self.twiddle1.re);
                            let m0205b = vmul_n_f32(x1m6, self.twiddle2.im);
                            let m0205b = vfms_n_f32(m0205b, x2m5, self.twiddle3.im);
                            let m0205b = vfms_n_f32(m0205b, x3m4, self.twiddle1.im);
                            let (y02, y05) = NeonButterfly::butterfly2h_f32(m0205a, m0205b);

                            let m0304a = vfma_n_f32(u0, x1p6, self.twiddle3.re);
                            let m0304a = vfma_n_f32(m0304a, x2p5, self.twiddle1.re);
                            let m0304a = vfma_n_f32(m0304a, x3p4, self.twiddle2.re);
                            let m0304b = vmul_n_f32(x1m6, self.twiddle3.im);
                            let m0304b = vfms_n_f32(m0304b, x2m5, self.twiddle1.im);
                            let m0304b = vfma_n_f32(m0304b, x3m4, self.twiddle2.im);
                            let (y03, y04) = NeonButterfly::butterfly2h_f32(m0304a, m0304b);

                            // Store results
                            vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            vst1_f32(
                                data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                                y01,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 2 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 3 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 4 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 5 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 6 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 6..];
                }

                chunk.copy_from_slice(&scratch);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_neon_fcma_radix7() {
        for i in 1..7 {
            let size = 7usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaRadix7::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix7::new(size, FftDirection::Inverse).unwrap();
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
    fn test_neon_fcma_radix7_f64() {
        for i in 1..7 {
            let size = 7usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaRadix7::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix7::new(size, FftDirection::Inverse).unwrap();
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
