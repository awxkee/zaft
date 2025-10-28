/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use crate::radix13::Radix13Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{bitreversed_transpose, compute_twiddle, is_power_of_thirteen};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::aarch64::*;
use std::fmt::Display;

pub(crate) struct NeonFcmaRadix13<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    direction: FftDirection,
    butterfly: Box<dyn CompositeFftExecutor<T> + Send + Sync>,
}

impl<
    T: Default
        + Clone
        + Radix13Twiddles
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
> NeonFcmaRadix13<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonFcmaRadix13<T>, ZaftError> {
        assert!(
            is_power_of_thirteen(size as u64),
            "Input length must be a power of 13"
        );

        let twiddles = create_neon_twiddles::<T, 13>(13, size, fft_direction)?;

        Ok(NeonFcmaRadix13 {
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 13, fft_direction),
            twiddle2: compute_twiddle(2, 13, fft_direction),
            twiddle3: compute_twiddle(3, 13, fft_direction),
            twiddle4: compute_twiddle(4, 13, fft_direction),
            twiddle5: compute_twiddle(5, 13, fft_direction),
            twiddle6: compute_twiddle(6, 13, fft_direction),
            direction: fft_direction,
            butterfly: T::butterfly13(fft_direction)?,
        })
    }
}

impl FftExecutor<f64> for NeonFcmaRadix13<f64> {
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

impl NeonFcmaRadix13<f64> {
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
                bitreversed_transpose::<Complex<f64>, 13>(13, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = 13;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 13;
                    let thirteenth = len / 13;

                    for data in chunk.chunks_exact_mut(len) {
                        for j in 0..thirteenth {
                            let tw0 = vld1q_f64(m_twiddles.get_unchecked(12 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f64(m_twiddles.get_unchecked(12 * j + 1..).as_ptr().cast());
                            let tw2 =
                                vld1q_f64(m_twiddles.get_unchecked(12 * j + 2..).as_ptr().cast());
                            let tw3 =
                                vld1q_f64(m_twiddles.get_unchecked(12 * j + 3..).as_ptr().cast());
                            let tw4 =
                                vld1q_f64(m_twiddles.get_unchecked(12 * j + 4..).as_ptr().cast());
                            let tw5 =
                                vld1q_f64(m_twiddles.get_unchecked(12 * j + 5..).as_ptr().cast());
                            let tw6 =
                                vld1q_f64(m_twiddles.get_unchecked(12 * j + 6..).as_ptr().cast());
                            let tw7 =
                                vld1q_f64(m_twiddles.get_unchecked(12 * j + 7..).as_ptr().cast());
                            let tw8 =
                                vld1q_f64(m_twiddles.get_unchecked(12 * j + 8..).as_ptr().cast());
                            let tw9 =
                                vld1q_f64(m_twiddles.get_unchecked(12 * j + 9..).as_ptr().cast());
                            let tw10 =
                                vld1q_f64(m_twiddles.get_unchecked(12 * j + 10..).as_ptr().cast());
                            let tw11 =
                                vld1q_f64(m_twiddles.get_unchecked(12 * j + 11..).as_ptr().cast());

                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + thirteenth..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * thirteenth..).as_ptr().cast()),
                                tw1,
                            );
                            let u3 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 3 * thirteenth..).as_ptr().cast()),
                                tw2,
                            );
                            let u4 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 4 * thirteenth..).as_ptr().cast()),
                                tw3,
                            );
                            let u5 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 5 * thirteenth..).as_ptr().cast()),
                                tw4,
                            );
                            let u6 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 6 * thirteenth..).as_ptr().cast()),
                                tw5,
                            );
                            let u7 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 7 * thirteenth..).as_ptr().cast()),
                                tw6,
                            );
                            let u8 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 8 * thirteenth..).as_ptr().cast()),
                                tw7,
                            );
                            let u9 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 9 * thirteenth..).as_ptr().cast()),
                                tw8,
                            );
                            let u10 = vfcmulq_fcma_f64(
                                vld1q_f64(
                                    data.get_unchecked(j + 10 * thirteenth..).as_ptr().cast(),
                                ),
                                tw9,
                            );
                            let u11 = vfcmulq_fcma_f64(
                                vld1q_f64(
                                    data.get_unchecked(j + 11 * thirteenth..).as_ptr().cast(),
                                ),
                                tw10,
                            );
                            let u12 = vfcmulq_fcma_f64(
                                vld1q_f64(
                                    data.get_unchecked(j + 12 * thirteenth..).as_ptr().cast(),
                                ),
                                tw11,
                            );

                            // Radix-13 butterfly

                            let y00 = u0;
                            let (x1p12, x1m12) = NeonButterfly::butterfly2_f64(u1, u12);
                            let x1m12 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m12);
                            let y00 = vaddq_f64(y00, x1p12);
                            let (x2p11, x2m11) = NeonButterfly::butterfly2_f64(u2, u11);
                            let x2m11 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m11);
                            let y00 = vaddq_f64(y00, x2p11);
                            let (x3p10, x3m10) = NeonButterfly::butterfly2_f64(u3, u10);
                            let x3m10 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m10);
                            let y00 = vaddq_f64(y00, x3p10);
                            let (x4p9, x4m9) = NeonButterfly::butterfly2_f64(u4, u9);
                            let x4m9 = vcaddq_rot90_f64(vdupq_n_f64(0.), x4m9);
                            let y00 = vaddq_f64(y00, x4p9);
                            let (x5p8, x5m8) = NeonButterfly::butterfly2_f64(u5, u8);
                            let x5m8 = vcaddq_rot90_f64(vdupq_n_f64(0.), x5m8);
                            let y00 = vaddq_f64(y00, x5p8);
                            let (x6p7, x6m7) = NeonButterfly::butterfly2_f64(u6, u7);
                            let x6m7 = vcaddq_rot90_f64(vdupq_n_f64(0.), x6m7);
                            let y00 = vaddq_f64(y00, x6p7);

                            let m0112a = vfmaq_n_f64(u0, x1p12, self.twiddle1.re);
                            let m0112a = vfmaq_n_f64(m0112a, x2p11, self.twiddle2.re);
                            let m0112a = vfmaq_n_f64(m0112a, x3p10, self.twiddle3.re);
                            let m0112a = vfmaq_n_f64(m0112a, x4p9, self.twiddle4.re);
                            let m0112a = vfmaq_n_f64(m0112a, x5p8, self.twiddle5.re);
                            let m0112a = vfmaq_n_f64(m0112a, x6p7, self.twiddle6.re);
                            let m0112b = vmulq_n_f64(x1m12, self.twiddle1.im);
                            let m0112b = vfmaq_n_f64(m0112b, x2m11, self.twiddle2.im);
                            let m0112b = vfmaq_n_f64(m0112b, x3m10, self.twiddle3.im);
                            let m0112b = vfmaq_n_f64(m0112b, x4m9, self.twiddle4.im);
                            let m0112b = vfmaq_n_f64(m0112b, x5m8, self.twiddle5.im);
                            let m0112b = vfmaq_n_f64(m0112b, x6m7, self.twiddle6.im);
                            let (y01, y12) = NeonButterfly::butterfly2_f64(m0112a, m0112b);

                            let m0211a = vfmaq_n_f64(u0, x1p12, self.twiddle2.re);
                            let m0211a = vfmaq_n_f64(m0211a, x2p11, self.twiddle4.re);
                            let m0211a = vfmaq_n_f64(m0211a, x3p10, self.twiddle6.re);
                            let m0211a = vfmaq_n_f64(m0211a, x4p9, self.twiddle5.re);
                            let m0211a = vfmaq_n_f64(m0211a, x5p8, self.twiddle3.re);
                            let m0211a = vfmaq_n_f64(m0211a, x6p7, self.twiddle1.re);
                            let m0211b = vmulq_n_f64(x1m12, self.twiddle2.im);
                            let m0211b = vfmaq_n_f64(m0211b, x2m11, self.twiddle4.im);
                            let m0211b = vfmaq_n_f64(m0211b, x3m10, self.twiddle6.im);
                            let m0211b = vfmsq_n_f64(m0211b, x4m9, self.twiddle5.im);
                            let m0211b = vfmsq_n_f64(m0211b, x5m8, self.twiddle3.im);
                            let m0211b = vfmsq_n_f64(m0211b, x6m7, self.twiddle1.im);
                            let (y02, y11) = NeonButterfly::butterfly2_f64(m0211a, m0211b);

                            let m0310a = vfmaq_n_f64(u0, x1p12, self.twiddle3.re);
                            let m0310a = vfmaq_n_f64(m0310a, x2p11, self.twiddle6.re);
                            let m0310a = vfmaq_n_f64(m0310a, x3p10, self.twiddle4.re);
                            let m0310a = vfmaq_n_f64(m0310a, x4p9, self.twiddle1.re);
                            let m0310a = vfmaq_n_f64(m0310a, x5p8, self.twiddle2.re);
                            let m0310a = vfmaq_n_f64(m0310a, x6p7, self.twiddle5.re);
                            let m0310b = vmulq_n_f64(x1m12, self.twiddle3.im);
                            let m0310b = vfmaq_n_f64(m0310b, x2m11, self.twiddle6.im);
                            let m0310b = vfmsq_n_f64(m0310b, x3m10, self.twiddle4.im);
                            let m0310b = vfmsq_n_f64(m0310b, x4m9, self.twiddle1.im);
                            let m0310b = vfmaq_n_f64(m0310b, x5m8, self.twiddle2.im);
                            let m0310b = vfmaq_n_f64(m0310b, x6m7, self.twiddle5.im);
                            let (y03, y10) = NeonButterfly::butterfly2_f64(m0310a, m0310b);

                            let m0409a = vfmaq_n_f64(u0, x1p12, self.twiddle4.re);
                            let m0409a = vfmaq_n_f64(m0409a, x2p11, self.twiddle5.re);
                            let m0409a = vfmaq_n_f64(m0409a, x3p10, self.twiddle1.re);
                            let m0409a = vfmaq_n_f64(m0409a, x4p9, self.twiddle3.re);
                            let m0409a = vfmaq_n_f64(m0409a, x5p8, self.twiddle6.re);
                            let m0409a = vfmaq_n_f64(m0409a, x6p7, self.twiddle2.re);
                            let m0409b = vmulq_n_f64(x1m12, self.twiddle4.im);
                            let m0409b = vfmsq_n_f64(m0409b, x2m11, self.twiddle5.im);
                            let m0409b = vfmsq_n_f64(m0409b, x3m10, self.twiddle1.im);
                            let m0409b = vfmaq_n_f64(m0409b, x4m9, self.twiddle3.im);
                            let m0409b = vfmsq_n_f64(m0409b, x5m8, self.twiddle6.im);
                            let m0409b = vfmsq_n_f64(m0409b, x6m7, self.twiddle2.im);
                            let (y04, y09) = NeonButterfly::butterfly2_f64(m0409a, m0409b);

                            let m0508a = vfmaq_n_f64(u0, x1p12, self.twiddle5.re);
                            let m0508a = vfmaq_n_f64(m0508a, x2p11, self.twiddle3.re);
                            let m0508a = vfmaq_n_f64(m0508a, x3p10, self.twiddle2.re);
                            let m0508a = vfmaq_n_f64(m0508a, x4p9, self.twiddle6.re);
                            let m0508a = vfmaq_n_f64(m0508a, x5p8, self.twiddle1.re);
                            let m0508a = vfmaq_n_f64(m0508a, x6p7, self.twiddle4.re);
                            let m0508b = vmulq_n_f64(x1m12, self.twiddle5.im);
                            let m0508b = vfmsq_n_f64(m0508b, x2m11, self.twiddle3.im);
                            let m0508b = vfmaq_n_f64(m0508b, x3m10, self.twiddle2.im);
                            let m0508b = vfmsq_n_f64(m0508b, x4m9, self.twiddle6.im);
                            let m0508b = vfmsq_n_f64(m0508b, x5m8, self.twiddle1.im);
                            let m0508b = vfmaq_n_f64(m0508b, x6m7, self.twiddle4.im);
                            let (y05, y08) = NeonButterfly::butterfly2_f64(m0508a, m0508b);

                            let m0607a = vfmaq_n_f64(u0, x1p12, self.twiddle6.re);
                            let m0607a = vfmaq_n_f64(m0607a, x2p11, self.twiddle1.re);
                            let m0607a = vfmaq_n_f64(m0607a, x3p10, self.twiddle5.re);
                            let m0607a = vfmaq_n_f64(m0607a, x4p9, self.twiddle2.re);
                            let m0607a = vfmaq_n_f64(m0607a, x5p8, self.twiddle4.re);
                            let m0607a = vfmaq_n_f64(m0607a, x6p7, self.twiddle3.re);
                            let m0607b = vmulq_n_f64(x1m12, self.twiddle6.im);
                            let m0607b = vfmsq_n_f64(m0607b, x2m11, self.twiddle1.im);
                            let m0607b = vfmaq_n_f64(m0607b, x3m10, self.twiddle5.im);
                            let m0607b = vfmsq_n_f64(m0607b, x4m9, self.twiddle2.im);
                            let m0607b = vfmaq_n_f64(m0607b, x5m8, self.twiddle4.im);
                            let m0607b = vfmsq_n_f64(m0607b, x6m7, self.twiddle3.im);
                            let (y06, y07) = NeonButterfly::butterfly2_f64(m0607a, m0607b);

                            // Store results
                            vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            vst1q_f64(
                                data.get_unchecked_mut(j + thirteenth..).as_mut_ptr().cast(),
                                y01,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 3 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 4 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 5 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 6 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 7 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y07,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 8 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y08,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 9 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y09,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 10 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 11 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y11,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 12 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y12,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 12..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaRadix13<f32> {
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

impl NeonFcmaRadix13<f32> {
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
                bitreversed_transpose::<Complex<f32>, 13>(13, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = 13;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 13;
                    let thirteenth = len / 13;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < thirteenth {
                            let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(12 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 2..).as_ptr().cast());
                            let tw2 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 4..).as_ptr().cast());
                            let tw3 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 6..).as_ptr().cast());
                            let tw4 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 8..).as_ptr().cast());
                            let tw5 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 10..).as_ptr().cast());
                            let tw6 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 12..).as_ptr().cast());
                            let tw7 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 14..).as_ptr().cast());
                            let tw8 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 16..).as_ptr().cast());
                            let tw9 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 18..).as_ptr().cast());
                            let tw10 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 20..).as_ptr().cast());
                            let tw11 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 22..).as_ptr().cast());

                            let u1 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + thirteenth..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * thirteenth..).as_ptr().cast()),
                                tw1,
                            );
                            let u3 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 3 * thirteenth..).as_ptr().cast()),
                                tw2,
                            );
                            let u4 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 4 * thirteenth..).as_ptr().cast()),
                                tw3,
                            );
                            let u5 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 5 * thirteenth..).as_ptr().cast()),
                                tw4,
                            );
                            let u6 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 6 * thirteenth..).as_ptr().cast()),
                                tw5,
                            );
                            let u7 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 7 * thirteenth..).as_ptr().cast()),
                                tw6,
                            );
                            let u8 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 8 * thirteenth..).as_ptr().cast()),
                                tw7,
                            );
                            let u9 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 9 * thirteenth..).as_ptr().cast()),
                                tw8,
                            );
                            let u10 = vfcmulq_fcma_f32(
                                vld1q_f32(
                                    data.get_unchecked(j + 10 * thirteenth..).as_ptr().cast(),
                                ),
                                tw9,
                            );
                            let u11 = vfcmulq_fcma_f32(
                                vld1q_f32(
                                    data.get_unchecked(j + 11 * thirteenth..).as_ptr().cast(),
                                ),
                                tw10,
                            );
                            let u12 = vfcmulq_fcma_f32(
                                vld1q_f32(
                                    data.get_unchecked(j + 12 * thirteenth..).as_ptr().cast(),
                                ),
                                tw11,
                            );

                            // Radix-13 butterfly

                            let y00 = u0;
                            let (x1p12, x1m12) = NeonButterfly::butterfly2_f32(u1, u12);
                            let x1m12 = vcaddq_rot90_f32(vdupq_n_f32(0.), x1m12);
                            let y00 = vaddq_f32(y00, x1p12);
                            let (x2p11, x2m11) = NeonButterfly::butterfly2_f32(u2, u11);
                            let x2m11 = vcaddq_rot90_f32(vdupq_n_f32(0.), x2m11);
                            let y00 = vaddq_f32(y00, x2p11);
                            let (x3p10, x3m10) = NeonButterfly::butterfly2_f32(u3, u10);
                            let x3m10 = vcaddq_rot90_f32(vdupq_n_f32(0.), x3m10);
                            let y00 = vaddq_f32(y00, x3p10);
                            let (x4p9, x4m9) = NeonButterfly::butterfly2_f32(u4, u9);
                            let x4m9 = vcaddq_rot90_f32(vdupq_n_f32(0.), x4m9);
                            let y00 = vaddq_f32(y00, x4p9);
                            let (x5p8, x5m8) = NeonButterfly::butterfly2_f32(u5, u8);
                            let x5m8 = vcaddq_rot90_f32(vdupq_n_f32(0.), x5m8);
                            let y00 = vaddq_f32(y00, x5p8);
                            let (x6p7, x6m7) = NeonButterfly::butterfly2_f32(u6, u7);
                            let x6m7 = vcaddq_rot90_f32(vdupq_n_f32(0.), x6m7);
                            let y00 = vaddq_f32(y00, x6p7);

                            let m0112a = vfmaq_n_f32(u0, x1p12, self.twiddle1.re);
                            let m0112a = vfmaq_n_f32(m0112a, x2p11, self.twiddle2.re);
                            let m0112a = vfmaq_n_f32(m0112a, x3p10, self.twiddle3.re);
                            let m0112a = vfmaq_n_f32(m0112a, x4p9, self.twiddle4.re);
                            let m0112a = vfmaq_n_f32(m0112a, x5p8, self.twiddle5.re);
                            let m0112a = vfmaq_n_f32(m0112a, x6p7, self.twiddle6.re);
                            let m0112b = vmulq_n_f32(x1m12, self.twiddle1.im);
                            let m0112b = vfmaq_n_f32(m0112b, x2m11, self.twiddle2.im);
                            let m0112b = vfmaq_n_f32(m0112b, x3m10, self.twiddle3.im);
                            let m0112b = vfmaq_n_f32(m0112b, x4m9, self.twiddle4.im);
                            let m0112b = vfmaq_n_f32(m0112b, x5m8, self.twiddle5.im);
                            let m0112b = vfmaq_n_f32(m0112b, x6m7, self.twiddle6.im);
                            let (y01, y12) = NeonButterfly::butterfly2_f32(m0112a, m0112b);

                            let m0211a = vfmaq_n_f32(u0, x1p12, self.twiddle2.re);
                            let m0211a = vfmaq_n_f32(m0211a, x2p11, self.twiddle4.re);
                            let m0211a = vfmaq_n_f32(m0211a, x3p10, self.twiddle6.re);
                            let m0211a = vfmaq_n_f32(m0211a, x4p9, self.twiddle5.re);
                            let m0211a = vfmaq_n_f32(m0211a, x5p8, self.twiddle3.re);
                            let m0211a = vfmaq_n_f32(m0211a, x6p7, self.twiddle1.re);
                            let m0211b = vmulq_n_f32(x1m12, self.twiddle2.im);
                            let m0211b = vfmaq_n_f32(m0211b, x2m11, self.twiddle4.im);
                            let m0211b = vfmaq_n_f32(m0211b, x3m10, self.twiddle6.im);
                            let m0211b = vfmsq_n_f32(m0211b, x4m9, self.twiddle5.im);
                            let m0211b = vfmsq_n_f32(m0211b, x5m8, self.twiddle3.im);
                            let m0211b = vfmsq_n_f32(m0211b, x6m7, self.twiddle1.im);
                            let (y02, y11) = NeonButterfly::butterfly2_f32(m0211a, m0211b);

                            let m0310a = vfmaq_n_f32(u0, x1p12, self.twiddle3.re);
                            let m0310a = vfmaq_n_f32(m0310a, x2p11, self.twiddle6.re);
                            let m0310a = vfmaq_n_f32(m0310a, x3p10, self.twiddle4.re);
                            let m0310a = vfmaq_n_f32(m0310a, x4p9, self.twiddle1.re);
                            let m0310a = vfmaq_n_f32(m0310a, x5p8, self.twiddle2.re);
                            let m0310a = vfmaq_n_f32(m0310a, x6p7, self.twiddle5.re);
                            let m0310b = vmulq_n_f32(x1m12, self.twiddle3.im);
                            let m0310b = vfmaq_n_f32(m0310b, x2m11, self.twiddle6.im);
                            let m0310b = vfmsq_n_f32(m0310b, x3m10, self.twiddle4.im);
                            let m0310b = vfmsq_n_f32(m0310b, x4m9, self.twiddle1.im);
                            let m0310b = vfmaq_n_f32(m0310b, x5m8, self.twiddle2.im);
                            let m0310b = vfmaq_n_f32(m0310b, x6m7, self.twiddle5.im);
                            let (y03, y10) = NeonButterfly::butterfly2_f32(m0310a, m0310b);

                            let m0409a = vfmaq_n_f32(u0, x1p12, self.twiddle4.re);
                            let m0409a = vfmaq_n_f32(m0409a, x2p11, self.twiddle5.re);
                            let m0409a = vfmaq_n_f32(m0409a, x3p10, self.twiddle1.re);
                            let m0409a = vfmaq_n_f32(m0409a, x4p9, self.twiddle3.re);
                            let m0409a = vfmaq_n_f32(m0409a, x5p8, self.twiddle6.re);
                            let m0409a = vfmaq_n_f32(m0409a, x6p7, self.twiddle2.re);
                            let m0409b = vmulq_n_f32(x1m12, self.twiddle4.im);
                            let m0409b = vfmsq_n_f32(m0409b, x2m11, self.twiddle5.im);
                            let m0409b = vfmsq_n_f32(m0409b, x3m10, self.twiddle1.im);
                            let m0409b = vfmaq_n_f32(m0409b, x4m9, self.twiddle3.im);
                            let m0409b = vfmsq_n_f32(m0409b, x5m8, self.twiddle6.im);
                            let m0409b = vfmsq_n_f32(m0409b, x6m7, self.twiddle2.im);
                            let (y04, y09) = NeonButterfly::butterfly2_f32(m0409a, m0409b);

                            let m0508a = vfmaq_n_f32(u0, x1p12, self.twiddle5.re);
                            let m0508a = vfmaq_n_f32(m0508a, x2p11, self.twiddle3.re);
                            let m0508a = vfmaq_n_f32(m0508a, x3p10, self.twiddle2.re);
                            let m0508a = vfmaq_n_f32(m0508a, x4p9, self.twiddle6.re);
                            let m0508a = vfmaq_n_f32(m0508a, x5p8, self.twiddle1.re);
                            let m0508a = vfmaq_n_f32(m0508a, x6p7, self.twiddle4.re);
                            let m0508b = vmulq_n_f32(x1m12, self.twiddle5.im);
                            let m0508b = vfmsq_n_f32(m0508b, x2m11, self.twiddle3.im);
                            let m0508b = vfmaq_n_f32(m0508b, x3m10, self.twiddle2.im);
                            let m0508b = vfmsq_n_f32(m0508b, x4m9, self.twiddle6.im);
                            let m0508b = vfmsq_n_f32(m0508b, x5m8, self.twiddle1.im);
                            let m0508b = vfmaq_n_f32(m0508b, x6m7, self.twiddle4.im);
                            let (y05, y08) = NeonButterfly::butterfly2_f32(m0508a, m0508b);

                            let m0607a = vfmaq_n_f32(u0, x1p12, self.twiddle6.re);
                            let m0607a = vfmaq_n_f32(m0607a, x2p11, self.twiddle1.re);
                            let m0607a = vfmaq_n_f32(m0607a, x3p10, self.twiddle5.re);
                            let m0607a = vfmaq_n_f32(m0607a, x4p9, self.twiddle2.re);
                            let m0607a = vfmaq_n_f32(m0607a, x5p8, self.twiddle4.re);
                            let m0607a = vfmaq_n_f32(m0607a, x6p7, self.twiddle3.re);
                            let m0607b = vmulq_n_f32(x1m12, self.twiddle6.im);
                            let m0607b = vfmsq_n_f32(m0607b, x2m11, self.twiddle1.im);
                            let m0607b = vfmaq_n_f32(m0607b, x3m10, self.twiddle5.im);
                            let m0607b = vfmsq_n_f32(m0607b, x4m9, self.twiddle2.im);
                            let m0607b = vfmaq_n_f32(m0607b, x5m8, self.twiddle4.im);
                            let m0607b = vfmsq_n_f32(m0607b, x6m7, self.twiddle3.im);
                            let (y06, y07) = NeonButterfly::butterfly2_f32(m0607a, m0607b);

                            // Store results
                            vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            vst1q_f32(
                                data.get_unchecked_mut(j + thirteenth..).as_mut_ptr().cast(),
                                y01,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 3 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 4 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 5 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 6 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 7 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y07,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 8 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y08,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 9 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y09,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 10 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 11 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y11,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 12 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y12,
                            );

                            j += 2;
                        }

                        for j in j..thirteenth {
                            let u0 = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                            let w0w1 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j..).as_ptr().cast());
                            let w2w3 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 2..).as_ptr().cast());
                            let w4w5 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 4..).as_ptr().cast());
                            let w6w7 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 6..).as_ptr().cast());
                            let w8w9 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 8..).as_ptr().cast());
                            let w10w11 =
                                vld1q_f32(m_twiddles.get_unchecked(12 * j + 10..).as_ptr().cast());

                            let u1u2 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + thirteenth..).as_ptr().cast()),
                                    vld1_f32(
                                        data.get_unchecked(j + 2 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                w0w1,
                            );
                            let u3u4 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(
                                        data.get_unchecked(j + 3 * thirteenth..).as_ptr().cast(),
                                    ),
                                    vld1_f32(
                                        data.get_unchecked(j + 4 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                w2w3,
                            );
                            let u5u6 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(
                                        data.get_unchecked(j + 5 * thirteenth..).as_ptr().cast(),
                                    ),
                                    vld1_f32(
                                        data.get_unchecked(j + 6 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                w4w5,
                            );
                            let u7u8 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(
                                        data.get_unchecked(j + 7 * thirteenth..).as_ptr().cast(),
                                    ),
                                    vld1_f32(
                                        data.get_unchecked(j + 8 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                w6w7,
                            );
                            let u9u10 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(
                                        data.get_unchecked(j + 9 * thirteenth..).as_ptr().cast(),
                                    ),
                                    vld1_f32(
                                        data.get_unchecked(j + 10 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                w8w9,
                            );
                            let u11u12 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(
                                        data.get_unchecked(j + 11 * thirteenth..).as_ptr().cast(),
                                    ),
                                    vld1_f32(
                                        data.get_unchecked(j + 12 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                w10w11,
                            );

                            let u1 = vget_low_f32(u1u2);
                            let u2 = vget_high_f32(u1u2);
                            let u3 = vget_low_f32(u3u4);
                            let u4 = vget_high_f32(u3u4);
                            let u5 = vget_low_f32(u5u6);
                            let u6 = vget_high_f32(u5u6);
                            let u7 = vget_low_f32(u7u8);
                            let u8 = vget_high_f32(u7u8);
                            let u9 = vget_low_f32(u9u10);
                            let u10 = vget_high_f32(u9u10);
                            let u11 = vget_low_f32(u11u12);
                            let u12 = vget_high_f32(u11u12);

                            // Radix-13 butterfly

                            let y00 = u0;
                            let (x1p12, x1m12) = NeonButterfly::butterfly2h_f32(u1, u12);
                            let x1m12 = vcadd_rot90_f32(vdup_n_f32(0.), x1m12);
                            let y00 = vadd_f32(y00, x1p12);
                            let (x2p11, x2m11) = NeonButterfly::butterfly2h_f32(u2, u11);
                            let x2m11 = vcadd_rot90_f32(vdup_n_f32(0.), x2m11);
                            let y00 = vadd_f32(y00, x2p11);
                            let (x3p10, x3m10) = NeonButterfly::butterfly2h_f32(u3, u10);
                            let x3m10 = vcadd_rot90_f32(vdup_n_f32(0.), x3m10);
                            let y00 = vadd_f32(y00, x3p10);
                            let (x4p9, x4m9) = NeonButterfly::butterfly2h_f32(u4, u9);
                            let x4m9 = vcadd_rot90_f32(vdup_n_f32(0.), x4m9);
                            let y00 = vadd_f32(y00, x4p9);
                            let (x5p8, x5m8) = NeonButterfly::butterfly2h_f32(u5, u8);
                            let x5m8 = vcadd_rot90_f32(vdup_n_f32(0.), x5m8);
                            let y00 = vadd_f32(y00, x5p8);
                            let (x6p7, x6m7) = NeonButterfly::butterfly2h_f32(u6, u7);
                            let x6m7 = vcadd_rot90_f32(vdup_n_f32(0.), x6m7);
                            let y00 = vadd_f32(y00, x6p7);

                            let m0112a = vfma_n_f32(u0, x1p12, self.twiddle1.re);
                            let m0112a = vfma_n_f32(m0112a, x2p11, self.twiddle2.re);
                            let m0112a = vfma_n_f32(m0112a, x3p10, self.twiddle3.re);
                            let m0112a = vfma_n_f32(m0112a, x4p9, self.twiddle4.re);
                            let m0112a = vfma_n_f32(m0112a, x5p8, self.twiddle5.re);
                            let m0112a = vfma_n_f32(m0112a, x6p7, self.twiddle6.re);
                            let m0112b = vmul_n_f32(x1m12, self.twiddle1.im);
                            let m0112b = vfma_n_f32(m0112b, x2m11, self.twiddle2.im);
                            let m0112b = vfma_n_f32(m0112b, x3m10, self.twiddle3.im);
                            let m0112b = vfma_n_f32(m0112b, x4m9, self.twiddle4.im);
                            let m0112b = vfma_n_f32(m0112b, x5m8, self.twiddle5.im);
                            let m0112b = vfma_n_f32(m0112b, x6m7, self.twiddle6.im);
                            let (y01, y12) = NeonButterfly::butterfly2h_f32(m0112a, m0112b);

                            let m0211a = vfma_n_f32(u0, x1p12, self.twiddle2.re);
                            let m0211a = vfma_n_f32(m0211a, x2p11, self.twiddle4.re);
                            let m0211a = vfma_n_f32(m0211a, x3p10, self.twiddle6.re);
                            let m0211a = vfma_n_f32(m0211a, x4p9, self.twiddle5.re);
                            let m0211a = vfma_n_f32(m0211a, x5p8, self.twiddle3.re);
                            let m0211a = vfma_n_f32(m0211a, x6p7, self.twiddle1.re);
                            let m0211b = vmul_n_f32(x1m12, self.twiddle2.im);
                            let m0211b = vfma_n_f32(m0211b, x2m11, self.twiddle4.im);
                            let m0211b = vfma_n_f32(m0211b, x3m10, self.twiddle6.im);
                            let m0211b = vfms_n_f32(m0211b, x4m9, self.twiddle5.im);
                            let m0211b = vfms_n_f32(m0211b, x5m8, self.twiddle3.im);
                            let m0211b = vfms_n_f32(m0211b, x6m7, self.twiddle1.im);
                            let (y02, y11) = NeonButterfly::butterfly2h_f32(m0211a, m0211b);

                            let m0310a = vfma_n_f32(u0, x1p12, self.twiddle3.re);
                            let m0310a = vfma_n_f32(m0310a, x2p11, self.twiddle6.re);
                            let m0310a = vfma_n_f32(m0310a, x3p10, self.twiddle4.re);
                            let m0310a = vfma_n_f32(m0310a, x4p9, self.twiddle1.re);
                            let m0310a = vfma_n_f32(m0310a, x5p8, self.twiddle2.re);
                            let m0310a = vfma_n_f32(m0310a, x6p7, self.twiddle5.re);
                            let m0310b = vmul_n_f32(x1m12, self.twiddle3.im);
                            let m0310b = vfma_n_f32(m0310b, x2m11, self.twiddle6.im);
                            let m0310b = vfms_n_f32(m0310b, x3m10, self.twiddle4.im);
                            let m0310b = vfms_n_f32(m0310b, x4m9, self.twiddle1.im);
                            let m0310b = vfma_n_f32(m0310b, x5m8, self.twiddle2.im);
                            let m0310b = vfma_n_f32(m0310b, x6m7, self.twiddle5.im);
                            let (y03, y10) = NeonButterfly::butterfly2h_f32(m0310a, m0310b);

                            let m0409a = vfma_n_f32(u0, x1p12, self.twiddle4.re);
                            let m0409a = vfma_n_f32(m0409a, x2p11, self.twiddle5.re);
                            let m0409a = vfma_n_f32(m0409a, x3p10, self.twiddle1.re);
                            let m0409a = vfma_n_f32(m0409a, x4p9, self.twiddle3.re);
                            let m0409a = vfma_n_f32(m0409a, x5p8, self.twiddle6.re);
                            let m0409a = vfma_n_f32(m0409a, x6p7, self.twiddle2.re);
                            let m0409b = vmul_n_f32(x1m12, self.twiddle4.im);
                            let m0409b = vfms_n_f32(m0409b, x2m11, self.twiddle5.im);
                            let m0409b = vfms_n_f32(m0409b, x3m10, self.twiddle1.im);
                            let m0409b = vfma_n_f32(m0409b, x4m9, self.twiddle3.im);
                            let m0409b = vfms_n_f32(m0409b, x5m8, self.twiddle6.im);
                            let m0409b = vfms_n_f32(m0409b, x6m7, self.twiddle2.im);
                            let (y04, y09) = NeonButterfly::butterfly2h_f32(m0409a, m0409b);

                            let m0508a = vfma_n_f32(u0, x1p12, self.twiddle5.re);
                            let m0508a = vfma_n_f32(m0508a, x2p11, self.twiddle3.re);
                            let m0508a = vfma_n_f32(m0508a, x3p10, self.twiddle2.re);
                            let m0508a = vfma_n_f32(m0508a, x4p9, self.twiddle6.re);
                            let m0508a = vfma_n_f32(m0508a, x5p8, self.twiddle1.re);
                            let m0508a = vfma_n_f32(m0508a, x6p7, self.twiddle4.re);
                            let m0508b = vmul_n_f32(x1m12, self.twiddle5.im);
                            let m0508b = vfms_n_f32(m0508b, x2m11, self.twiddle3.im);
                            let m0508b = vfma_n_f32(m0508b, x3m10, self.twiddle2.im);
                            let m0508b = vfms_n_f32(m0508b, x4m9, self.twiddle6.im);
                            let m0508b = vfms_n_f32(m0508b, x5m8, self.twiddle1.im);
                            let m0508b = vfma_n_f32(m0508b, x6m7, self.twiddle4.im);
                            let (y05, y08) = NeonButterfly::butterfly2h_f32(m0508a, m0508b);

                            let m0607a = vfma_n_f32(u0, x1p12, self.twiddle6.re);
                            let m0607a = vfma_n_f32(m0607a, x2p11, self.twiddle1.re);
                            let m0607a = vfma_n_f32(m0607a, x3p10, self.twiddle5.re);
                            let m0607a = vfma_n_f32(m0607a, x4p9, self.twiddle2.re);
                            let m0607a = vfma_n_f32(m0607a, x5p8, self.twiddle4.re);
                            let m0607a = vfma_n_f32(m0607a, x6p7, self.twiddle3.re);
                            let m0607b = vmul_n_f32(x1m12, self.twiddle6.im);
                            let m0607b = vfms_n_f32(m0607b, x2m11, self.twiddle1.im);
                            let m0607b = vfma_n_f32(m0607b, x3m10, self.twiddle5.im);
                            let m0607b = vfms_n_f32(m0607b, x4m9, self.twiddle2.im);
                            let m0607b = vfma_n_f32(m0607b, x5m8, self.twiddle4.im);
                            let m0607b = vfms_n_f32(m0607b, x6m7, self.twiddle3.im);
                            let (y06, y07) = NeonButterfly::butterfly2h_f32(m0607a, m0607b);

                            // Store results
                            vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            vst1_f32(
                                data.get_unchecked_mut(j + thirteenth..).as_mut_ptr().cast(),
                                y01,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 2 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 3 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 4 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 5 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 6 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 7 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y07,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 8 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y08,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 9 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y09,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 10 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 11 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y11,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 12 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y12,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 12..];
                }
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
    fn test_neon_radix13f() {
        for i in 1..5 {
            let size = 13usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaRadix13::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix13::new(size, FftDirection::Inverse).unwrap();
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
    fn test_neon_radix13_f64() {
        for i in 1..5 {
            let size = 13usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaRadix13::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix13::new(size, FftDirection::Inverse).unwrap();
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
