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
use crate::neon::radix11::neon_bitreversed_transpose_f32_radix11;
use crate::neon::util::{create_neon_twiddles, vfcmulq_f64, vfcmulq_fcma_f32, vfcmulq_fcma_f64};
use crate::radix11::Radix11Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{bitreversed_transpose, compute_logarithm, compute_twiddle, is_power_of_eleven};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::aarch64::*;
use std::fmt::Display;
use std::sync::Arc;

pub(crate) struct NeonFcmaRadix11<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    direction: FftDirection,
    butterfly: Arc<dyn CompositeFftExecutor<T> + Send + Sync>,
    butterfly_length: usize,
}

impl<
    T: Default
        + Clone
        + Radix11Twiddles
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
> NeonFcmaRadix11<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonFcmaRadix11<T>, ZaftError> {
        assert!(
            is_power_of_eleven(size as u64),
            "Input length must be a power of 11"
        );

        let log11 = compute_logarithm::<11>(size).unwrap();
        let butterfly = match log11 {
            0 => T::butterfly1(fft_direction)?,
            1 => T::butterfly11(fft_direction)?,
            _ => {
                T::butterfly121(fft_direction).map_or_else(|| T::butterfly11(fft_direction), Ok)?
            }
        };

        let butterfly_length = butterfly.length();

        let twiddles = create_neon_twiddles::<T, 11>(butterfly_length, size, fft_direction)?;

        Ok(NeonFcmaRadix11 {
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 11, fft_direction),
            twiddle2: compute_twiddle(2, 11, fft_direction),
            twiddle3: compute_twiddle(3, 11, fft_direction),
            twiddle4: compute_twiddle(4, 11, fft_direction),
            twiddle5: compute_twiddle(5, 11, fft_direction),
            direction: fft_direction,
            butterfly,
            butterfly_length,
        })
    }
}

impl FftExecutor<f64> for NeonFcmaRadix11<f64> {
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

impl NeonFcmaRadix11<f64> {
    #[target_feature(enable = "fcma")]
    fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
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
                bitreversed_transpose::<Complex<f64>, 11>(
                    self.butterfly_length,
                    chunk,
                    &mut scratch,
                );

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.butterfly_length;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 11;
                    let eleventh = len / 11;

                    for data in chunk.chunks_exact_mut(len) {
                        for j in 0..eleventh {
                            let tw0 = vld1q_f64(m_twiddles.get_unchecked(10 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f64(m_twiddles.get_unchecked(10 * j + 1..).as_ptr().cast());
                            let tw2 =
                                vld1q_f64(m_twiddles.get_unchecked(10 * j + 2..).as_ptr().cast());
                            let tw3 =
                                vld1q_f64(m_twiddles.get_unchecked(10 * j + 3..).as_ptr().cast());
                            let tw4 =
                                vld1q_f64(m_twiddles.get_unchecked(10 * j + 4..).as_ptr().cast());
                            let tw5 =
                                vld1q_f64(m_twiddles.get_unchecked(10 * j + 5..).as_ptr().cast());
                            let tw6 =
                                vld1q_f64(m_twiddles.get_unchecked(10 * j + 6..).as_ptr().cast());
                            let tw7 =
                                vld1q_f64(m_twiddles.get_unchecked(10 * j + 7..).as_ptr().cast());
                            let tw8 =
                                vld1q_f64(m_twiddles.get_unchecked(10 * j + 8..).as_ptr().cast());
                            let tw9 =
                                vld1q_f64(m_twiddles.get_unchecked(10 * j + 9..).as_ptr().cast());

                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + eleventh..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * eleventh..).as_ptr().cast()),
                                tw1,
                            );
                            let u3 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 3 * eleventh..).as_ptr().cast()),
                                tw2,
                            );
                            let u4 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 4 * eleventh..).as_ptr().cast()),
                                tw3,
                            );
                            let u5 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 5 * eleventh..).as_ptr().cast()),
                                tw4,
                            );
                            let u6 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 6 * eleventh..).as_ptr().cast()),
                                tw5,
                            );
                            let u7 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 7 * eleventh..).as_ptr().cast()),
                                tw6,
                            );
                            let u8 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 8 * eleventh..).as_ptr().cast()),
                                tw7,
                            );
                            let u9 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 9 * eleventh..).as_ptr().cast()),
                                tw8,
                            );
                            let u10 = vfcmulq_fcma_f64(
                                vld1q_f64(data.get_unchecked(j + 10 * eleventh..).as_ptr().cast()),
                                tw9,
                            );

                            // Radix-11 butterfly

                            let y00 = u0;
                            let (x1p10, x1m10) = NeonButterfly::butterfly2_f64(u1, u10);
                            let x1m10 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m10);
                            let y00 = vaddq_f64(y00, x1p10);
                            let (x2p9, x2m9) = NeonButterfly::butterfly2_f64(u2, u9);
                            let x2m9 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m9);
                            let y00 = vaddq_f64(y00, x2p9);
                            let (x3p8, x3m8) = NeonButterfly::butterfly2_f64(u3, u8);
                            let x3m8 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m8);
                            let y00 = vaddq_f64(y00, x3p8);
                            let (x4p7, x4m7) = NeonButterfly::butterfly2_f64(u4, u7);
                            let x4m7 = vcaddq_rot90_f64(vdupq_n_f64(0.), x4m7);
                            let y00 = vaddq_f64(y00, x4p7);
                            let (x5p6, x5m6) = NeonButterfly::butterfly2_f64(u5, u6);
                            let x5m6 = vcaddq_rot90_f64(vdupq_n_f64(0.), x5m6);
                            let y00 = vaddq_f64(y00, x5p6);

                            let m0110a = vfmaq_n_f64(u0, x1p10, self.twiddle1.re);
                            let m0110a = vfmaq_n_f64(m0110a, x2p9, self.twiddle2.re);
                            let m0110a = vfmaq_n_f64(m0110a, x3p8, self.twiddle3.re);
                            let m0110a = vfmaq_n_f64(m0110a, x4p7, self.twiddle4.re);
                            let m0110a = vfmaq_n_f64(m0110a, x5p6, self.twiddle5.re);
                            let m0110b = vmulq_n_f64(x1m10, self.twiddle1.im);
                            let m0110b = vfmaq_n_f64(m0110b, x2m9, self.twiddle2.im);
                            let m0110b = vfmaq_n_f64(m0110b, x3m8, self.twiddle3.im);
                            let m0110b = vfmaq_n_f64(m0110b, x4m7, self.twiddle4.im);
                            let m0110b = vfmaq_n_f64(m0110b, x5m6, self.twiddle5.im);
                            let (y01, y10) = NeonButterfly::butterfly2_f64(m0110a, m0110b);

                            let m0209a = vfmaq_n_f64(u0, x1p10, self.twiddle2.re);
                            let m0209a = vfmaq_n_f64(m0209a, x2p9, self.twiddle4.re);
                            let m0209a = vfmaq_n_f64(m0209a, x3p8, self.twiddle5.re);
                            let m0209a = vfmaq_n_f64(m0209a, x4p7, self.twiddle3.re);
                            let m0209a = vfmaq_n_f64(m0209a, x5p6, self.twiddle1.re);
                            let m0209b = vmulq_n_f64(x1m10, self.twiddle2.im);
                            let m0209b = vfmaq_n_f64(m0209b, x2m9, self.twiddle4.im);
                            let m0209b = vfmsq_n_f64(m0209b, x3m8, self.twiddle5.im);
                            let m0209b = vfmsq_n_f64(m0209b, x4m7, self.twiddle3.im);
                            let m0209b = vfmsq_n_f64(m0209b, x5m6, self.twiddle1.im);
                            let (y02, y09) = NeonButterfly::butterfly2_f64(m0209a, m0209b);

                            let m0308a = vfmaq_n_f64(u0, x1p10, self.twiddle3.re);
                            let m0308a = vfmaq_n_f64(m0308a, x2p9, self.twiddle5.re);
                            let m0308a = vfmaq_n_f64(m0308a, x3p8, self.twiddle2.re);
                            let m0308a = vfmaq_n_f64(m0308a, x4p7, self.twiddle1.re);
                            let m0308a = vfmaq_n_f64(m0308a, x5p6, self.twiddle4.re);
                            let m0308b = vmulq_n_f64(x1m10, self.twiddle3.im);
                            let m0308b = vfmsq_n_f64(m0308b, x2m9, self.twiddle5.im);
                            let m0308b = vfmsq_n_f64(m0308b, x3m8, self.twiddle2.im);
                            let m0308b = vfmaq_n_f64(m0308b, x4m7, self.twiddle1.im);
                            let m0308b = vfmaq_n_f64(m0308b, x5m6, self.twiddle4.im);
                            let (y03, y08) = NeonButterfly::butterfly2_f64(m0308a, m0308b);

                            let m0407a = vfmaq_n_f64(u0, x1p10, self.twiddle4.re);
                            let m0407a = vfmaq_n_f64(m0407a, x2p9, self.twiddle3.re);
                            let m0407a = vfmaq_n_f64(m0407a, x3p8, self.twiddle1.re);
                            let m0407a = vfmaq_n_f64(m0407a, x4p7, self.twiddle5.re);
                            let m0407a = vfmaq_n_f64(m0407a, x5p6, self.twiddle2.re);
                            let m0407b = vmulq_n_f64(x1m10, self.twiddle4.im);
                            let m0407b = vfmsq_n_f64(m0407b, x2m9, self.twiddle3.im);
                            let m0407b = vfmaq_n_f64(m0407b, x3m8, self.twiddle1.im);
                            let m0407b = vfmaq_n_f64(m0407b, x4m7, self.twiddle5.im);
                            let m0407b = vfmsq_n_f64(m0407b, x5m6, self.twiddle2.im);
                            let (y04, y07) = NeonButterfly::butterfly2_f64(m0407a, m0407b);

                            let m0506a = vfmaq_n_f64(u0, x1p10, self.twiddle5.re);
                            let m0506a = vfmaq_n_f64(m0506a, x2p9, self.twiddle1.re);
                            let m0506a = vfmaq_n_f64(m0506a, x3p8, self.twiddle4.re);
                            let m0506a = vfmaq_n_f64(m0506a, x4p7, self.twiddle2.re);
                            let m0506a = vfmaq_n_f64(m0506a, x5p6, self.twiddle3.re);
                            let m0506b = vmulq_n_f64(x1m10, self.twiddle5.im);
                            let m0506b = vfmsq_n_f64(m0506b, x2m9, self.twiddle1.im);
                            let m0506b = vfmaq_n_f64(m0506b, x3m8, self.twiddle4.im);
                            let m0506b = vfmsq_n_f64(m0506b, x4m7, self.twiddle2.im);
                            let m0506b = vfmaq_n_f64(m0506b, x5m6, self.twiddle3.im);
                            let (y05, y06) = NeonButterfly::butterfly2_f64(m0506a, m0506b);

                            // Store results
                            vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            vst1q_f64(
                                data.get_unchecked_mut(j + eleventh..).as_mut_ptr().cast(),
                                y01,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 3 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 4 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 5 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 6 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 7 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y07,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 8 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y08,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 9 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y09,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 10 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 10..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaRadix11<f32> {
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

impl NeonFcmaRadix11<f32> {
    #[target_feature(enable = "fcma")]
    fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
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
                neon_bitreversed_transpose_f32_radix11(self.butterfly_length, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.butterfly_length;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 11;
                    let eleventh = len / 11;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < eleventh {
                            let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(10 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 2..).as_ptr().cast());
                            let tw2 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 4..).as_ptr().cast());
                            let tw3 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 6..).as_ptr().cast());
                            let tw4 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 8..).as_ptr().cast());
                            let tw5 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 10..).as_ptr().cast());
                            let tw6 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 12..).as_ptr().cast());
                            let tw7 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 14..).as_ptr().cast());
                            let tw8 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 16..).as_ptr().cast());
                            let tw9 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 18..).as_ptr().cast());

                            let u1 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + eleventh..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * eleventh..).as_ptr().cast()),
                                tw1,
                            );
                            let u3 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 3 * eleventh..).as_ptr().cast()),
                                tw2,
                            );
                            let u4 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 4 * eleventh..).as_ptr().cast()),
                                tw3,
                            );
                            let u5 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 5 * eleventh..).as_ptr().cast()),
                                tw4,
                            );
                            let u6 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 6 * eleventh..).as_ptr().cast()),
                                tw5,
                            );
                            let u7 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 7 * eleventh..).as_ptr().cast()),
                                tw6,
                            );
                            let u8 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 8 * eleventh..).as_ptr().cast()),
                                tw7,
                            );
                            let u9 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 9 * eleventh..).as_ptr().cast()),
                                tw8,
                            );
                            let u10 = vfcmulq_fcma_f32(
                                vld1q_f32(data.get_unchecked(j + 10 * eleventh..).as_ptr().cast()),
                                tw9,
                            );

                            // Radix-11 butterfly

                            let y00 = u0;
                            let (x1p10, x1m10) = NeonButterfly::butterfly2_f32(u1, u10);
                            let x1m10 = vcaddq_rot90_f32(vdupq_n_f32(0.), x1m10);
                            let y00 = vaddq_f32(y00, x1p10);
                            let (x2p9, x2m9) = NeonButterfly::butterfly2_f32(u2, u9);
                            let x2m9 = vcaddq_rot90_f32(vdupq_n_f32(0.), x2m9);
                            let y00 = vaddq_f32(y00, x2p9);
                            let (x3p8, x3m8) = NeonButterfly::butterfly2_f32(u3, u8);
                            let x3m8 = vcaddq_rot90_f32(vdupq_n_f32(0.), x3m8);
                            let y00 = vaddq_f32(y00, x3p8);
                            let (x4p7, x4m7) = NeonButterfly::butterfly2_f32(u4, u7);
                            let x4m7 = vcaddq_rot90_f32(vdupq_n_f32(0.), x4m7);
                            let y00 = vaddq_f32(y00, x4p7);
                            let (x5p6, x5m6) = NeonButterfly::butterfly2_f32(u5, u6);
                            let x5m6 = vcaddq_rot90_f32(vdupq_n_f32(0.), x5m6);
                            let y00 = vaddq_f32(y00, x5p6);

                            let m0110a = vfmaq_n_f32(u0, x1p10, self.twiddle1.re);
                            let m0110a = vfmaq_n_f32(m0110a, x2p9, self.twiddle2.re);
                            let m0110a = vfmaq_n_f32(m0110a, x3p8, self.twiddle3.re);
                            let m0110a = vfmaq_n_f32(m0110a, x4p7, self.twiddle4.re);
                            let m0110a = vfmaq_n_f32(m0110a, x5p6, self.twiddle5.re);
                            let m0110b = vmulq_n_f32(x1m10, self.twiddle1.im);
                            let m0110b = vfmaq_n_f32(m0110b, x2m9, self.twiddle2.im);
                            let m0110b = vfmaq_n_f32(m0110b, x3m8, self.twiddle3.im);
                            let m0110b = vfmaq_n_f32(m0110b, x4m7, self.twiddle4.im);
                            let m0110b = vfmaq_n_f32(m0110b, x5m6, self.twiddle5.im);
                            let (y01, y10) = NeonButterfly::butterfly2_f32(m0110a, m0110b);

                            let m0209a = vfmaq_n_f32(u0, x1p10, self.twiddle2.re);
                            let m0209a = vfmaq_n_f32(m0209a, x2p9, self.twiddle4.re);
                            let m0209a = vfmaq_n_f32(m0209a, x3p8, self.twiddle5.re);
                            let m0209a = vfmaq_n_f32(m0209a, x4p7, self.twiddle3.re);
                            let m0209a = vfmaq_n_f32(m0209a, x5p6, self.twiddle1.re);
                            let m0209b = vmulq_n_f32(x1m10, self.twiddle2.im);
                            let m0209b = vfmaq_n_f32(m0209b, x2m9, self.twiddle4.im);
                            let m0209b = vfmsq_n_f32(m0209b, x3m8, self.twiddle5.im);
                            let m0209b = vfmsq_n_f32(m0209b, x4m7, self.twiddle3.im);
                            let m0209b = vfmsq_n_f32(m0209b, x5m6, self.twiddle1.im);
                            let (y02, y09) = NeonButterfly::butterfly2_f32(m0209a, m0209b);

                            let m0308a = vfmaq_n_f32(u0, x1p10, self.twiddle3.re);
                            let m0308a = vfmaq_n_f32(m0308a, x2p9, self.twiddle5.re);
                            let m0308a = vfmaq_n_f32(m0308a, x3p8, self.twiddle2.re);
                            let m0308a = vfmaq_n_f32(m0308a, x4p7, self.twiddle1.re);
                            let m0308a = vfmaq_n_f32(m0308a, x5p6, self.twiddle4.re);
                            let m0308b = vmulq_n_f32(x1m10, self.twiddle3.im);
                            let m0308b = vfmsq_n_f32(m0308b, x2m9, self.twiddle5.im);
                            let m0308b = vfmsq_n_f32(m0308b, x3m8, self.twiddle2.im);
                            let m0308b = vfmaq_n_f32(m0308b, x4m7, self.twiddle1.im);
                            let m0308b = vfmaq_n_f32(m0308b, x5m6, self.twiddle4.im);
                            let (y03, y08) = NeonButterfly::butterfly2_f32(m0308a, m0308b);

                            let m0407a = vfmaq_n_f32(u0, x1p10, self.twiddle4.re);
                            let m0407a = vfmaq_n_f32(m0407a, x2p9, self.twiddle3.re);
                            let m0407a = vfmaq_n_f32(m0407a, x3p8, self.twiddle1.re);
                            let m0407a = vfmaq_n_f32(m0407a, x4p7, self.twiddle5.re);
                            let m0407a = vfmaq_n_f32(m0407a, x5p6, self.twiddle2.re);
                            let m0407b = vmulq_n_f32(x1m10, self.twiddle4.im);
                            let m0407b = vfmsq_n_f32(m0407b, x2m9, self.twiddle3.im);
                            let m0407b = vfmaq_n_f32(m0407b, x3m8, self.twiddle1.im);
                            let m0407b = vfmaq_n_f32(m0407b, x4m7, self.twiddle5.im);
                            let m0407b = vfmsq_n_f32(m0407b, x5m6, self.twiddle2.im);
                            let (y04, y07) = NeonButterfly::butterfly2_f32(m0407a, m0407b);

                            let m0506a = vfmaq_n_f32(u0, x1p10, self.twiddle5.re);
                            let m0506a = vfmaq_n_f32(m0506a, x2p9, self.twiddle1.re);
                            let m0506a = vfmaq_n_f32(m0506a, x3p8, self.twiddle4.re);
                            let m0506a = vfmaq_n_f32(m0506a, x4p7, self.twiddle2.re);
                            let m0506a = vfmaq_n_f32(m0506a, x5p6, self.twiddle3.re);
                            let m0506b = vmulq_n_f32(x1m10, self.twiddle5.im);
                            let m0506b = vfmsq_n_f32(m0506b, x2m9, self.twiddle1.im);
                            let m0506b = vfmaq_n_f32(m0506b, x3m8, self.twiddle4.im);
                            let m0506b = vfmsq_n_f32(m0506b, x4m7, self.twiddle2.im);
                            let m0506b = vfmaq_n_f32(m0506b, x5m6, self.twiddle3.im);
                            let (y05, y06) = NeonButterfly::butterfly2_f32(m0506a, m0506b);

                            // Store results
                            vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            vst1q_f32(
                                data.get_unchecked_mut(j + eleventh..).as_mut_ptr().cast(),
                                y01,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 3 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 4 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 5 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 6 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 7 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y07,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 8 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y08,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 9 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y09,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 10 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );

                            j += 2;
                        }

                        for j in j..eleventh {
                            let u0 = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                            let w0w1 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j..).as_ptr().cast());
                            let w2w3 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 2..).as_ptr().cast());
                            let w4w5 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 4..).as_ptr().cast());
                            let w6w7 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 6..).as_ptr().cast());
                            let w8w9 =
                                vld1q_f32(m_twiddles.get_unchecked(10 * j + 8..).as_ptr().cast());

                            let u1u2 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + eleventh..).as_ptr().cast()),
                                    vld1_f32(
                                        data.get_unchecked(j + 2 * eleventh..).as_ptr().cast(),
                                    ),
                                ),
                                w0w1,
                            );
                            let u3u4 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(
                                        data.get_unchecked(j + 3 * eleventh..).as_ptr().cast(),
                                    ),
                                    vld1_f32(
                                        data.get_unchecked(j + 4 * eleventh..).as_ptr().cast(),
                                    ),
                                ),
                                w2w3,
                            );
                            let u5u6 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(
                                        data.get_unchecked(j + 5 * eleventh..).as_ptr().cast(),
                                    ),
                                    vld1_f32(
                                        data.get_unchecked(j + 6 * eleventh..).as_ptr().cast(),
                                    ),
                                ),
                                w4w5,
                            );
                            let u7u8 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(
                                        data.get_unchecked(j + 7 * eleventh..).as_ptr().cast(),
                                    ),
                                    vld1_f32(
                                        data.get_unchecked(j + 8 * eleventh..).as_ptr().cast(),
                                    ),
                                ),
                                w6w7,
                            );
                            let u9u10 = vfcmulq_fcma_f32(
                                vcombine_f32(
                                    vld1_f32(
                                        data.get_unchecked(j + 9 * eleventh..).as_ptr().cast(),
                                    ),
                                    vld1_f32(
                                        data.get_unchecked(j + 10 * eleventh..).as_ptr().cast(),
                                    ),
                                ),
                                w8w9,
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

                            // Radix-11 butterfly

                            let y00 = u0;
                            let (x1p10, x1m10) = NeonButterfly::butterfly2h_f32(u1, u10);
                            let x1m10 = vcadd_rot90_f32(vdup_n_f32(0.), x1m10);
                            let y00 = vadd_f32(y00, x1p10);
                            let (x2p9, x2m9) = NeonButterfly::butterfly2h_f32(u2, u9);
                            let x2m9 = vcadd_rot90_f32(vdup_n_f32(0.), x2m9);
                            let y00 = vadd_f32(y00, x2p9);
                            let (x3p8, x3m8) = NeonButterfly::butterfly2h_f32(u3, u8);
                            let x3m8 = vcadd_rot90_f32(vdup_n_f32(0.), x3m8);
                            let y00 = vadd_f32(y00, x3p8);
                            let (x4p7, x4m7) = NeonButterfly::butterfly2h_f32(u4, u7);
                            let x4m7 = vcadd_rot90_f32(vdup_n_f32(0.), x4m7);
                            let y00 = vadd_f32(y00, x4p7);
                            let (x5p6, x5m6) = NeonButterfly::butterfly2h_f32(u5, u6);
                            let x5m6 = vcadd_rot90_f32(vdup_n_f32(0.), x5m6);
                            let y00 = vadd_f32(y00, x5p6);

                            let m0110a = vfma_n_f32(u0, x1p10, self.twiddle1.re);
                            let m0110a = vfma_n_f32(m0110a, x2p9, self.twiddle2.re);
                            let m0110a = vfma_n_f32(m0110a, x3p8, self.twiddle3.re);
                            let m0110a = vfma_n_f32(m0110a, x4p7, self.twiddle4.re);
                            let m0110a = vfma_n_f32(m0110a, x5p6, self.twiddle5.re);
                            let m0110b = vmul_n_f32(x1m10, self.twiddle1.im);
                            let m0110b = vfma_n_f32(m0110b, x2m9, self.twiddle2.im);
                            let m0110b = vfma_n_f32(m0110b, x3m8, self.twiddle3.im);
                            let m0110b = vfma_n_f32(m0110b, x4m7, self.twiddle4.im);
                            let m0110b = vfma_n_f32(m0110b, x5m6, self.twiddle5.im);
                            let (y01, y10) = NeonButterfly::butterfly2h_f32(m0110a, m0110b);

                            let m0209a = vfma_n_f32(u0, x1p10, self.twiddle2.re);
                            let m0209a = vfma_n_f32(m0209a, x2p9, self.twiddle4.re);
                            let m0209a = vfma_n_f32(m0209a, x3p8, self.twiddle5.re);
                            let m0209a = vfma_n_f32(m0209a, x4p7, self.twiddle3.re);
                            let m0209a = vfma_n_f32(m0209a, x5p6, self.twiddle1.re);
                            let m0209b = vmul_n_f32(x1m10, self.twiddle2.im);
                            let m0209b = vfma_n_f32(m0209b, x2m9, self.twiddle4.im);
                            let m0209b = vfms_n_f32(m0209b, x3m8, self.twiddle5.im);
                            let m0209b = vfms_n_f32(m0209b, x4m7, self.twiddle3.im);
                            let m0209b = vfms_n_f32(m0209b, x5m6, self.twiddle1.im);
                            let (y02, y09) = NeonButterfly::butterfly2h_f32(m0209a, m0209b);

                            let m0308a = vfma_n_f32(u0, x1p10, self.twiddle3.re);
                            let m0308a = vfma_n_f32(m0308a, x2p9, self.twiddle5.re);
                            let m0308a = vfma_n_f32(m0308a, x3p8, self.twiddle2.re);
                            let m0308a = vfma_n_f32(m0308a, x4p7, self.twiddle1.re);
                            let m0308a = vfma_n_f32(m0308a, x5p6, self.twiddle4.re);
                            let m0308b = vmul_n_f32(x1m10, self.twiddle3.im);
                            let m0308b = vfms_n_f32(m0308b, x2m9, self.twiddle5.im);
                            let m0308b = vfms_n_f32(m0308b, x3m8, self.twiddle2.im);
                            let m0308b = vfma_n_f32(m0308b, x4m7, self.twiddle1.im);
                            let m0308b = vfma_n_f32(m0308b, x5m6, self.twiddle4.im);
                            let (y03, y08) = NeonButterfly::butterfly2h_f32(m0308a, m0308b);

                            let m0407a = vfma_n_f32(u0, x1p10, self.twiddle4.re);
                            let m0407a = vfma_n_f32(m0407a, x2p9, self.twiddle3.re);
                            let m0407a = vfma_n_f32(m0407a, x3p8, self.twiddle1.re);
                            let m0407a = vfma_n_f32(m0407a, x4p7, self.twiddle5.re);
                            let m0407a = vfma_n_f32(m0407a, x5p6, self.twiddle2.re);
                            let m0407b = vmul_n_f32(x1m10, self.twiddle4.im);
                            let m0407b = vfms_n_f32(m0407b, x2m9, self.twiddle3.im);
                            let m0407b = vfma_n_f32(m0407b, x3m8, self.twiddle1.im);
                            let m0407b = vfma_n_f32(m0407b, x4m7, self.twiddle5.im);
                            let m0407b = vfms_n_f32(m0407b, x5m6, self.twiddle2.im);
                            let (y04, y07) = NeonButterfly::butterfly2h_f32(m0407a, m0407b);

                            let m0506a = vfma_n_f32(u0, x1p10, self.twiddle5.re);
                            let m0506a = vfma_n_f32(m0506a, x2p9, self.twiddle1.re);
                            let m0506a = vfma_n_f32(m0506a, x3p8, self.twiddle4.re);
                            let m0506a = vfma_n_f32(m0506a, x4p7, self.twiddle2.re);
                            let m0506a = vfma_n_f32(m0506a, x5p6, self.twiddle3.re);
                            let m0506b = vmul_n_f32(x1m10, self.twiddle5.im);
                            let m0506b = vfms_n_f32(m0506b, x2m9, self.twiddle1.im);
                            let m0506b = vfma_n_f32(m0506b, x3m8, self.twiddle4.im);
                            let m0506b = vfms_n_f32(m0506b, x4m7, self.twiddle2.im);
                            let m0506b = vfma_n_f32(m0506b, x5m6, self.twiddle3.im);
                            let (y05, y06) = NeonButterfly::butterfly2h_f32(m0506a, m0506b);

                            // Store results
                            vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            vst1_f32(
                                data.get_unchecked_mut(j + eleventh..).as_mut_ptr().cast(),
                                y01,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 2 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 3 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 4 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 5 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 6 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 7 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y07,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 8 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y08,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 9 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y09,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 10 * eleventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 10..];
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neon::test_fcma_radix;

    test_fcma_radix!(test_neon_radix11, f32, NeonFcmaRadix11, 3, 11, 1e-2);
    test_fcma_radix!(test_neon_radix11_f64, f64, NeonFcmaRadix11, 3, 11, 1e-8);
}
