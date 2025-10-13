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
use crate::neon::butterflies::{NeonButterfly, NeonFastButterfly5};
use crate::neon::util::{fcma_complex_f32, fcma_complex_f64, fcmah_complex_f32};
use crate::radix10::Radix10Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{bitreversed_transpose, is_power_of_ten};
use crate::{FftDirection, FftExecutor, Zaft, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::aarch64::*;
use std::fmt::Display;

pub(crate) struct NeonFcmaRadix10<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    bf5: NeonFastButterfly5<T>,
    direction: FftDirection,
    butterfly: Box<dyn FftExecutor<T> + Send + Sync>,
}

impl<
    T: Default
        + Clone
        + Radix10Twiddles
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
> NeonFcmaRadix10<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonFcmaRadix10<T>, ZaftError> {
        assert!(
            is_power_of_ten(size as u64),
            "Input length must be a power of 10"
        );

        let twiddles = T::make_twiddles_with_base(10, size, fft_direction)?;

        Ok(NeonFcmaRadix10 {
            execution_length: size,
            twiddles,
            bf5: NeonFastButterfly5::new(fft_direction),
            direction: fft_direction,
            butterfly: Zaft::strategy(10, fft_direction)?,
        })
    }
}

impl FftExecutor<f64> for NeonFcmaRadix10<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }
}

impl NeonFcmaRadix10<f64> {
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
                bitreversed_transpose::<Complex<f64>, 10>(10, chunk, &mut scratch);

                self.butterfly.execute(&mut scratch)?;

                let mut len = 10;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 10;
                    let tenth = len / 10;

                    for data in scratch.chunks_exact_mut(len) {
                        for j in 0..tenth {
                            let td = 9 * j;
                            let tw0 = vld1q_f64(m_twiddles.get_unchecked(td..).as_ptr().cast());
                            let tw1 = vld1q_f64(m_twiddles.get_unchecked(td + 1..).as_ptr().cast());
                            let tw2 = vld1q_f64(m_twiddles.get_unchecked(td + 2..).as_ptr().cast());
                            let tw3 = vld1q_f64(m_twiddles.get_unchecked(td + 3..).as_ptr().cast());
                            let tw4 = vld1q_f64(m_twiddles.get_unchecked(td + 4..).as_ptr().cast());
                            let tw5 = vld1q_f64(m_twiddles.get_unchecked(td + 5..).as_ptr().cast());
                            let tw6 = vld1q_f64(m_twiddles.get_unchecked(td + 6..).as_ptr().cast());
                            let tw7 = vld1q_f64(m_twiddles.get_unchecked(td + 7..).as_ptr().cast());
                            let tw8 = vld1q_f64(m_twiddles.get_unchecked(td + 8..).as_ptr().cast());

                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + tenth..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * tenth..).as_ptr().cast()),
                                tw1,
                            );
                            let u3 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 3 * tenth..).as_ptr().cast()),
                                tw2,
                            );
                            let u4 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 4 * tenth..).as_ptr().cast()),
                                tw3,
                            );
                            let u5 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 5 * tenth..).as_ptr().cast()),
                                tw4,
                            );
                            let u6 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 6 * tenth..).as_ptr().cast()),
                                tw5,
                            );
                            let u7 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 7 * tenth..).as_ptr().cast()),
                                tw6,
                            );
                            let u8 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 8 * tenth..).as_ptr().cast()),
                                tw7,
                            );
                            let u9 = fcma_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 9 * tenth..).as_ptr().cast()),
                                tw8,
                            );

                            // Radix-10 butterfly

                            let mid0 = self.bf5.exec_fcma(u0, u2, u4, u6, u8);
                            let mid1 = self.bf5.exec_fcma(u5, u7, u9, u1, u3);

                            // Since this is good-thomas algorithm, we don't need twiddle factors
                            let (y0, y5) = NeonButterfly::butterfly2_f64(mid0.0, mid1.0);
                            let (y6, y1) = NeonButterfly::butterfly2_f64(mid0.1, mid1.1);
                            let (y2, y7) = NeonButterfly::butterfly2_f64(mid0.2, mid1.2);
                            let (y8, y3) = NeonButterfly::butterfly2_f64(mid0.3, mid1.3);
                            let (y4, y9) = NeonButterfly::butterfly2_f64(mid0.4, mid1.4);

                            // Store results
                            vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            vst1q_f64(data.get_unchecked_mut(j + tenth..).as_mut_ptr().cast(), y1);
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * tenth..).as_mut_ptr().cast(),
                                y2,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 3 * tenth..).as_mut_ptr().cast(),
                                y3,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 4 * tenth..).as_mut_ptr().cast(),
                                y4,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 5 * tenth..).as_mut_ptr().cast(),
                                y5,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 6 * tenth..).as_mut_ptr().cast(),
                                y6,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 7 * tenth..).as_mut_ptr().cast(),
                                y7,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 8 * tenth..).as_mut_ptr().cast(),
                                y8,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 9 * tenth..).as_mut_ptr().cast(),
                                y9,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 9..];
                }

                chunk.copy_from_slice(&scratch);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaRadix10<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }
}

impl NeonFcmaRadix10<f32> {
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
                bitreversed_transpose::<Complex<f32>, 10>(10, chunk, &mut scratch);

                self.butterfly.execute(&mut scratch)?;

                let mut len = 10;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 10;
                    let tenth = len / 10;

                    for data in scratch.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < tenth {
                            let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw = 9 * j;

                            let w0w1 = vld1q_f32(m_twiddles.get_unchecked(tw..).as_ptr().cast());
                            let w2w3 =
                                vld1q_f32(m_twiddles.get_unchecked(tw + 2..).as_ptr().cast());
                            let w4w5 =
                                vld1q_f32(m_twiddles.get_unchecked(tw + 4..).as_ptr().cast());
                            let w6w7 =
                                vld1q_f32(m_twiddles.get_unchecked(tw + 6..).as_ptr().cast());
                            let w8w9 =
                                vld1q_f32(m_twiddles.get_unchecked(tw + 8..).as_ptr().cast());
                            let w10w11 =
                                vld1q_f32(m_twiddles.get_unchecked(tw + 10..).as_ptr().cast());
                            let w12w13 =
                                vld1q_f32(m_twiddles.get_unchecked(tw + 12..).as_ptr().cast());
                            let w14w15 =
                                vld1q_f32(m_twiddles.get_unchecked(tw + 14..).as_ptr().cast());
                            let w16w17 =
                                vld1q_f32(m_twiddles.get_unchecked(tw + 16..).as_ptr().cast());

                            let ww0 = vcombine_f32(vget_low_f32(w0w1), vget_high_f32(w8w9));
                            let ww1 = vcombine_f32(vget_high_f32(w0w1), vget_low_f32(w10w11));
                            let ww2 = vcombine_f32(vget_low_f32(w2w3), vget_high_f32(w10w11));
                            let ww3 = vcombine_f32(vget_high_f32(w2w3), vget_low_f32(w12w13));
                            let ww4 = vcombine_f32(vget_low_f32(w4w5), vget_high_f32(w12w13));
                            let ww5 = vcombine_f32(vget_high_f32(w4w5), vget_low_f32(w14w15));
                            let ww6 = vcombine_f32(vget_low_f32(w6w7), vget_high_f32(w14w15));
                            let ww7 = vcombine_f32(vget_high_f32(w6w7), vget_low_f32(w16w17));
                            let ww8 = vcombine_f32(vget_low_f32(w8w9), vget_high_f32(w16w17));

                            let u1 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + tenth..).as_ptr().cast()),
                                ww0,
                            );
                            let u2 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * tenth..).as_ptr().cast()),
                                ww1,
                            );
                            let u3 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 3 * tenth..).as_ptr().cast()),
                                ww2,
                            );
                            let u4 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 4 * tenth..).as_ptr().cast()),
                                ww3,
                            );
                            let u5 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 5 * tenth..).as_ptr().cast()),
                                ww4,
                            );
                            let u6 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 6 * tenth..).as_ptr().cast()),
                                ww5,
                            );
                            let u7 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 7 * tenth..).as_ptr().cast()),
                                ww6,
                            );
                            let u8 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 8 * tenth..).as_ptr().cast()),
                                ww7,
                            );
                            let u9 = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 9 * tenth..).as_ptr().cast()),
                                ww8,
                            );

                            // Radix-10 butterfly

                            let mid0 = self.bf5.exec_fcma(u0, u2, u4, u6, u8);
                            let mid1 = self.bf5.exec_fcma(u5, u7, u9, u1, u3);

                            // Since this is good-thomas algorithm, we don't need twiddle factors
                            let (y0, y5) = NeonButterfly::butterfly2_f32(mid0.0, mid1.0);
                            let (y6, y1) = NeonButterfly::butterfly2_f32(mid0.1, mid1.1);
                            let (y2, y7) = NeonButterfly::butterfly2_f32(mid0.2, mid1.2);
                            let (y8, y3) = NeonButterfly::butterfly2_f32(mid0.3, mid1.3);
                            let (y4, y9) = NeonButterfly::butterfly2_f32(mid0.4, mid1.4);

                            // Store results
                            vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            vst1q_f32(data.get_unchecked_mut(j + tenth..).as_mut_ptr().cast(), y1);
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * tenth..).as_mut_ptr().cast(),
                                y2,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 3 * tenth..).as_mut_ptr().cast(),
                                y3,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 4 * tenth..).as_mut_ptr().cast(),
                                y4,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 5 * tenth..).as_mut_ptr().cast(),
                                y5,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 6 * tenth..).as_mut_ptr().cast(),
                                y6,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 7 * tenth..).as_mut_ptr().cast(),
                                y7,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 8 * tenth..).as_mut_ptr().cast(),
                                y8,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 9 * tenth..).as_mut_ptr().cast(),
                                y9,
                            );

                            j += 2;
                        }

                        for j in j..tenth {
                            let u0 = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                            let ti = 9 * j;

                            let w0w1 = vld1q_f32(m_twiddles.get_unchecked(ti..).as_ptr().cast());
                            let w2w3 =
                                vld1q_f32(m_twiddles.get_unchecked(ti + 2..).as_ptr().cast());
                            let w4w5 =
                                vld1q_f32(m_twiddles.get_unchecked(ti + 4..).as_ptr().cast());
                            let w6w7 =
                                vld1q_f32(m_twiddles.get_unchecked(ti + 6..).as_ptr().cast());
                            let w8 = vld1_f32(m_twiddles.get_unchecked(ti + 8..).as_ptr().cast());

                            let u1u2 = fcma_complex_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + tenth..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 2 * tenth..).as_ptr().cast()),
                                ),
                                w0w1,
                            );
                            let u3u4 = fcma_complex_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + 3 * tenth..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 4 * tenth..).as_ptr().cast()),
                                ),
                                w2w3,
                            );
                            let u5u6 = fcma_complex_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + 5 * tenth..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 6 * tenth..).as_ptr().cast()),
                                ),
                                w4w5,
                            );
                            let u7u8 = fcma_complex_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + 7 * tenth..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 8 * tenth..).as_ptr().cast()),
                                ),
                                w6w7,
                            );
                            let u9 = fcmah_complex_f32(
                                vld1_f32(data.get_unchecked(j + 9 * tenth..).as_ptr().cast()),
                                w8,
                            );

                            let u1 = vget_low_f32(u1u2);
                            let u2 = vget_high_f32(u1u2);
                            let u3 = vget_low_f32(u3u4);
                            let u4 = vget_high_f32(u3u4);
                            let u5 = vget_low_f32(u5u6);
                            let u6 = vget_high_f32(u5u6);
                            let u7 = vget_low_f32(u7u8);
                            let u8 = vget_high_f32(u7u8);

                            // Radix-10 butterfly

                            let mid0 = self.bf5.exec_fcma(
                                vcombine_f32(u0, u5),
                                vcombine_f32(u2, u7),
                                vcombine_f32(u4, u9),
                                vcombine_f32(u6, u1),
                                vcombine_f32(u8, u3),
                            );

                            // Since this is good-thomas algorithm, we don't need twiddle factors
                            let (y0, y5) = NeonButterfly::butterfly2h_f32(
                                vget_low_f32(mid0.0),
                                vget_high_f32(mid0.0),
                            );
                            let (y6, y1) = NeonButterfly::butterfly2h_f32(
                                vget_low_f32(mid0.1),
                                vget_high_f32(mid0.1),
                            );
                            let (y2, y7) = NeonButterfly::butterfly2h_f32(
                                vget_low_f32(mid0.2),
                                vget_high_f32(mid0.2),
                            );
                            let (y8, y3) = NeonButterfly::butterfly2h_f32(
                                vget_low_f32(mid0.3),
                                vget_high_f32(mid0.3),
                            );
                            let (y4, y9) = NeonButterfly::butterfly2h_f32(
                                vget_low_f32(mid0.4),
                                vget_high_f32(mid0.4),
                            );

                            // Store results
                            vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            vst1_f32(data.get_unchecked_mut(j + tenth..).as_mut_ptr().cast(), y1);
                            vst1_f32(
                                data.get_unchecked_mut(j + 2 * tenth..).as_mut_ptr().cast(),
                                y2,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 3 * tenth..).as_mut_ptr().cast(),
                                y3,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 4 * tenth..).as_mut_ptr().cast(),
                                y4,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 5 * tenth..).as_mut_ptr().cast(),
                                y5,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 6 * tenth..).as_mut_ptr().cast(),
                                y6,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 7 * tenth..).as_mut_ptr().cast(),
                                y7,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 8 * tenth..).as_mut_ptr().cast(),
                                y8,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 9 * tenth..).as_mut_ptr().cast(),
                                y9,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 9..];
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
    fn test_neon_radix10() {
        for i in 1..5 {
            let size = 10usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaRadix10::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix10::new(size, FftDirection::Inverse).unwrap();
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
    fn test_neon_radix10_f64() {
        for i in 1..5 {
            let size = 10usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaRadix10::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix10::new(size, FftDirection::Inverse).unwrap();
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
