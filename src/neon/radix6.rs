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
use crate::neon::butterflies::NeonButterfly;
use crate::neon::util::{mul_complex_f32, mul_complex_f64, mulh_complex_f32};
use crate::radix6::Radix6Twiddles;
use crate::traits::FftTrigonometry;
use crate::util::{compute_twiddle, digit_reverse_indices, is_power_of_six, permute_inplace};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonRadix6<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    twiddle_re: T,
    twiddle_im: [T; 4],
    direction: FftDirection,
}

impl<T: Default + Clone + Radix6Twiddles + 'static + Copy + FftTrigonometry + Float> NeonRadix6<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonRadix6<T>, ZaftError> {
        assert!(
            is_power_of_six(size as u64),
            "Input length must be a power of 6"
        );

        let twiddles = T::make_twiddles(size, fft_direction)?;
        let rev = digit_reverse_indices(size, 6)?;

        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);

        Ok(NeonRadix6 {
            permutations: rev,
            execution_length: size,
            twiddles,
            twiddle_re: twiddle.re,
            twiddle_im: [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im],
            direction: fft_direction,
        })
    }
}

impl FftExecutor<f64> for NeonRadix6<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let twiddle_re = unsafe { vdupq_n_f64(self.twiddle_re) };
        let twiddle_w_2 = unsafe { vld1q_f64(self.twiddle_im.as_ptr().cast()) };

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // Digit-reversal permutation
            permute_inplace(chunk, &self.permutations);

            let mut len = 6;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len <= self.execution_length {
                    let sixth = len / 6;

                    for data in chunk.chunks_exact_mut(len) {
                        for j in 0..sixth {
                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = mul_complex_f64(
                                vld1q_f64(data.get_unchecked(j + sixth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(5 * j..).as_ptr().cast()),
                            );
                            let u2 = mul_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * sixth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(5 * j + 1..).as_ptr().cast()),
                            );
                            let u3 = mul_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 3 * sixth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(5 * j + 2..).as_ptr().cast()),
                            );
                            let u4 = mul_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 4 * sixth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(5 * j + 3..).as_ptr().cast()),
                            );
                            let u5 = mul_complex_f64(
                                vld1q_f64(data.get_unchecked(j + 5 * sixth..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(5 * j + 4..).as_ptr().cast()),
                            );

                            let (t0, t2, t4) =
                                NeonButterfly::butterfly3_f64(u0, u2, u4, twiddle_re, twiddle_w_2);
                            let (t1, t3, t5) =
                                NeonButterfly::butterfly3_f64(u3, u5, u1, twiddle_re, twiddle_w_2);
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

                    m_twiddles = &m_twiddles[sixth * 5..];
                    len *= 6;
                }
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

impl FftExecutor<f32> for NeonRadix6<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let twiddle_re = unsafe { vdupq_n_f32(self.twiddle_re) };
        let twiddle_w_2 = unsafe { vld1q_f32(self.twiddle_im.as_ptr().cast()) };

        // Digit-reversal permutation
        permute_inplace(in_place, &self.permutations);

        let mut len = 6;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            while len <= self.execution_length {
                let sixth = len / 6;

                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 2 < sixth {
                        let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                        let tw0 = vld1q_f32(m_twiddles.get_unchecked(5 * j..).as_ptr().cast());
                        let tw1 =
                            vld1q_f32(m_twiddles.get_unchecked(5 * (j + 1)..).as_ptr().cast());
                        let tw2 = vld1q_f32(m_twiddles.get_unchecked(5 * j + 2..).as_ptr().cast());
                        let tw3 =
                            vld1q_f32(m_twiddles.get_unchecked(5 * (j + 1) + 2..).as_ptr().cast());

                        let u1 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + sixth..).as_ptr().cast()),
                            vcombine_f32(vget_low_f32(tw0), vget_low_f32(tw1)),
                        );
                        let u2 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + 2 * sixth..).as_ptr().cast()),
                            vcombine_f32(vget_high_f32(tw0), vget_high_f32(tw1)),
                        );
                        let u3 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + 3 * sixth..).as_ptr().cast()),
                            vcombine_f32(vget_low_f32(tw2), vget_low_f32(tw3)),
                        );
                        let u4 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + 4 * sixth..).as_ptr().cast()),
                            vcombine_f32(vget_high_f32(tw2), vget_high_f32(tw3)),
                        );
                        let u5 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + 5 * sixth..).as_ptr().cast()),
                            vcombine_f32(
                                vld1_f32(m_twiddles.get_unchecked(5 * j + 4..).as_ptr().cast()),
                                vld1_f32(
                                    m_twiddles.get_unchecked(5 * (j + 1) + 4..).as_ptr().cast(),
                                ),
                            ),
                        );

                        let (t0, t2, t4) =
                            NeonButterfly::butterfly3_f32(u0, u2, u4, twiddle_re, twiddle_w_2);
                        let (t1, t3, t5) =
                            NeonButterfly::butterfly3_f32(u3, u5, u1, twiddle_re, twiddle_w_2);
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
                        let tw1 = vld1q_f32(m_twiddles.get_unchecked(5 * j + 2..).as_ptr().cast());

                        let u1 = mulh_complex_f32(
                            vld1_f32(data.get_unchecked(j + sixth..).as_ptr().cast()),
                            vget_low_f32(tw0),
                        );
                        let u2 = mulh_complex_f32(
                            vld1_f32(data.get_unchecked(j + 2 * sixth..).as_ptr().cast()),
                            vget_high_f32(tw0),
                        );
                        let u3 = mulh_complex_f32(
                            vld1_f32(data.get_unchecked(j + 3 * sixth..).as_ptr().cast()),
                            vget_low_f32(tw1),
                        );
                        let u4 = mulh_complex_f32(
                            vld1_f32(data.get_unchecked(j + 4 * sixth..).as_ptr().cast()),
                            vget_high_f32(tw1),
                        );
                        let u5 = mulh_complex_f32(
                            vld1_f32(data.get_unchecked(j + 5 * sixth..).as_ptr().cast()),
                            vld1_f32(m_twiddles.get_unchecked(5 * j + 4..).as_ptr().cast()),
                        );

                        let (t0, t2, t4) = NeonButterfly::butterfly3h_f32(
                            u0,
                            u2,
                            u4,
                            vget_low_f32(twiddle_re),
                            vget_low_f32(twiddle_w_2),
                        );
                        let (t1, t3, t5) = NeonButterfly::butterfly3h_f32(
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

                m_twiddles = &m_twiddles[sixth * 5..];
                len *= 6;
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
    fn test_neon_radix6() {
        for i in 1..7 {
            let size = 6usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonRadix6::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonRadix6::new(size, FftDirection::Inverse).unwrap();
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
    fn test_neon_radix6_f64() {
        for i in 1..7 {
            let size = 6usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonRadix6::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonRadix6::new(size, FftDirection::Inverse).unwrap();
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
