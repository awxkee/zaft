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
use crate::neon::util::{mul_complex_f32, mul_complex_f64, mulh_complex_f32};
use crate::radix2::Radix2Twiddles;
use crate::util::permute_inplace;
use crate::{FftDirection, FftExecutor, ZaftError, bit_reverse_indices};
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) struct NeonRadix2<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
}

impl<T: Default + Clone + Radix2Twiddles> NeonRadix2<T> {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonRadix2<T>, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");

        let twiddles = T::make_twiddles(size, fft_direction)?;

        // Bit-reversal permutation
        let rev = bit_reverse_indices(size);

        Ok(NeonRadix2 {
            permutations: rev,
            execution_length: size,
            twiddles,
        })
    }
}

impl FftExecutor<f64> for NeonRadix2<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        permute_inplace(in_place, &self.permutations);

        let mut len = 2;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();
            while len <= self.execution_length {
                let half = len / 2;
                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;
                    while j + 2 < half {
                        let u01 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                        let u02 = vld1q_f64(data.get_unchecked(j + 1..).as_ptr().cast());
                        let tw0 = vld1q_f64(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let tw1 = vld1q_f64(m_twiddles.get_unchecked(j + 1..).as_ptr().cast());
                        let u11 = vld1q_f64(data.get_unchecked(j + half..).as_ptr().cast());
                        let u12 = vld1q_f64(data.get_unchecked(j + half + 1..).as_ptr().cast());
                        let t0 = mul_complex_f64(tw0, u11);
                        let t1 = mul_complex_f64(tw1, u12);
                        let y0_1 = vaddq_f64(u01, t0);
                        let y1_1 = vsubq_f64(u01, t0);
                        let y0_2 = vaddq_f64(u02, t1);
                        let y1_2 = vsubq_f64(u02, t1);
                        vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0_1);
                        vst1q_f64(data.get_unchecked_mut(j + 1..).as_mut_ptr().cast(), y0_2);
                        vst1q_f64(data.get_unchecked_mut(j + half..).as_mut_ptr().cast(), y1_1);
                        vst1q_f64(
                            data.get_unchecked_mut(j + half + 1..).as_mut_ptr().cast(),
                            y1_2,
                        );
                        j += 2;
                    }
                    for j in j..half {
                        let u = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                        let tw = vld1q_f64(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let u1 = vld1q_f64(data.get_unchecked(j + half..).as_ptr().cast());
                        let t = mul_complex_f64(tw, u1);
                        let y0 = vaddq_f64(u, t);
                        let y1 = vsubq_f64(u, t);
                        vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        vst1q_f64(data.get_unchecked_mut(j + half..).as_mut_ptr().cast(), y1);
                    }
                }

                len *= 2;
                m_twiddles = &m_twiddles[half..];
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonRadix2<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        permute_inplace(in_place, &self.permutations);

        let mut len = 2;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            while len <= self.execution_length {
                let half = len / 2;
                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 4 < half {
                        let u0_0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());
                        let u1_0 = vld1q_f32(data.get_unchecked(j + half..).as_ptr().cast());

                        let u0_1 = vld1q_f32(data.get_unchecked(j + 2..).as_ptr().cast());
                        let u1_1 = vld1q_f32(data.get_unchecked(j + half + 2..).as_ptr().cast());

                        let tw0 = vld1q_f32(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let tw1 = vld1q_f32(m_twiddles.get_unchecked(j + 2..).as_ptr().cast());

                        let t_0 = mul_complex_f32(tw0, u1_0);
                        let t_1 = mul_complex_f32(tw1, u1_1);

                        let y0 = vaddq_f32(u0_0, t_0);
                        let y1 = vsubq_f32(u0_0, t_0);
                        let y2 = vaddq_f32(u0_1, t_1);
                        let y3 = vsubq_f32(u0_1, t_1);

                        vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        vst1q_f32(data.get_unchecked_mut(j + half..).as_mut_ptr().cast(), y1);
                        vst1q_f32(data.get_unchecked_mut(j + 2..).as_mut_ptr().cast(), y2);
                        vst1q_f32(
                            data.get_unchecked_mut(j + half + 2..).as_mut_ptr().cast(),
                            y3,
                        );

                        j += 4;
                    }

                    while j + 2 < half {
                        let u = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());
                        let tw = vld1q_f32(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let u1 = vld1q_f32(data.get_unchecked(j + half..).as_ptr().cast());
                        let t = mul_complex_f32(tw, u1);
                        let y0 = vaddq_f32(u, t);
                        let y1 = vsubq_f32(u, t);
                        vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        vst1q_f32(data.get_unchecked_mut(j + half..).as_mut_ptr().cast(), y1);

                        j += 2;
                    }

                    for j in j..half {
                        let u = vld1_f32(data.get_unchecked(j..).as_ptr().cast());
                        let tw = vld1_f32(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let u1 = vld1_f32(data.get_unchecked(j + half..).as_ptr().cast());
                        let t = mulh_complex_f32(tw, u1);
                        let y0 = vadd_f32(u, t);
                        let y1 = vsub_f32(u, t);
                        vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        vst1_f32(data.get_unchecked_mut(j + half..).as_mut_ptr().cast(), y1);
                    }
                }

                len *= 2;
                m_twiddles = &m_twiddles[half..];
            }
        }
        Ok(())
    }
}
