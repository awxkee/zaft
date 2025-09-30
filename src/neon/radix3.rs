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
use crate::radix3::Radix3Twiddles;
use crate::traits::FftTrigonometry;
use crate::util::{compute_twiddle, digit_reverse_indices, permute_inplace};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonRadix3<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    twiddle_re: T,
    twiddle_im: [T; 4],
}

impl<T: Default + Clone + Radix3Twiddles + 'static + Copy + FftTrigonometry + Float> NeonRadix3<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonRadix3<T>, ZaftError> {
        assert!(
            size.is_power_of_two() || size % 3 == 0,
            "Input length must be divisible by 3"
        );

        let twiddles = T::make_twiddles(size, fft_direction)?;
        let rev = digit_reverse_indices(size, 3)?;

        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);

        Ok(NeonRadix3 {
            permutations: rev,
            execution_length: size,
            twiddles,
            twiddle_re: twiddle.re,
            twiddle_im: [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im],
        })
    }
}

impl FftExecutor<f64> for NeonRadix3<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        // Trit-reversal permutation
        permute_inplace(in_place, &self.permutations);

        let mut len = 3;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            let twiddle_re = vdupq_n_f64(self.twiddle_re);
            let twiddle_w_2 = vld1q_f64(self.twiddle_im.as_ptr().cast());

            while len <= self.execution_length {
                let third = len / 3;
                for data in in_place.chunks_exact_mut(len) {
                    for j in 0..third {
                        let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                        let u1 = mul_complex_f64(
                            vld1q_f64(data.get_unchecked(j + third..).as_ptr().cast()),
                            vld1q_f64(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                        );
                        let u2 = mul_complex_f64(
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
        Ok(())
    }
}

impl FftExecutor<f32> for NeonRadix3<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        // Trit-reversal permutation
        permute_inplace(in_place, &self.permutations);

        let mut len = 3;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            let twiddle_re = vdupq_n_f32(self.twiddle_re);
            let twiddle_w_2 = vld1q_f32(self.twiddle_im.as_ptr().cast());

            while len <= self.execution_length {
                let third = len / 3;
                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 3 < third {
                        let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());
                        let u0_1 = vld1_f32(data.get_unchecked(j + 2..).as_ptr().cast());
                        let u1 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + third..).as_ptr().cast()),
                            vld1q_f32(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                        );
                        let u1_1 = mulh_complex_f32(
                            vld1_f32(data.get_unchecked(j + third + 2..).as_ptr().cast()),
                            vld1_f32(m_twiddles.get_unchecked(2 * j + 2..).as_ptr().cast()),
                        );
                        let u2 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                            vld1q_f32(m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast()),
                        );
                        let u2_1 = mulh_complex_f32(
                            vld1_f32(data.get_unchecked(j + 2 * third + 2..).as_ptr().cast()),
                            vld1_f32(m_twiddles.get_unchecked(2 * j + 1 + 2..).as_ptr().cast()),
                        );

                        // Radix-3 butterfly
                        let xp_0 = vaddq_f32(u1, u2);
                        let xn_0 = vsubq_f32(u1, u2);
                        let sum_0 = vaddq_f32(u0, xp_0);

                        let xp_1 = vadd_f32(u1_1, u2_1);
                        let xn_1 = vsub_f32(u1_1, u2_1);
                        let sum_1 = vadd_f32(u0_1, xp_1);

                        let vw_1_1 = vfmaq_f32(u0, twiddle_re, xp_0);
                        let vw_2_1 = vmulq_f32(twiddle_w_2, vrev64q_f32(xn_0));

                        let vw_1_2 = vfma_f32(u0_1, vget_low_f32(twiddle_re), xp_1);
                        let vw_2_2 = vmul_f32(vget_low_f32(twiddle_w_2), vext_f32::<1>(xn_1, xn_1));

                        let vy0 = sum_0;
                        let vy1 = vaddq_f32(vw_1_1, vw_2_1);
                        let vy2 = vsubq_f32(vw_1_1, vw_2_1);

                        let vy0_1 = sum_1;
                        let vy1_1 = vadd_f32(vw_1_2, vw_2_2);
                        let vy2_1 = vsub_f32(vw_1_2, vw_2_2);

                        vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                        vst1_f32(data.get_unchecked_mut(j + 2..).as_mut_ptr().cast(), vy0_1);
                        vst1q_f32(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                        vst1_f32(
                            data.get_unchecked_mut(j + third + 2..).as_mut_ptr().cast(),
                            vy1_1,
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                            vy2,
                        );
                        vst1_f32(
                            data.get_unchecked_mut(j + 2 * third + 2..)
                                .as_mut_ptr()
                                .cast(),
                            vy2_1,
                        );

                        j += 3;
                    }

                    for j in j..third {
                        let u0 = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                        let tw = vld1q_f32(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());

                        let u1 = mulh_complex_f32(
                            vld1_f32(data.get_unchecked(j + third..).as_ptr().cast()),
                            vget_low_f32(tw),
                        );
                        let u2 = mulh_complex_f32(
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
        Ok(())
    }
}
