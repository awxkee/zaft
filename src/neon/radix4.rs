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
use crate::radix4::Radix4Twiddles;
use crate::util::{digit_reverse_indices, permute_inplace};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) struct NeonRadix4<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    direction: FftDirection,
}

impl<T: Default + Clone + Radix4Twiddles> NeonRadix4<T> {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonRadix4<T>, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");
        assert_eq!(size.trailing_zeros() % 2, 0, "Radix-4 requires power of 4");

        let twiddles = T::make_twiddles(size, fft_direction)?;
        let rev = digit_reverse_indices(size, 4)?;

        Ok(NeonRadix4 {
            permutations: rev,
            execution_length: size,
            twiddles,
            direction: fft_direction,
        })
    }
}

impl FftExecutor<f64> for NeonRadix4<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        // bit reversal first
        permute_inplace(in_place, &self.permutations);

        let mut len = 4;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            let v_i_multiplier = vreinterpretq_u64_f64(match self.direction {
                FftDirection::Forward => vld1q_f64([-0.0, 0.0].as_ptr()),
                FftDirection::Inverse => vld1q_f64([0.0, -0.0].as_ptr()),
            });

            while len <= self.execution_length {
                let quarter = len / 4;

                for data in in_place.chunks_exact_mut(len) {
                    for j in 0..quarter {
                        let a = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                        let b = mul_complex_f64(
                            vld1q_f64(data.get_unchecked(j + quarter..).as_ptr().cast()),
                            vld1q_f64(m_twiddles.get_unchecked(3 * j..).as_ptr().cast()),
                        );
                        let c = mul_complex_f64(
                            vld1q_f64(data.get_unchecked(j + 2 * quarter..).as_ptr().cast()),
                            vld1q_f64(m_twiddles.get_unchecked(3 * j + 1..).as_ptr().cast()),
                        );
                        let d = mul_complex_f64(
                            vld1q_f64(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                            vld1q_f64(m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast()),
                        );

                        // radix-4 butterfly
                        let t0 = vaddq_f64(a, c);
                        let t1 = vsubq_f64(a, c);
                        let t2 = vaddq_f64(b, d);
                        let mut t3 = vsubq_f64(b, d);
                        t3 = vreinterpretq_f64_u64(veorq_u64(
                            vreinterpretq_u64_f64(vcombine_f64(
                                vget_high_f64(t3),
                                vget_low_f64(t3),
                            )),
                            v_i_multiplier,
                        ));

                        vst1q_f64(
                            data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                            vaddq_f64(t0, t2),
                        );
                        vst1q_f64(
                            data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                            vaddq_f64(t1, t3),
                        );
                        vst1q_f64(
                            data.get_unchecked_mut(j + 2 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            vsubq_f64(t0, t2),
                        );
                        vst1q_f64(
                            data.get_unchecked_mut(j + 3 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            vsubq_f64(t1, t3),
                        );
                    }
                }

                m_twiddles = &m_twiddles[quarter * 3..];
                len *= 4;
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonRadix4<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        // bit reversal first
        permute_inplace(in_place, &self.permutations);

        let mut len = 4;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            let v_i_multiplier = vreinterpretq_u32_f32(match self.direction {
                FftDirection::Forward => vld1q_f32([-0.0, 0.0, -0.0, 0.0].as_ptr()),
                FftDirection::Inverse => vld1q_f32([0.0, -0.0, 0.0, -0.0].as_ptr()),
            });

            while len <= self.execution_length {
                let quarter = len / 4;

                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 4 < quarter {
                        let a0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());
                        let a1 = vld1q_f32(data.get_unchecked(j + 2..).as_ptr().cast());
                        let b0 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + quarter..).as_ptr().cast()),
                            vld1q_f32(m_twiddles.get_unchecked(3 * j..).as_ptr().cast()),
                        );
                        let b1 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + quarter + 2..).as_ptr().cast()),
                            vld1q_f32(m_twiddles.get_unchecked(3 * (j + 1)..).as_ptr().cast()),
                        );
                        let c0 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + 2 * quarter..).as_ptr().cast()),
                            vld1q_f32(m_twiddles.get_unchecked(3 * j + 1..).as_ptr().cast()),
                        );
                        let c1 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + 2 * quarter + 2..).as_ptr().cast()),
                            vld1q_f32(m_twiddles.get_unchecked(3 * (j + 1) + 1..).as_ptr().cast()),
                        );
                        let d0 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                            vld1q_f32(m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast()),
                        );
                        let d1 = mul_complex_f32(
                            vld1q_f32(data.get_unchecked(j + 3 * quarter + 2..).as_ptr().cast()),
                            vld1q_f32(m_twiddles.get_unchecked(3 * (j + 1) + 2..).as_ptr().cast()),
                        );

                        // radix-4 butterfly
                        let q0t0 = vaddq_f32(a0, c0);
                        let q0t1 = vsubq_f32(a0, c0);
                        let q0t2 = vaddq_f32(b0, d0);
                        let mut q0t3 = vsubq_f32(b0, d0);
                        q0t3 = vreinterpretq_f32_u32(veorq_u32(
                            vrev64q_u32(vreinterpretq_u32_f32(q0t3)),
                            v_i_multiplier,
                        ));

                        let q1t0 = vaddq_f32(a1, c1);
                        let q1t1 = vsubq_f32(a1, c1);
                        let q1t2 = vaddq_f32(b1, d1);
                        let mut q1t3 = vsubq_f32(b1, d1);
                        q1t3 = vreinterpretq_f32_u32(veorq_u32(
                            vrev64q_u32(vreinterpretq_u32_f32(q1t3)),
                            v_i_multiplier,
                        ));

                        vst1q_f32(
                            data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                            vaddq_f32(q0t0, q0t2),
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 2..).as_mut_ptr().cast(),
                            vaddq_f32(q1t0, q1t2),
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                            vaddq_f32(q0t1, q0t3),
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + quarter + 2..)
                                .as_mut_ptr()
                                .cast(),
                            vaddq_f32(q1t1, q1t3),
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 2 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            vsubq_f32(q0t0, q0t2),
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 2 * quarter + 2..)
                                .as_mut_ptr()
                                .cast(),
                            vsubq_f32(q1t0, q1t2),
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 3 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            vsubq_f32(q0t1, q0t3),
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 3 * quarter + 2..)
                                .as_mut_ptr()
                                .cast(),
                            vsubq_f32(q1t1, q1t3),
                        );

                        j += 4;
                    }

                    for j in j..quarter {
                        let a = vld1_f32(data.get_unchecked(j..).as_ptr().cast());
                        let b = mulh_complex_f32(
                            vld1_f32(data.get_unchecked(j + quarter..).as_ptr().cast()),
                            vld1_f32(m_twiddles.get_unchecked(3 * j..).as_ptr().cast()),
                        );
                        let c = mulh_complex_f32(
                            vld1_f32(data.get_unchecked(j + 2 * quarter..).as_ptr().cast()),
                            vld1_f32(m_twiddles.get_unchecked(3 * j + 1..).as_ptr().cast()),
                        );
                        let d = mulh_complex_f32(
                            vld1_f32(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                            vld1_f32(m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast()),
                        );

                        // radix-4 butterfly
                        let t0 = vadd_f32(a, c);
                        let t1 = vsub_f32(a, c);
                        let t2 = vadd_f32(b, d);
                        let mut t3 = vsub_f32(b, d);
                        t3 = vreinterpret_f32_u32(veor_u32(
                            vrev64_u32(vreinterpret_u32_f32(t3)),
                            vget_low_u32(v_i_multiplier),
                        ));

                        vst1_f32(
                            data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                            vadd_f32(t0, t2),
                        );
                        vst1_f32(
                            data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                            vadd_f32(t1, t3),
                        );
                        vst1_f32(
                            data.get_unchecked_mut(j + 2 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            vsub_f32(t0, t2),
                        );
                        vst1_f32(
                            data.get_unchecked_mut(j + 3 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            vsub_f32(t1, t3),
                        );
                    }
                }

                m_twiddles = &m_twiddles[quarter * 3..];
                len *= 4;
            }
        }
        Ok(())
    }
}
