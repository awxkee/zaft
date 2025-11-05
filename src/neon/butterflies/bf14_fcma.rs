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
use crate::neon::butterflies::NeonButterfly;
use crate::neon::butterflies::fast_bf7::NeonFastButterfly7;
use crate::neon::util::vqtrnq_f32;
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaButterfly14<T> {
    direction: FftDirection,
    bf7: NeonFastButterfly7<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonFcmaButterfly14<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf7: NeonFastButterfly7::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly14<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        14
    }
}

impl NeonFcmaButterfly14<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 14 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(14) {
                let u0 = vld1q_f64(chunk.as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u5 = vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast());
                let u6 = vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast());
                let u7 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u8 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                let u9 = vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast());
                let u10 = vld1q_f64(chunk.get_unchecked(12..).as_ptr().cast());
                let u11 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                let u12 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                let u13 = vld1q_f64(chunk.get_unchecked(13..).as_ptr().cast());

                let (u0, u1) = NeonButterfly::butterfly2_f64(u0, u1);
                let (u2, u3) = NeonButterfly::butterfly2_f64(u2, u3);
                let (u4, u5) = NeonButterfly::butterfly2_f64(u4, u5);
                let (u6, u7) = NeonButterfly::butterfly2_f64(u6, u7);
                let (u8, u9) = NeonButterfly::butterfly2_f64(u8, u9);
                let (u10, u11) = NeonButterfly::butterfly2_f64(u10, u11);
                let (u12, u13) = NeonButterfly::butterfly2_f64(u12, u13);

                // Outer 7-point butterflies
                let (y0, y2, y4, y6, y8, y10, y12) =
                    self.bf7.exec_fcma(u0, u2, u4, u6, u8, u10, u12); // (v0, v1, v2, v3, v4, v5, v6)
                let (y7, y9, y11, y13, y1, y3, y5) =
                    self.bf7.exec_fcma(u1, u3, u5, u7, u9, u11, u13); // (v7, v8, v9, v10, v11, v12, v13)

                vst1q_f64(chunk.as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3);
                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y5);
                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y6);
                vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y7);
                vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y8);
                vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y9);
                vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);
                vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), y11);
                vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y12);
                vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), y13);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaButterfly14<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        14
    }
}

impl NeonFcmaButterfly14<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 14 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(28) {
                let u0u3 = vld1q_f32(chunk.as_mut_ptr().cast());
                let u4u7 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u8u11 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u12u1 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u2u5 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u6u9 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u10u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());

                let u0u3_2 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());
                let u4u7_2 = vld1q_f32(chunk.get_unchecked(16..).as_ptr().cast());
                let u8u11_2 = vld1q_f32(chunk.get_unchecked(18..).as_ptr().cast());
                let u12u1_2 = vld1q_f32(chunk.get_unchecked(20..).as_ptr().cast());
                let u2u5_2 = vld1q_f32(chunk.get_unchecked(22..).as_ptr().cast());
                let u6u9_2 = vld1q_f32(chunk.get_unchecked(24..).as_ptr().cast());
                let u10u13_2 = vld1q_f32(chunk.get_unchecked(26..).as_ptr().cast());

                let (u0, u3) = vqtrnq_f32(u0u3, u0u3_2);
                let (u4, u7) = vqtrnq_f32(u4u7, u4u7_2);
                let (u8, u11) = vqtrnq_f32(u8u11, u8u11_2);
                let (u12, u1) = vqtrnq_f32(u12u1, u12u1_2);
                let (u2, u5) = vqtrnq_f32(u2u5, u2u5_2);
                let (u6, u9) = vqtrnq_f32(u6u9, u6u9_2);
                let (u10, u13) = vqtrnq_f32(u10u13, u10u13_2);

                let (u0, u1) = NeonButterfly::butterfly2_f32(u0, u1);
                let (u2, u3) = NeonButterfly::butterfly2_f32(u2, u3);
                let (u4, u5) = NeonButterfly::butterfly2_f32(u4, u5);
                let (u6, u7) = NeonButterfly::butterfly2_f32(u6, u7);
                let (u8, u9) = NeonButterfly::butterfly2_f32(u8, u9);
                let (u10, u11) = NeonButterfly::butterfly2_f32(u10, u11);
                let (u12, u13) = NeonButterfly::butterfly2_f32(u12, u13);

                // Outer 7-point butterflies
                let (y0, y2, y4, y6, y8, y10, y12) =
                    self.bf7.exec_fcma(u0, u2, u4, u6, u8, u10, u12); // (v0, v1, v2, v3, v4, v5, v6)
                let (y7, y9, y11, y13, y1, y3, y5) =
                    self.bf7.exec_fcma(u1, u3, u5, u7, u9, u11, u13); // (v7, v8, v9, v10, v11, v12, v13)

                let (q0, q7) = vqtrnq_f32(y0, y1);
                let (q1, q8) = vqtrnq_f32(y2, y3);
                let (q2, q9) = vqtrnq_f32(y4, y5);
                let (q3, q10) = vqtrnq_f32(y6, y7);
                let (q4, q11) = vqtrnq_f32(y8, y9);
                let (q5, q12) = vqtrnq_f32(y10, y11);
                let (q6, q13) = vqtrnq_f32(y12, y13);

                vst1q_f32(chunk.as_mut_ptr().cast(), q0);
                vst1q_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), q1);
                vst1q_f32(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), q2);
                vst1q_f32(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), q3);
                vst1q_f32(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), q4);
                vst1q_f32(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), q5);
                vst1q_f32(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), q6);
                vst1q_f32(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), q7);
                vst1q_f32(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), q8);
                vst1q_f32(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), q9);
                vst1q_f32(chunk.get_unchecked_mut(20..).as_mut_ptr().cast(), q10);
                vst1q_f32(chunk.get_unchecked_mut(22..).as_mut_ptr().cast(), q11);
                vst1q_f32(chunk.get_unchecked_mut(24..).as_mut_ptr().cast(), q12);
                vst1q_f32(chunk.get_unchecked_mut(26..).as_mut_ptr().cast(), q13);
            }

            let rem = in_place.chunks_exact_mut(28).into_remainder();

            for chunk in rem.chunks_exact_mut(14) {
                let u0u3 = vld1q_f32(chunk.as_mut_ptr().cast());
                let u4u7 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u8u11 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u12u1 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u2u5 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u6u9 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u10u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());

                let u0 = vget_low_f32(u0u3);
                let u1 = vget_high_f32(u12u1);
                let u2 = vget_low_f32(u2u5);
                let u3 = vget_high_f32(u0u3);
                let u4 = vget_low_f32(u4u7);
                let u5 = vget_high_f32(u2u5);
                let u6 = vget_low_f32(u6u9);
                let u7 = vget_high_f32(u4u7);
                let u8 = vget_low_f32(u8u11);
                let u9 = vget_high_f32(u6u9);
                let u10 = vget_low_f32(u10u13);
                let u11 = vget_high_f32(u8u11);
                let u12 = vget_low_f32(u12u1);
                let u13 = vget_high_f32(u10u13);

                let (u0, u1) = NeonButterfly::butterfly2h_f32(u0, u1);
                let (u2, u3) = NeonButterfly::butterfly2h_f32(u2, u3);
                let (u4, u5) = NeonButterfly::butterfly2h_f32(u4, u5);
                let (u6, u7) = NeonButterfly::butterfly2h_f32(u6, u7);
                let (u8, u9) = NeonButterfly::butterfly2h_f32(u8, u9);
                let (u10, u11) = NeonButterfly::butterfly2h_f32(u10, u11);
                let (u12, u13) = NeonButterfly::butterfly2h_f32(u12, u13);

                // // Outer 7-point butterflies
                let (y0y7, y2y9, y4y11, y6y13, y8y1, y10y3, y12y5) = self.bf7.exec_fcma(
                    vcombine_f32(u0, u1),
                    vcombine_f32(u2, u3),
                    vcombine_f32(u4, u5),
                    vcombine_f32(u6, u7),
                    vcombine_f32(u8, u9),
                    vcombine_f32(u10, u11),
                    vcombine_f32(u12, u13),
                ); // (v0, v1, v2, v3, v4, v5, v6)

                vst1q_f32(
                    chunk.as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y0y7), vget_high_f32(y8y1)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y2y9), vget_high_f32(y10y3)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y4y11), vget_high_f32(y12y5)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y6y13), vget_high_f32(y0y7)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y8y1), vget_high_f32(y2y9)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y10y3), vget_high_f32(y4y11)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y12y5), vget_high_f32(y6y13)),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neon::butterflies::test_fcma_butterfly;
    use rand::Rng;

    test_fcma_butterfly!(test_fcma_butterfly14, f32, NeonFcmaButterfly14, 14, 1e-5);
    test_fcma_butterfly!(
        test_fcma_butterfly14_f64,
        f64,
        NeonFcmaButterfly14,
        14,
        1e-7
    );
}
