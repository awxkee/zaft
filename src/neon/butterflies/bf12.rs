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
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonButterfly12<T> {
    direction: FftDirection,
    twiddle_re: T,
    twiddle_im: [T; 4],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonButterfly12<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);
        Self {
            direction: fft_direction,
            twiddle_re: twiddle.re,
            twiddle_im: [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im],
        }
    }
}

impl FftExecutor<f64> for NeonButterfly12<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 12 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            static INVERSE_ROT: [f64; 2] = [-0.0, 0.0];
            static FORWARD_ROT: [f64; 2] = [0.0, -0.0];

            let rot_sign = vld1q_f64(match self.direction {
                FftDirection::Inverse => INVERSE_ROT.as_ptr(),
                FftDirection::Forward => FORWARD_ROT.as_ptr(),
            });

            let tw3_re = vdupq_n_f64(self.twiddle_re);
            let tw3_wim = vld1q_f64(self.twiddle_im.as_ptr().cast());

            for chunk in in_place.chunks_exact_mut(12) {
                let u0 = vld1q_f64(chunk.as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast());

                let u4 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());
                let u6 = vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast());
                let u7 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());

                let u8 = vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast());
                let u9 = vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast());
                let u10 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u11 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());

                let (u0, u1, u2, u3) = NeonButterfly::butterfly4_f64(u0, u1, u2, u3, rot_sign);
                let (u4, u5, u6, u7) = NeonButterfly::butterfly4_f64(u4, u5, u6, u7, rot_sign);
                let (u8, u9, u10, u11) = NeonButterfly::butterfly4_f64(u8, u9, u10, u11, rot_sign);

                let (y0, y4, y8) = NeonButterfly::butterfly3_f64(u0, u4, u8, tw3_re, tw3_wim);
                let (y9, y1, y5) = NeonButterfly::butterfly3_f64(u1, u5, u9, tw3_re, tw3_wim);
                let (y6, y10, y2) = NeonButterfly::butterfly3_f64(u2, u6, u10, tw3_re, tw3_wim);
                let (y3, y7, y11) = NeonButterfly::butterfly3_f64(u3, u7, u11, tw3_re, tw3_wim);

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
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        12
    }
}

impl FftExecutor<f32> for NeonButterfly12<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 12 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            static INVERSE_ROT: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            static FORWARD_ROT: [f32; 4] = [0.0, -0.0, 0.0, -0.0];

            let rot_sign = vld1q_f32(match self.direction {
                FftDirection::Inverse => INVERSE_ROT.as_ptr(),
                FftDirection::Forward => FORWARD_ROT.as_ptr(),
            });

            let tw3_re = vdupq_n_f32(self.twiddle_re);
            let tw3_wim = vld1q_f32(self.twiddle_im.as_ptr().cast());

            for chunk in in_place.chunks_exact_mut(24) {
                let u0u7 = vld1q_f32(chunk.as_mut_ptr().cast());
                let u10u1 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u11 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u2u5 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u3 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u6u9 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());
                let u16u17 = vld1q_f32(chunk.get_unchecked(16..).as_ptr().cast());
                let u18u19 = vld1q_f32(chunk.get_unchecked(18..).as_ptr().cast());
                let u20u21 = vld1q_f32(chunk.get_unchecked(20..).as_ptr().cast());
                let u22u23 = vld1q_f32(chunk.get_unchecked(22..).as_ptr().cast());

                let u0 = vcombine_f32(vget_low_f32(u0u7), vget_low_f32(u12u13));
                let u1 = vcombine_f32(vget_high_f32(u10u1), vget_high_f32(u14u15));
                let u2 = vcombine_f32(vget_low_f32(u2u5), vget_low_f32(u18u19));
                let u3 = vcombine_f32(vget_high_f32(u8u3), vget_high_f32(u20u21));
                let u4 = vcombine_f32(vget_low_f32(u4u11), vget_low_f32(u16u17));
                let u5 = vcombine_f32(vget_high_f32(u2u5), vget_high_f32(u18u19));
                let u6 = vcombine_f32(vget_low_f32(u6u9), vget_low_f32(u22u23));
                let u7 = vcombine_f32(vget_high_f32(u0u7), vget_high_f32(u12u13));
                let u8 = vcombine_f32(vget_low_f32(u8u3), vget_low_f32(u20u21));
                let u9 = vcombine_f32(vget_high_f32(u6u9), vget_high_f32(u22u23));
                let u10 = vcombine_f32(vget_low_f32(u10u1), vget_low_f32(u14u15));
                let u11 = vcombine_f32(vget_high_f32(u4u11), vget_high_f32(u16u17));

                let (u0, u1, u2, u3) = NeonButterfly::butterfly4_f32(u0, u1, u2, u3, rot_sign);
                let (u4, u5, u6, u7) = NeonButterfly::butterfly4_f32(u4, u5, u6, u7, rot_sign);
                let (u8, u9, u10, u11) = NeonButterfly::butterfly4_f32(u8, u9, u10, u11, rot_sign);

                let (y0, y4, y8) = NeonButterfly::butterfly3_f32(u0, u4, u8, tw3_re, tw3_wim);
                let (y9, y1, y5) = NeonButterfly::butterfly3_f32(u1, u5, u9, tw3_re, tw3_wim);
                let (y6, y10, y2) = NeonButterfly::butterfly3_f32(u2, u6, u10, tw3_re, tw3_wim);
                let (y3, y7, y11) = NeonButterfly::butterfly3_f32(u3, u7, u11, tw3_re, tw3_wim);

                let qy0 = vcombine_f32(vget_low_f32(y0), vget_low_f32(y1));
                let qy1 = vcombine_f32(vget_low_f32(y2), vget_low_f32(y3));
                let qy2 = vcombine_f32(vget_low_f32(y4), vget_low_f32(y5));
                let qy3 = vcombine_f32(vget_low_f32(y6), vget_low_f32(y7));
                let qy4 = vcombine_f32(vget_low_f32(y8), vget_low_f32(y9));
                let qy5 = vcombine_f32(vget_low_f32(y10), vget_low_f32(y11));
                let qy6 = vcombine_f32(vget_high_f32(y0), vget_high_f32(y1));
                let qy7 = vcombine_f32(vget_high_f32(y2), vget_high_f32(y3));
                let qy8 = vcombine_f32(vget_high_f32(y4), vget_high_f32(y5));
                let qy9 = vcombine_f32(vget_high_f32(y6), vget_high_f32(y7));
                let qy10 = vcombine_f32(vget_high_f32(y8), vget_high_f32(y9));
                let qy11 = vcombine_f32(vget_high_f32(y10), vget_high_f32(y11));

                vst1q_f32(chunk.as_mut_ptr().cast(), qy0);
                vst1q_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), qy1);
                vst1q_f32(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), qy2);
                vst1q_f32(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), qy3);
                vst1q_f32(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), qy4);
                vst1q_f32(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), qy5);
                vst1q_f32(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), qy6);
                vst1q_f32(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), qy7);
                vst1q_f32(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), qy8);
                vst1q_f32(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), qy9);
                vst1q_f32(chunk.get_unchecked_mut(20..).as_mut_ptr().cast(), qy10);
                vst1q_f32(chunk.get_unchecked_mut(22..).as_mut_ptr().cast(), qy11);
            }

            let rem = in_place.chunks_exact_mut(24).into_remainder();

            for chunk in rem.chunks_exact_mut(12) {
                let u0u7 = vld1q_f32(chunk.as_mut_ptr().cast());
                let u10u1 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u11 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u2u5 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u3 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u6u9 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());

                let u0 = vget_low_f32(u0u7);
                let u1 = vget_high_f32(u10u1);
                let u2 = vget_low_f32(u2u5);
                let u3 = vget_high_f32(u8u3);
                let u4 = vget_low_f32(u4u11);
                let u5 = vget_high_f32(u2u5);
                let u6 = vget_low_f32(u6u9);
                let u7 = vget_high_f32(u0u7);
                let u8 = vget_low_f32(u8u3);
                let u9 = vget_high_f32(u6u9);
                let u10 = vget_low_f32(u10u1);
                let u11 = vget_high_f32(u4u11);

                let (u0, u1, u2, u3) =
                    NeonButterfly::butterfly4h_f32(u0, u1, u2, u3, vget_low_f32(rot_sign));
                let (u4, u5, u6, u7) =
                    NeonButterfly::butterfly4h_f32(u4, u5, u6, u7, vget_low_f32(rot_sign));
                let (u8, u9, u10, u11) =
                    NeonButterfly::butterfly4h_f32(u8, u9, u10, u11, vget_low_f32(rot_sign));

                let (y0, y4, y8) = NeonButterfly::butterfly3h_f32(
                    u0,
                    u4,
                    u8,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_wim),
                );
                let (y9, y1, y5) = NeonButterfly::butterfly3h_f32(
                    u1,
                    u5,
                    u9,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_wim),
                );
                let (y6, y10, y2) = NeonButterfly::butterfly3h_f32(
                    u2,
                    u6,
                    u10,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_wim),
                );
                let (y3, y7, y11) = NeonButterfly::butterfly3h_f32(
                    u3,
                    u7,
                    u11,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_wim),
                );

                vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y0, y1));
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y2, y3),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y4, y5),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y6, y7),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y8, y9),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(y10, y11),
                );
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        12
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    use rand::Rng;

    test_butterfly!(test_neon_butterfly12, f32, NeonButterfly12, 12, 1e-5);
    test_butterfly!(test_neon_butterfly12_f64, f64, NeonButterfly12, 12, 1e-7);
}
