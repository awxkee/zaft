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
use crate::neon::butterflies::shared::NeonButterfly;
use crate::neon::util::{v_rotate90_f32, v_rotate90_f64, vh_rotate90_f32, vqtrnq_f32};
use crate::traits::FftTrigonometry;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;
use std::sync::Arc;

pub(crate) struct NeonButterfly8<T> {
    direction: FftDirection,
    root2: T,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonButterfly8<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            root2: 0.5f64.sqrt().as_(),
        }
    }
}

impl FftExecutor<f64> for NeonButterfly8<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 8 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            static ROT_270: [f64; 2] = [0.0, -0.0];
            let rot_sign = vld1q_f64(match self.direction {
                FftDirection::Inverse => ROT_90.as_ptr(),
                FftDirection::Forward => ROT_270.as_ptr(),
            });

            for chunk in in_place.chunks_exact_mut(8) {
                let u0 = vld1q_f64(chunk.as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                let u6 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                let u7 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());

                let (u0, u2, u4, u6) = NeonButterfly::butterfly4_f64(u0, u2, u4, u6, rot_sign);
                let (u1, mut u3, mut u5, mut u7) =
                    NeonButterfly::butterfly4_f64(u1, u3, u5, u7, rot_sign);

                u3 = vmulq_n_f64(vaddq_f64(v_rotate90_f64(u3, rot_sign), u3), self.root2);
                u5 = v_rotate90_f64(u5, rot_sign);
                u7 = vmulq_n_f64(vsubq_f64(v_rotate90_f64(u7, rot_sign), u7), self.root2);

                let (y0, y1) = NeonButterfly::butterfly2_f64(u0, u1);
                let (y2, y3) = NeonButterfly::butterfly2_f64(u2, u3);
                let (y4, y5) = NeonButterfly::butterfly2_f64(u4, u5);
                let (y6, y7) = NeonButterfly::butterfly2_f64(u6, u7);

                vst1q_f64(chunk.as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y2);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y4);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y6);
                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y1);
                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y3);
                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y5);
                vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y7);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        8
    }
}

impl FftExecutorOutOfPlace<f64> for NeonButterfly8<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 8 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 8 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            static ROT_270: [f64; 2] = [0.0, -0.0];
            let rot_sign = vld1q_f64(match self.direction {
                FftDirection::Inverse => ROT_90.as_ptr(),
                FftDirection::Forward => ROT_270.as_ptr(),
            });

            for (dst, src) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
                let u0 = vld1q_f64(src.as_ptr().cast());
                let u1 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(src.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(src.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(src.get_unchecked(5..).as_ptr().cast());
                let u6 = vld1q_f64(src.get_unchecked(6..).as_ptr().cast());
                let u7 = vld1q_f64(src.get_unchecked(7..).as_ptr().cast());

                let (u0, u2, u4, u6) = NeonButterfly::butterfly4_f64(u0, u2, u4, u6, rot_sign);
                let (u1, mut u3, mut u5, mut u7) =
                    NeonButterfly::butterfly4_f64(u1, u3, u5, u7, rot_sign);

                u3 = vmulq_n_f64(vaddq_f64(v_rotate90_f64(u3, rot_sign), u3), self.root2);
                u5 = v_rotate90_f64(u5, rot_sign);
                u7 = vmulq_n_f64(vsubq_f64(v_rotate90_f64(u7, rot_sign), u7), self.root2);

                let (y0, y1) = NeonButterfly::butterfly2_f64(u0, u1);
                let (y2, y3) = NeonButterfly::butterfly2_f64(u2, u3);
                let (y4, y5) = NeonButterfly::butterfly2_f64(u4, u5);
                let (y6, y7) = NeonButterfly::butterfly2_f64(u6, u7);

                vst1q_f64(dst.as_mut_ptr().cast(), y0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), y2);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y4);
                vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), y6);
                vst1q_f64(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y1);
                vst1q_f64(dst.get_unchecked_mut(5..).as_mut_ptr().cast(), y3);
                vst1q_f64(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), y5);
                vst1q_f64(dst.get_unchecked_mut(7..).as_mut_ptr().cast(), y7);
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for NeonButterfly8<f64> {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for NeonButterfly8<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 8 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let rot_sign = vld1q_f32(match self.direction {
                FftDirection::Inverse => [-0.0, 0.0, -0.0, 0.0].as_ptr(),
                FftDirection::Forward => [0.0, -0.0, 0.0, -0.0].as_ptr(),
            });

            for chunk in in_place.chunks_exact_mut(16) {
                let u0u1 = vld1q_f32(chunk.as_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());

                let (u0, u1) = vqtrnq_f32(u0u1, u8u9);
                let (u2, u3) = vqtrnq_f32(u2u3, u10u11);
                let (u4, u5) = vqtrnq_f32(u4u5, u12u13);
                let (u6, u7) = vqtrnq_f32(u6u7, u14u15);

                let (u0, u2, u4, u6) = NeonButterfly::butterfly4_f32(u0, u2, u4, u6, rot_sign);
                let (u1, mut u3, mut u5, mut u7) =
                    NeonButterfly::butterfly4_f32(u1, u3, u5, u7, rot_sign);

                u3 = vmulq_n_f32(vaddq_f32(v_rotate90_f32(u3, rot_sign), u3), self.root2);
                u5 = v_rotate90_f32(u5, rot_sign);
                u7 = vmulq_n_f32(vsubq_f32(v_rotate90_f32(u7, rot_sign), u7), self.root2);

                let (zy0, zy1) = NeonButterfly::butterfly2_f32(u0, u1);
                let (zy2, zy3) = NeonButterfly::butterfly2_f32(u2, u3);
                let (zy4, zy5) = NeonButterfly::butterfly2_f32(u4, u5);
                let (zy6, zy7) = NeonButterfly::butterfly2_f32(u6, u7);

                let (y0, y1) = vqtrnq_f32(zy0, zy2);
                let (y2, y3) = vqtrnq_f32(zy4, zy6);
                let (y4, y5) = vqtrnq_f32(zy1, zy3);
                let (y6, y7) = vqtrnq_f32(zy5, zy7);

                vst1q_f32(chunk.as_mut_ptr().cast(), y0);
                vst1q_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f32(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
                vst1q_f32(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y6);
                vst1q_f32(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y1);
                vst1q_f32(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y3);
                vst1q_f32(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y5);
                vst1q_f32(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y7);
            }

            let rem = in_place.chunks_exact_mut(16).into_remainder();

            for chunk in rem.chunks_exact_mut(8) {
                let u0u1 = vld1q_f32(chunk.as_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());

                let (u0, u1) = (vget_low_f32(u0u1), vget_high_f32(u0u1));
                let (u2, u3) = (vget_low_f32(u2u3), vget_high_f32(u2u3));
                let (u4, u5) = (vget_low_f32(u4u5), vget_high_f32(u4u5));
                let (u6, u7) = (vget_low_f32(u6u7), vget_high_f32(u6u7));

                let (u0, u2, u4, u6) =
                    NeonButterfly::butterfly4h_f32(u0, u2, u4, u6, vget_low_f32(rot_sign));
                let (u1, mut u3, mut u5, mut u7) =
                    NeonButterfly::butterfly4h_f32(u1, u3, u5, u7, vget_low_f32(rot_sign));

                u3 = vmul_n_f32(
                    vadd_f32(vh_rotate90_f32(u3, vget_low_f32(rot_sign)), u3),
                    self.root2,
                );
                u5 = vh_rotate90_f32(u5, vget_low_f32(rot_sign));
                u7 = vmul_n_f32(
                    vsub_f32(vh_rotate90_f32(u7, vget_low_f32(rot_sign)), u7),
                    self.root2,
                );

                let (y0, y1) = NeonButterfly::butterfly2h_f32(u0, u1);
                let (y2, y3) = NeonButterfly::butterfly2h_f32(u2, u3);
                let (y4, y5) = NeonButterfly::butterfly2h_f32(u4, u5);
                let (y6, y7) = NeonButterfly::butterfly2h_f32(u6, u7);

                vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y0, y2));
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y4, y6),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y1, y3),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y5, y7),
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
        8
    }
}

impl FftExecutorOutOfPlace<f32> for NeonButterfly8<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 8 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 8 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let rot_sign = vld1q_f32(match self.direction {
                FftDirection::Inverse => [-0.0, 0.0, -0.0, 0.0].as_ptr(),
                FftDirection::Forward => [0.0, -0.0, 0.0, -0.0].as_ptr(),
            });

            for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
                let u0u1 = vld1q_f32(src.as_ptr().cast());
                let u2u3 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(src.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(src.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(src.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(src.get_unchecked(14..).as_ptr().cast());

                let (u0, u1) = vqtrnq_f32(u0u1, u8u9);
                let (u2, u3) = vqtrnq_f32(u2u3, u10u11);
                let (u4, u5) = vqtrnq_f32(u4u5, u12u13);
                let (u6, u7) = vqtrnq_f32(u6u7, u14u15);

                let (u0, u2, u4, u6) = NeonButterfly::butterfly4_f32(u0, u2, u4, u6, rot_sign);
                let (u1, mut u3, mut u5, mut u7) =
                    NeonButterfly::butterfly4_f32(u1, u3, u5, u7, rot_sign);

                u3 = vmulq_n_f32(vaddq_f32(v_rotate90_f32(u3, rot_sign), u3), self.root2);
                u5 = v_rotate90_f32(u5, rot_sign);
                u7 = vmulq_n_f32(vsubq_f32(v_rotate90_f32(u7, rot_sign), u7), self.root2);

                let (zy0, zy1) = NeonButterfly::butterfly2_f32(u0, u1);
                let (zy2, zy3) = NeonButterfly::butterfly2_f32(u2, u3);
                let (zy4, zy5) = NeonButterfly::butterfly2_f32(u4, u5);
                let (zy6, zy7) = NeonButterfly::butterfly2_f32(u6, u7);

                let (y0, y1) = vqtrnq_f32(zy0, zy2);
                let (y2, y3) = vqtrnq_f32(zy4, zy6);
                let (y4, y5) = vqtrnq_f32(zy1, zy3);
                let (y6, y7) = vqtrnq_f32(zy5, zy7);

                vst1q_f32(dst.as_mut_ptr().cast(), y0);
                vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f32(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
                vst1q_f32(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), y6);
                vst1q_f32(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), y1);
                vst1q_f32(dst.get_unchecked_mut(10..).as_mut_ptr().cast(), y3);
                vst1q_f32(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), y5);
                vst1q_f32(dst.get_unchecked_mut(14..).as_mut_ptr().cast(), y7);
            }

            let rem_src = src.chunks_exact(16).remainder();
            let rem_dst = dst.chunks_exact_mut(16).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(8).zip(rem_src.chunks_exact(8)) {
                let u0u1 = vld1q_f32(src.as_ptr().cast());
                let u2u3 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());

                let (u0, u1) = (vget_low_f32(u0u1), vget_high_f32(u0u1));
                let (u2, u3) = (vget_low_f32(u2u3), vget_high_f32(u2u3));
                let (u4, u5) = (vget_low_f32(u4u5), vget_high_f32(u4u5));
                let (u6, u7) = (vget_low_f32(u6u7), vget_high_f32(u6u7));

                let (u0, u2, u4, u6) =
                    NeonButterfly::butterfly4h_f32(u0, u2, u4, u6, vget_low_f32(rot_sign));
                let (u1, mut u3, mut u5, mut u7) =
                    NeonButterfly::butterfly4h_f32(u1, u3, u5, u7, vget_low_f32(rot_sign));

                u3 = vmul_n_f32(
                    vadd_f32(vh_rotate90_f32(u3, vget_low_f32(rot_sign)), u3),
                    self.root2,
                );
                u5 = vh_rotate90_f32(u5, vget_low_f32(rot_sign));
                u7 = vmul_n_f32(
                    vsub_f32(vh_rotate90_f32(u7, vget_low_f32(rot_sign)), u7),
                    self.root2,
                );

                let (y0, y1) = NeonButterfly::butterfly2h_f32(u0, u1);
                let (y2, y3) = NeonButterfly::butterfly2h_f32(u2, u3);
                let (y4, y5) = NeonButterfly::butterfly2h_f32(u4, u5);
                let (y6, y7) = NeonButterfly::butterfly2h_f32(u6, u7);

                vst1q_f32(dst.as_mut_ptr().cast(), vcombine_f32(y0, y2));
                vst1q_f32(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y4, y6),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y1, y3),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y5, y7),
                );
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for NeonButterfly8<f32> {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_neon_butterfly8, f32, NeonButterfly8, 8, 1e-5);
    test_butterfly!(test_neon_butterfly8_f64, f64, NeonButterfly8, 8, 1e-7);
    test_oof_butterfly!(test_oof_butterfly8, f32, NeonButterfly8, 8, 1e-5);
    test_oof_butterfly!(test_oof_butterfly8_f64, f64, NeonButterfly8, 8, 1e-9);
}
