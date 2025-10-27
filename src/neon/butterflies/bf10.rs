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
use crate::neon::butterflies::fast_bf5::NeonFastButterfly5;
use crate::neon::util::vqtrnq_f32;
use crate::traits::FftTrigonometry;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonButterfly10<T> {
    direction: FftDirection,
    bf5: NeonFastButterfly5<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonButterfly10<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf5: NeonFastButterfly5::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for NeonButterfly10<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 10 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            let rot_sign = vld1q_f64(ROT_90.as_ptr());

            for chunk in in_place.chunks_exact_mut(10) {
                let u0 = vld1q_f64(chunk.as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                let u6 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                let u7 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());
                let u8 = vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast());
                let u9 = vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast());

                let mid0 = self.bf5.exec(u0, u2, u4, u6, u8, rot_sign);
                let mid1 = self.bf5.exec(u5, u7, u9, u1, u3, rot_sign);

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) = NeonButterfly::butterfly2_f64(mid0.0, mid1.0);
                let (y2, y3) = NeonButterfly::butterfly2_f64(mid0.1, mid1.1);
                let (y4, y5) = NeonButterfly::butterfly2_f64(mid0.2, mid1.2);
                let (y6, y7) = NeonButterfly::butterfly2_f64(mid0.3, mid1.3);
                let (y8, y9) = NeonButterfly::butterfly2_f64(mid0.4, mid1.4);

                vst1q_f64(chunk.as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y3);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y4);

                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y7);
                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y8);
                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y1);

                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y2);
                vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y5);
                vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y6);

                vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y9);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        10
    }
}

impl FftExecutorOutOfPlace<f64> for NeonButterfly10<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 10 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 10 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            let rot_sign = vld1q_f64(ROT_90.as_ptr());

            for (dst, src) in dst.chunks_exact_mut(10).zip(src.chunks_exact(10)) {
                let u0 = vld1q_f64(src.as_ptr().cast());
                let u1 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(src.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(src.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(src.get_unchecked(5..).as_ptr().cast());
                let u6 = vld1q_f64(src.get_unchecked(6..).as_ptr().cast());
                let u7 = vld1q_f64(src.get_unchecked(7..).as_ptr().cast());
                let u8 = vld1q_f64(src.get_unchecked(8..).as_ptr().cast());
                let u9 = vld1q_f64(src.get_unchecked(9..).as_ptr().cast());

                let mid0 = self.bf5.exec(u0, u2, u4, u6, u8, rot_sign);
                let mid1 = self.bf5.exec(u5, u7, u9, u1, u3, rot_sign);

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) = NeonButterfly::butterfly2_f64(mid0.0, mid1.0);
                let (y2, y3) = NeonButterfly::butterfly2_f64(mid0.1, mid1.1);
                let (y4, y5) = NeonButterfly::butterfly2_f64(mid0.2, mid1.2);
                let (y6, y7) = NeonButterfly::butterfly2_f64(mid0.3, mid1.3);
                let (y8, y9) = NeonButterfly::butterfly2_f64(mid0.4, mid1.4);

                vst1q_f64(dst.as_mut_ptr().cast(), y0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), y3);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y4);

                vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), y7);
                vst1q_f64(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y8);
                vst1q_f64(dst.get_unchecked_mut(5..).as_mut_ptr().cast(), y1);

                vst1q_f64(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), y2);
                vst1q_f64(dst.get_unchecked_mut(7..).as_mut_ptr().cast(), y5);
                vst1q_f64(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), y6);

                vst1q_f64(dst.get_unchecked_mut(9..).as_mut_ptr().cast(), y9);
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for NeonButterfly10<f64> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for NeonButterfly10<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 10 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            static ROT_90: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            let rot_sign = vld1q_f32(ROT_90.as_ptr());

            for chunk in in_place.chunks_exact_mut(20) {
                let u0u1 = vld1q_f32(chunk.as_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());
                let u16u17 = vld1q_f32(chunk.get_unchecked(16..).as_ptr().cast());
                let u18u19 = vld1q_f32(chunk.get_unchecked(18..).as_ptr().cast());

                let (u0, u1) = vqtrnq_f32(u0u1, u10u11);
                let (u2, u3) = vqtrnq_f32(u2u3, u12u13);
                let (u4, u5) = vqtrnq_f32(u4u5, u14u15);
                let (u6, u7) = vqtrnq_f32(u6u7, u16u17);
                let (u8, u9) = vqtrnq_f32(u8u9, u18u19);

                let mid0 = self.bf5.exec(u0, u2, u4, u6, u8, rot_sign);
                let mid1 = self.bf5.exec(u5, u7, u9, u1, u3, rot_sign);

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (zy0, zy1) = NeonButterfly::butterfly2_f32(mid0.0, mid1.0);
                let (zy2, zy3) = NeonButterfly::butterfly2_f32(mid0.1, mid1.1);
                let (zy4, zy5) = NeonButterfly::butterfly2_f32(mid0.2, mid1.2);
                let (zy6, zy7) = NeonButterfly::butterfly2_f32(mid0.3, mid1.3);
                let (zy8, zy9) = NeonButterfly::butterfly2_f32(mid0.4, mid1.4);

                let (y0, y1) = vqtrnq_f32(zy0, zy3);
                let (y2, y3) = vqtrnq_f32(zy4, zy7);
                let (y4, y5) = vqtrnq_f32(zy8, zy1);
                let (y6, y7) = vqtrnq_f32(zy2, zy5);
                let (y8, y9) = vqtrnq_f32(zy6, zy9);

                vst1q_f32(chunk.as_mut_ptr().cast(), y0);
                vst1q_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f32(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
                vst1q_f32(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y6);
                vst1q_f32(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y8);
                vst1q_f32(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y1);
                vst1q_f32(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y3);
                vst1q_f32(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y5);
                vst1q_f32(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), y7);
                vst1q_f32(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), y9);
            }

            let rem = in_place.chunks_exact_mut(20).into_remainder();

            for chunk in rem.chunks_exact_mut(10) {
                let u0u1 = vld1q_f32(chunk.as_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());

                let (u0, u1) = (vget_low_f32(u0u1), vget_high_f32(u0u1));
                let (u2, u3) = (vget_low_f32(u2u3), vget_high_f32(u2u3));
                let (u4, u5) = (vget_low_f32(u4u5), vget_high_f32(u4u5));
                let (u6, u7) = (vget_low_f32(u6u7), vget_high_f32(u6u7));
                let (u8, u9) = (vget_low_f32(u8u9), vget_high_f32(u8u9));

                let mid0 = self.bf5.exec(
                    vcombine_f32(u0, u5),
                    vcombine_f32(u2, u7),
                    vcombine_f32(u4, u9),
                    vcombine_f32(u6, u1),
                    vcombine_f32(u8, u3),
                    rot_sign,
                );

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) =
                    NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.0), vget_high_f32(mid0.0));
                let (y2, y3) =
                    NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.1), vget_high_f32(mid0.1));
                let (y4, y5) =
                    NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.2), vget_high_f32(mid0.2));
                let (y6, y7) =
                    NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.3), vget_high_f32(mid0.3));
                let (y8, y9) =
                    NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.4), vget_high_f32(mid0.4));

                vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y0, y3));
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y4, y7),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y8, y1),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y2, y5),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y6, y9),
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
        10
    }
}

impl FftExecutorOutOfPlace<f32> for NeonButterfly10<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 10 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 10 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            static ROT_90: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            let rot_sign = vld1q_f32(ROT_90.as_ptr());

            for (dst, src) in dst.chunks_exact_mut(20).zip(src.chunks_exact(20)) {
                let u0u1 = vld1q_f32(src.as_ptr().cast());
                let u2u3 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(src.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(src.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(src.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(src.get_unchecked(14..).as_ptr().cast());
                let u16u17 = vld1q_f32(src.get_unchecked(16..).as_ptr().cast());
                let u18u19 = vld1q_f32(src.get_unchecked(18..).as_ptr().cast());

                let (u0, u1) = vqtrnq_f32(u0u1, u10u11);
                let (u2, u3) = vqtrnq_f32(u2u3, u12u13);
                let (u4, u5) = vqtrnq_f32(u4u5, u14u15);
                let (u6, u7) = vqtrnq_f32(u6u7, u16u17);
                let (u8, u9) = vqtrnq_f32(u8u9, u18u19);

                let mid0 = self.bf5.exec(u0, u2, u4, u6, u8, rot_sign);
                let mid1 = self.bf5.exec(u5, u7, u9, u1, u3, rot_sign);

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (zy0, zy1) = NeonButterfly::butterfly2_f32(mid0.0, mid1.0);
                let (zy2, zy3) = NeonButterfly::butterfly2_f32(mid0.1, mid1.1);
                let (zy4, zy5) = NeonButterfly::butterfly2_f32(mid0.2, mid1.2);
                let (zy6, zy7) = NeonButterfly::butterfly2_f32(mid0.3, mid1.3);
                let (zy8, zy9) = NeonButterfly::butterfly2_f32(mid0.4, mid1.4);

                let (y0, y1) = vqtrnq_f32(zy0, zy3);
                let (y2, y3) = vqtrnq_f32(zy4, zy7);
                let (y4, y5) = vqtrnq_f32(zy8, zy1);
                let (y6, y7) = vqtrnq_f32(zy2, zy5);
                let (y8, y9) = vqtrnq_f32(zy6, zy9);

                vst1q_f32(dst.as_mut_ptr().cast(), y0);
                vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f32(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
                vst1q_f32(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), y6);
                vst1q_f32(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), y8);
                vst1q_f32(dst.get_unchecked_mut(10..).as_mut_ptr().cast(), y1);
                vst1q_f32(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), y3);
                vst1q_f32(dst.get_unchecked_mut(14..).as_mut_ptr().cast(), y5);
                vst1q_f32(dst.get_unchecked_mut(16..).as_mut_ptr().cast(), y7);
                vst1q_f32(dst.get_unchecked_mut(18..).as_mut_ptr().cast(), y9);
            }

            let rem_dst = dst.chunks_exact_mut(20).into_remainder();
            let rem_src = src.chunks_exact(20).remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(10).zip(rem_src.chunks_exact(10)) {
                let u0u1 = vld1q_f32(src.as_ptr().cast());
                let u2u3 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(src.get_unchecked(8..).as_ptr().cast());

                let (u0, u1) = (vget_low_f32(u0u1), vget_high_f32(u0u1));
                let (u2, u3) = (vget_low_f32(u2u3), vget_high_f32(u2u3));
                let (u4, u5) = (vget_low_f32(u4u5), vget_high_f32(u4u5));
                let (u6, u7) = (vget_low_f32(u6u7), vget_high_f32(u6u7));
                let (u8, u9) = (vget_low_f32(u8u9), vget_high_f32(u8u9));

                let mid0 = self.bf5.exec(
                    vcombine_f32(u0, u5),
                    vcombine_f32(u2, u7),
                    vcombine_f32(u4, u9),
                    vcombine_f32(u6, u1),
                    vcombine_f32(u8, u3),
                    rot_sign,
                );

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) =
                    NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.0), vget_high_f32(mid0.0));
                let (y2, y3) =
                    NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.1), vget_high_f32(mid0.1));
                let (y4, y5) =
                    NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.2), vget_high_f32(mid0.2));
                let (y6, y7) =
                    NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.3), vget_high_f32(mid0.3));
                let (y8, y9) =
                    NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.4), vget_high_f32(mid0.4));

                vst1q_f32(dst.as_mut_ptr().cast(), vcombine_f32(y0, y3));
                vst1q_f32(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y4, y7),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y8, y1),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y2, y5),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y6, y9),
                );
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for NeonButterfly10<f32> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Butterfly10;
    use crate::dft::Dft;
    use rand::Rng;

    #[test]
    fn test_butterfly10_f32() {
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
            let mut z_ref = input.to_vec();

            let radix5_reference = Butterfly10::new(FftDirection::Forward);
            let radix5_inv_reference = Butterfly10::new(FftDirection::Inverse);

            let radix_forward = NeonButterfly10::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly10::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix5_reference.execute(&mut z_ref).unwrap();

            input.iter().zip(z_ref.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}, reference failed",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}, reference failed",
                    a.im,
                    b.im,
                    size
                );
            });

            radix_inverse.execute(&mut input).unwrap();
            radix5_inv_reference.execute(&mut z_ref).unwrap();

            input.iter().zip(z_ref.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}, reference inv failed",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}, reference inv failed",
                    a.im,
                    b.im,
                    size
                );
            });

            input = input.iter().map(|&x| x * (1.0 / 10f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly10_f64() {
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
            let mut z_ref = input.to_vec();

            let radix10_reference = Butterfly10::new(FftDirection::Forward);

            let radix_forward = NeonButterfly10::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly10::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();

            radix10_reference.execute(&mut z_ref).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {}, reference failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {}, reference failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 10f64)).collect();

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

    #[test]
    fn test_butterfly10_out_of_place_f64() {
        for i in 1..4 {
            let size = 10usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut out_of_place = vec![Complex::<f64>::default(); size];
            let mut ref_input = input.to_vec();
            let radix_forward = NeonButterfly10::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly10::new(FftDirection::Inverse);

            let reference_dft = Dft::new(10, FftDirection::Forward).unwrap();
            reference_dft.execute(&mut ref_input).unwrap();

            radix_forward
                .execute_out_of_place(&input, &mut out_of_place)
                .unwrap();

            out_of_place
                .iter()
                .zip(ref_input.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse
                .execute_out_of_place(&out_of_place, &mut input)
                .unwrap();

            input = input.iter().map(|&x| x * (1.0 / 10f64)).collect();

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

    #[test]
    fn test_butterfly10_out_of_place_f32() {
        for i in 1..4 {
            let size = 10usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut out_of_place = vec![Complex::<f32>::default(); size];
            let mut ref_input = input.to_vec();
            let radix_forward = NeonButterfly10::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly10::new(FftDirection::Inverse);

            let reference_dft = Dft::new(10, FftDirection::Forward).unwrap();
            reference_dft.execute(&mut ref_input).unwrap();

            radix_forward
                .execute_out_of_place(&input, &mut out_of_place)
                .unwrap();

            out_of_place
                .iter()
                .zip(ref_input.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-4,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-4,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse
                .execute_out_of_place(&out_of_place, &mut input)
                .unwrap();

            input = input.iter().map(|&x| x * (1.0 / 10f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }
}
