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
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaButterfly15<T> {
    direction: FftDirection,
    bf5: NeonFastButterfly5<T>,
    tw3_re: T,
    tw3_im: [T; 4],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonFcmaButterfly15<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);
        Self {
            direction: fft_direction,
            bf5: NeonFastButterfly5::new(fft_direction),
            tw3_re: twiddle.re,
            tw3_im: [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im],
        }
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly15<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        15
    }
}

impl NeonFcmaButterfly15<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 15 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let tw3_re = vdupq_n_f64(self.tw3_re);
            let tw3_im = vld1q_f64(self.tw3_im.as_ptr().cast());

            for chunk in in_place.chunks_exact_mut(15) {
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
                let u10 = vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast());
                let u11 = vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast());
                let u12 = vld1q_f64(chunk.get_unchecked(12..).as_ptr().cast());
                let u13 = vld1q_f64(chunk.get_unchecked(13..).as_ptr().cast());
                let u14 = vld1q_f64(chunk.get_unchecked(14..).as_ptr().cast());

                let mid0 = self.bf5.exec_fcma(u0, u3, u6, u9, u12);
                let mid1 = self.bf5.exec_fcma(u5, u8, u11, u14, u2);
                let mid2 = self.bf5.exec_fcma(u10, u13, u1, u4, u7);

                // Since this is good-thomas algorithm, we don't need twiddle factors

                // Transpose the data and do size-3 FFTs down the columns
                let (y0, y1, y2) =
                    NeonButterfly::butterfly3_f64_fcma(mid0.0, mid1.0, mid2.0, tw3_re, tw3_im);
                let (y3, y4, y5) =
                    NeonButterfly::butterfly3_f64_fcma(mid0.1, mid1.1, mid2.1, tw3_re, tw3_im);
                let (y6, y7, y8) =
                    NeonButterfly::butterfly3_f64_fcma(mid0.2, mid1.2, mid2.2, tw3_re, tw3_im);
                let (y9, y10, y11) =
                    NeonButterfly::butterfly3_f64_fcma(mid0.3, mid1.3, mid2.3, tw3_re, tw3_im);
                let (y12, y13, y14) =
                    NeonButterfly::butterfly3_f64_fcma(mid0.4, mid1.4, mid2.4, tw3_re, tw3_im);

                vst1q_f64(chunk.as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y4);

                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y8);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y9);

                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y13);
                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y2);

                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y3);
                vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y7);

                vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y11);
                vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y12);

                vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y1);
                vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), y5);

                vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y6);
                vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), y10);

                vst1q_f64(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y14);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaButterfly15<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        15
    }
}

impl NeonFcmaButterfly15<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 15 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let tw3_re = vdupq_n_f32(self.tw3_re);
            let tw3_im = vld1q_f32(self.tw3_im.as_ptr().cast());

            for chunk in in_place.chunks_exact_mut(30) {
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
                let u20u21 = vld1q_f32(chunk.get_unchecked(20..).as_ptr().cast());
                let u22u23 = vld1q_f32(chunk.get_unchecked(22..).as_ptr().cast());
                let u24u25 = vld1q_f32(chunk.get_unchecked(24..).as_ptr().cast());
                let u26u27 = vld1q_f32(chunk.get_unchecked(26..).as_ptr().cast());
                let u28u29 = vld1q_f32(chunk.get_unchecked(28..).as_ptr().cast());

                let u0 = vcombine_f32(vget_low_f32(u0u1), vget_high_f32(u14u15));
                let u1 = vcombine_f32(vget_high_f32(u0u1), vget_low_f32(u16u17));
                let u2 = vcombine_f32(vget_low_f32(u2u3), vget_high_f32(u16u17));
                let u3 = vcombine_f32(vget_high_f32(u2u3), vget_low_f32(u18u19));
                let u4 = vcombine_f32(vget_low_f32(u4u5), vget_high_f32(u18u19));
                let u5 = vcombine_f32(vget_high_f32(u4u5), vget_low_f32(u20u21));
                let u6 = vcombine_f32(vget_low_f32(u6u7), vget_high_f32(u20u21));
                let u7 = vcombine_f32(vget_high_f32(u6u7), vget_low_f32(u22u23));
                let u8 = vcombine_f32(vget_low_f32(u8u9), vget_high_f32(u22u23));
                let u9 = vcombine_f32(vget_high_f32(u8u9), vget_low_f32(u24u25));
                let u10 = vcombine_f32(vget_low_f32(u10u11), vget_high_f32(u24u25));
                let u11 = vcombine_f32(vget_high_f32(u10u11), vget_low_f32(u26u27));
                let u12 = vcombine_f32(vget_low_f32(u12u13), vget_high_f32(u26u27));
                let u13 = vcombine_f32(vget_high_f32(u12u13), vget_low_f32(u28u29));
                let u14 = vcombine_f32(vget_low_f32(u14u15), vget_high_f32(u28u29));

                let mid0 = self.bf5.exec_fcma(u0, u3, u6, u9, u12);
                let mid1 = self.bf5.exec_fcma(u5, u8, u11, u14, u2);
                let mid2 = self.bf5.exec_fcma(u10, u13, u1, u4, u7);

                // Since this is good-thomas algorithm, we don't need twiddle factors

                // Transpose the data and do size-3 FFTs down the columns
                let (y0, y1, y2) =
                    NeonButterfly::butterfly3_f32_fcma(mid0.0, mid1.0, mid2.0, tw3_re, tw3_im);
                let (y3, y4, y5) =
                    NeonButterfly::butterfly3_f32_fcma(mid0.1, mid1.1, mid2.1, tw3_re, tw3_im);
                let (y6, y7, y8) =
                    NeonButterfly::butterfly3_f32_fcma(mid0.2, mid1.2, mid2.2, tw3_re, tw3_im);
                let (y9, y10, y11) =
                    NeonButterfly::butterfly3_f32_fcma(mid0.3, mid1.3, mid2.3, tw3_re, tw3_im);
                let (y12, y13, y14) =
                    NeonButterfly::butterfly3_f32_fcma(mid0.4, mid1.4, mid2.4, tw3_re, tw3_im);

                vst1q_f32(
                    chunk.as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y0), vget_low_f32(y4)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y8), vget_low_f32(y9)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y13), vget_low_f32(y2)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y3), vget_low_f32(y7)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y11), vget_low_f32(y12)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y1), vget_low_f32(y5)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y6), vget_low_f32(y10)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y14), vget_high_f32(y0)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y4), vget_high_f32(y8)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y9), vget_high_f32(y13)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y2), vget_high_f32(y3)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y7), vget_high_f32(y11)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y12), vget_high_f32(y1)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y5), vget_high_f32(y6)),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y10), vget_high_f32(y14)),
                );
            }

            let rem = in_place.chunks_exact_mut(30).into_remainder();

            for chunk in rem.chunks_exact_mut(15) {
                let u0u1 = vld1q_f32(chunk.as_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                let u14 = vld1_f32(chunk.get_unchecked(14..).as_ptr().cast());

                let (u0, u1) = (vget_low_f32(u0u1), vget_high_f32(u0u1));
                let (u2, u3) = (vget_low_f32(u2u3), vget_high_f32(u2u3));
                let (u4, u5) = (vget_low_f32(u4u5), vget_high_f32(u4u5));
                let (u6, u7) = (vget_low_f32(u6u7), vget_high_f32(u6u7));
                let (u8, u9) = (vget_low_f32(u8u9), vget_high_f32(u8u9));
                let (u10, u11) = (vget_low_f32(u10u11), vget_high_f32(u10u11));
                let (u12, u13) = (vget_low_f32(u12u13), vget_high_f32(u12u13));

                let mid0 = self.bf5.exech_fcma(u0, u3, u6, u9, u12);
                let mid1 = self.bf5.exech_fcma(u5, u8, u11, u14, u2);
                let mid2 = self.bf5.exech_fcma(u10, u13, u1, u4, u7);

                // Since this is good-thomas algorithm, we don't need twiddle factors

                // Transpose the data and do size-3 FFTs down the columns
                let (y0, y1, y2) = NeonButterfly::butterfly3h_f32_fcma(
                    mid0.0,
                    mid1.0,
                    mid2.0,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_im),
                );
                let (y3, y4, y5) = NeonButterfly::butterfly3h_f32_fcma(
                    mid0.1,
                    mid1.1,
                    mid2.1,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_im),
                );
                let (y6, y7, y8) = NeonButterfly::butterfly3h_f32_fcma(
                    mid0.2,
                    mid1.2,
                    mid2.2,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_im),
                );
                let (y9, y10, y11) = NeonButterfly::butterfly3h_f32_fcma(
                    mid0.3,
                    mid1.3,
                    mid2.3,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_im),
                );
                let (y12, y13, y14) = NeonButterfly::butterfly3h_f32_fcma(
                    mid0.4,
                    mid1.4,
                    mid2.4,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_im),
                );

                vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y0, y4));

                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y8, y9),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y13, y2),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y3, y7),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y11, y12),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(y1, y5),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vcombine_f32(y6, y10),
                );
                vst1_f32(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y14);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neon::butterflies::test_fcma_butterfly;

    test_fcma_butterfly!(test_fcma_butterfly15, f32, NeonFcmaButterfly15, 15, 1e-5);
    test_fcma_butterfly!(
        test_fcma_butterfly15_f64,
        f64,
        NeonFcmaButterfly15,
        15,
        1e-7
    );
}
