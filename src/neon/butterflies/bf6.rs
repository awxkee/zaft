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
use crate::neon::util::{pack_complex_hi, pack_complex_lo};
use crate::radix6::Radix6Twiddles;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonButterfly6<T> {
    direction: FftDirection,
    twiddle_re: T,
    twiddle_im: [T; 4],
}

impl<T: Default + Clone + Radix6Twiddles + 'static + Copy + FftTrigonometry + Float>
    NeonButterfly6<T>
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

impl FftExecutor<f32> for NeonButterfly6<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 6 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let twiddle_re = unsafe { vdupq_n_f32(self.twiddle_re) };
        let twiddle_w_2 = unsafe { vld1q_f32(self.twiddle_im.as_ptr().cast()) };

        for chunk in in_place.chunks_exact_mut(12) {
            unsafe {
                let uz0 = vld1q_f32(chunk.as_ptr().cast());
                let uz1 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let uz2 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let uz3 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let uz4 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let uz10 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());

                let (u0, u1) = (pack_complex_lo(uz0, uz3), pack_complex_hi(uz0, uz3));
                let (u2, u3) = (pack_complex_lo(uz1, uz4), pack_complex_hi(uz1, uz4));
                let (u4, u5) = (pack_complex_lo(uz2, uz10), pack_complex_hi(uz2, uz10));

                let (t0, t2, t4) =
                    NeonButterfly::butterfly3_f32(u0, u2, u4, twiddle_re, twiddle_w_2);
                let (t1, t3, t5) =
                    NeonButterfly::butterfly3_f32(u3, u5, u1, twiddle_re, twiddle_w_2);
                let (y0, y3) = NeonButterfly::butterfly2_f32(t0, t1);
                let (y4, y1) = NeonButterfly::butterfly2_f32(t2, t3);
                let (y2, y5) = NeonButterfly::butterfly2_f32(t4, t5);

                let row0 = pack_complex_lo(y0, y1);
                let row1 = pack_complex_lo(y2, y3);
                let row2 = pack_complex_lo(y4, y5);
                let row3 = pack_complex_hi(y0, y1);
                let row4 = pack_complex_hi(y2, y3);
                let row5 = pack_complex_hi(y4, y5);

                vst1q_f32(chunk.as_mut_ptr().cast(), row0);
                vst1q_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), row1);
                vst1q_f32(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), row2);
                vst1q_f32(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), row3);
                vst1q_f32(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), row4);
                vst1q_f32(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), row5);
            }
        }

        let rem = in_place.chunks_exact_mut(12).into_remainder();

        for chunk in rem.chunks_exact_mut(6) {
            unsafe {
                let uz0 = vld1q_f32(chunk.as_ptr().cast());
                let uz1 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let uz2 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());

                let (u0, u1) = (vget_low_f32(uz0), vget_high_f32(uz0));
                let (u2, u3) = (vget_low_f32(uz1), vget_high_f32(uz1));
                let (u4, u5) = (vget_low_f32(uz2), vget_high_f32(uz2));

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

                vst1q_f32(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(y0, y1),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y2, y3),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y4, y5),
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
        6
    }
}

impl FftExecutorOutOfPlace<f32> for NeonButterfly6<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 6 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 6 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        let twiddle_re = unsafe { vdupq_n_f32(self.twiddle_re) };
        let twiddle_w_2 = unsafe { vld1q_f32(self.twiddle_im.as_ptr().cast()) };

        for (dst, src) in dst.chunks_exact_mut(12).zip(src.chunks_exact(12)) {
            unsafe {
                let uz0 = vld1q_f32(src.get_unchecked(0..).as_ptr().cast());
                let uz1 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let uz2 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let uz3 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());
                let uz4 = vld1q_f32(src.get_unchecked(8..).as_ptr().cast());
                let uz10 = vld1q_f32(src.get_unchecked(10..).as_ptr().cast());

                let (u0, u1) = (pack_complex_lo(uz0, uz3), pack_complex_hi(uz0, uz3));
                let (u2, u3) = (pack_complex_lo(uz1, uz4), pack_complex_hi(uz1, uz4));
                let (u4, u5) = (pack_complex_lo(uz2, uz10), pack_complex_hi(uz2, uz10));

                let (t0, t2, t4) =
                    NeonButterfly::butterfly3_f32(u0, u2, u4, twiddle_re, twiddle_w_2);
                let (t1, t3, t5) =
                    NeonButterfly::butterfly3_f32(u3, u5, u1, twiddle_re, twiddle_w_2);
                let (y0, y3) = NeonButterfly::butterfly2_f32(t0, t1);
                let (y4, y1) = NeonButterfly::butterfly2_f32(t2, t3);
                let (y2, y5) = NeonButterfly::butterfly2_f32(t4, t5);

                let row0 = pack_complex_lo(y0, y1);
                let row1 = pack_complex_lo(y2, y3);
                let row2 = pack_complex_lo(y4, y5);
                let row3 = pack_complex_hi(y0, y1);
                let row4 = pack_complex_hi(y2, y3);
                let row5 = pack_complex_hi(y4, y5);

                vst1q_f32(dst.as_mut_ptr().cast(), row0);
                vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), row1);
                vst1q_f32(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), row2);
                vst1q_f32(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), row3);
                vst1q_f32(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), row4);
                vst1q_f32(dst.get_unchecked_mut(10..).as_mut_ptr().cast(), row5);
            }
        }

        let rem_src = src.chunks_exact(12).remainder();
        let rem_dst = dst.chunks_exact_mut(12).into_remainder();

        for (dst, src) in rem_dst.chunks_exact_mut(6).zip(rem_src.chunks_exact(6)) {
            unsafe {
                let uz0 = vld1q_f32(src.as_ptr().cast());
                let uz1 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let uz2 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());

                let (u0, u1) = (vget_low_f32(uz0), vget_high_f32(uz0));
                let (u2, u3) = (vget_low_f32(uz1), vget_high_f32(uz1));
                let (u4, u5) = (vget_low_f32(uz2), vget_high_f32(uz2));

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

                vst1q_f32(
                    dst.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(y0, y1),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y2, y3),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y4, y5),
                );
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for NeonButterfly6<f32> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl FftExecutor<f64> for NeonButterfly6<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 6 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let twiddle_re = unsafe { vdupq_n_f64(self.twiddle_re) };
        let twiddle_w_2 = unsafe { vld1q_f64(self.twiddle_im.as_ptr().cast()) };

        for chunk in in_place.chunks_exact_mut(6) {
            unsafe {
                let u0 = vld1q_f64(chunk.as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());

                let (t0, t2, t4) =
                    NeonButterfly::butterfly3_f64(u0, u2, u4, twiddle_re, twiddle_w_2);
                let (t1, t3, t5) =
                    NeonButterfly::butterfly3_f64(u3, u5, u1, twiddle_re, twiddle_w_2);
                let (y0, y3) = NeonButterfly::butterfly2_f64(t0, t1);
                let (y4, y1) = NeonButterfly::butterfly2_f64(t2, t3);
                let (y2, y5) = NeonButterfly::butterfly2_f64(t4, t5);

                vst1q_f64(chunk.as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3);
                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y5);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        6
    }
}

impl FftExecutorOutOfPlace<f64> for NeonButterfly6<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 6 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 6 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        let twiddle_re = unsafe { vdupq_n_f64(self.twiddle_re) };
        let twiddle_w_2 = unsafe { vld1q_f64(self.twiddle_im.as_ptr().cast()) };

        for (dst, src) in dst.chunks_exact_mut(6).zip(src.chunks_exact(6)) {
            unsafe {
                let u0 = vld1q_f64(src.as_ptr().cast());
                let u1 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(src.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(src.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(src.get_unchecked(5..).as_ptr().cast());

                let (t0, t2, t4) =
                    NeonButterfly::butterfly3_f64(u0, u2, u4, twiddle_re, twiddle_w_2);
                let (t1, t3, t5) =
                    NeonButterfly::butterfly3_f64(u3, u5, u1, twiddle_re, twiddle_w_2);
                let (y0, y3) = NeonButterfly::butterfly2_f64(t0, t1);
                let (y4, y1) = NeonButterfly::butterfly2_f64(t2, t3);
                let (y2, y5) = NeonButterfly::butterfly2_f64(t4, t5);

                vst1q_f64(dst.as_mut_ptr().cast(), y0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), y3);
                vst1q_f64(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
                vst1q_f64(dst.get_unchecked_mut(5..).as_mut_ptr().cast(), y5);
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for NeonButterfly6<f64> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_neon_butterfly6, f32, NeonButterfly6, 6, 1e-5);
    test_butterfly!(test_neon_butterfly6_f64, f64, NeonButterfly6, 6, 1e-7);
    test_oof_butterfly!(test_oof_butterfly6, f32, NeonButterfly6, 6, 1e-5);
    test_oof_butterfly!(test_oof_butterfly6_f64, f64, NeonButterfly6, 6, 1e-9);
}
