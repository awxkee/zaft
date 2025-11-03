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
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaButterfly5<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    direction: FftDirection,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonFcmaButterfly5<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 5, fft_direction),
            twiddle2: compute_twiddle(2, 5, fft_direction),
        }
    }
}

impl FftExecutor<f32> for NeonFcmaButterfly5<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        5
    }
}

impl NeonFcmaButterfly5<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 5 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }
        unsafe {
            let tw1_re = vdupq_n_f32(self.twiddle1.re);
            let tw1_im = vdupq_n_f32(self.twiddle1.im);
            let tw2_re = vdupq_n_f32(self.twiddle2.re);
            let tw2_im = vdupq_n_f32(self.twiddle2.im);

            let tw1_tw2_im = vcombine_f32(vget_low_f32(tw1_im), vget_low_f32(tw2_im));
            let tw2_ntw1_im = vcombine_f32(vget_low_f32(tw2_im), vdup_n_f32(-self.twiddle1.im));
            let a1_a2 = vcombine_f32(vget_low_f32(tw1_re), vget_low_f32(tw2_re));
            let a1_a2_2 = vcombine_f32(vget_low_f32(tw2_re), vget_low_f32(tw1_re));

            for chunk in in_place.chunks_exact_mut(10) {
                let uz0 = vld1q_f32(chunk.get_unchecked(0..).as_ptr().cast());
                let uz1 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let uz2 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let uz3 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let uz4 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());

                // Combine into u0..u4
                let u0 = vcombine_f32(vget_low_f32(uz0), vget_high_f32(uz2));
                let u1 = vcombine_f32(vget_high_f32(uz0), vget_low_f32(uz3));
                let u2 = vcombine_f32(vget_low_f32(uz1), vget_high_f32(uz3));
                let u3 = vcombine_f32(vget_high_f32(uz1), vget_low_f32(uz4));
                let u4 = vcombine_f32(vget_low_f32(uz2), vget_high_f32(uz4));

                // Radix-5 butterfly

                let x14p = vaddq_f32(u1, u4);
                let x14n = vsubq_f32(u1, u4);
                let x23p = vaddq_f32(u2, u3);
                let x23n = vsubq_f32(u2, u3);
                let y0 = vaddq_f32(vaddq_f32(u0, x14p), x23p);

                let temp_b1_1 = vmulq_f32(tw1_im, x14n);
                let temp_b2_1 = vmulq_f32(tw2_im, x14n);

                let temp_a1 = vfmaq_f32(vfmaq_f32(u0, tw1_re, x14p), tw2_re, x23p);
                let temp_a2 = vfmaq_f32(vfmaq_f32(u0, tw2_re, x14p), tw1_re, x23p);

                let temp_b1 = vfmaq_f32(temp_b1_1, tw2_im, x23n);
                let temp_b2 = vfmsq_f32(temp_b2_1, tw1_im, x23n);

                let y1 = vcaddq_rot90_f32(temp_a1, temp_b1);
                let y2 = vcaddq_rot90_f32(temp_a2, temp_b2);
                let y3 = vcaddq_rot270_f32(temp_a2, temp_b2);
                let y4 = vcaddq_rot270_f32(temp_a1, temp_b1);

                vst1q_f32(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y0), vget_low_f32(y1)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y2), vget_low_f32(y3)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y4), vget_high_f32(y0)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y1), vget_high_f32(y2)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y3), vget_high_f32(y4)),
                );
            }

            let rem = in_place.chunks_exact_mut(10).into_remainder();

            for chunk in rem.chunks_exact_mut(5) {
                let uz0 = vld1q_f32(chunk.get_unchecked(0..).as_ptr().cast());
                let uz1 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());

                let u0 = vget_low_f32(uz0);
                let u1 = vget_high_f32(uz0);
                let u2 = vget_low_f32(uz1);
                let u3 = vget_high_f32(uz1);
                let u4 = vld1_f32(chunk.get_unchecked(4..).as_ptr().cast());

                // Radix-5 butterfly

                let x14px23p = vaddq_f32(vcombine_f32(u1, u2), vcombine_f32(u4, u3));
                let x14nx23n = vsubq_f32(vcombine_f32(u1, u2), vcombine_f32(u4, u3));
                let y0 = vadd_f32(
                    vadd_f32(u0, vget_low_f32(x14px23p)),
                    vget_high_f32(x14px23p),
                );

                let temp_b1_1_b2_1 = vmulq_f32(
                    tw1_tw2_im,
                    vcombine_f32(vget_low_f32(x14nx23n), vget_low_f32(x14nx23n)),
                );

                let wx23p = vcombine_f32(vget_high_f32(x14px23p), vget_high_f32(x14px23p));
                let wx23n = vcombine_f32(vget_high_f32(x14nx23n), vget_high_f32(x14nx23n));

                let temp_a1_a2 = vfmaq_f32(
                    vfmaq_f32(
                        vcombine_f32(u0, u0),
                        a1_a2,
                        vcombine_f32(vget_low_f32(x14px23p), vget_low_f32(x14px23p)),
                    ),
                    a1_a2_2,
                    wx23p,
                );

                let temp_b1_b2 = vfmaq_f32(temp_b1_1_b2_1, tw2_ntw1_im, wx23n);

                let y1y2 = vcaddq_rot90_f32(temp_a1_a2, temp_b1_b2);
                let y4y3 = vcaddq_rot270_f32(temp_a1_a2, temp_b1_b2);

                vst1_f32(chunk.as_mut_ptr().cast(), y0);
                vst1q_f32(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1y2);
                vst1q_f32(
                    chunk.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y4y3), vget_low_f32(y4y3)),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_out_of_place_f32(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 5 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 5 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let tw1_re = vdupq_n_f32(self.twiddle1.re);
            let tw1_im = vdupq_n_f32(self.twiddle1.im);
            let tw2_re = vdupq_n_f32(self.twiddle2.re);
            let tw2_im = vdupq_n_f32(self.twiddle2.im);

            let tw1_tw2_im = vcombine_f32(vget_low_f32(tw1_im), vget_low_f32(tw2_im));
            let tw2_ntw1_im = vcombine_f32(vget_low_f32(tw2_im), vdup_n_f32(-self.twiddle1.im));
            let a1_a2 = vcombine_f32(vget_low_f32(tw1_re), vget_low_f32(tw2_re));
            let a1_a2_2 = vcombine_f32(vget_low_f32(tw2_re), vget_low_f32(tw1_re));

            for (dst, src) in dst.chunks_exact_mut(10).zip(src.chunks_exact(10)) {
                let uz0 = vld1q_f32(src.get_unchecked(0..).as_ptr().cast());
                let uz1 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let uz2 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let uz3 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());
                let uz4 = vld1q_f32(src.get_unchecked(8..).as_ptr().cast());

                // Combine into u0..u4
                let u0 = vcombine_f32(vget_low_f32(uz0), vget_high_f32(uz2));
                let u1 = vcombine_f32(vget_high_f32(uz0), vget_low_f32(uz3));
                let u2 = vcombine_f32(vget_low_f32(uz1), vget_high_f32(uz3));
                let u3 = vcombine_f32(vget_high_f32(uz1), vget_low_f32(uz4));
                let u4 = vcombine_f32(vget_low_f32(uz2), vget_high_f32(uz4));

                // Radix-5 butterfly

                let x14p = vaddq_f32(u1, u4);
                let x14n = vsubq_f32(u1, u4);
                let x23p = vaddq_f32(u2, u3);
                let x23n = vsubq_f32(u2, u3);
                let y0 = vaddq_f32(vaddq_f32(u0, x14p), x23p);

                let temp_b1_1 = vmulq_f32(tw1_im, x14n);
                let temp_b2_1 = vmulq_f32(tw2_im, x14n);

                let temp_a1 = vfmaq_f32(vfmaq_f32(u0, tw1_re, x14p), tw2_re, x23p);
                let temp_a2 = vfmaq_f32(vfmaq_f32(u0, tw2_re, x14p), tw1_re, x23p);

                let temp_b1 = vfmaq_f32(temp_b1_1, tw2_im, x23n);
                let temp_b2 = vfmsq_f32(temp_b2_1, tw1_im, x23n);

                let y1 = vcaddq_rot90_f32(temp_a1, temp_b1);
                let y2 = vcaddq_rot90_f32(temp_a2, temp_b2);
                let y3 = vcaddq_rot270_f32(temp_a2, temp_b2);
                let y4 = vcaddq_rot270_f32(temp_a1, temp_b1);

                vst1q_f32(
                    dst.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y0), vget_low_f32(y1)),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y2), vget_low_f32(y3)),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y4), vget_high_f32(y0)),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y1), vget_high_f32(y2)),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y3), vget_high_f32(y4)),
                );
            }

            let rem_dst = dst.chunks_exact_mut(10).into_remainder();
            let rem_src = src.chunks_exact(10).remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(5).zip(rem_src.chunks_exact(5)) {
                let uz0 = vld1q_f32(src.get_unchecked(0..).as_ptr().cast());
                let uz1 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());

                let u0 = vget_low_f32(uz0);
                let u1 = vget_high_f32(uz0);
                let u2 = vget_low_f32(uz1);
                let u3 = vget_high_f32(uz1);
                let u4 = vld1_f32(src.get_unchecked(4..).as_ptr().cast());

                // Radix-5 butterfly

                let x14px23p = vaddq_f32(vcombine_f32(u1, u2), vcombine_f32(u4, u3));
                let x14nx23n = vsubq_f32(vcombine_f32(u1, u2), vcombine_f32(u4, u3));
                let y0 = vadd_f32(
                    vadd_f32(u0, vget_low_f32(x14px23p)),
                    vget_high_f32(x14px23p),
                );

                let temp_b1_1_b2_1 = vmulq_f32(
                    tw1_tw2_im,
                    vcombine_f32(vget_low_f32(x14nx23n), vget_low_f32(x14nx23n)),
                );

                let wx23p = vcombine_f32(vget_high_f32(x14px23p), vget_high_f32(x14px23p));
                let wx23n = vcombine_f32(vget_high_f32(x14nx23n), vget_high_f32(x14nx23n));

                let temp_a1_a2 = vfmaq_f32(
                    vfmaq_f32(
                        vcombine_f32(u0, u0),
                        a1_a2,
                        vcombine_f32(vget_low_f32(x14px23p), vget_low_f32(x14px23p)),
                    ),
                    a1_a2_2,
                    wx23p,
                );

                let temp_b1_b2 = vfmaq_f32(temp_b1_1_b2_1, tw2_ntw1_im, wx23n);

                let y1y2 = vcaddq_rot90_f32(temp_a1_a2, temp_b1_b2);
                let y4y3 = vcaddq_rot270_f32(temp_a1_a2, temp_b1_b2);

                vst1_f32(dst.as_mut_ptr().cast(), y0);
                vst1q_f32(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), y1y2);
                vst1q_f32(
                    dst.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y4y3), vget_low_f32(y4y3)),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for NeonFcmaButterfly5<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for NeonFcmaButterfly5<f32> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly5<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        5
    }
}

impl NeonFcmaButterfly5<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 5 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let tw1_re = vdupq_n_f64(self.twiddle1.re);
            let tw1_im = vdupq_n_f64(self.twiddle1.im);
            let tw2_re = vdupq_n_f64(self.twiddle2.re);
            let tw2_im = vdupq_n_f64(self.twiddle2.im);

            for chunk in in_place.chunks_exact_mut(5) {
                let u0 = vld1q_f64(chunk.get_unchecked(0..).as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());

                // Radix-5 butterfly

                let x14p = vaddq_f64(u1, u4);
                let x14n = vsubq_f64(u1, u4);
                let x23p = vaddq_f64(u2, u3);
                let x23n = vsubq_f64(u2, u3);
                let y0 = vaddq_f64(vaddq_f64(u0, x14p), x23p);

                let temp_b1_1 = vmulq_f64(tw1_im, x14n);
                let temp_b2_1 = vmulq_f64(tw2_im, x14n);

                let temp_a1 = vfmaq_f64(vfmaq_f64(u0, tw1_re, x14p), tw2_re, x23p);
                let temp_a2 = vfmaq_f64(vfmaq_f64(u0, tw2_re, x14p), tw1_re, x23p);

                let temp_b1 = vfmaq_f64(temp_b1_1, tw2_im, x23n);
                let temp_b2 = vfmsq_f64(temp_b2_1, tw1_im, x23n);

                let y1 = vcaddq_rot90_f64(temp_a1, temp_b1);
                let y2 = vcaddq_rot90_f64(temp_a2, temp_b2);
                let y3 = vcaddq_rot270_f64(temp_a2, temp_b2);
                let y4 = vcaddq_rot270_f64(temp_a1, temp_b1);

                vst1q_f64(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3);
                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_out_of_place_f64(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 5 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 5 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let tw1_re = vdupq_n_f64(self.twiddle1.re);
            let tw1_im = vdupq_n_f64(self.twiddle1.im);
            let tw2_re = vdupq_n_f64(self.twiddle2.re);
            let tw2_im = vdupq_n_f64(self.twiddle2.im);

            for (dst, src) in dst.chunks_exact_mut(5).zip(src.chunks_exact(5)) {
                let u0 = vld1q_f64(src.get_unchecked(0..).as_ptr().cast());
                let u1 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(src.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(src.get_unchecked(4..).as_ptr().cast());

                // Radix-5 butterfly

                let x14p = vaddq_f64(u1, u4);
                let x14n = vsubq_f64(u1, u4);
                let x23p = vaddq_f64(u2, u3);
                let x23n = vsubq_f64(u2, u3);
                let y0 = vaddq_f64(vaddq_f64(u0, x14p), x23p);

                let temp_b1_1 = vmulq_f64(tw1_im, x14n);
                let temp_b2_1 = vmulq_f64(tw2_im, x14n);

                let temp_a1 = vfmaq_f64(vfmaq_f64(u0, tw1_re, x14p), tw2_re, x23p);
                let temp_a2 = vfmaq_f64(vfmaq_f64(u0, tw2_re, x14p), tw1_re, x23p);

                let temp_b1 = vfmaq_f64(temp_b1_1, tw2_im, x23n);
                let temp_b2 = vfmsq_f64(temp_b2_1, tw1_im, x23n);

                let y1 = vcaddq_rot90_f64(temp_a1, temp_b1);
                let y2 = vcaddq_rot90_f64(temp_a2, temp_b2);
                let y3 = vcaddq_rot270_f64(temp_a2, temp_b2);
                let y4 = vcaddq_rot270_f64(temp_a1, temp_b1);

                vst1q_f64(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), y3);
                vst1q_f64(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for NeonFcmaButterfly5<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for NeonFcmaButterfly5<f64> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Butterfly5;
    use crate::dft::Dft;
    use rand::Rng;

    #[test]
    fn test_butterfly5_f32() {
        if !std::arch::is_aarch64_feature_detected!("fcma") {
            return;
        }
        for i in 1..6 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix5_reference = Butterfly5::new(FftDirection::Forward);
            let radix5_inv_reference = Butterfly5::new(FftDirection::Inverse);

            let radix_forward = NeonFcmaButterfly5::new(FftDirection::Forward);
            let radix_inverse = NeonFcmaButterfly5::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix5_reference.execute(&mut z_ref).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-5,
                        "a_re {} != b_re {} for size {}, reference failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-5,
                        "a_im {} != b_im {} for size {}, reference failed at {idx}",
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

            input = input.iter().map(|&x| x * (1.0 / 5f32)).collect();

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
    fn test_butterfly5_f64() {
        if !std::arch::is_aarch64_feature_detected!("fcma") {
            return;
        }
        for i in 1..6 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaButterfly5::new(FftDirection::Forward);
            let radix_inverse = NeonFcmaButterfly5::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 5f64)).collect();

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
    fn test_butterfly5_fcma_out_of_place_f64() {
        if !std::arch::is_aarch64_feature_detected!("fcma") {
            return;
        }
        for i in 1..6 {
            let size = 5usize.pow(i);
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
            let radix_forward = NeonFcmaButterfly5::new(FftDirection::Forward);
            let radix_inverse = NeonFcmaButterfly5::new(FftDirection::Inverse);

            let reference_dft = Dft::new(5, FftDirection::Forward).unwrap();
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

            input = input.iter().map(|&x| x * (1.0 / 5f64)).collect();

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
    fn test_butterfly5_fcma_out_of_place_f32() {
        if !std::arch::is_aarch64_feature_detected!("fcma") {
            return;
        }
        for i in 1..6 {
            let size = 5usize.pow(i);
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
            let radix_forward = NeonFcmaButterfly5::new(FftDirection::Forward);
            let radix_inverse = NeonFcmaButterfly5::new(FftDirection::Inverse);

            let reference_dft = Dft::new(5, FftDirection::Forward).unwrap();
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
                        (a.re - b.re).abs() < 1e-5,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-5,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse
                .execute_out_of_place(&out_of_place, &mut input)
                .unwrap();

            input = input.iter().map(|&x| x * (1.0 / 5f32)).collect();

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
