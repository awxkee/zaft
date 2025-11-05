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
use std::marker::PhantomData;

pub(crate) struct NeonButterfly3<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
    twiddle: Complex<T>,
    tw1: [T; 4],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonButterfly3<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle = compute_twiddle(1, 3, fft_direction);
        Self {
            direction: fft_direction,
            phantom_data: PhantomData,
            twiddle,
            tw1: [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im],
        }
    }
}

impl FftExecutor<f64> for NeonButterfly3<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), 3));
        }

        unsafe {
            let twiddle_re = vdupq_n_f64(self.twiddle.re);
            let tw1 = vld1q_f64(self.tw1.as_ptr());

            for chunk in in_place.chunks_exact_mut(3) {
                let u0 = vld1q_f64(chunk.get_unchecked(0..).as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());

                let xp = vaddq_f64(u1, u2);
                let xn = vsubq_f64(u1, u2);
                let sum = vaddq_f64(u0, xp);

                let w_1 = vfmaq_f64(u0, twiddle_re, xp);

                let xn_rot = vextq_f64::<1>(xn, xn);

                let y0 = sum;
                let y1 = vfmaq_f64(w_1, tw1, xn_rot);
                let y2 = vfmsq_f64(w_1, tw1, xn_rot);

                vst1q_f64(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        3
    }
}

impl FftExecutorOutOfPlace<f64> for NeonButterfly3<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), 3));
        }
        if dst.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), 3));
        }

        unsafe {
            let twiddle_re = vdupq_n_f64(self.twiddle.re);
            let tw1 = vld1q_f64(self.tw1.as_ptr());

            for (dst, src) in dst.chunks_exact_mut(3).zip(src.chunks_exact(3)) {
                let u0 = vld1q_f64(src.get_unchecked(0..).as_ptr().cast());
                let u1 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());

                let xp = vaddq_f64(u1, u2);
                let xn = vsubq_f64(u1, u2);
                let sum = vaddq_f64(u0, xp);

                let w_1 = vfmaq_f64(u0, twiddle_re, xp);

                let xn_rot = vextq_f64::<1>(xn, xn);

                let y0 = sum;
                let y1 = vfmaq_f64(w_1, tw1, xn_rot);
                let y2 = vfmsq_f64(w_1, tw1, xn_rot);

                vst1q_f64(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for NeonButterfly3<f64> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for NeonButterfly3<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let twiddle_re = vdupq_n_f32(self.twiddle.re);

            let tw1 = vld1q_f32(self.tw1.as_ptr());

            for chunk in in_place.chunks_exact_mut(12) {
                let uz0 = vld3q_f64(chunk.as_ptr().cast());
                let uz1 = vld3q_f64(chunk.get_unchecked(6..).as_ptr().cast());

                let u0 = vreinterpretq_f32_f64(uz0.0);
                let u1 = vreinterpretq_f32_f64(uz0.1);
                let u2 = vreinterpretq_f32_f64(uz0.2);

                let u3 = vreinterpretq_f32_f64(uz1.0);
                let u4 = vreinterpretq_f32_f64(uz1.1);
                let u5 = vreinterpretq_f32_f64(uz1.2);

                let xp0 = vaddq_f32(u1, u2);
                let xn0 = vsubq_f32(u1, u2);
                let sum0 = vaddq_f32(u0, xp0);

                let xp1 = vaddq_f32(u4, u5);
                let xn1 = vsubq_f32(u4, u5);
                let sum1 = vaddq_f32(u3, xp1);

                let w_1 = vfmaq_f32(u0, twiddle_re, xp0);
                let w_2 = vfmaq_f32(u3, twiddle_re, xp1);

                let xn_rot0 = vrev64q_f32(xn0);
                let xn_rot1 = vrev64q_f32(xn1);

                let y0 = sum0;
                let y1 = vfmaq_f32(w_1, tw1, xn_rot0);
                let y2 = vfmsq_f32(w_1, tw1, xn_rot0);

                let y3 = sum1;
                let y4 = vfmaq_f32(w_2, tw1, xn_rot1);
                let y5 = vfmsq_f32(w_2, tw1, xn_rot1);

                vst3q_f64(
                    chunk.as_mut_ptr().cast(),
                    float64x2x3_t(
                        vreinterpretq_f64_f32(y0),
                        vreinterpretq_f64_f32(y1),
                        vreinterpretq_f64_f32(y2),
                    ),
                );
                vst3q_f64(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    float64x2x3_t(
                        vreinterpretq_f64_f32(y3),
                        vreinterpretq_f64_f32(y4),
                        vreinterpretq_f64_f32(y5),
                    ),
                );
            }

            let rem = in_place.chunks_exact_mut(12).into_remainder();

            for chunk in rem.chunks_exact_mut(6) {
                let uz = vld3q_f64(chunk.as_ptr().cast());
                let u0 = vreinterpretq_f32_f64(uz.0);
                let u1 = vreinterpretq_f32_f64(uz.1);
                let u2 = vreinterpretq_f32_f64(uz.2);

                let xp = vaddq_f32(u1, u2);
                let xn = vsubq_f32(u1, u2);
                let sum = vaddq_f32(u0, xp);

                let w_1 = vfmaq_f32(u0, twiddle_re, xp);

                let xn_rot = vrev64q_f32(xn);

                let y0 = sum;
                let y1 = vfmaq_f32(w_1, tw1, xn_rot);
                let y2 = vfmsq_f32(w_1, tw1, xn_rot);

                vst3q_f64(
                    chunk.as_mut_ptr().cast(),
                    float64x2x3_t(
                        vreinterpretq_f64_f32(y0),
                        vreinterpretq_f64_f32(y1),
                        vreinterpretq_f64_f32(y2),
                    ),
                );
            }

            let rem = rem.chunks_exact_mut(6).into_remainder();

            for chunk in rem.chunks_exact_mut(3) {
                let uz0 = vld1q_f32(chunk.get_unchecked(0..).as_ptr().cast());

                let u0 = vget_low_f32(uz0);
                let u1 = vget_high_f32(uz0);
                let u2 = vld1_f32(chunk.get_unchecked(2..).as_ptr().cast());

                let xp = vadd_f32(u1, u2);
                let xn = vsub_f32(u1, u2);
                let sum = vadd_f32(u0, xp);

                let w_1 = vfma_f32(u0, vget_low_f32(twiddle_re), xp);

                let xn_rot = vext_f32::<1>(xn, xn);

                let y0 = sum;
                let y1 = vfma_f32(w_1, vget_low_f32(tw1), xn_rot);
                let y2 = vfms_f32(w_1, vget_low_f32(tw1), xn_rot);

                vst1q_f32(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(y0, y1),
                );
                vst1_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        3
    }
}

impl FftExecutorOutOfPlace<f32> for NeonButterfly3<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let twiddle_re = vdupq_n_f32(self.twiddle.re);

            let tw1 = vld1q_f32(self.tw1.as_ptr());

            for (dst, src) in dst.chunks_exact_mut(12).zip(src.chunks_exact(12)) {
                let uz0 = vld3q_f64(src.as_ptr().cast());
                let uz1 = vld3q_f64(src.get_unchecked(6..).as_ptr().cast());

                let u0 = vreinterpretq_f32_f64(uz0.0);
                let u1 = vreinterpretq_f32_f64(uz0.1);
                let u2 = vreinterpretq_f32_f64(uz0.2);

                let u3 = vreinterpretq_f32_f64(uz1.0);
                let u4 = vreinterpretq_f32_f64(uz1.1);
                let u5 = vreinterpretq_f32_f64(uz1.2);

                let xp0 = vaddq_f32(u1, u2);
                let xn0 = vsubq_f32(u1, u2);
                let sum0 = vaddq_f32(u0, xp0);

                let xp1 = vaddq_f32(u4, u5);
                let xn1 = vsubq_f32(u4, u5);
                let sum1 = vaddq_f32(u3, xp1);

                let w_1 = vfmaq_f32(u0, twiddle_re, xp0);
                let w_2 = vfmaq_f32(u3, twiddle_re, xp1);

                let xn_rot0 = vrev64q_f32(xn0);
                let xn_rot1 = vrev64q_f32(xn1);

                let y0 = sum0;
                let y1 = vfmaq_f32(w_1, tw1, xn_rot0);
                let y2 = vfmsq_f32(w_1, tw1, xn_rot0);

                let y3 = sum1;
                let y4 = vfmaq_f32(w_2, tw1, xn_rot1);
                let y5 = vfmsq_f32(w_2, tw1, xn_rot1);

                vst3q_f64(
                    dst.as_mut_ptr().cast(),
                    float64x2x3_t(
                        vreinterpretq_f64_f32(y0),
                        vreinterpretq_f64_f32(y1),
                        vreinterpretq_f64_f32(y2),
                    ),
                );
                vst3q_f64(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    float64x2x3_t(
                        vreinterpretq_f64_f32(y3),
                        vreinterpretq_f64_f32(y4),
                        vreinterpretq_f64_f32(y5),
                    ),
                );
            }

            let rem_src = src.chunks_exact(12).remainder();
            let rem_dst = dst.chunks_exact_mut(12).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(6).zip(rem_src.chunks_exact(6)) {
                let uz = vld3q_f64(src.as_ptr().cast());
                let u0 = vreinterpretq_f32_f64(uz.0);
                let u1 = vreinterpretq_f32_f64(uz.1);
                let u2 = vreinterpretq_f32_f64(uz.2);

                let xp = vaddq_f32(u1, u2);
                let xn = vsubq_f32(u1, u2);
                let sum = vaddq_f32(u0, xp);

                let w_1 = vfmaq_f32(u0, twiddle_re, xp);

                let xn_rot = vrev64q_f32(xn);

                let y0 = sum;
                let y1 = vfmaq_f32(w_1, tw1, xn_rot);
                let y2 = vfmsq_f32(w_1, tw1, xn_rot);

                vst3q_f64(
                    dst.as_mut_ptr().cast(),
                    float64x2x3_t(
                        vreinterpretq_f64_f32(y0),
                        vreinterpretq_f64_f32(y1),
                        vreinterpretq_f64_f32(y2),
                    ),
                );
            }

            let rem_src = rem_src.chunks_exact(6).remainder();
            let rem_dst = rem_dst.chunks_exact_mut(6).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(3).zip(rem_src.chunks_exact(3)) {
                let uz0 = vld1q_f32(src.get_unchecked(0..).as_ptr().cast());

                let u0 = vget_low_f32(uz0);
                let u1 = vget_high_f32(uz0);
                let u2 = vld1_f32(src.get_unchecked(2..).as_ptr().cast());

                let xp = vadd_f32(u1, u2);
                let xn = vsub_f32(u1, u2);
                let sum = vadd_f32(u0, xp);

                let w_1 = vfma_f32(u0, vget_low_f32(twiddle_re), xp);

                let xn_rot = vext_f32::<1>(xn, xn);

                let y0 = sum;
                let y1 = vfma_f32(w_1, vget_low_f32(tw1), xn_rot);
                let y2 = vfms_f32(w_1, vget_low_f32(tw1), xn_rot);

                vst1q_f32(
                    dst.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(y0, y1),
                );
                vst1_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for NeonButterfly3<f32> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    use rand::Rng;

    test_butterfly!(test_neon_butterfly3, f32, NeonButterfly3, 3, 1e-5);
    test_butterfly!(test_neon_butterfly3_f64, f64, NeonButterfly3, 3, 1e-7);
    test_oof_butterfly!(test_oof_butterfly3, f32, NeonButterfly3, 3, 1e-5);
    test_oof_butterfly!(test_oof_butterfly3_f64, f64, NeonButterfly3, 3, 1e-9);
}
