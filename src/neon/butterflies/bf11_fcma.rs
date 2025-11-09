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
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaButterfly11<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonFcmaButterfly11<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 11, fft_direction),
            twiddle2: compute_twiddle(2, 11, fft_direction),
            twiddle3: compute_twiddle(3, 11, fft_direction),
            twiddle4: compute_twiddle(4, 11, fft_direction),
            twiddle5: compute_twiddle(5, 11, fft_direction),
        }
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly11<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        11
    }
}

impl NeonFcmaButterfly11<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 11 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(11) {
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

                let y00 = u0;
                let (x1p10, x1m10) = NeonButterfly::butterfly2_f64(u1, u10);
                let x1m10 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m10);
                let y00 = vaddq_f64(y00, x1p10);
                let (x2p9, x2m9) = NeonButterfly::butterfly2_f64(u2, u9);
                let x2m9 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m9);
                let y00 = vaddq_f64(y00, x2p9);
                let (x3p8, x3m8) = NeonButterfly::butterfly2_f64(u3, u8);
                let x3m8 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m8);
                let y00 = vaddq_f64(y00, x3p8);
                let (x4p7, x4m7) = NeonButterfly::butterfly2_f64(u4, u7);
                let x4m7 = vcaddq_rot90_f64(vdupq_n_f64(0.), x4m7);
                let y00 = vaddq_f64(y00, x4p7);
                let (x5p6, x5m6) = NeonButterfly::butterfly2_f64(u5, u6);
                let x5m6 = vcaddq_rot90_f64(vdupq_n_f64(0.), x5m6);
                let y00 = vaddq_f64(y00, x5p6);

                let m0110a = vfmaq_n_f64(u0, x1p10, self.twiddle1.re);
                let m0110a = vfmaq_n_f64(m0110a, x2p9, self.twiddle2.re);
                let m0110a = vfmaq_n_f64(m0110a, x3p8, self.twiddle3.re);
                let m0110a = vfmaq_n_f64(m0110a, x4p7, self.twiddle4.re);
                let m0110a = vfmaq_n_f64(m0110a, x5p6, self.twiddle5.re);
                let m0110b = vmulq_n_f64(x1m10, self.twiddle1.im);
                let m0110b = vfmaq_n_f64(m0110b, x2m9, self.twiddle2.im);
                let m0110b = vfmaq_n_f64(m0110b, x3m8, self.twiddle3.im);
                let m0110b = vfmaq_n_f64(m0110b, x4m7, self.twiddle4.im);
                let m0110b = vfmaq_n_f64(m0110b, x5m6, self.twiddle5.im);
                let (y01, y10) = NeonButterfly::butterfly2_f64(m0110a, m0110b);

                let m0209a = vfmaq_n_f64(u0, x1p10, self.twiddle2.re);
                let m0209a = vfmaq_n_f64(m0209a, x2p9, self.twiddle4.re);
                let m0209a = vfmaq_n_f64(m0209a, x3p8, self.twiddle5.re);
                let m0209a = vfmaq_n_f64(m0209a, x4p7, self.twiddle3.re);
                let m0209a = vfmaq_n_f64(m0209a, x5p6, self.twiddle1.re);
                let m0209b = vmulq_n_f64(x1m10, self.twiddle2.im);
                let m0209b = vfmaq_n_f64(m0209b, x2m9, self.twiddle4.im);
                let m0209b = vfmsq_n_f64(m0209b, x3m8, self.twiddle5.im);
                let m0209b = vfmsq_n_f64(m0209b, x4m7, self.twiddle3.im);
                let m0209b = vfmsq_n_f64(m0209b, x5m6, self.twiddle1.im);
                let (y02, y09) = NeonButterfly::butterfly2_f64(m0209a, m0209b);

                let m0308a = vfmaq_n_f64(u0, x1p10, self.twiddle3.re);
                let m0308a = vfmaq_n_f64(m0308a, x2p9, self.twiddle5.re);
                let m0308a = vfmaq_n_f64(m0308a, x3p8, self.twiddle2.re);
                let m0308a = vfmaq_n_f64(m0308a, x4p7, self.twiddle1.re);
                let m0308a = vfmaq_n_f64(m0308a, x5p6, self.twiddle4.re);
                let m0308b = vmulq_n_f64(x1m10, self.twiddle3.im);
                let m0308b = vfmsq_n_f64(m0308b, x2m9, self.twiddle5.im);
                let m0308b = vfmsq_n_f64(m0308b, x3m8, self.twiddle2.im);
                let m0308b = vfmaq_n_f64(m0308b, x4m7, self.twiddle1.im);
                let m0308b = vfmaq_n_f64(m0308b, x5m6, self.twiddle4.im);
                let (y03, y08) = NeonButterfly::butterfly2_f64(m0308a, m0308b);

                let m0407a = vfmaq_n_f64(u0, x1p10, self.twiddle4.re);
                let m0407a = vfmaq_n_f64(m0407a, x2p9, self.twiddle3.re);
                let m0407a = vfmaq_n_f64(m0407a, x3p8, self.twiddle1.re);
                let m0407a = vfmaq_n_f64(m0407a, x4p7, self.twiddle5.re);
                let m0407a = vfmaq_n_f64(m0407a, x5p6, self.twiddle2.re);
                let m0407b = vmulq_n_f64(x1m10, self.twiddle4.im);
                let m0407b = vfmsq_n_f64(m0407b, x2m9, self.twiddle3.im);
                let m0407b = vfmaq_n_f64(m0407b, x3m8, self.twiddle1.im);
                let m0407b = vfmaq_n_f64(m0407b, x4m7, self.twiddle5.im);
                let m0407b = vfmsq_n_f64(m0407b, x5m6, self.twiddle2.im);
                let (y04, y07) = NeonButterfly::butterfly2_f64(m0407a, m0407b);

                let m0506a = vfmaq_n_f64(u0, x1p10, self.twiddle5.re);
                let m0506a = vfmaq_n_f64(m0506a, x2p9, self.twiddle1.re);
                let m0506a = vfmaq_n_f64(m0506a, x3p8, self.twiddle4.re);
                let m0506a = vfmaq_n_f64(m0506a, x4p7, self.twiddle2.re);
                let m0506a = vfmaq_n_f64(m0506a, x5p6, self.twiddle3.re);
                let m0506b = vmulq_n_f64(x1m10, self.twiddle5.im);
                let m0506b = vfmsq_n_f64(m0506b, x2m9, self.twiddle1.im);
                let m0506b = vfmaq_n_f64(m0506b, x3m8, self.twiddle4.im);
                let m0506b = vfmsq_n_f64(m0506b, x4m7, self.twiddle2.im);
                let m0506b = vfmaq_n_f64(m0506b, x5m6, self.twiddle3.im);
                let (y05, y06) = NeonButterfly::butterfly2_f64(m0506a, m0506b);

                vst1q_f64(chunk.as_mut_ptr().cast(), y00);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y01);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y02);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y03);
                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y04);
                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y05);
                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y06);
                vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y07);
                vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y08);
                vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y09);
                vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);
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
        if src.len() % 11 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 11 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(11).zip(src.chunks_exact(11)) {
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
                let u10 = vld1q_f64(src.get_unchecked(10..).as_ptr().cast());

                let y00 = u0;
                let (x1p10, x1m10) = NeonButterfly::butterfly2_f64(u1, u10);
                let x1m10 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m10);
                let y00 = vaddq_f64(y00, x1p10);
                let (x2p9, x2m9) = NeonButterfly::butterfly2_f64(u2, u9);
                let x2m9 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m9);
                let y00 = vaddq_f64(y00, x2p9);
                let (x3p8, x3m8) = NeonButterfly::butterfly2_f64(u3, u8);
                let x3m8 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m8);
                let y00 = vaddq_f64(y00, x3p8);
                let (x4p7, x4m7) = NeonButterfly::butterfly2_f64(u4, u7);
                let x4m7 = vcaddq_rot90_f64(vdupq_n_f64(0.), x4m7);
                let y00 = vaddq_f64(y00, x4p7);
                let (x5p6, x5m6) = NeonButterfly::butterfly2_f64(u5, u6);
                let x5m6 = vcaddq_rot90_f64(vdupq_n_f64(0.), x5m6);
                let y00 = vaddq_f64(y00, x5p6);

                let m0110a = vfmaq_n_f64(u0, x1p10, self.twiddle1.re);
                let m0110a = vfmaq_n_f64(m0110a, x2p9, self.twiddle2.re);
                let m0110a = vfmaq_n_f64(m0110a, x3p8, self.twiddle3.re);
                let m0110a = vfmaq_n_f64(m0110a, x4p7, self.twiddle4.re);
                let m0110a = vfmaq_n_f64(m0110a, x5p6, self.twiddle5.re);
                let m0110b = vmulq_n_f64(x1m10, self.twiddle1.im);
                let m0110b = vfmaq_n_f64(m0110b, x2m9, self.twiddle2.im);
                let m0110b = vfmaq_n_f64(m0110b, x3m8, self.twiddle3.im);
                let m0110b = vfmaq_n_f64(m0110b, x4m7, self.twiddle4.im);
                let m0110b = vfmaq_n_f64(m0110b, x5m6, self.twiddle5.im);
                let (y01, y10) = NeonButterfly::butterfly2_f64(m0110a, m0110b);

                let m0209a = vfmaq_n_f64(u0, x1p10, self.twiddle2.re);
                let m0209a = vfmaq_n_f64(m0209a, x2p9, self.twiddle4.re);
                let m0209a = vfmaq_n_f64(m0209a, x3p8, self.twiddle5.re);
                let m0209a = vfmaq_n_f64(m0209a, x4p7, self.twiddle3.re);
                let m0209a = vfmaq_n_f64(m0209a, x5p6, self.twiddle1.re);
                let m0209b = vmulq_n_f64(x1m10, self.twiddle2.im);
                let m0209b = vfmaq_n_f64(m0209b, x2m9, self.twiddle4.im);
                let m0209b = vfmsq_n_f64(m0209b, x3m8, self.twiddle5.im);
                let m0209b = vfmsq_n_f64(m0209b, x4m7, self.twiddle3.im);
                let m0209b = vfmsq_n_f64(m0209b, x5m6, self.twiddle1.im);
                let (y02, y09) = NeonButterfly::butterfly2_f64(m0209a, m0209b);

                let m0308a = vfmaq_n_f64(u0, x1p10, self.twiddle3.re);
                let m0308a = vfmaq_n_f64(m0308a, x2p9, self.twiddle5.re);
                let m0308a = vfmaq_n_f64(m0308a, x3p8, self.twiddle2.re);
                let m0308a = vfmaq_n_f64(m0308a, x4p7, self.twiddle1.re);
                let m0308a = vfmaq_n_f64(m0308a, x5p6, self.twiddle4.re);
                let m0308b = vmulq_n_f64(x1m10, self.twiddle3.im);
                let m0308b = vfmsq_n_f64(m0308b, x2m9, self.twiddle5.im);
                let m0308b = vfmsq_n_f64(m0308b, x3m8, self.twiddle2.im);
                let m0308b = vfmaq_n_f64(m0308b, x4m7, self.twiddle1.im);
                let m0308b = vfmaq_n_f64(m0308b, x5m6, self.twiddle4.im);
                let (y03, y08) = NeonButterfly::butterfly2_f64(m0308a, m0308b);

                let m0407a = vfmaq_n_f64(u0, x1p10, self.twiddle4.re);
                let m0407a = vfmaq_n_f64(m0407a, x2p9, self.twiddle3.re);
                let m0407a = vfmaq_n_f64(m0407a, x3p8, self.twiddle1.re);
                let m0407a = vfmaq_n_f64(m0407a, x4p7, self.twiddle5.re);
                let m0407a = vfmaq_n_f64(m0407a, x5p6, self.twiddle2.re);
                let m0407b = vmulq_n_f64(x1m10, self.twiddle4.im);
                let m0407b = vfmsq_n_f64(m0407b, x2m9, self.twiddle3.im);
                let m0407b = vfmaq_n_f64(m0407b, x3m8, self.twiddle1.im);
                let m0407b = vfmaq_n_f64(m0407b, x4m7, self.twiddle5.im);
                let m0407b = vfmsq_n_f64(m0407b, x5m6, self.twiddle2.im);
                let (y04, y07) = NeonButterfly::butterfly2_f64(m0407a, m0407b);

                let m0506a = vfmaq_n_f64(u0, x1p10, self.twiddle5.re);
                let m0506a = vfmaq_n_f64(m0506a, x2p9, self.twiddle1.re);
                let m0506a = vfmaq_n_f64(m0506a, x3p8, self.twiddle4.re);
                let m0506a = vfmaq_n_f64(m0506a, x4p7, self.twiddle2.re);
                let m0506a = vfmaq_n_f64(m0506a, x5p6, self.twiddle3.re);
                let m0506b = vmulq_n_f64(x1m10, self.twiddle5.im);
                let m0506b = vfmsq_n_f64(m0506b, x2m9, self.twiddle1.im);
                let m0506b = vfmaq_n_f64(m0506b, x3m8, self.twiddle4.im);
                let m0506b = vfmsq_n_f64(m0506b, x4m7, self.twiddle2.im);
                let m0506b = vfmaq_n_f64(m0506b, x5m6, self.twiddle3.im);
                let (y05, y06) = NeonButterfly::butterfly2_f64(m0506a, m0506b);

                vst1q_f64(dst.as_mut_ptr().cast(), y00);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), y01);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y02);
                vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), y03);
                vst1q_f64(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y04);
                vst1q_f64(dst.get_unchecked_mut(5..).as_mut_ptr().cast(), y05);
                vst1q_f64(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), y06);
                vst1q_f64(dst.get_unchecked_mut(7..).as_mut_ptr().cast(), y07);
                vst1q_f64(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), y08);
                vst1q_f64(dst.get_unchecked_mut(9..).as_mut_ptr().cast(), y09);
                vst1q_f64(dst.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for NeonFcmaButterfly11<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for NeonFcmaButterfly11<f64> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for NeonFcmaButterfly11<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        11
    }
}

impl NeonFcmaButterfly11<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 11 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(22) {
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

                let u0 = vcombine_f32(vget_low_f32(u0u1), vget_high_f32(u10u11));
                let u1 = vcombine_f32(vget_high_f32(u0u1), vget_low_f32(u12u13));
                let u2 = vcombine_f32(vget_low_f32(u2u3), vget_high_f32(u12u13));
                let u3 = vcombine_f32(vget_high_f32(u2u3), vget_low_f32(u14u15));
                let u4 = vcombine_f32(vget_low_f32(u4u5), vget_high_f32(u14u15));
                let u5 = vcombine_f32(vget_high_f32(u4u5), vget_low_f32(u16u17));
                let u6 = vcombine_f32(vget_low_f32(u6u7), vget_high_f32(u16u17));
                let u7 = vcombine_f32(vget_high_f32(u6u7), vget_low_f32(u18u19));
                let u8 = vcombine_f32(vget_low_f32(u8u9), vget_high_f32(u18u19));
                let u9 = vcombine_f32(vget_high_f32(u8u9), vget_low_f32(u20u21));
                let u10 = vcombine_f32(vget_low_f32(u10u11), vget_high_f32(u20u21));

                let y00 = u0;
                let (x1p10, x1m10) = NeonButterfly::butterfly2_f32(u1, u10);
                let x1m10 = vcaddq_rot90_f32(vdupq_n_f32(0.), x1m10);
                let y00 = vaddq_f32(y00, x1p10);
                let (x2p9, x2m9) = NeonButterfly::butterfly2_f32(u2, u9);
                let x2m9 = vcaddq_rot90_f32(vdupq_n_f32(0.), x2m9);
                let y00 = vaddq_f32(y00, x2p9);
                let (x3p8, x3m8) = NeonButterfly::butterfly2_f32(u3, u8);
                let x3m8 = vcaddq_rot90_f32(vdupq_n_f32(0.), x3m8);
                let y00 = vaddq_f32(y00, x3p8);
                let (x4p7, x4m7) = NeonButterfly::butterfly2_f32(u4, u7);
                let x4m7 = vcaddq_rot90_f32(vdupq_n_f32(0.), x4m7);
                let y00 = vaddq_f32(y00, x4p7);
                let (x5p6, x5m6) = NeonButterfly::butterfly2_f32(u5, u6);
                let x5m6 = vcaddq_rot90_f32(vdupq_n_f32(0.), x5m6);
                let y00 = vaddq_f32(y00, x5p6);

                let m0110a = vfmaq_n_f32(u0, x1p10, self.twiddle1.re);
                let m0110a = vfmaq_n_f32(m0110a, x2p9, self.twiddle2.re);
                let m0110a = vfmaq_n_f32(m0110a, x3p8, self.twiddle3.re);
                let m0110a = vfmaq_n_f32(m0110a, x4p7, self.twiddle4.re);
                let m0110a = vfmaq_n_f32(m0110a, x5p6, self.twiddle5.re);
                let m0110b = vmulq_n_f32(x1m10, self.twiddle1.im);
                let m0110b = vfmaq_n_f32(m0110b, x2m9, self.twiddle2.im);
                let m0110b = vfmaq_n_f32(m0110b, x3m8, self.twiddle3.im);
                let m0110b = vfmaq_n_f32(m0110b, x4m7, self.twiddle4.im);
                let m0110b = vfmaq_n_f32(m0110b, x5m6, self.twiddle5.im);
                let (y01, y10) = NeonButterfly::butterfly2_f32(m0110a, m0110b);

                let m0209a = vfmaq_n_f32(u0, x1p10, self.twiddle2.re);
                let m0209a = vfmaq_n_f32(m0209a, x2p9, self.twiddle4.re);
                let m0209a = vfmaq_n_f32(m0209a, x3p8, self.twiddle5.re);
                let m0209a = vfmaq_n_f32(m0209a, x4p7, self.twiddle3.re);
                let m0209a = vfmaq_n_f32(m0209a, x5p6, self.twiddle1.re);
                let m0209b = vmulq_n_f32(x1m10, self.twiddle2.im);
                let m0209b = vfmaq_n_f32(m0209b, x2m9, self.twiddle4.im);
                let m0209b = vfmsq_n_f32(m0209b, x3m8, self.twiddle5.im);
                let m0209b = vfmsq_n_f32(m0209b, x4m7, self.twiddle3.im);
                let m0209b = vfmsq_n_f32(m0209b, x5m6, self.twiddle1.im);
                let (y02, y09) = NeonButterfly::butterfly2_f32(m0209a, m0209b);

                let m0308a = vfmaq_n_f32(u0, x1p10, self.twiddle3.re);
                let m0308a = vfmaq_n_f32(m0308a, x2p9, self.twiddle5.re);
                let m0308a = vfmaq_n_f32(m0308a, x3p8, self.twiddle2.re);
                let m0308a = vfmaq_n_f32(m0308a, x4p7, self.twiddle1.re);
                let m0308a = vfmaq_n_f32(m0308a, x5p6, self.twiddle4.re);
                let m0308b = vmulq_n_f32(x1m10, self.twiddle3.im);
                let m0308b = vfmsq_n_f32(m0308b, x2m9, self.twiddle5.im);
                let m0308b = vfmsq_n_f32(m0308b, x3m8, self.twiddle2.im);
                let m0308b = vfmaq_n_f32(m0308b, x4m7, self.twiddle1.im);
                let m0308b = vfmaq_n_f32(m0308b, x5m6, self.twiddle4.im);
                let (y03, y08) = NeonButterfly::butterfly2_f32(m0308a, m0308b);

                let m0407a = vfmaq_n_f32(u0, x1p10, self.twiddle4.re);
                let m0407a = vfmaq_n_f32(m0407a, x2p9, self.twiddle3.re);
                let m0407a = vfmaq_n_f32(m0407a, x3p8, self.twiddle1.re);
                let m0407a = vfmaq_n_f32(m0407a, x4p7, self.twiddle5.re);
                let m0407a = vfmaq_n_f32(m0407a, x5p6, self.twiddle2.re);
                let m0407b = vmulq_n_f32(x1m10, self.twiddle4.im);
                let m0407b = vfmsq_n_f32(m0407b, x2m9, self.twiddle3.im);
                let m0407b = vfmaq_n_f32(m0407b, x3m8, self.twiddle1.im);
                let m0407b = vfmaq_n_f32(m0407b, x4m7, self.twiddle5.im);
                let m0407b = vfmsq_n_f32(m0407b, x5m6, self.twiddle2.im);
                let (y04, y07) = NeonButterfly::butterfly2_f32(m0407a, m0407b);

                let m0506a = vfmaq_n_f32(u0, x1p10, self.twiddle5.re);
                let m0506a = vfmaq_n_f32(m0506a, x2p9, self.twiddle1.re);
                let m0506a = vfmaq_n_f32(m0506a, x3p8, self.twiddle4.re);
                let m0506a = vfmaq_n_f32(m0506a, x4p7, self.twiddle2.re);
                let m0506a = vfmaq_n_f32(m0506a, x5p6, self.twiddle3.re);
                let m0506b = vmulq_n_f32(x1m10, self.twiddle5.im);
                let m0506b = vfmsq_n_f32(m0506b, x2m9, self.twiddle1.im);
                let m0506b = vfmaq_n_f32(m0506b, x3m8, self.twiddle4.im);
                let m0506b = vfmsq_n_f32(m0506b, x4m7, self.twiddle2.im);
                let m0506b = vfmaq_n_f32(m0506b, x5m6, self.twiddle3.im);
                let (y05, y06) = NeonButterfly::butterfly2_f32(m0506a, m0506b);

                let qy0 = vcombine_f32(vget_low_f32(y00), vget_low_f32(y01));
                let qy1 = vcombine_f32(vget_low_f32(y02), vget_low_f32(y03));
                let qy2 = vcombine_f32(vget_low_f32(y04), vget_low_f32(y05));
                let qy3 = vcombine_f32(vget_low_f32(y06), vget_low_f32(y07));
                let qy4 = vcombine_f32(vget_low_f32(y08), vget_low_f32(y09));
                let qy5 = vcombine_f32(vget_low_f32(y10), vget_high_f32(y00));
                let qy6 = vcombine_f32(vget_high_f32(y01), vget_high_f32(y02));
                let qy7 = vcombine_f32(vget_high_f32(y03), vget_high_f32(y04));
                let qy8 = vcombine_f32(vget_high_f32(y05), vget_high_f32(y06));
                let qy9 = vcombine_f32(vget_high_f32(y07), vget_high_f32(y08));
                let qy10 = vcombine_f32(vget_high_f32(y09), vget_high_f32(y10));

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
            }

            let rem = in_place.chunks_exact_mut(22).into_remainder();

            for chunk in rem.chunks_exact_mut(11) {
                let u0u1 = vld1q_f32(chunk.as_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10 = vld1_f32(chunk.get_unchecked(10..).as_ptr().cast());

                let u0 = vget_low_f32(u0u1);
                let u1 = vget_high_f32(u0u1);
                let u2 = vget_low_f32(u2u3);
                let u3 = vget_high_f32(u2u3);
                let u4 = vget_low_f32(u4u5);
                let u5 = vget_high_f32(u4u5);
                let u6 = vget_low_f32(u6u7);
                let u7 = vget_high_f32(u6u7);
                let u8 = vget_low_f32(u8u9);
                let u9 = vget_high_f32(u8u9);

                let y00 = u0;
                let (x1p10, x1m10) = NeonButterfly::butterfly2h_f32(u1, u10);
                let x1m10 = vcadd_rot90_f32(vdup_n_f32(0.), x1m10);
                let y00 = vadd_f32(y00, x1p10);
                let (x2p9, x2m9) = NeonButterfly::butterfly2h_f32(u2, u9);
                let x2m9 = vcadd_rot90_f32(vdup_n_f32(0.), x2m9);
                let y00 = vadd_f32(y00, x2p9);
                let (x3p8, x3m8) = NeonButterfly::butterfly2h_f32(u3, u8);
                let x3m8 = vcadd_rot90_f32(vdup_n_f32(0.), x3m8);
                let y00 = vadd_f32(y00, x3p8);
                let (x4p7, x4m7) = NeonButterfly::butterfly2h_f32(u4, u7);
                let x4m7 = vcadd_rot90_f32(vdup_n_f32(0.), x4m7);
                let y00 = vadd_f32(y00, x4p7);
                let (x5p6, x5m6) = NeonButterfly::butterfly2h_f32(u5, u6);
                let x5m6 = vcadd_rot90_f32(vdup_n_f32(0.), x5m6);
                let y00 = vadd_f32(y00, x5p6);

                let m0110a = vfma_n_f32(u0, x1p10, self.twiddle1.re);
                let m0110a = vfma_n_f32(m0110a, x2p9, self.twiddle2.re);
                let m0110a = vfma_n_f32(m0110a, x3p8, self.twiddle3.re);
                let m0110a = vfma_n_f32(m0110a, x4p7, self.twiddle4.re);
                let m0110a = vfma_n_f32(m0110a, x5p6, self.twiddle5.re);
                let m0110b = vmul_n_f32(x1m10, self.twiddle1.im);
                let m0110b = vfma_n_f32(m0110b, x2m9, self.twiddle2.im);
                let m0110b = vfma_n_f32(m0110b, x3m8, self.twiddle3.im);
                let m0110b = vfma_n_f32(m0110b, x4m7, self.twiddle4.im);
                let m0110b = vfma_n_f32(m0110b, x5m6, self.twiddle5.im);
                let (y01, y10) = NeonButterfly::butterfly2h_f32(m0110a, m0110b);

                let m0209a = vfma_n_f32(u0, x1p10, self.twiddle2.re);
                let m0209a = vfma_n_f32(m0209a, x2p9, self.twiddle4.re);
                let m0209a = vfma_n_f32(m0209a, x3p8, self.twiddle5.re);
                let m0209a = vfma_n_f32(m0209a, x4p7, self.twiddle3.re);
                let m0209a = vfma_n_f32(m0209a, x5p6, self.twiddle1.re);
                let m0209b = vmul_n_f32(x1m10, self.twiddle2.im);
                let m0209b = vfma_n_f32(m0209b, x2m9, self.twiddle4.im);
                let m0209b = vfms_n_f32(m0209b, x3m8, self.twiddle5.im);
                let m0209b = vfms_n_f32(m0209b, x4m7, self.twiddle3.im);
                let m0209b = vfms_n_f32(m0209b, x5m6, self.twiddle1.im);
                let (y02, y09) = NeonButterfly::butterfly2h_f32(m0209a, m0209b);

                let m0308a = vfma_n_f32(u0, x1p10, self.twiddle3.re);
                let m0308a = vfma_n_f32(m0308a, x2p9, self.twiddle5.re);
                let m0308a = vfma_n_f32(m0308a, x3p8, self.twiddle2.re);
                let m0308a = vfma_n_f32(m0308a, x4p7, self.twiddle1.re);
                let m0308a = vfma_n_f32(m0308a, x5p6, self.twiddle4.re);
                let m0308b = vmul_n_f32(x1m10, self.twiddle3.im);
                let m0308b = vfms_n_f32(m0308b, x2m9, self.twiddle5.im);
                let m0308b = vfms_n_f32(m0308b, x3m8, self.twiddle2.im);
                let m0308b = vfma_n_f32(m0308b, x4m7, self.twiddle1.im);
                let m0308b = vfma_n_f32(m0308b, x5m6, self.twiddle4.im);
                let (y03, y08) = NeonButterfly::butterfly2h_f32(m0308a, m0308b);

                let m0407a = vfma_n_f32(u0, x1p10, self.twiddle4.re);
                let m0407a = vfma_n_f32(m0407a, x2p9, self.twiddle3.re);
                let m0407a = vfma_n_f32(m0407a, x3p8, self.twiddle1.re);
                let m0407a = vfma_n_f32(m0407a, x4p7, self.twiddle5.re);
                let m0407a = vfma_n_f32(m0407a, x5p6, self.twiddle2.re);
                let m0407b = vmul_n_f32(x1m10, self.twiddle4.im);
                let m0407b = vfms_n_f32(m0407b, x2m9, self.twiddle3.im);
                let m0407b = vfma_n_f32(m0407b, x3m8, self.twiddle1.im);
                let m0407b = vfma_n_f32(m0407b, x4m7, self.twiddle5.im);
                let m0407b = vfms_n_f32(m0407b, x5m6, self.twiddle2.im);
                let (y04, y07) = NeonButterfly::butterfly2h_f32(m0407a, m0407b);

                let m0506a = vfma_n_f32(u0, x1p10, self.twiddle5.re);
                let m0506a = vfma_n_f32(m0506a, x2p9, self.twiddle1.re);
                let m0506a = vfma_n_f32(m0506a, x3p8, self.twiddle4.re);
                let m0506a = vfma_n_f32(m0506a, x4p7, self.twiddle2.re);
                let m0506a = vfma_n_f32(m0506a, x5p6, self.twiddle3.re);
                let m0506b = vmul_n_f32(x1m10, self.twiddle5.im);
                let m0506b = vfms_n_f32(m0506b, x2m9, self.twiddle1.im);
                let m0506b = vfma_n_f32(m0506b, x3m8, self.twiddle4.im);
                let m0506b = vfms_n_f32(m0506b, x4m7, self.twiddle2.im);
                let m0506b = vfma_n_f32(m0506b, x5m6, self.twiddle3.im);
                let (y05, y06) = NeonButterfly::butterfly2h_f32(m0506a, m0506b);

                vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y00, y01));
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y02, y03),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y04, y05),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y06, y07),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y08, y09),
                );
                vst1_f32(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_of_place_f32(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 11 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 11 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(22).zip(src.chunks_exact(22)) {
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
                let u20u21 = vld1q_f32(src.get_unchecked(20..).as_ptr().cast());

                let u0 = vcombine_f32(vget_low_f32(u0u1), vget_high_f32(u10u11));
                let u1 = vcombine_f32(vget_high_f32(u0u1), vget_low_f32(u12u13));
                let u2 = vcombine_f32(vget_low_f32(u2u3), vget_high_f32(u12u13));
                let u3 = vcombine_f32(vget_high_f32(u2u3), vget_low_f32(u14u15));
                let u4 = vcombine_f32(vget_low_f32(u4u5), vget_high_f32(u14u15));
                let u5 = vcombine_f32(vget_high_f32(u4u5), vget_low_f32(u16u17));
                let u6 = vcombine_f32(vget_low_f32(u6u7), vget_high_f32(u16u17));
                let u7 = vcombine_f32(vget_high_f32(u6u7), vget_low_f32(u18u19));
                let u8 = vcombine_f32(vget_low_f32(u8u9), vget_high_f32(u18u19));
                let u9 = vcombine_f32(vget_high_f32(u8u9), vget_low_f32(u20u21));
                let u10 = vcombine_f32(vget_low_f32(u10u11), vget_high_f32(u20u21));

                let y00 = u0;
                let (x1p10, x1m10) = NeonButterfly::butterfly2_f32(u1, u10);
                let x1m10 = vcaddq_rot90_f32(vdupq_n_f32(0.), x1m10);
                let y00 = vaddq_f32(y00, x1p10);
                let (x2p9, x2m9) = NeonButterfly::butterfly2_f32(u2, u9);
                let x2m9 = vcaddq_rot90_f32(vdupq_n_f32(0.), x2m9);
                let y00 = vaddq_f32(y00, x2p9);
                let (x3p8, x3m8) = NeonButterfly::butterfly2_f32(u3, u8);
                let x3m8 = vcaddq_rot90_f32(vdupq_n_f32(0.), x3m8);
                let y00 = vaddq_f32(y00, x3p8);
                let (x4p7, x4m7) = NeonButterfly::butterfly2_f32(u4, u7);
                let x4m7 = vcaddq_rot90_f32(vdupq_n_f32(0.), x4m7);
                let y00 = vaddq_f32(y00, x4p7);
                let (x5p6, x5m6) = NeonButterfly::butterfly2_f32(u5, u6);
                let x5m6 = vcaddq_rot90_f32(vdupq_n_f32(0.), x5m6);
                let y00 = vaddq_f32(y00, x5p6);

                let m0110a = vfmaq_n_f32(u0, x1p10, self.twiddle1.re);
                let m0110a = vfmaq_n_f32(m0110a, x2p9, self.twiddle2.re);
                let m0110a = vfmaq_n_f32(m0110a, x3p8, self.twiddle3.re);
                let m0110a = vfmaq_n_f32(m0110a, x4p7, self.twiddle4.re);
                let m0110a = vfmaq_n_f32(m0110a, x5p6, self.twiddle5.re);
                let m0110b = vmulq_n_f32(x1m10, self.twiddle1.im);
                let m0110b = vfmaq_n_f32(m0110b, x2m9, self.twiddle2.im);
                let m0110b = vfmaq_n_f32(m0110b, x3m8, self.twiddle3.im);
                let m0110b = vfmaq_n_f32(m0110b, x4m7, self.twiddle4.im);
                let m0110b = vfmaq_n_f32(m0110b, x5m6, self.twiddle5.im);
                let (y01, y10) = NeonButterfly::butterfly2_f32(m0110a, m0110b);

                let m0209a = vfmaq_n_f32(u0, x1p10, self.twiddle2.re);
                let m0209a = vfmaq_n_f32(m0209a, x2p9, self.twiddle4.re);
                let m0209a = vfmaq_n_f32(m0209a, x3p8, self.twiddle5.re);
                let m0209a = vfmaq_n_f32(m0209a, x4p7, self.twiddle3.re);
                let m0209a = vfmaq_n_f32(m0209a, x5p6, self.twiddle1.re);
                let m0209b = vmulq_n_f32(x1m10, self.twiddle2.im);
                let m0209b = vfmaq_n_f32(m0209b, x2m9, self.twiddle4.im);
                let m0209b = vfmsq_n_f32(m0209b, x3m8, self.twiddle5.im);
                let m0209b = vfmsq_n_f32(m0209b, x4m7, self.twiddle3.im);
                let m0209b = vfmsq_n_f32(m0209b, x5m6, self.twiddle1.im);
                let (y02, y09) = NeonButterfly::butterfly2_f32(m0209a, m0209b);

                let m0308a = vfmaq_n_f32(u0, x1p10, self.twiddle3.re);
                let m0308a = vfmaq_n_f32(m0308a, x2p9, self.twiddle5.re);
                let m0308a = vfmaq_n_f32(m0308a, x3p8, self.twiddle2.re);
                let m0308a = vfmaq_n_f32(m0308a, x4p7, self.twiddle1.re);
                let m0308a = vfmaq_n_f32(m0308a, x5p6, self.twiddle4.re);
                let m0308b = vmulq_n_f32(x1m10, self.twiddle3.im);
                let m0308b = vfmsq_n_f32(m0308b, x2m9, self.twiddle5.im);
                let m0308b = vfmsq_n_f32(m0308b, x3m8, self.twiddle2.im);
                let m0308b = vfmaq_n_f32(m0308b, x4m7, self.twiddle1.im);
                let m0308b = vfmaq_n_f32(m0308b, x5m6, self.twiddle4.im);
                let (y03, y08) = NeonButterfly::butterfly2_f32(m0308a, m0308b);

                let m0407a = vfmaq_n_f32(u0, x1p10, self.twiddle4.re);
                let m0407a = vfmaq_n_f32(m0407a, x2p9, self.twiddle3.re);
                let m0407a = vfmaq_n_f32(m0407a, x3p8, self.twiddle1.re);
                let m0407a = vfmaq_n_f32(m0407a, x4p7, self.twiddle5.re);
                let m0407a = vfmaq_n_f32(m0407a, x5p6, self.twiddle2.re);
                let m0407b = vmulq_n_f32(x1m10, self.twiddle4.im);
                let m0407b = vfmsq_n_f32(m0407b, x2m9, self.twiddle3.im);
                let m0407b = vfmaq_n_f32(m0407b, x3m8, self.twiddle1.im);
                let m0407b = vfmaq_n_f32(m0407b, x4m7, self.twiddle5.im);
                let m0407b = vfmsq_n_f32(m0407b, x5m6, self.twiddle2.im);
                let (y04, y07) = NeonButterfly::butterfly2_f32(m0407a, m0407b);

                let m0506a = vfmaq_n_f32(u0, x1p10, self.twiddle5.re);
                let m0506a = vfmaq_n_f32(m0506a, x2p9, self.twiddle1.re);
                let m0506a = vfmaq_n_f32(m0506a, x3p8, self.twiddle4.re);
                let m0506a = vfmaq_n_f32(m0506a, x4p7, self.twiddle2.re);
                let m0506a = vfmaq_n_f32(m0506a, x5p6, self.twiddle3.re);
                let m0506b = vmulq_n_f32(x1m10, self.twiddle5.im);
                let m0506b = vfmsq_n_f32(m0506b, x2m9, self.twiddle1.im);
                let m0506b = vfmaq_n_f32(m0506b, x3m8, self.twiddle4.im);
                let m0506b = vfmsq_n_f32(m0506b, x4m7, self.twiddle2.im);
                let m0506b = vfmaq_n_f32(m0506b, x5m6, self.twiddle3.im);
                let (y05, y06) = NeonButterfly::butterfly2_f32(m0506a, m0506b);

                let qy0 = vcombine_f32(vget_low_f32(y00), vget_low_f32(y01));
                let qy1 = vcombine_f32(vget_low_f32(y02), vget_low_f32(y03));
                let qy2 = vcombine_f32(vget_low_f32(y04), vget_low_f32(y05));
                let qy3 = vcombine_f32(vget_low_f32(y06), vget_low_f32(y07));
                let qy4 = vcombine_f32(vget_low_f32(y08), vget_low_f32(y09));
                let qy5 = vcombine_f32(vget_low_f32(y10), vget_high_f32(y00));
                let qy6 = vcombine_f32(vget_high_f32(y01), vget_high_f32(y02));
                let qy7 = vcombine_f32(vget_high_f32(y03), vget_high_f32(y04));
                let qy8 = vcombine_f32(vget_high_f32(y05), vget_high_f32(y06));
                let qy9 = vcombine_f32(vget_high_f32(y07), vget_high_f32(y08));
                let qy10 = vcombine_f32(vget_high_f32(y09), vget_high_f32(y10));

                vst1q_f32(dst.as_mut_ptr().cast(), qy0);
                vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), qy1);
                vst1q_f32(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), qy2);
                vst1q_f32(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), qy3);
                vst1q_f32(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), qy4);
                vst1q_f32(dst.get_unchecked_mut(10..).as_mut_ptr().cast(), qy5);
                vst1q_f32(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), qy6);
                vst1q_f32(dst.get_unchecked_mut(14..).as_mut_ptr().cast(), qy7);
                vst1q_f32(dst.get_unchecked_mut(16..).as_mut_ptr().cast(), qy8);
                vst1q_f32(dst.get_unchecked_mut(18..).as_mut_ptr().cast(), qy9);
                vst1q_f32(dst.get_unchecked_mut(20..).as_mut_ptr().cast(), qy10);
            }

            let rem_src = src.chunks_exact(22).remainder();
            let rem_dst = dst.chunks_exact_mut(22).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(11).zip(rem_src.chunks_exact(11)) {
                let u0u1 = vld1q_f32(src.as_ptr().cast());
                let u2u3 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(src.get_unchecked(8..).as_ptr().cast());
                let u10 = vld1_f32(src.get_unchecked(10..).as_ptr().cast());

                let u0 = vget_low_f32(u0u1);
                let u1 = vget_high_f32(u0u1);
                let u2 = vget_low_f32(u2u3);
                let u3 = vget_high_f32(u2u3);
                let u4 = vget_low_f32(u4u5);
                let u5 = vget_high_f32(u4u5);
                let u6 = vget_low_f32(u6u7);
                let u7 = vget_high_f32(u6u7);
                let u8 = vget_low_f32(u8u9);
                let u9 = vget_high_f32(u8u9);

                let y00 = u0;
                let (x1p10, x1m10) = NeonButterfly::butterfly2h_f32(u1, u10);
                let x1m10 = vcadd_rot90_f32(vdup_n_f32(0.), x1m10);
                let y00 = vadd_f32(y00, x1p10);
                let (x2p9, x2m9) = NeonButterfly::butterfly2h_f32(u2, u9);
                let x2m9 = vcadd_rot90_f32(vdup_n_f32(0.), x2m9);
                let y00 = vadd_f32(y00, x2p9);
                let (x3p8, x3m8) = NeonButterfly::butterfly2h_f32(u3, u8);
                let x3m8 = vcadd_rot90_f32(vdup_n_f32(0.), x3m8);
                let y00 = vadd_f32(y00, x3p8);
                let (x4p7, x4m7) = NeonButterfly::butterfly2h_f32(u4, u7);
                let x4m7 = vcadd_rot90_f32(vdup_n_f32(0.), x4m7);
                let y00 = vadd_f32(y00, x4p7);
                let (x5p6, x5m6) = NeonButterfly::butterfly2h_f32(u5, u6);
                let x5m6 = vcadd_rot90_f32(vdup_n_f32(0.), x5m6);
                let y00 = vadd_f32(y00, x5p6);

                let m0110a = vfma_n_f32(u0, x1p10, self.twiddle1.re);
                let m0110a = vfma_n_f32(m0110a, x2p9, self.twiddle2.re);
                let m0110a = vfma_n_f32(m0110a, x3p8, self.twiddle3.re);
                let m0110a = vfma_n_f32(m0110a, x4p7, self.twiddle4.re);
                let m0110a = vfma_n_f32(m0110a, x5p6, self.twiddle5.re);
                let m0110b = vmul_n_f32(x1m10, self.twiddle1.im);
                let m0110b = vfma_n_f32(m0110b, x2m9, self.twiddle2.im);
                let m0110b = vfma_n_f32(m0110b, x3m8, self.twiddle3.im);
                let m0110b = vfma_n_f32(m0110b, x4m7, self.twiddle4.im);
                let m0110b = vfma_n_f32(m0110b, x5m6, self.twiddle5.im);
                let (y01, y10) = NeonButterfly::butterfly2h_f32(m0110a, m0110b);

                let m0209a = vfma_n_f32(u0, x1p10, self.twiddle2.re);
                let m0209a = vfma_n_f32(m0209a, x2p9, self.twiddle4.re);
                let m0209a = vfma_n_f32(m0209a, x3p8, self.twiddle5.re);
                let m0209a = vfma_n_f32(m0209a, x4p7, self.twiddle3.re);
                let m0209a = vfma_n_f32(m0209a, x5p6, self.twiddle1.re);
                let m0209b = vmul_n_f32(x1m10, self.twiddle2.im);
                let m0209b = vfma_n_f32(m0209b, x2m9, self.twiddle4.im);
                let m0209b = vfms_n_f32(m0209b, x3m8, self.twiddle5.im);
                let m0209b = vfms_n_f32(m0209b, x4m7, self.twiddle3.im);
                let m0209b = vfms_n_f32(m0209b, x5m6, self.twiddle1.im);
                let (y02, y09) = NeonButterfly::butterfly2h_f32(m0209a, m0209b);

                let m0308a = vfma_n_f32(u0, x1p10, self.twiddle3.re);
                let m0308a = vfma_n_f32(m0308a, x2p9, self.twiddle5.re);
                let m0308a = vfma_n_f32(m0308a, x3p8, self.twiddle2.re);
                let m0308a = vfma_n_f32(m0308a, x4p7, self.twiddle1.re);
                let m0308a = vfma_n_f32(m0308a, x5p6, self.twiddle4.re);
                let m0308b = vmul_n_f32(x1m10, self.twiddle3.im);
                let m0308b = vfms_n_f32(m0308b, x2m9, self.twiddle5.im);
                let m0308b = vfms_n_f32(m0308b, x3m8, self.twiddle2.im);
                let m0308b = vfma_n_f32(m0308b, x4m7, self.twiddle1.im);
                let m0308b = vfma_n_f32(m0308b, x5m6, self.twiddle4.im);
                let (y03, y08) = NeonButterfly::butterfly2h_f32(m0308a, m0308b);

                let m0407a = vfma_n_f32(u0, x1p10, self.twiddle4.re);
                let m0407a = vfma_n_f32(m0407a, x2p9, self.twiddle3.re);
                let m0407a = vfma_n_f32(m0407a, x3p8, self.twiddle1.re);
                let m0407a = vfma_n_f32(m0407a, x4p7, self.twiddle5.re);
                let m0407a = vfma_n_f32(m0407a, x5p6, self.twiddle2.re);
                let m0407b = vmul_n_f32(x1m10, self.twiddle4.im);
                let m0407b = vfms_n_f32(m0407b, x2m9, self.twiddle3.im);
                let m0407b = vfma_n_f32(m0407b, x3m8, self.twiddle1.im);
                let m0407b = vfma_n_f32(m0407b, x4m7, self.twiddle5.im);
                let m0407b = vfms_n_f32(m0407b, x5m6, self.twiddle2.im);
                let (y04, y07) = NeonButterfly::butterfly2h_f32(m0407a, m0407b);

                let m0506a = vfma_n_f32(u0, x1p10, self.twiddle5.re);
                let m0506a = vfma_n_f32(m0506a, x2p9, self.twiddle1.re);
                let m0506a = vfma_n_f32(m0506a, x3p8, self.twiddle4.re);
                let m0506a = vfma_n_f32(m0506a, x4p7, self.twiddle2.re);
                let m0506a = vfma_n_f32(m0506a, x5p6, self.twiddle3.re);
                let m0506b = vmul_n_f32(x1m10, self.twiddle5.im);
                let m0506b = vfms_n_f32(m0506b, x2m9, self.twiddle1.im);
                let m0506b = vfma_n_f32(m0506b, x3m8, self.twiddle4.im);
                let m0506b = vfms_n_f32(m0506b, x4m7, self.twiddle2.im);
                let m0506b = vfma_n_f32(m0506b, x5m6, self.twiddle3.im);
                let (y05, y06) = NeonButterfly::butterfly2h_f32(m0506a, m0506b);

                vst1q_f32(dst.as_mut_ptr().cast(), vcombine_f32(y00, y01));
                vst1q_f32(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y02, y03),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y04, y05),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y06, y07),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y08, y09),
                );
                vst1_f32(dst.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for NeonFcmaButterfly11<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for NeonFcmaButterfly11<f32> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_fcma_butterfly!(
        test_fcma_neon_butterfly11,
        f32,
        NeonFcmaButterfly11,
        11,
        1e-5
    );
    test_fcma_butterfly!(
        test_fcma_neon_butterfly11_f64,
        f64,
        NeonFcmaButterfly11,
        11,
        1e-7
    );
    test_oof_fcma_butterfly!(test_oof_butterfly11, f32, NeonFcmaButterfly11, 11, 1e-5);
    test_oof_fcma_butterfly!(test_oof_butterfly11_f64, f64, NeonFcmaButterfly11, 11, 1e-9);
}
