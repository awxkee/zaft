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

pub(crate) struct NeonFcmaButterfly17<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    twiddle7: Complex<T>,
    twiddle8: Complex<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonFcmaButterfly17<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 17, fft_direction),
            twiddle2: compute_twiddle(2, 17, fft_direction),
            twiddle3: compute_twiddle(3, 17, fft_direction),
            twiddle4: compute_twiddle(4, 17, fft_direction),
            twiddle5: compute_twiddle(5, 17, fft_direction),
            twiddle6: compute_twiddle(6, 17, fft_direction),
            twiddle7: compute_twiddle(7, 17, fft_direction),
            twiddle8: compute_twiddle(8, 17, fft_direction),
        }
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly17<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        17
    }
}

impl NeonFcmaButterfly17<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 17 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(17) {
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
                let u15 = vld1q_f64(chunk.get_unchecked(15..).as_ptr().cast());
                let u16 = vld1q_f64(chunk.get_unchecked(16..).as_ptr().cast());

                let y00 = u0;
                let (x1p16, x1m16) = NeonButterfly::butterfly2_f64(u1, u16);
                let x1m16 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m16);
                let y00 = vaddq_f64(y00, x1p16);
                let (x2p15, x2m15) = NeonButterfly::butterfly2_f64(u2, u15);
                let x2m15 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m15);
                let y00 = vaddq_f64(y00, x2p15);
                let (x3p14, x3m14) = NeonButterfly::butterfly2_f64(u3, u14);
                let x3m14 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m14);
                let y00 = vaddq_f64(y00, x3p14);
                let (x4p13, x4m13) = NeonButterfly::butterfly2_f64(u4, u13);
                let x4m13 = vcaddq_rot90_f64(vdupq_n_f64(0.), x4m13);
                let y00 = vaddq_f64(y00, x4p13);
                let (x5p12, x5m12) = NeonButterfly::butterfly2_f64(u5, u12);
                let x5m12 = vcaddq_rot90_f64(vdupq_n_f64(0.), x5m12);
                let y00 = vaddq_f64(y00, x5p12);
                let (x6p11, x6m11) = NeonButterfly::butterfly2_f64(u6, u11);
                let x6m11 = vcaddq_rot90_f64(vdupq_n_f64(0.), x6m11);
                let y00 = vaddq_f64(y00, x6p11);
                let (x7p10, x7m10) = NeonButterfly::butterfly2_f64(u7, u10);
                let x7m10 = vcaddq_rot90_f64(vdupq_n_f64(0.), x7m10);
                let y00 = vaddq_f64(y00, x7p10);
                let (x8p9, x8m9) = NeonButterfly::butterfly2_f64(u8, u9);
                let x8m9 = vcaddq_rot90_f64(vdupq_n_f64(0.), x8m9);
                let y00 = vaddq_f64(y00, x8p9);

                vst1q_f64(chunk.as_mut_ptr().cast(), y00);

                let m0116a = vfmaq_n_f64(u0, x1p16, self.twiddle1.re);
                let m0116a = vfmaq_n_f64(m0116a, x2p15, self.twiddle2.re);
                let m0116a = vfmaq_n_f64(m0116a, x3p14, self.twiddle3.re);
                let m0116a = vfmaq_n_f64(m0116a, x4p13, self.twiddle4.re);
                let m0116a = vfmaq_n_f64(m0116a, x5p12, self.twiddle5.re);
                let m0116a = vfmaq_n_f64(m0116a, x6p11, self.twiddle6.re);
                let m0116a = vfmaq_n_f64(m0116a, x7p10, self.twiddle7.re);
                let m0116a = vfmaq_n_f64(m0116a, x8p9, self.twiddle8.re);
                let m0116b = vmulq_n_f64(x1m16, self.twiddle1.im);
                let m0116b = vfmaq_n_f64(m0116b, x2m15, self.twiddle2.im);
                let m0116b = vfmaq_n_f64(m0116b, x3m14, self.twiddle3.im);
                let m0116b = vfmaq_n_f64(m0116b, x4m13, self.twiddle4.im);
                let m0116b = vfmaq_n_f64(m0116b, x5m12, self.twiddle5.im);
                let m0116b = vfmaq_n_f64(m0116b, x6m11, self.twiddle6.im);
                let m0116b = vfmaq_n_f64(m0116b, x7m10, self.twiddle7.im);
                let m0116b = vfmaq_n_f64(m0116b, x8m9, self.twiddle8.im);
                let (y01, y16) = NeonButterfly::butterfly2_f64(m0116a, m0116b);

                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y01);
                vst1q_f64(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), y16);

                let m0215a = vfmaq_n_f64(u0, x1p16, self.twiddle2.re);
                let m0215a = vfmaq_n_f64(m0215a, x2p15, self.twiddle4.re);
                let m0215a = vfmaq_n_f64(m0215a, x3p14, self.twiddle6.re);
                let m0215a = vfmaq_n_f64(m0215a, x4p13, self.twiddle8.re);
                let m0215a = vfmaq_n_f64(m0215a, x5p12, self.twiddle7.re);
                let m0215a = vfmaq_n_f64(m0215a, x6p11, self.twiddle5.re);
                let m0215a = vfmaq_n_f64(m0215a, x7p10, self.twiddle3.re);
                let m0215a = vfmaq_n_f64(m0215a, x8p9, self.twiddle1.re);
                let m0215b = vmulq_n_f64(x1m16, self.twiddle2.im);
                let m0215b = vfmaq_n_f64(m0215b, x2m15, self.twiddle4.im);
                let m0215b = vfmaq_n_f64(m0215b, x3m14, self.twiddle6.im);
                let m0215b = vfmaq_n_f64(m0215b, x4m13, self.twiddle8.im);
                let m0215b = vfmsq_n_f64(m0215b, x5m12, self.twiddle7.im);
                let m0215b = vfmsq_n_f64(m0215b, x6m11, self.twiddle5.im);
                let m0215b = vfmsq_n_f64(m0215b, x7m10, self.twiddle3.im);
                let m0215b = vfmsq_n_f64(m0215b, x8m9, self.twiddle1.im);
                let (y02, y15) = NeonButterfly::butterfly2_f64(m0215a, m0215b);

                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y02);
                vst1q_f64(chunk.get_unchecked_mut(15..).as_mut_ptr().cast(), y15);

                let m0314a = vfmaq_n_f64(u0, x1p16, self.twiddle3.re);
                let m0314a = vfmaq_n_f64(m0314a, x2p15, self.twiddle6.re);
                let m0314a = vfmaq_n_f64(m0314a, x3p14, self.twiddle8.re);
                let m0314a = vfmaq_n_f64(m0314a, x4p13, self.twiddle5.re);
                let m0314a = vfmaq_n_f64(m0314a, x5p12, self.twiddle2.re);
                let m0314a = vfmaq_n_f64(m0314a, x6p11, self.twiddle1.re);
                let m0314a = vfmaq_n_f64(m0314a, x7p10, self.twiddle4.re);
                let m0314a = vfmaq_n_f64(m0314a, x8p9, self.twiddle7.re);
                let m0314b = vmulq_n_f64(x1m16, self.twiddle3.im);
                let m0314b = vfmaq_n_f64(m0314b, x2m15, self.twiddle6.im);
                let m0314b = vfmsq_n_f64(m0314b, x3m14, self.twiddle8.im);
                let m0314b = vfmsq_n_f64(m0314b, x4m13, self.twiddle5.im);
                let m0314b = vfmsq_n_f64(m0314b, x5m12, self.twiddle2.im);
                let m0314b = vfmaq_n_f64(m0314b, x6m11, self.twiddle1.im);
                let m0314b = vfmaq_n_f64(m0314b, x7m10, self.twiddle4.im);
                let m0314b = vfmaq_n_f64(m0314b, x8m9, self.twiddle7.im);
                let (y03, y14) = NeonButterfly::butterfly2_f64(m0314a, m0314b);

                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y03);
                vst1q_f64(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y14);

                let m0413a = vfmaq_n_f64(u0, x1p16, self.twiddle4.re);
                let m0413a = vfmaq_n_f64(m0413a, x2p15, self.twiddle8.re);
                let m0413a = vfmaq_n_f64(m0413a, x3p14, self.twiddle5.re);
                let m0413a = vfmaq_n_f64(m0413a, x4p13, self.twiddle1.re);
                let m0413a = vfmaq_n_f64(m0413a, x5p12, self.twiddle3.re);
                let m0413a = vfmaq_n_f64(m0413a, x6p11, self.twiddle7.re);
                let m0413a = vfmaq_n_f64(m0413a, x7p10, self.twiddle6.re);
                let m0413a = vfmaq_n_f64(m0413a, x8p9, self.twiddle2.re);
                let m0413b = vmulq_n_f64(x1m16, self.twiddle4.im);
                let m0413b = vfmaq_n_f64(m0413b, x2m15, self.twiddle8.im);
                let m0413b = vfmsq_n_f64(m0413b, x3m14, self.twiddle5.im);
                let m0413b = vfmsq_n_f64(m0413b, x4m13, self.twiddle1.im);
                let m0413b = vfmaq_n_f64(m0413b, x5m12, self.twiddle3.im);
                let m0413b = vfmaq_n_f64(m0413b, x6m11, self.twiddle7.im);
                let m0413b = vfmsq_n_f64(m0413b, x7m10, self.twiddle6.im);
                let m0413b = vfmsq_n_f64(m0413b, x8m9, self.twiddle2.im);
                let (y04, y13) = NeonButterfly::butterfly2_f64(m0413a, m0413b);

                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y04);
                vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), y13);

                let m0512a = vfmaq_n_f64(u0, x1p16, self.twiddle5.re);
                let m0512a = vfmaq_n_f64(m0512a, x2p15, self.twiddle7.re);
                let m0512a = vfmaq_n_f64(m0512a, x3p14, self.twiddle2.re);
                let m0512a = vfmaq_n_f64(m0512a, x4p13, self.twiddle3.re);
                let m0512a = vfmaq_n_f64(m0512a, x5p12, self.twiddle8.re);
                let m0512a = vfmaq_n_f64(m0512a, x6p11, self.twiddle4.re);
                let m0512a = vfmaq_n_f64(m0512a, x7p10, self.twiddle1.re);
                let m0512a = vfmaq_n_f64(m0512a, x8p9, self.twiddle6.re);
                let m0512b = vmulq_n_f64(x1m16, self.twiddle5.im);
                let m0512b = vfmsq_n_f64(m0512b, x2m15, self.twiddle7.im);
                let m0512b = vfmsq_n_f64(m0512b, x3m14, self.twiddle2.im);
                let m0512b = vfmaq_n_f64(m0512b, x4m13, self.twiddle3.im);
                let m0512b = vfmaq_n_f64(m0512b, x5m12, self.twiddle8.im);
                let m0512b = vfmsq_n_f64(m0512b, x6m11, self.twiddle4.im);
                let m0512b = vfmaq_n_f64(m0512b, x7m10, self.twiddle1.im);
                let m0512b = vfmaq_n_f64(m0512b, x8m9, self.twiddle6.im);
                let (y05, y12) = NeonButterfly::butterfly2_f64(m0512a, m0512b);

                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y05);
                vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y12);

                let m0611a = vfmaq_n_f64(u0, x1p16, self.twiddle6.re);
                let m0611a = vfmaq_n_f64(m0611a, x2p15, self.twiddle5.re);
                let m0611a = vfmaq_n_f64(m0611a, x3p14, self.twiddle1.re);
                let m0611a = vfmaq_n_f64(m0611a, x4p13, self.twiddle7.re);
                let m0611a = vfmaq_n_f64(m0611a, x5p12, self.twiddle4.re);
                let m0611a = vfmaq_n_f64(m0611a, x6p11, self.twiddle2.re);
                let m0611a = vfmaq_n_f64(m0611a, x7p10, self.twiddle8.re);
                let m0611a = vfmaq_n_f64(m0611a, x8p9, self.twiddle3.re);
                let m0611b = vmulq_n_f64(x1m16, self.twiddle6.im);
                let m0611b = vfmsq_n_f64(m0611b, x2m15, self.twiddle5.im);
                let m0611b = vfmaq_n_f64(m0611b, x3m14, self.twiddle1.im);
                let m0611b = vfmaq_n_f64(m0611b, x4m13, self.twiddle7.im);
                let m0611b = vfmsq_n_f64(m0611b, x5m12, self.twiddle4.im);
                let m0611b = vfmaq_n_f64(m0611b, x6m11, self.twiddle2.im);
                let m0611b = vfmaq_n_f64(m0611b, x7m10, self.twiddle8.im);
                let m0611b = vfmsq_n_f64(m0611b, x8m9, self.twiddle3.im);
                let (y06, y11) = NeonButterfly::butterfly2_f64(m0611a, m0611b);

                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y06);
                vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), y11);

                let m0710a = vfmaq_n_f64(u0, x1p16, self.twiddle7.re);
                let m0710a = vfmaq_n_f64(m0710a, x2p15, self.twiddle3.re);
                let m0710a = vfmaq_n_f64(m0710a, x3p14, self.twiddle4.re);
                let m0710a = vfmaq_n_f64(m0710a, x4p13, self.twiddle6.re);
                let m0710a = vfmaq_n_f64(m0710a, x5p12, self.twiddle1.re);
                let m0710a = vfmaq_n_f64(m0710a, x6p11, self.twiddle8.re);
                let m0710a = vfmaq_n_f64(m0710a, x7p10, self.twiddle2.re);
                let m0710a = vfmaq_n_f64(m0710a, x8p9, self.twiddle5.re);
                let m0710b = vmulq_n_f64(x1m16, self.twiddle7.im);
                let m0710b = vfmsq_n_f64(m0710b, x2m15, self.twiddle3.im);
                let m0710b = vfmaq_n_f64(m0710b, x3m14, self.twiddle4.im);
                let m0710b = vfmsq_n_f64(m0710b, x4m13, self.twiddle6.im);
                let m0710b = vfmaq_n_f64(m0710b, x5m12, self.twiddle1.im);
                let m0710b = vfmaq_n_f64(m0710b, x6m11, self.twiddle8.im);
                let m0710b = vfmsq_n_f64(m0710b, x7m10, self.twiddle2.im);
                let m0710b = vfmaq_n_f64(m0710b, x8m9, self.twiddle5.im);
                let (y07, y10) = NeonButterfly::butterfly2_f64(m0710a, m0710b);

                vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y07);
                vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);

                let m0809a = vfmaq_n_f64(u0, x1p16, self.twiddle8.re);
                let m0809a = vfmaq_n_f64(m0809a, x2p15, self.twiddle1.re);
                let m0809a = vfmaq_n_f64(m0809a, x3p14, self.twiddle7.re);
                let m0809a = vfmaq_n_f64(m0809a, x4p13, self.twiddle2.re);
                let m0809a = vfmaq_n_f64(m0809a, x5p12, self.twiddle6.re);
                let m0809a = vfmaq_n_f64(m0809a, x6p11, self.twiddle3.re);
                let m0809a = vfmaq_n_f64(m0809a, x7p10, self.twiddle5.re);
                let m0809a = vfmaq_n_f64(m0809a, x8p9, self.twiddle4.re);
                let m0809b = vmulq_n_f64(x1m16, self.twiddle8.im);
                let m0809b = vfmsq_n_f64(m0809b, x2m15, self.twiddle1.im);
                let m0809b = vfmaq_n_f64(m0809b, x3m14, self.twiddle7.im);
                let m0809b = vfmsq_n_f64(m0809b, x4m13, self.twiddle2.im);
                let m0809b = vfmaq_n_f64(m0809b, x5m12, self.twiddle6.im);
                let m0809b = vfmsq_n_f64(m0809b, x6m11, self.twiddle3.im);
                let m0809b = vfmaq_n_f64(m0809b, x7m10, self.twiddle5.im);
                let m0809b = vfmsq_n_f64(m0809b, x8m9, self.twiddle4.im);
                let (y08, y09) = NeonButterfly::butterfly2_f64(m0809a, m0809b);

                vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y08);
                vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y09);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaButterfly17<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        17
    }
}

impl NeonFcmaButterfly17<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 17 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(34) {
                let u0u1 = vld1q_f32(chunk.as_mut_ptr().cast());
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
                let u30u31 = vld1q_f32(chunk.get_unchecked(30..).as_ptr().cast());
                let u32u33 = vld1q_f32(chunk.get_unchecked(32..).as_ptr().cast());

                let u0 = vcombine_f32(vget_low_f32(u0u1), vget_high_f32(u16u17));
                let u1 = vcombine_f32(vget_high_f32(u0u1), vget_low_f32(u18u19));
                let u2 = vcombine_f32(vget_low_f32(u2u3), vget_high_f32(u18u19));
                let u3 = vcombine_f32(vget_high_f32(u2u3), vget_low_f32(u20u21));
                let u4 = vcombine_f32(vget_low_f32(u4u5), vget_high_f32(u20u21));
                let u5 = vcombine_f32(vget_high_f32(u4u5), vget_low_f32(u22u23));
                let u6 = vcombine_f32(vget_low_f32(u6u7), vget_high_f32(u22u23));
                let u7 = vcombine_f32(vget_high_f32(u6u7), vget_low_f32(u24u25));
                let u8 = vcombine_f32(vget_low_f32(u8u9), vget_high_f32(u24u25));
                let u9 = vcombine_f32(vget_high_f32(u8u9), vget_low_f32(u26u27));
                let u10 = vcombine_f32(vget_low_f32(u10u11), vget_high_f32(u26u27));
                let u11 = vcombine_f32(vget_high_f32(u10u11), vget_low_f32(u28u29));
                let u12 = vcombine_f32(vget_low_f32(u12u13), vget_high_f32(u28u29));
                let u13 = vcombine_f32(vget_high_f32(u12u13), vget_low_f32(u30u31));
                let u14 = vcombine_f32(vget_low_f32(u14u15), vget_high_f32(u30u31));
                let u15 = vcombine_f32(vget_high_f32(u14u15), vget_low_f32(u32u33));
                let u16 = vcombine_f32(vget_low_f32(u16u17), vget_high_f32(u32u33));

                let y00 = u0;
                let (x1p16, x1m16) = NeonButterfly::butterfly2_f32(u1, u16);
                let x1m16 = vcaddq_rot90_f32(vdupq_n_f32(0.), x1m16);
                let y00 = vaddq_f32(y00, x1p16);
                let (x2p15, x2m15) = NeonButterfly::butterfly2_f32(u2, u15);
                let x2m15 = vcaddq_rot90_f32(vdupq_n_f32(0.), x2m15);
                let y00 = vaddq_f32(y00, x2p15);
                let (x3p14, x3m14) = NeonButterfly::butterfly2_f32(u3, u14);
                let x3m14 = vcaddq_rot90_f32(vdupq_n_f32(0.), x3m14);
                let y00 = vaddq_f32(y00, x3p14);
                let (x4p13, x4m13) = NeonButterfly::butterfly2_f32(u4, u13);
                let x4m13 = vcaddq_rot90_f32(vdupq_n_f32(0.), x4m13);
                let y00 = vaddq_f32(y00, x4p13);
                let (x5p12, x5m12) = NeonButterfly::butterfly2_f32(u5, u12);
                let x5m12 = vcaddq_rot90_f32(vdupq_n_f32(0.), x5m12);
                let y00 = vaddq_f32(y00, x5p12);
                let (x6p11, x6m11) = NeonButterfly::butterfly2_f32(u6, u11);
                let x6m11 = vcaddq_rot90_f32(vdupq_n_f32(0.), x6m11);
                let y00 = vaddq_f32(y00, x6p11);
                let (x7p10, x7m10) = NeonButterfly::butterfly2_f32(u7, u10);
                let x7m10 = vcaddq_rot90_f32(vdupq_n_f32(0.), x7m10);
                let y00 = vaddq_f32(y00, x7p10);
                let (x8p9, x8m9) = NeonButterfly::butterfly2_f32(u8, u9);
                let x8m9 = vcaddq_rot90_f32(vdupq_n_f32(0.), x8m9);
                let y00 = vaddq_f32(y00, x8p9);

                let m0116a = vfmaq_n_f32(u0, x1p16, self.twiddle1.re);
                let m0116a = vfmaq_n_f32(m0116a, x2p15, self.twiddle2.re);
                let m0116a = vfmaq_n_f32(m0116a, x3p14, self.twiddle3.re);
                let m0116a = vfmaq_n_f32(m0116a, x4p13, self.twiddle4.re);
                let m0116a = vfmaq_n_f32(m0116a, x5p12, self.twiddle5.re);
                let m0116a = vfmaq_n_f32(m0116a, x6p11, self.twiddle6.re);
                let m0116a = vfmaq_n_f32(m0116a, x7p10, self.twiddle7.re);
                let m0116a = vfmaq_n_f32(m0116a, x8p9, self.twiddle8.re);
                let m0116b = vmulq_n_f32(x1m16, self.twiddle1.im);
                let m0116b = vfmaq_n_f32(m0116b, x2m15, self.twiddle2.im);
                let m0116b = vfmaq_n_f32(m0116b, x3m14, self.twiddle3.im);
                let m0116b = vfmaq_n_f32(m0116b, x4m13, self.twiddle4.im);
                let m0116b = vfmaq_n_f32(m0116b, x5m12, self.twiddle5.im);
                let m0116b = vfmaq_n_f32(m0116b, x6m11, self.twiddle6.im);
                let m0116b = vfmaq_n_f32(m0116b, x7m10, self.twiddle7.im);
                let m0116b = vfmaq_n_f32(m0116b, x8m9, self.twiddle8.im);
                let (y01, y16) = NeonButterfly::butterfly2_f32(m0116a, m0116b);

                vst1q_f32(
                    chunk.as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y00), vget_low_f32(y01)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y16), vget_high_f32(y00)),
                );

                let m0215a = vfmaq_n_f32(u0, x1p16, self.twiddle2.re);
                let m0215a = vfmaq_n_f32(m0215a, x2p15, self.twiddle4.re);
                let m0215a = vfmaq_n_f32(m0215a, x3p14, self.twiddle6.re);
                let m0215a = vfmaq_n_f32(m0215a, x4p13, self.twiddle8.re);
                let m0215a = vfmaq_n_f32(m0215a, x5p12, self.twiddle7.re);
                let m0215a = vfmaq_n_f32(m0215a, x6p11, self.twiddle5.re);
                let m0215a = vfmaq_n_f32(m0215a, x7p10, self.twiddle3.re);
                let m0215a = vfmaq_n_f32(m0215a, x8p9, self.twiddle1.re);
                let m0215b = vmulq_n_f32(x1m16, self.twiddle2.im);
                let m0215b = vfmaq_n_f32(m0215b, x2m15, self.twiddle4.im);
                let m0215b = vfmaq_n_f32(m0215b, x3m14, self.twiddle6.im);
                let m0215b = vfmaq_n_f32(m0215b, x4m13, self.twiddle8.im);
                let m0215b = vfmsq_n_f32(m0215b, x5m12, self.twiddle7.im);
                let m0215b = vfmsq_n_f32(m0215b, x6m11, self.twiddle5.im);
                let m0215b = vfmsq_n_f32(m0215b, x7m10, self.twiddle3.im);
                let m0215b = vfmsq_n_f32(m0215b, x8m9, self.twiddle1.im);
                let (y02, y15) = NeonButterfly::butterfly2_f32(m0215a, m0215b);

                vst1q_f32(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y01), vget_high_f32(y02)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(32..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y15), vget_high_f32(y16)),
                );

                let m0314a = vfmaq_n_f32(u0, x1p16, self.twiddle3.re);
                let m0314a = vfmaq_n_f32(m0314a, x2p15, self.twiddle6.re);
                let m0314a = vfmaq_n_f32(m0314a, x3p14, self.twiddle8.re);
                let m0314a = vfmaq_n_f32(m0314a, x4p13, self.twiddle5.re);
                let m0314a = vfmaq_n_f32(m0314a, x5p12, self.twiddle2.re);
                let m0314a = vfmaq_n_f32(m0314a, x6p11, self.twiddle1.re);
                let m0314a = vfmaq_n_f32(m0314a, x7p10, self.twiddle4.re);
                let m0314a = vfmaq_n_f32(m0314a, x8p9, self.twiddle7.re);
                let m0314b = vmulq_n_f32(x1m16, self.twiddle3.im);
                let m0314b = vfmaq_n_f32(m0314b, x2m15, self.twiddle6.im);
                let m0314b = vfmsq_n_f32(m0314b, x3m14, self.twiddle8.im);
                let m0314b = vfmsq_n_f32(m0314b, x4m13, self.twiddle5.im);
                let m0314b = vfmsq_n_f32(m0314b, x5m12, self.twiddle2.im);
                let m0314b = vfmaq_n_f32(m0314b, x6m11, self.twiddle1.im);
                let m0314b = vfmaq_n_f32(m0314b, x7m10, self.twiddle4.im);
                let m0314b = vfmaq_n_f32(m0314b, x8m9, self.twiddle7.im);
                let (y03, y14) = NeonButterfly::butterfly2_f32(m0314a, m0314b);

                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y02), vget_low_f32(y03)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y14), vget_low_f32(y15)),
                );

                let m0413a = vfmaq_n_f32(u0, x1p16, self.twiddle4.re);
                let m0413a = vfmaq_n_f32(m0413a, x2p15, self.twiddle8.re);
                let m0413a = vfmaq_n_f32(m0413a, x3p14, self.twiddle5.re);
                let m0413a = vfmaq_n_f32(m0413a, x4p13, self.twiddle1.re);
                let m0413a = vfmaq_n_f32(m0413a, x5p12, self.twiddle3.re);
                let m0413a = vfmaq_n_f32(m0413a, x6p11, self.twiddle7.re);
                let m0413a = vfmaq_n_f32(m0413a, x7p10, self.twiddle6.re);
                let m0413a = vfmaq_n_f32(m0413a, x8p9, self.twiddle2.re);
                let m0413b = vmulq_n_f32(x1m16, self.twiddle4.im);
                let m0413b = vfmaq_n_f32(m0413b, x2m15, self.twiddle8.im);
                let m0413b = vfmsq_n_f32(m0413b, x3m14, self.twiddle5.im);
                let m0413b = vfmsq_n_f32(m0413b, x4m13, self.twiddle1.im);
                let m0413b = vfmaq_n_f32(m0413b, x5m12, self.twiddle3.im);
                let m0413b = vfmaq_n_f32(m0413b, x6m11, self.twiddle7.im);
                let m0413b = vfmsq_n_f32(m0413b, x7m10, self.twiddle6.im);
                let m0413b = vfmsq_n_f32(m0413b, x8m9, self.twiddle2.im);
                let (y04, y13) = NeonButterfly::butterfly2_f32(m0413a, m0413b);

                let m0512a = vfmaq_n_f32(u0, x1p16, self.twiddle5.re);
                let m0512a = vfmaq_n_f32(m0512a, x2p15, self.twiddle7.re);
                let m0512a = vfmaq_n_f32(m0512a, x3p14, self.twiddle2.re);
                let m0512a = vfmaq_n_f32(m0512a, x4p13, self.twiddle3.re);
                let m0512a = vfmaq_n_f32(m0512a, x5p12, self.twiddle8.re);
                let m0512a = vfmaq_n_f32(m0512a, x6p11, self.twiddle4.re);
                let m0512a = vfmaq_n_f32(m0512a, x7p10, self.twiddle1.re);
                let m0512a = vfmaq_n_f32(m0512a, x8p9, self.twiddle6.re);
                let m0512b = vmulq_n_f32(x1m16, self.twiddle5.im);
                let m0512b = vfmsq_n_f32(m0512b, x2m15, self.twiddle7.im);
                let m0512b = vfmsq_n_f32(m0512b, x3m14, self.twiddle2.im);
                let m0512b = vfmaq_n_f32(m0512b, x4m13, self.twiddle3.im);
                let m0512b = vfmaq_n_f32(m0512b, x5m12, self.twiddle8.im);
                let m0512b = vfmsq_n_f32(m0512b, x6m11, self.twiddle4.im);
                let m0512b = vfmaq_n_f32(m0512b, x7m10, self.twiddle1.im);
                let m0512b = vfmaq_n_f32(m0512b, x8m9, self.twiddle6.im);
                let (y05, y12) = NeonButterfly::butterfly2_f32(m0512a, m0512b);

                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y04), vget_low_f32(y05)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y03), vget_high_f32(y04)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y12), vget_low_f32(y13)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(30..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y13), vget_high_f32(y14)),
                );

                let m0611a = vfmaq_n_f32(u0, x1p16, self.twiddle6.re);
                let m0611a = vfmaq_n_f32(m0611a, x2p15, self.twiddle5.re);
                let m0611a = vfmaq_n_f32(m0611a, x3p14, self.twiddle1.re);
                let m0611a = vfmaq_n_f32(m0611a, x4p13, self.twiddle7.re);
                let m0611a = vfmaq_n_f32(m0611a, x5p12, self.twiddle4.re);
                let m0611a = vfmaq_n_f32(m0611a, x6p11, self.twiddle2.re);
                let m0611a = vfmaq_n_f32(m0611a, x7p10, self.twiddle8.re);
                let m0611a = vfmaq_n_f32(m0611a, x8p9, self.twiddle3.re);
                let m0611b = vmulq_n_f32(x1m16, self.twiddle6.im);
                let m0611b = vfmsq_n_f32(m0611b, x2m15, self.twiddle5.im);
                let m0611b = vfmaq_n_f32(m0611b, x3m14, self.twiddle1.im);
                let m0611b = vfmaq_n_f32(m0611b, x4m13, self.twiddle7.im);
                let m0611b = vfmsq_n_f32(m0611b, x5m12, self.twiddle4.im);
                let m0611b = vfmaq_n_f32(m0611b, x6m11, self.twiddle2.im);
                let m0611b = vfmaq_n_f32(m0611b, x7m10, self.twiddle8.im);
                let m0611b = vfmsq_n_f32(m0611b, x8m9, self.twiddle3.im);
                let (y06, y11) = NeonButterfly::butterfly2_f32(m0611a, m0611b);

                vst1q_f32(
                    chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y05), vget_high_f32(y06)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y11), vget_high_f32(y12)),
                );

                let m0710a = vfmaq_n_f32(u0, x1p16, self.twiddle7.re);
                let m0710a = vfmaq_n_f32(m0710a, x2p15, self.twiddle3.re);
                let m0710a = vfmaq_n_f32(m0710a, x3p14, self.twiddle4.re);
                let m0710a = vfmaq_n_f32(m0710a, x4p13, self.twiddle6.re);
                let m0710a = vfmaq_n_f32(m0710a, x5p12, self.twiddle1.re);
                let m0710a = vfmaq_n_f32(m0710a, x6p11, self.twiddle8.re);
                let m0710a = vfmaq_n_f32(m0710a, x7p10, self.twiddle2.re);
                let m0710a = vfmaq_n_f32(m0710a, x8p9, self.twiddle5.re);
                let m0710b = vmulq_n_f32(x1m16, self.twiddle7.im);
                let m0710b = vfmsq_n_f32(m0710b, x2m15, self.twiddle3.im);
                let m0710b = vfmaq_n_f32(m0710b, x3m14, self.twiddle4.im);
                let m0710b = vfmsq_n_f32(m0710b, x4m13, self.twiddle6.im);
                let m0710b = vfmaq_n_f32(m0710b, x5m12, self.twiddle1.im);
                let m0710b = vfmaq_n_f32(m0710b, x6m11, self.twiddle8.im);
                let m0710b = vfmsq_n_f32(m0710b, x7m10, self.twiddle2.im);
                let m0710b = vfmaq_n_f32(m0710b, x8m9, self.twiddle5.im);
                let (y07, y10) = NeonButterfly::butterfly2_f32(m0710a, m0710b);

                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y06), vget_low_f32(y07)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y10), vget_low_f32(y11)),
                );

                let m0809a = vfmaq_n_f32(u0, x1p16, self.twiddle8.re);
                let m0809a = vfmaq_n_f32(m0809a, x2p15, self.twiddle1.re);
                let m0809a = vfmaq_n_f32(m0809a, x3p14, self.twiddle7.re);
                let m0809a = vfmaq_n_f32(m0809a, x4p13, self.twiddle2.re);
                let m0809a = vfmaq_n_f32(m0809a, x5p12, self.twiddle6.re);
                let m0809a = vfmaq_n_f32(m0809a, x6p11, self.twiddle3.re);
                let m0809a = vfmaq_n_f32(m0809a, x7p10, self.twiddle5.re);
                let m0809a = vfmaq_n_f32(m0809a, x8p9, self.twiddle4.re);
                let m0809b = vmulq_n_f32(x1m16, self.twiddle8.im);
                let m0809b = vfmsq_n_f32(m0809b, x2m15, self.twiddle1.im);
                let m0809b = vfmaq_n_f32(m0809b, x3m14, self.twiddle7.im);
                let m0809b = vfmsq_n_f32(m0809b, x4m13, self.twiddle2.im);
                let m0809b = vfmaq_n_f32(m0809b, x5m12, self.twiddle6.im);
                let m0809b = vfmsq_n_f32(m0809b, x6m11, self.twiddle3.im);
                let m0809b = vfmaq_n_f32(m0809b, x7m10, self.twiddle5.im);
                let m0809b = vfmsq_n_f32(m0809b, x8m9, self.twiddle4.im);
                let (y08, y09) = NeonButterfly::butterfly2_f32(m0809a, m0809b);

                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y08), vget_low_f32(y09)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y07), vget_high_f32(y08)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(y09), vget_high_f32(y10)),
                );
            }

            let rem = in_place.chunks_exact_mut(34).into_remainder();

            for chunk in rem.chunks_exact_mut(17) {
                let u0u1 = vld1q_f32(chunk.as_mut_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());
                let u16 = vld1_f32(chunk.get_unchecked(16..).as_ptr().cast());

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
                let u10 = vget_low_f32(u10u11);
                let u11 = vget_high_f32(u10u11);
                let u12 = vget_low_f32(u12u13);
                let u13 = vget_high_f32(u12u13);
                let u14 = vget_low_f32(u14u15);
                let u15 = vget_high_f32(u14u15);

                let y00 = u0;
                let (x1p16, x1m16) = NeonButterfly::butterfly2h_f32(u1, u16);
                let x1m16 = vcadd_rot90_f32(vdup_n_f32(0.), x1m16);
                let y00 = vadd_f32(y00, x1p16);
                let (x2p15, x2m15) = NeonButterfly::butterfly2h_f32(u2, u15);
                let x2m15 = vcadd_rot90_f32(vdup_n_f32(0.), x2m15);
                let y00 = vadd_f32(y00, x2p15);
                let (x3p14, x3m14) = NeonButterfly::butterfly2h_f32(u3, u14);
                let x3m14 = vcadd_rot90_f32(vdup_n_f32(0.), x3m14);
                let y00 = vadd_f32(y00, x3p14);
                let (x4p13, x4m13) = NeonButterfly::butterfly2h_f32(u4, u13);
                let x4m13 = vcadd_rot90_f32(vdup_n_f32(0.), x4m13);
                let y00 = vadd_f32(y00, x4p13);
                let (x5p12, x5m12) = NeonButterfly::butterfly2h_f32(u5, u12);
                let x5m12 = vcadd_rot90_f32(vdup_n_f32(0.), x5m12);
                let y00 = vadd_f32(y00, x5p12);
                let (x6p11, x6m11) = NeonButterfly::butterfly2h_f32(u6, u11);
                let x6m11 = vcadd_rot90_f32(vdup_n_f32(0.), x6m11);
                let y00 = vadd_f32(y00, x6p11);
                let (x7p10, x7m10) = NeonButterfly::butterfly2h_f32(u7, u10);
                let x7m10 = vcadd_rot90_f32(vdup_n_f32(0.), x7m10);
                let y00 = vadd_f32(y00, x7p10);
                let (x8p9, x8m9) = NeonButterfly::butterfly2h_f32(u8, u9);
                let x8m9 = vcadd_rot90_f32(vdup_n_f32(0.), x8m9);
                let y00 = vadd_f32(y00, x8p9);

                let m0116a = vfma_n_f32(u0, x1p16, self.twiddle1.re);
                let m0116a = vfma_n_f32(m0116a, x2p15, self.twiddle2.re);
                let m0116a = vfma_n_f32(m0116a, x3p14, self.twiddle3.re);
                let m0116a = vfma_n_f32(m0116a, x4p13, self.twiddle4.re);
                let m0116a = vfma_n_f32(m0116a, x5p12, self.twiddle5.re);
                let m0116a = vfma_n_f32(m0116a, x6p11, self.twiddle6.re);
                let m0116a = vfma_n_f32(m0116a, x7p10, self.twiddle7.re);
                let m0116a = vfma_n_f32(m0116a, x8p9, self.twiddle8.re);
                let m0116b = vmul_n_f32(x1m16, self.twiddle1.im);
                let m0116b = vfma_n_f32(m0116b, x2m15, self.twiddle2.im);
                let m0116b = vfma_n_f32(m0116b, x3m14, self.twiddle3.im);
                let m0116b = vfma_n_f32(m0116b, x4m13, self.twiddle4.im);
                let m0116b = vfma_n_f32(m0116b, x5m12, self.twiddle5.im);
                let m0116b = vfma_n_f32(m0116b, x6m11, self.twiddle6.im);
                let m0116b = vfma_n_f32(m0116b, x7m10, self.twiddle7.im);
                let m0116b = vfma_n_f32(m0116b, x8m9, self.twiddle8.im);
                let (y01, y16) = NeonButterfly::butterfly2h_f32(m0116a, m0116b);

                vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y00, y01));
                vst1_f32(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), y16);

                let m0215a = vfma_n_f32(u0, x1p16, self.twiddle2.re);
                let m0215a = vfma_n_f32(m0215a, x2p15, self.twiddle4.re);
                let m0215a = vfma_n_f32(m0215a, x3p14, self.twiddle6.re);
                let m0215a = vfma_n_f32(m0215a, x4p13, self.twiddle8.re);
                let m0215a = vfma_n_f32(m0215a, x5p12, self.twiddle7.re);
                let m0215a = vfma_n_f32(m0215a, x6p11, self.twiddle5.re);
                let m0215a = vfma_n_f32(m0215a, x7p10, self.twiddle3.re);
                let m0215a = vfma_n_f32(m0215a, x8p9, self.twiddle1.re);
                let m0215b = vmul_n_f32(x1m16, self.twiddle2.im);
                let m0215b = vfma_n_f32(m0215b, x2m15, self.twiddle4.im);
                let m0215b = vfma_n_f32(m0215b, x3m14, self.twiddle6.im);
                let m0215b = vfma_n_f32(m0215b, x4m13, self.twiddle8.im);
                let m0215b = vfms_n_f32(m0215b, x5m12, self.twiddle7.im);
                let m0215b = vfms_n_f32(m0215b, x6m11, self.twiddle5.im);
                let m0215b = vfms_n_f32(m0215b, x7m10, self.twiddle3.im);
                let m0215b = vfms_n_f32(m0215b, x8m9, self.twiddle1.im);
                let (y02, y15) = NeonButterfly::butterfly2h_f32(m0215a, m0215b);

                let m0314a = vfma_n_f32(u0, x1p16, self.twiddle3.re);
                let m0314a = vfma_n_f32(m0314a, x2p15, self.twiddle6.re);
                let m0314a = vfma_n_f32(m0314a, x3p14, self.twiddle8.re);
                let m0314a = vfma_n_f32(m0314a, x4p13, self.twiddle5.re);
                let m0314a = vfma_n_f32(m0314a, x5p12, self.twiddle2.re);
                let m0314a = vfma_n_f32(m0314a, x6p11, self.twiddle1.re);
                let m0314a = vfma_n_f32(m0314a, x7p10, self.twiddle4.re);
                let m0314a = vfma_n_f32(m0314a, x8p9, self.twiddle7.re);
                let m0314b = vmul_n_f32(x1m16, self.twiddle3.im);
                let m0314b = vfma_n_f32(m0314b, x2m15, self.twiddle6.im);
                let m0314b = vfms_n_f32(m0314b, x3m14, self.twiddle8.im);
                let m0314b = vfms_n_f32(m0314b, x4m13, self.twiddle5.im);
                let m0314b = vfms_n_f32(m0314b, x5m12, self.twiddle2.im);
                let m0314b = vfma_n_f32(m0314b, x6m11, self.twiddle1.im);
                let m0314b = vfma_n_f32(m0314b, x7m10, self.twiddle4.im);
                let m0314b = vfma_n_f32(m0314b, x8m9, self.twiddle7.im);
                let (y03, y14) = NeonButterfly::butterfly2h_f32(m0314a, m0314b);

                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y02, y03),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vcombine_f32(y14, y15),
                );

                let m0413a = vfma_n_f32(u0, x1p16, self.twiddle4.re);
                let m0413a = vfma_n_f32(m0413a, x2p15, self.twiddle8.re);
                let m0413a = vfma_n_f32(m0413a, x3p14, self.twiddle5.re);
                let m0413a = vfma_n_f32(m0413a, x4p13, self.twiddle1.re);
                let m0413a = vfma_n_f32(m0413a, x5p12, self.twiddle3.re);
                let m0413a = vfma_n_f32(m0413a, x6p11, self.twiddle7.re);
                let m0413a = vfma_n_f32(m0413a, x7p10, self.twiddle6.re);
                let m0413a = vfma_n_f32(m0413a, x8p9, self.twiddle2.re);
                let m0413b = vmul_n_f32(x1m16, self.twiddle4.im);
                let m0413b = vfma_n_f32(m0413b, x2m15, self.twiddle8.im);
                let m0413b = vfms_n_f32(m0413b, x3m14, self.twiddle5.im);
                let m0413b = vfms_n_f32(m0413b, x4m13, self.twiddle1.im);
                let m0413b = vfma_n_f32(m0413b, x5m12, self.twiddle3.im);
                let m0413b = vfma_n_f32(m0413b, x6m11, self.twiddle7.im);
                let m0413b = vfms_n_f32(m0413b, x7m10, self.twiddle6.im);
                let m0413b = vfms_n_f32(m0413b, x8m9, self.twiddle2.im);
                let (y04, y13) = NeonButterfly::butterfly2h_f32(m0413a, m0413b);

                let m0512a = vfma_n_f32(u0, x1p16, self.twiddle5.re);
                let m0512a = vfma_n_f32(m0512a, x2p15, self.twiddle7.re);
                let m0512a = vfma_n_f32(m0512a, x3p14, self.twiddle2.re);
                let m0512a = vfma_n_f32(m0512a, x4p13, self.twiddle3.re);
                let m0512a = vfma_n_f32(m0512a, x5p12, self.twiddle8.re);
                let m0512a = vfma_n_f32(m0512a, x6p11, self.twiddle4.re);
                let m0512a = vfma_n_f32(m0512a, x7p10, self.twiddle1.re);
                let m0512a = vfma_n_f32(m0512a, x8p9, self.twiddle6.re);
                let m0512b = vmul_n_f32(x1m16, self.twiddle5.im);
                let m0512b = vfms_n_f32(m0512b, x2m15, self.twiddle7.im);
                let m0512b = vfms_n_f32(m0512b, x3m14, self.twiddle2.im);
                let m0512b = vfma_n_f32(m0512b, x4m13, self.twiddle3.im);
                let m0512b = vfma_n_f32(m0512b, x5m12, self.twiddle8.im);
                let m0512b = vfms_n_f32(m0512b, x6m11, self.twiddle4.im);
                let m0512b = vfma_n_f32(m0512b, x7m10, self.twiddle1.im);
                let m0512b = vfma_n_f32(m0512b, x8m9, self.twiddle6.im);
                let (y05, y12) = NeonButterfly::butterfly2h_f32(m0512a, m0512b);

                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y04, y05),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vcombine_f32(y12, y13),
                );

                let m0611a = vfma_n_f32(u0, x1p16, self.twiddle6.re);
                let m0611a = vfma_n_f32(m0611a, x2p15, self.twiddle5.re);
                let m0611a = vfma_n_f32(m0611a, x3p14, self.twiddle1.re);
                let m0611a = vfma_n_f32(m0611a, x4p13, self.twiddle7.re);
                let m0611a = vfma_n_f32(m0611a, x5p12, self.twiddle4.re);
                let m0611a = vfma_n_f32(m0611a, x6p11, self.twiddle2.re);
                let m0611a = vfma_n_f32(m0611a, x7p10, self.twiddle8.re);
                let m0611a = vfma_n_f32(m0611a, x8p9, self.twiddle3.re);
                let m0611b = vmul_n_f32(x1m16, self.twiddle6.im);
                let m0611b = vfms_n_f32(m0611b, x2m15, self.twiddle5.im);
                let m0611b = vfma_n_f32(m0611b, x3m14, self.twiddle1.im);
                let m0611b = vfma_n_f32(m0611b, x4m13, self.twiddle7.im);
                let m0611b = vfms_n_f32(m0611b, x5m12, self.twiddle4.im);
                let m0611b = vfma_n_f32(m0611b, x6m11, self.twiddle2.im);
                let m0611b = vfma_n_f32(m0611b, x7m10, self.twiddle8.im);
                let m0611b = vfms_n_f32(m0611b, x8m9, self.twiddle3.im);
                let (y06, y11) = NeonButterfly::butterfly2h_f32(m0611a, m0611b);

                let m0710a = vfma_n_f32(u0, x1p16, self.twiddle7.re);
                let m0710a = vfma_n_f32(m0710a, x2p15, self.twiddle3.re);
                let m0710a = vfma_n_f32(m0710a, x3p14, self.twiddle4.re);
                let m0710a = vfma_n_f32(m0710a, x4p13, self.twiddle6.re);
                let m0710a = vfma_n_f32(m0710a, x5p12, self.twiddle1.re);
                let m0710a = vfma_n_f32(m0710a, x6p11, self.twiddle8.re);
                let m0710a = vfma_n_f32(m0710a, x7p10, self.twiddle2.re);
                let m0710a = vfma_n_f32(m0710a, x8p9, self.twiddle5.re);
                let m0710b = vmul_n_f32(x1m16, self.twiddle7.im);
                let m0710b = vfms_n_f32(m0710b, x2m15, self.twiddle3.im);
                let m0710b = vfma_n_f32(m0710b, x3m14, self.twiddle4.im);
                let m0710b = vfms_n_f32(m0710b, x4m13, self.twiddle6.im);
                let m0710b = vfma_n_f32(m0710b, x5m12, self.twiddle1.im);
                let m0710b = vfma_n_f32(m0710b, x6m11, self.twiddle8.im);
                let m0710b = vfms_n_f32(m0710b, x7m10, self.twiddle2.im);
                let m0710b = vfma_n_f32(m0710b, x8m9, self.twiddle5.im);
                let (y07, y10) = NeonButterfly::butterfly2h_f32(m0710a, m0710b);

                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y06, y07),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(y10, y11),
                );

                let m0809a = vfma_n_f32(u0, x1p16, self.twiddle8.re);
                let m0809a = vfma_n_f32(m0809a, x2p15, self.twiddle1.re);
                let m0809a = vfma_n_f32(m0809a, x3p14, self.twiddle7.re);
                let m0809a = vfma_n_f32(m0809a, x4p13, self.twiddle2.re);
                let m0809a = vfma_n_f32(m0809a, x5p12, self.twiddle6.re);
                let m0809a = vfma_n_f32(m0809a, x6p11, self.twiddle3.re);
                let m0809a = vfma_n_f32(m0809a, x7p10, self.twiddle5.re);
                let m0809a = vfma_n_f32(m0809a, x8p9, self.twiddle4.re);
                let m0809b = vmul_n_f32(x1m16, self.twiddle8.im);
                let m0809b = vfms_n_f32(m0809b, x2m15, self.twiddle1.im);
                let m0809b = vfma_n_f32(m0809b, x3m14, self.twiddle7.im);
                let m0809b = vfms_n_f32(m0809b, x4m13, self.twiddle2.im);
                let m0809b = vfma_n_f32(m0809b, x5m12, self.twiddle6.im);
                let m0809b = vfms_n_f32(m0809b, x6m11, self.twiddle3.im);
                let m0809b = vfma_n_f32(m0809b, x7m10, self.twiddle5.im);
                let m0809b = vfms_n_f32(m0809b, x8m9, self.twiddle4.im);
                let (y08, y09) = NeonButterfly::butterfly2h_f32(m0809a, m0809b);

                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y08, y09),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Butterfly17;
    use rand::Rng;

    #[test]
    fn test_butterfly17_f32() {
        for i in 1..5 {
            let size = 17usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix13_reference = Butterfly17::new(FftDirection::Forward);
            let radix13_inv_reference = Butterfly17::new(FftDirection::Inverse);

            let radix_forward = NeonFcmaButterfly17::new(FftDirection::Forward);
            let radix_inverse = NeonFcmaButterfly17::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix13_reference.execute(&mut z_ref).unwrap();

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
            radix13_inv_reference.execute(&mut z_ref).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-5,
                        "a_re {} != b_re {} for size {}, reference inv failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-5,
                        "a_im {} != b_im {} for size {}, reference inv failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            input = input.iter().map(|&x| x * (1.0 / 17f32)).collect();

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
    fn test_butterfly17_f64() {
        for i in 1..5 {
            let size = 17usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();

            let mut z_ref = input.to_vec();

            let radix9_reference = Butterfly17::new(FftDirection::Forward);

            let radix_forward = NeonFcmaButterfly17::new(FftDirection::Forward);
            let radix_inverse = NeonFcmaButterfly17::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix9_reference.execute(&mut z_ref).unwrap();

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

            input = input.iter().map(|&x| x * (1.0 / 17f64)).collect();

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
}
