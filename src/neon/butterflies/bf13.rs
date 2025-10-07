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
use crate::neon::util::{v_rotate90_f32, v_rotate90_f64, vh_rotate90_f32};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonButterfly13<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonButterfly13<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 13, fft_direction),
            twiddle2: compute_twiddle(2, 13, fft_direction),
            twiddle3: compute_twiddle(3, 13, fft_direction),
            twiddle4: compute_twiddle(4, 13, fft_direction),
            twiddle5: compute_twiddle(5, 13, fft_direction),
            twiddle6: compute_twiddle(6, 13, fft_direction),
        }
    }
}

impl FftExecutor<f64> for NeonButterfly13<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 13 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            let rot_sign = vld1q_f64(ROT_90.as_ptr());

            for chunk in in_place.chunks_exact_mut(13) {
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

                let y00 = u0;
                let (x1p12, x1m12) = NeonButterfly::butterfly2_f64(u1, u12);
                let x1m12 = v_rotate90_f64(x1m12, rot_sign);
                let y00 = vaddq_f64(y00, x1p12);
                let (x2p11, x2m11) = NeonButterfly::butterfly2_f64(u2, u11);
                let x2m11 = v_rotate90_f64(x2m11, rot_sign);
                let y00 = vaddq_f64(y00, x2p11);
                let (x3p10, x3m10) = NeonButterfly::butterfly2_f64(u3, u10);
                let x3m10 = v_rotate90_f64(x3m10, rot_sign);
                let y00 = vaddq_f64(y00, x3p10);
                let (x4p9, x4m9) = NeonButterfly::butterfly2_f64(u4, u9);
                let x4m9 = v_rotate90_f64(x4m9, rot_sign);
                let y00 = vaddq_f64(y00, x4p9);
                let (x5p8, x5m8) = NeonButterfly::butterfly2_f64(u5, u8);
                let x5m8 = v_rotate90_f64(x5m8, rot_sign);
                let y00 = vaddq_f64(y00, x5p8);
                let (x6p7, x6m7) = NeonButterfly::butterfly2_f64(u6, u7);
                let x6m7 = v_rotate90_f64(x6m7, rot_sign);
                let y00 = vaddq_f64(y00, x6p7);

                let m0112a = vfmaq_n_f64(u0, x1p12, self.twiddle1.re);
                let m0112a = vfmaq_n_f64(m0112a, x2p11, self.twiddle2.re);
                let m0112a = vfmaq_n_f64(m0112a, x3p10, self.twiddle3.re);
                let m0112a = vfmaq_n_f64(m0112a, x4p9, self.twiddle4.re);
                let m0112a = vfmaq_n_f64(m0112a, x5p8, self.twiddle5.re);
                let m0112a = vfmaq_n_f64(m0112a, x6p7, self.twiddle6.re);
                let m0112b = vmulq_n_f64(x1m12, self.twiddle1.im);
                let m0112b = vfmaq_n_f64(m0112b, x2m11, self.twiddle2.im);
                let m0112b = vfmaq_n_f64(m0112b, x3m10, self.twiddle3.im);
                let m0112b = vfmaq_n_f64(m0112b, x4m9, self.twiddle4.im);
                let m0112b = vfmaq_n_f64(m0112b, x5m8, self.twiddle5.im);
                let m0112b = vfmaq_n_f64(m0112b, x6m7, self.twiddle6.im);
                let (y01, y12) = NeonButterfly::butterfly2_f64(m0112a, m0112b);

                let m0211a = vfmaq_n_f64(u0, x1p12, self.twiddle2.re);
                let m0211a = vfmaq_n_f64(m0211a, x2p11, self.twiddle4.re);
                let m0211a = vfmaq_n_f64(m0211a, x3p10, self.twiddle6.re);
                let m0211a = vfmaq_n_f64(m0211a, x4p9, self.twiddle5.re);
                let m0211a = vfmaq_n_f64(m0211a, x5p8, self.twiddle3.re);
                let m0211a = vfmaq_n_f64(m0211a, x6p7, self.twiddle1.re);
                let m0211b = vmulq_n_f64(x1m12, self.twiddle2.im);
                let m0211b = vfmaq_n_f64(m0211b, x2m11, self.twiddle4.im);
                let m0211b = vfmaq_n_f64(m0211b, x3m10, self.twiddle6.im);
                let m0211b = vfmaq_n_f64(m0211b, x4m9, -self.twiddle5.im);
                let m0211b = vfmaq_n_f64(m0211b, x5m8, -self.twiddle3.im);
                let m0211b = vfmaq_n_f64(m0211b, x6m7, -self.twiddle1.im);
                let (y02, y11) = NeonButterfly::butterfly2_f64(m0211a, m0211b);

                let m0310a = vfmaq_n_f64(u0, x1p12, self.twiddle3.re);
                let m0310a = vfmaq_n_f64(m0310a, x2p11, self.twiddle6.re);
                let m0310a = vfmaq_n_f64(m0310a, x3p10, self.twiddle4.re);
                let m0310a = vfmaq_n_f64(m0310a, x4p9, self.twiddle1.re);
                let m0310a = vfmaq_n_f64(m0310a, x5p8, self.twiddle2.re);
                let m0310a = vfmaq_n_f64(m0310a, x6p7, self.twiddle5.re);
                let m0310b = vmulq_n_f64(x1m12, self.twiddle3.im);
                let m0310b = vfmaq_n_f64(m0310b, x2m11, self.twiddle6.im);
                let m0310b = vfmsq_n_f64(m0310b, x3m10, self.twiddle4.im);
                let m0310b = vfmsq_n_f64(m0310b, x4m9, self.twiddle1.im);
                let m0310b = vfmaq_n_f64(m0310b, x5m8, self.twiddle2.im);
                let m0310b = vfmaq_n_f64(m0310b, x6m7, self.twiddle5.im);
                let (y03, y10) = NeonButterfly::butterfly2_f64(m0310a, m0310b);

                let m0409a = vfmaq_n_f64(u0, x1p12, self.twiddle4.re);
                let m0409a = vfmaq_n_f64(m0409a, x2p11, self.twiddle5.re);
                let m0409a = vfmaq_n_f64(m0409a, x3p10, self.twiddle1.re);
                let m0409a = vfmaq_n_f64(m0409a, x4p9, self.twiddle3.re);
                let m0409a = vfmaq_n_f64(m0409a, x5p8, self.twiddle6.re);
                let m0409a = vfmaq_n_f64(m0409a, x6p7, self.twiddle2.re);
                let m0409b = vmulq_n_f64(x1m12, self.twiddle4.im);
                let m0409b = vfmsq_n_f64(m0409b, x2m11, self.twiddle5.im);
                let m0409b = vfmsq_n_f64(m0409b, x3m10, self.twiddle1.im);
                let m0409b = vfmaq_n_f64(m0409b, x4m9, self.twiddle3.im);
                let m0409b = vfmsq_n_f64(m0409b, x5m8, self.twiddle6.im);
                let m0409b = vfmsq_n_f64(m0409b, x6m7, self.twiddle2.im);
                let (y04, y09) = NeonButterfly::butterfly2_f64(m0409a, m0409b);

                let m0508a = vfmaq_n_f64(u0, x1p12, self.twiddle5.re);
                let m0508a = vfmaq_n_f64(m0508a, x2p11, self.twiddle3.re);
                let m0508a = vfmaq_n_f64(m0508a, x3p10, self.twiddle2.re);
                let m0508a = vfmaq_n_f64(m0508a, x4p9, self.twiddle6.re);
                let m0508a = vfmaq_n_f64(m0508a, x5p8, self.twiddle1.re);
                let m0508a = vfmaq_n_f64(m0508a, x6p7, self.twiddle4.re);
                let m0508b = vmulq_n_f64(x1m12, self.twiddle5.im);
                let m0508b = vfmsq_n_f64(m0508b, x2m11, self.twiddle3.im);
                let m0508b = vfmaq_n_f64(m0508b, x3m10, self.twiddle2.im);
                let m0508b = vfmsq_n_f64(m0508b, x4m9, self.twiddle6.im);
                let m0508b = vfmsq_n_f64(m0508b, x5m8, self.twiddle1.im);
                let m0508b = vfmaq_n_f64(m0508b, x6m7, self.twiddle4.im);
                let (y05, y08) = NeonButterfly::butterfly2_f64(m0508a, m0508b);

                let m0607a = vfmaq_n_f64(u0, x1p12, self.twiddle6.re);
                let m0607a = vfmaq_n_f64(m0607a, x2p11, self.twiddle1.re);
                let m0607a = vfmaq_n_f64(m0607a, x3p10, self.twiddle5.re);
                let m0607a = vfmaq_n_f64(m0607a, x4p9, self.twiddle2.re);
                let m0607a = vfmaq_n_f64(m0607a, x5p8, self.twiddle4.re);
                let m0607a = vfmaq_n_f64(m0607a, x6p7, self.twiddle3.re);
                let m0607b = vmulq_n_f64(x1m12, self.twiddle6.im);
                let m0607b = vfmsq_n_f64(m0607b, x2m11, self.twiddle1.im);
                let m0607b = vfmaq_n_f64(m0607b, x3m10, self.twiddle5.im);
                let m0607b = vfmsq_n_f64(m0607b, x4m9, self.twiddle2.im);
                let m0607b = vfmaq_n_f64(m0607b, x5m8, self.twiddle4.im);
                let m0607b = vfmsq_n_f64(m0607b, x6m7, self.twiddle3.im);
                let (y06, y07) = NeonButterfly::butterfly2_f64(m0607a, m0607b);

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
                vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), y11);
                vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y12);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        13
    }
}

impl FftExecutor<f32> for NeonButterfly13<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 13 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            static ROT_90: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            let rot_sign = vld1q_f32(ROT_90.as_ptr());

            for chunk in in_place.chunks_exact_mut(26) {
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

                let u0 = vcombine_f32(vget_low_f32(u0u1), vget_high_f32(u12u13));
                let u1 = vcombine_f32(vget_high_f32(u0u1), vget_low_f32(u14u15));
                let u2 = vcombine_f32(vget_low_f32(u2u3), vget_high_f32(u14u15));
                let u3 = vcombine_f32(vget_high_f32(u2u3), vget_low_f32(u16u17));
                let u4 = vcombine_f32(vget_low_f32(u4u5), vget_high_f32(u16u17));
                let u5 = vcombine_f32(vget_high_f32(u4u5), vget_low_f32(u18u19));
                let u6 = vcombine_f32(vget_low_f32(u6u7), vget_high_f32(u18u19));
                let u7 = vcombine_f32(vget_high_f32(u6u7), vget_low_f32(u20u21));
                let u8 = vcombine_f32(vget_low_f32(u8u9), vget_high_f32(u20u21));
                let u9 = vcombine_f32(vget_high_f32(u8u9), vget_low_f32(u22u23));
                let u10 = vcombine_f32(vget_low_f32(u10u11), vget_high_f32(u22u23));
                let u11 = vcombine_f32(vget_high_f32(u10u11), vget_low_f32(u24u25));
                let u12 = vcombine_f32(vget_low_f32(u12u13), vget_high_f32(u24u25));

                let y00 = u0;
                let (x1p12, x1m12) = NeonButterfly::butterfly2_f32(u1, u12);
                let x1m12 = v_rotate90_f32(x1m12, rot_sign);
                let y00 = vaddq_f32(y00, x1p12);
                let (x2p11, x2m11) = NeonButterfly::butterfly2_f32(u2, u11);
                let x2m11 = v_rotate90_f32(x2m11, rot_sign);
                let y00 = vaddq_f32(y00, x2p11);
                let (x3p10, x3m10) = NeonButterfly::butterfly2_f32(u3, u10);
                let x3m10 = v_rotate90_f32(x3m10, rot_sign);
                let y00 = vaddq_f32(y00, x3p10);
                let (x4p9, x4m9) = NeonButterfly::butterfly2_f32(u4, u9);
                let x4m9 = v_rotate90_f32(x4m9, rot_sign);
                let y00 = vaddq_f32(y00, x4p9);
                let (x5p8, x5m8) = NeonButterfly::butterfly2_f32(u5, u8);
                let x5m8 = v_rotate90_f32(x5m8, rot_sign);
                let y00 = vaddq_f32(y00, x5p8);
                let (x6p7, x6m7) = NeonButterfly::butterfly2_f32(u6, u7);
                let x6m7 = v_rotate90_f32(x6m7, rot_sign);
                let y00 = vaddq_f32(y00, x6p7);

                let m0112a = vfmaq_n_f32(u0, x1p12, self.twiddle1.re);
                let m0112a = vfmaq_n_f32(m0112a, x2p11, self.twiddle2.re);
                let m0112a = vfmaq_n_f32(m0112a, x3p10, self.twiddle3.re);
                let m0112a = vfmaq_n_f32(m0112a, x4p9, self.twiddle4.re);
                let m0112a = vfmaq_n_f32(m0112a, x5p8, self.twiddle5.re);
                let m0112a = vfmaq_n_f32(m0112a, x6p7, self.twiddle6.re);
                let m0112b = vmulq_n_f32(x1m12, self.twiddle1.im);
                let m0112b = vfmaq_n_f32(m0112b, x2m11, self.twiddle2.im);
                let m0112b = vfmaq_n_f32(m0112b, x3m10, self.twiddle3.im);
                let m0112b = vfmaq_n_f32(m0112b, x4m9, self.twiddle4.im);
                let m0112b = vfmaq_n_f32(m0112b, x5m8, self.twiddle5.im);
                let m0112b = vfmaq_n_f32(m0112b, x6m7, self.twiddle6.im);
                let (y01, y12) = NeonButterfly::butterfly2_f32(m0112a, m0112b);

                let m0211a = vfmaq_n_f32(u0, x1p12, self.twiddle2.re);
                let m0211a = vfmaq_n_f32(m0211a, x2p11, self.twiddle4.re);
                let m0211a = vfmaq_n_f32(m0211a, x3p10, self.twiddle6.re);
                let m0211a = vfmaq_n_f32(m0211a, x4p9, self.twiddle5.re);
                let m0211a = vfmaq_n_f32(m0211a, x5p8, self.twiddle3.re);
                let m0211a = vfmaq_n_f32(m0211a, x6p7, self.twiddle1.re);
                let m0211b = vmulq_n_f32(x1m12, self.twiddle2.im);
                let m0211b = vfmaq_n_f32(m0211b, x2m11, self.twiddle4.im);
                let m0211b = vfmaq_n_f32(m0211b, x3m10, self.twiddle6.im);
                let m0211b = vfmsq_n_f32(m0211b, x4m9, self.twiddle5.im);
                let m0211b = vfmsq_n_f32(m0211b, x5m8, self.twiddle3.im);
                let m0211b = vfmsq_n_f32(m0211b, x6m7, self.twiddle1.im);
                let (y02, y11) = NeonButterfly::butterfly2_f32(m0211a, m0211b);

                let m0310a = vfmaq_n_f32(u0, x1p12, self.twiddle3.re);
                let m0310a = vfmaq_n_f32(m0310a, x2p11, self.twiddle6.re);
                let m0310a = vfmaq_n_f32(m0310a, x3p10, self.twiddle4.re);
                let m0310a = vfmaq_n_f32(m0310a, x4p9, self.twiddle1.re);
                let m0310a = vfmaq_n_f32(m0310a, x5p8, self.twiddle2.re);
                let m0310a = vfmaq_n_f32(m0310a, x6p7, self.twiddle5.re);
                let m0310b = vmulq_n_f32(x1m12, self.twiddle3.im);
                let m0310b = vfmaq_n_f32(m0310b, x2m11, self.twiddle6.im);
                let m0310b = vfmsq_n_f32(m0310b, x3m10, self.twiddle4.im);
                let m0310b = vfmsq_n_f32(m0310b, x4m9, self.twiddle1.im);
                let m0310b = vfmaq_n_f32(m0310b, x5m8, self.twiddle2.im);
                let m0310b = vfmaq_n_f32(m0310b, x6m7, self.twiddle5.im);
                let (y03, y10) = NeonButterfly::butterfly2_f32(m0310a, m0310b);

                let m0409a = vfmaq_n_f32(u0, x1p12, self.twiddle4.re);
                let m0409a = vfmaq_n_f32(m0409a, x2p11, self.twiddle5.re);
                let m0409a = vfmaq_n_f32(m0409a, x3p10, self.twiddle1.re);
                let m0409a = vfmaq_n_f32(m0409a, x4p9, self.twiddle3.re);
                let m0409a = vfmaq_n_f32(m0409a, x5p8, self.twiddle6.re);
                let m0409a = vfmaq_n_f32(m0409a, x6p7, self.twiddle2.re);
                let m0409b = vmulq_n_f32(x1m12, self.twiddle4.im);
                let m0409b = vfmsq_n_f32(m0409b, x2m11, self.twiddle5.im);
                let m0409b = vfmsq_n_f32(m0409b, x3m10, self.twiddle1.im);
                let m0409b = vfmaq_n_f32(m0409b, x4m9, self.twiddle3.im);
                let m0409b = vfmsq_n_f32(m0409b, x5m8, self.twiddle6.im);
                let m0409b = vfmsq_n_f32(m0409b, x6m7, self.twiddle2.im);
                let (y04, y09) = NeonButterfly::butterfly2_f32(m0409a, m0409b);

                let m0508a = vfmaq_n_f32(u0, x1p12, self.twiddle5.re);
                let m0508a = vfmaq_n_f32(m0508a, x2p11, self.twiddle3.re);
                let m0508a = vfmaq_n_f32(m0508a, x3p10, self.twiddle2.re);
                let m0508a = vfmaq_n_f32(m0508a, x4p9, self.twiddle6.re);
                let m0508a = vfmaq_n_f32(m0508a, x5p8, self.twiddle1.re);
                let m0508a = vfmaq_n_f32(m0508a, x6p7, self.twiddle4.re);
                let m0508b = vmulq_n_f32(x1m12, self.twiddle5.im);
                let m0508b = vfmsq_n_f32(m0508b, x2m11, self.twiddle3.im);
                let m0508b = vfmaq_n_f32(m0508b, x3m10, self.twiddle2.im);
                let m0508b = vfmsq_n_f32(m0508b, x4m9, self.twiddle6.im);
                let m0508b = vfmsq_n_f32(m0508b, x5m8, self.twiddle1.im);
                let m0508b = vfmaq_n_f32(m0508b, x6m7, self.twiddle4.im);
                let (y05, y08) = NeonButterfly::butterfly2_f32(m0508a, m0508b);

                let m0607a = vfmaq_n_f32(u0, x1p12, self.twiddle6.re);
                let m0607a = vfmaq_n_f32(m0607a, x2p11, self.twiddle1.re);
                let m0607a = vfmaq_n_f32(m0607a, x3p10, self.twiddle5.re);
                let m0607a = vfmaq_n_f32(m0607a, x4p9, self.twiddle2.re);
                let m0607a = vfmaq_n_f32(m0607a, x5p8, self.twiddle4.re);
                let m0607a = vfmaq_n_f32(m0607a, x6p7, self.twiddle3.re);
                let m0607b = vmulq_n_f32(x1m12, self.twiddle6.im);
                let m0607b = vfmsq_n_f32(m0607b, x2m11, self.twiddle1.im);
                let m0607b = vfmaq_n_f32(m0607b, x3m10, self.twiddle5.im);
                let m0607b = vfmsq_n_f32(m0607b, x4m9, self.twiddle2.im);
                let m0607b = vfmaq_n_f32(m0607b, x5m8, self.twiddle4.im);
                let m0607b = vfmsq_n_f32(m0607b, x6m7, self.twiddle3.im);
                let (y06, y07) = NeonButterfly::butterfly2_f32(m0607a, m0607b);

                let qy0 = vcombine_f32(vget_low_f32(y00), vget_low_f32(y01));
                let qy1 = vcombine_f32(vget_low_f32(y02), vget_low_f32(y03));
                let qy2 = vcombine_f32(vget_low_f32(y04), vget_low_f32(y05));
                let qy3 = vcombine_f32(vget_low_f32(y06), vget_low_f32(y07));
                let qy4 = vcombine_f32(vget_low_f32(y08), vget_low_f32(y09));
                let qy5 = vcombine_f32(vget_low_f32(y10), vget_low_f32(y11));
                let qy6 = vcombine_f32(vget_low_f32(y12), vget_high_f32(y00));
                let qy7 = vcombine_f32(vget_high_f32(y01), vget_high_f32(y02));
                let qy8 = vcombine_f32(vget_high_f32(y03), vget_high_f32(y04));
                let qy9 = vcombine_f32(vget_high_f32(y05), vget_high_f32(y06));
                let qy10 = vcombine_f32(vget_high_f32(y07), vget_high_f32(y08));
                let qy11 = vcombine_f32(vget_high_f32(y09), vget_high_f32(y10));
                let qy12 = vcombine_f32(vget_high_f32(y11), vget_high_f32(y12));

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
                vst1q_f32(chunk.get_unchecked_mut(24..).as_mut_ptr().cast(), qy12);
            }

            let rem = in_place.chunks_exact_mut(26).into_remainder();

            for chunk in rem.chunks_exact_mut(13) {
                let u0u1 = vld1q_f32(chunk.as_mut_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12 = vld1_f32(chunk.get_unchecked(12..).as_ptr().cast());

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

                let y00 = u0;
                let (x1p12, x1m12) = NeonButterfly::butterfly2h_f32(u1, u12);
                let x1m12 = vh_rotate90_f32(x1m12, vget_low_f32(rot_sign));
                let y00 = vadd_f32(y00, x1p12);
                let (x2p11, x2m11) = NeonButterfly::butterfly2h_f32(u2, u11);
                let x2m11 = vh_rotate90_f32(x2m11, vget_low_f32(rot_sign));
                let y00 = vadd_f32(y00, x2p11);
                let (x3p10, x3m10) = NeonButterfly::butterfly2h_f32(u3, u10);
                let x3m10 = vh_rotate90_f32(x3m10, vget_low_f32(rot_sign));
                let y00 = vadd_f32(y00, x3p10);
                let (x4p9, x4m9) = NeonButterfly::butterfly2h_f32(u4, u9);
                let x4m9 = vh_rotate90_f32(x4m9, vget_low_f32(rot_sign));
                let y00 = vadd_f32(y00, x4p9);
                let (x5p8, x5m8) = NeonButterfly::butterfly2h_f32(u5, u8);
                let x5m8 = vh_rotate90_f32(x5m8, vget_low_f32(rot_sign));
                let y00 = vadd_f32(y00, x5p8);
                let (x6p7, x6m7) = NeonButterfly::butterfly2h_f32(u6, u7);
                let x6m7 = vh_rotate90_f32(x6m7, vget_low_f32(rot_sign));
                let y00 = vadd_f32(y00, x6p7);

                let m0112a = vfma_n_f32(u0, x1p12, self.twiddle1.re);
                let m0112a = vfma_n_f32(m0112a, x2p11, self.twiddle2.re);
                let m0112a = vfma_n_f32(m0112a, x3p10, self.twiddle3.re);
                let m0112a = vfma_n_f32(m0112a, x4p9, self.twiddle4.re);
                let m0112a = vfma_n_f32(m0112a, x5p8, self.twiddle5.re);
                let m0112a = vfma_n_f32(m0112a, x6p7, self.twiddle6.re);
                let m0112b = vmul_n_f32(x1m12, self.twiddle1.im);
                let m0112b = vfma_n_f32(m0112b, x2m11, self.twiddle2.im);
                let m0112b = vfma_n_f32(m0112b, x3m10, self.twiddle3.im);
                let m0112b = vfma_n_f32(m0112b, x4m9, self.twiddle4.im);
                let m0112b = vfma_n_f32(m0112b, x5m8, self.twiddle5.im);
                let m0112b = vfma_n_f32(m0112b, x6m7, self.twiddle6.im);
                let (y01, y12) = NeonButterfly::butterfly2h_f32(m0112a, m0112b);

                let m0211a = vfma_n_f32(u0, x1p12, self.twiddle2.re);
                let m0211a = vfma_n_f32(m0211a, x2p11, self.twiddle4.re);
                let m0211a = vfma_n_f32(m0211a, x3p10, self.twiddle6.re);
                let m0211a = vfma_n_f32(m0211a, x4p9, self.twiddle5.re);
                let m0211a = vfma_n_f32(m0211a, x5p8, self.twiddle3.re);
                let m0211a = vfma_n_f32(m0211a, x6p7, self.twiddle1.re);
                let m0211b = vmul_n_f32(x1m12, self.twiddle2.im);
                let m0211b = vfma_n_f32(m0211b, x2m11, self.twiddle4.im);
                let m0211b = vfma_n_f32(m0211b, x3m10, self.twiddle6.im);
                let m0211b = vfms_n_f32(m0211b, x4m9, self.twiddle5.im);
                let m0211b = vfms_n_f32(m0211b, x5m8, self.twiddle3.im);
                let m0211b = vfms_n_f32(m0211b, x6m7, self.twiddle1.im);
                let (y02, y11) = NeonButterfly::butterfly2h_f32(m0211a, m0211b);

                let m0310a = vfma_n_f32(u0, x1p12, self.twiddle3.re);
                let m0310a = vfma_n_f32(m0310a, x2p11, self.twiddle6.re);
                let m0310a = vfma_n_f32(m0310a, x3p10, self.twiddle4.re);
                let m0310a = vfma_n_f32(m0310a, x4p9, self.twiddle1.re);
                let m0310a = vfma_n_f32(m0310a, x5p8, self.twiddle2.re);
                let m0310a = vfma_n_f32(m0310a, x6p7, self.twiddle5.re);
                let m0310b = vmul_n_f32(x1m12, self.twiddle3.im);
                let m0310b = vfma_n_f32(m0310b, x2m11, self.twiddle6.im);
                let m0310b = vfms_n_f32(m0310b, x3m10, self.twiddle4.im);
                let m0310b = vfms_n_f32(m0310b, x4m9, self.twiddle1.im);
                let m0310b = vfma_n_f32(m0310b, x5m8, self.twiddle2.im);
                let m0310b = vfma_n_f32(m0310b, x6m7, self.twiddle5.im);
                let (y03, y10) = NeonButterfly::butterfly2h_f32(m0310a, m0310b);

                let m0409a = vfma_n_f32(u0, x1p12, self.twiddle4.re);
                let m0409a = vfma_n_f32(m0409a, x2p11, self.twiddle5.re);
                let m0409a = vfma_n_f32(m0409a, x3p10, self.twiddle1.re);
                let m0409a = vfma_n_f32(m0409a, x4p9, self.twiddle3.re);
                let m0409a = vfma_n_f32(m0409a, x5p8, self.twiddle6.re);
                let m0409a = vfma_n_f32(m0409a, x6p7, self.twiddle2.re);
                let m0409b = vmul_n_f32(x1m12, self.twiddle4.im);
                let m0409b = vfms_n_f32(m0409b, x2m11, self.twiddle5.im);
                let m0409b = vfms_n_f32(m0409b, x3m10, self.twiddle1.im);
                let m0409b = vfma_n_f32(m0409b, x4m9, self.twiddle3.im);
                let m0409b = vfms_n_f32(m0409b, x5m8, self.twiddle6.im);
                let m0409b = vfms_n_f32(m0409b, x6m7, self.twiddle2.im);
                let (y04, y09) = NeonButterfly::butterfly2h_f32(m0409a, m0409b);

                let m0508a = vfma_n_f32(u0, x1p12, self.twiddle5.re);
                let m0508a = vfma_n_f32(m0508a, x2p11, self.twiddle3.re);
                let m0508a = vfma_n_f32(m0508a, x3p10, self.twiddle2.re);
                let m0508a = vfma_n_f32(m0508a, x4p9, self.twiddle6.re);
                let m0508a = vfma_n_f32(m0508a, x5p8, self.twiddle1.re);
                let m0508a = vfma_n_f32(m0508a, x6p7, self.twiddle4.re);
                let m0508b = vmul_n_f32(x1m12, self.twiddle5.im);
                let m0508b = vfms_n_f32(m0508b, x2m11, self.twiddle3.im);
                let m0508b = vfma_n_f32(m0508b, x3m10, self.twiddle2.im);
                let m0508b = vfms_n_f32(m0508b, x4m9, self.twiddle6.im);
                let m0508b = vfms_n_f32(m0508b, x5m8, self.twiddle1.im);
                let m0508b = vfma_n_f32(m0508b, x6m7, self.twiddle4.im);
                let (y05, y08) = NeonButterfly::butterfly2h_f32(m0508a, m0508b);

                let m0607a = vfma_n_f32(u0, x1p12, self.twiddle6.re);
                let m0607a = vfma_n_f32(m0607a, x2p11, self.twiddle1.re);
                let m0607a = vfma_n_f32(m0607a, x3p10, self.twiddle5.re);
                let m0607a = vfma_n_f32(m0607a, x4p9, self.twiddle2.re);
                let m0607a = vfma_n_f32(m0607a, x5p8, self.twiddle4.re);
                let m0607a = vfma_n_f32(m0607a, x6p7, self.twiddle3.re);
                let m0607b = vmul_n_f32(x1m12, self.twiddle6.im);
                let m0607b = vfms_n_f32(m0607b, x2m11, self.twiddle1.im);
                let m0607b = vfma_n_f32(m0607b, x3m10, self.twiddle5.im);
                let m0607b = vfms_n_f32(m0607b, x4m9, self.twiddle2.im);
                let m0607b = vfma_n_f32(m0607b, x5m8, self.twiddle4.im);
                let m0607b = vfms_n_f32(m0607b, x6m7, self.twiddle3.im);
                let (y06, y07) = NeonButterfly::butterfly2h_f32(m0607a, m0607b);

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
                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(y10, y11),
                );
                vst1_f32(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y12);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        13
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Butterfly13;
    use rand::Rng;

    #[test]
    fn test_butterfly13_f32() {
        for i in 1..5 {
            let size = 13usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix13_reference = Butterfly13::new(FftDirection::Forward);
            let radix13_inv_reference = Butterfly13::new(FftDirection::Inverse);

            let radix_forward = NeonButterfly13::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly13::new(FftDirection::Inverse);
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

            input = input.iter().map(|&x| x * (1.0 / 13f32)).collect();

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
    fn test_butterfly13_f64() {
        for i in 1..5 {
            let size = 13usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();

            let mut z_ref = input.to_vec();

            let radix9_reference = Butterfly13::new(FftDirection::Forward);

            let radix_forward = NeonButterfly13::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly13::new(FftDirection::Inverse);
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

            input = input.iter().map(|&x| x * (1.0 / 13f64)).collect();

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
