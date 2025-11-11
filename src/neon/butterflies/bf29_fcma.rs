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

pub(crate) struct NeonFcmaButterfly29<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    twiddle7: Complex<T>,
    twiddle8: Complex<T>,
    twiddle9: Complex<T>,
    twiddle10: Complex<T>,
    twiddle11: Complex<T>,
    twiddle12: Complex<T>,
    twiddle13: Complex<T>,
    twiddle14: Complex<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonFcmaButterfly29<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 29, fft_direction),
            twiddle2: compute_twiddle(2, 29, fft_direction),
            twiddle3: compute_twiddle(3, 29, fft_direction),
            twiddle4: compute_twiddle(4, 29, fft_direction),
            twiddle5: compute_twiddle(5, 29, fft_direction),
            twiddle6: compute_twiddle(6, 29, fft_direction),
            twiddle7: compute_twiddle(7, 29, fft_direction),
            twiddle8: compute_twiddle(8, 29, fft_direction),
            twiddle9: compute_twiddle(9, 29, fft_direction),
            twiddle10: compute_twiddle(10, 29, fft_direction),
            twiddle11: compute_twiddle(11, 29, fft_direction),
            twiddle12: compute_twiddle(12, 29, fft_direction),
            twiddle13: compute_twiddle(13, 29, fft_direction),
            twiddle14: compute_twiddle(14, 29, fft_direction),
        }
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly29<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        29
    }
}

impl NeonFcmaButterfly29<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 29 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(29) {
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
                let u17 = vld1q_f64(chunk.get_unchecked(17..).as_ptr().cast());
                let u18 = vld1q_f64(chunk.get_unchecked(18..).as_ptr().cast());
                let u19 = vld1q_f64(chunk.get_unchecked(19..).as_ptr().cast());
                let u20 = vld1q_f64(chunk.get_unchecked(20..).as_ptr().cast());
                let u21 = vld1q_f64(chunk.get_unchecked(21..).as_ptr().cast());
                let u22 = vld1q_f64(chunk.get_unchecked(22..).as_ptr().cast());
                let u23 = vld1q_f64(chunk.get_unchecked(23..).as_ptr().cast());
                let u24 = vld1q_f64(chunk.get_unchecked(24..).as_ptr().cast());
                let u25 = vld1q_f64(chunk.get_unchecked(25..).as_ptr().cast());
                let u26 = vld1q_f64(chunk.get_unchecked(26..).as_ptr().cast());
                let u27 = vld1q_f64(chunk.get_unchecked(27..).as_ptr().cast());
                let u28 = vld1q_f64(chunk.get_unchecked(28..).as_ptr().cast());

                let y00 = u0;
                let (x1p28, x1m28) = NeonButterfly::butterfly2_f64(u1, u28);
                let x1m28 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m28);
                let y00 = vaddq_f64(y00, x1p28);
                let (x2p27, x2m27) = NeonButterfly::butterfly2_f64(u2, u27);
                let x2m27 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m27);
                let y00 = vaddq_f64(y00, x2p27);
                let (x3p26, x3m26) = NeonButterfly::butterfly2_f64(u3, u26);
                let x3m26 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m26);
                let y00 = vaddq_f64(y00, x3p26);
                let (x4p25, x4m25) = NeonButterfly::butterfly2_f64(u4, u25);
                let x4m25 = vcaddq_rot90_f64(vdupq_n_f64(0.), x4m25);
                let y00 = vaddq_f64(y00, x4p25);
                let (x5p24, x5m24) = NeonButterfly::butterfly2_f64(u5, u24);
                let x5m24 = vcaddq_rot90_f64(vdupq_n_f64(0.), x5m24);
                let y00 = vaddq_f64(y00, x5p24);
                let (x6p23, x6m23) = NeonButterfly::butterfly2_f64(u6, u23);
                let x6m23 = vcaddq_rot90_f64(vdupq_n_f64(0.), x6m23);
                let y00 = vaddq_f64(y00, x6p23);
                let (x7p22, x7m22) = NeonButterfly::butterfly2_f64(u7, u22);
                let x7m22 = vcaddq_rot90_f64(vdupq_n_f64(0.), x7m22);
                let y00 = vaddq_f64(y00, x7p22);
                let (x8p21, x8m21) = NeonButterfly::butterfly2_f64(u8, u21);
                let x8m21 = vcaddq_rot90_f64(vdupq_n_f64(0.), x8m21);
                let y00 = vaddq_f64(y00, x8p21);
                let (x9p20, x9m20) = NeonButterfly::butterfly2_f64(u9, u20);
                let x9m20 = vcaddq_rot90_f64(vdupq_n_f64(0.), x9m20);
                let y00 = vaddq_f64(y00, x9p20);
                let (x10p19, x10m19) = NeonButterfly::butterfly2_f64(u10, u19);
                let x10m19 = vcaddq_rot90_f64(vdupq_n_f64(0.), x10m19);
                let y00 = vaddq_f64(y00, x10p19);
                let (x11p18, x11m18) = NeonButterfly::butterfly2_f64(u11, u18);
                let x11m18 = vcaddq_rot90_f64(vdupq_n_f64(0.), x11m18);
                let y00 = vaddq_f64(y00, x11p18);
                let (x12p17, x12m17) = NeonButterfly::butterfly2_f64(u12, u17);
                let x12m17 = vcaddq_rot90_f64(vdupq_n_f64(0.), x12m17);
                let y00 = vaddq_f64(y00, x12p17);
                let (x13p16, x13m16) = NeonButterfly::butterfly2_f64(u13, u16);
                let x13m16 = vcaddq_rot90_f64(vdupq_n_f64(0.), x13m16);
                let y00 = vaddq_f64(y00, x13p16);
                let (x14p15, x14m15) = NeonButterfly::butterfly2_f64(u14, u15);
                let x14m15 = vcaddq_rot90_f64(vdupq_n_f64(0.), x14m15);
                let y00 = vaddq_f64(y00, x14p15);

                vst1q_f64(chunk.as_mut_ptr().cast(), y00);

                let m0128a = vfmaq_n_f64(u0, x1p28, self.twiddle1.re);
                let m0128a = vfmaq_n_f64(m0128a, x2p27, self.twiddle2.re);
                let m0128a = vfmaq_n_f64(m0128a, x3p26, self.twiddle3.re);
                let m0128a = vfmaq_n_f64(m0128a, x4p25, self.twiddle4.re);
                let m0128a = vfmaq_n_f64(m0128a, x5p24, self.twiddle5.re);
                let m0128a = vfmaq_n_f64(m0128a, x6p23, self.twiddle6.re);
                let m0128a = vfmaq_n_f64(m0128a, x7p22, self.twiddle7.re);
                let m0128a = vfmaq_n_f64(m0128a, x8p21, self.twiddle8.re);
                let m0128a = vfmaq_n_f64(m0128a, x9p20, self.twiddle9.re);
                let m0128a = vfmaq_n_f64(m0128a, x10p19, self.twiddle10.re);
                let m0128a = vfmaq_n_f64(m0128a, x11p18, self.twiddle11.re);
                let m0128a = vfmaq_n_f64(m0128a, x12p17, self.twiddle12.re);
                let m0128a = vfmaq_n_f64(m0128a, x13p16, self.twiddle13.re);
                let m0128a = vfmaq_n_f64(m0128a, x14p15, self.twiddle14.re);
                let m0128b = vmulq_n_f64(x1m28, self.twiddle1.im);
                let m0128b = vfmaq_n_f64(m0128b, x2m27, self.twiddle2.im);
                let m0128b = vfmaq_n_f64(m0128b, x3m26, self.twiddle3.im);
                let m0128b = vfmaq_n_f64(m0128b, x4m25, self.twiddle4.im);
                let m0128b = vfmaq_n_f64(m0128b, x5m24, self.twiddle5.im);
                let m0128b = vfmaq_n_f64(m0128b, x6m23, self.twiddle6.im);
                let m0128b = vfmaq_n_f64(m0128b, x7m22, self.twiddle7.im);
                let m0128b = vfmaq_n_f64(m0128b, x8m21, self.twiddle8.im);
                let m0128b = vfmaq_n_f64(m0128b, x9m20, self.twiddle9.im);
                let m0128b = vfmaq_n_f64(m0128b, x10m19, self.twiddle10.im);
                let m0128b = vfmaq_n_f64(m0128b, x11m18, self.twiddle11.im);
                let m0128b = vfmaq_n_f64(m0128b, x12m17, self.twiddle12.im);
                let m0128b = vfmaq_n_f64(m0128b, x13m16, self.twiddle13.im);
                let m0128b = vfmaq_n_f64(m0128b, x14m15, self.twiddle14.im);
                let (y01, y28) = NeonButterfly::butterfly2_f64(m0128a, m0128b);

                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y01);
                vst1q_f64(chunk.get_unchecked_mut(28..).as_mut_ptr().cast(), y28);

                let m0227a = vfmaq_n_f64(u0, x1p28, self.twiddle2.re);
                let m0227a = vfmaq_n_f64(m0227a, x2p27, self.twiddle4.re);
                let m0227a = vfmaq_n_f64(m0227a, x3p26, self.twiddle6.re);
                let m0227a = vfmaq_n_f64(m0227a, x4p25, self.twiddle8.re);
                let m0227a = vfmaq_n_f64(m0227a, x5p24, self.twiddle10.re);
                let m0227a = vfmaq_n_f64(m0227a, x6p23, self.twiddle12.re);
                let m0227a = vfmaq_n_f64(m0227a, x7p22, self.twiddle14.re);
                let m0227a = vfmaq_n_f64(m0227a, x8p21, self.twiddle13.re);
                let m0227a = vfmaq_n_f64(m0227a, x9p20, self.twiddle11.re);
                let m0227a = vfmaq_n_f64(m0227a, x10p19, self.twiddle9.re);
                let m0227a = vfmaq_n_f64(m0227a, x11p18, self.twiddle7.re);
                let m0227a = vfmaq_n_f64(m0227a, x12p17, self.twiddle5.re);
                let m0227a = vfmaq_n_f64(m0227a, x13p16, self.twiddle3.re);
                let m0227a = vfmaq_n_f64(m0227a, x14p15, self.twiddle1.re);
                let m0227b = vmulq_n_f64(x1m28, self.twiddle2.im);
                let m0227b = vfmaq_n_f64(m0227b, x2m27, self.twiddle4.im);
                let m0227b = vfmaq_n_f64(m0227b, x3m26, self.twiddle6.im);
                let m0227b = vfmaq_n_f64(m0227b, x4m25, self.twiddle8.im);
                let m0227b = vfmaq_n_f64(m0227b, x5m24, self.twiddle10.im);
                let m0227b = vfmaq_n_f64(m0227b, x6m23, self.twiddle12.im);
                let m0227b = vfmaq_n_f64(m0227b, x7m22, self.twiddle14.im);
                let m0227b = vfmsq_n_f64(m0227b, x8m21, self.twiddle13.im);
                let m0227b = vfmsq_n_f64(m0227b, x9m20, self.twiddle11.im);
                let m0227b = vfmsq_n_f64(m0227b, x10m19, self.twiddle9.im);
                let m0227b = vfmsq_n_f64(m0227b, x11m18, self.twiddle7.im);
                let m0227b = vfmsq_n_f64(m0227b, x12m17, self.twiddle5.im);
                let m0227b = vfmsq_n_f64(m0227b, x13m16, self.twiddle3.im);
                let m0227b = vfmsq_n_f64(m0227b, x14m15, self.twiddle1.im);
                let (y02, y27) = NeonButterfly::butterfly2_f64(m0227a, m0227b);

                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y02);
                vst1q_f64(chunk.get_unchecked_mut(27..).as_mut_ptr().cast(), y27);

                let m0326a = vfmaq_n_f64(u0, x1p28, self.twiddle3.re);
                let m0326a = vfmaq_n_f64(m0326a, x2p27, self.twiddle6.re);
                let m0326a = vfmaq_n_f64(m0326a, x3p26, self.twiddle9.re);
                let m0326a = vfmaq_n_f64(m0326a, x4p25, self.twiddle12.re);
                let m0326a = vfmaq_n_f64(m0326a, x5p24, self.twiddle14.re);
                let m0326a = vfmaq_n_f64(m0326a, x6p23, self.twiddle11.re);
                let m0326a = vfmaq_n_f64(m0326a, x7p22, self.twiddle8.re);
                let m0326a = vfmaq_n_f64(m0326a, x8p21, self.twiddle5.re);
                let m0326a = vfmaq_n_f64(m0326a, x9p20, self.twiddle2.re);
                let m0326a = vfmaq_n_f64(m0326a, x10p19, self.twiddle1.re);
                let m0326a = vfmaq_n_f64(m0326a, x11p18, self.twiddle4.re);
                let m0326a = vfmaq_n_f64(m0326a, x12p17, self.twiddle7.re);
                let m0326a = vfmaq_n_f64(m0326a, x13p16, self.twiddle10.re);
                let m0326a = vfmaq_n_f64(m0326a, x14p15, self.twiddle13.re);
                let m0326b = vmulq_n_f64(x1m28, self.twiddle3.im);
                let m0326b = vfmaq_n_f64(m0326b, x2m27, self.twiddle6.im);
                let m0326b = vfmaq_n_f64(m0326b, x3m26, self.twiddle9.im);
                let m0326b = vfmaq_n_f64(m0326b, x4m25, self.twiddle12.im);
                let m0326b = vfmsq_n_f64(m0326b, x5m24, self.twiddle14.im);
                let m0326b = vfmsq_n_f64(m0326b, x6m23, self.twiddle11.im);
                let m0326b = vfmsq_n_f64(m0326b, x7m22, self.twiddle8.im);
                let m0326b = vfmsq_n_f64(m0326b, x8m21, self.twiddle5.im);
                let m0326b = vfmsq_n_f64(m0326b, x9m20, self.twiddle2.im);
                let m0326b = vfmaq_n_f64(m0326b, x10m19, self.twiddle1.im);
                let m0326b = vfmaq_n_f64(m0326b, x11m18, self.twiddle4.im);
                let m0326b = vfmaq_n_f64(m0326b, x12m17, self.twiddle7.im);
                let m0326b = vfmaq_n_f64(m0326b, x13m16, self.twiddle10.im);
                let m0326b = vfmaq_n_f64(m0326b, x14m15, self.twiddle13.im);
                let (y03, y26) = NeonButterfly::butterfly2_f64(m0326a, m0326b);

                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y03);
                vst1q_f64(chunk.get_unchecked_mut(26..).as_mut_ptr().cast(), y26);

                let m0425a = vfmaq_n_f64(u0, x1p28, self.twiddle4.re);
                let m0425a = vfmaq_n_f64(m0425a, x2p27, self.twiddle8.re);
                let m0425a = vfmaq_n_f64(m0425a, x3p26, self.twiddle12.re);
                let m0425a = vfmaq_n_f64(m0425a, x4p25, self.twiddle13.re);
                let m0425a = vfmaq_n_f64(m0425a, x5p24, self.twiddle9.re);
                let m0425a = vfmaq_n_f64(m0425a, x6p23, self.twiddle5.re);
                let m0425a = vfmaq_n_f64(m0425a, x7p22, self.twiddle1.re);
                let m0425a = vfmaq_n_f64(m0425a, x8p21, self.twiddle3.re);
                let m0425a = vfmaq_n_f64(m0425a, x9p20, self.twiddle7.re);
                let m0425a = vfmaq_n_f64(m0425a, x10p19, self.twiddle11.re);
                let m0425a = vfmaq_n_f64(m0425a, x11p18, self.twiddle14.re);
                let m0425a = vfmaq_n_f64(m0425a, x12p17, self.twiddle10.re);
                let m0425a = vfmaq_n_f64(m0425a, x13p16, self.twiddle6.re);
                let m0425a = vfmaq_n_f64(m0425a, x14p15, self.twiddle2.re);
                let m0425b = vmulq_n_f64(x1m28, self.twiddle4.im);
                let m0425b = vfmaq_n_f64(m0425b, x2m27, self.twiddle8.im);
                let m0425b = vfmaq_n_f64(m0425b, x3m26, self.twiddle12.im);
                let m0425b = vfmsq_n_f64(m0425b, x4m25, self.twiddle13.im);
                let m0425b = vfmsq_n_f64(m0425b, x5m24, self.twiddle9.im);
                let m0425b = vfmsq_n_f64(m0425b, x6m23, self.twiddle5.im);
                let m0425b = vfmsq_n_f64(m0425b, x7m22, self.twiddle1.im);
                let m0425b = vfmaq_n_f64(m0425b, x8m21, self.twiddle3.im);
                let m0425b = vfmaq_n_f64(m0425b, x9m20, self.twiddle7.im);
                let m0425b = vfmaq_n_f64(m0425b, x10m19, self.twiddle11.im);
                let m0425b = vfmsq_n_f64(m0425b, x11m18, self.twiddle14.im);
                let m0425b = vfmsq_n_f64(m0425b, x12m17, self.twiddle10.im);
                let m0425b = vfmsq_n_f64(m0425b, x13m16, self.twiddle6.im);
                let m0425b = vfmsq_n_f64(m0425b, x14m15, self.twiddle2.im);
                let (y04, y25) = NeonButterfly::butterfly2_f64(m0425a, m0425b);

                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y04);
                vst1q_f64(chunk.get_unchecked_mut(25..).as_mut_ptr().cast(), y25);

                let m0524a = vfmaq_n_f64(u0, x1p28, self.twiddle5.re);
                let m0524a = vfmaq_n_f64(m0524a, x2p27, self.twiddle10.re);
                let m0524a = vfmaq_n_f64(m0524a, x3p26, self.twiddle14.re);
                let m0524a = vfmaq_n_f64(m0524a, x4p25, self.twiddle9.re);
                let m0524a = vfmaq_n_f64(m0524a, x5p24, self.twiddle4.re);
                let m0524a = vfmaq_n_f64(m0524a, x6p23, self.twiddle1.re);
                let m0524a = vfmaq_n_f64(m0524a, x7p22, self.twiddle6.re);
                let m0524a = vfmaq_n_f64(m0524a, x8p21, self.twiddle11.re);
                let m0524a = vfmaq_n_f64(m0524a, x9p20, self.twiddle13.re);
                let m0524a = vfmaq_n_f64(m0524a, x10p19, self.twiddle8.re);
                let m0524a = vfmaq_n_f64(m0524a, x11p18, self.twiddle3.re);
                let m0524a = vfmaq_n_f64(m0524a, x12p17, self.twiddle2.re);
                let m0524a = vfmaq_n_f64(m0524a, x13p16, self.twiddle7.re);
                let m0524a = vfmaq_n_f64(m0524a, x14p15, self.twiddle12.re);
                let m0524b = vmulq_n_f64(x1m28, self.twiddle5.im);
                let m0524b = vfmaq_n_f64(m0524b, x2m27, self.twiddle10.im);
                let m0524b = vfmsq_n_f64(m0524b, x3m26, self.twiddle14.im);
                let m0524b = vfmsq_n_f64(m0524b, x4m25, self.twiddle9.im);
                let m0524b = vfmsq_n_f64(m0524b, x5m24, self.twiddle4.im);
                let m0524b = vfmaq_n_f64(m0524b, x6m23, self.twiddle1.im);
                let m0524b = vfmaq_n_f64(m0524b, x7m22, self.twiddle6.im);
                let m0524b = vfmaq_n_f64(m0524b, x8m21, self.twiddle11.im);
                let m0524b = vfmsq_n_f64(m0524b, x9m20, self.twiddle13.im);
                let m0524b = vfmsq_n_f64(m0524b, x10m19, self.twiddle8.im);
                let m0524b = vfmsq_n_f64(m0524b, x11m18, self.twiddle3.im);
                let m0524b = vfmaq_n_f64(m0524b, x12m17, self.twiddle2.im);
                let m0524b = vfmaq_n_f64(m0524b, x13m16, self.twiddle7.im);
                let m0524b = vfmaq_n_f64(m0524b, x14m15, self.twiddle12.im);
                let (y05, y24) = NeonButterfly::butterfly2_f64(m0524a, m0524b);

                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y05);
                vst1q_f64(chunk.get_unchecked_mut(24..).as_mut_ptr().cast(), y24);

                let m0623a = vfmaq_n_f64(u0, x1p28, self.twiddle6.re);
                let m0623a = vfmaq_n_f64(m0623a, x2p27, self.twiddle12.re);
                let m0623a = vfmaq_n_f64(m0623a, x3p26, self.twiddle11.re);
                let m0623a = vfmaq_n_f64(m0623a, x4p25, self.twiddle5.re);
                let m0623a = vfmaq_n_f64(m0623a, x5p24, self.twiddle1.re);
                let m0623a = vfmaq_n_f64(m0623a, x6p23, self.twiddle7.re);
                let m0623a = vfmaq_n_f64(m0623a, x7p22, self.twiddle13.re);
                let m0623a = vfmaq_n_f64(m0623a, x8p21, self.twiddle10.re);
                let m0623a = vfmaq_n_f64(m0623a, x9p20, self.twiddle4.re);
                let m0623a = vfmaq_n_f64(m0623a, x10p19, self.twiddle2.re);
                let m0623a = vfmaq_n_f64(m0623a, x11p18, self.twiddle8.re);
                let m0623a = vfmaq_n_f64(m0623a, x12p17, self.twiddle14.re);
                let m0623a = vfmaq_n_f64(m0623a, x13p16, self.twiddle9.re);
                let m0623a = vfmaq_n_f64(m0623a, x14p15, self.twiddle3.re);
                let m0623b = vmulq_n_f64(x1m28, self.twiddle6.im);
                let m0623b = vfmaq_n_f64(m0623b, x2m27, self.twiddle12.im);
                let m0623b = vfmsq_n_f64(m0623b, x3m26, self.twiddle11.im);
                let m0623b = vfmsq_n_f64(m0623b, x4m25, self.twiddle5.im);
                let m0623b = vfmaq_n_f64(m0623b, x5m24, self.twiddle1.im);
                let m0623b = vfmaq_n_f64(m0623b, x6m23, self.twiddle7.im);
                let m0623b = vfmaq_n_f64(m0623b, x7m22, self.twiddle13.im);
                let m0623b = vfmsq_n_f64(m0623b, x8m21, self.twiddle10.im);
                let m0623b = vfmsq_n_f64(m0623b, x9m20, self.twiddle4.im);
                let m0623b = vfmaq_n_f64(m0623b, x10m19, self.twiddle2.im);
                let m0623b = vfmaq_n_f64(m0623b, x11m18, self.twiddle8.im);
                let m0623b = vfmaq_n_f64(m0623b, x12m17, self.twiddle14.im);
                let m0623b = vfmsq_n_f64(m0623b, x13m16, self.twiddle9.im);
                let m0623b = vfmsq_n_f64(m0623b, x14m15, self.twiddle3.im);
                let (y06, y23) = NeonButterfly::butterfly2_f64(m0623a, m0623b);

                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y06);
                vst1q_f64(chunk.get_unchecked_mut(23..).as_mut_ptr().cast(), y23);

                let m0722a = vfmaq_n_f64(u0, x1p28, self.twiddle7.re);
                let m0722a = vfmaq_n_f64(m0722a, x2p27, self.twiddle14.re);
                let m0722a = vfmaq_n_f64(m0722a, x3p26, self.twiddle8.re);
                let m0722a = vfmaq_n_f64(m0722a, x4p25, self.twiddle1.re);
                let m0722a = vfmaq_n_f64(m0722a, x5p24, self.twiddle6.re);
                let m0722a = vfmaq_n_f64(m0722a, x6p23, self.twiddle13.re);
                let m0722a = vfmaq_n_f64(m0722a, x7p22, self.twiddle9.re);
                let m0722a = vfmaq_n_f64(m0722a, x8p21, self.twiddle2.re);
                let m0722a = vfmaq_n_f64(m0722a, x9p20, self.twiddle5.re);
                let m0722a = vfmaq_n_f64(m0722a, x10p19, self.twiddle12.re);
                let m0722a = vfmaq_n_f64(m0722a, x11p18, self.twiddle10.re);
                let m0722a = vfmaq_n_f64(m0722a, x12p17, self.twiddle3.re);
                let m0722a = vfmaq_n_f64(m0722a, x13p16, self.twiddle4.re);
                let m0722a = vfmaq_n_f64(m0722a, x14p15, self.twiddle11.re);
                let m0722b = vmulq_n_f64(x1m28, self.twiddle7.im);
                let m0722b = vfmaq_n_f64(m0722b, x2m27, self.twiddle14.im);
                let m0722b = vfmsq_n_f64(m0722b, x3m26, self.twiddle8.im);
                let m0722b = vfmsq_n_f64(m0722b, x4m25, self.twiddle1.im);
                let m0722b = vfmaq_n_f64(m0722b, x5m24, self.twiddle6.im);
                let m0722b = vfmaq_n_f64(m0722b, x6m23, self.twiddle13.im);
                let m0722b = vfmsq_n_f64(m0722b, x7m22, self.twiddle9.im);
                let m0722b = vfmsq_n_f64(m0722b, x8m21, self.twiddle2.im);
                let m0722b = vfmaq_n_f64(m0722b, x9m20, self.twiddle5.im);
                let m0722b = vfmaq_n_f64(m0722b, x10m19, self.twiddle12.im);
                let m0722b = vfmsq_n_f64(m0722b, x11m18, self.twiddle10.im);
                let m0722b = vfmsq_n_f64(m0722b, x12m17, self.twiddle3.im);
                let m0722b = vfmaq_n_f64(m0722b, x13m16, self.twiddle4.im);
                let m0722b = vfmaq_n_f64(m0722b, x14m15, self.twiddle11.im);
                let (y07, y22) = NeonButterfly::butterfly2_f64(m0722a, m0722b);

                vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y07);
                vst1q_f64(chunk.get_unchecked_mut(22..).as_mut_ptr().cast(), y22);

                let m0821a = vfmaq_n_f64(u0, x1p28, self.twiddle8.re);
                let m0821a = vfmaq_n_f64(m0821a, x2p27, self.twiddle13.re);
                let m0821a = vfmaq_n_f64(m0821a, x3p26, self.twiddle5.re);
                let m0821a = vfmaq_n_f64(m0821a, x4p25, self.twiddle3.re);
                let m0821a = vfmaq_n_f64(m0821a, x5p24, self.twiddle11.re);
                let m0821a = vfmaq_n_f64(m0821a, x6p23, self.twiddle10.re);
                let m0821a = vfmaq_n_f64(m0821a, x7p22, self.twiddle2.re);
                let m0821a = vfmaq_n_f64(m0821a, x8p21, self.twiddle6.re);
                let m0821a = vfmaq_n_f64(m0821a, x9p20, self.twiddle14.re);
                let m0821a = vfmaq_n_f64(m0821a, x10p19, self.twiddle7.re);
                let m0821a = vfmaq_n_f64(m0821a, x11p18, self.twiddle1.re);
                let m0821a = vfmaq_n_f64(m0821a, x12p17, self.twiddle9.re);
                let m0821a = vfmaq_n_f64(m0821a, x13p16, self.twiddle12.re);
                let m0821a = vfmaq_n_f64(m0821a, x14p15, self.twiddle4.re);
                let m0821b = vmulq_n_f64(x1m28, self.twiddle8.im);
                let m0821b = vfmsq_n_f64(m0821b, x2m27, self.twiddle13.im);
                let m0821b = vfmsq_n_f64(m0821b, x3m26, self.twiddle5.im);
                let m0821b = vfmaq_n_f64(m0821b, x4m25, self.twiddle3.im);
                let m0821b = vfmaq_n_f64(m0821b, x5m24, self.twiddle11.im);
                let m0821b = vfmsq_n_f64(m0821b, x6m23, self.twiddle10.im);
                let m0821b = vfmsq_n_f64(m0821b, x7m22, self.twiddle2.im);
                let m0821b = vfmaq_n_f64(m0821b, x8m21, self.twiddle6.im);
                let m0821b = vfmaq_n_f64(m0821b, x9m20, self.twiddle14.im);
                let m0821b = vfmsq_n_f64(m0821b, x10m19, self.twiddle7.im);
                let m0821b = vfmaq_n_f64(m0821b, x11m18, self.twiddle1.im);
                let m0821b = vfmaq_n_f64(m0821b, x12m17, self.twiddle9.im);
                let m0821b = vfmsq_n_f64(m0821b, x13m16, self.twiddle12.im);
                let m0821b = vfmsq_n_f64(m0821b, x14m15, self.twiddle4.im);
                let (y08, y21) = NeonButterfly::butterfly2_f64(m0821a, m0821b);

                vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y08);
                vst1q_f64(chunk.get_unchecked_mut(21..).as_mut_ptr().cast(), y21);

                let m0920a = vfmaq_n_f64(u0, x1p28, self.twiddle9.re);
                let m0920a = vfmaq_n_f64(m0920a, x2p27, self.twiddle11.re);
                let m0920a = vfmaq_n_f64(m0920a, x3p26, self.twiddle2.re);
                let m0920a = vfmaq_n_f64(m0920a, x4p25, self.twiddle7.re);
                let m0920a = vfmaq_n_f64(m0920a, x5p24, self.twiddle13.re);
                let m0920a = vfmaq_n_f64(m0920a, x6p23, self.twiddle4.re);
                let m0920a = vfmaq_n_f64(m0920a, x7p22, self.twiddle5.re);
                let m0920a = vfmaq_n_f64(m0920a, x8p21, self.twiddle14.re);
                let m0920a = vfmaq_n_f64(m0920a, x9p20, self.twiddle6.re);
                let m0920a = vfmaq_n_f64(m0920a, x10p19, self.twiddle3.re);
                let m0920a = vfmaq_n_f64(m0920a, x11p18, self.twiddle12.re);
                let m0920a = vfmaq_n_f64(m0920a, x12p17, self.twiddle8.re);
                let m0920a = vfmaq_n_f64(m0920a, x13p16, self.twiddle1.re);
                let m0920a = vfmaq_n_f64(m0920a, x14p15, self.twiddle10.re);
                let m0920b = vmulq_n_f64(x1m28, self.twiddle9.im);
                let m0920b = vfmsq_n_f64(m0920b, x2m27, self.twiddle11.im);
                let m0920b = vfmsq_n_f64(m0920b, x3m26, self.twiddle2.im);
                let m0920b = vfmaq_n_f64(m0920b, x4m25, self.twiddle7.im);
                let m0920b = vfmsq_n_f64(m0920b, x5m24, self.twiddle13.im);
                let m0920b = vfmsq_n_f64(m0920b, x6m23, self.twiddle4.im);
                let m0920b = vfmaq_n_f64(m0920b, x7m22, self.twiddle5.im);
                let m0920b = vfmaq_n_f64(m0920b, x8m21, self.twiddle14.im);
                let m0920b = vfmsq_n_f64(m0920b, x9m20, self.twiddle6.im);
                let m0920b = vfmaq_n_f64(m0920b, x10m19, self.twiddle3.im);
                let m0920b = vfmaq_n_f64(m0920b, x11m18, self.twiddle12.im);
                let m0920b = vfmsq_n_f64(m0920b, x12m17, self.twiddle8.im);
                let m0920b = vfmaq_n_f64(m0920b, x13m16, self.twiddle1.im);
                let m0920b = vfmaq_n_f64(m0920b, x14m15, self.twiddle10.im);
                let (y09, y20) = NeonButterfly::butterfly2_f64(m0920a, m0920b);

                vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y09);
                vst1q_f64(chunk.get_unchecked_mut(20..).as_mut_ptr().cast(), y20);

                let m1019a = vfmaq_n_f64(u0, x1p28, self.twiddle10.re);
                let m1019a = vfmaq_n_f64(m1019a, x2p27, self.twiddle9.re);
                let m1019a = vfmaq_n_f64(m1019a, x3p26, self.twiddle1.re);
                let m1019a = vfmaq_n_f64(m1019a, x4p25, self.twiddle11.re);
                let m1019a = vfmaq_n_f64(m1019a, x5p24, self.twiddle8.re);
                let m1019a = vfmaq_n_f64(m1019a, x6p23, self.twiddle2.re);
                let m1019a = vfmaq_n_f64(m1019a, x7p22, self.twiddle12.re);
                let m1019a = vfmaq_n_f64(m1019a, x8p21, self.twiddle7.re);
                let m1019a = vfmaq_n_f64(m1019a, x9p20, self.twiddle3.re);
                let m1019a = vfmaq_n_f64(m1019a, x10p19, self.twiddle13.re);
                let m1019a = vfmaq_n_f64(m1019a, x11p18, self.twiddle6.re);
                let m1019a = vfmaq_n_f64(m1019a, x12p17, self.twiddle4.re);
                let m1019a = vfmaq_n_f64(m1019a, x13p16, self.twiddle14.re);
                let m1019a = vfmaq_n_f64(m1019a, x14p15, self.twiddle5.re);
                let m1019b = vmulq_n_f64(x1m28, self.twiddle10.im);
                let m1019b = vfmsq_n_f64(m1019b, x2m27, self.twiddle9.im);
                let m1019b = vfmaq_n_f64(m1019b, x3m26, self.twiddle1.im);
                let m1019b = vfmaq_n_f64(m1019b, x4m25, self.twiddle11.im);
                let m1019b = vfmsq_n_f64(m1019b, x5m24, self.twiddle8.im);
                let m1019b = vfmaq_n_f64(m1019b, x6m23, self.twiddle2.im);
                let m1019b = vfmaq_n_f64(m1019b, x7m22, self.twiddle12.im);
                let m1019b = vfmsq_n_f64(m1019b, x8m21, self.twiddle7.im);
                let m1019b = vfmaq_n_f64(m1019b, x9m20, self.twiddle3.im);
                let m1019b = vfmaq_n_f64(m1019b, x10m19, self.twiddle13.im);
                let m1019b = vfmsq_n_f64(m1019b, x11m18, self.twiddle6.im);
                let m1019b = vfmaq_n_f64(m1019b, x12m17, self.twiddle4.im);
                let m1019b = vfmaq_n_f64(m1019b, x13m16, self.twiddle14.im);
                let m1019b = vfmsq_n_f64(m1019b, x14m15, self.twiddle5.im);
                let (y10, y19) = NeonButterfly::butterfly2_f64(m1019a, m1019b);

                vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);
                vst1q_f64(chunk.get_unchecked_mut(19..).as_mut_ptr().cast(), y19);

                let m1118a = vfmaq_n_f64(u0, x1p28, self.twiddle11.re);
                let m1118a = vfmaq_n_f64(m1118a, x2p27, self.twiddle7.re);
                let m1118a = vfmaq_n_f64(m1118a, x3p26, self.twiddle4.re);
                let m1118a = vfmaq_n_f64(m1118a, x4p25, self.twiddle14.re);
                let m1118a = vfmaq_n_f64(m1118a, x5p24, self.twiddle3.re);
                let m1118a = vfmaq_n_f64(m1118a, x6p23, self.twiddle8.re);
                let m1118a = vfmaq_n_f64(m1118a, x7p22, self.twiddle10.re);
                let m1118a = vfmaq_n_f64(m1118a, x8p21, self.twiddle1.re);
                let m1118a = vfmaq_n_f64(m1118a, x9p20, self.twiddle12.re);
                let m1118a = vfmaq_n_f64(m1118a, x10p19, self.twiddle6.re);
                let m1118a = vfmaq_n_f64(m1118a, x11p18, self.twiddle5.re);
                let m1118a = vfmaq_n_f64(m1118a, x12p17, self.twiddle13.re);
                let m1118a = vfmaq_n_f64(m1118a, x13p16, self.twiddle2.re);
                let m1118a = vfmaq_n_f64(m1118a, x14p15, self.twiddle9.re);
                let m1118b = vmulq_n_f64(x1m28, self.twiddle11.im);
                let m1118b = vfmsq_n_f64(m1118b, x2m27, self.twiddle7.im);
                let m1118b = vfmaq_n_f64(m1118b, x3m26, self.twiddle4.im);
                let m1118b = vfmsq_n_f64(m1118b, x4m25, self.twiddle14.im);
                let m1118b = vfmsq_n_f64(m1118b, x5m24, self.twiddle3.im);
                let m1118b = vfmaq_n_f64(m1118b, x6m23, self.twiddle8.im);
                let m1118b = vfmsq_n_f64(m1118b, x7m22, self.twiddle10.im);
                let m1118b = vfmaq_n_f64(m1118b, x8m21, self.twiddle1.im);
                let m1118b = vfmaq_n_f64(m1118b, x9m20, self.twiddle12.im);
                let m1118b = vfmsq_n_f64(m1118b, x10m19, self.twiddle6.im);
                let m1118b = vfmaq_n_f64(m1118b, x11m18, self.twiddle5.im);
                let m1118b = vfmsq_n_f64(m1118b, x12m17, self.twiddle13.im);
                let m1118b = vfmsq_n_f64(m1118b, x13m16, self.twiddle2.im);
                let m1118b = vfmaq_n_f64(m1118b, x14m15, self.twiddle9.im);
                let (y11, y18) = NeonButterfly::butterfly2_f64(m1118a, m1118b);

                vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), y11);
                vst1q_f64(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), y18);

                let m1217a = vfmaq_n_f64(u0, x1p28, self.twiddle12.re);
                let m1217a = vfmaq_n_f64(m1217a, x2p27, self.twiddle5.re);
                let m1217a = vfmaq_n_f64(m1217a, x3p26, self.twiddle7.re);
                let m1217a = vfmaq_n_f64(m1217a, x4p25, self.twiddle10.re);
                let m1217a = vfmaq_n_f64(m1217a, x5p24, self.twiddle2.re);
                let m1217a = vfmaq_n_f64(m1217a, x6p23, self.twiddle14.re);
                let m1217a = vfmaq_n_f64(m1217a, x7p22, self.twiddle3.re);
                let m1217a = vfmaq_n_f64(m1217a, x8p21, self.twiddle9.re);
                let m1217a = vfmaq_n_f64(m1217a, x9p20, self.twiddle8.re);
                let m1217a = vfmaq_n_f64(m1217a, x10p19, self.twiddle4.re);
                let m1217a = vfmaq_n_f64(m1217a, x11p18, self.twiddle13.re);
                let m1217a = vfmaq_n_f64(m1217a, x12p17, self.twiddle1.re);
                let m1217a = vfmaq_n_f64(m1217a, x13p16, self.twiddle11.re);
                let m1217a = vfmaq_n_f64(m1217a, x14p15, self.twiddle6.re);
                let m1217b = vmulq_n_f64(x1m28, self.twiddle12.im);
                let m1217b = vfmsq_n_f64(m1217b, x2m27, self.twiddle5.im);
                let m1217b = vfmaq_n_f64(m1217b, x3m26, self.twiddle7.im);
                let m1217b = vfmsq_n_f64(m1217b, x4m25, self.twiddle10.im);
                let m1217b = vfmaq_n_f64(m1217b, x5m24, self.twiddle2.im);
                let m1217b = vfmaq_n_f64(m1217b, x6m23, self.twiddle14.im);
                let m1217b = vfmsq_n_f64(m1217b, x7m22, self.twiddle3.im);
                let m1217b = vfmaq_n_f64(m1217b, x8m21, self.twiddle9.im);
                let m1217b = vfmsq_n_f64(m1217b, x9m20, self.twiddle8.im);
                let m1217b = vfmaq_n_f64(m1217b, x10m19, self.twiddle4.im);
                let m1217b = vfmsq_n_f64(m1217b, x11m18, self.twiddle13.im);
                let m1217b = vfmsq_n_f64(m1217b, x12m17, self.twiddle1.im);
                let m1217b = vfmaq_n_f64(m1217b, x13m16, self.twiddle11.im);
                let m1217b = vfmsq_n_f64(m1217b, x14m15, self.twiddle6.im);
                let (y12, y17) = NeonButterfly::butterfly2_f64(m1217a, m1217b);

                vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y12);
                vst1q_f64(chunk.get_unchecked_mut(17..).as_mut_ptr().cast(), y17);

                let m1316a = vfmaq_n_f64(u0, x1p28, self.twiddle13.re);
                let m1316a = vfmaq_n_f64(m1316a, x2p27, self.twiddle3.re);
                let m1316a = vfmaq_n_f64(m1316a, x3p26, self.twiddle10.re);
                let m1316a = vfmaq_n_f64(m1316a, x4p25, self.twiddle6.re);
                let m1316a = vfmaq_n_f64(m1316a, x5p24, self.twiddle7.re);
                let m1316a = vfmaq_n_f64(m1316a, x6p23, self.twiddle9.re);
                let m1316a = vfmaq_n_f64(m1316a, x7p22, self.twiddle4.re);
                let m1316a = vfmaq_n_f64(m1316a, x8p21, self.twiddle12.re);
                let m1316a = vfmaq_n_f64(m1316a, x9p20, self.twiddle1.re);
                let m1316a = vfmaq_n_f64(m1316a, x10p19, self.twiddle14.re);
                let m1316a = vfmaq_n_f64(m1316a, x11p18, self.twiddle2.re);
                let m1316a = vfmaq_n_f64(m1316a, x12p17, self.twiddle11.re);
                let m1316a = vfmaq_n_f64(m1316a, x13p16, self.twiddle5.re);
                let m1316a = vfmaq_n_f64(m1316a, x14p15, self.twiddle8.re);
                let m1316b = vmulq_n_f64(x1m28, self.twiddle13.im);
                let m1316b = vfmsq_n_f64(m1316b, x2m27, self.twiddle3.im);
                let m1316b = vfmaq_n_f64(m1316b, x3m26, self.twiddle10.im);
                let m1316b = vfmsq_n_f64(m1316b, x4m25, self.twiddle6.im);
                let m1316b = vfmaq_n_f64(m1316b, x5m24, self.twiddle7.im);
                let m1316b = vfmsq_n_f64(m1316b, x6m23, self.twiddle9.im);
                let m1316b = vfmaq_n_f64(m1316b, x7m22, self.twiddle4.im);
                let m1316b = vfmsq_n_f64(m1316b, x8m21, self.twiddle12.im);
                let m1316b = vfmaq_n_f64(m1316b, x9m20, self.twiddle1.im);
                let m1316b = vfmaq_n_f64(m1316b, x10m19, self.twiddle14.im);
                let m1316b = vfmsq_n_f64(m1316b, x11m18, self.twiddle2.im);
                let m1316b = vfmaq_n_f64(m1316b, x12m17, self.twiddle11.im);
                let m1316b = vfmsq_n_f64(m1316b, x13m16, self.twiddle5.im);
                let m1316b = vfmaq_n_f64(m1316b, x14m15, self.twiddle8.im);
                let (y13, y16) = NeonButterfly::butterfly2_f64(m1316a, m1316b);

                vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), y13);
                vst1q_f64(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), y16);

                let m1415a = vfmaq_n_f64(u0, x1p28, self.twiddle14.re);
                let m1415a = vfmaq_n_f64(m1415a, x2p27, self.twiddle1.re);
                let m1415a = vfmaq_n_f64(m1415a, x3p26, self.twiddle13.re);
                let m1415a = vfmaq_n_f64(m1415a, x4p25, self.twiddle2.re);
                let m1415a = vfmaq_n_f64(m1415a, x5p24, self.twiddle12.re);
                let m1415a = vfmaq_n_f64(m1415a, x6p23, self.twiddle3.re);
                let m1415a = vfmaq_n_f64(m1415a, x7p22, self.twiddle11.re);
                let m1415a = vfmaq_n_f64(m1415a, x8p21, self.twiddle4.re);
                let m1415a = vfmaq_n_f64(m1415a, x9p20, self.twiddle10.re);
                let m1415a = vfmaq_n_f64(m1415a, x10p19, self.twiddle5.re);
                let m1415a = vfmaq_n_f64(m1415a, x11p18, self.twiddle9.re);
                let m1415a = vfmaq_n_f64(m1415a, x12p17, self.twiddle6.re);
                let m1415a = vfmaq_n_f64(m1415a, x13p16, self.twiddle8.re);
                let m1415a = vfmaq_n_f64(m1415a, x14p15, self.twiddle7.re);
                let m1415b = vmulq_n_f64(x1m28, self.twiddle14.im);
                let m1415b = vfmsq_n_f64(m1415b, x2m27, self.twiddle1.im);
                let m1415b = vfmaq_n_f64(m1415b, x3m26, self.twiddle13.im);
                let m1415b = vfmsq_n_f64(m1415b, x4m25, self.twiddle2.im);
                let m1415b = vfmaq_n_f64(m1415b, x5m24, self.twiddle12.im);
                let m1415b = vfmsq_n_f64(m1415b, x6m23, self.twiddle3.im);
                let m1415b = vfmaq_n_f64(m1415b, x7m22, self.twiddle11.im);
                let m1415b = vfmsq_n_f64(m1415b, x8m21, self.twiddle4.im);
                let m1415b = vfmaq_n_f64(m1415b, x9m20, self.twiddle10.im);
                let m1415b = vfmsq_n_f64(m1415b, x10m19, self.twiddle5.im);
                let m1415b = vfmaq_n_f64(m1415b, x11m18, self.twiddle9.im);
                let m1415b = vfmsq_n_f64(m1415b, x12m17, self.twiddle6.im);
                let m1415b = vfmaq_n_f64(m1415b, x13m16, self.twiddle8.im);
                let m1415b = vfmsq_n_f64(m1415b, x14m15, self.twiddle7.im);
                let (y14, y15) = NeonButterfly::butterfly2_f64(m1415a, m1415b);

                vst1q_f64(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y14);
                vst1q_f64(chunk.get_unchecked_mut(15..).as_mut_ptr().cast(), y15);
            }
        }
        Ok(())
    }
}

macro_rules! shift_load {
    ($chunk: expr, $offset0: expr) => {{
        let q0 = vld1q_f32($chunk.get_unchecked($offset0..).as_ptr().cast());
        let q1 = vld1q_f32($chunk.get_unchecked($offset0 + 29..).as_ptr().cast());
        (
            vcombine_f32(vget_low_f32(q0), vget_low_f32(q1)),
            vcombine_f32(vget_high_f32(q0), vget_high_f32(q1)),
        )
    }};
}

macro_rules! shift_store {
    ($chunk: expr, $offset0: expr, $r0: expr, $r1: expr) => {{
        vst1q_f32(
            $chunk.get_unchecked_mut($offset0..).as_mut_ptr().cast(),
            vcombine_f32(vget_low_f32($r0), vget_low_f32($r1)),
        );
        vst1q_f32(
            $chunk
                .get_unchecked_mut($offset0 + 29..)
                .as_mut_ptr()
                .cast(),
            vcombine_f32(vget_high_f32($r0), vget_high_f32($r1)),
        );
    }};
}

macro_rules! shift_storel {
    ($chunk: expr, $offset0: expr, $r0: expr) => {{
        vst1_f32(
            $chunk.get_unchecked_mut($offset0..).as_mut_ptr().cast(),
            vget_low_f32($r0),
        );
        vst1_f32(
            $chunk
                .get_unchecked_mut($offset0 + 29..)
                .as_mut_ptr()
                .cast(),
            vget_high_f32($r0),
        );
    }};
}

macro_rules! shift_loadl {
    ($chunk: expr, $offset0: expr) => {{
        let q0 = vld1_f32($chunk.get_unchecked($offset0..).as_ptr().cast());
        let q1 = vld1_f32($chunk.get_unchecked($offset0 + 29..).as_ptr().cast());
        vcombine_f32(q0, q1)
    }};
}

impl FftExecutor<f32> for NeonFcmaButterfly29<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        29
    }
}

impl NeonFcmaButterfly29<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 29 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(58) {
                let (u0, u1) = shift_load!(chunk, 0);
                let (u2, u3) = shift_load!(chunk, 2);
                let (u4, u5) = shift_load!(chunk, 4);
                let (u6, u7) = shift_load!(chunk, 6);
                let (u8, u9) = shift_load!(chunk, 8);
                let (u10, u11) = shift_load!(chunk, 10);
                let (u12, u13) = shift_load!(chunk, 12);
                let (u14, u15) = shift_load!(chunk, 14);
                let (u16, u17) = shift_load!(chunk, 16);
                let (u18, u19) = shift_load!(chunk, 18);
                let (u20, u21) = shift_load!(chunk, 20);
                let (u22, u23) = shift_load!(chunk, 22);
                let (u24, u25) = shift_load!(chunk, 24);
                let (u26, u27) = shift_load!(chunk, 26);
                let u28 = shift_loadl!(chunk, 28);

                let y00 = u0;
                let (x1p28, x1m28) = NeonButterfly::butterfly2_f32(u1, u28);
                let x1m28 = vcaddq_rot90_f32(vdupq_n_f32(0.), x1m28);
                let y00 = vaddq_f32(y00, x1p28);
                let (x2p27, x2m27) = NeonButterfly::butterfly2_f32(u2, u27);
                let x2m27 = vcaddq_rot90_f32(vdupq_n_f32(0.), x2m27);
                let y00 = vaddq_f32(y00, x2p27);
                let (x3p26, x3m26) = NeonButterfly::butterfly2_f32(u3, u26);
                let x3m26 = vcaddq_rot90_f32(vdupq_n_f32(0.), x3m26);
                let y00 = vaddq_f32(y00, x3p26);
                let (x4p25, x4m25) = NeonButterfly::butterfly2_f32(u4, u25);
                let x4m25 = vcaddq_rot90_f32(vdupq_n_f32(0.), x4m25);
                let y00 = vaddq_f32(y00, x4p25);
                let (x5p24, x5m24) = NeonButterfly::butterfly2_f32(u5, u24);
                let x5m24 = vcaddq_rot90_f32(vdupq_n_f32(0.), x5m24);
                let y00 = vaddq_f32(y00, x5p24);
                let (x6p23, x6m23) = NeonButterfly::butterfly2_f32(u6, u23);
                let x6m23 = vcaddq_rot90_f32(vdupq_n_f32(0.), x6m23);
                let y00 = vaddq_f32(y00, x6p23);
                let (x7p22, x7m22) = NeonButterfly::butterfly2_f32(u7, u22);
                let x7m22 = vcaddq_rot90_f32(vdupq_n_f32(0.), x7m22);
                let y00 = vaddq_f32(y00, x7p22);
                let (x8p21, x8m21) = NeonButterfly::butterfly2_f32(u8, u21);
                let x8m21 = vcaddq_rot90_f32(vdupq_n_f32(0.), x8m21);
                let y00 = vaddq_f32(y00, x8p21);
                let (x9p20, x9m20) = NeonButterfly::butterfly2_f32(u9, u20);
                let x9m20 = vcaddq_rot90_f32(vdupq_n_f32(0.), x9m20);
                let y00 = vaddq_f32(y00, x9p20);
                let (x10p19, x10m19) = NeonButterfly::butterfly2_f32(u10, u19);
                let x10m19 = vcaddq_rot90_f32(vdupq_n_f32(0.), x10m19);
                let y00 = vaddq_f32(y00, x10p19);
                let (x11p18, x11m18) = NeonButterfly::butterfly2_f32(u11, u18);
                let x11m18 = vcaddq_rot90_f32(vdupq_n_f32(0.), x11m18);
                let y00 = vaddq_f32(y00, x11p18);
                let (x12p17, x12m17) = NeonButterfly::butterfly2_f32(u12, u17);
                let x12m17 = vcaddq_rot90_f32(vdupq_n_f32(0.), x12m17);
                let y00 = vaddq_f32(y00, x12p17);
                let (x13p16, x13m16) = NeonButterfly::butterfly2_f32(u13, u16);
                let x13m16 = vcaddq_rot90_f32(vdupq_n_f32(0.), x13m16);
                let y00 = vaddq_f32(y00, x13p16);
                let (x14p15, x14m15) = NeonButterfly::butterfly2_f32(u14, u15);
                let x14m15 = vcaddq_rot90_f32(vdupq_n_f32(0.), x14m15);
                let y00 = vaddq_f32(y00, x14p15);

                let m0128a = vfmaq_n_f32(u0, x1p28, self.twiddle1.re);
                let m0128a = vfmaq_n_f32(m0128a, x2p27, self.twiddle2.re);
                let m0128a = vfmaq_n_f32(m0128a, x3p26, self.twiddle3.re);
                let m0128a = vfmaq_n_f32(m0128a, x4p25, self.twiddle4.re);
                let m0128a = vfmaq_n_f32(m0128a, x5p24, self.twiddle5.re);
                let m0128a = vfmaq_n_f32(m0128a, x6p23, self.twiddle6.re);
                let m0128a = vfmaq_n_f32(m0128a, x7p22, self.twiddle7.re);
                let m0128a = vfmaq_n_f32(m0128a, x8p21, self.twiddle8.re);
                let m0128a = vfmaq_n_f32(m0128a, x9p20, self.twiddle9.re);
                let m0128a = vfmaq_n_f32(m0128a, x10p19, self.twiddle10.re);
                let m0128a = vfmaq_n_f32(m0128a, x11p18, self.twiddle11.re);
                let m0128a = vfmaq_n_f32(m0128a, x12p17, self.twiddle12.re);
                let m0128a = vfmaq_n_f32(m0128a, x13p16, self.twiddle13.re);
                let m0128a = vfmaq_n_f32(m0128a, x14p15, self.twiddle14.re);
                let m0128b = vmulq_n_f32(x1m28, self.twiddle1.im);
                let m0128b = vfmaq_n_f32(m0128b, x2m27, self.twiddle2.im);
                let m0128b = vfmaq_n_f32(m0128b, x3m26, self.twiddle3.im);
                let m0128b = vfmaq_n_f32(m0128b, x4m25, self.twiddle4.im);
                let m0128b = vfmaq_n_f32(m0128b, x5m24, self.twiddle5.im);
                let m0128b = vfmaq_n_f32(m0128b, x6m23, self.twiddle6.im);
                let m0128b = vfmaq_n_f32(m0128b, x7m22, self.twiddle7.im);
                let m0128b = vfmaq_n_f32(m0128b, x8m21, self.twiddle8.im);
                let m0128b = vfmaq_n_f32(m0128b, x9m20, self.twiddle9.im);
                let m0128b = vfmaq_n_f32(m0128b, x10m19, self.twiddle10.im);
                let m0128b = vfmaq_n_f32(m0128b, x11m18, self.twiddle11.im);
                let m0128b = vfmaq_n_f32(m0128b, x12m17, self.twiddle12.im);
                let m0128b = vfmaq_n_f32(m0128b, x13m16, self.twiddle13.im);
                let m0128b = vfmaq_n_f32(m0128b, x14m15, self.twiddle14.im);
                let (y01, y28) = NeonButterfly::butterfly2_f32(m0128a, m0128b);

                shift_store!(chunk, 0, y00, y01);
                shift_storel!(chunk, 28, y28);

                let m0227a = vfmaq_n_f32(u0, x1p28, self.twiddle2.re);
                let m0227a = vfmaq_n_f32(m0227a, x2p27, self.twiddle4.re);
                let m0227a = vfmaq_n_f32(m0227a, x3p26, self.twiddle6.re);
                let m0227a = vfmaq_n_f32(m0227a, x4p25, self.twiddle8.re);
                let m0227a = vfmaq_n_f32(m0227a, x5p24, self.twiddle10.re);
                let m0227a = vfmaq_n_f32(m0227a, x6p23, self.twiddle12.re);
                let m0227a = vfmaq_n_f32(m0227a, x7p22, self.twiddle14.re);
                let m0227a = vfmaq_n_f32(m0227a, x8p21, self.twiddle13.re);
                let m0227a = vfmaq_n_f32(m0227a, x9p20, self.twiddle11.re);
                let m0227a = vfmaq_n_f32(m0227a, x10p19, self.twiddle9.re);
                let m0227a = vfmaq_n_f32(m0227a, x11p18, self.twiddle7.re);
                let m0227a = vfmaq_n_f32(m0227a, x12p17, self.twiddle5.re);
                let m0227a = vfmaq_n_f32(m0227a, x13p16, self.twiddle3.re);
                let m0227a = vfmaq_n_f32(m0227a, x14p15, self.twiddle1.re);
                let m0227b = vmulq_n_f32(x1m28, self.twiddle2.im);
                let m0227b = vfmaq_n_f32(m0227b, x2m27, self.twiddle4.im);
                let m0227b = vfmaq_n_f32(m0227b, x3m26, self.twiddle6.im);
                let m0227b = vfmaq_n_f32(m0227b, x4m25, self.twiddle8.im);
                let m0227b = vfmaq_n_f32(m0227b, x5m24, self.twiddle10.im);
                let m0227b = vfmaq_n_f32(m0227b, x6m23, self.twiddle12.im);
                let m0227b = vfmaq_n_f32(m0227b, x7m22, self.twiddle14.im);
                let m0227b = vfmsq_n_f32(m0227b, x8m21, self.twiddle13.im);
                let m0227b = vfmsq_n_f32(m0227b, x9m20, self.twiddle11.im);
                let m0227b = vfmsq_n_f32(m0227b, x10m19, self.twiddle9.im);
                let m0227b = vfmsq_n_f32(m0227b, x11m18, self.twiddle7.im);
                let m0227b = vfmsq_n_f32(m0227b, x12m17, self.twiddle5.im);
                let m0227b = vfmsq_n_f32(m0227b, x13m16, self.twiddle3.im);
                let m0227b = vfmsq_n_f32(m0227b, x14m15, self.twiddle1.im);
                let (y02, y27) = NeonButterfly::butterfly2_f32(m0227a, m0227b);

                let m0326a = vfmaq_n_f32(u0, x1p28, self.twiddle3.re);
                let m0326a = vfmaq_n_f32(m0326a, x2p27, self.twiddle6.re);
                let m0326a = vfmaq_n_f32(m0326a, x3p26, self.twiddle9.re);
                let m0326a = vfmaq_n_f32(m0326a, x4p25, self.twiddle12.re);
                let m0326a = vfmaq_n_f32(m0326a, x5p24, self.twiddle14.re);
                let m0326a = vfmaq_n_f32(m0326a, x6p23, self.twiddle11.re);
                let m0326a = vfmaq_n_f32(m0326a, x7p22, self.twiddle8.re);
                let m0326a = vfmaq_n_f32(m0326a, x8p21, self.twiddle5.re);
                let m0326a = vfmaq_n_f32(m0326a, x9p20, self.twiddle2.re);
                let m0326a = vfmaq_n_f32(m0326a, x10p19, self.twiddle1.re);
                let m0326a = vfmaq_n_f32(m0326a, x11p18, self.twiddle4.re);
                let m0326a = vfmaq_n_f32(m0326a, x12p17, self.twiddle7.re);
                let m0326a = vfmaq_n_f32(m0326a, x13p16, self.twiddle10.re);
                let m0326a = vfmaq_n_f32(m0326a, x14p15, self.twiddle13.re);
                let m0326b = vmulq_n_f32(x1m28, self.twiddle3.im);
                let m0326b = vfmaq_n_f32(m0326b, x2m27, self.twiddle6.im);
                let m0326b = vfmaq_n_f32(m0326b, x3m26, self.twiddle9.im);
                let m0326b = vfmaq_n_f32(m0326b, x4m25, self.twiddle12.im);
                let m0326b = vfmsq_n_f32(m0326b, x5m24, self.twiddle14.im);
                let m0326b = vfmsq_n_f32(m0326b, x6m23, self.twiddle11.im);
                let m0326b = vfmsq_n_f32(m0326b, x7m22, self.twiddle8.im);
                let m0326b = vfmsq_n_f32(m0326b, x8m21, self.twiddle5.im);
                let m0326b = vfmsq_n_f32(m0326b, x9m20, self.twiddle2.im);
                let m0326b = vfmaq_n_f32(m0326b, x10m19, self.twiddle1.im);
                let m0326b = vfmaq_n_f32(m0326b, x11m18, self.twiddle4.im);
                let m0326b = vfmaq_n_f32(m0326b, x12m17, self.twiddle7.im);
                let m0326b = vfmaq_n_f32(m0326b, x13m16, self.twiddle10.im);
                let m0326b = vfmaq_n_f32(m0326b, x14m15, self.twiddle13.im);
                let (y03, y26) = NeonButterfly::butterfly2_f32(m0326a, m0326b);

                shift_store!(chunk, 2, y02, y03);
                shift_store!(chunk, 26, y26, y27);

                let m0425a = vfmaq_n_f32(u0, x1p28, self.twiddle4.re);
                let m0425a = vfmaq_n_f32(m0425a, x2p27, self.twiddle8.re);
                let m0425a = vfmaq_n_f32(m0425a, x3p26, self.twiddle12.re);
                let m0425a = vfmaq_n_f32(m0425a, x4p25, self.twiddle13.re);
                let m0425a = vfmaq_n_f32(m0425a, x5p24, self.twiddle9.re);
                let m0425a = vfmaq_n_f32(m0425a, x6p23, self.twiddle5.re);
                let m0425a = vfmaq_n_f32(m0425a, x7p22, self.twiddle1.re);
                let m0425a = vfmaq_n_f32(m0425a, x8p21, self.twiddle3.re);
                let m0425a = vfmaq_n_f32(m0425a, x9p20, self.twiddle7.re);
                let m0425a = vfmaq_n_f32(m0425a, x10p19, self.twiddle11.re);
                let m0425a = vfmaq_n_f32(m0425a, x11p18, self.twiddle14.re);
                let m0425a = vfmaq_n_f32(m0425a, x12p17, self.twiddle10.re);
                let m0425a = vfmaq_n_f32(m0425a, x13p16, self.twiddle6.re);
                let m0425a = vfmaq_n_f32(m0425a, x14p15, self.twiddle2.re);
                let m0425b = vmulq_n_f32(x1m28, self.twiddle4.im);
                let m0425b = vfmaq_n_f32(m0425b, x2m27, self.twiddle8.im);
                let m0425b = vfmaq_n_f32(m0425b, x3m26, self.twiddle12.im);
                let m0425b = vfmsq_n_f32(m0425b, x4m25, self.twiddle13.im);
                let m0425b = vfmsq_n_f32(m0425b, x5m24, self.twiddle9.im);
                let m0425b = vfmsq_n_f32(m0425b, x6m23, self.twiddle5.im);
                let m0425b = vfmsq_n_f32(m0425b, x7m22, self.twiddle1.im);
                let m0425b = vfmaq_n_f32(m0425b, x8m21, self.twiddle3.im);
                let m0425b = vfmaq_n_f32(m0425b, x9m20, self.twiddle7.im);
                let m0425b = vfmaq_n_f32(m0425b, x10m19, self.twiddle11.im);
                let m0425b = vfmsq_n_f32(m0425b, x11m18, self.twiddle14.im);
                let m0425b = vfmsq_n_f32(m0425b, x12m17, self.twiddle10.im);
                let m0425b = vfmsq_n_f32(m0425b, x13m16, self.twiddle6.im);
                let m0425b = vfmsq_n_f32(m0425b, x14m15, self.twiddle2.im);
                let (y04, y25) = NeonButterfly::butterfly2_f32(m0425a, m0425b);

                let m0524a = vfmaq_n_f32(u0, x1p28, self.twiddle5.re);
                let m0524a = vfmaq_n_f32(m0524a, x2p27, self.twiddle10.re);
                let m0524a = vfmaq_n_f32(m0524a, x3p26, self.twiddle14.re);
                let m0524a = vfmaq_n_f32(m0524a, x4p25, self.twiddle9.re);
                let m0524a = vfmaq_n_f32(m0524a, x5p24, self.twiddle4.re);
                let m0524a = vfmaq_n_f32(m0524a, x6p23, self.twiddle1.re);
                let m0524a = vfmaq_n_f32(m0524a, x7p22, self.twiddle6.re);
                let m0524a = vfmaq_n_f32(m0524a, x8p21, self.twiddle11.re);
                let m0524a = vfmaq_n_f32(m0524a, x9p20, self.twiddle13.re);
                let m0524a = vfmaq_n_f32(m0524a, x10p19, self.twiddle8.re);
                let m0524a = vfmaq_n_f32(m0524a, x11p18, self.twiddle3.re);
                let m0524a = vfmaq_n_f32(m0524a, x12p17, self.twiddle2.re);
                let m0524a = vfmaq_n_f32(m0524a, x13p16, self.twiddle7.re);
                let m0524a = vfmaq_n_f32(m0524a, x14p15, self.twiddle12.re);
                let m0524b = vmulq_n_f32(x1m28, self.twiddle5.im);
                let m0524b = vfmaq_n_f32(m0524b, x2m27, self.twiddle10.im);
                let m0524b = vfmsq_n_f32(m0524b, x3m26, self.twiddle14.im);
                let m0524b = vfmsq_n_f32(m0524b, x4m25, self.twiddle9.im);
                let m0524b = vfmsq_n_f32(m0524b, x5m24, self.twiddle4.im);
                let m0524b = vfmaq_n_f32(m0524b, x6m23, self.twiddle1.im);
                let m0524b = vfmaq_n_f32(m0524b, x7m22, self.twiddle6.im);
                let m0524b = vfmaq_n_f32(m0524b, x8m21, self.twiddle11.im);
                let m0524b = vfmsq_n_f32(m0524b, x9m20, self.twiddle13.im);
                let m0524b = vfmsq_n_f32(m0524b, x10m19, self.twiddle8.im);
                let m0524b = vfmsq_n_f32(m0524b, x11m18, self.twiddle3.im);
                let m0524b = vfmaq_n_f32(m0524b, x12m17, self.twiddle2.im);
                let m0524b = vfmaq_n_f32(m0524b, x13m16, self.twiddle7.im);
                let m0524b = vfmaq_n_f32(m0524b, x14m15, self.twiddle12.im);
                let (y05, y24) = NeonButterfly::butterfly2_f32(m0524a, m0524b);

                shift_store!(chunk, 4, y04, y05);
                shift_store!(chunk, 24, y24, y25);

                let m0623a = vfmaq_n_f32(u0, x1p28, self.twiddle6.re);
                let m0623a = vfmaq_n_f32(m0623a, x2p27, self.twiddle12.re);
                let m0623a = vfmaq_n_f32(m0623a, x3p26, self.twiddle11.re);
                let m0623a = vfmaq_n_f32(m0623a, x4p25, self.twiddle5.re);
                let m0623a = vfmaq_n_f32(m0623a, x5p24, self.twiddle1.re);
                let m0623a = vfmaq_n_f32(m0623a, x6p23, self.twiddle7.re);
                let m0623a = vfmaq_n_f32(m0623a, x7p22, self.twiddle13.re);
                let m0623a = vfmaq_n_f32(m0623a, x8p21, self.twiddle10.re);
                let m0623a = vfmaq_n_f32(m0623a, x9p20, self.twiddle4.re);
                let m0623a = vfmaq_n_f32(m0623a, x10p19, self.twiddle2.re);
                let m0623a = vfmaq_n_f32(m0623a, x11p18, self.twiddle8.re);
                let m0623a = vfmaq_n_f32(m0623a, x12p17, self.twiddle14.re);
                let m0623a = vfmaq_n_f32(m0623a, x13p16, self.twiddle9.re);
                let m0623a = vfmaq_n_f32(m0623a, x14p15, self.twiddle3.re);
                let m0623b = vmulq_n_f32(x1m28, self.twiddle6.im);
                let m0623b = vfmaq_n_f32(m0623b, x2m27, self.twiddle12.im);
                let m0623b = vfmsq_n_f32(m0623b, x3m26, self.twiddle11.im);
                let m0623b = vfmsq_n_f32(m0623b, x4m25, self.twiddle5.im);
                let m0623b = vfmaq_n_f32(m0623b, x5m24, self.twiddle1.im);
                let m0623b = vfmaq_n_f32(m0623b, x6m23, self.twiddle7.im);
                let m0623b = vfmaq_n_f32(m0623b, x7m22, self.twiddle13.im);
                let m0623b = vfmsq_n_f32(m0623b, x8m21, self.twiddle10.im);
                let m0623b = vfmsq_n_f32(m0623b, x9m20, self.twiddle4.im);
                let m0623b = vfmaq_n_f32(m0623b, x10m19, self.twiddle2.im);
                let m0623b = vfmaq_n_f32(m0623b, x11m18, self.twiddle8.im);
                let m0623b = vfmaq_n_f32(m0623b, x12m17, self.twiddle14.im);
                let m0623b = vfmsq_n_f32(m0623b, x13m16, self.twiddle9.im);
                let m0623b = vfmsq_n_f32(m0623b, x14m15, self.twiddle3.im);
                let (y06, y23) = NeonButterfly::butterfly2_f32(m0623a, m0623b);

                let m0722a = vfmaq_n_f32(u0, x1p28, self.twiddle7.re);
                let m0722a = vfmaq_n_f32(m0722a, x2p27, self.twiddle14.re);
                let m0722a = vfmaq_n_f32(m0722a, x3p26, self.twiddle8.re);
                let m0722a = vfmaq_n_f32(m0722a, x4p25, self.twiddle1.re);
                let m0722a = vfmaq_n_f32(m0722a, x5p24, self.twiddle6.re);
                let m0722a = vfmaq_n_f32(m0722a, x6p23, self.twiddle13.re);
                let m0722a = vfmaq_n_f32(m0722a, x7p22, self.twiddle9.re);
                let m0722a = vfmaq_n_f32(m0722a, x8p21, self.twiddle2.re);
                let m0722a = vfmaq_n_f32(m0722a, x9p20, self.twiddle5.re);
                let m0722a = vfmaq_n_f32(m0722a, x10p19, self.twiddle12.re);
                let m0722a = vfmaq_n_f32(m0722a, x11p18, self.twiddle10.re);
                let m0722a = vfmaq_n_f32(m0722a, x12p17, self.twiddle3.re);
                let m0722a = vfmaq_n_f32(m0722a, x13p16, self.twiddle4.re);
                let m0722a = vfmaq_n_f32(m0722a, x14p15, self.twiddle11.re);
                let m0722b = vmulq_n_f32(x1m28, self.twiddle7.im);
                let m0722b = vfmaq_n_f32(m0722b, x2m27, self.twiddle14.im);
                let m0722b = vfmsq_n_f32(m0722b, x3m26, self.twiddle8.im);
                let m0722b = vfmsq_n_f32(m0722b, x4m25, self.twiddle1.im);
                let m0722b = vfmaq_n_f32(m0722b, x5m24, self.twiddle6.im);
                let m0722b = vfmaq_n_f32(m0722b, x6m23, self.twiddle13.im);
                let m0722b = vfmsq_n_f32(m0722b, x7m22, self.twiddle9.im);
                let m0722b = vfmsq_n_f32(m0722b, x8m21, self.twiddle2.im);
                let m0722b = vfmaq_n_f32(m0722b, x9m20, self.twiddle5.im);
                let m0722b = vfmaq_n_f32(m0722b, x10m19, self.twiddle12.im);
                let m0722b = vfmsq_n_f32(m0722b, x11m18, self.twiddle10.im);
                let m0722b = vfmsq_n_f32(m0722b, x12m17, self.twiddle3.im);
                let m0722b = vfmaq_n_f32(m0722b, x13m16, self.twiddle4.im);
                let m0722b = vfmaq_n_f32(m0722b, x14m15, self.twiddle11.im);
                let (y07, y22) = NeonButterfly::butterfly2_f32(m0722a, m0722b);

                shift_store!(chunk, 6, y06, y07);
                shift_store!(chunk, 22, y22, y23);

                let m0821a = vfmaq_n_f32(u0, x1p28, self.twiddle8.re);
                let m0821a = vfmaq_n_f32(m0821a, x2p27, self.twiddle13.re);
                let m0821a = vfmaq_n_f32(m0821a, x3p26, self.twiddle5.re);
                let m0821a = vfmaq_n_f32(m0821a, x4p25, self.twiddle3.re);
                let m0821a = vfmaq_n_f32(m0821a, x5p24, self.twiddle11.re);
                let m0821a = vfmaq_n_f32(m0821a, x6p23, self.twiddle10.re);
                let m0821a = vfmaq_n_f32(m0821a, x7p22, self.twiddle2.re);
                let m0821a = vfmaq_n_f32(m0821a, x8p21, self.twiddle6.re);
                let m0821a = vfmaq_n_f32(m0821a, x9p20, self.twiddle14.re);
                let m0821a = vfmaq_n_f32(m0821a, x10p19, self.twiddle7.re);
                let m0821a = vfmaq_n_f32(m0821a, x11p18, self.twiddle1.re);
                let m0821a = vfmaq_n_f32(m0821a, x12p17, self.twiddle9.re);
                let m0821a = vfmaq_n_f32(m0821a, x13p16, self.twiddle12.re);
                let m0821a = vfmaq_n_f32(m0821a, x14p15, self.twiddle4.re);
                let m0821b = vmulq_n_f32(x1m28, self.twiddle8.im);
                let m0821b = vfmsq_n_f32(m0821b, x2m27, self.twiddle13.im);
                let m0821b = vfmsq_n_f32(m0821b, x3m26, self.twiddle5.im);
                let m0821b = vfmaq_n_f32(m0821b, x4m25, self.twiddle3.im);
                let m0821b = vfmaq_n_f32(m0821b, x5m24, self.twiddle11.im);
                let m0821b = vfmsq_n_f32(m0821b, x6m23, self.twiddle10.im);
                let m0821b = vfmsq_n_f32(m0821b, x7m22, self.twiddle2.im);
                let m0821b = vfmaq_n_f32(m0821b, x8m21, self.twiddle6.im);
                let m0821b = vfmaq_n_f32(m0821b, x9m20, self.twiddle14.im);
                let m0821b = vfmsq_n_f32(m0821b, x10m19, self.twiddle7.im);
                let m0821b = vfmaq_n_f32(m0821b, x11m18, self.twiddle1.im);
                let m0821b = vfmaq_n_f32(m0821b, x12m17, self.twiddle9.im);
                let m0821b = vfmsq_n_f32(m0821b, x13m16, self.twiddle12.im);
                let m0821b = vfmsq_n_f32(m0821b, x14m15, self.twiddle4.im);
                let (y08, y21) = NeonButterfly::butterfly2_f32(m0821a, m0821b);

                let m0920a = vfmaq_n_f32(u0, x1p28, self.twiddle9.re);
                let m0920a = vfmaq_n_f32(m0920a, x2p27, self.twiddle11.re);
                let m0920a = vfmaq_n_f32(m0920a, x3p26, self.twiddle2.re);
                let m0920a = vfmaq_n_f32(m0920a, x4p25, self.twiddle7.re);
                let m0920a = vfmaq_n_f32(m0920a, x5p24, self.twiddle13.re);
                let m0920a = vfmaq_n_f32(m0920a, x6p23, self.twiddle4.re);
                let m0920a = vfmaq_n_f32(m0920a, x7p22, self.twiddle5.re);
                let m0920a = vfmaq_n_f32(m0920a, x8p21, self.twiddle14.re);
                let m0920a = vfmaq_n_f32(m0920a, x9p20, self.twiddle6.re);
                let m0920a = vfmaq_n_f32(m0920a, x10p19, self.twiddle3.re);
                let m0920a = vfmaq_n_f32(m0920a, x11p18, self.twiddle12.re);
                let m0920a = vfmaq_n_f32(m0920a, x12p17, self.twiddle8.re);
                let m0920a = vfmaq_n_f32(m0920a, x13p16, self.twiddle1.re);
                let m0920a = vfmaq_n_f32(m0920a, x14p15, self.twiddle10.re);
                let m0920b = vmulq_n_f32(x1m28, self.twiddle9.im);
                let m0920b = vfmsq_n_f32(m0920b, x2m27, self.twiddle11.im);
                let m0920b = vfmsq_n_f32(m0920b, x3m26, self.twiddle2.im);
                let m0920b = vfmaq_n_f32(m0920b, x4m25, self.twiddle7.im);
                let m0920b = vfmsq_n_f32(m0920b, x5m24, self.twiddle13.im);
                let m0920b = vfmsq_n_f32(m0920b, x6m23, self.twiddle4.im);
                let m0920b = vfmaq_n_f32(m0920b, x7m22, self.twiddle5.im);
                let m0920b = vfmaq_n_f32(m0920b, x8m21, self.twiddle14.im);
                let m0920b = vfmsq_n_f32(m0920b, x9m20, self.twiddle6.im);
                let m0920b = vfmaq_n_f32(m0920b, x10m19, self.twiddle3.im);
                let m0920b = vfmaq_n_f32(m0920b, x11m18, self.twiddle12.im);
                let m0920b = vfmsq_n_f32(m0920b, x12m17, self.twiddle8.im);
                let m0920b = vfmaq_n_f32(m0920b, x13m16, self.twiddle1.im);
                let m0920b = vfmaq_n_f32(m0920b, x14m15, self.twiddle10.im);
                let (y09, y20) = NeonButterfly::butterfly2_f32(m0920a, m0920b);

                shift_store!(chunk, 8, y08, y09);
                shift_store!(chunk, 20, y20, y21);

                let m1019a = vfmaq_n_f32(u0, x1p28, self.twiddle10.re);
                let m1019a = vfmaq_n_f32(m1019a, x2p27, self.twiddle9.re);
                let m1019a = vfmaq_n_f32(m1019a, x3p26, self.twiddle1.re);
                let m1019a = vfmaq_n_f32(m1019a, x4p25, self.twiddle11.re);
                let m1019a = vfmaq_n_f32(m1019a, x5p24, self.twiddle8.re);
                let m1019a = vfmaq_n_f32(m1019a, x6p23, self.twiddle2.re);
                let m1019a = vfmaq_n_f32(m1019a, x7p22, self.twiddle12.re);
                let m1019a = vfmaq_n_f32(m1019a, x8p21, self.twiddle7.re);
                let m1019a = vfmaq_n_f32(m1019a, x9p20, self.twiddle3.re);
                let m1019a = vfmaq_n_f32(m1019a, x10p19, self.twiddle13.re);
                let m1019a = vfmaq_n_f32(m1019a, x11p18, self.twiddle6.re);
                let m1019a = vfmaq_n_f32(m1019a, x12p17, self.twiddle4.re);
                let m1019a = vfmaq_n_f32(m1019a, x13p16, self.twiddle14.re);
                let m1019a = vfmaq_n_f32(m1019a, x14p15, self.twiddle5.re);
                let m1019b = vmulq_n_f32(x1m28, self.twiddle10.im);
                let m1019b = vfmsq_n_f32(m1019b, x2m27, self.twiddle9.im);
                let m1019b = vfmaq_n_f32(m1019b, x3m26, self.twiddle1.im);
                let m1019b = vfmaq_n_f32(m1019b, x4m25, self.twiddle11.im);
                let m1019b = vfmsq_n_f32(m1019b, x5m24, self.twiddle8.im);
                let m1019b = vfmaq_n_f32(m1019b, x6m23, self.twiddle2.im);
                let m1019b = vfmaq_n_f32(m1019b, x7m22, self.twiddle12.im);
                let m1019b = vfmsq_n_f32(m1019b, x8m21, self.twiddle7.im);
                let m1019b = vfmaq_n_f32(m1019b, x9m20, self.twiddle3.im);
                let m1019b = vfmaq_n_f32(m1019b, x10m19, self.twiddle13.im);
                let m1019b = vfmsq_n_f32(m1019b, x11m18, self.twiddle6.im);
                let m1019b = vfmaq_n_f32(m1019b, x12m17, self.twiddle4.im);
                let m1019b = vfmaq_n_f32(m1019b, x13m16, self.twiddle14.im);
                let m1019b = vfmsq_n_f32(m1019b, x14m15, self.twiddle5.im);
                let (y10, y19) = NeonButterfly::butterfly2_f32(m1019a, m1019b);

                let m1118a = vfmaq_n_f32(u0, x1p28, self.twiddle11.re);
                let m1118a = vfmaq_n_f32(m1118a, x2p27, self.twiddle7.re);
                let m1118a = vfmaq_n_f32(m1118a, x3p26, self.twiddle4.re);
                let m1118a = vfmaq_n_f32(m1118a, x4p25, self.twiddle14.re);
                let m1118a = vfmaq_n_f32(m1118a, x5p24, self.twiddle3.re);
                let m1118a = vfmaq_n_f32(m1118a, x6p23, self.twiddle8.re);
                let m1118a = vfmaq_n_f32(m1118a, x7p22, self.twiddle10.re);
                let m1118a = vfmaq_n_f32(m1118a, x8p21, self.twiddle1.re);
                let m1118a = vfmaq_n_f32(m1118a, x9p20, self.twiddle12.re);
                let m1118a = vfmaq_n_f32(m1118a, x10p19, self.twiddle6.re);
                let m1118a = vfmaq_n_f32(m1118a, x11p18, self.twiddle5.re);
                let m1118a = vfmaq_n_f32(m1118a, x12p17, self.twiddle13.re);
                let m1118a = vfmaq_n_f32(m1118a, x13p16, self.twiddle2.re);
                let m1118a = vfmaq_n_f32(m1118a, x14p15, self.twiddle9.re);
                let m1118b = vmulq_n_f32(x1m28, self.twiddle11.im);
                let m1118b = vfmsq_n_f32(m1118b, x2m27, self.twiddle7.im);
                let m1118b = vfmaq_n_f32(m1118b, x3m26, self.twiddle4.im);
                let m1118b = vfmsq_n_f32(m1118b, x4m25, self.twiddle14.im);
                let m1118b = vfmsq_n_f32(m1118b, x5m24, self.twiddle3.im);
                let m1118b = vfmaq_n_f32(m1118b, x6m23, self.twiddle8.im);
                let m1118b = vfmsq_n_f32(m1118b, x7m22, self.twiddle10.im);
                let m1118b = vfmaq_n_f32(m1118b, x8m21, self.twiddle1.im);
                let m1118b = vfmaq_n_f32(m1118b, x9m20, self.twiddle12.im);
                let m1118b = vfmsq_n_f32(m1118b, x10m19, self.twiddle6.im);
                let m1118b = vfmaq_n_f32(m1118b, x11m18, self.twiddle5.im);
                let m1118b = vfmsq_n_f32(m1118b, x12m17, self.twiddle13.im);
                let m1118b = vfmsq_n_f32(m1118b, x13m16, self.twiddle2.im);
                let m1118b = vfmaq_n_f32(m1118b, x14m15, self.twiddle9.im);
                let (y11, y18) = NeonButterfly::butterfly2_f32(m1118a, m1118b);

                shift_store!(chunk, 10, y10, y11);
                shift_store!(chunk, 18, y18, y19);

                let m1217a = vfmaq_n_f32(u0, x1p28, self.twiddle12.re);
                let m1217a = vfmaq_n_f32(m1217a, x2p27, self.twiddle5.re);
                let m1217a = vfmaq_n_f32(m1217a, x3p26, self.twiddle7.re);
                let m1217a = vfmaq_n_f32(m1217a, x4p25, self.twiddle10.re);
                let m1217a = vfmaq_n_f32(m1217a, x5p24, self.twiddle2.re);
                let m1217a = vfmaq_n_f32(m1217a, x6p23, self.twiddle14.re);
                let m1217a = vfmaq_n_f32(m1217a, x7p22, self.twiddle3.re);
                let m1217a = vfmaq_n_f32(m1217a, x8p21, self.twiddle9.re);
                let m1217a = vfmaq_n_f32(m1217a, x9p20, self.twiddle8.re);
                let m1217a = vfmaq_n_f32(m1217a, x10p19, self.twiddle4.re);
                let m1217a = vfmaq_n_f32(m1217a, x11p18, self.twiddle13.re);
                let m1217a = vfmaq_n_f32(m1217a, x12p17, self.twiddle1.re);
                let m1217a = vfmaq_n_f32(m1217a, x13p16, self.twiddle11.re);
                let m1217a = vfmaq_n_f32(m1217a, x14p15, self.twiddle6.re);
                let m1217b = vmulq_n_f32(x1m28, self.twiddle12.im);
                let m1217b = vfmsq_n_f32(m1217b, x2m27, self.twiddle5.im);
                let m1217b = vfmaq_n_f32(m1217b, x3m26, self.twiddle7.im);
                let m1217b = vfmsq_n_f32(m1217b, x4m25, self.twiddle10.im);
                let m1217b = vfmaq_n_f32(m1217b, x5m24, self.twiddle2.im);
                let m1217b = vfmaq_n_f32(m1217b, x6m23, self.twiddle14.im);
                let m1217b = vfmsq_n_f32(m1217b, x7m22, self.twiddle3.im);
                let m1217b = vfmaq_n_f32(m1217b, x8m21, self.twiddle9.im);
                let m1217b = vfmsq_n_f32(m1217b, x9m20, self.twiddle8.im);
                let m1217b = vfmaq_n_f32(m1217b, x10m19, self.twiddle4.im);
                let m1217b = vfmsq_n_f32(m1217b, x11m18, self.twiddle13.im);
                let m1217b = vfmsq_n_f32(m1217b, x12m17, self.twiddle1.im);
                let m1217b = vfmaq_n_f32(m1217b, x13m16, self.twiddle11.im);
                let m1217b = vfmsq_n_f32(m1217b, x14m15, self.twiddle6.im);
                let (y12, y17) = NeonButterfly::butterfly2_f32(m1217a, m1217b);

                let m1316a = vfmaq_n_f32(u0, x1p28, self.twiddle13.re);
                let m1316a = vfmaq_n_f32(m1316a, x2p27, self.twiddle3.re);
                let m1316a = vfmaq_n_f32(m1316a, x3p26, self.twiddle10.re);
                let m1316a = vfmaq_n_f32(m1316a, x4p25, self.twiddle6.re);
                let m1316a = vfmaq_n_f32(m1316a, x5p24, self.twiddle7.re);
                let m1316a = vfmaq_n_f32(m1316a, x6p23, self.twiddle9.re);
                let m1316a = vfmaq_n_f32(m1316a, x7p22, self.twiddle4.re);
                let m1316a = vfmaq_n_f32(m1316a, x8p21, self.twiddle12.re);
                let m1316a = vfmaq_n_f32(m1316a, x9p20, self.twiddle1.re);
                let m1316a = vfmaq_n_f32(m1316a, x10p19, self.twiddle14.re);
                let m1316a = vfmaq_n_f32(m1316a, x11p18, self.twiddle2.re);
                let m1316a = vfmaq_n_f32(m1316a, x12p17, self.twiddle11.re);
                let m1316a = vfmaq_n_f32(m1316a, x13p16, self.twiddle5.re);
                let m1316a = vfmaq_n_f32(m1316a, x14p15, self.twiddle8.re);
                let m1316b = vmulq_n_f32(x1m28, self.twiddle13.im);
                let m1316b = vfmsq_n_f32(m1316b, x2m27, self.twiddle3.im);
                let m1316b = vfmaq_n_f32(m1316b, x3m26, self.twiddle10.im);
                let m1316b = vfmsq_n_f32(m1316b, x4m25, self.twiddle6.im);
                let m1316b = vfmaq_n_f32(m1316b, x5m24, self.twiddle7.im);
                let m1316b = vfmsq_n_f32(m1316b, x6m23, self.twiddle9.im);
                let m1316b = vfmaq_n_f32(m1316b, x7m22, self.twiddle4.im);
                let m1316b = vfmsq_n_f32(m1316b, x8m21, self.twiddle12.im);
                let m1316b = vfmaq_n_f32(m1316b, x9m20, self.twiddle1.im);
                let m1316b = vfmaq_n_f32(m1316b, x10m19, self.twiddle14.im);
                let m1316b = vfmsq_n_f32(m1316b, x11m18, self.twiddle2.im);
                let m1316b = vfmaq_n_f32(m1316b, x12m17, self.twiddle11.im);
                let m1316b = vfmsq_n_f32(m1316b, x13m16, self.twiddle5.im);
                let m1316b = vfmaq_n_f32(m1316b, x14m15, self.twiddle8.im);
                let (y13, y16) = NeonButterfly::butterfly2_f32(m1316a, m1316b);

                shift_store!(chunk, 12, y12, y13);
                shift_store!(chunk, 16, y16, y17);

                let m1415a = vfmaq_n_f32(u0, x1p28, self.twiddle14.re);
                let m1415a = vfmaq_n_f32(m1415a, x2p27, self.twiddle1.re);
                let m1415a = vfmaq_n_f32(m1415a, x3p26, self.twiddle13.re);
                let m1415a = vfmaq_n_f32(m1415a, x4p25, self.twiddle2.re);
                let m1415a = vfmaq_n_f32(m1415a, x5p24, self.twiddle12.re);
                let m1415a = vfmaq_n_f32(m1415a, x6p23, self.twiddle3.re);
                let m1415a = vfmaq_n_f32(m1415a, x7p22, self.twiddle11.re);
                let m1415a = vfmaq_n_f32(m1415a, x8p21, self.twiddle4.re);
                let m1415a = vfmaq_n_f32(m1415a, x9p20, self.twiddle10.re);
                let m1415a = vfmaq_n_f32(m1415a, x10p19, self.twiddle5.re);
                let m1415a = vfmaq_n_f32(m1415a, x11p18, self.twiddle9.re);
                let m1415a = vfmaq_n_f32(m1415a, x12p17, self.twiddle6.re);
                let m1415a = vfmaq_n_f32(m1415a, x13p16, self.twiddle8.re);
                let m1415a = vfmaq_n_f32(m1415a, x14p15, self.twiddle7.re);
                let m1415b = vmulq_n_f32(x1m28, self.twiddle14.im);
                let m1415b = vfmsq_n_f32(m1415b, x2m27, self.twiddle1.im);
                let m1415b = vfmaq_n_f32(m1415b, x3m26, self.twiddle13.im);
                let m1415b = vfmsq_n_f32(m1415b, x4m25, self.twiddle2.im);
                let m1415b = vfmaq_n_f32(m1415b, x5m24, self.twiddle12.im);
                let m1415b = vfmsq_n_f32(m1415b, x6m23, self.twiddle3.im);
                let m1415b = vfmaq_n_f32(m1415b, x7m22, self.twiddle11.im);
                let m1415b = vfmsq_n_f32(m1415b, x8m21, self.twiddle4.im);
                let m1415b = vfmaq_n_f32(m1415b, x9m20, self.twiddle10.im);
                let m1415b = vfmsq_n_f32(m1415b, x10m19, self.twiddle5.im);
                let m1415b = vfmaq_n_f32(m1415b, x11m18, self.twiddle9.im);
                let m1415b = vfmsq_n_f32(m1415b, x12m17, self.twiddle6.im);
                let m1415b = vfmaq_n_f32(m1415b, x13m16, self.twiddle8.im);
                let m1415b = vfmsq_n_f32(m1415b, x14m15, self.twiddle7.im);
                let (y14, y15) = NeonButterfly::butterfly2_f32(m1415a, m1415b);

                shift_store!(chunk, 14, y14, y15);
            }

            let rem = in_place.chunks_exact_mut(58).into_remainder();

            for chunk in rem.chunks_exact_mut(29) {
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
                let u28 = vld1_f32(chunk.get_unchecked(28..).as_ptr().cast());

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
                let u16 = vget_low_f32(u16u17);
                let u17 = vget_high_f32(u16u17);
                let u18 = vget_low_f32(u18u19);
                let u19 = vget_high_f32(u18u19);
                let u20 = vget_low_f32(u20u21);
                let u21 = vget_high_f32(u20u21);
                let u22 = vget_low_f32(u22u23);
                let u23 = vget_high_f32(u22u23);
                let u24 = vget_low_f32(u24u25);
                let u25 = vget_high_f32(u24u25);
                let u26 = vget_low_f32(u26u27);
                let u27 = vget_high_f32(u26u27);

                let y00 = u0;
                let (x1p28, x1m28) = NeonButterfly::butterfly2h_f32(u1, u28);
                let x1m28 = vcadd_rot90_f32(vdup_n_f32(0.), x1m28);
                let y00 = vadd_f32(y00, x1p28);
                let (x2p27, x2m27) = NeonButterfly::butterfly2h_f32(u2, u27);
                let x2m27 = vcadd_rot90_f32(vdup_n_f32(0.), x2m27);
                let y00 = vadd_f32(y00, x2p27);
                let (x3p26, x3m26) = NeonButterfly::butterfly2h_f32(u3, u26);
                let x3m26 = vcadd_rot90_f32(vdup_n_f32(0.), x3m26);
                let y00 = vadd_f32(y00, x3p26);
                let (x4p25, x4m25) = NeonButterfly::butterfly2h_f32(u4, u25);
                let x4m25 = vcadd_rot90_f32(vdup_n_f32(0.), x4m25);
                let y00 = vadd_f32(y00, x4p25);
                let (x5p24, x5m24) = NeonButterfly::butterfly2h_f32(u5, u24);
                let x5m24 = vcadd_rot90_f32(vdup_n_f32(0.), x5m24);
                let y00 = vadd_f32(y00, x5p24);
                let (x6p23, x6m23) = NeonButterfly::butterfly2h_f32(u6, u23);
                let x6m23 = vcadd_rot90_f32(vdup_n_f32(0.), x6m23);
                let y00 = vadd_f32(y00, x6p23);
                let (x7p22, x7m22) = NeonButterfly::butterfly2h_f32(u7, u22);
                let x7m22 = vcadd_rot90_f32(vdup_n_f32(0.), x7m22);
                let y00 = vadd_f32(y00, x7p22);
                let (x8p21, x8m21) = NeonButterfly::butterfly2h_f32(u8, u21);
                let x8m21 = vcadd_rot90_f32(vdup_n_f32(0.), x8m21);
                let y00 = vadd_f32(y00, x8p21);
                let (x9p20, x9m20) = NeonButterfly::butterfly2h_f32(u9, u20);
                let x9m20 = vcadd_rot90_f32(vdup_n_f32(0.), x9m20);
                let y00 = vadd_f32(y00, x9p20);
                let (x10p19, x10m19) = NeonButterfly::butterfly2h_f32(u10, u19);
                let x10m19 = vcadd_rot90_f32(vdup_n_f32(0.), x10m19);
                let y00 = vadd_f32(y00, x10p19);
                let (x11p18, x11m18) = NeonButterfly::butterfly2h_f32(u11, u18);
                let x11m18 = vcadd_rot90_f32(vdup_n_f32(0.), x11m18);
                let y00 = vadd_f32(y00, x11p18);
                let (x12p17, x12m17) = NeonButterfly::butterfly2h_f32(u12, u17);
                let x12m17 = vcadd_rot90_f32(vdup_n_f32(0.), x12m17);
                let y00 = vadd_f32(y00, x12p17);
                let (x13p16, x13m16) = NeonButterfly::butterfly2h_f32(u13, u16);
                let x13m16 = vcadd_rot90_f32(vdup_n_f32(0.), x13m16);
                let y00 = vadd_f32(y00, x13p16);
                let (x14p15, x14m15) = NeonButterfly::butterfly2h_f32(u14, u15);
                let x14m15 = vcadd_rot90_f32(vdup_n_f32(0.), x14m15);
                let y00 = vadd_f32(y00, x14p15);

                let m0128a = vfma_n_f32(u0, x1p28, self.twiddle1.re);
                let m0128a = vfma_n_f32(m0128a, x2p27, self.twiddle2.re);
                let m0128a = vfma_n_f32(m0128a, x3p26, self.twiddle3.re);
                let m0128a = vfma_n_f32(m0128a, x4p25, self.twiddle4.re);
                let m0128a = vfma_n_f32(m0128a, x5p24, self.twiddle5.re);
                let m0128a = vfma_n_f32(m0128a, x6p23, self.twiddle6.re);
                let m0128a = vfma_n_f32(m0128a, x7p22, self.twiddle7.re);
                let m0128a = vfma_n_f32(m0128a, x8p21, self.twiddle8.re);
                let m0128a = vfma_n_f32(m0128a, x9p20, self.twiddle9.re);
                let m0128a = vfma_n_f32(m0128a, x10p19, self.twiddle10.re);
                let m0128a = vfma_n_f32(m0128a, x11p18, self.twiddle11.re);
                let m0128a = vfma_n_f32(m0128a, x12p17, self.twiddle12.re);
                let m0128a = vfma_n_f32(m0128a, x13p16, self.twiddle13.re);
                let m0128a = vfma_n_f32(m0128a, x14p15, self.twiddle14.re);
                let m0128b = vmul_n_f32(x1m28, self.twiddle1.im);
                let m0128b = vfma_n_f32(m0128b, x2m27, self.twiddle2.im);
                let m0128b = vfma_n_f32(m0128b, x3m26, self.twiddle3.im);
                let m0128b = vfma_n_f32(m0128b, x4m25, self.twiddle4.im);
                let m0128b = vfma_n_f32(m0128b, x5m24, self.twiddle5.im);
                let m0128b = vfma_n_f32(m0128b, x6m23, self.twiddle6.im);
                let m0128b = vfma_n_f32(m0128b, x7m22, self.twiddle7.im);
                let m0128b = vfma_n_f32(m0128b, x8m21, self.twiddle8.im);
                let m0128b = vfma_n_f32(m0128b, x9m20, self.twiddle9.im);
                let m0128b = vfma_n_f32(m0128b, x10m19, self.twiddle10.im);
                let m0128b = vfma_n_f32(m0128b, x11m18, self.twiddle11.im);
                let m0128b = vfma_n_f32(m0128b, x12m17, self.twiddle12.im);
                let m0128b = vfma_n_f32(m0128b, x13m16, self.twiddle13.im);
                let m0128b = vfma_n_f32(m0128b, x14m15, self.twiddle14.im);
                let (y01, y28) = NeonButterfly::butterfly2h_f32(m0128a, m0128b);

                vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y00, y01));
                vst1_f32(chunk.get_unchecked_mut(28..).as_mut_ptr().cast(), y28);

                let m0227a = vfma_n_f32(u0, x1p28, self.twiddle2.re);
                let m0227a = vfma_n_f32(m0227a, x2p27, self.twiddle4.re);
                let m0227a = vfma_n_f32(m0227a, x3p26, self.twiddle6.re);
                let m0227a = vfma_n_f32(m0227a, x4p25, self.twiddle8.re);
                let m0227a = vfma_n_f32(m0227a, x5p24, self.twiddle10.re);
                let m0227a = vfma_n_f32(m0227a, x6p23, self.twiddle12.re);
                let m0227a = vfma_n_f32(m0227a, x7p22, self.twiddle14.re);
                let m0227a = vfma_n_f32(m0227a, x8p21, self.twiddle13.re);
                let m0227a = vfma_n_f32(m0227a, x9p20, self.twiddle11.re);
                let m0227a = vfma_n_f32(m0227a, x10p19, self.twiddle9.re);
                let m0227a = vfma_n_f32(m0227a, x11p18, self.twiddle7.re);
                let m0227a = vfma_n_f32(m0227a, x12p17, self.twiddle5.re);
                let m0227a = vfma_n_f32(m0227a, x13p16, self.twiddle3.re);
                let m0227a = vfma_n_f32(m0227a, x14p15, self.twiddle1.re);
                let m0227b = vmul_n_f32(x1m28, self.twiddle2.im);
                let m0227b = vfma_n_f32(m0227b, x2m27, self.twiddle4.im);
                let m0227b = vfma_n_f32(m0227b, x3m26, self.twiddle6.im);
                let m0227b = vfma_n_f32(m0227b, x4m25, self.twiddle8.im);
                let m0227b = vfma_n_f32(m0227b, x5m24, self.twiddle10.im);
                let m0227b = vfma_n_f32(m0227b, x6m23, self.twiddle12.im);
                let m0227b = vfma_n_f32(m0227b, x7m22, self.twiddle14.im);
                let m0227b = vfms_n_f32(m0227b, x8m21, self.twiddle13.im);
                let m0227b = vfms_n_f32(m0227b, x9m20, self.twiddle11.im);
                let m0227b = vfms_n_f32(m0227b, x10m19, self.twiddle9.im);
                let m0227b = vfms_n_f32(m0227b, x11m18, self.twiddle7.im);
                let m0227b = vfms_n_f32(m0227b, x12m17, self.twiddle5.im);
                let m0227b = vfms_n_f32(m0227b, x13m16, self.twiddle3.im);
                let m0227b = vfms_n_f32(m0227b, x14m15, self.twiddle1.im);
                let (y02, y27) = NeonButterfly::butterfly2h_f32(m0227a, m0227b);

                let m0326a = vfma_n_f32(u0, x1p28, self.twiddle3.re);
                let m0326a = vfma_n_f32(m0326a, x2p27, self.twiddle6.re);
                let m0326a = vfma_n_f32(m0326a, x3p26, self.twiddle9.re);
                let m0326a = vfma_n_f32(m0326a, x4p25, self.twiddle12.re);
                let m0326a = vfma_n_f32(m0326a, x5p24, self.twiddle14.re);
                let m0326a = vfma_n_f32(m0326a, x6p23, self.twiddle11.re);
                let m0326a = vfma_n_f32(m0326a, x7p22, self.twiddle8.re);
                let m0326a = vfma_n_f32(m0326a, x8p21, self.twiddle5.re);
                let m0326a = vfma_n_f32(m0326a, x9p20, self.twiddle2.re);
                let m0326a = vfma_n_f32(m0326a, x10p19, self.twiddle1.re);
                let m0326a = vfma_n_f32(m0326a, x11p18, self.twiddle4.re);
                let m0326a = vfma_n_f32(m0326a, x12p17, self.twiddle7.re);
                let m0326a = vfma_n_f32(m0326a, x13p16, self.twiddle10.re);
                let m0326a = vfma_n_f32(m0326a, x14p15, self.twiddle13.re);
                let m0326b = vmul_n_f32(x1m28, self.twiddle3.im);
                let m0326b = vfma_n_f32(m0326b, x2m27, self.twiddle6.im);
                let m0326b = vfma_n_f32(m0326b, x3m26, self.twiddle9.im);
                let m0326b = vfma_n_f32(m0326b, x4m25, self.twiddle12.im);
                let m0326b = vfms_n_f32(m0326b, x5m24, self.twiddle14.im);
                let m0326b = vfms_n_f32(m0326b, x6m23, self.twiddle11.im);
                let m0326b = vfms_n_f32(m0326b, x7m22, self.twiddle8.im);
                let m0326b = vfms_n_f32(m0326b, x8m21, self.twiddle5.im);
                let m0326b = vfms_n_f32(m0326b, x9m20, self.twiddle2.im);
                let m0326b = vfma_n_f32(m0326b, x10m19, self.twiddle1.im);
                let m0326b = vfma_n_f32(m0326b, x11m18, self.twiddle4.im);
                let m0326b = vfma_n_f32(m0326b, x12m17, self.twiddle7.im);
                let m0326b = vfma_n_f32(m0326b, x13m16, self.twiddle10.im);
                let m0326b = vfma_n_f32(m0326b, x14m15, self.twiddle13.im);
                let (y03, y26) = NeonButterfly::butterfly2h_f32(m0326a, m0326b);

                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y02, y03),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    vcombine_f32(y26, y27),
                );

                let m0425a = vfma_n_f32(u0, x1p28, self.twiddle4.re);
                let m0425a = vfma_n_f32(m0425a, x2p27, self.twiddle8.re);
                let m0425a = vfma_n_f32(m0425a, x3p26, self.twiddle12.re);
                let m0425a = vfma_n_f32(m0425a, x4p25, self.twiddle13.re);
                let m0425a = vfma_n_f32(m0425a, x5p24, self.twiddle9.re);
                let m0425a = vfma_n_f32(m0425a, x6p23, self.twiddle5.re);
                let m0425a = vfma_n_f32(m0425a, x7p22, self.twiddle1.re);
                let m0425a = vfma_n_f32(m0425a, x8p21, self.twiddle3.re);
                let m0425a = vfma_n_f32(m0425a, x9p20, self.twiddle7.re);
                let m0425a = vfma_n_f32(m0425a, x10p19, self.twiddle11.re);
                let m0425a = vfma_n_f32(m0425a, x11p18, self.twiddle14.re);
                let m0425a = vfma_n_f32(m0425a, x12p17, self.twiddle10.re);
                let m0425a = vfma_n_f32(m0425a, x13p16, self.twiddle6.re);
                let m0425a = vfma_n_f32(m0425a, x14p15, self.twiddle2.re);
                let m0425b = vmul_n_f32(x1m28, self.twiddle4.im);
                let m0425b = vfma_n_f32(m0425b, x2m27, self.twiddle8.im);
                let m0425b = vfma_n_f32(m0425b, x3m26, self.twiddle12.im);
                let m0425b = vfms_n_f32(m0425b, x4m25, self.twiddle13.im);
                let m0425b = vfms_n_f32(m0425b, x5m24, self.twiddle9.im);
                let m0425b = vfms_n_f32(m0425b, x6m23, self.twiddle5.im);
                let m0425b = vfms_n_f32(m0425b, x7m22, self.twiddle1.im);
                let m0425b = vfma_n_f32(m0425b, x8m21, self.twiddle3.im);
                let m0425b = vfma_n_f32(m0425b, x9m20, self.twiddle7.im);
                let m0425b = vfma_n_f32(m0425b, x10m19, self.twiddle11.im);
                let m0425b = vfms_n_f32(m0425b, x11m18, self.twiddle14.im);
                let m0425b = vfms_n_f32(m0425b, x12m17, self.twiddle10.im);
                let m0425b = vfms_n_f32(m0425b, x13m16, self.twiddle6.im);
                let m0425b = vfms_n_f32(m0425b, x14m15, self.twiddle2.im);
                let (y04, y25) = NeonButterfly::butterfly2h_f32(m0425a, m0425b);

                let m0524a = vfma_n_f32(u0, x1p28, self.twiddle5.re);
                let m0524a = vfma_n_f32(m0524a, x2p27, self.twiddle10.re);
                let m0524a = vfma_n_f32(m0524a, x3p26, self.twiddle14.re);
                let m0524a = vfma_n_f32(m0524a, x4p25, self.twiddle9.re);
                let m0524a = vfma_n_f32(m0524a, x5p24, self.twiddle4.re);
                let m0524a = vfma_n_f32(m0524a, x6p23, self.twiddle1.re);
                let m0524a = vfma_n_f32(m0524a, x7p22, self.twiddle6.re);
                let m0524a = vfma_n_f32(m0524a, x8p21, self.twiddle11.re);
                let m0524a = vfma_n_f32(m0524a, x9p20, self.twiddle13.re);
                let m0524a = vfma_n_f32(m0524a, x10p19, self.twiddle8.re);
                let m0524a = vfma_n_f32(m0524a, x11p18, self.twiddle3.re);
                let m0524a = vfma_n_f32(m0524a, x12p17, self.twiddle2.re);
                let m0524a = vfma_n_f32(m0524a, x13p16, self.twiddle7.re);
                let m0524a = vfma_n_f32(m0524a, x14p15, self.twiddle12.re);
                let m0524b = vmul_n_f32(x1m28, self.twiddle5.im);
                let m0524b = vfma_n_f32(m0524b, x2m27, self.twiddle10.im);
                let m0524b = vfms_n_f32(m0524b, x3m26, self.twiddle14.im);
                let m0524b = vfms_n_f32(m0524b, x4m25, self.twiddle9.im);
                let m0524b = vfms_n_f32(m0524b, x5m24, self.twiddle4.im);
                let m0524b = vfma_n_f32(m0524b, x6m23, self.twiddle1.im);
                let m0524b = vfma_n_f32(m0524b, x7m22, self.twiddle6.im);
                let m0524b = vfma_n_f32(m0524b, x8m21, self.twiddle11.im);
                let m0524b = vfms_n_f32(m0524b, x9m20, self.twiddle13.im);
                let m0524b = vfms_n_f32(m0524b, x10m19, self.twiddle8.im);
                let m0524b = vfms_n_f32(m0524b, x11m18, self.twiddle3.im);
                let m0524b = vfma_n_f32(m0524b, x12m17, self.twiddle2.im);
                let m0524b = vfma_n_f32(m0524b, x13m16, self.twiddle7.im);
                let m0524b = vfma_n_f32(m0524b, x14m15, self.twiddle12.im);
                let (y05, y24) = NeonButterfly::butterfly2h_f32(m0524a, m0524b);

                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y04, y05),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    vcombine_f32(y24, y25),
                );

                let m0623a = vfma_n_f32(u0, x1p28, self.twiddle6.re);
                let m0623a = vfma_n_f32(m0623a, x2p27, self.twiddle12.re);
                let m0623a = vfma_n_f32(m0623a, x3p26, self.twiddle11.re);
                let m0623a = vfma_n_f32(m0623a, x4p25, self.twiddle5.re);
                let m0623a = vfma_n_f32(m0623a, x5p24, self.twiddle1.re);
                let m0623a = vfma_n_f32(m0623a, x6p23, self.twiddle7.re);
                let m0623a = vfma_n_f32(m0623a, x7p22, self.twiddle13.re);
                let m0623a = vfma_n_f32(m0623a, x8p21, self.twiddle10.re);
                let m0623a = vfma_n_f32(m0623a, x9p20, self.twiddle4.re);
                let m0623a = vfma_n_f32(m0623a, x10p19, self.twiddle2.re);
                let m0623a = vfma_n_f32(m0623a, x11p18, self.twiddle8.re);
                let m0623a = vfma_n_f32(m0623a, x12p17, self.twiddle14.re);
                let m0623a = vfma_n_f32(m0623a, x13p16, self.twiddle9.re);
                let m0623a = vfma_n_f32(m0623a, x14p15, self.twiddle3.re);
                let m0623b = vmul_n_f32(x1m28, self.twiddle6.im);
                let m0623b = vfma_n_f32(m0623b, x2m27, self.twiddle12.im);
                let m0623b = vfms_n_f32(m0623b, x3m26, self.twiddle11.im);
                let m0623b = vfms_n_f32(m0623b, x4m25, self.twiddle5.im);
                let m0623b = vfma_n_f32(m0623b, x5m24, self.twiddle1.im);
                let m0623b = vfma_n_f32(m0623b, x6m23, self.twiddle7.im);
                let m0623b = vfma_n_f32(m0623b, x7m22, self.twiddle13.im);
                let m0623b = vfms_n_f32(m0623b, x8m21, self.twiddle10.im);
                let m0623b = vfms_n_f32(m0623b, x9m20, self.twiddle4.im);
                let m0623b = vfma_n_f32(m0623b, x10m19, self.twiddle2.im);
                let m0623b = vfma_n_f32(m0623b, x11m18, self.twiddle8.im);
                let m0623b = vfma_n_f32(m0623b, x12m17, self.twiddle14.im);
                let m0623b = vfms_n_f32(m0623b, x13m16, self.twiddle9.im);
                let m0623b = vfms_n_f32(m0623b, x14m15, self.twiddle3.im);
                let (y06, y23) = NeonButterfly::butterfly2h_f32(m0623a, m0623b);

                let m0722a = vfma_n_f32(u0, x1p28, self.twiddle7.re);
                let m0722a = vfma_n_f32(m0722a, x2p27, self.twiddle14.re);
                let m0722a = vfma_n_f32(m0722a, x3p26, self.twiddle8.re);
                let m0722a = vfma_n_f32(m0722a, x4p25, self.twiddle1.re);
                let m0722a = vfma_n_f32(m0722a, x5p24, self.twiddle6.re);
                let m0722a = vfma_n_f32(m0722a, x6p23, self.twiddle13.re);
                let m0722a = vfma_n_f32(m0722a, x7p22, self.twiddle9.re);
                let m0722a = vfma_n_f32(m0722a, x8p21, self.twiddle2.re);
                let m0722a = vfma_n_f32(m0722a, x9p20, self.twiddle5.re);
                let m0722a = vfma_n_f32(m0722a, x10p19, self.twiddle12.re);
                let m0722a = vfma_n_f32(m0722a, x11p18, self.twiddle10.re);
                let m0722a = vfma_n_f32(m0722a, x12p17, self.twiddle3.re);
                let m0722a = vfma_n_f32(m0722a, x13p16, self.twiddle4.re);
                let m0722a = vfma_n_f32(m0722a, x14p15, self.twiddle11.re);
                let m0722b = vmul_n_f32(x1m28, self.twiddle7.im);
                let m0722b = vfma_n_f32(m0722b, x2m27, self.twiddle14.im);
                let m0722b = vfms_n_f32(m0722b, x3m26, self.twiddle8.im);
                let m0722b = vfms_n_f32(m0722b, x4m25, self.twiddle1.im);
                let m0722b = vfma_n_f32(m0722b, x5m24, self.twiddle6.im);
                let m0722b = vfma_n_f32(m0722b, x6m23, self.twiddle13.im);
                let m0722b = vfms_n_f32(m0722b, x7m22, self.twiddle9.im);
                let m0722b = vfms_n_f32(m0722b, x8m21, self.twiddle2.im);
                let m0722b = vfma_n_f32(m0722b, x9m20, self.twiddle5.im);
                let m0722b = vfma_n_f32(m0722b, x10m19, self.twiddle12.im);
                let m0722b = vfms_n_f32(m0722b, x11m18, self.twiddle10.im);
                let m0722b = vfms_n_f32(m0722b, x12m17, self.twiddle3.im);
                let m0722b = vfma_n_f32(m0722b, x13m16, self.twiddle4.im);
                let m0722b = vfma_n_f32(m0722b, x14m15, self.twiddle11.im);
                let (y07, y22) = NeonButterfly::butterfly2h_f32(m0722a, m0722b);

                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y06, y07),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    vcombine_f32(y22, y23),
                );

                let m0821a = vfma_n_f32(u0, x1p28, self.twiddle8.re);
                let m0821a = vfma_n_f32(m0821a, x2p27, self.twiddle13.re);
                let m0821a = vfma_n_f32(m0821a, x3p26, self.twiddle5.re);
                let m0821a = vfma_n_f32(m0821a, x4p25, self.twiddle3.re);
                let m0821a = vfma_n_f32(m0821a, x5p24, self.twiddle11.re);
                let m0821a = vfma_n_f32(m0821a, x6p23, self.twiddle10.re);
                let m0821a = vfma_n_f32(m0821a, x7p22, self.twiddle2.re);
                let m0821a = vfma_n_f32(m0821a, x8p21, self.twiddle6.re);
                let m0821a = vfma_n_f32(m0821a, x9p20, self.twiddle14.re);
                let m0821a = vfma_n_f32(m0821a, x10p19, self.twiddle7.re);
                let m0821a = vfma_n_f32(m0821a, x11p18, self.twiddle1.re);
                let m0821a = vfma_n_f32(m0821a, x12p17, self.twiddle9.re);
                let m0821a = vfma_n_f32(m0821a, x13p16, self.twiddle12.re);
                let m0821a = vfma_n_f32(m0821a, x14p15, self.twiddle4.re);
                let m0821b = vmul_n_f32(x1m28, self.twiddle8.im);
                let m0821b = vfms_n_f32(m0821b, x2m27, self.twiddle13.im);
                let m0821b = vfms_n_f32(m0821b, x3m26, self.twiddle5.im);
                let m0821b = vfma_n_f32(m0821b, x4m25, self.twiddle3.im);
                let m0821b = vfma_n_f32(m0821b, x5m24, self.twiddle11.im);
                let m0821b = vfms_n_f32(m0821b, x6m23, self.twiddle10.im);
                let m0821b = vfms_n_f32(m0821b, x7m22, self.twiddle2.im);
                let m0821b = vfma_n_f32(m0821b, x8m21, self.twiddle6.im);
                let m0821b = vfma_n_f32(m0821b, x9m20, self.twiddle14.im);
                let m0821b = vfms_n_f32(m0821b, x10m19, self.twiddle7.im);
                let m0821b = vfma_n_f32(m0821b, x11m18, self.twiddle1.im);
                let m0821b = vfma_n_f32(m0821b, x12m17, self.twiddle9.im);
                let m0821b = vfms_n_f32(m0821b, x13m16, self.twiddle12.im);
                let m0821b = vfms_n_f32(m0821b, x14m15, self.twiddle4.im);
                let (y08, y21) = NeonButterfly::butterfly2h_f32(m0821a, m0821b);

                let m0920a = vfma_n_f32(u0, x1p28, self.twiddle9.re);
                let m0920a = vfma_n_f32(m0920a, x2p27, self.twiddle11.re);
                let m0920a = vfma_n_f32(m0920a, x3p26, self.twiddle2.re);
                let m0920a = vfma_n_f32(m0920a, x4p25, self.twiddle7.re);
                let m0920a = vfma_n_f32(m0920a, x5p24, self.twiddle13.re);
                let m0920a = vfma_n_f32(m0920a, x6p23, self.twiddle4.re);
                let m0920a = vfma_n_f32(m0920a, x7p22, self.twiddle5.re);
                let m0920a = vfma_n_f32(m0920a, x8p21, self.twiddle14.re);
                let m0920a = vfma_n_f32(m0920a, x9p20, self.twiddle6.re);
                let m0920a = vfma_n_f32(m0920a, x10p19, self.twiddle3.re);
                let m0920a = vfma_n_f32(m0920a, x11p18, self.twiddle12.re);
                let m0920a = vfma_n_f32(m0920a, x12p17, self.twiddle8.re);
                let m0920a = vfma_n_f32(m0920a, x13p16, self.twiddle1.re);
                let m0920a = vfma_n_f32(m0920a, x14p15, self.twiddle10.re);
                let m0920b = vmul_n_f32(x1m28, self.twiddle9.im);
                let m0920b = vfms_n_f32(m0920b, x2m27, self.twiddle11.im);
                let m0920b = vfms_n_f32(m0920b, x3m26, self.twiddle2.im);
                let m0920b = vfma_n_f32(m0920b, x4m25, self.twiddle7.im);
                let m0920b = vfms_n_f32(m0920b, x5m24, self.twiddle13.im);
                let m0920b = vfms_n_f32(m0920b, x6m23, self.twiddle4.im);
                let m0920b = vfma_n_f32(m0920b, x7m22, self.twiddle5.im);
                let m0920b = vfma_n_f32(m0920b, x8m21, self.twiddle14.im);
                let m0920b = vfms_n_f32(m0920b, x9m20, self.twiddle6.im);
                let m0920b = vfma_n_f32(m0920b, x10m19, self.twiddle3.im);
                let m0920b = vfma_n_f32(m0920b, x11m18, self.twiddle12.im);
                let m0920b = vfms_n_f32(m0920b, x12m17, self.twiddle8.im);
                let m0920b = vfma_n_f32(m0920b, x13m16, self.twiddle1.im);
                let m0920b = vfma_n_f32(m0920b, x14m15, self.twiddle10.im);
                let (y09, y20) = NeonButterfly::butterfly2h_f32(m0920a, m0920b);

                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y08, y09),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vcombine_f32(y20, y21),
                );

                let m1019a = vfma_n_f32(u0, x1p28, self.twiddle10.re);
                let m1019a = vfma_n_f32(m1019a, x2p27, self.twiddle9.re);
                let m1019a = vfma_n_f32(m1019a, x3p26, self.twiddle1.re);
                let m1019a = vfma_n_f32(m1019a, x4p25, self.twiddle11.re);
                let m1019a = vfma_n_f32(m1019a, x5p24, self.twiddle8.re);
                let m1019a = vfma_n_f32(m1019a, x6p23, self.twiddle2.re);
                let m1019a = vfma_n_f32(m1019a, x7p22, self.twiddle12.re);
                let m1019a = vfma_n_f32(m1019a, x8p21, self.twiddle7.re);
                let m1019a = vfma_n_f32(m1019a, x9p20, self.twiddle3.re);
                let m1019a = vfma_n_f32(m1019a, x10p19, self.twiddle13.re);
                let m1019a = vfma_n_f32(m1019a, x11p18, self.twiddle6.re);
                let m1019a = vfma_n_f32(m1019a, x12p17, self.twiddle4.re);
                let m1019a = vfma_n_f32(m1019a, x13p16, self.twiddle14.re);
                let m1019a = vfma_n_f32(m1019a, x14p15, self.twiddle5.re);
                let m1019b = vmul_n_f32(x1m28, self.twiddle10.im);
                let m1019b = vfms_n_f32(m1019b, x2m27, self.twiddle9.im);
                let m1019b = vfma_n_f32(m1019b, x3m26, self.twiddle1.im);
                let m1019b = vfma_n_f32(m1019b, x4m25, self.twiddle11.im);
                let m1019b = vfms_n_f32(m1019b, x5m24, self.twiddle8.im);
                let m1019b = vfma_n_f32(m1019b, x6m23, self.twiddle2.im);
                let m1019b = vfma_n_f32(m1019b, x7m22, self.twiddle12.im);
                let m1019b = vfms_n_f32(m1019b, x8m21, self.twiddle7.im);
                let m1019b = vfma_n_f32(m1019b, x9m20, self.twiddle3.im);
                let m1019b = vfma_n_f32(m1019b, x10m19, self.twiddle13.im);
                let m1019b = vfms_n_f32(m1019b, x11m18, self.twiddle6.im);
                let m1019b = vfma_n_f32(m1019b, x12m17, self.twiddle4.im);
                let m1019b = vfma_n_f32(m1019b, x13m16, self.twiddle14.im);
                let m1019b = vfms_n_f32(m1019b, x14m15, self.twiddle5.im);
                let (y10, y19) = NeonButterfly::butterfly2h_f32(m1019a, m1019b);

                let m1118a = vfma_n_f32(u0, x1p28, self.twiddle11.re);
                let m1118a = vfma_n_f32(m1118a, x2p27, self.twiddle7.re);
                let m1118a = vfma_n_f32(m1118a, x3p26, self.twiddle4.re);
                let m1118a = vfma_n_f32(m1118a, x4p25, self.twiddle14.re);
                let m1118a = vfma_n_f32(m1118a, x5p24, self.twiddle3.re);
                let m1118a = vfma_n_f32(m1118a, x6p23, self.twiddle8.re);
                let m1118a = vfma_n_f32(m1118a, x7p22, self.twiddle10.re);
                let m1118a = vfma_n_f32(m1118a, x8p21, self.twiddle1.re);
                let m1118a = vfma_n_f32(m1118a, x9p20, self.twiddle12.re);
                let m1118a = vfma_n_f32(m1118a, x10p19, self.twiddle6.re);
                let m1118a = vfma_n_f32(m1118a, x11p18, self.twiddle5.re);
                let m1118a = vfma_n_f32(m1118a, x12p17, self.twiddle13.re);
                let m1118a = vfma_n_f32(m1118a, x13p16, self.twiddle2.re);
                let m1118a = vfma_n_f32(m1118a, x14p15, self.twiddle9.re);
                let m1118b = vmul_n_f32(x1m28, self.twiddle11.im);
                let m1118b = vfms_n_f32(m1118b, x2m27, self.twiddle7.im);
                let m1118b = vfma_n_f32(m1118b, x3m26, self.twiddle4.im);
                let m1118b = vfms_n_f32(m1118b, x4m25, self.twiddle14.im);
                let m1118b = vfms_n_f32(m1118b, x5m24, self.twiddle3.im);
                let m1118b = vfma_n_f32(m1118b, x6m23, self.twiddle8.im);
                let m1118b = vfms_n_f32(m1118b, x7m22, self.twiddle10.im);
                let m1118b = vfma_n_f32(m1118b, x8m21, self.twiddle1.im);
                let m1118b = vfma_n_f32(m1118b, x9m20, self.twiddle12.im);
                let m1118b = vfms_n_f32(m1118b, x10m19, self.twiddle6.im);
                let m1118b = vfma_n_f32(m1118b, x11m18, self.twiddle5.im);
                let m1118b = vfms_n_f32(m1118b, x12m17, self.twiddle13.im);
                let m1118b = vfms_n_f32(m1118b, x13m16, self.twiddle2.im);
                let m1118b = vfma_n_f32(m1118b, x14m15, self.twiddle9.im);
                let (y11, y18) = NeonButterfly::butterfly2h_f32(m1118a, m1118b);

                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(y10, y11),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vcombine_f32(y18, y19),
                );

                let m1217a = vfma_n_f32(u0, x1p28, self.twiddle12.re);
                let m1217a = vfma_n_f32(m1217a, x2p27, self.twiddle5.re);
                let m1217a = vfma_n_f32(m1217a, x3p26, self.twiddle7.re);
                let m1217a = vfma_n_f32(m1217a, x4p25, self.twiddle10.re);
                let m1217a = vfma_n_f32(m1217a, x5p24, self.twiddle2.re);
                let m1217a = vfma_n_f32(m1217a, x6p23, self.twiddle14.re);
                let m1217a = vfma_n_f32(m1217a, x7p22, self.twiddle3.re);
                let m1217a = vfma_n_f32(m1217a, x8p21, self.twiddle9.re);
                let m1217a = vfma_n_f32(m1217a, x9p20, self.twiddle8.re);
                let m1217a = vfma_n_f32(m1217a, x10p19, self.twiddle4.re);
                let m1217a = vfma_n_f32(m1217a, x11p18, self.twiddle13.re);
                let m1217a = vfma_n_f32(m1217a, x12p17, self.twiddle1.re);
                let m1217a = vfma_n_f32(m1217a, x13p16, self.twiddle11.re);
                let m1217a = vfma_n_f32(m1217a, x14p15, self.twiddle6.re);
                let m1217b = vmul_n_f32(x1m28, self.twiddle12.im);
                let m1217b = vfms_n_f32(m1217b, x2m27, self.twiddle5.im);
                let m1217b = vfma_n_f32(m1217b, x3m26, self.twiddle7.im);
                let m1217b = vfms_n_f32(m1217b, x4m25, self.twiddle10.im);
                let m1217b = vfma_n_f32(m1217b, x5m24, self.twiddle2.im);
                let m1217b = vfma_n_f32(m1217b, x6m23, self.twiddle14.im);
                let m1217b = vfms_n_f32(m1217b, x7m22, self.twiddle3.im);
                let m1217b = vfma_n_f32(m1217b, x8m21, self.twiddle9.im);
                let m1217b = vfms_n_f32(m1217b, x9m20, self.twiddle8.im);
                let m1217b = vfma_n_f32(m1217b, x10m19, self.twiddle4.im);
                let m1217b = vfms_n_f32(m1217b, x11m18, self.twiddle13.im);
                let m1217b = vfms_n_f32(m1217b, x12m17, self.twiddle1.im);
                let m1217b = vfma_n_f32(m1217b, x13m16, self.twiddle11.im);
                let m1217b = vfms_n_f32(m1217b, x14m15, self.twiddle6.im);
                let (y12, y17) = NeonButterfly::butterfly2h_f32(m1217a, m1217b);

                let m1316a = vfma_n_f32(u0, x1p28, self.twiddle13.re);
                let m1316a = vfma_n_f32(m1316a, x2p27, self.twiddle3.re);
                let m1316a = vfma_n_f32(m1316a, x3p26, self.twiddle10.re);
                let m1316a = vfma_n_f32(m1316a, x4p25, self.twiddle6.re);
                let m1316a = vfma_n_f32(m1316a, x5p24, self.twiddle7.re);
                let m1316a = vfma_n_f32(m1316a, x6p23, self.twiddle9.re);
                let m1316a = vfma_n_f32(m1316a, x7p22, self.twiddle4.re);
                let m1316a = vfma_n_f32(m1316a, x8p21, self.twiddle12.re);
                let m1316a = vfma_n_f32(m1316a, x9p20, self.twiddle1.re);
                let m1316a = vfma_n_f32(m1316a, x10p19, self.twiddle14.re);
                let m1316a = vfma_n_f32(m1316a, x11p18, self.twiddle2.re);
                let m1316a = vfma_n_f32(m1316a, x12p17, self.twiddle11.re);
                let m1316a = vfma_n_f32(m1316a, x13p16, self.twiddle5.re);
                let m1316a = vfma_n_f32(m1316a, x14p15, self.twiddle8.re);
                let m1316b = vmul_n_f32(x1m28, self.twiddle13.im);
                let m1316b = vfms_n_f32(m1316b, x2m27, self.twiddle3.im);
                let m1316b = vfma_n_f32(m1316b, x3m26, self.twiddle10.im);
                let m1316b = vfms_n_f32(m1316b, x4m25, self.twiddle6.im);
                let m1316b = vfma_n_f32(m1316b, x5m24, self.twiddle7.im);
                let m1316b = vfms_n_f32(m1316b, x6m23, self.twiddle9.im);
                let m1316b = vfma_n_f32(m1316b, x7m22, self.twiddle4.im);
                let m1316b = vfms_n_f32(m1316b, x8m21, self.twiddle12.im);
                let m1316b = vfma_n_f32(m1316b, x9m20, self.twiddle1.im);
                let m1316b = vfma_n_f32(m1316b, x10m19, self.twiddle14.im);
                let m1316b = vfms_n_f32(m1316b, x11m18, self.twiddle2.im);
                let m1316b = vfma_n_f32(m1316b, x12m17, self.twiddle11.im);
                let m1316b = vfms_n_f32(m1316b, x13m16, self.twiddle5.im);
                let m1316b = vfma_n_f32(m1316b, x14m15, self.twiddle8.im);
                let (y13, y16) = NeonButterfly::butterfly2h_f32(m1316a, m1316b);

                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vcombine_f32(y12, y13),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vcombine_f32(y16, y17),
                );

                let m1415a = vfma_n_f32(u0, x1p28, self.twiddle14.re);
                let m1415a = vfma_n_f32(m1415a, x2p27, self.twiddle1.re);
                let m1415a = vfma_n_f32(m1415a, x3p26, self.twiddle13.re);
                let m1415a = vfma_n_f32(m1415a, x4p25, self.twiddle2.re);
                let m1415a = vfma_n_f32(m1415a, x5p24, self.twiddle12.re);
                let m1415a = vfma_n_f32(m1415a, x6p23, self.twiddle3.re);
                let m1415a = vfma_n_f32(m1415a, x7p22, self.twiddle11.re);
                let m1415a = vfma_n_f32(m1415a, x8p21, self.twiddle4.re);
                let m1415a = vfma_n_f32(m1415a, x9p20, self.twiddle10.re);
                let m1415a = vfma_n_f32(m1415a, x10p19, self.twiddle5.re);
                let m1415a = vfma_n_f32(m1415a, x11p18, self.twiddle9.re);
                let m1415a = vfma_n_f32(m1415a, x12p17, self.twiddle6.re);
                let m1415a = vfma_n_f32(m1415a, x13p16, self.twiddle8.re);
                let m1415a = vfma_n_f32(m1415a, x14p15, self.twiddle7.re);
                let m1415b = vmul_n_f32(x1m28, self.twiddle14.im);
                let m1415b = vfms_n_f32(m1415b, x2m27, self.twiddle1.im);
                let m1415b = vfma_n_f32(m1415b, x3m26, self.twiddle13.im);
                let m1415b = vfms_n_f32(m1415b, x4m25, self.twiddle2.im);
                let m1415b = vfma_n_f32(m1415b, x5m24, self.twiddle12.im);
                let m1415b = vfms_n_f32(m1415b, x6m23, self.twiddle3.im);
                let m1415b = vfma_n_f32(m1415b, x7m22, self.twiddle11.im);
                let m1415b = vfms_n_f32(m1415b, x8m21, self.twiddle4.im);
                let m1415b = vfma_n_f32(m1415b, x9m20, self.twiddle10.im);
                let m1415b = vfms_n_f32(m1415b, x10m19, self.twiddle5.im);
                let m1415b = vfma_n_f32(m1415b, x11m18, self.twiddle9.im);
                let m1415b = vfms_n_f32(m1415b, x12m17, self.twiddle6.im);
                let m1415b = vfma_n_f32(m1415b, x13m16, self.twiddle8.im);
                let m1415b = vfms_n_f32(m1415b, x14m15, self.twiddle7.im);
                let (y14, y15) = NeonButterfly::butterfly2h_f32(m1415a, m1415b);

                vst1q_f32(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vcombine_f32(y14, y15),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neon::butterflies::test_fcma_butterfly;

    test_fcma_butterfly!(test_fcma_butterfly29, f32, NeonFcmaButterfly29, 29, 1e-5);
    test_fcma_butterfly!(
        test_fcma_butterfly29_f64,
        f64,
        NeonFcmaButterfly29,
        29,
        1e-7
    );
}
