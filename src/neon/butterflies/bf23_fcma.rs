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

pub(crate) struct NeonFcmaButterfly23<T> {
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
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonFcmaButterfly23<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 23, fft_direction),
            twiddle2: compute_twiddle(2, 23, fft_direction),
            twiddle3: compute_twiddle(3, 23, fft_direction),
            twiddle4: compute_twiddle(4, 23, fft_direction),
            twiddle5: compute_twiddle(5, 23, fft_direction),
            twiddle6: compute_twiddle(6, 23, fft_direction),
            twiddle7: compute_twiddle(7, 23, fft_direction),
            twiddle8: compute_twiddle(8, 23, fft_direction),
            twiddle9: compute_twiddle(9, 23, fft_direction),
            twiddle10: compute_twiddle(10, 23, fft_direction),
            twiddle11: compute_twiddle(11, 23, fft_direction),
        }
    }
}
impl FftExecutor<f64> for NeonFcmaButterfly23<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        23
    }
}

impl NeonFcmaButterfly23<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 23 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(23) {
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

                let y00 = u0;
                let (x1p22, x1m22) = NeonButterfly::butterfly2_f64(u1, u22);
                let x1m22 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m22);
                let y00 = vaddq_f64(y00, x1p22);
                let (x2p21, x2m21) = NeonButterfly::butterfly2_f64(u2, u21);
                let x2m21 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m21);
                let y00 = vaddq_f64(y00, x2p21);
                let (x3p20, x3m20) = NeonButterfly::butterfly2_f64(u3, u20);
                let x3m20 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m20);
                let y00 = vaddq_f64(y00, x3p20);
                let (x4p19, x4m19) = NeonButterfly::butterfly2_f64(u4, u19);
                let x4m19 = vcaddq_rot90_f64(vdupq_n_f64(0.), x4m19);
                let y00 = vaddq_f64(y00, x4p19);
                let (x5p18, x5m18) = NeonButterfly::butterfly2_f64(u5, u18);
                let x5m18 = vcaddq_rot90_f64(vdupq_n_f64(0.), x5m18);
                let y00 = vaddq_f64(y00, x5p18);
                let (x6p17, x6m17) = NeonButterfly::butterfly2_f64(u6, u17);
                let x6m17 = vcaddq_rot90_f64(vdupq_n_f64(0.), x6m17);
                let y00 = vaddq_f64(y00, x6p17);
                let (x7p16, x7m16) = NeonButterfly::butterfly2_f64(u7, u16);
                let x7m16 = vcaddq_rot90_f64(vdupq_n_f64(0.), x7m16);
                let y00 = vaddq_f64(y00, x7p16);
                let (x8p15, x8m15) = NeonButterfly::butterfly2_f64(u8, u15);
                let x8m15 = vcaddq_rot90_f64(vdupq_n_f64(0.), x8m15);
                let y00 = vaddq_f64(y00, x8p15);
                let (x9p14, x9m14) = NeonButterfly::butterfly2_f64(u9, u14);
                let x9m14 = vcaddq_rot90_f64(vdupq_n_f64(0.), x9m14);
                let y00 = vaddq_f64(y00, x9p14);
                let (x10p13, x10m13) = NeonButterfly::butterfly2_f64(u10, u13);
                let x10m13 = vcaddq_rot90_f64(vdupq_n_f64(0.), x10m13);
                let y00 = vaddq_f64(y00, x10p13);
                let (x11p12, x11m12) = NeonButterfly::butterfly2_f64(u11, u12);
                let x11m12 = vcaddq_rot90_f64(vdupq_n_f64(0.), x11m12);
                let y00 = vaddq_f64(y00, x11p12);

                vst1q_f64(chunk.as_mut_ptr().cast(), y00);

                let m0122a = vfmaq_n_f64(u0, x1p22, self.twiddle1.re);
                let m0122a = vfmaq_n_f64(m0122a, x2p21, self.twiddle2.re);
                let m0122a = vfmaq_n_f64(m0122a, x3p20, self.twiddle3.re);
                let m0122a = vfmaq_n_f64(m0122a, x4p19, self.twiddle4.re);
                let m0122a = vfmaq_n_f64(m0122a, x5p18, self.twiddle5.re);
                let m0122a = vfmaq_n_f64(m0122a, x6p17, self.twiddle6.re);
                let m0122a = vfmaq_n_f64(m0122a, x7p16, self.twiddle7.re);
                let m0122a = vfmaq_n_f64(m0122a, x8p15, self.twiddle8.re);
                let m0122a = vfmaq_n_f64(m0122a, x9p14, self.twiddle9.re);
                let m0122a = vfmaq_n_f64(m0122a, x10p13, self.twiddle10.re);
                let m0122a = vfmaq_n_f64(m0122a, x11p12, self.twiddle11.re);
                let m0122b = vmulq_n_f64(x1m22, self.twiddle1.im);
                let m0122b = vfmaq_n_f64(m0122b, x2m21, self.twiddle2.im);
                let m0122b = vfmaq_n_f64(m0122b, x3m20, self.twiddle3.im);
                let m0122b = vfmaq_n_f64(m0122b, x4m19, self.twiddle4.im);
                let m0122b = vfmaq_n_f64(m0122b, x5m18, self.twiddle5.im);
                let m0122b = vfmaq_n_f64(m0122b, x6m17, self.twiddle6.im);
                let m0122b = vfmaq_n_f64(m0122b, x7m16, self.twiddle7.im);
                let m0122b = vfmaq_n_f64(m0122b, x8m15, self.twiddle8.im);
                let m0122b = vfmaq_n_f64(m0122b, x9m14, self.twiddle9.im);
                let m0122b = vfmaq_n_f64(m0122b, x10m13, self.twiddle10.im);
                let m0122b = vfmaq_n_f64(m0122b, x11m12, self.twiddle11.im);
                let (y01, y22) = NeonButterfly::butterfly2_f64(m0122a, m0122b);

                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y01);
                vst1q_f64(chunk.get_unchecked_mut(22..).as_mut_ptr().cast(), y22);

                let m0221a = vfmaq_n_f64(u0, x1p22, self.twiddle2.re);
                let m0221a = vfmaq_n_f64(m0221a, x2p21, self.twiddle4.re);
                let m0221a = vfmaq_n_f64(m0221a, x3p20, self.twiddle6.re);
                let m0221a = vfmaq_n_f64(m0221a, x4p19, self.twiddle8.re);
                let m0221a = vfmaq_n_f64(m0221a, x5p18, self.twiddle10.re);
                let m0221a = vfmaq_n_f64(m0221a, x6p17, self.twiddle11.re);
                let m0221a = vfmaq_n_f64(m0221a, x7p16, self.twiddle9.re);
                let m0221a = vfmaq_n_f64(m0221a, x8p15, self.twiddle7.re);
                let m0221a = vfmaq_n_f64(m0221a, x9p14, self.twiddle5.re);
                let m0221a = vfmaq_n_f64(m0221a, x10p13, self.twiddle3.re);
                let m0221a = vfmaq_n_f64(m0221a, x11p12, self.twiddle1.re);
                let m0221b = vmulq_n_f64(x1m22, self.twiddle2.im);
                let m0221b = vfmaq_n_f64(m0221b, x2m21, self.twiddle4.im);
                let m0221b = vfmaq_n_f64(m0221b, x3m20, self.twiddle6.im);
                let m0221b = vfmaq_n_f64(m0221b, x4m19, self.twiddle8.im);
                let m0221b = vfmaq_n_f64(m0221b, x5m18, self.twiddle10.im);
                let m0221b = vfmsq_n_f64(m0221b, x6m17, self.twiddle11.im);
                let m0221b = vfmsq_n_f64(m0221b, x7m16, self.twiddle9.im);
                let m0221b = vfmsq_n_f64(m0221b, x8m15, self.twiddle7.im);
                let m0221b = vfmsq_n_f64(m0221b, x9m14, self.twiddle5.im);
                let m0221b = vfmsq_n_f64(m0221b, x10m13, self.twiddle3.im);
                let m0221b = vfmsq_n_f64(m0221b, x11m12, self.twiddle1.im);
                let (y02, y21) = NeonButterfly::butterfly2_f64(m0221a, m0221b);

                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y02);
                vst1q_f64(chunk.get_unchecked_mut(21..).as_mut_ptr().cast(), y21);

                let m0320a = vfmaq_n_f64(u0, x1p22, self.twiddle3.re);
                let m0320a = vfmaq_n_f64(m0320a, x2p21, self.twiddle6.re);
                let m0320a = vfmaq_n_f64(m0320a, x3p20, self.twiddle9.re);
                let m0320a = vfmaq_n_f64(m0320a, x4p19, self.twiddle11.re);
                let m0320a = vfmaq_n_f64(m0320a, x5p18, self.twiddle8.re);
                let m0320a = vfmaq_n_f64(m0320a, x6p17, self.twiddle5.re);
                let m0320a = vfmaq_n_f64(m0320a, x7p16, self.twiddle2.re);
                let m0320a = vfmaq_n_f64(m0320a, x8p15, self.twiddle1.re);
                let m0320a = vfmaq_n_f64(m0320a, x9p14, self.twiddle4.re);
                let m0320a = vfmaq_n_f64(m0320a, x10p13, self.twiddle7.re);
                let m0320a = vfmaq_n_f64(m0320a, x11p12, self.twiddle10.re);
                let m0320b = vmulq_n_f64(x1m22, self.twiddle3.im);
                let m0320b = vfmaq_n_f64(m0320b, x2m21, self.twiddle6.im);
                let m0320b = vfmaq_n_f64(m0320b, x3m20, self.twiddle9.im);
                let m0320b = vfmsq_n_f64(m0320b, x4m19, self.twiddle11.im);
                let m0320b = vfmsq_n_f64(m0320b, x5m18, self.twiddle8.im);
                let m0320b = vfmsq_n_f64(m0320b, x6m17, self.twiddle5.im);
                let m0320b = vfmsq_n_f64(m0320b, x7m16, self.twiddle2.im);
                let m0320b = vfmaq_n_f64(m0320b, x8m15, self.twiddle1.im);
                let m0320b = vfmaq_n_f64(m0320b, x9m14, self.twiddle4.im);
                let m0320b = vfmaq_n_f64(m0320b, x10m13, self.twiddle7.im);
                let m0320b = vfmaq_n_f64(m0320b, x11m12, self.twiddle10.im);
                let (y03, y20) = NeonButterfly::butterfly2_f64(m0320a, m0320b);

                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y03);
                vst1q_f64(chunk.get_unchecked_mut(20..).as_mut_ptr().cast(), y20);

                let m0419a = vfmaq_n_f64(u0, x1p22, self.twiddle4.re);
                let m0419a = vfmaq_n_f64(m0419a, x2p21, self.twiddle8.re);
                let m0419a = vfmaq_n_f64(m0419a, x3p20, self.twiddle11.re);
                let m0419a = vfmaq_n_f64(m0419a, x4p19, self.twiddle7.re);
                let m0419a = vfmaq_n_f64(m0419a, x5p18, self.twiddle3.re);
                let m0419a = vfmaq_n_f64(m0419a, x6p17, self.twiddle1.re);
                let m0419a = vfmaq_n_f64(m0419a, x7p16, self.twiddle5.re);
                let m0419a = vfmaq_n_f64(m0419a, x8p15, self.twiddle9.re);
                let m0419a = vfmaq_n_f64(m0419a, x9p14, self.twiddle10.re);
                let m0419a = vfmaq_n_f64(m0419a, x10p13, self.twiddle6.re);
                let m0419a = vfmaq_n_f64(m0419a, x11p12, self.twiddle2.re);
                let m0419b = vmulq_n_f64(x1m22, self.twiddle4.im);
                let m0419b = vfmaq_n_f64(m0419b, x2m21, self.twiddle8.im);
                let m0419b = vfmsq_n_f64(m0419b, x3m20, self.twiddle11.im);
                let m0419b = vfmsq_n_f64(m0419b, x4m19, self.twiddle7.im);
                let m0419b = vfmsq_n_f64(m0419b, x5m18, self.twiddle3.im);
                let m0419b = vfmaq_n_f64(m0419b, x6m17, self.twiddle1.im);
                let m0419b = vfmaq_n_f64(m0419b, x7m16, self.twiddle5.im);
                let m0419b = vfmaq_n_f64(m0419b, x8m15, self.twiddle9.im);
                let m0419b = vfmsq_n_f64(m0419b, x9m14, self.twiddle10.im);
                let m0419b = vfmsq_n_f64(m0419b, x10m13, self.twiddle6.im);
                let m0419b = vfmsq_n_f64(m0419b, x11m12, self.twiddle2.im);
                let (y04, y19) = NeonButterfly::butterfly2_f64(m0419a, m0419b);

                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y04);
                vst1q_f64(chunk.get_unchecked_mut(19..).as_mut_ptr().cast(), y19);

                let m0518a = vfmaq_n_f64(u0, x1p22, self.twiddle5.re);
                let m0518a = vfmaq_n_f64(m0518a, x2p21, self.twiddle10.re);
                let m0518a = vfmaq_n_f64(m0518a, x3p20, self.twiddle8.re);
                let m0518a = vfmaq_n_f64(m0518a, x4p19, self.twiddle3.re);
                let m0518a = vfmaq_n_f64(m0518a, x5p18, self.twiddle2.re);
                let m0518a = vfmaq_n_f64(m0518a, x6p17, self.twiddle7.re);
                let m0518a = vfmaq_n_f64(m0518a, x7p16, self.twiddle11.re);
                let m0518a = vfmaq_n_f64(m0518a, x8p15, self.twiddle6.re);
                let m0518a = vfmaq_n_f64(m0518a, x9p14, self.twiddle1.re);
                let m0518a = vfmaq_n_f64(m0518a, x10p13, self.twiddle4.re);
                let m0518a = vfmaq_n_f64(m0518a, x11p12, self.twiddle9.re);
                let m0518b = vmulq_n_f64(x1m22, self.twiddle5.im);
                let m0518b = vfmaq_n_f64(m0518b, x2m21, self.twiddle10.im);
                let m0518b = vfmsq_n_f64(m0518b, x3m20, self.twiddle8.im);
                let m0518b = vfmsq_n_f64(m0518b, x4m19, self.twiddle3.im);
                let m0518b = vfmaq_n_f64(m0518b, x5m18, self.twiddle2.im);
                let m0518b = vfmaq_n_f64(m0518b, x6m17, self.twiddle7.im);
                let m0518b = vfmsq_n_f64(m0518b, x7m16, self.twiddle11.im);
                let m0518b = vfmsq_n_f64(m0518b, x8m15, self.twiddle6.im);
                let m0518b = vfmsq_n_f64(m0518b, x9m14, self.twiddle1.im);
                let m0518b = vfmaq_n_f64(m0518b, x10m13, self.twiddle4.im);
                let m0518b = vfmaq_n_f64(m0518b, x11m12, self.twiddle9.im);
                let (y05, y18) = NeonButterfly::butterfly2_f64(m0518a, m0518b);

                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y05);
                vst1q_f64(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), y18);

                let m0617a = vfmaq_n_f64(u0, x1p22, self.twiddle6.re);
                let m0617a = vfmaq_n_f64(m0617a, x2p21, self.twiddle11.re);
                let m0617a = vfmaq_n_f64(m0617a, x3p20, self.twiddle5.re);
                let m0617a = vfmaq_n_f64(m0617a, x4p19, self.twiddle1.re);
                let m0617a = vfmaq_n_f64(m0617a, x5p18, self.twiddle7.re);
                let m0617a = vfmaq_n_f64(m0617a, x6p17, self.twiddle10.re);
                let m0617a = vfmaq_n_f64(m0617a, x7p16, self.twiddle4.re);
                let m0617a = vfmaq_n_f64(m0617a, x8p15, self.twiddle2.re);
                let m0617a = vfmaq_n_f64(m0617a, x9p14, self.twiddle8.re);
                let m0617a = vfmaq_n_f64(m0617a, x10p13, self.twiddle9.re);
                let m0617a = vfmaq_n_f64(m0617a, x11p12, self.twiddle3.re);
                let m0617b = vmulq_n_f64(x1m22, self.twiddle6.im);
                let m0617b = vfmsq_n_f64(m0617b, x2m21, self.twiddle11.im);
                let m0617b = vfmsq_n_f64(m0617b, x3m20, self.twiddle5.im);
                let m0617b = vfmaq_n_f64(m0617b, x4m19, self.twiddle1.im);
                let m0617b = vfmaq_n_f64(m0617b, x5m18, self.twiddle7.im);
                let m0617b = vfmsq_n_f64(m0617b, x6m17, self.twiddle10.im);
                let m0617b = vfmsq_n_f64(m0617b, x7m16, self.twiddle4.im);
                let m0617b = vfmaq_n_f64(m0617b, x8m15, self.twiddle2.im);
                let m0617b = vfmaq_n_f64(m0617b, x9m14, self.twiddle8.im);
                let m0617b = vfmsq_n_f64(m0617b, x10m13, self.twiddle9.im);
                let m0617b = vfmsq_n_f64(m0617b, x11m12, self.twiddle3.im);
                let (y06, y17) = NeonButterfly::butterfly2_f64(m0617a, m0617b);

                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y06);
                vst1q_f64(chunk.get_unchecked_mut(17..).as_mut_ptr().cast(), y17);

                let m0716a = vfmaq_n_f64(u0, x1p22, self.twiddle7.re);
                let m0716a = vfmaq_n_f64(m0716a, x2p21, self.twiddle9.re);
                let m0716a = vfmaq_n_f64(m0716a, x3p20, self.twiddle2.re);
                let m0716a = vfmaq_n_f64(m0716a, x4p19, self.twiddle5.re);
                let m0716a = vfmaq_n_f64(m0716a, x5p18, self.twiddle11.re);
                let m0716a = vfmaq_n_f64(m0716a, x6p17, self.twiddle4.re);
                let m0716a = vfmaq_n_f64(m0716a, x7p16, self.twiddle3.re);
                let m0716a = vfmaq_n_f64(m0716a, x8p15, self.twiddle10.re);
                let m0716a = vfmaq_n_f64(m0716a, x9p14, self.twiddle6.re);
                let m0716a = vfmaq_n_f64(m0716a, x10p13, self.twiddle1.re);
                let m0716a = vfmaq_n_f64(m0716a, x11p12, self.twiddle8.re);
                let m0716b = vmulq_n_f64(x1m22, self.twiddle7.im);
                let m0716b = vfmsq_n_f64(m0716b, x2m21, self.twiddle9.im);
                let m0716b = vfmsq_n_f64(m0716b, x3m20, self.twiddle2.im);
                let m0716b = vfmaq_n_f64(m0716b, x4m19, self.twiddle5.im);
                let m0716b = vfmsq_n_f64(m0716b, x5m18, self.twiddle11.im);
                let m0716b = vfmsq_n_f64(m0716b, x6m17, self.twiddle4.im);
                let m0716b = vfmaq_n_f64(m0716b, x7m16, self.twiddle3.im);
                let m0716b = vfmaq_n_f64(m0716b, x8m15, self.twiddle10.im);
                let m0716b = vfmsq_n_f64(m0716b, x9m14, self.twiddle6.im);
                let m0716b = vfmaq_n_f64(m0716b, x10m13, self.twiddle1.im);
                let m0716b = vfmaq_n_f64(m0716b, x11m12, self.twiddle8.im);
                let (y07, y16) = NeonButterfly::butterfly2_f64(m0716a, m0716b);

                vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y07);
                vst1q_f64(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), y16);

                let m0815a = vfmaq_n_f64(u0, x1p22, self.twiddle8.re);
                let m0815a = vfmaq_n_f64(m0815a, x2p21, self.twiddle7.re);
                let m0815a = vfmaq_n_f64(m0815a, x3p20, self.twiddle1.re);
                let m0815a = vfmaq_n_f64(m0815a, x4p19, self.twiddle9.re);
                let m0815a = vfmaq_n_f64(m0815a, x5p18, self.twiddle6.re);
                let m0815a = vfmaq_n_f64(m0815a, x6p17, self.twiddle2.re);
                let m0815a = vfmaq_n_f64(m0815a, x7p16, self.twiddle10.re);
                let m0815a = vfmaq_n_f64(m0815a, x8p15, self.twiddle5.re);
                let m0815a = vfmaq_n_f64(m0815a, x9p14, self.twiddle3.re);
                let m0815a = vfmaq_n_f64(m0815a, x10p13, self.twiddle11.re);
                let m0815a = vfmaq_n_f64(m0815a, x11p12, self.twiddle4.re);
                let m0815b = vmulq_n_f64(x1m22, self.twiddle8.im);
                let m0815b = vfmsq_n_f64(m0815b, x2m21, self.twiddle7.im);
                let m0815b = vfmaq_n_f64(m0815b, x3m20, self.twiddle1.im);
                let m0815b = vfmaq_n_f64(m0815b, x4m19, self.twiddle9.im);
                let m0815b = vfmsq_n_f64(m0815b, x5m18, self.twiddle6.im);
                let m0815b = vfmaq_n_f64(m0815b, x6m17, self.twiddle2.im);
                let m0815b = vfmaq_n_f64(m0815b, x7m16, self.twiddle10.im);
                let m0815b = vfmsq_n_f64(m0815b, x8m15, self.twiddle5.im);
                let m0815b = vfmaq_n_f64(m0815b, x9m14, self.twiddle3.im);
                let m0815b = vfmaq_n_f64(m0815b, x10m13, self.twiddle11.im);
                let m0815b = vfmsq_n_f64(m0815b, x11m12, self.twiddle4.im);
                let (y08, y15) = NeonButterfly::butterfly2_f64(m0815a, m0815b);

                vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y08);
                vst1q_f64(chunk.get_unchecked_mut(15..).as_mut_ptr().cast(), y15);

                let m0914a = vfmaq_n_f64(u0, x1p22, self.twiddle9.re);
                let m0914a = vfmaq_n_f64(m0914a, x2p21, self.twiddle5.re);
                let m0914a = vfmaq_n_f64(m0914a, x3p20, self.twiddle4.re);
                let m0914a = vfmaq_n_f64(m0914a, x4p19, self.twiddle10.re);
                let m0914a = vfmaq_n_f64(m0914a, x5p18, self.twiddle1.re);
                let m0914a = vfmaq_n_f64(m0914a, x6p17, self.twiddle8.re);
                let m0914a = vfmaq_n_f64(m0914a, x7p16, self.twiddle6.re);
                let m0914a = vfmaq_n_f64(m0914a, x8p15, self.twiddle3.re);
                let m0914a = vfmaq_n_f64(m0914a, x9p14, self.twiddle11.re);
                let m0914a = vfmaq_n_f64(m0914a, x10p13, self.twiddle2.re);
                let m0914a = vfmaq_n_f64(m0914a, x11p12, self.twiddle7.re);
                let m0914b = vmulq_n_f64(x1m22, self.twiddle9.im);
                let m0914b = vfmsq_n_f64(m0914b, x2m21, self.twiddle5.im);
                let m0914b = vfmaq_n_f64(m0914b, x3m20, self.twiddle4.im);
                let m0914b = vfmsq_n_f64(m0914b, x4m19, self.twiddle10.im);
                let m0914b = vfmsq_n_f64(m0914b, x5m18, self.twiddle1.im);
                let m0914b = vfmaq_n_f64(m0914b, x6m17, self.twiddle8.im);
                let m0914b = vfmsq_n_f64(m0914b, x7m16, self.twiddle6.im);
                let m0914b = vfmaq_n_f64(m0914b, x8m15, self.twiddle3.im);
                let m0914b = vfmsq_n_f64(m0914b, x9m14, self.twiddle11.im);
                let m0914b = vfmsq_n_f64(m0914b, x10m13, self.twiddle2.im);
                let m0914b = vfmaq_n_f64(m0914b, x11m12, self.twiddle7.im);
                let (y09, y14) = NeonButterfly::butterfly2_f64(m0914a, m0914b);

                vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y09);
                vst1q_f64(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y14);

                let m1013a = vfmaq_n_f64(u0, x1p22, self.twiddle10.re);
                let m1013a = vfmaq_n_f64(m1013a, x2p21, self.twiddle3.re);
                let m1013a = vfmaq_n_f64(m1013a, x3p20, self.twiddle7.re);
                let m1013a = vfmaq_n_f64(m1013a, x4p19, self.twiddle6.re);
                let m1013a = vfmaq_n_f64(m1013a, x5p18, self.twiddle4.re);
                let m1013a = vfmaq_n_f64(m1013a, x6p17, self.twiddle9.re);
                let m1013a = vfmaq_n_f64(m1013a, x7p16, self.twiddle1.re);
                let m1013a = vfmaq_n_f64(m1013a, x8p15, self.twiddle11.re);
                let m1013a = vfmaq_n_f64(m1013a, x9p14, self.twiddle2.re);
                let m1013a = vfmaq_n_f64(m1013a, x10p13, self.twiddle8.re);
                let m1013a = vfmaq_n_f64(m1013a, x11p12, self.twiddle5.re);
                let m1013b = vmulq_n_f64(x1m22, self.twiddle10.im);
                let m1013b = vfmsq_n_f64(m1013b, x2m21, self.twiddle3.im);
                let m1013b = vfmaq_n_f64(m1013b, x3m20, self.twiddle7.im);
                let m1013b = vfmsq_n_f64(m1013b, x4m19, self.twiddle6.im);
                let m1013b = vfmaq_n_f64(m1013b, x5m18, self.twiddle4.im);
                let m1013b = vfmsq_n_f64(m1013b, x6m17, self.twiddle9.im);
                let m1013b = vfmaq_n_f64(m1013b, x7m16, self.twiddle1.im);
                let m1013b = vfmaq_n_f64(m1013b, x8m15, self.twiddle11.im);
                let m1013b = vfmsq_n_f64(m1013b, x9m14, self.twiddle2.im);
                let m1013b = vfmaq_n_f64(m1013b, x10m13, self.twiddle8.im);
                let m1013b = vfmsq_n_f64(m1013b, x11m12, self.twiddle5.im);
                let (y10, y13) = NeonButterfly::butterfly2_f64(m1013a, m1013b);

                vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);
                vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), y13);

                let m1112a = vfmaq_n_f64(u0, x1p22, self.twiddle11.re);
                let m1112a = vfmaq_n_f64(m1112a, x2p21, self.twiddle1.re);
                let m1112a = vfmaq_n_f64(m1112a, x3p20, self.twiddle10.re);
                let m1112a = vfmaq_n_f64(m1112a, x4p19, self.twiddle2.re);
                let m1112a = vfmaq_n_f64(m1112a, x5p18, self.twiddle9.re);
                let m1112a = vfmaq_n_f64(m1112a, x6p17, self.twiddle3.re);
                let m1112a = vfmaq_n_f64(m1112a, x7p16, self.twiddle8.re);
                let m1112a = vfmaq_n_f64(m1112a, x8p15, self.twiddle4.re);
                let m1112a = vfmaq_n_f64(m1112a, x9p14, self.twiddle7.re);
                let m1112a = vfmaq_n_f64(m1112a, x10p13, self.twiddle5.re);
                let m1112a = vfmaq_n_f64(m1112a, x11p12, self.twiddle6.re);
                let m1112b = vmulq_n_f64(x1m22, self.twiddle11.im);
                let m1112b = vfmsq_n_f64(m1112b, x2m21, self.twiddle1.im);
                let m1112b = vfmaq_n_f64(m1112b, x3m20, self.twiddle10.im);
                let m1112b = vfmsq_n_f64(m1112b, x4m19, self.twiddle2.im);
                let m1112b = vfmaq_n_f64(m1112b, x5m18, self.twiddle9.im);
                let m1112b = vfmsq_n_f64(m1112b, x6m17, self.twiddle3.im);
                let m1112b = vfmaq_n_f64(m1112b, x7m16, self.twiddle8.im);
                let m1112b = vfmsq_n_f64(m1112b, x8m15, self.twiddle4.im);
                let m1112b = vfmaq_n_f64(m1112b, x9m14, self.twiddle7.im);
                let m1112b = vfmsq_n_f64(m1112b, x10m13, self.twiddle5.im);
                let m1112b = vfmaq_n_f64(m1112b, x11m12, self.twiddle6.im);
                let (y11, y12) = NeonButterfly::butterfly2_f64(m1112a, m1112b);

                vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), y11);
                vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y12);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaButterfly23<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        23
    }
}

impl NeonFcmaButterfly23<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 23 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(23) {
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
                let u22 = vld1_f32(chunk.get_unchecked(22..).as_ptr().cast());

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

                let y00 = u0;
                let (x1p22, x1m22) = NeonButterfly::butterfly2h_f32(u1, u22);
                let x1m22 = vcadd_rot90_f32(vdup_n_f32(0.), x1m22);
                let y00 = vadd_f32(y00, x1p22);
                let (x2p21, x2m21) = NeonButterfly::butterfly2h_f32(u2, u21);
                let x2m21 = vcadd_rot90_f32(vdup_n_f32(0.), x2m21);
                let y00 = vadd_f32(y00, x2p21);
                let (x3p20, x3m20) = NeonButterfly::butterfly2h_f32(u3, u20);
                let x3m20 = vcadd_rot90_f32(vdup_n_f32(0.), x3m20);
                let y00 = vadd_f32(y00, x3p20);
                let (x4p19, x4m19) = NeonButterfly::butterfly2h_f32(u4, u19);
                let x4m19 = vcadd_rot90_f32(vdup_n_f32(0.), x4m19);
                let y00 = vadd_f32(y00, x4p19);
                let (x5p18, x5m18) = NeonButterfly::butterfly2h_f32(u5, u18);
                let x5m18 = vcadd_rot90_f32(vdup_n_f32(0.), x5m18);
                let y00 = vadd_f32(y00, x5p18);
                let (x6p17, x6m17) = NeonButterfly::butterfly2h_f32(u6, u17);
                let x6m17 = vcadd_rot90_f32(vdup_n_f32(0.), x6m17);
                let y00 = vadd_f32(y00, x6p17);
                let (x7p16, x7m16) = NeonButterfly::butterfly2h_f32(u7, u16);
                let x7m16 = vcadd_rot90_f32(vdup_n_f32(0.), x7m16);
                let y00 = vadd_f32(y00, x7p16);
                let (x8p15, x8m15) = NeonButterfly::butterfly2h_f32(u8, u15);
                let x8m15 = vcadd_rot90_f32(vdup_n_f32(0.), x8m15);
                let y00 = vadd_f32(y00, x8p15);
                let (x9p14, x9m14) = NeonButterfly::butterfly2h_f32(u9, u14);
                let x9m14 = vcadd_rot90_f32(vdup_n_f32(0.), x9m14);
                let y00 = vadd_f32(y00, x9p14);
                let (x10p13, x10m13) = NeonButterfly::butterfly2h_f32(u10, u13);
                let x10m13 = vcadd_rot90_f32(vdup_n_f32(0.), x10m13);
                let y00 = vadd_f32(y00, x10p13);
                let (x11p12, x11m12) = NeonButterfly::butterfly2h_f32(u11, u12);
                let x11m12 = vcadd_rot90_f32(vdup_n_f32(0.), x11m12);
                let y00 = vadd_f32(y00, x11p12);

                let m0122a = vfma_n_f32(u0, x1p22, self.twiddle1.re);
                let m0122a = vfma_n_f32(m0122a, x2p21, self.twiddle2.re);
                let m0122a = vfma_n_f32(m0122a, x3p20, self.twiddle3.re);
                let m0122a = vfma_n_f32(m0122a, x4p19, self.twiddle4.re);
                let m0122a = vfma_n_f32(m0122a, x5p18, self.twiddle5.re);
                let m0122a = vfma_n_f32(m0122a, x6p17, self.twiddle6.re);
                let m0122a = vfma_n_f32(m0122a, x7p16, self.twiddle7.re);
                let m0122a = vfma_n_f32(m0122a, x8p15, self.twiddle8.re);
                let m0122a = vfma_n_f32(m0122a, x9p14, self.twiddle9.re);
                let m0122a = vfma_n_f32(m0122a, x10p13, self.twiddle10.re);
                let m0122a = vfma_n_f32(m0122a, x11p12, self.twiddle11.re);
                let m0122b = vmul_n_f32(x1m22, self.twiddle1.im);
                let m0122b = vfma_n_f32(m0122b, x2m21, self.twiddle2.im);
                let m0122b = vfma_n_f32(m0122b, x3m20, self.twiddle3.im);
                let m0122b = vfma_n_f32(m0122b, x4m19, self.twiddle4.im);
                let m0122b = vfma_n_f32(m0122b, x5m18, self.twiddle5.im);
                let m0122b = vfma_n_f32(m0122b, x6m17, self.twiddle6.im);
                let m0122b = vfma_n_f32(m0122b, x7m16, self.twiddle7.im);
                let m0122b = vfma_n_f32(m0122b, x8m15, self.twiddle8.im);
                let m0122b = vfma_n_f32(m0122b, x9m14, self.twiddle9.im);
                let m0122b = vfma_n_f32(m0122b, x10m13, self.twiddle10.im);
                let m0122b = vfma_n_f32(m0122b, x11m12, self.twiddle11.im);
                let (y01, y22) = NeonButterfly::butterfly2h_f32(m0122a, m0122b);

                vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y00, y01));
                vst1_f32(chunk.get_unchecked_mut(22..).as_mut_ptr().cast(), y22);

                let m0221a = vfma_n_f32(u0, x1p22, self.twiddle2.re);
                let m0221a = vfma_n_f32(m0221a, x2p21, self.twiddle4.re);
                let m0221a = vfma_n_f32(m0221a, x3p20, self.twiddle6.re);
                let m0221a = vfma_n_f32(m0221a, x4p19, self.twiddle8.re);
                let m0221a = vfma_n_f32(m0221a, x5p18, self.twiddle10.re);
                let m0221a = vfma_n_f32(m0221a, x6p17, self.twiddle11.re);
                let m0221a = vfma_n_f32(m0221a, x7p16, self.twiddle9.re);
                let m0221a = vfma_n_f32(m0221a, x8p15, self.twiddle7.re);
                let m0221a = vfma_n_f32(m0221a, x9p14, self.twiddle5.re);
                let m0221a = vfma_n_f32(m0221a, x10p13, self.twiddle3.re);
                let m0221a = vfma_n_f32(m0221a, x11p12, self.twiddle1.re);
                let m0221b = vmul_n_f32(x1m22, self.twiddle2.im);
                let m0221b = vfma_n_f32(m0221b, x2m21, self.twiddle4.im);
                let m0221b = vfma_n_f32(m0221b, x3m20, self.twiddle6.im);
                let m0221b = vfma_n_f32(m0221b, x4m19, self.twiddle8.im);
                let m0221b = vfma_n_f32(m0221b, x5m18, self.twiddle10.im);
                let m0221b = vfms_n_f32(m0221b, x6m17, self.twiddle11.im);
                let m0221b = vfms_n_f32(m0221b, x7m16, self.twiddle9.im);
                let m0221b = vfms_n_f32(m0221b, x8m15, self.twiddle7.im);
                let m0221b = vfms_n_f32(m0221b, x9m14, self.twiddle5.im);
                let m0221b = vfms_n_f32(m0221b, x10m13, self.twiddle3.im);
                let m0221b = vfms_n_f32(m0221b, x11m12, self.twiddle1.im);
                let (y02, y21) = NeonButterfly::butterfly2h_f32(m0221a, m0221b);

                let m0320a = vfma_n_f32(u0, x1p22, self.twiddle3.re);
                let m0320a = vfma_n_f32(m0320a, x2p21, self.twiddle6.re);
                let m0320a = vfma_n_f32(m0320a, x3p20, self.twiddle9.re);
                let m0320a = vfma_n_f32(m0320a, x4p19, self.twiddle11.re);
                let m0320a = vfma_n_f32(m0320a, x5p18, self.twiddle8.re);
                let m0320a = vfma_n_f32(m0320a, x6p17, self.twiddle5.re);
                let m0320a = vfma_n_f32(m0320a, x7p16, self.twiddle2.re);
                let m0320a = vfma_n_f32(m0320a, x8p15, self.twiddle1.re);
                let m0320a = vfma_n_f32(m0320a, x9p14, self.twiddle4.re);
                let m0320a = vfma_n_f32(m0320a, x10p13, self.twiddle7.re);
                let m0320a = vfma_n_f32(m0320a, x11p12, self.twiddle10.re);
                let m0320b = vmul_n_f32(x1m22, self.twiddle3.im);
                let m0320b = vfma_n_f32(m0320b, x2m21, self.twiddle6.im);
                let m0320b = vfma_n_f32(m0320b, x3m20, self.twiddle9.im);
                let m0320b = vfms_n_f32(m0320b, x4m19, self.twiddle11.im);
                let m0320b = vfms_n_f32(m0320b, x5m18, self.twiddle8.im);
                let m0320b = vfms_n_f32(m0320b, x6m17, self.twiddle5.im);
                let m0320b = vfms_n_f32(m0320b, x7m16, self.twiddle2.im);
                let m0320b = vfma_n_f32(m0320b, x8m15, self.twiddle1.im);
                let m0320b = vfma_n_f32(m0320b, x9m14, self.twiddle4.im);
                let m0320b = vfma_n_f32(m0320b, x10m13, self.twiddle7.im);
                let m0320b = vfma_n_f32(m0320b, x11m12, self.twiddle10.im);
                let (y03, y20) = NeonButterfly::butterfly2h_f32(m0320a, m0320b);

                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y02, y03),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vcombine_f32(y20, y21),
                );

                let m0419a = vfma_n_f32(u0, x1p22, self.twiddle4.re);
                let m0419a = vfma_n_f32(m0419a, x2p21, self.twiddle8.re);
                let m0419a = vfma_n_f32(m0419a, x3p20, self.twiddle11.re);
                let m0419a = vfma_n_f32(m0419a, x4p19, self.twiddle7.re);
                let m0419a = vfma_n_f32(m0419a, x5p18, self.twiddle3.re);
                let m0419a = vfma_n_f32(m0419a, x6p17, self.twiddle1.re);
                let m0419a = vfma_n_f32(m0419a, x7p16, self.twiddle5.re);
                let m0419a = vfma_n_f32(m0419a, x8p15, self.twiddle9.re);
                let m0419a = vfma_n_f32(m0419a, x9p14, self.twiddle10.re);
                let m0419a = vfma_n_f32(m0419a, x10p13, self.twiddle6.re);
                let m0419a = vfma_n_f32(m0419a, x11p12, self.twiddle2.re);
                let m0419b = vmul_n_f32(x1m22, self.twiddle4.im);
                let m0419b = vfma_n_f32(m0419b, x2m21, self.twiddle8.im);
                let m0419b = vfms_n_f32(m0419b, x3m20, self.twiddle11.im);
                let m0419b = vfms_n_f32(m0419b, x4m19, self.twiddle7.im);
                let m0419b = vfms_n_f32(m0419b, x5m18, self.twiddle3.im);
                let m0419b = vfma_n_f32(m0419b, x6m17, self.twiddle1.im);
                let m0419b = vfma_n_f32(m0419b, x7m16, self.twiddle5.im);
                let m0419b = vfma_n_f32(m0419b, x8m15, self.twiddle9.im);
                let m0419b = vfms_n_f32(m0419b, x9m14, self.twiddle10.im);
                let m0419b = vfms_n_f32(m0419b, x10m13, self.twiddle6.im);
                let m0419b = vfms_n_f32(m0419b, x11m12, self.twiddle2.im);
                let (y04, y19) = NeonButterfly::butterfly2h_f32(m0419a, m0419b);

                let m0518a = vfma_n_f32(u0, x1p22, self.twiddle5.re);
                let m0518a = vfma_n_f32(m0518a, x2p21, self.twiddle10.re);
                let m0518a = vfma_n_f32(m0518a, x3p20, self.twiddle8.re);
                let m0518a = vfma_n_f32(m0518a, x4p19, self.twiddle3.re);
                let m0518a = vfma_n_f32(m0518a, x5p18, self.twiddle2.re);
                let m0518a = vfma_n_f32(m0518a, x6p17, self.twiddle7.re);
                let m0518a = vfma_n_f32(m0518a, x7p16, self.twiddle11.re);
                let m0518a = vfma_n_f32(m0518a, x8p15, self.twiddle6.re);
                let m0518a = vfma_n_f32(m0518a, x9p14, self.twiddle1.re);
                let m0518a = vfma_n_f32(m0518a, x10p13, self.twiddle4.re);
                let m0518a = vfma_n_f32(m0518a, x11p12, self.twiddle9.re);
                let m0518b = vmul_n_f32(x1m22, self.twiddle5.im);
                let m0518b = vfma_n_f32(m0518b, x2m21, self.twiddle10.im);
                let m0518b = vfms_n_f32(m0518b, x3m20, self.twiddle8.im);
                let m0518b = vfms_n_f32(m0518b, x4m19, self.twiddle3.im);
                let m0518b = vfma_n_f32(m0518b, x5m18, self.twiddle2.im);
                let m0518b = vfma_n_f32(m0518b, x6m17, self.twiddle7.im);
                let m0518b = vfms_n_f32(m0518b, x7m16, self.twiddle11.im);
                let m0518b = vfms_n_f32(m0518b, x8m15, self.twiddle6.im);
                let m0518b = vfms_n_f32(m0518b, x9m14, self.twiddle1.im);
                let m0518b = vfma_n_f32(m0518b, x10m13, self.twiddle4.im);
                let m0518b = vfma_n_f32(m0518b, x11m12, self.twiddle9.im);
                let (y05, y18) = NeonButterfly::butterfly2h_f32(m0518a, m0518b);

                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y04, y05),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vcombine_f32(y18, y19),
                );

                let m0617a = vfma_n_f32(u0, x1p22, self.twiddle6.re);
                let m0617a = vfma_n_f32(m0617a, x2p21, self.twiddle11.re);
                let m0617a = vfma_n_f32(m0617a, x3p20, self.twiddle5.re);
                let m0617a = vfma_n_f32(m0617a, x4p19, self.twiddle1.re);
                let m0617a = vfma_n_f32(m0617a, x5p18, self.twiddle7.re);
                let m0617a = vfma_n_f32(m0617a, x6p17, self.twiddle10.re);
                let m0617a = vfma_n_f32(m0617a, x7p16, self.twiddle4.re);
                let m0617a = vfma_n_f32(m0617a, x8p15, self.twiddle2.re);
                let m0617a = vfma_n_f32(m0617a, x9p14, self.twiddle8.re);
                let m0617a = vfma_n_f32(m0617a, x10p13, self.twiddle9.re);
                let m0617a = vfma_n_f32(m0617a, x11p12, self.twiddle3.re);
                let m0617b = vmul_n_f32(x1m22, self.twiddle6.im);
                let m0617b = vfms_n_f32(m0617b, x2m21, self.twiddle11.im);
                let m0617b = vfms_n_f32(m0617b, x3m20, self.twiddle5.im);
                let m0617b = vfma_n_f32(m0617b, x4m19, self.twiddle1.im);
                let m0617b = vfma_n_f32(m0617b, x5m18, self.twiddle7.im);
                let m0617b = vfms_n_f32(m0617b, x6m17, self.twiddle10.im);
                let m0617b = vfms_n_f32(m0617b, x7m16, self.twiddle4.im);
                let m0617b = vfma_n_f32(m0617b, x8m15, self.twiddle2.im);
                let m0617b = vfma_n_f32(m0617b, x9m14, self.twiddle8.im);
                let m0617b = vfms_n_f32(m0617b, x10m13, self.twiddle9.im);
                let m0617b = vfms_n_f32(m0617b, x11m12, self.twiddle3.im);
                let (y06, y17) = NeonButterfly::butterfly2h_f32(m0617a, m0617b);

                let m0716a = vfma_n_f32(u0, x1p22, self.twiddle7.re);
                let m0716a = vfma_n_f32(m0716a, x2p21, self.twiddle9.re);
                let m0716a = vfma_n_f32(m0716a, x3p20, self.twiddle2.re);
                let m0716a = vfma_n_f32(m0716a, x4p19, self.twiddle5.re);
                let m0716a = vfma_n_f32(m0716a, x5p18, self.twiddle11.re);
                let m0716a = vfma_n_f32(m0716a, x6p17, self.twiddle4.re);
                let m0716a = vfma_n_f32(m0716a, x7p16, self.twiddle3.re);
                let m0716a = vfma_n_f32(m0716a, x8p15, self.twiddle10.re);
                let m0716a = vfma_n_f32(m0716a, x9p14, self.twiddle6.re);
                let m0716a = vfma_n_f32(m0716a, x10p13, self.twiddle1.re);
                let m0716a = vfma_n_f32(m0716a, x11p12, self.twiddle8.re);
                let m0716b = vmul_n_f32(x1m22, self.twiddle7.im);
                let m0716b = vfms_n_f32(m0716b, x2m21, self.twiddle9.im);
                let m0716b = vfms_n_f32(m0716b, x3m20, self.twiddle2.im);
                let m0716b = vfma_n_f32(m0716b, x4m19, self.twiddle5.im);
                let m0716b = vfms_n_f32(m0716b, x5m18, self.twiddle11.im);
                let m0716b = vfms_n_f32(m0716b, x6m17, self.twiddle4.im);
                let m0716b = vfma_n_f32(m0716b, x7m16, self.twiddle3.im);
                let m0716b = vfma_n_f32(m0716b, x8m15, self.twiddle10.im);
                let m0716b = vfms_n_f32(m0716b, x9m14, self.twiddle6.im);
                let m0716b = vfma_n_f32(m0716b, x10m13, self.twiddle1.im);
                let m0716b = vfma_n_f32(m0716b, x11m12, self.twiddle8.im);
                let (y07, y16) = NeonButterfly::butterfly2h_f32(m0716a, m0716b);

                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y06, y07),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vcombine_f32(y16, y17),
                );

                let m0815a = vfma_n_f32(u0, x1p22, self.twiddle8.re);
                let m0815a = vfma_n_f32(m0815a, x2p21, self.twiddle7.re);
                let m0815a = vfma_n_f32(m0815a, x3p20, self.twiddle1.re);
                let m0815a = vfma_n_f32(m0815a, x4p19, self.twiddle9.re);
                let m0815a = vfma_n_f32(m0815a, x5p18, self.twiddle6.re);
                let m0815a = vfma_n_f32(m0815a, x6p17, self.twiddle2.re);
                let m0815a = vfma_n_f32(m0815a, x7p16, self.twiddle10.re);
                let m0815a = vfma_n_f32(m0815a, x8p15, self.twiddle5.re);
                let m0815a = vfma_n_f32(m0815a, x9p14, self.twiddle3.re);
                let m0815a = vfma_n_f32(m0815a, x10p13, self.twiddle11.re);
                let m0815a = vfma_n_f32(m0815a, x11p12, self.twiddle4.re);
                let m0815b = vmul_n_f32(x1m22, self.twiddle8.im);
                let m0815b = vfms_n_f32(m0815b, x2m21, self.twiddle7.im);
                let m0815b = vfma_n_f32(m0815b, x3m20, self.twiddle1.im);
                let m0815b = vfma_n_f32(m0815b, x4m19, self.twiddle9.im);
                let m0815b = vfms_n_f32(m0815b, x5m18, self.twiddle6.im);
                let m0815b = vfma_n_f32(m0815b, x6m17, self.twiddle2.im);
                let m0815b = vfma_n_f32(m0815b, x7m16, self.twiddle10.im);
                let m0815b = vfms_n_f32(m0815b, x8m15, self.twiddle5.im);
                let m0815b = vfma_n_f32(m0815b, x9m14, self.twiddle3.im);
                let m0815b = vfma_n_f32(m0815b, x10m13, self.twiddle11.im);
                let m0815b = vfms_n_f32(m0815b, x11m12, self.twiddle4.im);
                let (y08, y15) = NeonButterfly::butterfly2h_f32(m0815a, m0815b);

                let m0914a = vfma_n_f32(u0, x1p22, self.twiddle9.re);
                let m0914a = vfma_n_f32(m0914a, x2p21, self.twiddle5.re);
                let m0914a = vfma_n_f32(m0914a, x3p20, self.twiddle4.re);
                let m0914a = vfma_n_f32(m0914a, x4p19, self.twiddle10.re);
                let m0914a = vfma_n_f32(m0914a, x5p18, self.twiddle1.re);
                let m0914a = vfma_n_f32(m0914a, x6p17, self.twiddle8.re);
                let m0914a = vfma_n_f32(m0914a, x7p16, self.twiddle6.re);
                let m0914a = vfma_n_f32(m0914a, x8p15, self.twiddle3.re);
                let m0914a = vfma_n_f32(m0914a, x9p14, self.twiddle11.re);
                let m0914a = vfma_n_f32(m0914a, x10p13, self.twiddle2.re);
                let m0914a = vfma_n_f32(m0914a, x11p12, self.twiddle7.re);
                let m0914b = vmul_n_f32(x1m22, self.twiddle9.im);
                let m0914b = vfms_n_f32(m0914b, x2m21, self.twiddle5.im);
                let m0914b = vfma_n_f32(m0914b, x3m20, self.twiddle4.im);
                let m0914b = vfms_n_f32(m0914b, x4m19, self.twiddle10.im);
                let m0914b = vfms_n_f32(m0914b, x5m18, self.twiddle1.im);
                let m0914b = vfma_n_f32(m0914b, x6m17, self.twiddle8.im);
                let m0914b = vfms_n_f32(m0914b, x7m16, self.twiddle6.im);
                let m0914b = vfma_n_f32(m0914b, x8m15, self.twiddle3.im);
                let m0914b = vfms_n_f32(m0914b, x9m14, self.twiddle11.im);
                let m0914b = vfms_n_f32(m0914b, x10m13, self.twiddle2.im);
                let m0914b = vfma_n_f32(m0914b, x11m12, self.twiddle7.im);
                let (y09, y14) = NeonButterfly::butterfly2h_f32(m0914a, m0914b);

                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y08, y09),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vcombine_f32(y14, y15),
                );

                let m1013a = vfma_n_f32(u0, x1p22, self.twiddle10.re);
                let m1013a = vfma_n_f32(m1013a, x2p21, self.twiddle3.re);
                let m1013a = vfma_n_f32(m1013a, x3p20, self.twiddle7.re);
                let m1013a = vfma_n_f32(m1013a, x4p19, self.twiddle6.re);
                let m1013a = vfma_n_f32(m1013a, x5p18, self.twiddle4.re);
                let m1013a = vfma_n_f32(m1013a, x6p17, self.twiddle9.re);
                let m1013a = vfma_n_f32(m1013a, x7p16, self.twiddle1.re);
                let m1013a = vfma_n_f32(m1013a, x8p15, self.twiddle11.re);
                let m1013a = vfma_n_f32(m1013a, x9p14, self.twiddle2.re);
                let m1013a = vfma_n_f32(m1013a, x10p13, self.twiddle8.re);
                let m1013a = vfma_n_f32(m1013a, x11p12, self.twiddle5.re);
                let m1013b = vmul_n_f32(x1m22, self.twiddle10.im);
                let m1013b = vfms_n_f32(m1013b, x2m21, self.twiddle3.im);
                let m1013b = vfma_n_f32(m1013b, x3m20, self.twiddle7.im);
                let m1013b = vfms_n_f32(m1013b, x4m19, self.twiddle6.im);
                let m1013b = vfma_n_f32(m1013b, x5m18, self.twiddle4.im);
                let m1013b = vfms_n_f32(m1013b, x6m17, self.twiddle9.im);
                let m1013b = vfma_n_f32(m1013b, x7m16, self.twiddle1.im);
                let m1013b = vfma_n_f32(m1013b, x8m15, self.twiddle11.im);
                let m1013b = vfms_n_f32(m1013b, x9m14, self.twiddle2.im);
                let m1013b = vfma_n_f32(m1013b, x10m13, self.twiddle8.im);
                let m1013b = vfms_n_f32(m1013b, x11m12, self.twiddle5.im);
                let (y10, y13) = NeonButterfly::butterfly2h_f32(m1013a, m1013b);

                let m1112a = vfma_n_f32(u0, x1p22, self.twiddle11.re);
                let m1112a = vfma_n_f32(m1112a, x2p21, self.twiddle1.re);
                let m1112a = vfma_n_f32(m1112a, x3p20, self.twiddle10.re);
                let m1112a = vfma_n_f32(m1112a, x4p19, self.twiddle2.re);
                let m1112a = vfma_n_f32(m1112a, x5p18, self.twiddle9.re);
                let m1112a = vfma_n_f32(m1112a, x6p17, self.twiddle3.re);
                let m1112a = vfma_n_f32(m1112a, x7p16, self.twiddle8.re);
                let m1112a = vfma_n_f32(m1112a, x8p15, self.twiddle4.re);
                let m1112a = vfma_n_f32(m1112a, x9p14, self.twiddle7.re);
                let m1112a = vfma_n_f32(m1112a, x10p13, self.twiddle5.re);
                let m1112a = vfma_n_f32(m1112a, x11p12, self.twiddle6.re);
                let m1112b = vmul_n_f32(x1m22, self.twiddle11.im);
                let m1112b = vfms_n_f32(m1112b, x2m21, self.twiddle1.im);
                let m1112b = vfma_n_f32(m1112b, x3m20, self.twiddle10.im);
                let m1112b = vfms_n_f32(m1112b, x4m19, self.twiddle2.im);
                let m1112b = vfma_n_f32(m1112b, x5m18, self.twiddle9.im);
                let m1112b = vfms_n_f32(m1112b, x6m17, self.twiddle3.im);
                let m1112b = vfma_n_f32(m1112b, x7m16, self.twiddle8.im);
                let m1112b = vfms_n_f32(m1112b, x8m15, self.twiddle4.im);
                let m1112b = vfma_n_f32(m1112b, x9m14, self.twiddle7.im);
                let m1112b = vfms_n_f32(m1112b, x10m13, self.twiddle5.im);
                let m1112b = vfma_n_f32(m1112b, x11m12, self.twiddle6.im);
                let (y11, y12) = NeonButterfly::butterfly2h_f32(m1112a, m1112b);

                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(y10, y11),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vcombine_f32(y12, y13),
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
    use rand::Rng;

    test_fcma_butterfly!(test_fcma_butterfly23, f32, NeonFcmaButterfly23, 23, 1e-5);
    test_fcma_butterfly!(
        test_fcma_butterfly23_f64,
        f64,
        NeonFcmaButterfly23,
        23,
        1e-7
    );
}
