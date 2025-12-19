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
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

macro_rules! gen_bf19d {
    ($name: ident, $features: literal, $rot: ident) => {
        use crate::neon::butterflies::shared::$rot;

        pub(crate) struct $name {
            direction: FftDirection,
            twiddle1: Complex<f64>,
            twiddle2: Complex<f64>,
            twiddle3: Complex<f64>,
            twiddle4: Complex<f64>,
            twiddle5: Complex<f64>,
            twiddle6: Complex<f64>,
            twiddle7: Complex<f64>,
            twiddle8: Complex<f64>,
            twiddle9: Complex<f64>,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddle1: compute_twiddle(1, 19, fft_direction),
                    twiddle2: compute_twiddle(2, 19, fft_direction),
                    twiddle3: compute_twiddle(3, 19, fft_direction),
                    twiddle4: compute_twiddle(4, 19, fft_direction),
                    twiddle5: compute_twiddle(5, 19, fft_direction),
                    twiddle6: compute_twiddle(6, 19, fft_direction),
                    twiddle7: compute_twiddle(7, 19, fft_direction),
                    twiddle8: compute_twiddle(8, 19, fft_direction),
                    twiddle9: compute_twiddle(9, 19, fft_direction),
                }
            }
        }
        impl FftExecutor<f64> for $name {
            fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                unsafe { self.execute_impl(in_place) }
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                19
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(19) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let rotate = $rot::new();

                    for chunk in in_place.chunks_exact_mut(19) {
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

                        let y00 = u0;
                        let (x1p18, x1m18) = NeonButterfly::butterfly2_f64(u1, u18);
                        let x1m18 = rotate.rot(x1m18);
                        let y00 = vaddq_f64(y00, x1p18);
                        let (x2p17, x2m17) = NeonButterfly::butterfly2_f64(u2, u17);
                        let x2m17 = rotate.rot(x2m17);
                        let y00 = vaddq_f64(y00, x2p17);
                        let (x3p16, x3m16) = NeonButterfly::butterfly2_f64(u3, u16);
                        let x3m16 = rotate.rot(x3m16);
                        let y00 = vaddq_f64(y00, x3p16);
                        let (x4p15, x4m15) = NeonButterfly::butterfly2_f64(u4, u15);
                        let x4m15 = rotate.rot(x4m15);
                        let y00 = vaddq_f64(y00, x4p15);
                        let (x5p14, x5m14) = NeonButterfly::butterfly2_f64(u5, u14);
                        let x5m14 = rotate.rot(x5m14);
                        let y00 = vaddq_f64(y00, x5p14);
                        let (x6p13, x6m13) = NeonButterfly::butterfly2_f64(u6, u13);
                        let x6m13 = rotate.rot(x6m13);
                        let y00 = vaddq_f64(y00, x6p13);
                        let (x7p12, x7m12) = NeonButterfly::butterfly2_f64(u7, u12);
                        let x7m12 = rotate.rot(x7m12);
                        let y00 = vaddq_f64(y00, x7p12);
                        let (x8p11, x8m11) = NeonButterfly::butterfly2_f64(u8, u11);
                        let x8m11 = rotate.rot(x8m11);
                        let y00 = vaddq_f64(y00, x8p11);
                        let (x9p10, x9m10) = NeonButterfly::butterfly2_f64(u9, u10);
                        let x9m10 = rotate.rot(x9m10);
                        let y00 = vaddq_f64(y00, x9p10);

                        vst1q_f64(chunk.as_mut_ptr().cast(), y00);

                        let m0118a = vfmaq_n_f64(u0, x1p18, self.twiddle1.re);
                        let m0118a = vfmaq_n_f64(m0118a, x2p17, self.twiddle2.re);
                        let m0118a = vfmaq_n_f64(m0118a, x3p16, self.twiddle3.re);
                        let m0118a = vfmaq_n_f64(m0118a, x4p15, self.twiddle4.re);
                        let m0118a = vfmaq_n_f64(m0118a, x5p14, self.twiddle5.re);
                        let m0118a = vfmaq_n_f64(m0118a, x6p13, self.twiddle6.re);
                        let m0118a = vfmaq_n_f64(m0118a, x7p12, self.twiddle7.re);
                        let m0118a = vfmaq_n_f64(m0118a, x8p11, self.twiddle8.re);
                        let m0118a = vfmaq_n_f64(m0118a, x9p10, self.twiddle9.re);
                        let m0118b = vmulq_n_f64(x1m18, self.twiddle1.im);
                        let m0118b = vfmaq_n_f64(m0118b, x2m17, self.twiddle2.im);
                        let m0118b = vfmaq_n_f64(m0118b, x3m16, self.twiddle3.im);
                        let m0118b = vfmaq_n_f64(m0118b, x4m15, self.twiddle4.im);
                        let m0118b = vfmaq_n_f64(m0118b, x5m14, self.twiddle5.im);
                        let m0118b = vfmaq_n_f64(m0118b, x6m13, self.twiddle6.im);
                        let m0118b = vfmaq_n_f64(m0118b, x7m12, self.twiddle7.im);
                        let m0118b = vfmaq_n_f64(m0118b, x8m11, self.twiddle8.im);
                        let m0118b = vfmaq_n_f64(m0118b, x9m10, self.twiddle9.im);
                        let (y01, y18) = NeonButterfly::butterfly2_f64(m0118a, m0118b);

                        vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y01);
                        vst1q_f64(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), y18);

                        let m0217a = vfmaq_n_f64(u0, x1p18, self.twiddle2.re);
                        let m0217a = vfmaq_n_f64(m0217a, x2p17, self.twiddle4.re);
                        let m0217a = vfmaq_n_f64(m0217a, x3p16, self.twiddle6.re);
                        let m0217a = vfmaq_n_f64(m0217a, x4p15, self.twiddle8.re);
                        let m0217a = vfmaq_n_f64(m0217a, x5p14, self.twiddle9.re);
                        let m0217a = vfmaq_n_f64(m0217a, x6p13, self.twiddle7.re);
                        let m0217a = vfmaq_n_f64(m0217a, x7p12, self.twiddle5.re);
                        let m0217a = vfmaq_n_f64(m0217a, x8p11, self.twiddle3.re);
                        let m0217a = vfmaq_n_f64(m0217a, x9p10, self.twiddle1.re);
                        let m0217b = vmulq_n_f64(x1m18, self.twiddle2.im);
                        let m0217b = vfmaq_n_f64(m0217b, x2m17, self.twiddle4.im);
                        let m0217b = vfmaq_n_f64(m0217b, x3m16, self.twiddle6.im);
                        let m0217b = vfmaq_n_f64(m0217b, x4m15, self.twiddle8.im);
                        let m0217b = vfmsq_n_f64(m0217b, x5m14, self.twiddle9.im);
                        let m0217b = vfmsq_n_f64(m0217b, x6m13, self.twiddle7.im);
                        let m0217b = vfmsq_n_f64(m0217b, x7m12, self.twiddle5.im);
                        let m0217b = vfmsq_n_f64(m0217b, x8m11, self.twiddle3.im);
                        let m0217b = vfmsq_n_f64(m0217b, x9m10, self.twiddle1.im);
                        let (y02, y17) = NeonButterfly::butterfly2_f64(m0217a, m0217b);

                        vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y02);
                        vst1q_f64(chunk.get_unchecked_mut(17..).as_mut_ptr().cast(), y17);

                        let m0316a = vfmaq_n_f64(u0, x1p18, self.twiddle3.re);
                        let m0316a = vfmaq_n_f64(m0316a, x2p17, self.twiddle6.re);
                        let m0316a = vfmaq_n_f64(m0316a, x3p16, self.twiddle9.re);
                        let m0316a = vfmaq_n_f64(m0316a, x4p15, self.twiddle7.re);
                        let m0316a = vfmaq_n_f64(m0316a, x5p14, self.twiddle4.re);
                        let m0316a = vfmaq_n_f64(m0316a, x6p13, self.twiddle1.re);
                        let m0316a = vfmaq_n_f64(m0316a, x7p12, self.twiddle2.re);
                        let m0316a = vfmaq_n_f64(m0316a, x8p11, self.twiddle5.re);
                        let m0316a = vfmaq_n_f64(m0316a, x9p10, self.twiddle8.re);
                        let m0316b = vmulq_n_f64(x1m18, self.twiddle3.im);
                        let m0316b = vfmaq_n_f64(m0316b, x2m17, self.twiddle6.im);
                        let m0316b = vfmaq_n_f64(m0316b, x3m16, self.twiddle9.im);
                        let m0316b = vfmsq_n_f64(m0316b, x4m15, self.twiddle7.im);
                        let m0316b = vfmsq_n_f64(m0316b, x5m14, self.twiddle4.im);
                        let m0316b = vfmsq_n_f64(m0316b, x6m13, self.twiddle1.im);
                        let m0316b = vfmaq_n_f64(m0316b, x7m12, self.twiddle2.im);
                        let m0316b = vfmaq_n_f64(m0316b, x8m11, self.twiddle5.im);
                        let m0316b = vfmaq_n_f64(m0316b, x9m10, self.twiddle8.im);
                        let (y03, y16) = NeonButterfly::butterfly2_f64(m0316a, m0316b);

                        vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y03);
                        vst1q_f64(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), y16);

                        let m0415a = vfmaq_n_f64(u0, x1p18, self.twiddle4.re);
                        let m0415a = vfmaq_n_f64(m0415a, x2p17, self.twiddle8.re);
                        let m0415a = vfmaq_n_f64(m0415a, x3p16, self.twiddle7.re);
                        let m0415a = vfmaq_n_f64(m0415a, x4p15, self.twiddle3.re);
                        let m0415a = vfmaq_n_f64(m0415a, x5p14, self.twiddle1.re);
                        let m0415a = vfmaq_n_f64(m0415a, x6p13, self.twiddle5.re);
                        let m0415a = vfmaq_n_f64(m0415a, x7p12, self.twiddle9.re);
                        let m0415a = vfmaq_n_f64(m0415a, x8p11, self.twiddle6.re);
                        let m0415a = vfmaq_n_f64(m0415a, x9p10, self.twiddle2.re);
                        let m0415b = vmulq_n_f64(x1m18, self.twiddle4.im);
                        let m0415b = vfmaq_n_f64(m0415b, x2m17, self.twiddle8.im);
                        let m0415b = vfmsq_n_f64(m0415b, x3m16, self.twiddle7.im);
                        let m0415b = vfmsq_n_f64(m0415b, x4m15, self.twiddle3.im);
                        let m0415b = vfmaq_n_f64(m0415b, x5m14, self.twiddle1.im);
                        let m0415b = vfmaq_n_f64(m0415b, x6m13, self.twiddle5.im);
                        let m0415b = vfmaq_n_f64(m0415b, x7m12, self.twiddle9.im);
                        let m0415b = vfmsq_n_f64(m0415b, x8m11, self.twiddle6.im);
                        let m0415b = vfmsq_n_f64(m0415b, x9m10, self.twiddle2.im);
                        let (y04, y15) = NeonButterfly::butterfly2_f64(m0415a, m0415b);

                        vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y04);
                        vst1q_f64(chunk.get_unchecked_mut(15..).as_mut_ptr().cast(), y15);

                        let m0514a = vfmaq_n_f64(u0, x1p18, self.twiddle5.re);
                        let m0514a = vfmaq_n_f64(m0514a, x2p17, self.twiddle9.re);
                        let m0514a = vfmaq_n_f64(m0514a, x3p16, self.twiddle4.re);
                        let m0514a = vfmaq_n_f64(m0514a, x4p15, self.twiddle1.re);
                        let m0514a = vfmaq_n_f64(m0514a, x5p14, self.twiddle6.re);
                        let m0514a = vfmaq_n_f64(m0514a, x6p13, self.twiddle8.re);
                        let m0514a = vfmaq_n_f64(m0514a, x7p12, self.twiddle3.re);
                        let m0514a = vfmaq_n_f64(m0514a, x8p11, self.twiddle2.re);
                        let m0514a = vfmaq_n_f64(m0514a, x9p10, self.twiddle7.re);
                        let m0514b = vmulq_n_f64(x1m18, self.twiddle5.im);
                        let m0514b = vfmsq_n_f64(m0514b, x2m17, self.twiddle9.im);
                        let m0514b = vfmsq_n_f64(m0514b, x3m16, self.twiddle4.im);
                        let m0514b = vfmaq_n_f64(m0514b, x4m15, self.twiddle1.im);
                        let m0514b = vfmaq_n_f64(m0514b, x5m14, self.twiddle6.im);
                        let m0514b = vfmsq_n_f64(m0514b, x6m13, self.twiddle8.im);
                        let m0514b = vfmsq_n_f64(m0514b, x7m12, self.twiddle3.im);
                        let m0514b = vfmaq_n_f64(m0514b, x8m11, self.twiddle2.im);
                        let m0514b = vfmaq_n_f64(m0514b, x9m10, self.twiddle7.im);
                        let (y05, y14) = NeonButterfly::butterfly2_f64(m0514a, m0514b);

                        vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y05);
                        vst1q_f64(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y14);

                        let m0613a = vfmaq_n_f64(u0, x1p18, self.twiddle6.re);
                        let m0613a = vfmaq_n_f64(m0613a, x2p17, self.twiddle7.re);
                        let m0613a = vfmaq_n_f64(m0613a, x3p16, self.twiddle1.re);
                        let m0613a = vfmaq_n_f64(m0613a, x4p15, self.twiddle5.re);
                        let m0613a = vfmaq_n_f64(m0613a, x5p14, self.twiddle8.re);
                        let m0613a = vfmaq_n_f64(m0613a, x6p13, self.twiddle2.re);
                        let m0613a = vfmaq_n_f64(m0613a, x7p12, self.twiddle4.re);
                        let m0613a = vfmaq_n_f64(m0613a, x8p11, self.twiddle9.re);
                        let m0613a = vfmaq_n_f64(m0613a, x9p10, self.twiddle3.re);
                        let m0613b = vmulq_n_f64(x1m18, self.twiddle6.im);
                        let m0613b = vfmsq_n_f64(m0613b, x2m17, self.twiddle7.im);
                        let m0613b = vfmsq_n_f64(m0613b, x3m16, self.twiddle1.im);
                        let m0613b = vfmaq_n_f64(m0613b, x4m15, self.twiddle5.im);
                        let m0613b = vfmsq_n_f64(m0613b, x5m14, self.twiddle8.im);
                        let m0613b = vfmsq_n_f64(m0613b, x6m13, self.twiddle2.im);
                        let m0613b = vfmaq_n_f64(m0613b, x7m12, self.twiddle4.im);
                        let m0613b = vfmsq_n_f64(m0613b, x8m11, self.twiddle9.im);
                        let m0613b = vfmsq_n_f64(m0613b, x9m10, self.twiddle3.im);
                        let (y06, y13) = NeonButterfly::butterfly2_f64(m0613a, m0613b);

                        vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y06);
                        vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), y13);

                        let m0712a = vfmaq_n_f64(u0, x1p18, self.twiddle7.re);
                        let m0712a = vfmaq_n_f64(m0712a, x2p17, self.twiddle5.re);
                        let m0712a = vfmaq_n_f64(m0712a, x3p16, self.twiddle2.re);
                        let m0712a = vfmaq_n_f64(m0712a, x4p15, self.twiddle9.re);
                        let m0712a = vfmaq_n_f64(m0712a, x5p14, self.twiddle3.re);
                        let m0712a = vfmaq_n_f64(m0712a, x6p13, self.twiddle4.re);
                        let m0712a = vfmaq_n_f64(m0712a, x7p12, self.twiddle8.re);
                        let m0712a = vfmaq_n_f64(m0712a, x8p11, self.twiddle1.re);
                        let m0712a = vfmaq_n_f64(m0712a, x9p10, self.twiddle6.re);
                        let m0712b = vmulq_n_f64(x1m18, self.twiddle7.im);
                        let m0712b = vfmsq_n_f64(m0712b, x2m17, self.twiddle5.im);
                        let m0712b = vfmaq_n_f64(m0712b, x3m16, self.twiddle2.im);
                        let m0712b = vfmaq_n_f64(m0712b, x4m15, self.twiddle9.im);
                        let m0712b = vfmsq_n_f64(m0712b, x5m14, self.twiddle3.im);
                        let m0712b = vfmaq_n_f64(m0712b, x6m13, self.twiddle4.im);
                        let m0712b = vfmsq_n_f64(m0712b, x7m12, self.twiddle8.im);
                        let m0712b = vfmsq_n_f64(m0712b, x8m11, self.twiddle1.im);
                        let m0712b = vfmaq_n_f64(m0712b, x9m10, self.twiddle6.im);
                        let (y07, y12) = NeonButterfly::butterfly2_f64(m0712a, m0712b);

                        vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y07);
                        vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y12);

                        let m0811a = vfmaq_n_f64(u0, x1p18, self.twiddle8.re);
                        let m0811a = vfmaq_n_f64(m0811a, x2p17, self.twiddle3.re);
                        let m0811a = vfmaq_n_f64(m0811a, x3p16, self.twiddle5.re);
                        let m0811a = vfmaq_n_f64(m0811a, x4p15, self.twiddle6.re);
                        let m0811a = vfmaq_n_f64(m0811a, x5p14, self.twiddle2.re);
                        let m0811a = vfmaq_n_f64(m0811a, x6p13, self.twiddle9.re);
                        let m0811a = vfmaq_n_f64(m0811a, x7p12, self.twiddle1.re);
                        let m0811a = vfmaq_n_f64(m0811a, x8p11, self.twiddle7.re);
                        let m0811a = vfmaq_n_f64(m0811a, x9p10, self.twiddle4.re);
                        let m0811b = vmulq_n_f64(x1m18, self.twiddle8.im);
                        let m0811b = vfmsq_n_f64(m0811b, x2m17, self.twiddle3.im);
                        let m0811b = vfmaq_n_f64(m0811b, x3m16, self.twiddle5.im);
                        let m0811b = vfmsq_n_f64(m0811b, x4m15, self.twiddle6.im);
                        let m0811b = vfmaq_n_f64(m0811b, x5m14, self.twiddle2.im);
                        let m0811b = vfmsq_n_f64(m0811b, x6m13, self.twiddle9.im);
                        let m0811b = vfmsq_n_f64(m0811b, x7m12, self.twiddle1.im);
                        let m0811b = vfmaq_n_f64(m0811b, x8m11, self.twiddle7.im);
                        let m0811b = vfmsq_n_f64(m0811b, x9m10, self.twiddle4.im);
                        let (y08, y11) = NeonButterfly::butterfly2_f64(m0811a, m0811b);

                        vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y08);
                        vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), y11);

                        let m0910a = vfmaq_n_f64(u0, x1p18, self.twiddle9.re);
                        let m0910a = vfmaq_n_f64(m0910a, x2p17, self.twiddle1.re);
                        let m0910a = vfmaq_n_f64(m0910a, x3p16, self.twiddle8.re);
                        let m0910a = vfmaq_n_f64(m0910a, x4p15, self.twiddle2.re);
                        let m0910a = vfmaq_n_f64(m0910a, x5p14, self.twiddle7.re);
                        let m0910a = vfmaq_n_f64(m0910a, x6p13, self.twiddle3.re);
                        let m0910a = vfmaq_n_f64(m0910a, x7p12, self.twiddle6.re);
                        let m0910a = vfmaq_n_f64(m0910a, x8p11, self.twiddle4.re);
                        let m0910a = vfmaq_n_f64(m0910a, x9p10, self.twiddle5.re);
                        let m0910b = vmulq_n_f64(x1m18, self.twiddle9.im);
                        let m0910b = vfmsq_n_f64(m0910b, x2m17, self.twiddle1.im);
                        let m0910b = vfmaq_n_f64(m0910b, x3m16, self.twiddle8.im);
                        let m0910b = vfmsq_n_f64(m0910b, x4m15, self.twiddle2.im);
                        let m0910b = vfmaq_n_f64(m0910b, x5m14, self.twiddle7.im);
                        let m0910b = vfmsq_n_f64(m0910b, x6m13, self.twiddle3.im);
                        let m0910b = vfmaq_n_f64(m0910b, x7m12, self.twiddle6.im);
                        let m0910b = vfmsq_n_f64(m0910b, x8m11, self.twiddle4.im);
                        let m0910b = vfmaq_n_f64(m0910b, x9m10, self.twiddle5.im);
                        let (y09, y10) = NeonButterfly::butterfly2_f64(m0910a, m0910b);

                        vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y09);
                        vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf19d!(NeonButterfly19d, "neon", NeonRotate90D);
#[cfg(feature = "fcma")]
gen_bf19d!(NeonFcmaButterfly19d, "fcma", NeonFcmaRotate90D);

macro_rules! gen_bf19f {
    ($name: ident, $features: literal, $rot: ident) => {
        use crate::neon::butterflies::shared::$rot;

        pub(crate) struct $name {
            direction: FftDirection,
            twiddle1: Complex<f32>,
            twiddle2: Complex<f32>,
            twiddle3: Complex<f32>,
            twiddle4: Complex<f32>,
            twiddle5: Complex<f32>,
            twiddle6: Complex<f32>,
            twiddle7: Complex<f32>,
            twiddle8: Complex<f32>,
            twiddle9: Complex<f32>,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddle1: compute_twiddle(1, 19, fft_direction),
                    twiddle2: compute_twiddle(2, 19, fft_direction),
                    twiddle3: compute_twiddle(3, 19, fft_direction),
                    twiddle4: compute_twiddle(4, 19, fft_direction),
                    twiddle5: compute_twiddle(5, 19, fft_direction),
                    twiddle6: compute_twiddle(6, 19, fft_direction),
                    twiddle7: compute_twiddle(7, 19, fft_direction),
                    twiddle8: compute_twiddle(8, 19, fft_direction),
                    twiddle9: compute_twiddle(9, 19, fft_direction),
                }
            }
        }

        impl FftExecutor<f32> for $name {
            fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                unsafe { self.execute_impl(in_place) }
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                19
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(19) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let rotate = $rot::new();

                    for chunk in in_place.chunks_exact_mut(38) {
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
                        let u34u35 = vld1q_f32(chunk.get_unchecked(34..).as_ptr().cast());
                        let u36u37 = vld1q_f32(chunk.get_unchecked(36..).as_ptr().cast());

                        let u0 = vcombine_f32(vget_low_f32(u0u1), vget_high_f32(u18u19));
                        let u1 = vcombine_f32(vget_high_f32(u0u1), vget_low_f32(u20u21));
                        let u2 = vcombine_f32(vget_low_f32(u2u3), vget_high_f32(u20u21));
                        let u3 = vcombine_f32(vget_high_f32(u2u3), vget_low_f32(u22u23));
                        let u4 = vcombine_f32(vget_low_f32(u4u5), vget_high_f32(u22u23));
                        let u5 = vcombine_f32(vget_high_f32(u4u5), vget_low_f32(u24u25));
                        let u6 = vcombine_f32(vget_low_f32(u6u7), vget_high_f32(u24u25));
                        let u7 = vcombine_f32(vget_high_f32(u6u7), vget_low_f32(u26u27));
                        let u8 = vcombine_f32(vget_low_f32(u8u9), vget_high_f32(u26u27));
                        let u9 = vcombine_f32(vget_high_f32(u8u9), vget_low_f32(u28u29));
                        let u10 = vcombine_f32(vget_low_f32(u10u11), vget_high_f32(u28u29));
                        let u11 = vcombine_f32(vget_high_f32(u10u11), vget_low_f32(u30u31));
                        let u12 = vcombine_f32(vget_low_f32(u12u13), vget_high_f32(u30u31));
                        let u13 = vcombine_f32(vget_high_f32(u12u13), vget_low_f32(u32u33));
                        let u14 = vcombine_f32(vget_low_f32(u14u15), vget_high_f32(u32u33));
                        let u15 = vcombine_f32(vget_high_f32(u14u15), vget_low_f32(u34u35));
                        let u16 = vcombine_f32(vget_low_f32(u16u17), vget_high_f32(u34u35));
                        let u17 = vcombine_f32(vget_high_f32(u16u17), vget_low_f32(u36u37));
                        let u18 = vcombine_f32(vget_low_f32(u18u19), vget_high_f32(u36u37));

                        let y00 = u0;
                        let (x1p18, x1m18) = NeonButterfly::butterfly2_f32(u1, u18);
                        let x1m18 = rotate.rot(x1m18);
                        let y00 = vaddq_f32(y00, x1p18);
                        let (x2p17, x2m17) = NeonButterfly::butterfly2_f32(u2, u17);
                        let x2m17 = rotate.rot(x2m17);
                        let y00 = vaddq_f32(y00, x2p17);
                        let (x3p16, x3m16) = NeonButterfly::butterfly2_f32(u3, u16);
                        let x3m16 = rotate.rot(x3m16);
                        let y00 = vaddq_f32(y00, x3p16);
                        let (x4p15, x4m15) = NeonButterfly::butterfly2_f32(u4, u15);
                        let x4m15 = rotate.rot(x4m15);
                        let y00 = vaddq_f32(y00, x4p15);
                        let (x5p14, x5m14) = NeonButterfly::butterfly2_f32(u5, u14);
                        let x5m14 = rotate.rot(x5m14);
                        let y00 = vaddq_f32(y00, x5p14);
                        let (x6p13, x6m13) = NeonButterfly::butterfly2_f32(u6, u13);
                        let x6m13 = rotate.rot(x6m13);
                        let y00 = vaddq_f32(y00, x6p13);
                        let (x7p12, x7m12) = NeonButterfly::butterfly2_f32(u7, u12);
                        let x7m12 = rotate.rot(x7m12);
                        let y00 = vaddq_f32(y00, x7p12);
                        let (x8p11, x8m11) = NeonButterfly::butterfly2_f32(u8, u11);
                        let x8m11 = rotate.rot(x8m11);
                        let y00 = vaddq_f32(y00, x8p11);
                        let (x9p10, x9m10) = NeonButterfly::butterfly2_f32(u9, u10);
                        let x9m10 = rotate.rot(x9m10);
                        let y00 = vaddq_f32(y00, x9p10);

                        let m0118a = vfmaq_n_f32(u0, x1p18, self.twiddle1.re);
                        let m0118a = vfmaq_n_f32(m0118a, x2p17, self.twiddle2.re);
                        let m0118a = vfmaq_n_f32(m0118a, x3p16, self.twiddle3.re);
                        let m0118a = vfmaq_n_f32(m0118a, x4p15, self.twiddle4.re);
                        let m0118a = vfmaq_n_f32(m0118a, x5p14, self.twiddle5.re);
                        let m0118a = vfmaq_n_f32(m0118a, x6p13, self.twiddle6.re);
                        let m0118a = vfmaq_n_f32(m0118a, x7p12, self.twiddle7.re);
                        let m0118a = vfmaq_n_f32(m0118a, x8p11, self.twiddle8.re);
                        let m0118a = vfmaq_n_f32(m0118a, x9p10, self.twiddle9.re);
                        let m0118b = vmulq_n_f32(x1m18, self.twiddle1.im);
                        let m0118b = vfmaq_n_f32(m0118b, x2m17, self.twiddle2.im);
                        let m0118b = vfmaq_n_f32(m0118b, x3m16, self.twiddle3.im);
                        let m0118b = vfmaq_n_f32(m0118b, x4m15, self.twiddle4.im);
                        let m0118b = vfmaq_n_f32(m0118b, x5m14, self.twiddle5.im);
                        let m0118b = vfmaq_n_f32(m0118b, x6m13, self.twiddle6.im);
                        let m0118b = vfmaq_n_f32(m0118b, x7m12, self.twiddle7.im);
                        let m0118b = vfmaq_n_f32(m0118b, x8m11, self.twiddle8.im);
                        let m0118b = vfmaq_n_f32(m0118b, x9m10, self.twiddle9.im);
                        let (y01, y18) = NeonButterfly::butterfly2_f32(m0118a, m0118b);

                        vst1q_f32(
                            chunk.as_mut_ptr().cast(),
                            vcombine_f32(vget_low_f32(y00), vget_low_f32(y01)),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                            vcombine_f32(vget_low_f32(y18), vget_high_f32(y00)),
                        );

                        let m0217a = vfmaq_n_f32(u0, x1p18, self.twiddle2.re);
                        let m0217a = vfmaq_n_f32(m0217a, x2p17, self.twiddle4.re);
                        let m0217a = vfmaq_n_f32(m0217a, x3p16, self.twiddle6.re);
                        let m0217a = vfmaq_n_f32(m0217a, x4p15, self.twiddle8.re);
                        let m0217a = vfmaq_n_f32(m0217a, x5p14, self.twiddle9.re);
                        let m0217a = vfmaq_n_f32(m0217a, x6p13, self.twiddle7.re);
                        let m0217a = vfmaq_n_f32(m0217a, x7p12, self.twiddle5.re);
                        let m0217a = vfmaq_n_f32(m0217a, x8p11, self.twiddle3.re);
                        let m0217a = vfmaq_n_f32(m0217a, x9p10, self.twiddle1.re);
                        let m0217b = vmulq_n_f32(x1m18, self.twiddle2.im);
                        let m0217b = vfmaq_n_f32(m0217b, x2m17, self.twiddle4.im);
                        let m0217b = vfmaq_n_f32(m0217b, x3m16, self.twiddle6.im);
                        let m0217b = vfmaq_n_f32(m0217b, x4m15, self.twiddle8.im);
                        let m0217b = vfmsq_n_f32(m0217b, x5m14, self.twiddle9.im);
                        let m0217b = vfmsq_n_f32(m0217b, x6m13, self.twiddle7.im);
                        let m0217b = vfmsq_n_f32(m0217b, x7m12, self.twiddle5.im);
                        let m0217b = vfmsq_n_f32(m0217b, x8m11, self.twiddle3.im);
                        let m0217b = vfmsq_n_f32(m0217b, x9m10, self.twiddle1.im);
                        let (y02, y17) = NeonButterfly::butterfly2_f32(m0217a, m0217b);

                        vst1q_f32(
                            chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                            vcombine_f32(vget_high_f32(y01), vget_high_f32(y02)),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(36..).as_mut_ptr().cast(),
                            vcombine_f32(vget_high_f32(y17), vget_high_f32(y18)),
                        );

                        let m0316a = vfmaq_n_f32(u0, x1p18, self.twiddle3.re);
                        let m0316a = vfmaq_n_f32(m0316a, x2p17, self.twiddle6.re);
                        let m0316a = vfmaq_n_f32(m0316a, x3p16, self.twiddle9.re);
                        let m0316a = vfmaq_n_f32(m0316a, x4p15, self.twiddle7.re);
                        let m0316a = vfmaq_n_f32(m0316a, x5p14, self.twiddle4.re);
                        let m0316a = vfmaq_n_f32(m0316a, x6p13, self.twiddle1.re);
                        let m0316a = vfmaq_n_f32(m0316a, x7p12, self.twiddle2.re);
                        let m0316a = vfmaq_n_f32(m0316a, x8p11, self.twiddle5.re);
                        let m0316a = vfmaq_n_f32(m0316a, x9p10, self.twiddle8.re);
                        let m0316b = vmulq_n_f32(x1m18, self.twiddle3.im);
                        let m0316b = vfmaq_n_f32(m0316b, x2m17, self.twiddle6.im);
                        let m0316b = vfmaq_n_f32(m0316b, x3m16, self.twiddle9.im);
                        let m0316b = vfmsq_n_f32(m0316b, x4m15, self.twiddle7.im);
                        let m0316b = vfmsq_n_f32(m0316b, x5m14, self.twiddle4.im);
                        let m0316b = vfmsq_n_f32(m0316b, x6m13, self.twiddle1.im);
                        let m0316b = vfmaq_n_f32(m0316b, x7m12, self.twiddle2.im);
                        let m0316b = vfmaq_n_f32(m0316b, x8m11, self.twiddle5.im);
                        let m0316b = vfmaq_n_f32(m0316b, x9m10, self.twiddle8.im);
                        let (y03, y16) = NeonButterfly::butterfly2_f32(m0316a, m0316b);

                        vst1q_f32(
                            chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                            vcombine_f32(vget_low_f32(y02), vget_low_f32(y03)),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                            vcombine_f32(vget_low_f32(y16), vget_low_f32(y17)),
                        );

                        let m0415a = vfmaq_n_f32(u0, x1p18, self.twiddle4.re);
                        let m0415a = vfmaq_n_f32(m0415a, x2p17, self.twiddle8.re);
                        let m0415a = vfmaq_n_f32(m0415a, x3p16, self.twiddle7.re);
                        let m0415a = vfmaq_n_f32(m0415a, x4p15, self.twiddle3.re);
                        let m0415a = vfmaq_n_f32(m0415a, x5p14, self.twiddle1.re);
                        let m0415a = vfmaq_n_f32(m0415a, x6p13, self.twiddle5.re);
                        let m0415a = vfmaq_n_f32(m0415a, x7p12, self.twiddle9.re);
                        let m0415a = vfmaq_n_f32(m0415a, x8p11, self.twiddle6.re);
                        let m0415a = vfmaq_n_f32(m0415a, x9p10, self.twiddle2.re);
                        let m0415b = vmulq_n_f32(x1m18, self.twiddle4.im);
                        let m0415b = vfmaq_n_f32(m0415b, x2m17, self.twiddle8.im);
                        let m0415b = vfmsq_n_f32(m0415b, x3m16, self.twiddle7.im);
                        let m0415b = vfmsq_n_f32(m0415b, x4m15, self.twiddle3.im);
                        let m0415b = vfmaq_n_f32(m0415b, x5m14, self.twiddle1.im);
                        let m0415b = vfmaq_n_f32(m0415b, x6m13, self.twiddle5.im);
                        let m0415b = vfmaq_n_f32(m0415b, x7m12, self.twiddle9.im);
                        let m0415b = vfmsq_n_f32(m0415b, x8m11, self.twiddle6.im);
                        let m0415b = vfmsq_n_f32(m0415b, x9m10, self.twiddle2.im);
                        let (y04, y15) = NeonButterfly::butterfly2_f32(m0415a, m0415b);

                        vst1q_f32(
                            chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                            vcombine_f32(vget_high_f32(y03), vget_high_f32(y04)),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(34..).as_mut_ptr().cast(),
                            vcombine_f32(vget_high_f32(y15), vget_high_f32(y16)),
                        );

                        let m0514a = vfmaq_n_f32(u0, x1p18, self.twiddle5.re);
                        let m0514a = vfmaq_n_f32(m0514a, x2p17, self.twiddle9.re);
                        let m0514a = vfmaq_n_f32(m0514a, x3p16, self.twiddle4.re);
                        let m0514a = vfmaq_n_f32(m0514a, x4p15, self.twiddle1.re);
                        let m0514a = vfmaq_n_f32(m0514a, x5p14, self.twiddle6.re);
                        let m0514a = vfmaq_n_f32(m0514a, x6p13, self.twiddle8.re);
                        let m0514a = vfmaq_n_f32(m0514a, x7p12, self.twiddle3.re);
                        let m0514a = vfmaq_n_f32(m0514a, x8p11, self.twiddle2.re);
                        let m0514a = vfmaq_n_f32(m0514a, x9p10, self.twiddle7.re);
                        let m0514b = vmulq_n_f32(x1m18, self.twiddle5.im);
                        let m0514b = vfmsq_n_f32(m0514b, x2m17, self.twiddle9.im);
                        let m0514b = vfmsq_n_f32(m0514b, x3m16, self.twiddle4.im);
                        let m0514b = vfmaq_n_f32(m0514b, x4m15, self.twiddle1.im);
                        let m0514b = vfmaq_n_f32(m0514b, x5m14, self.twiddle6.im);
                        let m0514b = vfmsq_n_f32(m0514b, x6m13, self.twiddle8.im);
                        let m0514b = vfmsq_n_f32(m0514b, x7m12, self.twiddle3.im);
                        let m0514b = vfmaq_n_f32(m0514b, x8m11, self.twiddle2.im);
                        let m0514b = vfmaq_n_f32(m0514b, x9m10, self.twiddle7.im);
                        let (y05, y14) = NeonButterfly::butterfly2_f32(m0514a, m0514b);

                        vst1q_f32(
                            chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                            vcombine_f32(vget_low_f32(y04), vget_low_f32(y05)),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                            vcombine_f32(vget_low_f32(y14), vget_low_f32(y15)),
                        );

                        let m0613a = vfmaq_n_f32(u0, x1p18, self.twiddle6.re);
                        let m0613a = vfmaq_n_f32(m0613a, x2p17, self.twiddle7.re);
                        let m0613a = vfmaq_n_f32(m0613a, x3p16, self.twiddle1.re);
                        let m0613a = vfmaq_n_f32(m0613a, x4p15, self.twiddle5.re);
                        let m0613a = vfmaq_n_f32(m0613a, x5p14, self.twiddle8.re);
                        let m0613a = vfmaq_n_f32(m0613a, x6p13, self.twiddle2.re);
                        let m0613a = vfmaq_n_f32(m0613a, x7p12, self.twiddle4.re);
                        let m0613a = vfmaq_n_f32(m0613a, x8p11, self.twiddle9.re);
                        let m0613a = vfmaq_n_f32(m0613a, x9p10, self.twiddle3.re);
                        let m0613b = vmulq_n_f32(x1m18, self.twiddle6.im);
                        let m0613b = vfmsq_n_f32(m0613b, x2m17, self.twiddle7.im);
                        let m0613b = vfmsq_n_f32(m0613b, x3m16, self.twiddle1.im);
                        let m0613b = vfmaq_n_f32(m0613b, x4m15, self.twiddle5.im);
                        let m0613b = vfmsq_n_f32(m0613b, x5m14, self.twiddle8.im);
                        let m0613b = vfmsq_n_f32(m0613b, x6m13, self.twiddle2.im);
                        let m0613b = vfmaq_n_f32(m0613b, x7m12, self.twiddle4.im);
                        let m0613b = vfmsq_n_f32(m0613b, x8m11, self.twiddle9.im);
                        let m0613b = vfmsq_n_f32(m0613b, x9m10, self.twiddle3.im);
                        let (y06, y13) = NeonButterfly::butterfly2_f32(m0613a, m0613b);

                        vst1q_f32(
                            chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                            vcombine_f32(vget_high_f32(y05), vget_high_f32(y06)),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(32..).as_mut_ptr().cast(),
                            vcombine_f32(vget_high_f32(y13), vget_high_f32(y14)),
                        );

                        let m0712a = vfmaq_n_f32(u0, x1p18, self.twiddle7.re);
                        let m0712a = vfmaq_n_f32(m0712a, x2p17, self.twiddle5.re);
                        let m0712a = vfmaq_n_f32(m0712a, x3p16, self.twiddle2.re);
                        let m0712a = vfmaq_n_f32(m0712a, x4p15, self.twiddle9.re);
                        let m0712a = vfmaq_n_f32(m0712a, x5p14, self.twiddle3.re);
                        let m0712a = vfmaq_n_f32(m0712a, x6p13, self.twiddle4.re);
                        let m0712a = vfmaq_n_f32(m0712a, x7p12, self.twiddle8.re);
                        let m0712a = vfmaq_n_f32(m0712a, x8p11, self.twiddle1.re);
                        let m0712a = vfmaq_n_f32(m0712a, x9p10, self.twiddle6.re);
                        let m0712b = vmulq_n_f32(x1m18, self.twiddle7.im);
                        let m0712b = vfmsq_n_f32(m0712b, x2m17, self.twiddle5.im);
                        let m0712b = vfmaq_n_f32(m0712b, x3m16, self.twiddle2.im);
                        let m0712b = vfmaq_n_f32(m0712b, x4m15, self.twiddle9.im);
                        let m0712b = vfmsq_n_f32(m0712b, x5m14, self.twiddle3.im);
                        let m0712b = vfmaq_n_f32(m0712b, x6m13, self.twiddle4.im);
                        let m0712b = vfmsq_n_f32(m0712b, x7m12, self.twiddle8.im);
                        let m0712b = vfmsq_n_f32(m0712b, x8m11, self.twiddle1.im);
                        let m0712b = vfmaq_n_f32(m0712b, x9m10, self.twiddle6.im);
                        let (y07, y12) = NeonButterfly::butterfly2_f32(m0712a, m0712b);

                        vst1q_f32(
                            chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                            vcombine_f32(vget_low_f32(y06), vget_low_f32(y07)),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                            vcombine_f32(vget_low_f32(y12), vget_low_f32(y13)),
                        );

                        let m0811a = vfmaq_n_f32(u0, x1p18, self.twiddle8.re);
                        let m0811a = vfmaq_n_f32(m0811a, x2p17, self.twiddle3.re);
                        let m0811a = vfmaq_n_f32(m0811a, x3p16, self.twiddle5.re);
                        let m0811a = vfmaq_n_f32(m0811a, x4p15, self.twiddle6.re);
                        let m0811a = vfmaq_n_f32(m0811a, x5p14, self.twiddle2.re);
                        let m0811a = vfmaq_n_f32(m0811a, x6p13, self.twiddle9.re);
                        let m0811a = vfmaq_n_f32(m0811a, x7p12, self.twiddle1.re);
                        let m0811a = vfmaq_n_f32(m0811a, x8p11, self.twiddle7.re);
                        let m0811a = vfmaq_n_f32(m0811a, x9p10, self.twiddle4.re);
                        let m0811b = vmulq_n_f32(x1m18, self.twiddle8.im);
                        let m0811b = vfmsq_n_f32(m0811b, x2m17, self.twiddle3.im);
                        let m0811b = vfmaq_n_f32(m0811b, x3m16, self.twiddle5.im);
                        let m0811b = vfmsq_n_f32(m0811b, x4m15, self.twiddle6.im);
                        let m0811b = vfmaq_n_f32(m0811b, x5m14, self.twiddle2.im);
                        let m0811b = vfmsq_n_f32(m0811b, x6m13, self.twiddle9.im);
                        let m0811b = vfmsq_n_f32(m0811b, x7m12, self.twiddle1.im);
                        let m0811b = vfmaq_n_f32(m0811b, x8m11, self.twiddle7.im);
                        let m0811b = vfmsq_n_f32(m0811b, x9m10, self.twiddle4.im);
                        let (y08, y11) = NeonButterfly::butterfly2_f32(m0811a, m0811b);

                        vst1q_f32(
                            chunk.get_unchecked_mut(26..).as_mut_ptr().cast(),
                            vcombine_f32(vget_high_f32(y07), vget_high_f32(y08)),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(30..).as_mut_ptr().cast(),
                            vcombine_f32(vget_high_f32(y11), vget_high_f32(y12)),
                        );

                        let m0910a = vfmaq_n_f32(u0, x1p18, self.twiddle9.re);
                        let m0910a = vfmaq_n_f32(m0910a, x2p17, self.twiddle1.re);
                        let m0910a = vfmaq_n_f32(m0910a, x3p16, self.twiddle8.re);
                        let m0910a = vfmaq_n_f32(m0910a, x4p15, self.twiddle2.re);
                        let m0910a = vfmaq_n_f32(m0910a, x5p14, self.twiddle7.re);
                        let m0910a = vfmaq_n_f32(m0910a, x6p13, self.twiddle3.re);
                        let m0910a = vfmaq_n_f32(m0910a, x7p12, self.twiddle6.re);
                        let m0910a = vfmaq_n_f32(m0910a, x8p11, self.twiddle4.re);
                        let m0910a = vfmaq_n_f32(m0910a, x9p10, self.twiddle5.re);
                        let m0910b = vmulq_n_f32(x1m18, self.twiddle9.im);
                        let m0910b = vfmsq_n_f32(m0910b, x2m17, self.twiddle1.im);
                        let m0910b = vfmaq_n_f32(m0910b, x3m16, self.twiddle8.im);
                        let m0910b = vfmsq_n_f32(m0910b, x4m15, self.twiddle2.im);
                        let m0910b = vfmaq_n_f32(m0910b, x5m14, self.twiddle7.im);
                        let m0910b = vfmsq_n_f32(m0910b, x6m13, self.twiddle3.im);
                        let m0910b = vfmaq_n_f32(m0910b, x7m12, self.twiddle6.im);
                        let m0910b = vfmsq_n_f32(m0910b, x8m11, self.twiddle4.im);
                        let m0910b = vfmaq_n_f32(m0910b, x9m10, self.twiddle5.im);
                        let (y09, y10) = NeonButterfly::butterfly2_f32(m0910a, m0910b);

                        vst1q_f32(
                            chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                            vcombine_f32(vget_low_f32(y08), vget_low_f32(y09)),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                            vcombine_f32(vget_low_f32(y10), vget_low_f32(y11)),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(28..).as_mut_ptr().cast(),
                            vcombine_f32(vget_high_f32(y09), vget_high_f32(y10)),
                        );
                    }

                    let rem = in_place.chunks_exact_mut(38).into_remainder();

                    for chunk in rem.chunks_exact_mut(19) {
                        let u0u1 = vld1q_f32(chunk.as_mut_ptr().cast());
                        let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                        let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                        let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                        let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                        let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                        let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                        let u14u15 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());
                        let u16u17 = vld1q_f32(chunk.get_unchecked(16..).as_ptr().cast());
                        let u18 = vld1_f32(chunk.get_unchecked(18..).as_ptr().cast());

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

                        let y00 = u0;
                        let (x1p18, x1m18) = NeonButterfly::butterfly2h_f32(u1, u18);
                        let x1m18 = rotate.roth(x1m18);
                        let y00 = vadd_f32(y00, x1p18);
                        let (x2p17, x2m17) = NeonButterfly::butterfly2h_f32(u2, u17);
                        let x2m17 = rotate.roth(x2m17);
                        let y00 = vadd_f32(y00, x2p17);
                        let (x3p16, x3m16) = NeonButterfly::butterfly2h_f32(u3, u16);
                        let x3m16 = rotate.roth(x3m16);
                        let y00 = vadd_f32(y00, x3p16);
                        let (x4p15, x4m15) = NeonButterfly::butterfly2h_f32(u4, u15);
                        let x4m15 = rotate.roth(x4m15);
                        let y00 = vadd_f32(y00, x4p15);
                        let (x5p14, x5m14) = NeonButterfly::butterfly2h_f32(u5, u14);
                        let x5m14 = rotate.roth(x5m14);
                        let y00 = vadd_f32(y00, x5p14);
                        let (x6p13, x6m13) = NeonButterfly::butterfly2h_f32(u6, u13);
                        let x6m13 = rotate.roth(x6m13);
                        let y00 = vadd_f32(y00, x6p13);
                        let (x7p12, x7m12) = NeonButterfly::butterfly2h_f32(u7, u12);
                        let x7m12 = rotate.roth(x7m12);
                        let y00 = vadd_f32(y00, x7p12);
                        let (x8p11, x8m11) = NeonButterfly::butterfly2h_f32(u8, u11);
                        let x8m11 = rotate.roth(x8m11);
                        let y00 = vadd_f32(y00, x8p11);
                        let (x9p10, x9m10) = NeonButterfly::butterfly2h_f32(u9, u10);
                        let x9m10 = rotate.roth(x9m10);
                        let y00 = vadd_f32(y00, x9p10);

                        let m0118a = vfma_n_f32(u0, x1p18, self.twiddle1.re);
                        let m0118a = vfma_n_f32(m0118a, x2p17, self.twiddle2.re);
                        let m0118a = vfma_n_f32(m0118a, x3p16, self.twiddle3.re);
                        let m0118a = vfma_n_f32(m0118a, x4p15, self.twiddle4.re);
                        let m0118a = vfma_n_f32(m0118a, x5p14, self.twiddle5.re);
                        let m0118a = vfma_n_f32(m0118a, x6p13, self.twiddle6.re);
                        let m0118a = vfma_n_f32(m0118a, x7p12, self.twiddle7.re);
                        let m0118a = vfma_n_f32(m0118a, x8p11, self.twiddle8.re);
                        let m0118a = vfma_n_f32(m0118a, x9p10, self.twiddle9.re);
                        let m0118b = vmul_n_f32(x1m18, self.twiddle1.im);
                        let m0118b = vfma_n_f32(m0118b, x2m17, self.twiddle2.im);
                        let m0118b = vfma_n_f32(m0118b, x3m16, self.twiddle3.im);
                        let m0118b = vfma_n_f32(m0118b, x4m15, self.twiddle4.im);
                        let m0118b = vfma_n_f32(m0118b, x5m14, self.twiddle5.im);
                        let m0118b = vfma_n_f32(m0118b, x6m13, self.twiddle6.im);
                        let m0118b = vfma_n_f32(m0118b, x7m12, self.twiddle7.im);
                        let m0118b = vfma_n_f32(m0118b, x8m11, self.twiddle8.im);
                        let m0118b = vfma_n_f32(m0118b, x9m10, self.twiddle9.im);
                        let (y01, y18) = NeonButterfly::butterfly2h_f32(m0118a, m0118b);

                        vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y00, y01));
                        vst1_f32(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), y18);

                        let m0217a = vfma_n_f32(u0, x1p18, self.twiddle2.re);
                        let m0217a = vfma_n_f32(m0217a, x2p17, self.twiddle4.re);
                        let m0217a = vfma_n_f32(m0217a, x3p16, self.twiddle6.re);
                        let m0217a = vfma_n_f32(m0217a, x4p15, self.twiddle8.re);
                        let m0217a = vfma_n_f32(m0217a, x5p14, self.twiddle9.re);
                        let m0217a = vfma_n_f32(m0217a, x6p13, self.twiddle7.re);
                        let m0217a = vfma_n_f32(m0217a, x7p12, self.twiddle5.re);
                        let m0217a = vfma_n_f32(m0217a, x8p11, self.twiddle3.re);
                        let m0217a = vfma_n_f32(m0217a, x9p10, self.twiddle1.re);
                        let m0217b = vmul_n_f32(x1m18, self.twiddle2.im);
                        let m0217b = vfma_n_f32(m0217b, x2m17, self.twiddle4.im);
                        let m0217b = vfma_n_f32(m0217b, x3m16, self.twiddle6.im);
                        let m0217b = vfma_n_f32(m0217b, x4m15, self.twiddle8.im);
                        let m0217b = vfms_n_f32(m0217b, x5m14, self.twiddle9.im);
                        let m0217b = vfms_n_f32(m0217b, x6m13, self.twiddle7.im);
                        let m0217b = vfms_n_f32(m0217b, x7m12, self.twiddle5.im);
                        let m0217b = vfms_n_f32(m0217b, x8m11, self.twiddle3.im);
                        let m0217b = vfms_n_f32(m0217b, x9m10, self.twiddle1.im);
                        let (y02, y17) = NeonButterfly::butterfly2h_f32(m0217a, m0217b);

                        let m0316a = vfma_n_f32(u0, x1p18, self.twiddle3.re);
                        let m0316a = vfma_n_f32(m0316a, x2p17, self.twiddle6.re);
                        let m0316a = vfma_n_f32(m0316a, x3p16, self.twiddle9.re);
                        let m0316a = vfma_n_f32(m0316a, x4p15, self.twiddle7.re);
                        let m0316a = vfma_n_f32(m0316a, x5p14, self.twiddle4.re);
                        let m0316a = vfma_n_f32(m0316a, x6p13, self.twiddle1.re);
                        let m0316a = vfma_n_f32(m0316a, x7p12, self.twiddle2.re);
                        let m0316a = vfma_n_f32(m0316a, x8p11, self.twiddle5.re);
                        let m0316a = vfma_n_f32(m0316a, x9p10, self.twiddle8.re);
                        let m0316b = vmul_n_f32(x1m18, self.twiddle3.im);
                        let m0316b = vfma_n_f32(m0316b, x2m17, self.twiddle6.im);
                        let m0316b = vfma_n_f32(m0316b, x3m16, self.twiddle9.im);
                        let m0316b = vfms_n_f32(m0316b, x4m15, self.twiddle7.im);
                        let m0316b = vfms_n_f32(m0316b, x5m14, self.twiddle4.im);
                        let m0316b = vfms_n_f32(m0316b, x6m13, self.twiddle1.im);
                        let m0316b = vfma_n_f32(m0316b, x7m12, self.twiddle2.im);
                        let m0316b = vfma_n_f32(m0316b, x8m11, self.twiddle5.im);
                        let m0316b = vfma_n_f32(m0316b, x9m10, self.twiddle8.im);
                        let (y03, y16) = NeonButterfly::butterfly2h_f32(m0316a, m0316b);

                        vst1q_f32(
                            chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                            vcombine_f32(y02, y03),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                            vcombine_f32(y16, y17),
                        );

                        let m0415a = vfma_n_f32(u0, x1p18, self.twiddle4.re);
                        let m0415a = vfma_n_f32(m0415a, x2p17, self.twiddle8.re);
                        let m0415a = vfma_n_f32(m0415a, x3p16, self.twiddle7.re);
                        let m0415a = vfma_n_f32(m0415a, x4p15, self.twiddle3.re);
                        let m0415a = vfma_n_f32(m0415a, x5p14, self.twiddle1.re);
                        let m0415a = vfma_n_f32(m0415a, x6p13, self.twiddle5.re);
                        let m0415a = vfma_n_f32(m0415a, x7p12, self.twiddle9.re);
                        let m0415a = vfma_n_f32(m0415a, x8p11, self.twiddle6.re);
                        let m0415a = vfma_n_f32(m0415a, x9p10, self.twiddle2.re);
                        let m0415b = vmul_n_f32(x1m18, self.twiddle4.im);
                        let m0415b = vfma_n_f32(m0415b, x2m17, self.twiddle8.im);
                        let m0415b = vfms_n_f32(m0415b, x3m16, self.twiddle7.im);
                        let m0415b = vfms_n_f32(m0415b, x4m15, self.twiddle3.im);
                        let m0415b = vfma_n_f32(m0415b, x5m14, self.twiddle1.im);
                        let m0415b = vfma_n_f32(m0415b, x6m13, self.twiddle5.im);
                        let m0415b = vfma_n_f32(m0415b, x7m12, self.twiddle9.im);
                        let m0415b = vfms_n_f32(m0415b, x8m11, self.twiddle6.im);
                        let m0415b = vfms_n_f32(m0415b, x9m10, self.twiddle2.im);
                        let (y04, y15) = NeonButterfly::butterfly2h_f32(m0415a, m0415b);

                        let m0514a = vfma_n_f32(u0, x1p18, self.twiddle5.re);
                        let m0514a = vfma_n_f32(m0514a, x2p17, self.twiddle9.re);
                        let m0514a = vfma_n_f32(m0514a, x3p16, self.twiddle4.re);
                        let m0514a = vfma_n_f32(m0514a, x4p15, self.twiddle1.re);
                        let m0514a = vfma_n_f32(m0514a, x5p14, self.twiddle6.re);
                        let m0514a = vfma_n_f32(m0514a, x6p13, self.twiddle8.re);
                        let m0514a = vfma_n_f32(m0514a, x7p12, self.twiddle3.re);
                        let m0514a = vfma_n_f32(m0514a, x8p11, self.twiddle2.re);
                        let m0514a = vfma_n_f32(m0514a, x9p10, self.twiddle7.re);
                        let m0514b = vmul_n_f32(x1m18, self.twiddle5.im);
                        let m0514b = vfms_n_f32(m0514b, x2m17, self.twiddle9.im);
                        let m0514b = vfms_n_f32(m0514b, x3m16, self.twiddle4.im);
                        let m0514b = vfma_n_f32(m0514b, x4m15, self.twiddle1.im);
                        let m0514b = vfma_n_f32(m0514b, x5m14, self.twiddle6.im);
                        let m0514b = vfms_n_f32(m0514b, x6m13, self.twiddle8.im);
                        let m0514b = vfms_n_f32(m0514b, x7m12, self.twiddle3.im);
                        let m0514b = vfma_n_f32(m0514b, x8m11, self.twiddle2.im);
                        let m0514b = vfma_n_f32(m0514b, x9m10, self.twiddle7.im);
                        let (y05, y14) = NeonButterfly::butterfly2h_f32(m0514a, m0514b);

                        vst1q_f32(
                            chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                            vcombine_f32(y04, y05),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                            vcombine_f32(y14, y15),
                        );

                        let m0613a = vfma_n_f32(u0, x1p18, self.twiddle6.re);
                        let m0613a = vfma_n_f32(m0613a, x2p17, self.twiddle7.re);
                        let m0613a = vfma_n_f32(m0613a, x3p16, self.twiddle1.re);
                        let m0613a = vfma_n_f32(m0613a, x4p15, self.twiddle5.re);
                        let m0613a = vfma_n_f32(m0613a, x5p14, self.twiddle8.re);
                        let m0613a = vfma_n_f32(m0613a, x6p13, self.twiddle2.re);
                        let m0613a = vfma_n_f32(m0613a, x7p12, self.twiddle4.re);
                        let m0613a = vfma_n_f32(m0613a, x8p11, self.twiddle9.re);
                        let m0613a = vfma_n_f32(m0613a, x9p10, self.twiddle3.re);
                        let m0613b = vmul_n_f32(x1m18, self.twiddle6.im);
                        let m0613b = vfms_n_f32(m0613b, x2m17, self.twiddle7.im);
                        let m0613b = vfms_n_f32(m0613b, x3m16, self.twiddle1.im);
                        let m0613b = vfma_n_f32(m0613b, x4m15, self.twiddle5.im);
                        let m0613b = vfms_n_f32(m0613b, x5m14, self.twiddle8.im);
                        let m0613b = vfms_n_f32(m0613b, x6m13, self.twiddle2.im);
                        let m0613b = vfma_n_f32(m0613b, x7m12, self.twiddle4.im);
                        let m0613b = vfms_n_f32(m0613b, x8m11, self.twiddle9.im);
                        let m0613b = vfms_n_f32(m0613b, x9m10, self.twiddle3.im);
                        let (y06, y13) = NeonButterfly::butterfly2h_f32(m0613a, m0613b);

                        let m0712a = vfma_n_f32(u0, x1p18, self.twiddle7.re);
                        let m0712a = vfma_n_f32(m0712a, x2p17, self.twiddle5.re);
                        let m0712a = vfma_n_f32(m0712a, x3p16, self.twiddle2.re);
                        let m0712a = vfma_n_f32(m0712a, x4p15, self.twiddle9.re);
                        let m0712a = vfma_n_f32(m0712a, x5p14, self.twiddle3.re);
                        let m0712a = vfma_n_f32(m0712a, x6p13, self.twiddle4.re);
                        let m0712a = vfma_n_f32(m0712a, x7p12, self.twiddle8.re);
                        let m0712a = vfma_n_f32(m0712a, x8p11, self.twiddle1.re);
                        let m0712a = vfma_n_f32(m0712a, x9p10, self.twiddle6.re);
                        let m0712b = vmul_n_f32(x1m18, self.twiddle7.im);
                        let m0712b = vfms_n_f32(m0712b, x2m17, self.twiddle5.im);
                        let m0712b = vfma_n_f32(m0712b, x3m16, self.twiddle2.im);
                        let m0712b = vfma_n_f32(m0712b, x4m15, self.twiddle9.im);
                        let m0712b = vfms_n_f32(m0712b, x5m14, self.twiddle3.im);
                        let m0712b = vfma_n_f32(m0712b, x6m13, self.twiddle4.im);
                        let m0712b = vfms_n_f32(m0712b, x7m12, self.twiddle8.im);
                        let m0712b = vfms_n_f32(m0712b, x8m11, self.twiddle1.im);
                        let m0712b = vfma_n_f32(m0712b, x9m10, self.twiddle6.im);
                        let (y07, y12) = NeonButterfly::butterfly2h_f32(m0712a, m0712b);

                        vst1q_f32(
                            chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                            vcombine_f32(y06, y07),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                            vcombine_f32(y12, y13),
                        );

                        let m0811a = vfma_n_f32(u0, x1p18, self.twiddle8.re);
                        let m0811a = vfma_n_f32(m0811a, x2p17, self.twiddle3.re);
                        let m0811a = vfma_n_f32(m0811a, x3p16, self.twiddle5.re);
                        let m0811a = vfma_n_f32(m0811a, x4p15, self.twiddle6.re);
                        let m0811a = vfma_n_f32(m0811a, x5p14, self.twiddle2.re);
                        let m0811a = vfma_n_f32(m0811a, x6p13, self.twiddle9.re);
                        let m0811a = vfma_n_f32(m0811a, x7p12, self.twiddle1.re);
                        let m0811a = vfma_n_f32(m0811a, x8p11, self.twiddle7.re);
                        let m0811a = vfma_n_f32(m0811a, x9p10, self.twiddle4.re);
                        let m0811b = vmul_n_f32(x1m18, self.twiddle8.im);
                        let m0811b = vfms_n_f32(m0811b, x2m17, self.twiddle3.im);
                        let m0811b = vfma_n_f32(m0811b, x3m16, self.twiddle5.im);
                        let m0811b = vfms_n_f32(m0811b, x4m15, self.twiddle6.im);
                        let m0811b = vfma_n_f32(m0811b, x5m14, self.twiddle2.im);
                        let m0811b = vfms_n_f32(m0811b, x6m13, self.twiddle9.im);
                        let m0811b = vfms_n_f32(m0811b, x7m12, self.twiddle1.im);
                        let m0811b = vfma_n_f32(m0811b, x8m11, self.twiddle7.im);
                        let m0811b = vfms_n_f32(m0811b, x9m10, self.twiddle4.im);
                        let (y08, y11) = NeonButterfly::butterfly2h_f32(m0811a, m0811b);

                        let m0910a = vfma_n_f32(u0, x1p18, self.twiddle9.re);
                        let m0910a = vfma_n_f32(m0910a, x2p17, self.twiddle1.re);
                        let m0910a = vfma_n_f32(m0910a, x3p16, self.twiddle8.re);
                        let m0910a = vfma_n_f32(m0910a, x4p15, self.twiddle2.re);
                        let m0910a = vfma_n_f32(m0910a, x5p14, self.twiddle7.re);
                        let m0910a = vfma_n_f32(m0910a, x6p13, self.twiddle3.re);
                        let m0910a = vfma_n_f32(m0910a, x7p12, self.twiddle6.re);
                        let m0910a = vfma_n_f32(m0910a, x8p11, self.twiddle4.re);
                        let m0910a = vfma_n_f32(m0910a, x9p10, self.twiddle5.re);
                        let m0910b = vmul_n_f32(x1m18, self.twiddle9.im);
                        let m0910b = vfms_n_f32(m0910b, x2m17, self.twiddle1.im);
                        let m0910b = vfma_n_f32(m0910b, x3m16, self.twiddle8.im);
                        let m0910b = vfms_n_f32(m0910b, x4m15, self.twiddle2.im);
                        let m0910b = vfma_n_f32(m0910b, x5m14, self.twiddle7.im);
                        let m0910b = vfms_n_f32(m0910b, x6m13, self.twiddle3.im);
                        let m0910b = vfma_n_f32(m0910b, x7m12, self.twiddle6.im);
                        let m0910b = vfms_n_f32(m0910b, x8m11, self.twiddle4.im);
                        let m0910b = vfma_n_f32(m0910b, x9m10, self.twiddle5.im);
                        let (y09, y10) = NeonButterfly::butterfly2h_f32(m0910a, m0910b);

                        vst1q_f32(
                            chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                            vcombine_f32(y08, y09),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                            vcombine_f32(y10, y11),
                        );
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf19f!(NeonButterfly19f, "neon", NeonRotate90F);
#[cfg(feature = "fcma")]
gen_bf19f!(NeonFcmaButterfly19f, "fcma", NeonFcmaRotate90F);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly19, f32, NeonButterfly19f, 19, 1e-5);
    #[cfg(feature = "fcma")]
    test_butterfly!(test_fcma_butterfly19, f32, NeonFcmaButterfly19f, 19, 1e-5);
    test_butterfly!(test_neon_butterfly19_f64, f64, NeonButterfly19d, 19, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly19_f64,
        f64,
        NeonFcmaButterfly19d,
        19,
        1e-7
    );
}
