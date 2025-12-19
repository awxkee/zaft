/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
use crate::util::make_twiddles;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

macro_rules! gen_bf31d {
    ($name: ident, $feature: literal, $rot: ident) => {
        use crate::neon::butterflies::shared::$rot;

        pub(crate) struct $name {
            direction: FftDirection,
            twiddles_re: [f64; 15],
            twiddles_im: [f64; 15],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                let twiddles = make_twiddles::<15, f64>(31, fft_direction);
                Self {
                    direction: fft_direction,
                    twiddles_re: std::array::from_fn(|x| twiddles[x].re),
                    twiddles_im: std::array::from_fn(|x| twiddles[x].im),
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
                31
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(31) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let rotate = $rot::new();

                    for chunk in in_place.chunks_exact_mut(31) {
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
                        let u29 = vld1q_f64(chunk.get_unchecked(29..).as_ptr().cast());
                        let u30 = vld1q_f64(chunk.get_unchecked(30..).as_ptr().cast());

                        let y00 = u0;
                        let (x1p30, x1m30) = NeonButterfly::butterfly2_f64(u1, u30);
                        let x1m30 = rotate.rot(x1m30);
                        let y00 = vaddq_f64(y00, x1p30);
                        let (x2p29, x2m29) = NeonButterfly::butterfly2_f64(u2, u29);
                        let x2m29 = rotate.rot(x2m29);
                        let y00 = vaddq_f64(y00, x2p29);
                        let (x3p28, x3m28) = NeonButterfly::butterfly2_f64(u3, u28);
                        let x3m28 = rotate.rot(x3m28);
                        let y00 = vaddq_f64(y00, x3p28);
                        let (x4p27, x4m27) = NeonButterfly::butterfly2_f64(u4, u27);
                        let x4m27 = rotate.rot(x4m27);
                        let y00 = vaddq_f64(y00, x4p27);
                        let (x5p26, x5m26) = NeonButterfly::butterfly2_f64(u5, u26);
                        let x5m26 = rotate.rot(x5m26);
                        let y00 = vaddq_f64(y00, x5p26);
                        let (x6p25, x6m25) = NeonButterfly::butterfly2_f64(u6, u25);
                        let x6m25 = rotate.rot(x6m25);
                        let y00 = vaddq_f64(y00, x6p25);
                        let (x7p24, x7m24) = NeonButterfly::butterfly2_f64(u7, u24);
                        let x7m24 = rotate.rot(x7m24);
                        let y00 = vaddq_f64(y00, x7p24);
                        let (x8p23, x8m23) = NeonButterfly::butterfly2_f64(u8, u23);
                        let x8m23 = rotate.rot(x8m23);
                        let y00 = vaddq_f64(y00, x8p23);
                        let (x9p22, x9m22) = NeonButterfly::butterfly2_f64(u9, u22);
                        let x9m22 = rotate.rot(x9m22);
                        let y00 = vaddq_f64(y00, x9p22);
                        let (x10p21, x10m21) = NeonButterfly::butterfly2_f64(u10, u21);
                        let x10m21 = rotate.rot(x10m21);
                        let y00 = vaddq_f64(y00, x10p21);
                        let (x11p20, x11m20) = NeonButterfly::butterfly2_f64(u11, u20);
                        let x11m20 = rotate.rot(x11m20);
                        let y00 = vaddq_f64(y00, x11p20);
                        let (x12p19, x12m19) = NeonButterfly::butterfly2_f64(u12, u19);
                        let x12m19 = rotate.rot(x12m19);
                        let y00 = vaddq_f64(y00, x12p19);
                        let (x13p18, x13m18) = NeonButterfly::butterfly2_f64(u13, u18);
                        let x13m18 = rotate.rot(x13m18);
                        let y00 = vaddq_f64(y00, x13p18);
                        let (x14p17, x14m17) = NeonButterfly::butterfly2_f64(u14, u17);
                        let x14m17 = rotate.rot(x14m17);
                        let y00 = vaddq_f64(y00, x14p17);
                        let (x15p16, x15m16) = NeonButterfly::butterfly2_f64(u15, u16);
                        let x15m16 = rotate.rot(x15m16);
                        let y00 = vaddq_f64(y00, x15p16);
                        vst1q_f64(chunk.as_mut_ptr().cast(), y00);

                        let m0130a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[0]);
                        let m0130a = vfmaq_n_f64(m0130a, x2p29, self.twiddles_re[1]);
                        let m0130a = vfmaq_n_f64(m0130a, x3p28, self.twiddles_re[2]);
                        let m0130a = vfmaq_n_f64(m0130a, x4p27, self.twiddles_re[3]);
                        let m0130a = vfmaq_n_f64(m0130a, x5p26, self.twiddles_re[4]);
                        let m0130a = vfmaq_n_f64(m0130a, x6p25, self.twiddles_re[5]);
                        let m0130a = vfmaq_n_f64(m0130a, x7p24, self.twiddles_re[6]);
                        let m0130a = vfmaq_n_f64(m0130a, x8p23, self.twiddles_re[7]);
                        let m0130a = vfmaq_n_f64(m0130a, x9p22, self.twiddles_re[8]);
                        let m0130a = vfmaq_n_f64(m0130a, x10p21, self.twiddles_re[9]);
                        let m0130a = vfmaq_n_f64(m0130a, x11p20, self.twiddles_re[10]);
                        let m0130a = vfmaq_n_f64(m0130a, x12p19, self.twiddles_re[11]);
                        let m0130a = vfmaq_n_f64(m0130a, x13p18, self.twiddles_re[12]);
                        let m0130a = vfmaq_n_f64(m0130a, x14p17, self.twiddles_re[13]);
                        let m0130a = vfmaq_n_f64(m0130a, x15p16, self.twiddles_re[14]);
                        let m0130b = vmulq_n_f64(x1m30, self.twiddles_im[0]);
                        let m0130b = vfmaq_n_f64(m0130b, x2m29, self.twiddles_im[1]);
                        let m0130b = vfmaq_n_f64(m0130b, x3m28, self.twiddles_im[2]);
                        let m0130b = vfmaq_n_f64(m0130b, x4m27, self.twiddles_im[3]);
                        let m0130b = vfmaq_n_f64(m0130b, x5m26, self.twiddles_im[4]);
                        let m0130b = vfmaq_n_f64(m0130b, x6m25, self.twiddles_im[5]);
                        let m0130b = vfmaq_n_f64(m0130b, x7m24, self.twiddles_im[6]);
                        let m0130b = vfmaq_n_f64(m0130b, x8m23, self.twiddles_im[7]);
                        let m0130b = vfmaq_n_f64(m0130b, x9m22, self.twiddles_im[8]);
                        let m0130b = vfmaq_n_f64(m0130b, x10m21, self.twiddles_im[9]);
                        let m0130b = vfmaq_n_f64(m0130b, x11m20, self.twiddles_im[10]);
                        let m0130b = vfmaq_n_f64(m0130b, x12m19, self.twiddles_im[11]);
                        let m0130b = vfmaq_n_f64(m0130b, x13m18, self.twiddles_im[12]);
                        let m0130b = vfmaq_n_f64(m0130b, x14m17, self.twiddles_im[13]);
                        let m0130b = vfmaq_n_f64(m0130b, x15m16, self.twiddles_im[14]);
                        let (y01, y30) = NeonButterfly::butterfly2_f64(m0130a, m0130b);

                        vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y01);
                        vst1q_f64(chunk.get_unchecked_mut(30..).as_mut_ptr().cast(), y30);

                        let m0229a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[1]);
                        let m0229a = vfmaq_n_f64(m0229a, x2p29, self.twiddles_re[3]);
                        let m0229a = vfmaq_n_f64(m0229a, x3p28, self.twiddles_re[5]);
                        let m0229a = vfmaq_n_f64(m0229a, x4p27, self.twiddles_re[7]);
                        let m0229a = vfmaq_n_f64(m0229a, x5p26, self.twiddles_re[9]);
                        let m0229a = vfmaq_n_f64(m0229a, x6p25, self.twiddles_re[11]);
                        let m0229a = vfmaq_n_f64(m0229a, x7p24, self.twiddles_re[13]);
                        let m0229a = vfmaq_n_f64(m0229a, x8p23, self.twiddles_re[14]);
                        let m0229a = vfmaq_n_f64(m0229a, x9p22, self.twiddles_re[12]);
                        let m0229a = vfmaq_n_f64(m0229a, x10p21, self.twiddles_re[10]);
                        let m0229a = vfmaq_n_f64(m0229a, x11p20, self.twiddles_re[8]);
                        let m0229a = vfmaq_n_f64(m0229a, x12p19, self.twiddles_re[6]);
                        let m0229a = vfmaq_n_f64(m0229a, x13p18, self.twiddles_re[4]);
                        let m0229a = vfmaq_n_f64(m0229a, x14p17, self.twiddles_re[2]);
                        let m0229a = vfmaq_n_f64(m0229a, x15p16, self.twiddles_re[0]);
                        let m0229b = vmulq_n_f64(x1m30, self.twiddles_im[1]);
                        let m0229b = vfmaq_n_f64(m0229b, x2m29, self.twiddles_im[3]);
                        let m0229b = vfmaq_n_f64(m0229b, x3m28, self.twiddles_im[5]);
                        let m0229b = vfmaq_n_f64(m0229b, x4m27, self.twiddles_im[7]);
                        let m0229b = vfmaq_n_f64(m0229b, x5m26, self.twiddles_im[9]);
                        let m0229b = vfmaq_n_f64(m0229b, x6m25, self.twiddles_im[11]);
                        let m0229b = vfmaq_n_f64(m0229b, x7m24, self.twiddles_im[13]);
                        let m0229b = vfmsq_n_f64(m0229b, x8m23, self.twiddles_im[14]);
                        let m0229b = vfmsq_n_f64(m0229b, x9m22, self.twiddles_im[12]);
                        let m0229b = vfmsq_n_f64(m0229b, x10m21, self.twiddles_im[10]);
                        let m0229b = vfmsq_n_f64(m0229b, x11m20, self.twiddles_im[8]);
                        let m0229b = vfmsq_n_f64(m0229b, x12m19, self.twiddles_im[6]);
                        let m0229b = vfmsq_n_f64(m0229b, x13m18, self.twiddles_im[4]);
                        let m0229b = vfmsq_n_f64(m0229b, x14m17, self.twiddles_im[2]);
                        let m0229b = vfmsq_n_f64(m0229b, x15m16, self.twiddles_im[0]);
                        let (y02, y29) = NeonButterfly::butterfly2_f64(m0229a, m0229b);

                        vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y02);
                        vst1q_f64(chunk.get_unchecked_mut(29..).as_mut_ptr().cast(), y29);

                        let m0328a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[2]);
                        let m0328a = vfmaq_n_f64(m0328a, x2p29, self.twiddles_re[5]);
                        let m0328a = vfmaq_n_f64(m0328a, x3p28, self.twiddles_re[8]);
                        let m0328a = vfmaq_n_f64(m0328a, x4p27, self.twiddles_re[11]);
                        let m0328a = vfmaq_n_f64(m0328a, x5p26, self.twiddles_re[14]);
                        let m0328a = vfmaq_n_f64(m0328a, x6p25, self.twiddles_re[12]);
                        let m0328a = vfmaq_n_f64(m0328a, x7p24, self.twiddles_re[9]);
                        let m0328a = vfmaq_n_f64(m0328a, x8p23, self.twiddles_re[6]);
                        let m0328a = vfmaq_n_f64(m0328a, x9p22, self.twiddles_re[3]);
                        let m0328a = vfmaq_n_f64(m0328a, x10p21, self.twiddles_re[0]);
                        let m0328a = vfmaq_n_f64(m0328a, x11p20, self.twiddles_re[1]);
                        let m0328a = vfmaq_n_f64(m0328a, x12p19, self.twiddles_re[4]);
                        let m0328a = vfmaq_n_f64(m0328a, x13p18, self.twiddles_re[7]);
                        let m0328a = vfmaq_n_f64(m0328a, x14p17, self.twiddles_re[10]);
                        let m0328a = vfmaq_n_f64(m0328a, x15p16, self.twiddles_re[13]);
                        let m0328b = vmulq_n_f64(x1m30, self.twiddles_im[2]);
                        let m0328b = vfmaq_n_f64(m0328b, x2m29, self.twiddles_im[5]);
                        let m0328b = vfmaq_n_f64(m0328b, x3m28, self.twiddles_im[8]);
                        let m0328b = vfmaq_n_f64(m0328b, x4m27, self.twiddles_im[11]);
                        let m0328b = vfmaq_n_f64(m0328b, x5m26, self.twiddles_im[14]);
                        let m0328b = vfmsq_n_f64(m0328b, x6m25, self.twiddles_im[12]);
                        let m0328b = vfmsq_n_f64(m0328b, x7m24, self.twiddles_im[9]);
                        let m0328b = vfmsq_n_f64(m0328b, x8m23, self.twiddles_im[6]);
                        let m0328b = vfmsq_n_f64(m0328b, x9m22, self.twiddles_im[3]);
                        let m0328b = vfmsq_n_f64(m0328b, x10m21, self.twiddles_im[0]);
                        let m0328b = vfmaq_n_f64(m0328b, x11m20, self.twiddles_im[1]);
                        let m0328b = vfmaq_n_f64(m0328b, x12m19, self.twiddles_im[4]);
                        let m0328b = vfmaq_n_f64(m0328b, x13m18, self.twiddles_im[7]);
                        let m0328b = vfmaq_n_f64(m0328b, x14m17, self.twiddles_im[10]);
                        let m0328b = vfmaq_n_f64(m0328b, x15m16, self.twiddles_im[13]);
                        let (y03, y28) = NeonButterfly::butterfly2_f64(m0328a, m0328b);

                        vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y03);
                        vst1q_f64(chunk.get_unchecked_mut(28..).as_mut_ptr().cast(), y28);

                        let m0427a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[3]);
                        let m0427a = vfmaq_n_f64(m0427a, x2p29, self.twiddles_re[7]);
                        let m0427a = vfmaq_n_f64(m0427a, x3p28, self.twiddles_re[11]);
                        let m0427a = vfmaq_n_f64(m0427a, x4p27, self.twiddles_re[14]);
                        let m0427a = vfmaq_n_f64(m0427a, x5p26, self.twiddles_re[10]);
                        let m0427a = vfmaq_n_f64(m0427a, x6p25, self.twiddles_re[6]);
                        let m0427a = vfmaq_n_f64(m0427a, x7p24, self.twiddles_re[2]);
                        let m0427a = vfmaq_n_f64(m0427a, x8p23, self.twiddles_re[0]);
                        let m0427a = vfmaq_n_f64(m0427a, x9p22, self.twiddles_re[4]);
                        let m0427a = vfmaq_n_f64(m0427a, x10p21, self.twiddles_re[8]);
                        let m0427a = vfmaq_n_f64(m0427a, x11p20, self.twiddles_re[12]);
                        let m0427a = vfmaq_n_f64(m0427a, x12p19, self.twiddles_re[13]);
                        let m0427a = vfmaq_n_f64(m0427a, x13p18, self.twiddles_re[9]);
                        let m0427a = vfmaq_n_f64(m0427a, x14p17, self.twiddles_re[5]);
                        let m0427a = vfmaq_n_f64(m0427a, x15p16, self.twiddles_re[1]);
                        let m0427b = vmulq_n_f64(x1m30, self.twiddles_im[3]);
                        let m0427b = vfmaq_n_f64(m0427b, x2m29, self.twiddles_im[7]);
                        let m0427b = vfmaq_n_f64(m0427b, x3m28, self.twiddles_im[11]);
                        let m0427b = vfmsq_n_f64(m0427b, x4m27, self.twiddles_im[14]);
                        let m0427b = vfmsq_n_f64(m0427b, x5m26, self.twiddles_im[10]);
                        let m0427b = vfmsq_n_f64(m0427b, x6m25, self.twiddles_im[6]);
                        let m0427b = vfmsq_n_f64(m0427b, x7m24, self.twiddles_im[2]);
                        let m0427b = vfmaq_n_f64(m0427b, x8m23, self.twiddles_im[0]);
                        let m0427b = vfmaq_n_f64(m0427b, x9m22, self.twiddles_im[4]);
                        let m0427b = vfmaq_n_f64(m0427b, x10m21, self.twiddles_im[8]);
                        let m0427b = vfmaq_n_f64(m0427b, x11m20, self.twiddles_im[12]);
                        let m0427b = vfmsq_n_f64(m0427b, x12m19, self.twiddles_im[13]);
                        let m0427b = vfmsq_n_f64(m0427b, x13m18, self.twiddles_im[9]);
                        let m0427b = vfmsq_n_f64(m0427b, x14m17, self.twiddles_im[5]);
                        let m0427b = vfmsq_n_f64(m0427b, x15m16, self.twiddles_im[1]);
                        let (y04, y27) = NeonButterfly::butterfly2_f64(m0427a, m0427b);

                        vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y04);
                        vst1q_f64(chunk.get_unchecked_mut(27..).as_mut_ptr().cast(), y27);

                        let m0526a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[4]);
                        let m0526a = vfmaq_n_f64(m0526a, x2p29, self.twiddles_re[9]);
                        let m0526a = vfmaq_n_f64(m0526a, x3p28, self.twiddles_re[14]);
                        let m0526a = vfmaq_n_f64(m0526a, x4p27, self.twiddles_re[10]);
                        let m0526a = vfmaq_n_f64(m0526a, x5p26, self.twiddles_re[5]);
                        let m0526a = vfmaq_n_f64(m0526a, x6p25, self.twiddles_re[0]);
                        let m0526a = vfmaq_n_f64(m0526a, x7p24, self.twiddles_re[3]);
                        let m0526a = vfmaq_n_f64(m0526a, x8p23, self.twiddles_re[8]);
                        let m0526a = vfmaq_n_f64(m0526a, x9p22, self.twiddles_re[13]);
                        let m0526a = vfmaq_n_f64(m0526a, x10p21, self.twiddles_re[11]);
                        let m0526a = vfmaq_n_f64(m0526a, x11p20, self.twiddles_re[6]);
                        let m0526a = vfmaq_n_f64(m0526a, x12p19, self.twiddles_re[1]);
                        let m0526a = vfmaq_n_f64(m0526a, x13p18, self.twiddles_re[2]);
                        let m0526a = vfmaq_n_f64(m0526a, x14p17, self.twiddles_re[7]);
                        let m0526a = vfmaq_n_f64(m0526a, x15p16, self.twiddles_re[12]);
                        let m0526b = vmulq_n_f64(x1m30, self.twiddles_im[4]);
                        let m0526b = vfmaq_n_f64(m0526b, x2m29, self.twiddles_im[9]);
                        let m0526b = vfmaq_n_f64(m0526b, x3m28, self.twiddles_im[14]);
                        let m0526b = vfmsq_n_f64(m0526b, x4m27, self.twiddles_im[10]);
                        let m0526b = vfmsq_n_f64(m0526b, x5m26, self.twiddles_im[5]);
                        let m0526b = vfmsq_n_f64(m0526b, x6m25, self.twiddles_im[0]);
                        let m0526b = vfmaq_n_f64(m0526b, x7m24, self.twiddles_im[3]);
                        let m0526b = vfmaq_n_f64(m0526b, x8m23, self.twiddles_im[8]);
                        let m0526b = vfmaq_n_f64(m0526b, x9m22, self.twiddles_im[13]);
                        let m0526b = vfmsq_n_f64(m0526b, x10m21, self.twiddles_im[11]);
                        let m0526b = vfmsq_n_f64(m0526b, x11m20, self.twiddles_im[6]);
                        let m0526b = vfmsq_n_f64(m0526b, x12m19, self.twiddles_im[1]);
                        let m0526b = vfmaq_n_f64(m0526b, x13m18, self.twiddles_im[2]);
                        let m0526b = vfmaq_n_f64(m0526b, x14m17, self.twiddles_im[7]);
                        let m0526b = vfmaq_n_f64(m0526b, x15m16, self.twiddles_im[12]);
                        let (y05, y26) = NeonButterfly::butterfly2_f64(m0526a, m0526b);

                        vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y05);
                        vst1q_f64(chunk.get_unchecked_mut(26..).as_mut_ptr().cast(), y26);

                        let m0625a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[5]);
                        let m0625a = vfmaq_n_f64(m0625a, x2p29, self.twiddles_re[11]);
                        let m0625a = vfmaq_n_f64(m0625a, x3p28, self.twiddles_re[12]);
                        let m0625a = vfmaq_n_f64(m0625a, x4p27, self.twiddles_re[6]);
                        let m0625a = vfmaq_n_f64(m0625a, x5p26, self.twiddles_re[0]);
                        let m0625a = vfmaq_n_f64(m0625a, x6p25, self.twiddles_re[4]);
                        let m0625a = vfmaq_n_f64(m0625a, x7p24, self.twiddles_re[10]);
                        let m0625a = vfmaq_n_f64(m0625a, x8p23, self.twiddles_re[13]);
                        let m0625a = vfmaq_n_f64(m0625a, x9p22, self.twiddles_re[7]);
                        let m0625a = vfmaq_n_f64(m0625a, x10p21, self.twiddles_re[1]);
                        let m0625a = vfmaq_n_f64(m0625a, x11p20, self.twiddles_re[3]);
                        let m0625a = vfmaq_n_f64(m0625a, x12p19, self.twiddles_re[9]);
                        let m0625a = vfmaq_n_f64(m0625a, x13p18, self.twiddles_re[14]);
                        let m0625a = vfmaq_n_f64(m0625a, x14p17, self.twiddles_re[8]);
                        let m0625a = vfmaq_n_f64(m0625a, x15p16, self.twiddles_re[2]);
                        let m0625b = vmulq_n_f64(x1m30, self.twiddles_im[5]);
                        let m0625b = vfmaq_n_f64(m0625b, x2m29, self.twiddles_im[11]);
                        let m0625b = vfmsq_n_f64(m0625b, x3m28, self.twiddles_im[12]);
                        let m0625b = vfmsq_n_f64(m0625b, x4m27, self.twiddles_im[6]);
                        let m0625b = vfmsq_n_f64(m0625b, x5m26, self.twiddles_im[0]);
                        let m0625b = vfmaq_n_f64(m0625b, x6m25, self.twiddles_im[4]);
                        let m0625b = vfmaq_n_f64(m0625b, x7m24, self.twiddles_im[10]);
                        let m0625b = vfmsq_n_f64(m0625b, x8m23, self.twiddles_im[13]);
                        let m0625b = vfmsq_n_f64(m0625b, x9m22, self.twiddles_im[7]);
                        let m0625b = vfmsq_n_f64(m0625b, x10m21, self.twiddles_im[1]);
                        let m0625b = vfmaq_n_f64(m0625b, x11m20, self.twiddles_im[3]);
                        let m0625b = vfmaq_n_f64(m0625b, x12m19, self.twiddles_im[9]);
                        let m0625b = vfmsq_n_f64(m0625b, x13m18, self.twiddles_im[14]);
                        let m0625b = vfmsq_n_f64(m0625b, x14m17, self.twiddles_im[8]);
                        let m0625b = vfmsq_n_f64(m0625b, x15m16, self.twiddles_im[2]);
                        let (y06, y25) = NeonButterfly::butterfly2_f64(m0625a, m0625b);

                        vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y06);
                        vst1q_f64(chunk.get_unchecked_mut(25..).as_mut_ptr().cast(), y25);

                        let m0724a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[6]);
                        let m0724a = vfmaq_n_f64(m0724a, x2p29, self.twiddles_re[13]);
                        let m0724a = vfmaq_n_f64(m0724a, x3p28, self.twiddles_re[9]);
                        let m0724a = vfmaq_n_f64(m0724a, x4p27, self.twiddles_re[2]);
                        let m0724a = vfmaq_n_f64(m0724a, x5p26, self.twiddles_re[3]);
                        let m0724a = vfmaq_n_f64(m0724a, x6p25, self.twiddles_re[10]);
                        let m0724a = vfmaq_n_f64(m0724a, x7p24, self.twiddles_re[12]);
                        let m0724a = vfmaq_n_f64(m0724a, x8p23, self.twiddles_re[5]);
                        let m0724a = vfmaq_n_f64(m0724a, x9p22, self.twiddles_re[0]);
                        let m0724a = vfmaq_n_f64(m0724a, x10p21, self.twiddles_re[7]);
                        let m0724a = vfmaq_n_f64(m0724a, x11p20, self.twiddles_re[14]);
                        let m0724a = vfmaq_n_f64(m0724a, x12p19, self.twiddles_re[8]);
                        let m0724a = vfmaq_n_f64(m0724a, x13p18, self.twiddles_re[1]);
                        let m0724a = vfmaq_n_f64(m0724a, x14p17, self.twiddles_re[4]);
                        let m0724a = vfmaq_n_f64(m0724a, x15p16, self.twiddles_re[11]);
                        let m0724b = vmulq_n_f64(x1m30, self.twiddles_im[6]);
                        let m0724b = vfmaq_n_f64(m0724b, x2m29, self.twiddles_im[13]);
                        let m0724b = vfmsq_n_f64(m0724b, x3m28, self.twiddles_im[9]);
                        let m0724b = vfmsq_n_f64(m0724b, x4m27, self.twiddles_im[2]);
                        let m0724b = vfmaq_n_f64(m0724b, x5m26, self.twiddles_im[3]);
                        let m0724b = vfmaq_n_f64(m0724b, x6m25, self.twiddles_im[10]);
                        let m0724b = vfmsq_n_f64(m0724b, x7m24, self.twiddles_im[12]);
                        let m0724b = vfmsq_n_f64(m0724b, x8m23, self.twiddles_im[5]);
                        let m0724b = vfmaq_n_f64(m0724b, x9m22, self.twiddles_im[0]);
                        let m0724b = vfmaq_n_f64(m0724b, x10m21, self.twiddles_im[7]);
                        let m0724b = vfmaq_n_f64(m0724b, x11m20, self.twiddles_im[14]);
                        let m0724b = vfmsq_n_f64(m0724b, x12m19, self.twiddles_im[8]);
                        let m0724b = vfmsq_n_f64(m0724b, x13m18, self.twiddles_im[1]);
                        let m0724b = vfmaq_n_f64(m0724b, x14m17, self.twiddles_im[4]);
                        let m0724b = vfmaq_n_f64(m0724b, x15m16, self.twiddles_im[11]);
                        let (y07, y24) = NeonButterfly::butterfly2_f64(m0724a, m0724b);

                        vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y07);
                        vst1q_f64(chunk.get_unchecked_mut(24..).as_mut_ptr().cast(), y24);

                        let m0823a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[7]);
                        let m0823a = vfmaq_n_f64(m0823a, x2p29, self.twiddles_re[14]);
                        let m0823a = vfmaq_n_f64(m0823a, x3p28, self.twiddles_re[6]);
                        let m0823a = vfmaq_n_f64(m0823a, x4p27, self.twiddles_re[0]);
                        let m0823a = vfmaq_n_f64(m0823a, x5p26, self.twiddles_re[8]);
                        let m0823a = vfmaq_n_f64(m0823a, x6p25, self.twiddles_re[13]);
                        let m0823a = vfmaq_n_f64(m0823a, x7p24, self.twiddles_re[5]);
                        let m0823a = vfmaq_n_f64(m0823a, x8p23, self.twiddles_re[1]);
                        let m0823a = vfmaq_n_f64(m0823a, x9p22, self.twiddles_re[9]);
                        let m0823a = vfmaq_n_f64(m0823a, x10p21, self.twiddles_re[12]);
                        let m0823a = vfmaq_n_f64(m0823a, x11p20, self.twiddles_re[4]);
                        let m0823a = vfmaq_n_f64(m0823a, x12p19, self.twiddles_re[2]);
                        let m0823a = vfmaq_n_f64(m0823a, x13p18, self.twiddles_re[10]);
                        let m0823a = vfmaq_n_f64(m0823a, x14p17, self.twiddles_re[11]);
                        let m0823a = vfmaq_n_f64(m0823a, x15p16, self.twiddles_re[3]);
                        let m0823b = vmulq_n_f64(x1m30, self.twiddles_im[7]);
                        let m0823b = vfmsq_n_f64(m0823b, x2m29, self.twiddles_im[14]);
                        let m0823b = vfmsq_n_f64(m0823b, x3m28, self.twiddles_im[6]);
                        let m0823b = vfmaq_n_f64(m0823b, x4m27, self.twiddles_im[0]);
                        let m0823b = vfmaq_n_f64(m0823b, x5m26, self.twiddles_im[8]);
                        let m0823b = vfmsq_n_f64(m0823b, x6m25, self.twiddles_im[13]);
                        let m0823b = vfmsq_n_f64(m0823b, x7m24, self.twiddles_im[5]);
                        let m0823b = vfmaq_n_f64(m0823b, x8m23, self.twiddles_im[1]);
                        let m0823b = vfmaq_n_f64(m0823b, x9m22, self.twiddles_im[9]);
                        let m0823b = vfmsq_n_f64(m0823b, x10m21, self.twiddles_im[12]);
                        let m0823b = vfmsq_n_f64(m0823b, x11m20, self.twiddles_im[4]);
                        let m0823b = vfmaq_n_f64(m0823b, x12m19, self.twiddles_im[2]);
                        let m0823b = vfmaq_n_f64(m0823b, x13m18, self.twiddles_im[10]);
                        let m0823b = vfmsq_n_f64(m0823b, x14m17, self.twiddles_im[11]);
                        let m0823b = vfmsq_n_f64(m0823b, x15m16, self.twiddles_im[3]);
                        let (y08, y23) = NeonButterfly::butterfly2_f64(m0823a, m0823b);

                        vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y08);
                        vst1q_f64(chunk.get_unchecked_mut(23..).as_mut_ptr().cast(), y23);

                        let m0922a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[8]);
                        let m0922a = vfmaq_n_f64(m0922a, x2p29, self.twiddles_re[12]);
                        let m0922a = vfmaq_n_f64(m0922a, x3p28, self.twiddles_re[3]);
                        let m0922a = vfmaq_n_f64(m0922a, x4p27, self.twiddles_re[4]);
                        let m0922a = vfmaq_n_f64(m0922a, x5p26, self.twiddles_re[13]);
                        let m0922a = vfmaq_n_f64(m0922a, x6p25, self.twiddles_re[7]);
                        let m0922a = vfmaq_n_f64(m0922a, x7p24, self.twiddles_re[0]);
                        let m0922a = vfmaq_n_f64(m0922a, x8p23, self.twiddles_re[9]);
                        let m0922a = vfmaq_n_f64(m0922a, x9p22, self.twiddles_re[11]);
                        let m0922a = vfmaq_n_f64(m0922a, x10p21, self.twiddles_re[2]);
                        let m0922a = vfmaq_n_f64(m0922a, x11p20, self.twiddles_re[5]);
                        let m0922a = vfmaq_n_f64(m0922a, x12p19, self.twiddles_re[14]);
                        let m0922a = vfmaq_n_f64(m0922a, x13p18, self.twiddles_re[6]);
                        let m0922a = vfmaq_n_f64(m0922a, x14p17, self.twiddles_re[1]);
                        let m0922a = vfmaq_n_f64(m0922a, x15p16, self.twiddles_re[10]);
                        let m0922b = vmulq_n_f64(x1m30, self.twiddles_im[8]);
                        let m0922b = vfmsq_n_f64(m0922b, x2m29, self.twiddles_im[12]);
                        let m0922b = vfmsq_n_f64(m0922b, x3m28, self.twiddles_im[3]);
                        let m0922b = vfmaq_n_f64(m0922b, x4m27, self.twiddles_im[4]);
                        let m0922b = vfmaq_n_f64(m0922b, x5m26, self.twiddles_im[13]);
                        let m0922b = vfmsq_n_f64(m0922b, x6m25, self.twiddles_im[7]);
                        let m0922b = vfmaq_n_f64(m0922b, x7m24, self.twiddles_im[0]);
                        let m0922b = vfmaq_n_f64(m0922b, x8m23, self.twiddles_im[9]);
                        let m0922b = vfmsq_n_f64(m0922b, x9m22, self.twiddles_im[11]);
                        let m0922b = vfmsq_n_f64(m0922b, x10m21, self.twiddles_im[2]);
                        let m0922b = vfmaq_n_f64(m0922b, x11m20, self.twiddles_im[5]);
                        let m0922b = vfmaq_n_f64(m0922b, x12m19, self.twiddles_im[14]);
                        let m0922b = vfmsq_n_f64(m0922b, x13m18, self.twiddles_im[6]);
                        let m0922b = vfmaq_n_f64(m0922b, x14m17, self.twiddles_im[1]);
                        let m0922b = vfmaq_n_f64(m0922b, x15m16, self.twiddles_im[10]);
                        let (y09, y22) = NeonButterfly::butterfly2_f64(m0922a, m0922b);

                        vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y09);
                        vst1q_f64(chunk.get_unchecked_mut(22..).as_mut_ptr().cast(), y22);

                        let m1021a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[9]);
                        let m1021a = vfmaq_n_f64(m1021a, x2p29, self.twiddles_re[10]);
                        let m1021a = vfmaq_n_f64(m1021a, x3p28, self.twiddles_re[0]);
                        let m1021a = vfmaq_n_f64(m1021a, x4p27, self.twiddles_re[8]);
                        let m1021a = vfmaq_n_f64(m1021a, x5p26, self.twiddles_re[11]);
                        let m1021a = vfmaq_n_f64(m1021a, x6p25, self.twiddles_re[1]);
                        let m1021a = vfmaq_n_f64(m1021a, x7p24, self.twiddles_re[7]);
                        let m1021a = vfmaq_n_f64(m1021a, x8p23, self.twiddles_re[12]);
                        let m1021a = vfmaq_n_f64(m1021a, x9p22, self.twiddles_re[2]);
                        let m1021a = vfmaq_n_f64(m1021a, x10p21, self.twiddles_re[6]);
                        let m1021a = vfmaq_n_f64(m1021a, x11p20, self.twiddles_re[13]);
                        let m1021a = vfmaq_n_f64(m1021a, x12p19, self.twiddles_re[3]);
                        let m1021a = vfmaq_n_f64(m1021a, x13p18, self.twiddles_re[5]);
                        let m1021a = vfmaq_n_f64(m1021a, x14p17, self.twiddles_re[14]);
                        let m1021a = vfmaq_n_f64(m1021a, x15p16, self.twiddles_re[4]);
                        let m1021b = vmulq_n_f64(x1m30, self.twiddles_im[9]);
                        let m1021b = vfmsq_n_f64(m1021b, x2m29, self.twiddles_im[10]);
                        let m1021b = vfmsq_n_f64(m1021b, x3m28, self.twiddles_im[0]);
                        let m1021b = vfmaq_n_f64(m1021b, x4m27, self.twiddles_im[8]);
                        let m1021b = vfmsq_n_f64(m1021b, x5m26, self.twiddles_im[11]);
                        let m1021b = vfmsq_n_f64(m1021b, x6m25, self.twiddles_im[1]);
                        let m1021b = vfmaq_n_f64(m1021b, x7m24, self.twiddles_im[7]);
                        let m1021b = vfmsq_n_f64(m1021b, x8m23, self.twiddles_im[12]);
                        let m1021b = vfmsq_n_f64(m1021b, x9m22, self.twiddles_im[2]);
                        let m1021b = vfmaq_n_f64(m1021b, x10m21, self.twiddles_im[6]);
                        let m1021b = vfmsq_n_f64(m1021b, x11m20, self.twiddles_im[13]);
                        let m1021b = vfmsq_n_f64(m1021b, x12m19, self.twiddles_im[3]);
                        let m1021b = vfmaq_n_f64(m1021b, x13m18, self.twiddles_im[5]);
                        let m1021b = vfmsq_n_f64(m1021b, x14m17, self.twiddles_im[14]);
                        let m1021b = vfmsq_n_f64(m1021b, x15m16, self.twiddles_im[4]);
                        let (y10, y21) = NeonButterfly::butterfly2_f64(m1021a, m1021b);

                        vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);
                        vst1q_f64(chunk.get_unchecked_mut(21..).as_mut_ptr().cast(), y21);

                        let m1120a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[10]);
                        let m1120a = vfmaq_n_f64(m1120a, x2p29, self.twiddles_re[8]);
                        let m1120a = vfmaq_n_f64(m1120a, x3p28, self.twiddles_re[1]);
                        let m1120a = vfmaq_n_f64(m1120a, x4p27, self.twiddles_re[12]);
                        let m1120a = vfmaq_n_f64(m1120a, x5p26, self.twiddles_re[6]);
                        let m1120a = vfmaq_n_f64(m1120a, x6p25, self.twiddles_re[3]);
                        let m1120a = vfmaq_n_f64(m1120a, x7p24, self.twiddles_re[14]);
                        let m1120a = vfmaq_n_f64(m1120a, x8p23, self.twiddles_re[4]);
                        let m1120a = vfmaq_n_f64(m1120a, x9p22, self.twiddles_re[5]);
                        let m1120a = vfmaq_n_f64(m1120a, x10p21, self.twiddles_re[13]);
                        let m1120a = vfmaq_n_f64(m1120a, x11p20, self.twiddles_re[2]);
                        let m1120a = vfmaq_n_f64(m1120a, x12p19, self.twiddles_re[7]);
                        let m1120a = vfmaq_n_f64(m1120a, x13p18, self.twiddles_re[11]);
                        let m1120a = vfmaq_n_f64(m1120a, x14p17, self.twiddles_re[0]);
                        let m1120a = vfmaq_n_f64(m1120a, x15p16, self.twiddles_re[9]);
                        let m1120b = vmulq_n_f64(x1m30, self.twiddles_im[10]);
                        let m1120b = vfmsq_n_f64(m1120b, x2m29, self.twiddles_im[8]);
                        let m1120b = vfmaq_n_f64(m1120b, x3m28, self.twiddles_im[1]);
                        let m1120b = vfmaq_n_f64(m1120b, x4m27, self.twiddles_im[12]);
                        let m1120b = vfmsq_n_f64(m1120b, x5m26, self.twiddles_im[6]);
                        let m1120b = vfmaq_n_f64(m1120b, x6m25, self.twiddles_im[3]);
                        let m1120b = vfmaq_n_f64(m1120b, x7m24, self.twiddles_im[14]);
                        let m1120b = vfmsq_n_f64(m1120b, x8m23, self.twiddles_im[4]);
                        let m1120b = vfmaq_n_f64(m1120b, x9m22, self.twiddles_im[5]);
                        let m1120b = vfmsq_n_f64(m1120b, x10m21, self.twiddles_im[13]);
                        let m1120b = vfmsq_n_f64(m1120b, x11m20, self.twiddles_im[2]);
                        let m1120b = vfmaq_n_f64(m1120b, x12m19, self.twiddles_im[7]);
                        let m1120b = vfmsq_n_f64(m1120b, x13m18, self.twiddles_im[11]);
                        let m1120b = vfmsq_n_f64(m1120b, x14m17, self.twiddles_im[0]);
                        let m1120b = vfmaq_n_f64(m1120b, x15m16, self.twiddles_im[9]);
                        let (y11, y20) = NeonButterfly::butterfly2_f64(m1120a, m1120b);

                        vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), y11);
                        vst1q_f64(chunk.get_unchecked_mut(20..).as_mut_ptr().cast(), y20);

                        let m1219a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[11]);
                        let m1219a = vfmaq_n_f64(m1219a, x2p29, self.twiddles_re[6]);
                        let m1219a = vfmaq_n_f64(m1219a, x3p28, self.twiddles_re[4]);
                        let m1219a = vfmaq_n_f64(m1219a, x4p27, self.twiddles_re[13]);
                        let m1219a = vfmaq_n_f64(m1219a, x5p26, self.twiddles_re[1]);
                        let m1219a = vfmaq_n_f64(m1219a, x6p25, self.twiddles_re[9]);
                        let m1219a = vfmaq_n_f64(m1219a, x7p24, self.twiddles_re[8]);
                        let m1219a = vfmaq_n_f64(m1219a, x8p23, self.twiddles_re[2]);
                        let m1219a = vfmaq_n_f64(m1219a, x9p22, self.twiddles_re[14]);
                        let m1219a = vfmaq_n_f64(m1219a, x10p21, self.twiddles_re[3]);
                        let m1219a = vfmaq_n_f64(m1219a, x11p20, self.twiddles_re[7]);
                        let m1219a = vfmaq_n_f64(m1219a, x12p19, self.twiddles_re[10]);
                        let m1219a = vfmaq_n_f64(m1219a, x13p18, self.twiddles_re[0]);
                        let m1219a = vfmaq_n_f64(m1219a, x14p17, self.twiddles_re[12]);
                        let m1219a = vfmaq_n_f64(m1219a, x15p16, self.twiddles_re[5]);
                        let m1219b = vmulq_n_f64(x1m30, self.twiddles_im[11]);
                        let m1219b = vfmsq_n_f64(m1219b, x2m29, self.twiddles_im[6]);
                        let m1219b = vfmaq_n_f64(m1219b, x3m28, self.twiddles_im[4]);
                        let m1219b = vfmsq_n_f64(m1219b, x4m27, self.twiddles_im[13]);
                        let m1219b = vfmsq_n_f64(m1219b, x5m26, self.twiddles_im[1]);
                        let m1219b = vfmaq_n_f64(m1219b, x6m25, self.twiddles_im[9]);
                        let m1219b = vfmsq_n_f64(m1219b, x7m24, self.twiddles_im[8]);
                        let m1219b = vfmaq_n_f64(m1219b, x8m23, self.twiddles_im[2]);
                        let m1219b = vfmaq_n_f64(m1219b, x9m22, self.twiddles_im[14]);
                        let m1219b = vfmsq_n_f64(m1219b, x10m21, self.twiddles_im[3]);
                        let m1219b = vfmaq_n_f64(m1219b, x11m20, self.twiddles_im[7]);
                        let m1219b = vfmsq_n_f64(m1219b, x12m19, self.twiddles_im[10]);
                        let m1219b = vfmaq_n_f64(m1219b, x13m18, self.twiddles_im[0]);
                        let m1219b = vfmaq_n_f64(m1219b, x14m17, self.twiddles_im[12]);
                        let m1219b = vfmsq_n_f64(m1219b, x15m16, self.twiddles_im[5]);
                        let (y12, y19) = NeonButterfly::butterfly2_f64(m1219a, m1219b);

                        vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y12);
                        vst1q_f64(chunk.get_unchecked_mut(19..).as_mut_ptr().cast(), y19);

                        let m1318a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[12]);
                        let m1318a = vfmaq_n_f64(m1318a, x2p29, self.twiddles_re[4]);
                        let m1318a = vfmaq_n_f64(m1318a, x3p28, self.twiddles_re[7]);
                        let m1318a = vfmaq_n_f64(m1318a, x4p27, self.twiddles_re[9]);
                        let m1318a = vfmaq_n_f64(m1318a, x5p26, self.twiddles_re[2]);
                        let m1318a = vfmaq_n_f64(m1318a, x6p25, self.twiddles_re[14]);
                        let m1318a = vfmaq_n_f64(m1318a, x7p24, self.twiddles_re[1]);
                        let m1318a = vfmaq_n_f64(m1318a, x8p23, self.twiddles_re[10]);
                        let m1318a = vfmaq_n_f64(m1318a, x9p22, self.twiddles_re[6]);
                        let m1318a = vfmaq_n_f64(m1318a, x10p21, self.twiddles_re[5]);
                        let m1318a = vfmaq_n_f64(m1318a, x11p20, self.twiddles_re[11]);
                        let m1318a = vfmaq_n_f64(m1318a, x12p19, self.twiddles_re[0]);
                        let m1318a = vfmaq_n_f64(m1318a, x13p18, self.twiddles_re[13]);
                        let m1318a = vfmaq_n_f64(m1318a, x14p17, self.twiddles_re[3]);
                        let m1318a = vfmaq_n_f64(m1318a, x15p16, self.twiddles_re[8]);
                        let m1318b = vmulq_n_f64(x1m30, self.twiddles_im[12]);
                        let m1318b = vfmsq_n_f64(m1318b, x2m29, self.twiddles_im[4]);
                        let m1318b = vfmaq_n_f64(m1318b, x3m28, self.twiddles_im[7]);
                        let m1318b = vfmsq_n_f64(m1318b, x4m27, self.twiddles_im[9]);
                        let m1318b = vfmaq_n_f64(m1318b, x5m26, self.twiddles_im[2]);
                        let m1318b = vfmsq_n_f64(m1318b, x6m25, self.twiddles_im[14]);
                        let m1318b = vfmsq_n_f64(m1318b, x7m24, self.twiddles_im[1]);
                        let m1318b = vfmaq_n_f64(m1318b, x8m23, self.twiddles_im[10]);
                        let m1318b = vfmsq_n_f64(m1318b, x9m22, self.twiddles_im[6]);
                        let m1318b = vfmaq_n_f64(m1318b, x10m21, self.twiddles_im[5]);
                        let m1318b = vfmsq_n_f64(m1318b, x11m20, self.twiddles_im[11]);
                        let m1318b = vfmaq_n_f64(m1318b, x12m19, self.twiddles_im[0]);
                        let m1318b = vfmaq_n_f64(m1318b, x13m18, self.twiddles_im[13]);
                        let m1318b = vfmsq_n_f64(m1318b, x14m17, self.twiddles_im[3]);
                        let m1318b = vfmaq_n_f64(m1318b, x15m16, self.twiddles_im[8]);
                        let (y13, y18) = NeonButterfly::butterfly2_f64(m1318a, m1318b);

                        vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), y13);
                        vst1q_f64(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), y18);

                        let m1417a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[13]);
                        let m1417a = vfmaq_n_f64(m1417a, x2p29, self.twiddles_re[2]);
                        let m1417a = vfmaq_n_f64(m1417a, x3p28, self.twiddles_re[10]);
                        let m1417a = vfmaq_n_f64(m1417a, x4p27, self.twiddles_re[5]);
                        let m1417a = vfmaq_n_f64(m1417a, x5p26, self.twiddles_re[7]);
                        let m1417a = vfmaq_n_f64(m1417a, x6p25, self.twiddles_re[8]);
                        let m1417a = vfmaq_n_f64(m1417a, x7p24, self.twiddles_re[4]);
                        let m1417a = vfmaq_n_f64(m1417a, x8p23, self.twiddles_re[11]);
                        let m1417a = vfmaq_n_f64(m1417a, x9p22, self.twiddles_re[1]);
                        let m1417a = vfmaq_n_f64(m1417a, x10p21, self.twiddles_re[14]);
                        let m1417a = vfmaq_n_f64(m1417a, x11p20, self.twiddles_re[0]);
                        let m1417a = vfmaq_n_f64(m1417a, x12p19, self.twiddles_re[12]);
                        let m1417a = vfmaq_n_f64(m1417a, x13p18, self.twiddles_re[3]);
                        let m1417a = vfmaq_n_f64(m1417a, x14p17, self.twiddles_re[9]);
                        let m1417a = vfmaq_n_f64(m1417a, x15p16, self.twiddles_re[6]);
                        let m1417b = vmulq_n_f64(x1m30, self.twiddles_im[13]);
                        let m1417b = vfmsq_n_f64(m1417b, x2m29, self.twiddles_im[2]);
                        let m1417b = vfmaq_n_f64(m1417b, x3m28, self.twiddles_im[10]);
                        let m1417b = vfmsq_n_f64(m1417b, x4m27, self.twiddles_im[5]);
                        let m1417b = vfmaq_n_f64(m1417b, x5m26, self.twiddles_im[7]);
                        let m1417b = vfmsq_n_f64(m1417b, x6m25, self.twiddles_im[8]);
                        let m1417b = vfmaq_n_f64(m1417b, x7m24, self.twiddles_im[4]);
                        let m1417b = vfmsq_n_f64(m1417b, x8m23, self.twiddles_im[11]);
                        let m1417b = vfmaq_n_f64(m1417b, x9m22, self.twiddles_im[1]);
                        let m1417b = vfmsq_n_f64(m1417b, x10m21, self.twiddles_im[14]);
                        let m1417b = vfmsq_n_f64(m1417b, x11m20, self.twiddles_im[0]);
                        let m1417b = vfmaq_n_f64(m1417b, x12m19, self.twiddles_im[12]);
                        let m1417b = vfmsq_n_f64(m1417b, x13m18, self.twiddles_im[3]);
                        let m1417b = vfmaq_n_f64(m1417b, x14m17, self.twiddles_im[9]);
                        let m1417b = vfmsq_n_f64(m1417b, x15m16, self.twiddles_im[6]);
                        let (y14, y17) = NeonButterfly::butterfly2_f64(m1417a, m1417b);

                        vst1q_f64(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y14);
                        vst1q_f64(chunk.get_unchecked_mut(17..).as_mut_ptr().cast(), y17);

                        let m1516a = vfmaq_n_f64(u0, x1p30, self.twiddles_re[14]);
                        let m1516a = vfmaq_n_f64(m1516a, x2p29, self.twiddles_re[0]);
                        let m1516a = vfmaq_n_f64(m1516a, x3p28, self.twiddles_re[13]);
                        let m1516a = vfmaq_n_f64(m1516a, x4p27, self.twiddles_re[1]);
                        let m1516a = vfmaq_n_f64(m1516a, x5p26, self.twiddles_re[12]);
                        let m1516a = vfmaq_n_f64(m1516a, x6p25, self.twiddles_re[2]);
                        let m1516a = vfmaq_n_f64(m1516a, x7p24, self.twiddles_re[11]);
                        let m1516a = vfmaq_n_f64(m1516a, x8p23, self.twiddles_re[3]);
                        let m1516a = vfmaq_n_f64(m1516a, x9p22, self.twiddles_re[10]);
                        let m1516a = vfmaq_n_f64(m1516a, x10p21, self.twiddles_re[4]);
                        let m1516a = vfmaq_n_f64(m1516a, x11p20, self.twiddles_re[9]);
                        let m1516a = vfmaq_n_f64(m1516a, x12p19, self.twiddles_re[5]);
                        let m1516a = vfmaq_n_f64(m1516a, x13p18, self.twiddles_re[8]);
                        let m1516a = vfmaq_n_f64(m1516a, x14p17, self.twiddles_re[6]);
                        let m1516a = vfmaq_n_f64(m1516a, x15p16, self.twiddles_re[7]);
                        let m1516b = vmulq_n_f64(x1m30, self.twiddles_im[14]);
                        let m1516b = vfmsq_n_f64(m1516b, x2m29, self.twiddles_im[0]);
                        let m1516b = vfmaq_n_f64(m1516b, x3m28, self.twiddles_im[13]);
                        let m1516b = vfmsq_n_f64(m1516b, x4m27, self.twiddles_im[1]);
                        let m1516b = vfmaq_n_f64(m1516b, x5m26, self.twiddles_im[12]);
                        let m1516b = vfmsq_n_f64(m1516b, x6m25, self.twiddles_im[2]);
                        let m1516b = vfmaq_n_f64(m1516b, x7m24, self.twiddles_im[11]);
                        let m1516b = vfmsq_n_f64(m1516b, x8m23, self.twiddles_im[3]);
                        let m1516b = vfmaq_n_f64(m1516b, x9m22, self.twiddles_im[10]);
                        let m1516b = vfmsq_n_f64(m1516b, x10m21, self.twiddles_im[4]);
                        let m1516b = vfmaq_n_f64(m1516b, x11m20, self.twiddles_im[9]);
                        let m1516b = vfmsq_n_f64(m1516b, x12m19, self.twiddles_im[5]);
                        let m1516b = vfmaq_n_f64(m1516b, x13m18, self.twiddles_im[8]);
                        let m1516b = vfmsq_n_f64(m1516b, x14m17, self.twiddles_im[6]);
                        let m1516b = vfmaq_n_f64(m1516b, x15m16, self.twiddles_im[7]);
                        let (y15, y16) = NeonButterfly::butterfly2_f64(m1516a, m1516b);

                        vst1q_f64(chunk.get_unchecked_mut(15..).as_mut_ptr().cast(), y15);
                        vst1q_f64(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), y16);
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf31d!(NeonButterfly31d, "neon", NeonRotate90D);
#[cfg(feature = "fcma")]
gen_bf31d!(NeonFcmaButterfly31d, "fcma", NeonFcmaRotate90D);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly31_f64, f64, NeonButterfly31d, 31, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly31_f64,
        f64,
        NeonFcmaButterfly31d,
        31,
        1e-7
    );
}
