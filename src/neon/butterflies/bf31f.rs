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
use crate::neon::util::vh_rotate90_f32;
use crate::util::make_twiddles;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) struct NeonButterfly31f {
    direction: FftDirection,
    twiddles_re: [f32; 15],
    twiddles_im: [f32; 15],
}

impl NeonButterfly31f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddles = make_twiddles::<15, f32>(31, fft_direction);
        Self {
            direction: fft_direction,
            twiddles_re: std::array::from_fn(|x| twiddles[x].re),
            twiddles_im: std::array::from_fn(|x| twiddles[x].im),
        }
    }
}

impl FftExecutor<f32> for NeonButterfly31f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 31 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            static ROT_90: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            let rot_sign = vld1_f32(ROT_90.as_ptr());

            for chunk in in_place.chunks_exact_mut(31) {
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
                let u28u29 = vld1q_f32(chunk.get_unchecked(28..).as_ptr().cast());
                let u30 = vld1_f32(chunk.get_unchecked(30..).as_ptr().cast());

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
                let u28 = vget_low_f32(u28u29);
                let u29 = vget_high_f32(u28u29);

                let y00 = u0;
                let (x1p30, x1m30) = NeonButterfly::butterfly2h_f32(u1, u30);
                let x1m30 = vh_rotate90_f32(x1m30, rot_sign);
                let y00 = vadd_f32(y00, x1p30);
                let (x2p29, x2m29) = NeonButterfly::butterfly2h_f32(u2, u29);
                let x2m29 = vh_rotate90_f32(x2m29, rot_sign);
                let y00 = vadd_f32(y00, x2p29);
                let (x3p28, x3m28) = NeonButterfly::butterfly2h_f32(u3, u28);
                let x3m28 = vh_rotate90_f32(x3m28, rot_sign);
                let y00 = vadd_f32(y00, x3p28);
                let (x4p27, x4m27) = NeonButterfly::butterfly2h_f32(u4, u27);
                let x4m27 = vh_rotate90_f32(x4m27, rot_sign);
                let y00 = vadd_f32(y00, x4p27);
                let (x5p26, x5m26) = NeonButterfly::butterfly2h_f32(u5, u26);
                let x5m26 = vh_rotate90_f32(x5m26, rot_sign);
                let y00 = vadd_f32(y00, x5p26);
                let (x6p25, x6m25) = NeonButterfly::butterfly2h_f32(u6, u25);
                let x6m25 = vh_rotate90_f32(x6m25, rot_sign);
                let y00 = vadd_f32(y00, x6p25);
                let (x7p24, x7m24) = NeonButterfly::butterfly2h_f32(u7, u24);
                let x7m24 = vh_rotate90_f32(x7m24, rot_sign);
                let y00 = vadd_f32(y00, x7p24);
                let (x8p23, x8m23) = NeonButterfly::butterfly2h_f32(u8, u23);
                let x8m23 = vh_rotate90_f32(x8m23, rot_sign);
                let y00 = vadd_f32(y00, x8p23);
                let (x9p22, x9m22) = NeonButterfly::butterfly2h_f32(u9, u22);
                let x9m22 = vh_rotate90_f32(x9m22, rot_sign);
                let y00 = vadd_f32(y00, x9p22);
                let (x10p21, x10m21) = NeonButterfly::butterfly2h_f32(u10, u21);
                let x10m21 = vh_rotate90_f32(x10m21, rot_sign);
                let y00 = vadd_f32(y00, x10p21);
                let (x11p20, x11m20) = NeonButterfly::butterfly2h_f32(u11, u20);
                let x11m20 = vh_rotate90_f32(x11m20, rot_sign);
                let y00 = vadd_f32(y00, x11p20);
                let (x12p19, x12m19) = NeonButterfly::butterfly2h_f32(u12, u19);
                let x12m19 = vh_rotate90_f32(x12m19, rot_sign);
                let y00 = vadd_f32(y00, x12p19);
                let (x13p18, x13m18) = NeonButterfly::butterfly2h_f32(u13, u18);
                let x13m18 = vh_rotate90_f32(x13m18, rot_sign);
                let y00 = vadd_f32(y00, x13p18);
                let (x14p17, x14m17) = NeonButterfly::butterfly2h_f32(u14, u17);
                let x14m17 = vh_rotate90_f32(x14m17, rot_sign);
                let y00 = vadd_f32(y00, x14p17);
                let (x15p16, x15m16) = NeonButterfly::butterfly2h_f32(u15, u16);
                let x15m16 = vh_rotate90_f32(x15m16, rot_sign);
                let y00 = vadd_f32(y00, x15p16);

                let m0130a = vfma_n_f32(u0, x1p30, self.twiddles_re[0]);
                let m0130a = vfma_n_f32(m0130a, x2p29, self.twiddles_re[1]);
                let m0130a = vfma_n_f32(m0130a, x3p28, self.twiddles_re[2]);
                let m0130a = vfma_n_f32(m0130a, x4p27, self.twiddles_re[3]);
                let m0130a = vfma_n_f32(m0130a, x5p26, self.twiddles_re[4]);
                let m0130a = vfma_n_f32(m0130a, x6p25, self.twiddles_re[5]);
                let m0130a = vfma_n_f32(m0130a, x7p24, self.twiddles_re[6]);
                let m0130a = vfma_n_f32(m0130a, x8p23, self.twiddles_re[7]);
                let m0130a = vfma_n_f32(m0130a, x9p22, self.twiddles_re[8]);
                let m0130a = vfma_n_f32(m0130a, x10p21, self.twiddles_re[9]);
                let m0130a = vfma_n_f32(m0130a, x11p20, self.twiddles_re[10]);
                let m0130a = vfma_n_f32(m0130a, x12p19, self.twiddles_re[11]);
                let m0130a = vfma_n_f32(m0130a, x13p18, self.twiddles_re[12]);
                let m0130a = vfma_n_f32(m0130a, x14p17, self.twiddles_re[13]);
                let m0130a = vfma_n_f32(m0130a, x15p16, self.twiddles_re[14]);
                let m0130b = vmul_n_f32(x1m30, self.twiddles_im[0]);
                let m0130b = vfma_n_f32(m0130b, x2m29, self.twiddles_im[1]);
                let m0130b = vfma_n_f32(m0130b, x3m28, self.twiddles_im[2]);
                let m0130b = vfma_n_f32(m0130b, x4m27, self.twiddles_im[3]);
                let m0130b = vfma_n_f32(m0130b, x5m26, self.twiddles_im[4]);
                let m0130b = vfma_n_f32(m0130b, x6m25, self.twiddles_im[5]);
                let m0130b = vfma_n_f32(m0130b, x7m24, self.twiddles_im[6]);
                let m0130b = vfma_n_f32(m0130b, x8m23, self.twiddles_im[7]);
                let m0130b = vfma_n_f32(m0130b, x9m22, self.twiddles_im[8]);
                let m0130b = vfma_n_f32(m0130b, x10m21, self.twiddles_im[9]);
                let m0130b = vfma_n_f32(m0130b, x11m20, self.twiddles_im[10]);
                let m0130b = vfma_n_f32(m0130b, x12m19, self.twiddles_im[11]);
                let m0130b = vfma_n_f32(m0130b, x13m18, self.twiddles_im[12]);
                let m0130b = vfma_n_f32(m0130b, x14m17, self.twiddles_im[13]);
                let m0130b = vfma_n_f32(m0130b, x15m16, self.twiddles_im[14]);
                let (y01, y30) = NeonButterfly::butterfly2h_f32(m0130a, m0130b);

                vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y00, y01));
                vst1_f32(chunk.get_unchecked_mut(30..).as_mut_ptr().cast(), y30);

                let m0229a = vfma_n_f32(u0, x1p30, self.twiddles_re[1]);
                let m0229a = vfma_n_f32(m0229a, x2p29, self.twiddles_re[3]);
                let m0229a = vfma_n_f32(m0229a, x3p28, self.twiddles_re[5]);
                let m0229a = vfma_n_f32(m0229a, x4p27, self.twiddles_re[7]);
                let m0229a = vfma_n_f32(m0229a, x5p26, self.twiddles_re[9]);
                let m0229a = vfma_n_f32(m0229a, x6p25, self.twiddles_re[11]);
                let m0229a = vfma_n_f32(m0229a, x7p24, self.twiddles_re[13]);
                let m0229a = vfma_n_f32(m0229a, x8p23, self.twiddles_re[14]);
                let m0229a = vfma_n_f32(m0229a, x9p22, self.twiddles_re[12]);
                let m0229a = vfma_n_f32(m0229a, x10p21, self.twiddles_re[10]);
                let m0229a = vfma_n_f32(m0229a, x11p20, self.twiddles_re[8]);
                let m0229a = vfma_n_f32(m0229a, x12p19, self.twiddles_re[6]);
                let m0229a = vfma_n_f32(m0229a, x13p18, self.twiddles_re[4]);
                let m0229a = vfma_n_f32(m0229a, x14p17, self.twiddles_re[2]);
                let m0229a = vfma_n_f32(m0229a, x15p16, self.twiddles_re[0]);
                let m0229b = vmul_n_f32(x1m30, self.twiddles_im[1]);
                let m0229b = vfma_n_f32(m0229b, x2m29, self.twiddles_im[3]);
                let m0229b = vfma_n_f32(m0229b, x3m28, self.twiddles_im[5]);
                let m0229b = vfma_n_f32(m0229b, x4m27, self.twiddles_im[7]);
                let m0229b = vfma_n_f32(m0229b, x5m26, self.twiddles_im[9]);
                let m0229b = vfma_n_f32(m0229b, x6m25, self.twiddles_im[11]);
                let m0229b = vfma_n_f32(m0229b, x7m24, self.twiddles_im[13]);
                let m0229b = vfms_n_f32(m0229b, x8m23, self.twiddles_im[14]);
                let m0229b = vfms_n_f32(m0229b, x9m22, self.twiddles_im[12]);
                let m0229b = vfms_n_f32(m0229b, x10m21, self.twiddles_im[10]);
                let m0229b = vfms_n_f32(m0229b, x11m20, self.twiddles_im[8]);
                let m0229b = vfms_n_f32(m0229b, x12m19, self.twiddles_im[6]);
                let m0229b = vfms_n_f32(m0229b, x13m18, self.twiddles_im[4]);
                let m0229b = vfms_n_f32(m0229b, x14m17, self.twiddles_im[2]);
                let m0229b = vfms_n_f32(m0229b, x15m16, self.twiddles_im[0]);
                let (y02, y29) = NeonButterfly::butterfly2h_f32(m0229a, m0229b);

                let m0328a = vfma_n_f32(u0, x1p30, self.twiddles_re[2]);
                let m0328a = vfma_n_f32(m0328a, x2p29, self.twiddles_re[5]);
                let m0328a = vfma_n_f32(m0328a, x3p28, self.twiddles_re[8]);
                let m0328a = vfma_n_f32(m0328a, x4p27, self.twiddles_re[11]);
                let m0328a = vfma_n_f32(m0328a, x5p26, self.twiddles_re[14]);
                let m0328a = vfma_n_f32(m0328a, x6p25, self.twiddles_re[12]);
                let m0328a = vfma_n_f32(m0328a, x7p24, self.twiddles_re[9]);
                let m0328a = vfma_n_f32(m0328a, x8p23, self.twiddles_re[6]);
                let m0328a = vfma_n_f32(m0328a, x9p22, self.twiddles_re[3]);
                let m0328a = vfma_n_f32(m0328a, x10p21, self.twiddles_re[0]);
                let m0328a = vfma_n_f32(m0328a, x11p20, self.twiddles_re[1]);
                let m0328a = vfma_n_f32(m0328a, x12p19, self.twiddles_re[4]);
                let m0328a = vfma_n_f32(m0328a, x13p18, self.twiddles_re[7]);
                let m0328a = vfma_n_f32(m0328a, x14p17, self.twiddles_re[10]);
                let m0328a = vfma_n_f32(m0328a, x15p16, self.twiddles_re[13]);
                let m0328b = vmul_n_f32(x1m30, self.twiddles_im[2]);
                let m0328b = vfma_n_f32(m0328b, x2m29, self.twiddles_im[5]);
                let m0328b = vfma_n_f32(m0328b, x3m28, self.twiddles_im[8]);
                let m0328b = vfma_n_f32(m0328b, x4m27, self.twiddles_im[11]);
                let m0328b = vfma_n_f32(m0328b, x5m26, self.twiddles_im[14]);
                let m0328b = vfms_n_f32(m0328b, x6m25, self.twiddles_im[12]);
                let m0328b = vfms_n_f32(m0328b, x7m24, self.twiddles_im[9]);
                let m0328b = vfms_n_f32(m0328b, x8m23, self.twiddles_im[6]);
                let m0328b = vfms_n_f32(m0328b, x9m22, self.twiddles_im[3]);
                let m0328b = vfms_n_f32(m0328b, x10m21, self.twiddles_im[0]);
                let m0328b = vfma_n_f32(m0328b, x11m20, self.twiddles_im[1]);
                let m0328b = vfma_n_f32(m0328b, x12m19, self.twiddles_im[4]);
                let m0328b = vfma_n_f32(m0328b, x13m18, self.twiddles_im[7]);
                let m0328b = vfma_n_f32(m0328b, x14m17, self.twiddles_im[10]);
                let m0328b = vfma_n_f32(m0328b, x15m16, self.twiddles_im[13]);
                let (y03, y28) = NeonButterfly::butterfly2h_f32(m0328a, m0328b);

                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y02, y03),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    vcombine_f32(y28, y29),
                );

                let m0427a = vfma_n_f32(u0, x1p30, self.twiddles_re[3]);
                let m0427a = vfma_n_f32(m0427a, x2p29, self.twiddles_re[7]);
                let m0427a = vfma_n_f32(m0427a, x3p28, self.twiddles_re[11]);
                let m0427a = vfma_n_f32(m0427a, x4p27, self.twiddles_re[14]);
                let m0427a = vfma_n_f32(m0427a, x5p26, self.twiddles_re[10]);
                let m0427a = vfma_n_f32(m0427a, x6p25, self.twiddles_re[6]);
                let m0427a = vfma_n_f32(m0427a, x7p24, self.twiddles_re[2]);
                let m0427a = vfma_n_f32(m0427a, x8p23, self.twiddles_re[0]);
                let m0427a = vfma_n_f32(m0427a, x9p22, self.twiddles_re[4]);
                let m0427a = vfma_n_f32(m0427a, x10p21, self.twiddles_re[8]);
                let m0427a = vfma_n_f32(m0427a, x11p20, self.twiddles_re[12]);
                let m0427a = vfma_n_f32(m0427a, x12p19, self.twiddles_re[13]);
                let m0427a = vfma_n_f32(m0427a, x13p18, self.twiddles_re[9]);
                let m0427a = vfma_n_f32(m0427a, x14p17, self.twiddles_re[5]);
                let m0427a = vfma_n_f32(m0427a, x15p16, self.twiddles_re[1]);
                let m0427b = vmul_n_f32(x1m30, self.twiddles_im[3]);
                let m0427b = vfma_n_f32(m0427b, x2m29, self.twiddles_im[7]);
                let m0427b = vfma_n_f32(m0427b, x3m28, self.twiddles_im[11]);
                let m0427b = vfms_n_f32(m0427b, x4m27, self.twiddles_im[14]);
                let m0427b = vfms_n_f32(m0427b, x5m26, self.twiddles_im[10]);
                let m0427b = vfms_n_f32(m0427b, x6m25, self.twiddles_im[6]);
                let m0427b = vfms_n_f32(m0427b, x7m24, self.twiddles_im[2]);
                let m0427b = vfma_n_f32(m0427b, x8m23, self.twiddles_im[0]);
                let m0427b = vfma_n_f32(m0427b, x9m22, self.twiddles_im[4]);
                let m0427b = vfma_n_f32(m0427b, x10m21, self.twiddles_im[8]);
                let m0427b = vfma_n_f32(m0427b, x11m20, self.twiddles_im[12]);
                let m0427b = vfms_n_f32(m0427b, x12m19, self.twiddles_im[13]);
                let m0427b = vfms_n_f32(m0427b, x13m18, self.twiddles_im[9]);
                let m0427b = vfms_n_f32(m0427b, x14m17, self.twiddles_im[5]);
                let m0427b = vfms_n_f32(m0427b, x15m16, self.twiddles_im[1]);
                let (y04, y27) = NeonButterfly::butterfly2h_f32(m0427a, m0427b);

                let m0526a = vfma_n_f32(u0, x1p30, self.twiddles_re[4]);
                let m0526a = vfma_n_f32(m0526a, x2p29, self.twiddles_re[9]);
                let m0526a = vfma_n_f32(m0526a, x3p28, self.twiddles_re[14]);
                let m0526a = vfma_n_f32(m0526a, x4p27, self.twiddles_re[10]);
                let m0526a = vfma_n_f32(m0526a, x5p26, self.twiddles_re[5]);
                let m0526a = vfma_n_f32(m0526a, x6p25, self.twiddles_re[0]);
                let m0526a = vfma_n_f32(m0526a, x7p24, self.twiddles_re[3]);
                let m0526a = vfma_n_f32(m0526a, x8p23, self.twiddles_re[8]);
                let m0526a = vfma_n_f32(m0526a, x9p22, self.twiddles_re[13]);
                let m0526a = vfma_n_f32(m0526a, x10p21, self.twiddles_re[11]);
                let m0526a = vfma_n_f32(m0526a, x11p20, self.twiddles_re[6]);
                let m0526a = vfma_n_f32(m0526a, x12p19, self.twiddles_re[1]);
                let m0526a = vfma_n_f32(m0526a, x13p18, self.twiddles_re[2]);
                let m0526a = vfma_n_f32(m0526a, x14p17, self.twiddles_re[7]);
                let m0526a = vfma_n_f32(m0526a, x15p16, self.twiddles_re[12]);
                let m0526b = vmul_n_f32(x1m30, self.twiddles_im[4]);
                let m0526b = vfma_n_f32(m0526b, x2m29, self.twiddles_im[9]);
                let m0526b = vfma_n_f32(m0526b, x3m28, self.twiddles_im[14]);
                let m0526b = vfms_n_f32(m0526b, x4m27, self.twiddles_im[10]);
                let m0526b = vfms_n_f32(m0526b, x5m26, self.twiddles_im[5]);
                let m0526b = vfms_n_f32(m0526b, x6m25, self.twiddles_im[0]);
                let m0526b = vfma_n_f32(m0526b, x7m24, self.twiddles_im[3]);
                let m0526b = vfma_n_f32(m0526b, x8m23, self.twiddles_im[8]);
                let m0526b = vfma_n_f32(m0526b, x9m22, self.twiddles_im[13]);
                let m0526b = vfms_n_f32(m0526b, x10m21, self.twiddles_im[11]);
                let m0526b = vfms_n_f32(m0526b, x11m20, self.twiddles_im[6]);
                let m0526b = vfms_n_f32(m0526b, x12m19, self.twiddles_im[1]);
                let m0526b = vfma_n_f32(m0526b, x13m18, self.twiddles_im[2]);
                let m0526b = vfma_n_f32(m0526b, x14m17, self.twiddles_im[7]);
                let m0526b = vfma_n_f32(m0526b, x15m16, self.twiddles_im[12]);
                let (y05, y26) = NeonButterfly::butterfly2h_f32(m0526a, m0526b);

                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y04, y05),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    vcombine_f32(y26, y27),
                );

                let m0625a = vfma_n_f32(u0, x1p30, self.twiddles_re[5]);
                let m0625a = vfma_n_f32(m0625a, x2p29, self.twiddles_re[11]);
                let m0625a = vfma_n_f32(m0625a, x3p28, self.twiddles_re[12]);
                let m0625a = vfma_n_f32(m0625a, x4p27, self.twiddles_re[6]);
                let m0625a = vfma_n_f32(m0625a, x5p26, self.twiddles_re[0]);
                let m0625a = vfma_n_f32(m0625a, x6p25, self.twiddles_re[4]);
                let m0625a = vfma_n_f32(m0625a, x7p24, self.twiddles_re[10]);
                let m0625a = vfma_n_f32(m0625a, x8p23, self.twiddles_re[13]);
                let m0625a = vfma_n_f32(m0625a, x9p22, self.twiddles_re[7]);
                let m0625a = vfma_n_f32(m0625a, x10p21, self.twiddles_re[1]);
                let m0625a = vfma_n_f32(m0625a, x11p20, self.twiddles_re[3]);
                let m0625a = vfma_n_f32(m0625a, x12p19, self.twiddles_re[9]);
                let m0625a = vfma_n_f32(m0625a, x13p18, self.twiddles_re[14]);
                let m0625a = vfma_n_f32(m0625a, x14p17, self.twiddles_re[8]);
                let m0625a = vfma_n_f32(m0625a, x15p16, self.twiddles_re[2]);
                let m0625b = vmul_n_f32(x1m30, self.twiddles_im[5]);
                let m0625b = vfma_n_f32(m0625b, x2m29, self.twiddles_im[11]);
                let m0625b = vfms_n_f32(m0625b, x3m28, self.twiddles_im[12]);
                let m0625b = vfms_n_f32(m0625b, x4m27, self.twiddles_im[6]);
                let m0625b = vfms_n_f32(m0625b, x5m26, self.twiddles_im[0]);
                let m0625b = vfma_n_f32(m0625b, x6m25, self.twiddles_im[4]);
                let m0625b = vfma_n_f32(m0625b, x7m24, self.twiddles_im[10]);
                let m0625b = vfms_n_f32(m0625b, x8m23, self.twiddles_im[13]);
                let m0625b = vfms_n_f32(m0625b, x9m22, self.twiddles_im[7]);
                let m0625b = vfms_n_f32(m0625b, x10m21, self.twiddles_im[1]);
                let m0625b = vfma_n_f32(m0625b, x11m20, self.twiddles_im[3]);
                let m0625b = vfma_n_f32(m0625b, x12m19, self.twiddles_im[9]);
                let m0625b = vfms_n_f32(m0625b, x13m18, self.twiddles_im[14]);
                let m0625b = vfms_n_f32(m0625b, x14m17, self.twiddles_im[8]);
                let m0625b = vfms_n_f32(m0625b, x15m16, self.twiddles_im[2]);
                let (y06, y25) = NeonButterfly::butterfly2h_f32(m0625a, m0625b);

                let m0724a = vfma_n_f32(u0, x1p30, self.twiddles_re[6]);
                let m0724a = vfma_n_f32(m0724a, x2p29, self.twiddles_re[13]);
                let m0724a = vfma_n_f32(m0724a, x3p28, self.twiddles_re[9]);
                let m0724a = vfma_n_f32(m0724a, x4p27, self.twiddles_re[2]);
                let m0724a = vfma_n_f32(m0724a, x5p26, self.twiddles_re[3]);
                let m0724a = vfma_n_f32(m0724a, x6p25, self.twiddles_re[10]);
                let m0724a = vfma_n_f32(m0724a, x7p24, self.twiddles_re[12]);
                let m0724a = vfma_n_f32(m0724a, x8p23, self.twiddles_re[5]);
                let m0724a = vfma_n_f32(m0724a, x9p22, self.twiddles_re[0]);
                let m0724a = vfma_n_f32(m0724a, x10p21, self.twiddles_re[7]);
                let m0724a = vfma_n_f32(m0724a, x11p20, self.twiddles_re[14]);
                let m0724a = vfma_n_f32(m0724a, x12p19, self.twiddles_re[8]);
                let m0724a = vfma_n_f32(m0724a, x13p18, self.twiddles_re[1]);
                let m0724a = vfma_n_f32(m0724a, x14p17, self.twiddles_re[4]);
                let m0724a = vfma_n_f32(m0724a, x15p16, self.twiddles_re[11]);
                let m0724b = vmul_n_f32(x1m30, self.twiddles_im[6]);
                let m0724b = vfma_n_f32(m0724b, x2m29, self.twiddles_im[13]);
                let m0724b = vfms_n_f32(m0724b, x3m28, self.twiddles_im[9]);
                let m0724b = vfms_n_f32(m0724b, x4m27, self.twiddles_im[2]);
                let m0724b = vfma_n_f32(m0724b, x5m26, self.twiddles_im[3]);
                let m0724b = vfma_n_f32(m0724b, x6m25, self.twiddles_im[10]);
                let m0724b = vfms_n_f32(m0724b, x7m24, self.twiddles_im[12]);
                let m0724b = vfms_n_f32(m0724b, x8m23, self.twiddles_im[5]);
                let m0724b = vfma_n_f32(m0724b, x9m22, self.twiddles_im[0]);
                let m0724b = vfma_n_f32(m0724b, x10m21, self.twiddles_im[7]);
                let m0724b = vfma_n_f32(m0724b, x11m20, self.twiddles_im[14]);
                let m0724b = vfms_n_f32(m0724b, x12m19, self.twiddles_im[8]);
                let m0724b = vfms_n_f32(m0724b, x13m18, self.twiddles_im[1]);
                let m0724b = vfma_n_f32(m0724b, x14m17, self.twiddles_im[4]);
                let m0724b = vfma_n_f32(m0724b, x15m16, self.twiddles_im[11]);
                let (y07, y24) = NeonButterfly::butterfly2h_f32(m0724a, m0724b);

                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y06, y07),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    vcombine_f32(y24, y25),
                );

                let m0823a = vfma_n_f32(u0, x1p30, self.twiddles_re[7]);
                let m0823a = vfma_n_f32(m0823a, x2p29, self.twiddles_re[14]);
                let m0823a = vfma_n_f32(m0823a, x3p28, self.twiddles_re[6]);
                let m0823a = vfma_n_f32(m0823a, x4p27, self.twiddles_re[0]);
                let m0823a = vfma_n_f32(m0823a, x5p26, self.twiddles_re[8]);
                let m0823a = vfma_n_f32(m0823a, x6p25, self.twiddles_re[13]);
                let m0823a = vfma_n_f32(m0823a, x7p24, self.twiddles_re[5]);
                let m0823a = vfma_n_f32(m0823a, x8p23, self.twiddles_re[1]);
                let m0823a = vfma_n_f32(m0823a, x9p22, self.twiddles_re[9]);
                let m0823a = vfma_n_f32(m0823a, x10p21, self.twiddles_re[12]);
                let m0823a = vfma_n_f32(m0823a, x11p20, self.twiddles_re[4]);
                let m0823a = vfma_n_f32(m0823a, x12p19, self.twiddles_re[2]);
                let m0823a = vfma_n_f32(m0823a, x13p18, self.twiddles_re[10]);
                let m0823a = vfma_n_f32(m0823a, x14p17, self.twiddles_re[11]);
                let m0823a = vfma_n_f32(m0823a, x15p16, self.twiddles_re[3]);
                let m0823b = vmul_n_f32(x1m30, self.twiddles_im[7]);
                let m0823b = vfms_n_f32(m0823b, x2m29, self.twiddles_im[14]);
                let m0823b = vfms_n_f32(m0823b, x3m28, self.twiddles_im[6]);
                let m0823b = vfma_n_f32(m0823b, x4m27, self.twiddles_im[0]);
                let m0823b = vfma_n_f32(m0823b, x5m26, self.twiddles_im[8]);
                let m0823b = vfms_n_f32(m0823b, x6m25, self.twiddles_im[13]);
                let m0823b = vfms_n_f32(m0823b, x7m24, self.twiddles_im[5]);
                let m0823b = vfma_n_f32(m0823b, x8m23, self.twiddles_im[1]);
                let m0823b = vfma_n_f32(m0823b, x9m22, self.twiddles_im[9]);
                let m0823b = vfms_n_f32(m0823b, x10m21, self.twiddles_im[12]);
                let m0823b = vfms_n_f32(m0823b, x11m20, self.twiddles_im[4]);
                let m0823b = vfma_n_f32(m0823b, x12m19, self.twiddles_im[2]);
                let m0823b = vfma_n_f32(m0823b, x13m18, self.twiddles_im[10]);
                let m0823b = vfms_n_f32(m0823b, x14m17, self.twiddles_im[11]);
                let m0823b = vfms_n_f32(m0823b, x15m16, self.twiddles_im[3]);
                let (y08, y23) = NeonButterfly::butterfly2h_f32(m0823a, m0823b);

                let m0922a = vfma_n_f32(u0, x1p30, self.twiddles_re[8]);
                let m0922a = vfma_n_f32(m0922a, x2p29, self.twiddles_re[12]);
                let m0922a = vfma_n_f32(m0922a, x3p28, self.twiddles_re[3]);
                let m0922a = vfma_n_f32(m0922a, x4p27, self.twiddles_re[4]);
                let m0922a = vfma_n_f32(m0922a, x5p26, self.twiddles_re[13]);
                let m0922a = vfma_n_f32(m0922a, x6p25, self.twiddles_re[7]);
                let m0922a = vfma_n_f32(m0922a, x7p24, self.twiddles_re[0]);
                let m0922a = vfma_n_f32(m0922a, x8p23, self.twiddles_re[9]);
                let m0922a = vfma_n_f32(m0922a, x9p22, self.twiddles_re[11]);
                let m0922a = vfma_n_f32(m0922a, x10p21, self.twiddles_re[2]);
                let m0922a = vfma_n_f32(m0922a, x11p20, self.twiddles_re[5]);
                let m0922a = vfma_n_f32(m0922a, x12p19, self.twiddles_re[14]);
                let m0922a = vfma_n_f32(m0922a, x13p18, self.twiddles_re[6]);
                let m0922a = vfma_n_f32(m0922a, x14p17, self.twiddles_re[1]);
                let m0922a = vfma_n_f32(m0922a, x15p16, self.twiddles_re[10]);
                let m0922b = vmul_n_f32(x1m30, self.twiddles_im[8]);
                let m0922b = vfms_n_f32(m0922b, x2m29, self.twiddles_im[12]);
                let m0922b = vfms_n_f32(m0922b, x3m28, self.twiddles_im[3]);
                let m0922b = vfma_n_f32(m0922b, x4m27, self.twiddles_im[4]);
                let m0922b = vfma_n_f32(m0922b, x5m26, self.twiddles_im[13]);
                let m0922b = vfms_n_f32(m0922b, x6m25, self.twiddles_im[7]);
                let m0922b = vfma_n_f32(m0922b, x7m24, self.twiddles_im[0]);
                let m0922b = vfma_n_f32(m0922b, x8m23, self.twiddles_im[9]);
                let m0922b = vfms_n_f32(m0922b, x9m22, self.twiddles_im[11]);
                let m0922b = vfms_n_f32(m0922b, x10m21, self.twiddles_im[2]);
                let m0922b = vfma_n_f32(m0922b, x11m20, self.twiddles_im[5]);
                let m0922b = vfma_n_f32(m0922b, x12m19, self.twiddles_im[14]);
                let m0922b = vfms_n_f32(m0922b, x13m18, self.twiddles_im[6]);
                let m0922b = vfma_n_f32(m0922b, x14m17, self.twiddles_im[1]);
                let m0922b = vfma_n_f32(m0922b, x15m16, self.twiddles_im[10]);
                let (y09, y22) = NeonButterfly::butterfly2h_f32(m0922a, m0922b);

                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y08, y09),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    vcombine_f32(y22, y23),
                );

                let m1021a = vfma_n_f32(u0, x1p30, self.twiddles_re[9]);
                let m1021a = vfma_n_f32(m1021a, x2p29, self.twiddles_re[10]);
                let m1021a = vfma_n_f32(m1021a, x3p28, self.twiddles_re[0]);
                let m1021a = vfma_n_f32(m1021a, x4p27, self.twiddles_re[8]);
                let m1021a = vfma_n_f32(m1021a, x5p26, self.twiddles_re[11]);
                let m1021a = vfma_n_f32(m1021a, x6p25, self.twiddles_re[1]);
                let m1021a = vfma_n_f32(m1021a, x7p24, self.twiddles_re[7]);
                let m1021a = vfma_n_f32(m1021a, x8p23, self.twiddles_re[12]);
                let m1021a = vfma_n_f32(m1021a, x9p22, self.twiddles_re[2]);
                let m1021a = vfma_n_f32(m1021a, x10p21, self.twiddles_re[6]);
                let m1021a = vfma_n_f32(m1021a, x11p20, self.twiddles_re[13]);
                let m1021a = vfma_n_f32(m1021a, x12p19, self.twiddles_re[3]);
                let m1021a = vfma_n_f32(m1021a, x13p18, self.twiddles_re[5]);
                let m1021a = vfma_n_f32(m1021a, x14p17, self.twiddles_re[14]);
                let m1021a = vfma_n_f32(m1021a, x15p16, self.twiddles_re[4]);
                let m1021b = vmul_n_f32(x1m30, self.twiddles_im[9]);
                let m1021b = vfms_n_f32(m1021b, x2m29, self.twiddles_im[10]);
                let m1021b = vfms_n_f32(m1021b, x3m28, self.twiddles_im[0]);
                let m1021b = vfma_n_f32(m1021b, x4m27, self.twiddles_im[8]);
                let m1021b = vfms_n_f32(m1021b, x5m26, self.twiddles_im[11]);
                let m1021b = vfms_n_f32(m1021b, x6m25, self.twiddles_im[1]);
                let m1021b = vfma_n_f32(m1021b, x7m24, self.twiddles_im[7]);
                let m1021b = vfms_n_f32(m1021b, x8m23, self.twiddles_im[12]);
                let m1021b = vfms_n_f32(m1021b, x9m22, self.twiddles_im[2]);
                let m1021b = vfma_n_f32(m1021b, x10m21, self.twiddles_im[6]);
                let m1021b = vfms_n_f32(m1021b, x11m20, self.twiddles_im[13]);
                let m1021b = vfms_n_f32(m1021b, x12m19, self.twiddles_im[3]);
                let m1021b = vfma_n_f32(m1021b, x13m18, self.twiddles_im[5]);
                let m1021b = vfms_n_f32(m1021b, x14m17, self.twiddles_im[14]);
                let m1021b = vfms_n_f32(m1021b, x15m16, self.twiddles_im[4]);
                let (y10, y21) = NeonButterfly::butterfly2h_f32(m1021a, m1021b);

                let m1120a = vfma_n_f32(u0, x1p30, self.twiddles_re[10]);
                let m1120a = vfma_n_f32(m1120a, x2p29, self.twiddles_re[8]);
                let m1120a = vfma_n_f32(m1120a, x3p28, self.twiddles_re[1]);
                let m1120a = vfma_n_f32(m1120a, x4p27, self.twiddles_re[12]);
                let m1120a = vfma_n_f32(m1120a, x5p26, self.twiddles_re[6]);
                let m1120a = vfma_n_f32(m1120a, x6p25, self.twiddles_re[3]);
                let m1120a = vfma_n_f32(m1120a, x7p24, self.twiddles_re[14]);
                let m1120a = vfma_n_f32(m1120a, x8p23, self.twiddles_re[4]);
                let m1120a = vfma_n_f32(m1120a, x9p22, self.twiddles_re[5]);
                let m1120a = vfma_n_f32(m1120a, x10p21, self.twiddles_re[13]);
                let m1120a = vfma_n_f32(m1120a, x11p20, self.twiddles_re[2]);
                let m1120a = vfma_n_f32(m1120a, x12p19, self.twiddles_re[7]);
                let m1120a = vfma_n_f32(m1120a, x13p18, self.twiddles_re[11]);
                let m1120a = vfma_n_f32(m1120a, x14p17, self.twiddles_re[0]);
                let m1120a = vfma_n_f32(m1120a, x15p16, self.twiddles_re[9]);
                let m1120b = vmul_n_f32(x1m30, self.twiddles_im[10]);
                let m1120b = vfms_n_f32(m1120b, x2m29, self.twiddles_im[8]);
                let m1120b = vfma_n_f32(m1120b, x3m28, self.twiddles_im[1]);
                let m1120b = vfma_n_f32(m1120b, x4m27, self.twiddles_im[12]);
                let m1120b = vfms_n_f32(m1120b, x5m26, self.twiddles_im[6]);
                let m1120b = vfma_n_f32(m1120b, x6m25, self.twiddles_im[3]);
                let m1120b = vfma_n_f32(m1120b, x7m24, self.twiddles_im[14]);
                let m1120b = vfms_n_f32(m1120b, x8m23, self.twiddles_im[4]);
                let m1120b = vfma_n_f32(m1120b, x9m22, self.twiddles_im[5]);
                let m1120b = vfms_n_f32(m1120b, x10m21, self.twiddles_im[13]);
                let m1120b = vfms_n_f32(m1120b, x11m20, self.twiddles_im[2]);
                let m1120b = vfma_n_f32(m1120b, x12m19, self.twiddles_im[7]);
                let m1120b = vfms_n_f32(m1120b, x13m18, self.twiddles_im[11]);
                let m1120b = vfms_n_f32(m1120b, x14m17, self.twiddles_im[0]);
                let m1120b = vfma_n_f32(m1120b, x15m16, self.twiddles_im[9]);
                let (y11, y20) = NeonButterfly::butterfly2h_f32(m1120a, m1120b);

                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(y10, y11),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vcombine_f32(y20, y21),
                );

                let m1219a = vfma_n_f32(u0, x1p30, self.twiddles_re[11]);
                let m1219a = vfma_n_f32(m1219a, x2p29, self.twiddles_re[6]);
                let m1219a = vfma_n_f32(m1219a, x3p28, self.twiddles_re[4]);
                let m1219a = vfma_n_f32(m1219a, x4p27, self.twiddles_re[13]);
                let m1219a = vfma_n_f32(m1219a, x5p26, self.twiddles_re[1]);
                let m1219a = vfma_n_f32(m1219a, x6p25, self.twiddles_re[9]);
                let m1219a = vfma_n_f32(m1219a, x7p24, self.twiddles_re[8]);
                let m1219a = vfma_n_f32(m1219a, x8p23, self.twiddles_re[2]);
                let m1219a = vfma_n_f32(m1219a, x9p22, self.twiddles_re[14]);
                let m1219a = vfma_n_f32(m1219a, x10p21, self.twiddles_re[3]);
                let m1219a = vfma_n_f32(m1219a, x11p20, self.twiddles_re[7]);
                let m1219a = vfma_n_f32(m1219a, x12p19, self.twiddles_re[10]);
                let m1219a = vfma_n_f32(m1219a, x13p18, self.twiddles_re[0]);
                let m1219a = vfma_n_f32(m1219a, x14p17, self.twiddles_re[12]);
                let m1219a = vfma_n_f32(m1219a, x15p16, self.twiddles_re[5]);
                let m1219b = vmul_n_f32(x1m30, self.twiddles_im[11]);
                let m1219b = vfms_n_f32(m1219b, x2m29, self.twiddles_im[6]);
                let m1219b = vfma_n_f32(m1219b, x3m28, self.twiddles_im[4]);
                let m1219b = vfms_n_f32(m1219b, x4m27, self.twiddles_im[13]);
                let m1219b = vfms_n_f32(m1219b, x5m26, self.twiddles_im[1]);
                let m1219b = vfma_n_f32(m1219b, x6m25, self.twiddles_im[9]);
                let m1219b = vfms_n_f32(m1219b, x7m24, self.twiddles_im[8]);
                let m1219b = vfma_n_f32(m1219b, x8m23, self.twiddles_im[2]);
                let m1219b = vfma_n_f32(m1219b, x9m22, self.twiddles_im[14]);
                let m1219b = vfms_n_f32(m1219b, x10m21, self.twiddles_im[3]);
                let m1219b = vfma_n_f32(m1219b, x11m20, self.twiddles_im[7]);
                let m1219b = vfms_n_f32(m1219b, x12m19, self.twiddles_im[10]);
                let m1219b = vfma_n_f32(m1219b, x13m18, self.twiddles_im[0]);
                let m1219b = vfma_n_f32(m1219b, x14m17, self.twiddles_im[12]);
                let m1219b = vfms_n_f32(m1219b, x15m16, self.twiddles_im[5]);
                let (y12, y19) = NeonButterfly::butterfly2h_f32(m1219a, m1219b);

                let m1318a = vfma_n_f32(u0, x1p30, self.twiddles_re[12]);
                let m1318a = vfma_n_f32(m1318a, x2p29, self.twiddles_re[4]);
                let m1318a = vfma_n_f32(m1318a, x3p28, self.twiddles_re[7]);
                let m1318a = vfma_n_f32(m1318a, x4p27, self.twiddles_re[9]);
                let m1318a = vfma_n_f32(m1318a, x5p26, self.twiddles_re[2]);
                let m1318a = vfma_n_f32(m1318a, x6p25, self.twiddles_re[14]);
                let m1318a = vfma_n_f32(m1318a, x7p24, self.twiddles_re[1]);
                let m1318a = vfma_n_f32(m1318a, x8p23, self.twiddles_re[10]);
                let m1318a = vfma_n_f32(m1318a, x9p22, self.twiddles_re[6]);
                let m1318a = vfma_n_f32(m1318a, x10p21, self.twiddles_re[5]);
                let m1318a = vfma_n_f32(m1318a, x11p20, self.twiddles_re[11]);
                let m1318a = vfma_n_f32(m1318a, x12p19, self.twiddles_re[0]);
                let m1318a = vfma_n_f32(m1318a, x13p18, self.twiddles_re[13]);
                let m1318a = vfma_n_f32(m1318a, x14p17, self.twiddles_re[3]);
                let m1318a = vfma_n_f32(m1318a, x15p16, self.twiddles_re[8]);
                let m1318b = vmul_n_f32(x1m30, self.twiddles_im[12]);
                let m1318b = vfms_n_f32(m1318b, x2m29, self.twiddles_im[4]);
                let m1318b = vfma_n_f32(m1318b, x3m28, self.twiddles_im[7]);
                let m1318b = vfms_n_f32(m1318b, x4m27, self.twiddles_im[9]);
                let m1318b = vfma_n_f32(m1318b, x5m26, self.twiddles_im[2]);
                let m1318b = vfms_n_f32(m1318b, x6m25, self.twiddles_im[14]);
                let m1318b = vfms_n_f32(m1318b, x7m24, self.twiddles_im[1]);
                let m1318b = vfma_n_f32(m1318b, x8m23, self.twiddles_im[10]);
                let m1318b = vfms_n_f32(m1318b, x9m22, self.twiddles_im[6]);
                let m1318b = vfma_n_f32(m1318b, x10m21, self.twiddles_im[5]);
                let m1318b = vfms_n_f32(m1318b, x11m20, self.twiddles_im[11]);
                let m1318b = vfma_n_f32(m1318b, x12m19, self.twiddles_im[0]);
                let m1318b = vfma_n_f32(m1318b, x13m18, self.twiddles_im[13]);
                let m1318b = vfms_n_f32(m1318b, x14m17, self.twiddles_im[3]);
                let m1318b = vfma_n_f32(m1318b, x15m16, self.twiddles_im[8]);
                let (y13, y18) = NeonButterfly::butterfly2h_f32(m1318a, m1318b);

                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vcombine_f32(y12, y13),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vcombine_f32(y18, y19),
                );

                let m1417a = vfma_n_f32(u0, x1p30, self.twiddles_re[13]);
                let m1417a = vfma_n_f32(m1417a, x2p29, self.twiddles_re[2]);
                let m1417a = vfma_n_f32(m1417a, x3p28, self.twiddles_re[10]);
                let m1417a = vfma_n_f32(m1417a, x4p27, self.twiddles_re[5]);
                let m1417a = vfma_n_f32(m1417a, x5p26, self.twiddles_re[7]);
                let m1417a = vfma_n_f32(m1417a, x6p25, self.twiddles_re[8]);
                let m1417a = vfma_n_f32(m1417a, x7p24, self.twiddles_re[4]);
                let m1417a = vfma_n_f32(m1417a, x8p23, self.twiddles_re[11]);
                let m1417a = vfma_n_f32(m1417a, x9p22, self.twiddles_re[1]);
                let m1417a = vfma_n_f32(m1417a, x10p21, self.twiddles_re[14]);
                let m1417a = vfma_n_f32(m1417a, x11p20, self.twiddles_re[0]);
                let m1417a = vfma_n_f32(m1417a, x12p19, self.twiddles_re[12]);
                let m1417a = vfma_n_f32(m1417a, x13p18, self.twiddles_re[3]);
                let m1417a = vfma_n_f32(m1417a, x14p17, self.twiddles_re[9]);
                let m1417a = vfma_n_f32(m1417a, x15p16, self.twiddles_re[6]);
                let m1417b = vmul_n_f32(x1m30, self.twiddles_im[13]);
                let m1417b = vfms_n_f32(m1417b, x2m29, self.twiddles_im[2]);
                let m1417b = vfma_n_f32(m1417b, x3m28, self.twiddles_im[10]);
                let m1417b = vfms_n_f32(m1417b, x4m27, self.twiddles_im[5]);
                let m1417b = vfma_n_f32(m1417b, x5m26, self.twiddles_im[7]);
                let m1417b = vfms_n_f32(m1417b, x6m25, self.twiddles_im[8]);
                let m1417b = vfma_n_f32(m1417b, x7m24, self.twiddles_im[4]);
                let m1417b = vfms_n_f32(m1417b, x8m23, self.twiddles_im[11]);
                let m1417b = vfma_n_f32(m1417b, x9m22, self.twiddles_im[1]);
                let m1417b = vfms_n_f32(m1417b, x10m21, self.twiddles_im[14]);
                let m1417b = vfms_n_f32(m1417b, x11m20, self.twiddles_im[0]);
                let m1417b = vfma_n_f32(m1417b, x12m19, self.twiddles_im[12]);
                let m1417b = vfms_n_f32(m1417b, x13m18, self.twiddles_im[3]);
                let m1417b = vfma_n_f32(m1417b, x14m17, self.twiddles_im[9]);
                let m1417b = vfms_n_f32(m1417b, x15m16, self.twiddles_im[6]);
                let (y14, y17) = NeonButterfly::butterfly2h_f32(m1417a, m1417b);

                let m1516a = vfma_n_f32(u0, x1p30, self.twiddles_re[14]);
                let m1516a = vfma_n_f32(m1516a, x2p29, self.twiddles_re[0]);
                let m1516a = vfma_n_f32(m1516a, x3p28, self.twiddles_re[13]);
                let m1516a = vfma_n_f32(m1516a, x4p27, self.twiddles_re[1]);
                let m1516a = vfma_n_f32(m1516a, x5p26, self.twiddles_re[12]);
                let m1516a = vfma_n_f32(m1516a, x6p25, self.twiddles_re[2]);
                let m1516a = vfma_n_f32(m1516a, x7p24, self.twiddles_re[11]);
                let m1516a = vfma_n_f32(m1516a, x8p23, self.twiddles_re[3]);
                let m1516a = vfma_n_f32(m1516a, x9p22, self.twiddles_re[10]);
                let m1516a = vfma_n_f32(m1516a, x10p21, self.twiddles_re[4]);
                let m1516a = vfma_n_f32(m1516a, x11p20, self.twiddles_re[9]);
                let m1516a = vfma_n_f32(m1516a, x12p19, self.twiddles_re[5]);
                let m1516a = vfma_n_f32(m1516a, x13p18, self.twiddles_re[8]);
                let m1516a = vfma_n_f32(m1516a, x14p17, self.twiddles_re[6]);
                let m1516a = vfma_n_f32(m1516a, x15p16, self.twiddles_re[7]);
                let m1516b = vmul_n_f32(x1m30, self.twiddles_im[14]);
                let m1516b = vfms_n_f32(m1516b, x2m29, self.twiddles_im[0]);
                let m1516b = vfma_n_f32(m1516b, x3m28, self.twiddles_im[13]);
                let m1516b = vfms_n_f32(m1516b, x4m27, self.twiddles_im[1]);
                let m1516b = vfma_n_f32(m1516b, x5m26, self.twiddles_im[12]);
                let m1516b = vfms_n_f32(m1516b, x6m25, self.twiddles_im[2]);
                let m1516b = vfma_n_f32(m1516b, x7m24, self.twiddles_im[11]);
                let m1516b = vfms_n_f32(m1516b, x8m23, self.twiddles_im[3]);
                let m1516b = vfma_n_f32(m1516b, x9m22, self.twiddles_im[10]);
                let m1516b = vfms_n_f32(m1516b, x10m21, self.twiddles_im[4]);
                let m1516b = vfma_n_f32(m1516b, x11m20, self.twiddles_im[9]);
                let m1516b = vfms_n_f32(m1516b, x12m19, self.twiddles_im[5]);
                let m1516b = vfma_n_f32(m1516b, x13m18, self.twiddles_im[8]);
                let m1516b = vfms_n_f32(m1516b, x14m17, self.twiddles_im[6]);
                let m1516b = vfma_n_f32(m1516b, x15m16, self.twiddles_im[7]);
                let (y15, y16) = NeonButterfly::butterfly2h_f32(m1516a, m1516b);

                vst1q_f32(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vcombine_f32(y14, y15),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vcombine_f32(y16, y17),
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
        31
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;

    test_butterfly!(test_neon_butterfly31, f32, NeonButterfly31f, 31, 1e-2);
}
