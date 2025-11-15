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
use crate::FftDirection;
use crate::neon::util::vfcmulq_f32;
use crate::util::compute_twiddle;
use std::arch::aarch64::*;

pub(crate) struct NeonFastButterfly9f {
    tw3_re: f32,
    tw3_im: float32x4_t,
    twiddle1: float32x4_t,
    twiddle2: float32x4_t,
    twiddle4: float32x4_t,
}

impl NeonFastButterfly9f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle3 = compute_twiddle::<f32>(1, 3, fft_direction);
        let tw1 = compute_twiddle::<f32>(1, 9, fft_direction);
        let tw2 = compute_twiddle::<f32>(2, 9, fft_direction);
        let tw4 = compute_twiddle::<f32>(4, 9, fft_direction);
        unsafe {
            NeonFastButterfly9f {
                tw3_re: twiddle3.re,
                tw3_im: vld1q_f32(
                    [-twiddle3.im, twiddle3.im, -twiddle3.im, twiddle3.im]
                        .as_ptr()
                        .cast(),
                ),
                twiddle1: vld1q_f32([tw1.re, tw1.im, tw1.re, tw1.im].as_ptr().cast()),
                twiddle2: vld1q_f32([tw2.re, tw2.im, tw2.re, tw2.im].as_ptr().cast()),
                twiddle4: vld1q_f32([tw4.re, tw4.im, tw4.re, tw4.im].as_ptr().cast()),
            }
        }
    }
}

impl NeonFastButterfly9f {
    #[inline(always)]
    pub(crate) fn bf3(
        &self,
        u0: float32x4_t,
        u1: float32x4_t,
        u2: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t) {
        unsafe {
            let xp = vaddq_f32(u1, u2);
            let xn = vsubq_f32(u1, u2);
            let sum = vaddq_f32(u0, xp);

            let w_1 = vfmaq_n_f32(u0, xp, self.tw3_re);
            let xn_rot = vrev64q_f32(xn);

            let y0 = sum;
            let y1 = vfmaq_f32(w_1, self.tw3_im, xn_rot);
            let y2 = vfmsq_f32(w_1, self.tw3_im, xn_rot);
            (y0, y1, y2)
        }
    }

    #[inline(always)]
    pub(crate) fn bf3h(
        &self,
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
    ) -> (float32x2_t, float32x2_t, float32x2_t) {
        unsafe {
            let xp = vadd_f32(u1, u2);
            let xn = vsub_f32(u1, u2);
            let sum = vadd_f32(u0, xp);

            let w_1 = vfma_n_f32(u0, xp, self.tw3_re);
            let xn_rot = vext_f32::<1>(xn, xn);

            let y0 = sum;
            let y1 = vfma_f32(w_1, vget_low_f32(self.tw3_im), xn_rot);
            let y2 = vfms_f32(w_1, vget_low_f32(self.tw3_im), xn_rot);
            (y0, y1, y2)
        }
    }

    #[inline(always)]
    pub(crate) fn exec(
        &self,
        u0: float32x4_t,
        u1: float32x4_t,
        u2: float32x4_t,
        u3: float32x4_t,
        u4: float32x4_t,
        u5: float32x4_t,
        u6: float32x4_t,
        u7: float32x4_t,
        u8: float32x4_t,
    ) -> (
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
    ) {
        let (u0, u3, u6) = self.bf3(u0, u3, u6);
        let (u1, mut u4, mut u7) = self.bf3(u1, u4, u7);
        let (u2, mut u5, mut u8) = self.bf3(u2, u5, u8);

        u4 = vfcmulq_f32(u4, self.twiddle1);
        u7 = vfcmulq_f32(u7, self.twiddle2);
        u5 = vfcmulq_f32(u5, self.twiddle2);
        u8 = vfcmulq_f32(u8, self.twiddle4);

        let (y0, y3, y6) = self.bf3(u0, u1, u2);
        let (y1, y4, y7) = self.bf3(u3, u4, u5);
        let (y2, y5, y8) = self.bf3(u6, u7, u8);
        (y0, y1, y2, y3, y4, y5, y6, y7, y8)
    }

    #[inline(always)]
    pub(crate) fn exech(
        &self,
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
        u3: float32x2_t,
        u4: float32x2_t,
        u5: float32x2_t,
        u6: float32x2_t,
        u7: float32x2_t,
        u8: float32x2_t,
    ) -> (
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
    ) {
        unsafe {
            let (u0, u3, u6) = self.bf3h(u0, u3, u6);
            let (u1, mut u4, mut u7) = self.bf3h(u1, u4, u7);
            let (u2, mut u5, mut u8) = self.bf3h(u2, u5, u8);

            let u4u7 = vfcmulq_f32(
                vcombine_f32(u4, u7),
                vcombine_f32(vget_low_f32(self.twiddle1), vget_low_f32(self.twiddle2)),
            );
            u4 = vget_low_f32(u4u7);
            u7 = vget_high_f32(u4u7);
            let u5u8 = vfcmulq_f32(
                vcombine_f32(u5, u8),
                vcombine_f32(vget_low_f32(self.twiddle2), vget_low_f32(self.twiddle4)),
            );
            u5 = vget_low_f32(u5u8);
            u8 = vget_high_f32(u5u8);

            let (y0, y3, y6) = self.bf3h(u0, u1, u2);
            let (y1, y4, y7) = self.bf3h(u3, u4, u5);
            let (y2, y5, y8) = self.bf3h(u6, u7, u8);
            (y0, y1, y2, y3, y4, y5, y6, y7, y8)
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct NeonFcmaFastButterfly9f {
    tw3_re: f32,
    tw3_im: float32x4_t,
    n_tw3_im: float32x4_t,
    twiddle1: float32x4_t,
    twiddle2: float32x4_t,
    twiddle4: float32x4_t,
}

#[cfg(feature = "fcma")]
impl NeonFcmaFastButterfly9f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle3 = compute_twiddle::<f32>(1, 3, fft_direction);
        let tw1 = compute_twiddle::<f32>(1, 9, fft_direction);
        let tw2 = compute_twiddle::<f32>(2, 9, fft_direction);
        let tw4 = compute_twiddle::<f32>(4, 9, fft_direction);
        unsafe {
            let q = vld1q_f32(
                [-twiddle3.im, twiddle3.im, -twiddle3.im, twiddle3.im]
                    .as_ptr()
                    .cast(),
            );
            NeonFcmaFastButterfly9f {
                tw3_re: twiddle3.re,
                tw3_im: q,
                n_tw3_im: vnegq_f32(q),
                twiddle1: vld1q_f32([tw1.re, tw1.im, tw1.re, tw1.im].as_ptr().cast()),
                twiddle2: vld1q_f32([tw2.re, tw2.im, tw2.re, tw2.im].as_ptr().cast()),
                twiddle4: vld1q_f32([tw4.re, tw4.im, tw4.re, tw4.im].as_ptr().cast()),
            }
        }
    }
}

#[cfg(feature = "fcma")]
impl NeonFcmaFastButterfly9f {
    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn bf3(
        &self,
        u0: float32x4_t,
        u1: float32x4_t,
        u2: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t) {
        let xp = vaddq_f32(u1, u2);
        let xn = vsubq_f32(u1, u2);
        let sum = vaddq_f32(u0, xp);

        let w_1 = vfmaq_n_f32(u0, xp, self.tw3_re);

        let y0 = sum;
        let y1 = vcmlaq_rot90_f32(w_1, self.tw3_im, xn);
        let y2 = vcmlaq_rot90_f32(w_1, self.n_tw3_im, xn);
        (y0, y1, y2)
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn bf3h(
        &self,
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
    ) -> (float32x2_t, float32x2_t, float32x2_t) {
        let xp = vadd_f32(u1, u2);
        let xn = vsub_f32(u1, u2);
        let sum = vadd_f32(u0, xp);

        let w_1 = vfma_n_f32(u0, xp, self.tw3_re);

        let y0 = sum;
        let y1 = vcmla_rot90_f32(w_1, vget_low_f32(self.tw3_im), xn);
        let y2 = vcmla_rot90_f32(w_1, vget_low_f32(self.n_tw3_im), xn);
        (y0, y1, y2)
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(
        &self,
        u0: float32x4_t,
        u1: float32x4_t,
        u2: float32x4_t,
        u3: float32x4_t,
        u4: float32x4_t,
        u5: float32x4_t,
        u6: float32x4_t,
        u7: float32x4_t,
        u8: float32x4_t,
    ) -> (
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
    ) {
        let (u0, u3, u6) = self.bf3(u0, u3, u6);
        let (u1, mut u4, mut u7) = self.bf3(u1, u4, u7);
        let (u2, mut u5, mut u8) = self.bf3(u2, u5, u8);

        use crate::neon::util::vfcmulq_fcma_f32;

        u4 = vfcmulq_fcma_f32(u4, self.twiddle1);
        u7 = vfcmulq_fcma_f32(u7, self.twiddle2);
        u5 = vfcmulq_fcma_f32(u5, self.twiddle2);
        u8 = vfcmulq_fcma_f32(u8, self.twiddle4);

        let (y0, y3, y6) = self.bf3(u0, u1, u2);
        let (y1, y4, y7) = self.bf3(u3, u4, u5);
        let (y2, y5, y8) = self.bf3(u6, u7, u8);
        (y0, y1, y2, y3, y4, y5, y6, y7, y8)
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(
        &self,
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
        u3: float32x2_t,
        u4: float32x2_t,
        u5: float32x2_t,
        u6: float32x2_t,
        u7: float32x2_t,
        u8: float32x2_t,
    ) -> (
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
    ) {
        let (u0, u3, u6) = self.bf3h(u0, u3, u6);
        let (u1, mut u4, mut u7) = self.bf3h(u1, u4, u7);
        let (u2, mut u5, mut u8) = self.bf3h(u2, u5, u8);
        use crate::neon::util::vfcmulq_fcma_f32;
        let u4u7 = vfcmulq_fcma_f32(
            vcombine_f32(u4, u7),
            vcombine_f32(vget_low_f32(self.twiddle1), vget_low_f32(self.twiddle2)),
        );
        u4 = vget_low_f32(u4u7);
        u7 = vget_high_f32(u4u7);
        let u5u8 = vfcmulq_fcma_f32(
            vcombine_f32(u5, u8),
            vcombine_f32(vget_low_f32(self.twiddle2), vget_low_f32(self.twiddle4)),
        );
        u5 = vget_low_f32(u5u8);
        u8 = vget_high_f32(u5u8);

        let (y0, y3, y6) = self.bf3h(u0, u1, u2);
        let (y1, y4, y7) = self.bf3h(u3, u4, u5);
        let (y2, y5, y8) = self.bf3h(u6, u7, u8);
        (y0, y1, y2, y3, y4, y5, y6, y7, y8)
    }
}
