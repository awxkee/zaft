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
use crate::neon::util::mul_complex_f64;
use crate::util::compute_twiddle;
use std::arch::aarch64::*;

pub(crate) struct NeonFastButterfly9d {
    tw3_re: f64,
    tw3_im: float64x2_t,
    twiddle1: float64x2_t,
    twiddle2: float64x2_t,
    twiddle4: float64x2_t,
}

impl NeonFastButterfly9d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle3 = compute_twiddle::<f64>(1, 3, fft_direction);
        let tw1 = compute_twiddle::<f64>(1, 9, fft_direction);
        let tw2 = compute_twiddle::<f64>(2, 9, fft_direction);
        let tw4 = compute_twiddle::<f64>(4, 9, fft_direction);
        unsafe {
            NeonFastButterfly9d {
                tw3_re: twiddle3.re,
                tw3_im: vld1q_f64(
                    [-twiddle3.im, twiddle3.im, -twiddle3.im, twiddle3.im]
                        .as_ptr()
                        .cast(),
                ),
                twiddle1: vld1q_f64([tw1.re, tw1.im, tw1.re, tw1.im].as_ptr().cast()),
                twiddle2: vld1q_f64([tw2.re, tw2.im, tw2.re, tw2.im].as_ptr().cast()),
                twiddle4: vld1q_f64([tw4.re, tw4.im, tw4.re, tw4.im].as_ptr().cast()),
            }
        }
    }
}

impl NeonFastButterfly9d {
    #[inline(always)]
    pub(crate) fn bf3(
        &self,
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
    ) -> (float64x2_t, float64x2_t, float64x2_t) {
        unsafe {
            let xp = vaddq_f64(u1, u2);
            let xn = vsubq_f64(u1, u2);
            let sum = vaddq_f64(u0, xp);

            let w_1 = vfmaq_n_f64(u0, xp, self.tw3_re);
            let xn_rot = vextq_f64::<1>(xn, xn);

            let y0 = sum;
            let y1 = vfmaq_f64(w_1, self.tw3_im, xn_rot);
            let y2 = vfmsq_f64(w_1, self.tw3_im, xn_rot);
            (y0, y1, y2)
        }
    }

    #[inline(always)]
    pub(crate) fn exec(
        &self,
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        u3: float64x2_t,
        u4: float64x2_t,
        u5: float64x2_t,
        u6: float64x2_t,
        u7: float64x2_t,
        u8: float64x2_t,
    ) -> (
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
    ) {
        unsafe {
            let (u0, u3, u6) = self.bf3(u0, u3, u6);
            let (u1, mut u4, mut u7) = self.bf3(u1, u4, u7);
            let (u2, mut u5, mut u8) = self.bf3(u2, u5, u8);

            u4 = mul_complex_f64(u4, self.twiddle1);
            u7 = mul_complex_f64(u7, self.twiddle2);
            u5 = mul_complex_f64(u5, self.twiddle2);
            u8 = mul_complex_f64(u8, self.twiddle4);

            let (y0, y3, y6) = self.bf3(u0, u1, u2);
            let (y1, y4, y7) = self.bf3(u3, u4, u5);
            let (y2, y5, y8) = self.bf3(u6, u7, u8);
            (y0, y1, y2, y3, y4, y5, y6, y7, y8)
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct NeonFcmaFastButterfly9d {
    tw3_re: f64,
    tw3_im: float64x2_t,
    twiddle1: float64x2_t,
    twiddle2: float64x2_t,
    twiddle4: float64x2_t,
}

#[cfg(feature = "fcma")]
impl NeonFcmaFastButterfly9d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle3 = compute_twiddle::<f64>(1, 3, fft_direction);
        let tw1 = compute_twiddle::<f64>(1, 9, fft_direction);
        let tw2 = compute_twiddle::<f64>(2, 9, fft_direction);
        let tw4 = compute_twiddle::<f64>(4, 9, fft_direction);
        unsafe {
            NeonFcmaFastButterfly9d {
                tw3_re: twiddle3.re,
                tw3_im: vld1q_f64(
                    [-twiddle3.im, twiddle3.im, -twiddle3.im, twiddle3.im]
                        .as_ptr()
                        .cast(),
                ),
                twiddle1: vld1q_f64([tw1.re, tw1.im, tw1.re, tw1.im].as_ptr().cast()),
                twiddle2: vld1q_f64([tw2.re, tw2.im, tw2.re, tw2.im].as_ptr().cast()),
                twiddle4: vld1q_f64([tw4.re, tw4.im, tw4.re, tw4.im].as_ptr().cast()),
            }
        }
    }
}

#[cfg(feature = "fcma")]
impl NeonFcmaFastButterfly9d {
    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn bf3(
        &self,
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
    ) -> (float64x2_t, float64x2_t, float64x2_t) {
        let xp = vaddq_f64(u1, u2);
        let xn = vsubq_f64(u1, u2);
        let sum = vaddq_f64(u0, xp);

        let w_1 = vfmaq_n_f64(u0, xp, self.tw3_re);
        let xn_rot = vextq_f64::<1>(xn, xn);

        let y0 = sum;
        let y1 = vfmaq_f64(w_1, self.tw3_im, xn_rot);
        let y2 = vfmsq_f64(w_1, self.tw3_im, xn_rot);
        (y0, y1, y2)
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(
        &self,
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        u3: float64x2_t,
        u4: float64x2_t,
        u5: float64x2_t,
        u6: float64x2_t,
        u7: float64x2_t,
        u8: float64x2_t,
    ) -> (
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
    ) {
        unsafe {
            let (u0, u3, u6) = self.bf3(u0, u3, u6);
            let (u1, mut u4, mut u7) = self.bf3(u1, u4, u7);
            let (u2, mut u5, mut u8) = self.bf3(u2, u5, u8);

            use crate::neon::util::fcma_complex_f64;
            u4 = fcma_complex_f64(u4, self.twiddle1);
            u7 = fcma_complex_f64(u7, self.twiddle2);
            u5 = fcma_complex_f64(u5, self.twiddle2);
            u8 = fcma_complex_f64(u8, self.twiddle4);

            let (y0, y3, y6) = self.bf3(u0, u1, u2);
            let (y1, y4, y7) = self.bf3(u3, u4, u5);
            let (y2, y5, y8) = self.bf3(u6, u7, u8);
            (y0, y1, y2, y3, y4, y5, y6, y7, y8)
        }
    }
}
