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
use crate::neon::butterflies::NeonButterfly;
use crate::neon::util::{v_rotate90_f32, v_rotate90_f64};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

#[allow(unused)]
pub(crate) struct NeonFastButterfly7<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> NeonFastButterfly7<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        NeonFastButterfly7 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 7, fft_direction),
            twiddle2: compute_twiddle(2, 7, fft_direction),
            twiddle3: compute_twiddle(3, 7, fft_direction),
        }
    }
}

impl NeonFastButterfly7<f64> {
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
        rot: float64x2_t, // [-0.0, 0.0] - 90 rot
    ) -> (
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
    ) {
        unsafe {
            // Radix-7 butterfly

            let (x1p6, x1m6) = NeonButterfly::butterfly2_f64(u1, u6);
            let x1m6 = v_rotate90_f64(x1m6, rot);
            let y00 = vaddq_f64(u0, x1p6);
            let (x2p5, x2m5) = NeonButterfly::butterfly2_f64(u2, u5);
            let x2m5 = v_rotate90_f64(x2m5, rot);
            let y00 = vaddq_f64(y00, x2p5);
            let (x3p4, x3m4) = NeonButterfly::butterfly2_f64(u3, u4);
            let x3m4 = v_rotate90_f64(x3m4, rot);
            let y00 = vaddq_f64(y00, x3p4);

            let m0106a = vfmaq_n_f64(u0, x1p6, self.twiddle1.re);
            let m0106a = vfmaq_n_f64(m0106a, x2p5, self.twiddle2.re);
            let m0106a = vfmaq_n_f64(m0106a, x3p4, self.twiddle3.re);
            let m0106b = vmulq_n_f64(x1m6, self.twiddle1.im);
            let m0106b = vfmaq_n_f64(m0106b, x2m5, self.twiddle2.im);
            let m0106b = vfmaq_n_f64(m0106b, x3m4, self.twiddle3.im);
            let (y01, y06) = NeonButterfly::butterfly2_f64(m0106a, m0106b);

            let m0205a = vfmaq_n_f64(u0, x1p6, self.twiddle2.re);
            let m0205a = vfmaq_n_f64(m0205a, x2p5, self.twiddle3.re);
            let m0205a = vfmaq_n_f64(m0205a, x3p4, self.twiddle1.re);
            let m0205b = vmulq_n_f64(x1m6, self.twiddle2.im);
            let m0205b = vfmsq_n_f64(m0205b, x2m5, self.twiddle3.im);
            let m0205b = vfmsq_n_f64(m0205b, x3m4, self.twiddle1.im);
            let (y02, y05) = NeonButterfly::butterfly2_f64(m0205a, m0205b);

            let m0304a = vfmaq_n_f64(u0, x1p6, self.twiddle3.re);
            let m0304a = vfmaq_n_f64(m0304a, x2p5, self.twiddle1.re);
            let m0304a = vfmaq_n_f64(m0304a, x3p4, self.twiddle2.re);
            let m0304b = vmulq_n_f64(x1m6, self.twiddle3.im);
            let m0304b = vfmsq_n_f64(m0304b, x2m5, self.twiddle1.im);
            let m0304b = vfmaq_n_f64(m0304b, x3m4, self.twiddle2.im);
            let (y03, y04) = NeonButterfly::butterfly2_f64(m0304a, m0304b);
            (y00, y01, y02, y03, y04, y05, y06)
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    #[cfg(feature = "fcma")]
    pub(crate) fn exec_fcma(
        &self,
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        u3: float64x2_t,
        u4: float64x2_t,
        u5: float64x2_t,
        u6: float64x2_t,
    ) -> (
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
    ) {
        // Radix-7 butterfly

        let (x1p6, x1m6) = NeonButterfly::butterfly2_f64(u1, u6);
        let x1m6 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m6);
        let y00 = vaddq_f64(u0, x1p6);
        let (x2p5, x2m5) = NeonButterfly::butterfly2_f64(u2, u5);
        let x2m5 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m5);
        let y00 = vaddq_f64(y00, x2p5);
        let (x3p4, x3m4) = NeonButterfly::butterfly2_f64(u3, u4);
        let x3m4 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m4);
        let y00 = vaddq_f64(y00, x3p4);

        let m0106a = vfmaq_n_f64(u0, x1p6, self.twiddle1.re);
        let m0106a = vfmaq_n_f64(m0106a, x2p5, self.twiddle2.re);
        let m0106a = vfmaq_n_f64(m0106a, x3p4, self.twiddle3.re);
        let m0106b = vmulq_n_f64(x1m6, self.twiddle1.im);
        let m0106b = vfmaq_n_f64(m0106b, x2m5, self.twiddle2.im);
        let m0106b = vfmaq_n_f64(m0106b, x3m4, self.twiddle3.im);
        let (y01, y06) = NeonButterfly::butterfly2_f64(m0106a, m0106b);

        let m0205a = vfmaq_n_f64(u0, x1p6, self.twiddle2.re);
        let m0205a = vfmaq_n_f64(m0205a, x2p5, self.twiddle3.re);
        let m0205a = vfmaq_n_f64(m0205a, x3p4, self.twiddle1.re);
        let m0205b = vmulq_n_f64(x1m6, self.twiddle2.im);
        let m0205b = vfmsq_n_f64(m0205b, x2m5, self.twiddle3.im);
        let m0205b = vfmsq_n_f64(m0205b, x3m4, self.twiddle1.im);
        let (y02, y05) = NeonButterfly::butterfly2_f64(m0205a, m0205b);

        let m0304a = vfmaq_n_f64(u0, x1p6, self.twiddle3.re);
        let m0304a = vfmaq_n_f64(m0304a, x2p5, self.twiddle1.re);
        let m0304a = vfmaq_n_f64(m0304a, x3p4, self.twiddle2.re);
        let m0304b = vmulq_n_f64(x1m6, self.twiddle3.im);
        let m0304b = vfmsq_n_f64(m0304b, x2m5, self.twiddle1.im);
        let m0304b = vfmaq_n_f64(m0304b, x3m4, self.twiddle2.im);
        let (y03, y04) = NeonButterfly::butterfly2_f64(m0304a, m0304b);
        (y00, y01, y02, y03, y04, y05, y06)
    }
}
impl NeonFastButterfly7<f32> {
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
        rot: float32x4_t, // [-0.0, 0.0] - 90 rot
    ) -> (
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
    ) {
        unsafe {
            // Radix-7 butterfly

            let (x1p6, x1m6) = NeonButterfly::butterfly2_f32(u1, u6);
            let x1m6 = v_rotate90_f32(x1m6, rot);
            let y00 = vaddq_f32(u0, x1p6);
            let (x2p5, x2m5) = NeonButterfly::butterfly2_f32(u2, u5);
            let x2m5 = v_rotate90_f32(x2m5, rot);
            let y00 = vaddq_f32(y00, x2p5);
            let (x3p4, x3m4) = NeonButterfly::butterfly2_f32(u3, u4);
            let x3m4 = v_rotate90_f32(x3m4, rot);
            let y00 = vaddq_f32(y00, x3p4);

            let m0106a = vfmaq_n_f32(u0, x1p6, self.twiddle1.re);
            let m0106a = vfmaq_n_f32(m0106a, x2p5, self.twiddle2.re);
            let m0106a = vfmaq_n_f32(m0106a, x3p4, self.twiddle3.re);
            let m0106b = vmulq_n_f32(x1m6, self.twiddle1.im);
            let m0106b = vfmaq_n_f32(m0106b, x2m5, self.twiddle2.im);
            let m0106b = vfmaq_n_f32(m0106b, x3m4, self.twiddle3.im);
            let (y01, y06) = NeonButterfly::butterfly2_f32(m0106a, m0106b);

            let m0205a = vfmaq_n_f32(u0, x1p6, self.twiddle2.re);
            let m0205a = vfmaq_n_f32(m0205a, x2p5, self.twiddle3.re);
            let m0205a = vfmaq_n_f32(m0205a, x3p4, self.twiddle1.re);
            let m0205b = vmulq_n_f32(x1m6, self.twiddle2.im);
            let m0205b = vfmsq_n_f32(m0205b, x2m5, self.twiddle3.im);
            let m0205b = vfmsq_n_f32(m0205b, x3m4, self.twiddle1.im);
            let (y02, y05) = NeonButterfly::butterfly2_f32(m0205a, m0205b);

            let m0304a = vfmaq_n_f32(u0, x1p6, self.twiddle3.re);
            let m0304a = vfmaq_n_f32(m0304a, x2p5, self.twiddle1.re);
            let m0304a = vfmaq_n_f32(m0304a, x3p4, self.twiddle2.re);
            let m0304b = vmulq_n_f32(x1m6, self.twiddle3.im);
            let m0304b = vfmsq_n_f32(m0304b, x2m5, self.twiddle1.im);
            let m0304b = vfmaq_n_f32(m0304b, x3m4, self.twiddle2.im);
            let (y03, y04) = NeonButterfly::butterfly2_f32(m0304a, m0304b);

            (y00, y01, y02, y03, y04, y05, y06)
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    #[cfg(feature = "fcma")]
    pub(crate) fn exec_fcma(
        &self,
        u0: float32x4_t,
        u1: float32x4_t,
        u2: float32x4_t,
        u3: float32x4_t,
        u4: float32x4_t,
        u5: float32x4_t,
        u6: float32x4_t,
    ) -> (
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
    ) {
        // Radix-7 butterfly

        let (x1p6, x1m6) = NeonButterfly::butterfly2_f32(u1, u6);
        let x1m6 = vcaddq_rot90_f32(vdupq_n_f32(0.), x1m6);
        let y00 = vaddq_f32(u0, x1p6);
        let (x2p5, x2m5) = NeonButterfly::butterfly2_f32(u2, u5);
        let x2m5 = vcaddq_rot90_f32(vdupq_n_f32(0.), x2m5);
        let y00 = vaddq_f32(y00, x2p5);
        let (x3p4, x3m4) = NeonButterfly::butterfly2_f32(u3, u4);
        let x3m4 = vcaddq_rot90_f32(vdupq_n_f32(0.), x3m4);
        let y00 = vaddq_f32(y00, x3p4);

        let m0106a = vfmaq_n_f32(u0, x1p6, self.twiddle1.re);
        let m0106a = vfmaq_n_f32(m0106a, x2p5, self.twiddle2.re);
        let m0106a = vfmaq_n_f32(m0106a, x3p4, self.twiddle3.re);
        let m0106b = vmulq_n_f32(x1m6, self.twiddle1.im);
        let m0106b = vfmaq_n_f32(m0106b, x2m5, self.twiddle2.im);
        let m0106b = vfmaq_n_f32(m0106b, x3m4, self.twiddle3.im);
        let (y01, y06) = NeonButterfly::butterfly2_f32(m0106a, m0106b);

        let m0205a = vfmaq_n_f32(u0, x1p6, self.twiddle2.re);
        let m0205a = vfmaq_n_f32(m0205a, x2p5, self.twiddle3.re);
        let m0205a = vfmaq_n_f32(m0205a, x3p4, self.twiddle1.re);
        let m0205b = vmulq_n_f32(x1m6, self.twiddle2.im);
        let m0205b = vfmsq_n_f32(m0205b, x2m5, self.twiddle3.im);
        let m0205b = vfmsq_n_f32(m0205b, x3m4, self.twiddle1.im);
        let (y02, y05) = NeonButterfly::butterfly2_f32(m0205a, m0205b);

        let m0304a = vfmaq_n_f32(u0, x1p6, self.twiddle3.re);
        let m0304a = vfmaq_n_f32(m0304a, x2p5, self.twiddle1.re);
        let m0304a = vfmaq_n_f32(m0304a, x3p4, self.twiddle2.re);
        let m0304b = vmulq_n_f32(x1m6, self.twiddle3.im);
        let m0304b = vfmsq_n_f32(m0304b, x2m5, self.twiddle1.im);
        let m0304b = vfmaq_n_f32(m0304b, x3m4, self.twiddle2.im);
        let (y03, y04) = NeonButterfly::butterfly2_f32(m0304a, m0304b);

        (y00, y01, y02, y03, y04, y05, y06)
    }
}
