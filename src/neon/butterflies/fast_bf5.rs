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
use crate::neon::util::{v_rotate90_f32, v_rotate90_f64};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

#[allow(unused)]
pub(crate) struct NeonFastButterfly5<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> NeonFastButterfly5<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        NeonFastButterfly5 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 5, fft_direction),
            twiddle2: compute_twiddle(2, 5, fft_direction),
        }
    }
}

impl NeonFastButterfly5<f64> {
    #[inline(always)]
    pub(crate) fn exec(
        &self,
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        u3: float64x2_t,
        u4: float64x2_t,
        rot: float64x2_t, // [-0.0, 0.0] - 90 rot
    ) -> (
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
    ) {
        unsafe {
            let x14p = vaddq_f64(u1, u4);
            let x14n = vsubq_f64(u1, u4);
            let x23p = vaddq_f64(u2, u3);
            let x23n = vsubq_f64(u2, u3);
            let y0 = vaddq_f64(vaddq_f64(u0, x14p), x23p);

            let temp_b1_1 = vmulq_n_f64(x14n, self.twiddle1.im);
            let temp_b2_1 = vmulq_n_f64(x14n, self.twiddle2.im);

            let temp_a1 = vfmaq_n_f64(
                vfmaq_n_f64(u0, x14p, self.twiddle1.re),
                x23p,
                self.twiddle2.re,
            );
            let temp_a2 = vfmaq_n_f64(
                vfmaq_n_f64(u0, x14p, self.twiddle2.re),
                x23p,
                self.twiddle1.re,
            );

            let temp_b1 = vfmaq_n_f64(temp_b1_1, x23n, self.twiddle2.im);
            let temp_b2 = vfmsq_n_f64(temp_b2_1, x23n, self.twiddle1.im);

            let temp_b1_rot = v_rotate90_f64(temp_b1, rot);
            let temp_b2_rot = v_rotate90_f64(temp_b2, rot);

            let y1 = vaddq_f64(temp_a1, temp_b1_rot);
            let y2 = vaddq_f64(temp_a2, temp_b2_rot);
            let y3 = vsubq_f64(temp_a2, temp_b2_rot);
            let y4 = vsubq_f64(temp_a1, temp_b1_rot);
            (y0, y1, y2, y3, y4)
        }
    }

    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    #[inline]
    pub(crate) fn exec_fcma(
        &self,
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        u3: float64x2_t,
        u4: float64x2_t,
    ) -> (
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
    ) {
        let x14p = vaddq_f64(u1, u4);
        let x14n = vsubq_f64(u1, u4);
        let x23p = vaddq_f64(u2, u3);
        let x23n = vsubq_f64(u2, u3);
        let y0 = vaddq_f64(vaddq_f64(u0, x14p), x23p);

        let temp_b1_1 = vmulq_n_f64(x14n, self.twiddle1.im);
        let temp_b2_1 = vmulq_n_f64(x14n, self.twiddle2.im);

        let temp_a1 = vfmaq_n_f64(
            vfmaq_n_f64(u0, x14p, self.twiddle1.re),
            x23p,
            self.twiddle2.re,
        );
        let temp_a2 = vfmaq_n_f64(
            vfmaq_n_f64(u0, x14p, self.twiddle2.re),
            x23p,
            self.twiddle1.re,
        );

        let temp_b1 = vfmaq_n_f64(temp_b1_1, x23n, self.twiddle2.im);
        let temp_b2 = vfmsq_n_f64(temp_b2_1, x23n, self.twiddle1.im);

        let y1 = vcaddq_rot90_f64(temp_a1, temp_b1);
        let y2 = vcaddq_rot90_f64(temp_a2, temp_b2);
        let y3 = vcaddq_rot270_f64(temp_a2, temp_b2);
        let y4 = vcaddq_rot270_f64(temp_a1, temp_b1);
        (y0, y1, y2, y3, y4)
    }
}

impl NeonFastButterfly5<f32> {
    #[inline(always)]
    pub(crate) fn exec(
        &self,
        u0: float32x4_t,
        u1: float32x4_t,
        u2: float32x4_t,
        u3: float32x4_t,
        u4: float32x4_t,
        rot: float32x4_t, // [-0.0, 0.0] - 90 rot
    ) -> (
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
    ) {
        unsafe {
            let x14p = vaddq_f32(u1, u4);
            let x14n = vsubq_f32(u1, u4);
            let x23p = vaddq_f32(u2, u3);
            let x23n = vsubq_f32(u2, u3);
            let y0 = vaddq_f32(vaddq_f32(u0, x14p), x23p);

            let temp_b1_1 = vmulq_n_f32(x14n, self.twiddle1.im);
            let temp_b2_1 = vmulq_n_f32(x14n, self.twiddle2.im);

            let temp_a1 = vfmaq_n_f32(
                vfmaq_n_f32(u0, x14p, self.twiddle1.re),
                x23p,
                self.twiddle2.re,
            );
            let temp_a2 = vfmaq_n_f32(
                vfmaq_n_f32(u0, x14p, self.twiddle2.re),
                x23p,
                self.twiddle1.re,
            );

            let temp_b1 = vfmaq_n_f32(temp_b1_1, x23n, self.twiddle2.im);
            let temp_b2 = vfmsq_n_f32(temp_b2_1, x23n, self.twiddle1.im);

            let temp_b1_rot = v_rotate90_f32(temp_b1, rot);
            let temp_b2_rot = v_rotate90_f32(temp_b2, rot);

            let y1 = vaddq_f32(temp_a1, temp_b1_rot);
            let y2 = vaddq_f32(temp_a2, temp_b2_rot);
            let y3 = vsubq_f32(temp_a2, temp_b2_rot);
            let y4 = vsubq_f32(temp_a1, temp_b1_rot);
            (y0, y1, y2, y3, y4)
        }
    }

    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    #[inline]
    pub(crate) fn exec_fcma(
        &self,
        u0: float32x4_t,
        u1: float32x4_t,
        u2: float32x4_t,
        u3: float32x4_t,
        u4: float32x4_t,
    ) -> (
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
    ) {
        let x14p = vaddq_f32(u1, u4);
        let x14n = vsubq_f32(u1, u4);
        let x23p = vaddq_f32(u2, u3);
        let x23n = vsubq_f32(u2, u3);
        let y0 = vaddq_f32(vaddq_f32(u0, x14p), x23p);

        let temp_b1_1 = vmulq_n_f32(x14n, self.twiddle1.im);
        let temp_b2_1 = vmulq_n_f32(x14n, self.twiddle2.im);

        let temp_a1 = vfmaq_n_f32(
            vfmaq_n_f32(u0, x14p, self.twiddle1.re),
            x23p,
            self.twiddle2.re,
        );
        let temp_a2 = vfmaq_n_f32(
            vfmaq_n_f32(u0, x14p, self.twiddle2.re),
            x23p,
            self.twiddle1.re,
        );

        let temp_b1 = vfmaq_n_f32(temp_b1_1, x23n, self.twiddle2.im);
        let temp_b2 = vfmsq_n_f32(temp_b2_1, x23n, self.twiddle1.im);

        let y1 = vcaddq_rot90_f32(temp_a1, temp_b1);
        let y2 = vcaddq_rot90_f32(temp_a2, temp_b2);
        let y3 = vcaddq_rot270_f32(temp_a2, temp_b2);
        let y4 = vcaddq_rot270_f32(temp_a1, temp_b1);
        (y0, y1, y2, y3, y4)
    }
}
