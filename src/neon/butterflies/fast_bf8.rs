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
#[cfg(feature = "fcma")]
use crate::neon::butterflies::{FastFcmaBf4d, FastFcmaBf4f};
use crate::neon::util::{v_rotate90_f32, v_rotate90_f64, vh_rotate90_f32};
use crate::traits::FftTrigonometry;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

#[allow(unused)]
pub(crate) struct NeonFastButterfly8<T> {
    direction: FftDirection,
    root2: T,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> NeonFastButterfly8<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        NeonFastButterfly8 {
            direction: fft_direction,
            root2: 0.5f64.sqrt().as_(),
        }
    }
}

impl NeonFastButterfly8<f64> {
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
        rot: float64x2_t, // [-0.0, 0.0] - 90 rot
    ) -> (
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
            let (u0, u2, u4, u6) = NeonButterfly::butterfly4_f64(u0, u2, u4, u6, rot);
            let (u1, mut u3, mut u5, mut u7) = NeonButterfly::butterfly4_f64(u1, u3, u5, u7, rot);

            u3 = vmulq_n_f64(vaddq_f64(v_rotate90_f64(u3, rot), u3), self.root2);
            u5 = v_rotate90_f64(u5, rot);
            u7 = vmulq_n_f64(vsubq_f64(v_rotate90_f64(u7, rot), u7), self.root2);

            let (y0, y1) = NeonButterfly::butterfly2_f64(u0, u1);
            let (y2, y3) = NeonButterfly::butterfly2_f64(u2, u3);
            let (y4, y5) = NeonButterfly::butterfly2_f64(u4, u5);
            let (y6, y7) = NeonButterfly::butterfly2_f64(u6, u7);
            (y0, y2, y4, y6, y1, y3, y5, y7)
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
        u7: float64x2_t,
        bf4: &FastFcmaBf4d,
    ) -> (
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
    ) {
        let (u0, u2, u4, u6) = bf4.exec(u0, u2, u4, u6);
        let (u1, mut u3, mut u5, mut u7) = bf4.exec(u1, u3, u5, u7);

        u3 = vmulq_n_f64(vcmlaq_rot90_f64(u3, bf4.rot_sign, u3), self.root2);
        u5 = vcmlaq_rot90_f64(vdupq_n_f64(0.), bf4.rot_sign, u5);
        u7 = vmulq_n_f64(
            vsubq_f64(vcmlaq_rot90_f64(vdupq_n_f64(0.), bf4.rot_sign, u7), u7),
            self.root2,
        );

        let (y0, y1) = NeonButterfly::butterfly2_f64(u0, u1);
        let (y2, y3) = NeonButterfly::butterfly2_f64(u2, u3);
        let (y4, y5) = NeonButterfly::butterfly2_f64(u4, u5);
        let (y6, y7) = NeonButterfly::butterfly2_f64(u6, u7);
        (y0, y2, y4, y6, y1, y3, y5, y7)
    }
}

impl NeonFastButterfly8<f32> {
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
        rot: float32x2_t, // [-0.0, 0.0] - 90 rot
    ) -> (
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
            let (u0, u2, u4, u6) = NeonButterfly::butterfly4h_f32(u0, u2, u4, u6, rot);
            let (u1, mut u3, mut u5, mut u7) = NeonButterfly::butterfly4h_f32(u1, u3, u5, u7, rot);

            u3 = vmul_n_f32(vadd_f32(vh_rotate90_f32(u3, rot), u3), self.root2);
            u5 = vh_rotate90_f32(u5, rot);
            u7 = vmul_n_f32(vsub_f32(vh_rotate90_f32(u7, rot), u7), self.root2);

            let (y0, y1) = NeonButterfly::butterfly2h_f32(u0, u1);
            let (y2, y3) = NeonButterfly::butterfly2h_f32(u2, u3);
            let (y4, y5) = NeonButterfly::butterfly2h_f32(u4, u5);
            let (y6, y7) = NeonButterfly::butterfly2h_f32(u6, u7);
            (y0, y2, y4, y6, y1, y3, y5, y7)
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
        rot: float32x4_t, // [-0.0, 0.0] - 90 rot
    ) -> (
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
    ) {
        unsafe {
            let (u0, u2, u4, u6) = NeonButterfly::butterfly4_f32(u0, u2, u4, u6, rot);
            let (u1, mut u3, mut u5, mut u7) = NeonButterfly::butterfly4_f32(u1, u3, u5, u7, rot);

            u3 = vmulq_n_f32(vaddq_f32(v_rotate90_f32(u3, rot), u3), self.root2);
            u5 = v_rotate90_f32(u5, rot);
            u7 = vmulq_n_f32(vsubq_f32(v_rotate90_f32(u7, rot), u7), self.root2);

            let (y0, y1) = NeonButterfly::butterfly2_f32(u0, u1);
            let (y2, y3) = NeonButterfly::butterfly2_f32(u2, u3);
            let (y4, y5) = NeonButterfly::butterfly2_f32(u4, u5);
            let (y6, y7) = NeonButterfly::butterfly2_f32(u6, u7);
            (y0, y2, y4, y6, y1, y3, y5, y7)
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
        u7: float32x4_t,
        bf4: &FastFcmaBf4f,
    ) -> (
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
        float32x4_t,
    ) {
        let (u0, u2, u4, u6) = bf4.exec(u0, u2, u4, u6);
        let (u1, mut u3, mut u5, mut u7) = bf4.exec(u1, u3, u5, u7);

        u3 = vmulq_n_f32(vcmlaq_rot90_f32(u3, bf4.rot_sign, u3), self.root2);
        u5 = vcmlaq_rot90_f32(vdupq_n_f32(0.), bf4.rot_sign, u5);
        u7 = vmulq_n_f32(
            vsubq_f32(vcmlaq_rot90_f32(vdupq_n_f32(0.), bf4.rot_sign, u7), u7),
            self.root2,
        );

        let (y0, y1) = NeonButterfly::butterfly2_f32(u0, u1);
        let (y2, y3) = NeonButterfly::butterfly2_f32(u2, u3);
        let (y4, y5) = NeonButterfly::butterfly2_f32(u4, u5);
        let (y6, y7) = NeonButterfly::butterfly2_f32(u6, u7);
        (y0, y2, y4, y6, y1, y3, y5, y7)
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    #[cfg(feature = "fcma")]
    pub(crate) fn exech_fcma(
        &self,
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
        u3: float32x2_t,
        u4: float32x2_t,
        u5: float32x2_t,
        u6: float32x2_t,
        u7: float32x2_t,
        bf4: &FastFcmaBf4f,
    ) -> (
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
    ) {
        let (u0, u2, u4, u6) = bf4.exech(u0, u2, u4, u6);
        let (u1, mut u3, mut u5, mut u7) = bf4.exech(u1, u3, u5, u7);

        u3 = vmul_n_f32(
            vcmla_rot90_f32(u3, vget_low_f32(bf4.rot_sign), u3),
            self.root2,
        );
        u5 = vcmla_rot90_f32(vdup_n_f32(0.), vget_low_f32(bf4.rot_sign), u5);
        u7 = vmul_n_f32(
            vsub_f32(
                vcmla_rot90_f32(vdup_n_f32(0.), vget_low_f32(bf4.rot_sign), u7),
                u7,
            ),
            self.root2,
        );

        let (y0, y1) = NeonButterfly::butterfly2h_f32(u0, u1);
        let (y2, y3) = NeonButterfly::butterfly2h_f32(u2, u3);
        let (y4, y5) = NeonButterfly::butterfly2h_f32(u4, u5);
        let (y6, y7) = NeonButterfly::butterfly2h_f32(u6, u7);
        (y0, y2, y4, y6, y1, y3, y5, y7)
    }
}
