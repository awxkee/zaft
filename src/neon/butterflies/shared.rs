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
use std::arch::aarch64::*;

pub(crate) struct NeonButterfly {}

impl NeonButterfly {
    #[inline]
    pub(crate) fn butterfly3h_f32(
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
        tw_re: float32x2_t,
        tw_w_2: float32x2_t,
    ) -> (float32x2_t, float32x2_t, float32x2_t) {
        unsafe {
            let xp = vadd_f32(u1, u2);
            let xn = vsub_f32(u1, u2);
            let sum = vadd_f32(u0, xp);

            let w_1 = vfma_f32(u0, tw_re, xp);
            let vw_2 = vmul_f32(tw_w_2, vext_f32::<1>(xn, xn));

            let y0 = sum;
            let y1 = vadd_f32(w_1, vw_2);
            let y2 = vsub_f32(w_1, vw_2);
            (y0, y1, y2)
        }
    }

    #[inline]
    pub(crate) fn butterfly3_f32(
        u0: float32x4_t,
        u1: float32x4_t,
        u2: float32x4_t,
        tw_re: float32x4_t,
        tw_w_2: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t) {
        unsafe {
            let xp = vaddq_f32(u1, u2);
            let xn = vsubq_f32(u1, u2);
            let sum = vaddq_f32(u0, xp);

            let w_1 = vfmaq_f32(u0, tw_re, xp);
            let vw_2 = vmulq_f32(tw_w_2, vrev64q_f32(xn));

            let y0 = sum;
            let y1 = vaddq_f32(w_1, vw_2);
            let y2 = vsubq_f32(w_1, vw_2);
            (y0, y1, y2)
        }
    }

    #[inline]
    pub(crate) fn butterfly2h_f32(u0: float32x2_t, u1: float32x2_t) -> (float32x2_t, float32x2_t) {
        unsafe {
            let t = vadd_f32(u0, u1);

            let y1 = vsub_f32(u0, u1);
            let y0 = t;
            (y0, y1)
        }
    }

    #[inline]
    pub(crate) fn butterfly2_f32(u0: float32x4_t, u1: float32x4_t) -> (float32x4_t, float32x4_t) {
        unsafe {
            let t = vaddq_f32(u0, u1);

            let y1 = vsubq_f32(u0, u1);
            let y0 = t;
            (y0, y1)
        }
    }

    #[inline]
    pub(crate) fn butterfly3_f64(
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        tw_re: float64x2_t,
        tw_w_2: float64x2_t,
    ) -> (float64x2_t, float64x2_t, float64x2_t) {
        unsafe {
            let xp = vaddq_f64(u1, u2);
            let xn = vsubq_f64(u1, u2);
            let sum = vaddq_f64(u0, xp);

            let w_1 = vfmaq_f64(u0, tw_re, xp);
            let vw_2 = vmulq_f64(tw_w_2, vextq_f64::<1>(xn, xn));

            let y0 = sum;
            let y1 = vaddq_f64(w_1, vw_2);
            let y2 = vsubq_f64(w_1, vw_2);
            (y0, y1, y2)
        }
    }

    #[inline]
    pub(crate) fn butterfly4_f64(
        a: float64x2_t,
        b: float64x2_t,
        c: float64x2_t,
        d: float64x2_t,
        rotate: uint64x2_t,
    ) -> (float64x2_t, float64x2_t, float64x2_t, float64x2_t) {
        unsafe {
            let t0 = vaddq_f64(a, c);
            let t1 = vsubq_f64(a, c);
            let t2 = vaddq_f64(b, d);
            let mut t3 = vsubq_f64(b, d);
            t3 = vreinterpretq_f64_u64(veorq_u64(
                vreinterpretq_u64_f64(vextq_f64::<1>(t3, t3)),
                rotate,
            ));
            (
                vaddq_f64(t0, t2),
                vaddq_f64(t1, t3),
                vsubq_f64(t0, t2),
                vsubq_f64(t1, t3),
            )
        }
    }

    #[inline]
    pub(crate) fn butterfly4h_f32(
        a: float32x2_t,
        b: float32x2_t,
        c: float32x2_t,
        d: float32x2_t,
        rotate: uint32x2_t,
    ) -> (float32x2_t, float32x2_t, float32x2_t, float32x2_t) {
        unsafe {
            let t0 = vadd_f32(a, c);
            let t1 = vsub_f32(a, c);
            let t2 = vadd_f32(b, d);
            let mut t3 = vsub_f32(b, d);
            t3 = vreinterpret_f32_u32(veor_u32(
                vreinterpret_u32_f32(vext_f32::<1>(t3, t3)),
                rotate,
            ));
            (
                vadd_f32(t0, t2),
                vadd_f32(t1, t3),
                vsub_f32(t0, t2),
                vsub_f32(t1, t3),
            )
        }
    }

    #[inline]
    pub(crate) fn butterfly4_f32(
        a: float32x4_t,
        b: float32x4_t,
        c: float32x4_t,
        d: float32x4_t,
        rotate: uint32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
        unsafe {
            let t0 = vaddq_f32(a, c);
            let t1 = vsubq_f32(a, c);
            let t2 = vaddq_f32(b, d);
            let mut t3 = vsubq_f32(b, d);
            t3 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vrev64q_f32(t3)), rotate));
            (
                vaddq_f32(t0, t2),
                vaddq_f32(t1, t3),
                vsubq_f32(t0, t2),
                vsubq_f32(t1, t3),
            )
        }
    }

    #[inline]
    pub(crate) fn butterfly2_f64(u0: float64x2_t, u1: float64x2_t) -> (float64x2_t, float64x2_t) {
        unsafe {
            let t = vaddq_f64(u0, u1);

            let y1 = vsubq_f64(u0, u1);
            let y0 = t;
            (y0, y1)
        }
    }
}
