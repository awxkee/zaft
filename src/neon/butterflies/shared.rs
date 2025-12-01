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
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::util::compute_twiddle;
use std::arch::aarch64::*;

pub(crate) fn gen_butterfly_twiddles_f64<const N: usize>(
    rows: usize,
    cols: usize,
    direction: FftDirection,
    size: usize,
) -> [NeonStoreD; N] {
    let mut twiddles = [NeonStoreD::default(); N];
    let mut q = 0usize;
    let len_per_row = rows;
    const COMPLEX_PER_VECTOR: usize = 1;
    let quotient = len_per_row / COMPLEX_PER_VECTOR;
    let remainder = len_per_row % COMPLEX_PER_VECTOR;

    let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
    for x in 0..num_twiddle_columns {
        for y in 1..cols {
            twiddles[q] = NeonStoreD::from_complex(&compute_twiddle(
                y * (x * COMPLEX_PER_VECTOR),
                size,
                direction,
            ));
            q += 1;
        }
    }
    twiddles
}

pub(crate) fn gen_butterfly_twiddles_f32<const N: usize>(
    rows: usize,
    cols: usize,
    direction: FftDirection,
    size: usize,
) -> [NeonStoreF; N] {
    let mut twiddles = [NeonStoreF::default(); N];
    let mut q = 0usize;
    let len_per_row = rows;
    const COMPLEX_PER_VECTOR: usize = 2;
    let quotient = len_per_row / COMPLEX_PER_VECTOR;
    let remainder = len_per_row % COMPLEX_PER_VECTOR;

    let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
    for x in 0..num_twiddle_columns {
        for y in 1..cols {
            twiddles[q] = NeonStoreF::from_complex2(
                compute_twiddle(y * (x * COMPLEX_PER_VECTOR), size, direction),
                compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), size, direction),
            );
            q += 1;
        }
    }
    twiddles
}

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
            let xn_rot = vext_f32::<1>(xn, xn);

            let y0 = sum;
            let y1 = vfma_f32(w_1, tw_w_2, xn_rot);
            let y2 = vfms_f32(w_1, tw_w_2, xn_rot);
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
            let xn_rot = vrev64q_f32(xn);

            let y0 = sum;
            let y1 = vfmaq_f32(w_1, tw_w_2, xn_rot);
            let y2 = vfmsq_f32(w_1, tw_w_2, xn_rot);
            (y0, y1, y2)
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn butterfly3h_f32_fcma(
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
        tw_re: float32x2_t,
        tw_w_2: float32x2_t,
    ) -> (float32x2_t, float32x2_t, float32x2_t) {
        let xp = vadd_f32(u1, u2);
        let xn = vsub_f32(u1, u2);
        let sum = vadd_f32(u0, xp);

        let w_1 = vfma_f32(u0, tw_re, xp);

        let y0 = sum;
        let y1 = vcmla_rot90_f32(w_1, tw_w_2, xn);
        let y2 = vcmla_rot270_f32(w_1, tw_w_2, xn);
        (y0, y1, y2)
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn butterfly3_f32_fcma(
        u0: float32x4_t,
        u1: float32x4_t,
        u2: float32x4_t,
        tw_re: float32x4_t,
        tw_w_2: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t) {
        let xp = vaddq_f32(u1, u2);
        let xn = vsubq_f32(u1, u2);
        let sum = vaddq_f32(u0, xp);

        let w_1 = vfmaq_f32(u0, tw_re, xp);

        let y0 = sum;
        let y1 = vcmlaq_rot90_f32(w_1, tw_w_2, xn);
        let y2 = vcmlaq_rot270_f32(w_1, tw_w_2, xn);
        (y0, y1, y2)
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
            let xn_rot = vextq_f64::<1>(xn, xn);

            let y0 = sum;
            let y1 = vfmaq_f64(w_1, tw_w_2, xn_rot);
            let y2 = vfmsq_f64(w_1, tw_w_2, xn_rot);
            (y0, y1, y2)
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn butterfly3_f64_fcma(
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        tw_re: float64x2_t,
        tw_w_2: float64x2_t,
    ) -> (float64x2_t, float64x2_t, float64x2_t) {
        let xp = vaddq_f64(u1, u2);
        let xn = vsubq_f64(u1, u2);
        let sum = vaddq_f64(u0, xp);

        let w_1 = vfmaq_f64(u0, tw_re, xp);

        let y0 = sum;
        let y1 = vcmlaq_rot90_f64(w_1, tw_w_2, xn);
        let y2 = vcmlaq_rot270_f64(w_1, tw_w_2, xn);
        (y0, y1, y2)
    }

    #[inline]
    pub(crate) fn butterfly4_f64(
        a: float64x2_t,
        b: float64x2_t,
        c: float64x2_t,
        d: float64x2_t,
        rotate: float64x2_t,
    ) -> (float64x2_t, float64x2_t, float64x2_t, float64x2_t) {
        unsafe {
            let t0 = vaddq_f64(a, c);
            let t1 = vsubq_f64(a, c);
            let t2 = vaddq_f64(b, d);
            let mut t3 = vsubq_f64(b, d);
            t3 = vreinterpretq_f64_u64(veorq_u64(
                vreinterpretq_u64_f64(vextq_f64::<1>(t3, t3)),
                vreinterpretq_u64_f64(rotate),
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
        rotate: float32x2_t,
    ) -> (float32x2_t, float32x2_t, float32x2_t, float32x2_t) {
        unsafe {
            let t0 = vadd_f32(a, c);
            let t1 = vsub_f32(a, c);
            let t2 = vadd_f32(b, d);
            let mut t3 = vsub_f32(b, d);
            t3 = vreinterpret_f32_u32(veor_u32(
                vreinterpret_u32_f32(vext_f32::<1>(t3, t3)),
                vreinterpret_u32_f32(rotate),
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
        rotate: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
        unsafe {
            let t0 = vaddq_f32(a, c);
            let t1 = vsubq_f32(a, c);
            let t2 = vaddq_f32(b, d);
            let mut t3 = vsubq_f32(b, d);
            t3 = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(vrev64q_f32(t3)),
                vreinterpretq_u32_f32(rotate),
            ));
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

#[cfg(feature = "fcma")]
pub(crate) struct FastFcmaBf4f {
    pub(crate) rot_sign: float32x4_t,
}

#[cfg(feature = "fcma")]
impl FastFcmaBf4f {
    #[inline]
    pub(crate) fn new(direction: FftDirection) -> Self {
        Self {
            rot_sign: unsafe {
                match direction {
                    FftDirection::Forward => vdupq_n_f32(-1.0),
                    FftDirection::Inverse => vdupq_n_f32(1.0),
                }
            },
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(
        &self,
        a: float32x4_t,
        b: float32x4_t,
        c: float32x4_t,
        d: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
        let t0 = vaddq_f32(a, c);
        let t1 = vsubq_f32(a, c);
        let t2 = vaddq_f32(b, d);
        let t3 = vsubq_f32(b, d);
        (
            vaddq_f32(t0, t2),
            vcmlaq_rot90_f32(t1, self.rot_sign, t3),
            vsubq_f32(t0, t2),
            vcmlaq_rot270_f32(t1, self.rot_sign, t3),
        )
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(
        &self,
        a: float32x2_t,
        b: float32x2_t,
        c: float32x2_t,
        d: float32x2_t,
    ) -> (float32x2_t, float32x2_t, float32x2_t, float32x2_t) {
        let t0 = vadd_f32(a, c);
        let t1 = vsub_f32(a, c);
        let t2 = vadd_f32(b, d);
        let t3 = vsub_f32(b, d);
        (
            vadd_f32(t0, t2),
            vcmla_rot90_f32(t1, vget_low_f32(self.rot_sign), t3),
            vsub_f32(t0, t2),
            vcmla_rot270_f32(t1, vget_low_f32(self.rot_sign), t3),
        )
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct FastFcmaBf4d {
    pub(crate) rot_sign: float64x2_t,
}

#[cfg(feature = "fcma")]
impl FastFcmaBf4d {
    #[inline]
    pub(crate) fn new(direction: FftDirection) -> Self {
        Self {
            rot_sign: unsafe {
                match direction {
                    FftDirection::Forward => vdupq_n_f64(-1.0),
                    FftDirection::Inverse => vdupq_n_f64(1.0),
                }
            },
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(
        &self,
        a: float64x2_t,
        b: float64x2_t,
        c: float64x2_t,
        d: float64x2_t,
    ) -> (float64x2_t, float64x2_t, float64x2_t, float64x2_t) {
        let t0 = vaddq_f64(a, c);
        let t1 = vsubq_f64(a, c);
        let t2 = vaddq_f64(b, d);
        let t3 = vsubq_f64(b, d);
        (
            vaddq_f64(t0, t2),
            vcmlaq_rot90_f64(t1, self.rot_sign, t3),
            vsubq_f64(t0, t2),
            vcmlaq_rot270_f64(t1, self.rot_sign, t3),
        )
    }
}
