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
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF, NeonStoreFh};
use crate::neon::util::{v_rotate90_f32, v_rotate90_f64, vh_rotate90_f32};
use crate::util::compute_twiddle;
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly5d {
    rotate: float64x2_t,
    twiddle1: Complex<f64>,
    twiddle2: Complex<f64>,
}

impl ColumnButterfly5d {
    #[inline]
    pub(crate) fn new(direction: FftDirection) -> Self {
        unsafe {
            Self {
                rotate: vld1q_f64([-0.0f64, 0.0, -0.0f64, 0.0].as_ptr().cast()),
                twiddle1: compute_twiddle(1, 5, direction),
                twiddle2: compute_twiddle(2, 5, direction),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 5]) -> [NeonStoreD; 5] {
        unsafe {
            let x14p = vaddq_f64(store[1].v, store[4].v);
            let x14n = vsubq_f64(store[1].v, store[4].v);
            let x23p = vaddq_f64(store[2].v, store[3].v);
            let x23n = vsubq_f64(store[2].v, store[3].v);
            let y0 = vaddq_f64(vaddq_f64(store[0].v, x14p), x23p);

            let temp_b1_1 = vmulq_n_f64(x14n, self.twiddle1.im);
            let temp_b2_1 = vmulq_n_f64(x14n, self.twiddle2.im);

            let temp_a1 = vfmaq_n_f64(
                vfmaq_n_f64(store[0].v, x14p, self.twiddle1.re),
                x23p,
                self.twiddle2.re,
            );
            let temp_a2 = vfmaq_n_f64(
                vfmaq_n_f64(store[0].v, x14p, self.twiddle2.re),
                x23p,
                self.twiddle1.re,
            );

            let temp_b1 = vfmaq_n_f64(temp_b1_1, x23n, self.twiddle2.im);
            let temp_b2 = vfmsq_n_f64(temp_b2_1, x23n, self.twiddle1.im);

            let temp_b1_rot = v_rotate90_f64(temp_b1, self.rotate);
            let temp_b2_rot = v_rotate90_f64(temp_b2, self.rotate);

            let y1 = vaddq_f64(temp_a1, temp_b1_rot);
            let y2 = vaddq_f64(temp_a2, temp_b2_rot);
            let y3 = vsubq_f64(temp_a2, temp_b2_rot);
            let y4 = vsubq_f64(temp_a1, temp_b1_rot);
            [
                NeonStoreD::raw(y0),
                NeonStoreD::raw(y1),
                NeonStoreD::raw(y2),
                NeonStoreD::raw(y3),
                NeonStoreD::raw(y4),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly5d {
    twiddle1: Complex<f64>,
    twiddle2: Complex<f64>,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly5d {
    #[inline]
    pub(crate) fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: compute_twiddle(1, 5, direction),
            twiddle2: compute_twiddle(2, 5, direction),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 5]) -> [NeonStoreD; 5] {
        let x14p = vaddq_f64(store[1].v, store[4].v);
        let x14n = vsubq_f64(store[1].v, store[4].v);
        let x23p = vaddq_f64(store[2].v, store[3].v);
        let x23n = vsubq_f64(store[2].v, store[3].v);
        let y0 = vaddq_f64(vaddq_f64(store[0].v, x14p), x23p);

        let temp_b1_1 = vmulq_n_f64(x14n, self.twiddle1.im);
        let temp_b2_1 = vmulq_n_f64(x14n, self.twiddle2.im);

        let temp_a1 = vfmaq_n_f64(
            vfmaq_n_f64(store[0].v, x14p, self.twiddle1.re),
            x23p,
            self.twiddle2.re,
        );
        let temp_a2 = vfmaq_n_f64(
            vfmaq_n_f64(store[0].v, x14p, self.twiddle2.re),
            x23p,
            self.twiddle1.re,
        );

        let temp_b1 = vfmaq_n_f64(temp_b1_1, x23n, self.twiddle2.im);
        let temp_b2 = vfmsq_n_f64(temp_b2_1, x23n, self.twiddle1.im);

        let temp_b1_rot = vcaddq_rot90_f64(vdupq_n_f64(0.), temp_b1);
        let temp_b2_rot = vcaddq_rot90_f64(vdupq_n_f64(0.), temp_b2);

        let y1 = vaddq_f64(temp_a1, temp_b1_rot);
        let y2 = vaddq_f64(temp_a2, temp_b2_rot);
        let y3 = vsubq_f64(temp_a2, temp_b2_rot);
        let y4 = vsubq_f64(temp_a1, temp_b1_rot);
        [
            NeonStoreD::raw(y0),
            NeonStoreD::raw(y1),
            NeonStoreD::raw(y2),
            NeonStoreD::raw(y3),
            NeonStoreD::raw(y4),
        ]
    }
}

pub(crate) struct ColumnButterfly5f {
    rotate: float32x4_t,
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
}

impl ColumnButterfly5f {
    #[inline]
    pub(crate) fn new(direction: FftDirection) -> Self {
        unsafe {
            Self {
                rotate: vld1q_f32([-0.0f32, 0.0, -0.0f32, 0.0].as_ptr().cast()),
                twiddle1: compute_twiddle(1, 5, direction),
                twiddle2: compute_twiddle(2, 5, direction),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 5]) -> [NeonStoreF; 5] {
        unsafe {
            let x14p = vaddq_f32(store[1].v, store[4].v);
            let x14n = vsubq_f32(store[1].v, store[4].v);
            let x23p = vaddq_f32(store[2].v, store[3].v);
            let x23n = vsubq_f32(store[2].v, store[3].v);
            let y0 = vaddq_f32(vaddq_f32(store[0].v, x14p), x23p);

            let temp_b1_1 = vmulq_n_f32(x14n, self.twiddle1.im);
            let temp_b2_1 = vmulq_n_f32(x14n, self.twiddle2.im);

            let temp_a1 = vfmaq_n_f32(
                vfmaq_n_f32(store[0].v, x14p, self.twiddle1.re),
                x23p,
                self.twiddle2.re,
            );
            let temp_a2 = vfmaq_n_f32(
                vfmaq_n_f32(store[0].v, x14p, self.twiddle2.re),
                x23p,
                self.twiddle1.re,
            );

            let temp_b1 = vfmaq_n_f32(temp_b1_1, x23n, self.twiddle2.im);
            let temp_b2 = vfmsq_n_f32(temp_b2_1, x23n, self.twiddle1.im);

            let temp_b1_rot = v_rotate90_f32(temp_b1, self.rotate);
            let temp_b2_rot = v_rotate90_f32(temp_b2, self.rotate);

            let y1 = vaddq_f32(temp_a1, temp_b1_rot);
            let y2 = vaddq_f32(temp_a2, temp_b2_rot);
            let y3 = vsubq_f32(temp_a2, temp_b2_rot);
            let y4 = vsubq_f32(temp_a1, temp_b1_rot);
            [
                NeonStoreF::raw(y0),
                NeonStoreF::raw(y1),
                NeonStoreF::raw(y2),
                NeonStoreF::raw(y3),
                NeonStoreF::raw(y4),
            ]
        }
    }

    #[inline(always)]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 5]) -> [NeonStoreFh; 5] {
        unsafe {
            let x14p = vadd_f32(store[1].v, store[4].v);
            let x14n = vsub_f32(store[1].v, store[4].v);
            let x23p = vadd_f32(store[2].v, store[3].v);
            let x23n = vsub_f32(store[2].v, store[3].v);
            let y0 = vadd_f32(vadd_f32(store[0].v, x14p), x23p);

            let temp_b1_1 = vmul_n_f32(x14n, self.twiddle1.im);
            let temp_b2_1 = vmul_n_f32(x14n, self.twiddle2.im);

            let temp_a1 = vfma_n_f32(
                vfma_n_f32(store[0].v, x14p, self.twiddle1.re),
                x23p,
                self.twiddle2.re,
            );
            let temp_a2 = vfma_n_f32(
                vfma_n_f32(store[0].v, x14p, self.twiddle2.re),
                x23p,
                self.twiddle1.re,
            );

            let temp_b1 = vfma_n_f32(temp_b1_1, x23n, self.twiddle2.im);
            let temp_b2 = vfms_n_f32(temp_b2_1, x23n, self.twiddle1.im);

            let temp_b1_rot = vh_rotate90_f32(temp_b1, vget_low_f32(self.rotate));
            let temp_b2_rot = vh_rotate90_f32(temp_b2, vget_low_f32(self.rotate));

            let y1 = vadd_f32(temp_a1, temp_b1_rot);
            let y2 = vadd_f32(temp_a2, temp_b2_rot);
            let y3 = vsub_f32(temp_a2, temp_b2_rot);
            let y4 = vsub_f32(temp_a1, temp_b1_rot);
            [
                NeonStoreFh::raw(y0),
                NeonStoreFh::raw(y1),
                NeonStoreFh::raw(y2),
                NeonStoreFh::raw(y3),
                NeonStoreFh::raw(y4),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly5f {
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly5f {
    #[inline]
    pub(crate) fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: compute_twiddle(1, 5, direction),
            twiddle2: compute_twiddle(2, 5, direction),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 5]) -> [NeonStoreF; 5] {
        let x14p = vaddq_f32(store[1].v, store[4].v);
        let x14n = vsubq_f32(store[1].v, store[4].v);
        let x23p = vaddq_f32(store[2].v, store[3].v);
        let x23n = vsubq_f32(store[2].v, store[3].v);
        let y0 = vaddq_f32(vaddq_f32(store[0].v, x14p), x23p);

        let temp_b1_1 = vmulq_n_f32(x14n, self.twiddle1.im);
        let temp_b2_1 = vmulq_n_f32(x14n, self.twiddle2.im);

        let temp_a1 = vfmaq_n_f32(
            vfmaq_n_f32(store[0].v, x14p, self.twiddle1.re),
            x23p,
            self.twiddle2.re,
        );
        let temp_a2 = vfmaq_n_f32(
            vfmaq_n_f32(store[0].v, x14p, self.twiddle2.re),
            x23p,
            self.twiddle1.re,
        );

        let temp_b1 = vfmaq_n_f32(temp_b1_1, x23n, self.twiddle2.im);
        let temp_b2 = vfmsq_n_f32(temp_b2_1, x23n, self.twiddle1.im);

        let temp_b1_rot = vcaddq_rot90_f32(vdupq_n_f32(0.), temp_b1);
        let temp_b2_rot = vcaddq_rot90_f32(vdupq_n_f32(0.), temp_b2);

        let y1 = vaddq_f32(temp_a1, temp_b1_rot);
        let y2 = vaddq_f32(temp_a2, temp_b2_rot);
        let y3 = vsubq_f32(temp_a2, temp_b2_rot);
        let y4 = vsubq_f32(temp_a1, temp_b1_rot);
        [
            NeonStoreF::raw(y0),
            NeonStoreF::raw(y1),
            NeonStoreF::raw(y2),
            NeonStoreF::raw(y3),
            NeonStoreF::raw(y4),
        ]
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 5]) -> [NeonStoreFh; 5] {
        let x14p = vadd_f32(store[1].v, store[4].v);
        let x14n = vsub_f32(store[1].v, store[4].v);
        let x23p = vadd_f32(store[2].v, store[3].v);
        let x23n = vsub_f32(store[2].v, store[3].v);
        let y0 = vadd_f32(vadd_f32(store[0].v, x14p), x23p);

        let temp_b1_1 = vmul_n_f32(x14n, self.twiddle1.im);
        let temp_b2_1 = vmul_n_f32(x14n, self.twiddle2.im);

        let temp_a1 = vfma_n_f32(
            vfma_n_f32(store[0].v, x14p, self.twiddle1.re),
            x23p,
            self.twiddle2.re,
        );
        let temp_a2 = vfma_n_f32(
            vfma_n_f32(store[0].v, x14p, self.twiddle2.re),
            x23p,
            self.twiddle1.re,
        );

        let temp_b1 = vfma_n_f32(temp_b1_1, x23n, self.twiddle2.im);
        let temp_b2 = vfms_n_f32(temp_b2_1, x23n, self.twiddle1.im);

        let temp_b1_rot = vcadd_rot90_f32(vdup_n_f32(0.), temp_b1);
        let temp_b2_rot = vcadd_rot90_f32(vdup_n_f32(0.), temp_b2);

        let y1 = vadd_f32(temp_a1, temp_b1_rot);
        let y2 = vadd_f32(temp_a2, temp_b2_rot);
        let y3 = vsub_f32(temp_a2, temp_b2_rot);
        let y4 = vsub_f32(temp_a1, temp_b1_rot);
        [
            NeonStoreFh::raw(y0),
            NeonStoreFh::raw(y1),
            NeonStoreFh::raw(y2),
            NeonStoreFh::raw(y3),
            NeonStoreFh::raw(y4),
        ]
    }
}
