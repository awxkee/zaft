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
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF, NeonStoreFh};
use crate::neon::util::{v_rotate90_f32, v_rotate90_f64, vh_rotate90_f32};
use crate::util::compute_twiddle;
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly7d {
    rotate: float64x2_t,
    twiddle1: Complex<f64>,
    twiddle2: Complex<f64>,
    twiddle3: Complex<f64>,
}

impl ColumnButterfly7d {
    pub(crate) fn new(direction: FftDirection) -> Self {
        unsafe {
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            let rot_sign = vld1q_f64(ROT_90.as_ptr());

            Self {
                rotate: rot_sign,
                twiddle1: compute_twiddle(1, 7, direction),
                twiddle2: compute_twiddle(2, 7, direction),
                twiddle3: compute_twiddle(3, 7, direction),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 7]) -> [NeonStoreD; 7] {
        unsafe {
            let (x1p6, x1m6) = NeonButterfly::butterfly2_f64(store[1].v, store[6].v);
            let x1m6 = v_rotate90_f64(x1m6, self.rotate);
            let y00 = vaddq_f64(store[0].v, x1p6);
            let (x2p5, x2m5) = NeonButterfly::butterfly2_f64(store[2].v, store[5].v);
            let x2m5 = v_rotate90_f64(x2m5, self.rotate);
            let y00 = vaddq_f64(y00, x2p5);
            let (x3p4, x3m4) = NeonButterfly::butterfly2_f64(store[3].v, store[4].v);
            let x3m4 = v_rotate90_f64(x3m4, self.rotate);
            let y00 = vaddq_f64(y00, x3p4);

            let m0106a = vfmaq_n_f64(store[0].v, x1p6, self.twiddle1.re);
            let m0106a = vfmaq_n_f64(m0106a, x2p5, self.twiddle2.re);
            let m0106a = vfmaq_n_f64(m0106a, x3p4, self.twiddle3.re);
            let m0106b = vmulq_n_f64(x1m6, self.twiddle1.im);
            let m0106b = vfmaq_n_f64(m0106b, x2m5, self.twiddle2.im);
            let m0106b = vfmaq_n_f64(m0106b, x3m4, self.twiddle3.im);
            let (y01, y06) = NeonButterfly::butterfly2_f64(m0106a, m0106b);

            let m0205a = vfmaq_n_f64(store[0].v, x1p6, self.twiddle2.re);
            let m0205a = vfmaq_n_f64(m0205a, x2p5, self.twiddle3.re);
            let m0205a = vfmaq_n_f64(m0205a, x3p4, self.twiddle1.re);
            let m0205b = vmulq_n_f64(x1m6, self.twiddle2.im);
            let m0205b = vfmsq_n_f64(m0205b, x2m5, self.twiddle3.im);
            let m0205b = vfmsq_n_f64(m0205b, x3m4, self.twiddle1.im);
            let (y02, y05) = NeonButterfly::butterfly2_f64(m0205a, m0205b);

            let m0304a = vfmaq_n_f64(store[0].v, x1p6, self.twiddle3.re);
            let m0304a = vfmaq_n_f64(m0304a, x2p5, self.twiddle1.re);
            let m0304a = vfmaq_n_f64(m0304a, x3p4, self.twiddle2.re);
            let m0304b = vmulq_n_f64(x1m6, self.twiddle3.im);
            let m0304b = vfmsq_n_f64(m0304b, x2m5, self.twiddle1.im);
            let m0304b = vfmaq_n_f64(m0304b, x3m4, self.twiddle2.im);
            let (y03, y04) = NeonButterfly::butterfly2_f64(m0304a, m0304b);
            [
                NeonStoreD::raw(y00),
                NeonStoreD::raw(y01),
                NeonStoreD::raw(y02),
                NeonStoreD::raw(y03),
                NeonStoreD::raw(y04),
                NeonStoreD::raw(y05),
                NeonStoreD::raw(y06),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly7d {
    twiddle1: Complex<f64>,
    twiddle2: Complex<f64>,
    twiddle3: Complex<f64>,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly7d {
    pub(crate) fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: compute_twiddle(1, 7, direction),
            twiddle2: compute_twiddle(2, 7, direction),
            twiddle3: compute_twiddle(3, 7, direction),
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 7]) -> [NeonStoreD; 7] {
        let (x1p6, x1m6) = NeonButterfly::butterfly2_f64(store[1].v, store[6].v);
        let x1m6 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m6);
        let y00 = vaddq_f64(store[0].v, x1p6);
        let (x2p5, x2m5) = NeonButterfly::butterfly2_f64(store[2].v, store[5].v);
        let x2m5 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m5);
        let y00 = vaddq_f64(y00, x2p5);
        let (x3p4, x3m4) = NeonButterfly::butterfly2_f64(store[3].v, store[4].v);
        let x3m4 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m4);
        let y00 = vaddq_f64(y00, x3p4);

        let m0106a = vfmaq_n_f64(store[0].v, x1p6, self.twiddle1.re);
        let m0106a = vfmaq_n_f64(m0106a, x2p5, self.twiddle2.re);
        let m0106a = vfmaq_n_f64(m0106a, x3p4, self.twiddle3.re);
        let m0106b = vmulq_n_f64(x1m6, self.twiddle1.im);
        let m0106b = vfmaq_n_f64(m0106b, x2m5, self.twiddle2.im);
        let m0106b = vfmaq_n_f64(m0106b, x3m4, self.twiddle3.im);
        let (y01, y06) = NeonButterfly::butterfly2_f64(m0106a, m0106b);

        let m0205a = vfmaq_n_f64(store[0].v, x1p6, self.twiddle2.re);
        let m0205a = vfmaq_n_f64(m0205a, x2p5, self.twiddle3.re);
        let m0205a = vfmaq_n_f64(m0205a, x3p4, self.twiddle1.re);
        let m0205b = vmulq_n_f64(x1m6, self.twiddle2.im);
        let m0205b = vfmsq_n_f64(m0205b, x2m5, self.twiddle3.im);
        let m0205b = vfmsq_n_f64(m0205b, x3m4, self.twiddle1.im);
        let (y02, y05) = NeonButterfly::butterfly2_f64(m0205a, m0205b);

        let m0304a = vfmaq_n_f64(store[0].v, x1p6, self.twiddle3.re);
        let m0304a = vfmaq_n_f64(m0304a, x2p5, self.twiddle1.re);
        let m0304a = vfmaq_n_f64(m0304a, x3p4, self.twiddle2.re);
        let m0304b = vmulq_n_f64(x1m6, self.twiddle3.im);
        let m0304b = vfmsq_n_f64(m0304b, x2m5, self.twiddle1.im);
        let m0304b = vfmaq_n_f64(m0304b, x3m4, self.twiddle2.im);
        let (y03, y04) = NeonButterfly::butterfly2_f64(m0304a, m0304b);
        [
            NeonStoreD::raw(y00),
            NeonStoreD::raw(y01),
            NeonStoreD::raw(y02),
            NeonStoreD::raw(y03),
            NeonStoreD::raw(y04),
            NeonStoreD::raw(y05),
            NeonStoreD::raw(y06),
        ]
    }
}

pub(crate) struct ColumnButterfly7f {
    rotate: float32x4_t,
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
    twiddle3: Complex<f32>,
}

impl ColumnButterfly7f {
    pub(crate) fn new(direction: FftDirection) -> Self {
        unsafe {
            static ROT_90: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            let rot_sign = vld1q_f32(ROT_90.as_ptr());

            Self {
                rotate: rot_sign,
                twiddle1: compute_twiddle(1, 7, direction),
                twiddle2: compute_twiddle(2, 7, direction),
                twiddle3: compute_twiddle(3, 7, direction),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 7]) -> [NeonStoreF; 7] {
        unsafe {
            let (x1p6, x1m6) = NeonButterfly::butterfly2_f32(store[1].v, store[6].v);
            let x1m6 = v_rotate90_f32(x1m6, self.rotate);
            let y00 = vaddq_f32(store[0].v, x1p6);
            let (x2p5, x2m5) = NeonButterfly::butterfly2_f32(store[2].v, store[5].v);
            let x2m5 = v_rotate90_f32(x2m5, self.rotate);
            let y00 = vaddq_f32(y00, x2p5);
            let (x3p4, x3m4) = NeonButterfly::butterfly2_f32(store[3].v, store[4].v);
            let x3m4 = v_rotate90_f32(x3m4, self.rotate);
            let y00 = vaddq_f32(y00, x3p4);

            let m0106a = vfmaq_n_f32(store[0].v, x1p6, self.twiddle1.re);
            let m0106a = vfmaq_n_f32(m0106a, x2p5, self.twiddle2.re);
            let m0106a = vfmaq_n_f32(m0106a, x3p4, self.twiddle3.re);
            let m0106b = vmulq_n_f32(x1m6, self.twiddle1.im);
            let m0106b = vfmaq_n_f32(m0106b, x2m5, self.twiddle2.im);
            let m0106b = vfmaq_n_f32(m0106b, x3m4, self.twiddle3.im);
            let (y01, y06) = NeonButterfly::butterfly2_f32(m0106a, m0106b);

            let m0205a = vfmaq_n_f32(store[0].v, x1p6, self.twiddle2.re);
            let m0205a = vfmaq_n_f32(m0205a, x2p5, self.twiddle3.re);
            let m0205a = vfmaq_n_f32(m0205a, x3p4, self.twiddle1.re);
            let m0205b = vmulq_n_f32(x1m6, self.twiddle2.im);
            let m0205b = vfmsq_n_f32(m0205b, x2m5, self.twiddle3.im);
            let m0205b = vfmsq_n_f32(m0205b, x3m4, self.twiddle1.im);
            let (y02, y05) = NeonButterfly::butterfly2_f32(m0205a, m0205b);

            let m0304a = vfmaq_n_f32(store[0].v, x1p6, self.twiddle3.re);
            let m0304a = vfmaq_n_f32(m0304a, x2p5, self.twiddle1.re);
            let m0304a = vfmaq_n_f32(m0304a, x3p4, self.twiddle2.re);
            let m0304b = vmulq_n_f32(x1m6, self.twiddle3.im);
            let m0304b = vfmsq_n_f32(m0304b, x2m5, self.twiddle1.im);
            let m0304b = vfmaq_n_f32(m0304b, x3m4, self.twiddle2.im);
            let (y03, y04) = NeonButterfly::butterfly2_f32(m0304a, m0304b);
            [
                NeonStoreF::raw(y00),
                NeonStoreF::raw(y01),
                NeonStoreF::raw(y02),
                NeonStoreF::raw(y03),
                NeonStoreF::raw(y04),
                NeonStoreF::raw(y05),
                NeonStoreF::raw(y06),
            ]
        }
    }

    #[inline(always)]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 7]) -> [NeonStoreFh; 7] {
        unsafe {
            let (x1p6, x1m6) = NeonButterfly::butterfly2h_f32(store[1].v, store[6].v);
            let x1m6 = vh_rotate90_f32(x1m6, vget_low_f32(self.rotate));
            let y00 = vadd_f32(store[0].v, x1p6);
            let (x2p5, x2m5) = NeonButterfly::butterfly2h_f32(store[2].v, store[5].v);
            let x2m5 = vh_rotate90_f32(x2m5, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x2p5);
            let (x3p4, x3m4) = NeonButterfly::butterfly2h_f32(store[3].v, store[4].v);
            let x3m4 = vh_rotate90_f32(x3m4, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x3p4);

            let m0106a = vfma_n_f32(store[0].v, x1p6, self.twiddle1.re);
            let m0106a = vfma_n_f32(m0106a, x2p5, self.twiddle2.re);
            let m0106a = vfma_n_f32(m0106a, x3p4, self.twiddle3.re);
            let m0106b = vmul_n_f32(x1m6, self.twiddle1.im);
            let m0106b = vfma_n_f32(m0106b, x2m5, self.twiddle2.im);
            let m0106b = vfma_n_f32(m0106b, x3m4, self.twiddle3.im);
            let (y01, y06) = NeonButterfly::butterfly2h_f32(m0106a, m0106b);

            let m0205a = vfma_n_f32(store[0].v, x1p6, self.twiddle2.re);
            let m0205a = vfma_n_f32(m0205a, x2p5, self.twiddle3.re);
            let m0205a = vfma_n_f32(m0205a, x3p4, self.twiddle1.re);
            let m0205b = vmul_n_f32(x1m6, self.twiddle2.im);
            let m0205b = vfms_n_f32(m0205b, x2m5, self.twiddle3.im);
            let m0205b = vfms_n_f32(m0205b, x3m4, self.twiddle1.im);
            let (y02, y05) = NeonButterfly::butterfly2h_f32(m0205a, m0205b);

            let m0304a = vfma_n_f32(store[0].v, x1p6, self.twiddle3.re);
            let m0304a = vfma_n_f32(m0304a, x2p5, self.twiddle1.re);
            let m0304a = vfma_n_f32(m0304a, x3p4, self.twiddle2.re);
            let m0304b = vmul_n_f32(x1m6, self.twiddle3.im);
            let m0304b = vfms_n_f32(m0304b, x2m5, self.twiddle1.im);
            let m0304b = vfma_n_f32(m0304b, x3m4, self.twiddle2.im);
            let (y03, y04) = NeonButterfly::butterfly2h_f32(m0304a, m0304b);
            [
                NeonStoreFh::raw(y00),
                NeonStoreFh::raw(y01),
                NeonStoreFh::raw(y02),
                NeonStoreFh::raw(y03),
                NeonStoreFh::raw(y04),
                NeonStoreFh::raw(y05),
                NeonStoreFh::raw(y06),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly7f {
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
    twiddle3: Complex<f32>,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly7f {
    pub(crate) fn new(direction: FftDirection) -> Self {
        Self {
            twiddle1: compute_twiddle(1, 7, direction),
            twiddle2: compute_twiddle(2, 7, direction),
            twiddle3: compute_twiddle(3, 7, direction),
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 7]) -> [NeonStoreF; 7] {
        let (x1p6, x1m6) = NeonButterfly::butterfly2_f32(store[1].v, store[6].v);
        let x1m6 = vcaddq_rot90_f32(vdupq_n_f32(0.), x1m6);
        let y00 = vaddq_f32(store[0].v, x1p6);
        let (x2p5, x2m5) = NeonButterfly::butterfly2_f32(store[2].v, store[5].v);
        let x2m5 = vcaddq_rot90_f32(vdupq_n_f32(0.), x2m5);
        let y00 = vaddq_f32(y00, x2p5);
        let (x3p4, x3m4) = NeonButterfly::butterfly2_f32(store[3].v, store[4].v);
        let x3m4 = vcaddq_rot90_f32(vdupq_n_f32(0.), x3m4);
        let y00 = vaddq_f32(y00, x3p4);

        let m0106a = vfmaq_n_f32(store[0].v, x1p6, self.twiddle1.re);
        let m0106a = vfmaq_n_f32(m0106a, x2p5, self.twiddle2.re);
        let m0106a = vfmaq_n_f32(m0106a, x3p4, self.twiddle3.re);
        let m0106b = vmulq_n_f32(x1m6, self.twiddle1.im);
        let m0106b = vfmaq_n_f32(m0106b, x2m5, self.twiddle2.im);
        let m0106b = vfmaq_n_f32(m0106b, x3m4, self.twiddle3.im);
        let (y01, y06) = NeonButterfly::butterfly2_f32(m0106a, m0106b);

        let m0205a = vfmaq_n_f32(store[0].v, x1p6, self.twiddle2.re);
        let m0205a = vfmaq_n_f32(m0205a, x2p5, self.twiddle3.re);
        let m0205a = vfmaq_n_f32(m0205a, x3p4, self.twiddle1.re);
        let m0205b = vmulq_n_f32(x1m6, self.twiddle2.im);
        let m0205b = vfmsq_n_f32(m0205b, x2m5, self.twiddle3.im);
        let m0205b = vfmsq_n_f32(m0205b, x3m4, self.twiddle1.im);
        let (y02, y05) = NeonButterfly::butterfly2_f32(m0205a, m0205b);

        let m0304a = vfmaq_n_f32(store[0].v, x1p6, self.twiddle3.re);
        let m0304a = vfmaq_n_f32(m0304a, x2p5, self.twiddle1.re);
        let m0304a = vfmaq_n_f32(m0304a, x3p4, self.twiddle2.re);
        let m0304b = vmulq_n_f32(x1m6, self.twiddle3.im);
        let m0304b = vfmsq_n_f32(m0304b, x2m5, self.twiddle1.im);
        let m0304b = vfmaq_n_f32(m0304b, x3m4, self.twiddle2.im);
        let (y03, y04) = NeonButterfly::butterfly2_f32(m0304a, m0304b);
        [
            NeonStoreF::raw(y00),
            NeonStoreF::raw(y01),
            NeonStoreF::raw(y02),
            NeonStoreF::raw(y03),
            NeonStoreF::raw(y04),
            NeonStoreF::raw(y05),
            NeonStoreF::raw(y06),
        ]
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 7]) -> [NeonStoreFh; 7] {
        let (x1p6, x1m6) = NeonButterfly::butterfly2h_f32(store[1].v, store[6].v);
        let x1m6 = vcadd_rot90_f32(vdup_n_f32(0.), x1m6);
        let y00 = vadd_f32(store[0].v, x1p6);
        let (x2p5, x2m5) = NeonButterfly::butterfly2h_f32(store[2].v, store[5].v);
        let x2m5 = vcadd_rot90_f32(vdup_n_f32(0.), x2m5);
        let y00 = vadd_f32(y00, x2p5);
        let (x3p4, x3m4) = NeonButterfly::butterfly2h_f32(store[3].v, store[4].v);
        let x3m4 = vcadd_rot90_f32(vdup_n_f32(0.), x3m4);
        let y00 = vadd_f32(y00, x3p4);

        let m0106a = vfma_n_f32(store[0].v, x1p6, self.twiddle1.re);
        let m0106a = vfma_n_f32(m0106a, x2p5, self.twiddle2.re);
        let m0106a = vfma_n_f32(m0106a, x3p4, self.twiddle3.re);
        let m0106b = vmul_n_f32(x1m6, self.twiddle1.im);
        let m0106b = vfma_n_f32(m0106b, x2m5, self.twiddle2.im);
        let m0106b = vfma_n_f32(m0106b, x3m4, self.twiddle3.im);
        let (y01, y06) = NeonButterfly::butterfly2h_f32(m0106a, m0106b);

        let m0205a = vfma_n_f32(store[0].v, x1p6, self.twiddle2.re);
        let m0205a = vfma_n_f32(m0205a, x2p5, self.twiddle3.re);
        let m0205a = vfma_n_f32(m0205a, x3p4, self.twiddle1.re);
        let m0205b = vmul_n_f32(x1m6, self.twiddle2.im);
        let m0205b = vfms_n_f32(m0205b, x2m5, self.twiddle3.im);
        let m0205b = vfms_n_f32(m0205b, x3m4, self.twiddle1.im);
        let (y02, y05) = NeonButterfly::butterfly2h_f32(m0205a, m0205b);

        let m0304a = vfma_n_f32(store[0].v, x1p6, self.twiddle3.re);
        let m0304a = vfma_n_f32(m0304a, x2p5, self.twiddle1.re);
        let m0304a = vfma_n_f32(m0304a, x3p4, self.twiddle2.re);
        let m0304b = vmul_n_f32(x1m6, self.twiddle3.im);
        let m0304b = vfms_n_f32(m0304b, x2m5, self.twiddle1.im);
        let m0304b = vfma_n_f32(m0304b, x3m4, self.twiddle2.im);
        let (y03, y04) = NeonButterfly::butterfly2h_f32(m0304a, m0304b);
        [
            NeonStoreFh::raw(y00),
            NeonStoreFh::raw(y01),
            NeonStoreFh::raw(y02),
            NeonStoreFh::raw(y03),
            NeonStoreFh::raw(y04),
            NeonStoreFh::raw(y05),
            NeonStoreFh::raw(y06),
        ]
    }
}

pub(crate) struct ColumnRdftButterfly7f {
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
    twiddle3: Complex<f32>,
}

impl ColumnRdftButterfly7f {
    pub(crate) fn new() -> Self {
        Self {
            twiddle1: compute_twiddle(1, 7, FftDirection::Forward),
            twiddle2: compute_twiddle(2, 7, FftDirection::Forward),
            twiddle3: compute_twiddle(3, 7, FftDirection::Forward),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 7]) -> [[NeonStoreF; 4]; 2] {
        let x16p = store[1] + store[6];
        let x16n = store[1] - store[6];
        let x25p = store[2] + store[5];
        let x25n = store[2] - store[5];
        let x34p = store[3] + store[4];
        let x34n = store[3] - store[4];
        let y0 = store[0] + x16p + x25p + x34p;

        let x16re_a = x16p.mul_f32_add(
            self.twiddle1.re,
            x25p.mul_f32_add(
                self.twiddle2.re,
                x34p.mul_f32_add(self.twiddle3.re, store[0]),
            ),
        );
        let x25re_a = x34p.mul_f32_add(
            self.twiddle1.re,
            x16p.mul_f32_add(
                self.twiddle2.re,
                x25p.mul_f32_add(self.twiddle3.re, store[0]),
            ),
        );
        let x34re_a = x25p.mul_f32_add(
            self.twiddle1.re,
            x34p.mul_f32_add(
                self.twiddle2.re,
                x16p.mul_f32_add(self.twiddle3.re, store[0]),
            ),
        );
        let x16im_b = x16n.mul_f32_add(
            self.twiddle1.im,
            x25n.mul_f32_add(self.twiddle2.im, x34n * self.twiddle3.im),
        );
        let x25im_b = x34n.mul_f32_add(
            -self.twiddle1.im,
            x16n.mul_f32_add(self.twiddle2.im, x25n * -self.twiddle3.im),
        );
        let x34im_b = x25n.mul_f32_add(
            self.twiddle1.im,
            x34n.mul_f32_add(-self.twiddle2.im, x16n * -self.twiddle3.im),
        );
        let v0 = y0.zip_complex(NeonStoreF::zero());
        let v1 = x16re_a.zip_complex(x16im_b);
        let v2 = x25re_a.zip_complex(x25im_b);
        let v3 = x34re_a.zip_complex(-x34im_b);
        [[v0[0], v1[0], v2[0], v3[0]], [v0[1], v1[1], v2[1], v3[1]]]
    }
}

pub(crate) struct ColumnRdftButterfly7d {
    twiddle1: Complex<f64>,
    twiddle2: Complex<f64>,
    twiddle3: Complex<f64>,
}

impl ColumnRdftButterfly7d {
    pub(crate) fn new() -> Self {
        Self {
            twiddle1: compute_twiddle(1, 7, FftDirection::Forward),
            twiddle2: compute_twiddle(2, 7, FftDirection::Forward),
            twiddle3: compute_twiddle(3, 7, FftDirection::Forward),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 7]) -> [[NeonStoreD; 4]; 2] {
        let x16p = store[1] + store[6];
        let x16n = store[1] - store[6];
        let x25p = store[2] + store[5];
        let x25n = store[2] - store[5];
        let x34p = store[3] + store[4];
        let x34n = store[3] - store[4];
        let y0 = store[0] + x16p + x25p + x34p;

        let x16re_a = x16p.mul_f64_add(
            self.twiddle1.re,
            x25p.mul_f64_add(
                self.twiddle2.re,
                x34p.mul_f64_add(self.twiddle3.re, store[0]),
            ),
        );
        let x25re_a = x34p.mul_f64_add(
            self.twiddle1.re,
            x16p.mul_f64_add(
                self.twiddle2.re,
                x25p.mul_f64_add(self.twiddle3.re, store[0]),
            ),
        );
        let x34re_a = x25p.mul_f64_add(
            self.twiddle1.re,
            x34p.mul_f64_add(
                self.twiddle2.re,
                x16p.mul_f64_add(self.twiddle3.re, store[0]),
            ),
        );
        let x16im_b = x16n.mul_f64_add(
            self.twiddle1.im,
            x25n.mul_f64_add(self.twiddle2.im, x34n * self.twiddle3.im),
        );
        let x25im_b = x34n.mul_f64_add(
            -self.twiddle1.im,
            x16n.mul_f64_add(self.twiddle2.im, x25n * -self.twiddle3.im),
        );
        let x34im_b = x25n.mul_f64_add(
            self.twiddle1.im,
            x34n.mul_f64_add(-self.twiddle2.im, x16n * -self.twiddle3.im),
        );
        let v0 = y0.zip_complex(NeonStoreD::zero());
        let v1 = x16re_a.zip_complex(x16im_b);
        let v2 = x25re_a.zip_complex(x25im_b);
        let v3 = x34re_a.zip_complex(-x34im_b);
        [[v0[0], v1[0], v2[0], v3[0]], [v0[1], v1[1], v2[1], v3[1]]]
    }
}
