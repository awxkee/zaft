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
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly4d {
    pub(crate) rotate: float64x2_t,
}

impl ColumnButterfly4d {
    #[inline]
    pub(crate) fn new(direction: FftDirection) -> Self {
        unsafe {
            Self {
                rotate: vld1q_f64(match direction {
                    FftDirection::Inverse => [-0.0f64, 0.0].as_ptr().cast(),
                    FftDirection::Forward => [0.0f64, -0.0].as_ptr().cast(),
                }),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 4]) -> [NeonStoreD; 4] {
        unsafe {
            let t0 = vaddq_f64(store[0].v, store[2].v);
            let t1 = vsubq_f64(store[0].v, store[2].v);
            let t2 = vaddq_f64(store[1].v, store[3].v);
            let mut t3 = vsubq_f64(store[1].v, store[3].v);
            t3 = vreinterpretq_f64_u64(veorq_u64(
                vreinterpretq_u64_f64(vextq_f64::<1>(t3, t3)),
                vreinterpretq_u64_f64(self.rotate),
            ));
            [
                NeonStoreD::raw(vaddq_f64(t0, t2)),
                NeonStoreD::raw(vaddq_f64(t1, t3)),
                NeonStoreD::raw(vsubq_f64(t0, t2)),
                NeonStoreD::raw(vsubq_f64(t1, t3)),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly4d {
    pub(crate) rot_sign: float64x2_t,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly4d {
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
    pub(crate) fn exec(&self, store: [NeonStoreD; 4]) -> [NeonStoreD; 4] {
        let t0 = vaddq_f64(store[0].v, store[2].v);
        let t1 = vsubq_f64(store[0].v, store[2].v);
        let t2 = vaddq_f64(store[1].v, store[3].v);
        let t3 = vsubq_f64(store[1].v, store[3].v);
        [
            NeonStoreD::raw(vaddq_f64(t0, t2)),
            NeonStoreD::raw(vcmlaq_rot90_f64(t1, self.rot_sign, t3)),
            NeonStoreD::raw(vsubq_f64(t0, t2)),
            NeonStoreD::raw(vcmlaq_rot270_f64(t1, self.rot_sign, t3)),
        ]
    }
}

pub(crate) struct ColumnButterfly4f {
    pub(crate) rotate: float32x4_t,
}

impl ColumnButterfly4f {
    pub(crate) fn new(direction: FftDirection) -> Self {
        unsafe {
            Self {
                rotate: vld1q_f32(match direction {
                    FftDirection::Inverse => [-0.0f32, 0.0, -0.0f32, 0.0].as_ptr().cast(),
                    FftDirection::Forward => [0.0f32, -0.0, 0.0f32, -0.0].as_ptr().cast(),
                }),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 4]) -> [NeonStoreF; 4] {
        unsafe {
            let t0 = vaddq_f32(store[0].v, store[2].v);
            let t1 = vsubq_f32(store[0].v, store[2].v);
            let t2 = vaddq_f32(store[1].v, store[3].v);
            let mut t3 = vsubq_f32(store[1].v, store[3].v);
            t3 = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(vrev64q_f32(t3)),
                vreinterpretq_u32_f32(self.rotate),
            ));
            [
                NeonStoreF::raw(vaddq_f32(t0, t2)),
                NeonStoreF::raw(vaddq_f32(t1, t3)),
                NeonStoreF::raw(vsubq_f32(t0, t2)),
                NeonStoreF::raw(vsubq_f32(t1, t3)),
            ]
        }
    }

    #[inline(always)]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 4]) -> [NeonStoreFh; 4] {
        unsafe {
            let t0 = vadd_f32(store[0].v, store[2].v);
            let t1 = vsub_f32(store[0].v, store[2].v);
            let t2 = vadd_f32(store[1].v, store[3].v);
            let mut t3 = vsub_f32(store[1].v, store[3].v);
            t3 = vreinterpret_f32_u32(veor_u32(
                vreinterpret_u32_f32(vext_f32::<1>(t3, t3)),
                vreinterpret_u32_f32(vget_low_f32(self.rotate)),
            ));
            [
                NeonStoreFh::raw(vadd_f32(t0, t2)),
                NeonStoreFh::raw(vadd_f32(t1, t3)),
                NeonStoreFh::raw(vsub_f32(t0, t2)),
                NeonStoreFh::raw(vsub_f32(t1, t3)),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly4f {
    pub(crate) rot_sign: float32x4_t,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly4f {
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
    pub(crate) fn exec(&self, store: [NeonStoreF; 4]) -> [NeonStoreF; 4] {
        let t0 = vaddq_f32(store[0].v, store[2].v);
        let t1 = vsubq_f32(store[0].v, store[2].v);
        let t2 = vaddq_f32(store[1].v, store[3].v);
        let t3 = vsubq_f32(store[1].v, store[3].v);
        [
            NeonStoreF::raw(vaddq_f32(t0, t2)),
            NeonStoreF::raw(vcmlaq_rot90_f32(t1, self.rot_sign, t3)),
            NeonStoreF::raw(vsubq_f32(t0, t2)),
            NeonStoreF::raw(vcmlaq_rot270_f32(t1, self.rot_sign, t3)),
        ]
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 4]) -> [NeonStoreFh; 4] {
        let t0 = vadd_f32(store[0].v, store[2].v);
        let t1 = vsub_f32(store[0].v, store[2].v);
        let t2 = vadd_f32(store[1].v, store[3].v);
        let t3 = vsub_f32(store[1].v, store[3].v);
        [
            NeonStoreFh::raw(vadd_f32(t0, t2)),
            NeonStoreFh::raw(vcmla_rot90_f32(t1, vget_low_f32(self.rot_sign), t3)),
            NeonStoreFh::raw(vsub_f32(t0, t2)),
            NeonStoreFh::raw(vcmla_rot270_f32(t1, vget_low_f32(self.rot_sign), t3)),
        ]
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaForwardButterfly4f {}

#[cfg(feature = "fcma")]
impl ColumnFcmaForwardButterfly4f {
    #[inline]
    pub(crate) fn new(_: FftDirection) -> Self {
        Self {}
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 4]) -> [NeonStoreF; 4] {
        let t0 = vaddq_f32(store[0].v, store[2].v);
        let t1 = vsubq_f32(store[0].v, store[2].v);
        let t2 = vaddq_f32(store[1].v, store[3].v);
        let t3 = vsubq_f32(store[1].v, store[3].v);
        [
            NeonStoreF::raw(vaddq_f32(t0, t2)),
            NeonStoreF::raw(vcaddq_rot270_f32(t1, t3)),
            NeonStoreF::raw(vsubq_f32(t0, t2)),
            NeonStoreF::raw(vcaddq_rot90_f32(t1, t3)),
        ]
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 4]) -> [NeonStoreFh; 4] {
        let t0 = vadd_f32(store[0].v, store[2].v);
        let t1 = vsub_f32(store[0].v, store[2].v);
        let t2 = vadd_f32(store[1].v, store[3].v);
        let t3 = vsub_f32(store[1].v, store[3].v);
        [
            NeonStoreFh::raw(vadd_f32(t0, t2)),
            NeonStoreFh::raw(vcadd_rot270_f32(t1, t3)),
            NeonStoreFh::raw(vsub_f32(t0, t2)),
            NeonStoreFh::raw(vcadd_rot90_f32(t1, t3)),
        ]
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaInverseButterfly4f {}

#[cfg(feature = "fcma")]
impl ColumnFcmaInverseButterfly4f {
    #[inline]
    pub(crate) fn new(_: FftDirection) -> Self {
        Self {}
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 4]) -> [NeonStoreF; 4] {
        let t0 = vaddq_f32(store[0].v, store[2].v);
        let t1 = vsubq_f32(store[0].v, store[2].v);
        let t2 = vaddq_f32(store[1].v, store[3].v);
        let t3 = vsubq_f32(store[1].v, store[3].v);
        [
            NeonStoreF::raw(vaddq_f32(t0, t2)),
            NeonStoreF::raw(vcaddq_rot90_f32(t1, t3)),
            NeonStoreF::raw(vsubq_f32(t0, t2)),
            NeonStoreF::raw(vcaddq_rot270_f32(t1, t3)),
        ]
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 4]) -> [NeonStoreFh; 4] {
        let t0 = vadd_f32(store[0].v, store[2].v);
        let t1 = vsub_f32(store[0].v, store[2].v);
        let t2 = vadd_f32(store[1].v, store[3].v);
        let t3 = vsub_f32(store[1].v, store[3].v);
        [
            NeonStoreFh::raw(vadd_f32(t0, t2)),
            NeonStoreFh::raw(vcadd_rot90_f32(t1, t3)),
            NeonStoreFh::raw(vsub_f32(t0, t2)),
            NeonStoreFh::raw(vcadd_rot270_f32(t1, t3)),
        ]
    }
}
