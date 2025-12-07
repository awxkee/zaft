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
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF, NeonStoreFh};
use crate::neon::util::{v_rotate90_f32, v_rotate90_f64, vh_rotate90_f32};
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly8d {
    rotate: float64x2_t,
    root2: f64,
}

impl ColumnButterfly8d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            static ROT_270: [f64; 2] = [0.0, -0.0];
            let rot_sign = vld1q_f64(match fft_direction {
                FftDirection::Inverse => ROT_90.as_ptr(),
                FftDirection::Forward => ROT_270.as_ptr(),
            });
            Self {
                rotate: rot_sign,
                root2: 0.5f64.sqrt(),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 8]) -> [NeonStoreD; 8] {
        unsafe {
            let (u0, u2, u4, u6) = NeonButterfly::butterfly4_f64(
                store[0].v,
                store[2].v,
                store[4].v,
                store[6].v,
                self.rotate,
            );
            let (u1, mut u3, mut u5, mut u7) = NeonButterfly::butterfly4_f64(
                store[1].v,
                store[3].v,
                store[5].v,
                store[7].v,
                self.rotate,
            );

            u3 = vmulq_n_f64(vaddq_f64(v_rotate90_f64(u3, self.rotate), u3), self.root2);
            u5 = v_rotate90_f64(u5, self.rotate);
            u7 = vmulq_n_f64(vsubq_f64(v_rotate90_f64(u7, self.rotate), u7), self.root2);

            let (y0, y1) = NeonButterfly::butterfly2_f64(u0, u1);
            let (y2, y3) = NeonButterfly::butterfly2_f64(u2, u3);
            let (y4, y5) = NeonButterfly::butterfly2_f64(u4, u5);
            let (y6, y7) = NeonButterfly::butterfly2_f64(u6, u7);
            [
                NeonStoreD::raw(y0),
                NeonStoreD::raw(y2),
                NeonStoreD::raw(y4),
                NeonStoreD::raw(y6),
                NeonStoreD::raw(y1),
                NeonStoreD::raw(y3),
                NeonStoreD::raw(y5),
                NeonStoreD::raw(y7),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly8d {
    root2: f64,
    bf4: FastFcmaBf4d,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly8d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            root2: 0.5f64.sqrt(),
            bf4: FastFcmaBf4d::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 8]) -> [NeonStoreD; 8] {
        let (u0, u2, u4, u6) = self
            .bf4
            .exec(store[0].v, store[2].v, store[4].v, store[6].v);
        let (u1, mut u3, mut u5, mut u7) = self
            .bf4
            .exec(store[1].v, store[3].v, store[5].v, store[7].v);

        u3 = vmulq_n_f64(vcmlaq_rot90_f64(u3, self.bf4.rot_sign, u3), self.root2);
        u5 = vcmlaq_rot90_f64(vdupq_n_f64(0.), self.bf4.rot_sign, u5);
        u7 = vmulq_n_f64(
            vsubq_f64(vcmlaq_rot90_f64(vdupq_n_f64(0.), self.bf4.rot_sign, u7), u7),
            self.root2,
        );

        let (y0, y1) = NeonButterfly::butterfly2_f64(u0, u1);
        let (y2, y3) = NeonButterfly::butterfly2_f64(u2, u3);
        let (y4, y5) = NeonButterfly::butterfly2_f64(u4, u5);
        let (y6, y7) = NeonButterfly::butterfly2_f64(u6, u7);
        [
            NeonStoreD::raw(y0),
            NeonStoreD::raw(y2),
            NeonStoreD::raw(y4),
            NeonStoreD::raw(y6),
            NeonStoreD::raw(y1),
            NeonStoreD::raw(y3),
            NeonStoreD::raw(y5),
            NeonStoreD::raw(y7),
        ]
    }
}

pub(crate) struct ColumnButterfly8f {
    rotate: float32x4_t,
    root2: f32,
}

impl ColumnButterfly8f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            static ROT_90: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            static ROT_270: [f32; 4] = [0.0, -0.0, 0.0, -0.0];
            let rot_sign = vld1q_f32(match fft_direction {
                FftDirection::Inverse => ROT_90.as_ptr(),
                FftDirection::Forward => ROT_270.as_ptr(),
            });
            Self {
                rotate: rot_sign,
                root2: 0.5f32.sqrt(),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn rotate1(&self, v: NeonStoreF) -> NeonStoreF {
        unsafe {
            NeonStoreF::raw(vmulq_n_f32(
                vaddq_f32(v_rotate90_f32(v.v, self.rotate), v.v),
                self.root2,
            ))
        }
    }

    #[inline(always)]
    pub(crate) fn rotate3(&self, v: NeonStoreF) -> NeonStoreF {
        unsafe {
            NeonStoreF::raw(vmulq_n_f32(
                vsubq_f32(v_rotate90_f32(v.v, self.rotate), v.v),
                self.root2,
            ))
        }
    }

    #[inline(always)]
    pub(crate) fn rotate(&self, v: NeonStoreF) -> NeonStoreF {
        NeonStoreF::raw(v_rotate90_f32(v.v, self.rotate))
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 8]) -> [NeonStoreF; 8] {
        unsafe {
            let (u0, u2, u4, u6) = NeonButterfly::butterfly4_f32(
                store[0].v,
                store[2].v,
                store[4].v,
                store[6].v,
                self.rotate,
            );
            let (u1, mut u3, mut u5, mut u7) = NeonButterfly::butterfly4_f32(
                store[1].v,
                store[3].v,
                store[5].v,
                store[7].v,
                self.rotate,
            );

            u3 = vmulq_n_f32(vaddq_f32(v_rotate90_f32(u3, self.rotate), u3), self.root2);
            u5 = v_rotate90_f32(u5, self.rotate);
            u7 = vmulq_n_f32(vsubq_f32(v_rotate90_f32(u7, self.rotate), u7), self.root2);

            let (zy0, zy1) = NeonButterfly::butterfly2_f32(u0, u1);
            let (zy2, zy3) = NeonButterfly::butterfly2_f32(u2, u3);
            let (zy4, zy5) = NeonButterfly::butterfly2_f32(u4, u5);
            let (zy6, zy7) = NeonButterfly::butterfly2_f32(u6, u7);

            [
                NeonStoreF::raw(zy0),
                NeonStoreF::raw(zy2),
                NeonStoreF::raw(zy4),
                NeonStoreF::raw(zy6),
                NeonStoreF::raw(zy1),
                NeonStoreF::raw(zy3),
                NeonStoreF::raw(zy5),
                NeonStoreF::raw(zy7),
            ]
        }
    }

    #[inline(always)]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 8]) -> [NeonStoreFh; 8] {
        unsafe {
            let (u0, u2, u4, u6) = NeonButterfly::butterfly4h_f32(
                store[0].v,
                store[2].v,
                store[4].v,
                store[6].v,
                vget_low_f32(self.rotate),
            );
            let (u1, mut u3, mut u5, mut u7) = NeonButterfly::butterfly4h_f32(
                store[1].v,
                store[3].v,
                store[5].v,
                store[7].v,
                vget_low_f32(self.rotate),
            );

            u3 = vmul_n_f32(
                vadd_f32(vh_rotate90_f32(u3, vget_low_f32(self.rotate)), u3),
                self.root2,
            );
            u5 = vh_rotate90_f32(u5, vget_low_f32(self.rotate));
            u7 = vmul_n_f32(
                vsub_f32(vh_rotate90_f32(u7, vget_low_f32(self.rotate)), u7),
                self.root2,
            );

            let (zy0, zy1) = NeonButterfly::butterfly2h_f32(u0, u1);
            let (zy2, zy3) = NeonButterfly::butterfly2h_f32(u2, u3);
            let (zy4, zy5) = NeonButterfly::butterfly2h_f32(u4, u5);
            let (zy6, zy7) = NeonButterfly::butterfly2h_f32(u6, u7);

            [
                NeonStoreFh::raw(zy0),
                NeonStoreFh::raw(zy2),
                NeonStoreFh::raw(zy4),
                NeonStoreFh::raw(zy6),
                NeonStoreFh::raw(zy1),
                NeonStoreFh::raw(zy3),
                NeonStoreFh::raw(zy5),
                NeonStoreFh::raw(zy7),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly8f {
    root2: f32,
    bf4: FastFcmaBf4f,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly8f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            root2: 0.5f32.sqrt(),
            bf4: FastFcmaBf4f::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn rotate1(&self, v: NeonStoreF) -> NeonStoreF {
        NeonStoreF::raw(vmulq_n_f32(
            vcmlaq_rot90_f32(v.v, self.bf4.rot_sign, v.v),
            self.root2,
        ))
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn rotate3(&self, v: NeonStoreF) -> NeonStoreF {
        NeonStoreF::raw(vmulq_n_f32(
            vsubq_f32(
                vcmlaq_rot90_f32(vdupq_n_f32(0.), self.bf4.rot_sign, v.v),
                v.v,
            ),
            self.root2,
        ))
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn rotate(&self, v: NeonStoreF) -> NeonStoreF {
        NeonStoreF::raw(vcmlaq_rot90_f32(vdupq_n_f32(0.), self.bf4.rot_sign, v.v))
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 8]) -> [NeonStoreF; 8] {
        let (u0, u2, u4, u6) = self
            .bf4
            .exec(store[0].v, store[2].v, store[4].v, store[6].v);
        let (u1, mut u3, mut u5, mut u7) = self
            .bf4
            .exec(store[1].v, store[3].v, store[5].v, store[7].v);

        u3 = vmulq_n_f32(vcmlaq_rot90_f32(u3, self.bf4.rot_sign, u3), self.root2);
        u5 = vcmlaq_rot90_f32(vdupq_n_f32(0.), self.bf4.rot_sign, u5);
        u7 = vmulq_n_f32(
            vsubq_f32(vcmlaq_rot90_f32(vdupq_n_f32(0.), self.bf4.rot_sign, u7), u7),
            self.root2,
        );

        let (zy0, zy1) = NeonButterfly::butterfly2_f32(u0, u1);
        let (zy2, zy3) = NeonButterfly::butterfly2_f32(u2, u3);
        let (zy4, zy5) = NeonButterfly::butterfly2_f32(u4, u5);
        let (zy6, zy7) = NeonButterfly::butterfly2_f32(u6, u7);

        [
            NeonStoreF::raw(zy0),
            NeonStoreF::raw(zy2),
            NeonStoreF::raw(zy4),
            NeonStoreF::raw(zy6),
            NeonStoreF::raw(zy1),
            NeonStoreF::raw(zy3),
            NeonStoreF::raw(zy5),
            NeonStoreF::raw(zy7),
        ]
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 8]) -> [NeonStoreFh; 8] {
        let (u0, u2, u4, u6) = self
            .bf4
            .exech(store[0].v, store[2].v, store[4].v, store[6].v);
        let (u1, mut u3, mut u5, mut u7) = self
            .bf4
            .exech(store[1].v, store[3].v, store[5].v, store[7].v);

        u3 = vmul_n_f32(
            vcmla_rot90_f32(u3, vget_low_f32(self.bf4.rot_sign), u3),
            self.root2,
        );
        u5 = vcmla_rot90_f32(vdup_n_f32(0.), vget_low_f32(self.bf4.rot_sign), u5);
        u7 = vmul_n_f32(
            vsub_f32(
                vcmla_rot90_f32(vdup_n_f32(0.), vget_low_f32(self.bf4.rot_sign), u7),
                u7,
            ),
            self.root2,
        );

        let (zy0, zy1) = NeonButterfly::butterfly2h_f32(u0, u1);
        let (zy2, zy3) = NeonButterfly::butterfly2h_f32(u2, u3);
        let (zy4, zy5) = NeonButterfly::butterfly2h_f32(u4, u5);
        let (zy6, zy7) = NeonButterfly::butterfly2h_f32(u6, u7);

        [
            NeonStoreFh::raw(zy0),
            NeonStoreFh::raw(zy2),
            NeonStoreFh::raw(zy4),
            NeonStoreFh::raw(zy6),
            NeonStoreFh::raw(zy1),
            NeonStoreFh::raw(zy3),
            NeonStoreFh::raw(zy5),
            NeonStoreFh::raw(zy7),
        ]
    }
}
