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
use crate::neon::mixed::{ColumnButterfly4d, ColumnButterfly4f};
#[cfg(feature = "fcma")]
use crate::neon::mixed::{ColumnFcmaButterfly4d, ColumnFcmaButterfly4f};
use crate::neon::util::{v_rotate90_f32, v_rotate90_f64, vh_rotate90_f32};
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly8d {
    pub(crate) bf4: ColumnButterfly4d,
    root2: f64,
}

impl ColumnButterfly8d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            bf4: ColumnButterfly4d::new(fft_direction),
            root2: 0.5f64.sqrt(),
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 8]) -> [NeonStoreD; 8] {
        unsafe {
            let [u0, u2, u4, u6] = self.bf4.exec([store[0], store[2], store[4], store[6]]);
            let [u1, u3, u5, u7] = self.bf4.exec([store[1], store[3], store[5], store[7]]);

            let u3 = vmulq_n_f64(
                vaddq_f64(v_rotate90_f64(u3.v, self.bf4.rotate), u3.v),
                self.root2,
            );
            let u5 = v_rotate90_f64(u5.v, self.bf4.rotate);
            let u7 = vmulq_n_f64(
                vsubq_f64(v_rotate90_f64(u7.v, self.bf4.rotate), u7.v),
                self.root2,
            );

            let (y0, y1) = NeonButterfly::butterfly2_f64(u0.v, u1.v);
            let (y2, y3) = NeonButterfly::butterfly2_f64(u2.v, u3);
            let (y4, y5) = NeonButterfly::butterfly2_f64(u4.v, u5);
            let (y6, y7) = NeonButterfly::butterfly2_f64(u6.v, u7);
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
    pub(crate) bf4: ColumnFcmaButterfly4d,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly8d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            root2: 0.5f64.sqrt(),
            bf4: ColumnFcmaButterfly4d::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 8]) -> [NeonStoreD; 8] {
        let [u0, u2, u4, u6] = self.bf4.exec([store[0], store[2], store[4], store[6]]);
        let [u1, u3, u5, u7] = self.bf4.exec([store[1], store[3], store[5], store[7]]);

        let u3 = vmulq_n_f64(vcmlaq_rot90_f64(u3.v, self.bf4.rot_sign, u3.v), self.root2);
        let u5 = vcmlaq_rot90_f64(vdupq_n_f64(0.), self.bf4.rot_sign, u5.v);
        let u7 = vmulq_n_f64(
            vsubq_f64(
                vcmlaq_rot90_f64(vdupq_n_f64(0.), self.bf4.rot_sign, u7.v),
                u7.v,
            ),
            self.root2,
        );

        let (y0, y1) = NeonButterfly::butterfly2_f64(u0.v, u1.v);
        let (y2, y3) = NeonButterfly::butterfly2_f64(u2.v, u3);
        let (y4, y5) = NeonButterfly::butterfly2_f64(u4.v, u5);
        let (y6, y7) = NeonButterfly::butterfly2_f64(u6.v, u7);
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
    pub(crate) bf4: ColumnButterfly4f,
    root2: f32,
}

impl ColumnButterfly8f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            root2: 0.5f32.sqrt(),
            bf4: ColumnButterfly4f::new(fft_direction),
        }
    }

    #[inline(always)]
    pub(crate) fn rotate1(&self, v: NeonStoreF) -> NeonStoreF {
        unsafe {
            NeonStoreF::raw(vmulq_n_f32(
                vaddq_f32(v_rotate90_f32(v.v, self.bf4.rotate), v.v),
                self.root2,
            ))
        }
    }

    #[inline(always)]
    pub(crate) fn rotate3(&self, v: NeonStoreF) -> NeonStoreF {
        unsafe {
            NeonStoreF::raw(vmulq_n_f32(
                vsubq_f32(v_rotate90_f32(v.v, self.bf4.rotate), v.v),
                self.root2,
            ))
        }
    }

    #[inline(always)]
    pub(crate) fn rotate(&self, v: NeonStoreF) -> NeonStoreF {
        NeonStoreF::raw(v_rotate90_f32(v.v, self.bf4.rotate))
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 8]) -> [NeonStoreF; 8] {
        unsafe {
            let [u0, u2, u4, u6] = self.bf4.exec([store[0], store[2], store[4], store[6]]);
            let [u1, u3, u5, u7] = self.bf4.exec([store[1], store[3], store[5], store[7]]);

            let u3 = vmulq_n_f32(
                vaddq_f32(v_rotate90_f32(u3.v, self.bf4.rotate), u3.v),
                self.root2,
            );
            let u5 = v_rotate90_f32(u5.v, self.bf4.rotate);
            let u7 = vmulq_n_f32(
                vsubq_f32(v_rotate90_f32(u7.v, self.bf4.rotate), u7.v),
                self.root2,
            );

            let (zy0, zy1) = NeonButterfly::butterfly2_f32(u0.v, u1.v);
            let (zy2, zy3) = NeonButterfly::butterfly2_f32(u2.v, u3);
            let (zy4, zy5) = NeonButterfly::butterfly2_f32(u4.v, u5);
            let (zy6, zy7) = NeonButterfly::butterfly2_f32(u6.v, u7);

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
            let [u0, u2, u4, u6] = self.bf4.exech([store[0], store[2], store[4], store[6]]);
            let [u1, u3, u5, u7] = self.bf4.exech([store[1], store[3], store[5], store[7]]);

            let u3 = vmul_n_f32(
                vadd_f32(vh_rotate90_f32(u3.v, vget_low_f32(self.bf4.rotate)), u3.v),
                self.root2,
            );
            let u5 = vh_rotate90_f32(u5.v, vget_low_f32(self.bf4.rotate));
            let u7 = vmul_n_f32(
                vsub_f32(vh_rotate90_f32(u7.v, vget_low_f32(self.bf4.rotate)), u7.v),
                self.root2,
            );

            let (zy0, zy1) = NeonButterfly::butterfly2h_f32(u0.v, u1.v);
            let (zy2, zy3) = NeonButterfly::butterfly2h_f32(u2.v, u3);
            let (zy4, zy5) = NeonButterfly::butterfly2h_f32(u4.v, u5);
            let (zy6, zy7) = NeonButterfly::butterfly2h_f32(u6.v, u7);

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
    pub(crate) bf4: ColumnFcmaButterfly4f,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly8f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            root2: 0.5f32.sqrt(),
            bf4: ColumnFcmaButterfly4f::new(fft_direction),
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
        let [u0, u2, u4, u6] = self.bf4.exec([store[0], store[2], store[4], store[6]]);
        let [u1, u3, u5, u7] = self.bf4.exec([store[1], store[3], store[5], store[7]]);

        let u3 = vmulq_n_f32(vcmlaq_rot90_f32(u3.v, self.bf4.rot_sign, u3.v), self.root2);
        let u5 = vcmlaq_rot90_f32(vdupq_n_f32(0.), self.bf4.rot_sign, u5.v);
        let u7 = vmulq_n_f32(
            vsubq_f32(
                vcmlaq_rot90_f32(vdupq_n_f32(0.), self.bf4.rot_sign, u7.v),
                u7.v,
            ),
            self.root2,
        );

        let (zy0, zy1) = NeonButterfly::butterfly2_f32(u0.v, u1.v);
        let (zy2, zy3) = NeonButterfly::butterfly2_f32(u2.v, u3);
        let (zy4, zy5) = NeonButterfly::butterfly2_f32(u4.v, u5);
        let (zy6, zy7) = NeonButterfly::butterfly2_f32(u6.v, u7);

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
        let [u0, u2, u4, u6] = self.bf4.exech([store[0], store[2], store[4], store[6]]);
        let [u1, u3, u5, u7] = self.bf4.exech([store[1], store[3], store[5], store[7]]);

        let u3 = vmul_n_f32(
            vcmla_rot90_f32(u3.v, vget_low_f32(self.bf4.rot_sign), u3.v),
            self.root2,
        );
        let u5 = vcmla_rot90_f32(vdup_n_f32(0.), vget_low_f32(self.bf4.rot_sign), u5.v);
        let u7 = vmul_n_f32(
            vsub_f32(
                vcmla_rot90_f32(vdup_n_f32(0.), vget_low_f32(self.bf4.rot_sign), u7.v),
                u7.v,
            ),
            self.root2,
        );

        let (zy0, zy1) = NeonButterfly::butterfly2h_f32(u0.v, u1.v);
        let (zy2, zy3) = NeonButterfly::butterfly2h_f32(u2.v, u3);
        let (zy4, zy5) = NeonButterfly::butterfly2h_f32(u4.v, u5);
        let (zy6, zy7) = NeonButterfly::butterfly2h_f32(u6.v, u7);

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

#[cfg(feature = "fcma")]
use crate::neon::mixed::bf4::{ColumnFcmaForwardButterfly4f, ColumnFcmaInverseButterfly4f};

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaForwardButterfly8f {
    root2: f32,
    pub(crate) bf4: ColumnFcmaForwardButterfly4f,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaForwardButterfly8f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            root2: 0.5f32.sqrt(),
            bf4: ColumnFcmaForwardButterfly4f::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn rotate1(&self, v: NeonStoreF) -> NeonStoreF {
        NeonStoreF::raw(vmulq_n_f32(vcaddq_rot270_f32(v.v, v.v), self.root2))
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn rotate3(&self, v: NeonStoreF) -> NeonStoreF {
        NeonStoreF::raw(vmulq_n_f32(
            vsubq_f32(vcaddq_rot270_f32(vdupq_n_f32(0.), v.v), v.v),
            self.root2,
        ))
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn rotate(&self, v: NeonStoreF) -> NeonStoreF {
        NeonStoreF::raw(vcaddq_rot270_f32(vdupq_n_f32(0.), v.v))
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 8]) -> [NeonStoreF; 8] {
        let [u0, u2, u4, u6] = self.bf4.exec([store[0], store[2], store[4], store[6]]);
        let [u1, u3, u5, u7] = self.bf4.exec([store[1], store[3], store[5], store[7]]);

        let u3 = vmulq_n_f32(vcaddq_rot270_f32(u3.v, u3.v), self.root2);
        let u5 = vcaddq_rot270_f32(vdupq_n_f32(0.), u5.v);
        let u7 = vmulq_n_f32(
            vsubq_f32(vcaddq_rot270_f32(vdupq_n_f32(0.), u7.v), u7.v),
            self.root2,
        );

        let (zy0, zy1) = NeonButterfly::butterfly2_f32(u0.v, u1.v);
        let (zy2, zy3) = NeonButterfly::butterfly2_f32(u2.v, u3);
        let (zy4, zy5) = NeonButterfly::butterfly2_f32(u4.v, u5);
        let (zy6, zy7) = NeonButterfly::butterfly2_f32(u6.v, u7);

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
        let [u0, u2, u4, u6] = self.bf4.exech([store[0], store[2], store[4], store[6]]);
        let [u1, u3, u5, u7] = self.bf4.exech([store[1], store[3], store[5], store[7]]);

        let u3 = vmul_n_f32(vcadd_rot270_f32(u3.v, u3.v), self.root2);
        let u5 = vcadd_rot270_f32(vdup_n_f32(0.), u5.v);
        let u7 = vmul_n_f32(
            vsub_f32(vcadd_rot270_f32(vdup_n_f32(0.), u7.v), u7.v),
            self.root2,
        );

        let (zy0, zy1) = NeonButterfly::butterfly2h_f32(u0.v, u1.v);
        let (zy2, zy3) = NeonButterfly::butterfly2h_f32(u2.v, u3);
        let (zy4, zy5) = NeonButterfly::butterfly2h_f32(u4.v, u5);
        let (zy6, zy7) = NeonButterfly::butterfly2h_f32(u6.v, u7);

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

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaInverseButterfly8f {
    root2: f32,
    pub(crate) bf4: ColumnFcmaInverseButterfly4f,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaInverseButterfly8f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            root2: 0.5f32.sqrt(),
            bf4: ColumnFcmaInverseButterfly4f::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn rotate1(&self, v: NeonStoreF) -> NeonStoreF {
        NeonStoreF::raw(vmulq_n_f32(vcaddq_rot90_f32(v.v, v.v), self.root2))
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn rotate3(&self, v: NeonStoreF) -> NeonStoreF {
        NeonStoreF::raw(vmulq_n_f32(
            vsubq_f32(vcaddq_rot90_f32(vdupq_n_f32(0.), v.v), v.v),
            self.root2,
        ))
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn rotate(&self, v: NeonStoreF) -> NeonStoreF {
        NeonStoreF::raw(vcaddq_rot90_f32(vdupq_n_f32(0.), v.v))
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 8]) -> [NeonStoreF; 8] {
        let [u0, u2, u4, u6] = self.bf4.exec([store[0], store[2], store[4], store[6]]);
        let [u1, u3, u5, u7] = self.bf4.exec([store[1], store[3], store[5], store[7]]);

        let u3 = vmulq_n_f32(vcaddq_rot90_f32(u3.v, u3.v), self.root2);
        let u5 = vcaddq_rot90_f32(vdupq_n_f32(0.), u5.v);
        let u7 = vmulq_n_f32(
            vsubq_f32(vcaddq_rot90_f32(vdupq_n_f32(0.), u7.v), u7.v),
            self.root2,
        );

        let (zy0, zy1) = NeonButterfly::butterfly2_f32(u0.v, u1.v);
        let (zy2, zy3) = NeonButterfly::butterfly2_f32(u2.v, u3);
        let (zy4, zy5) = NeonButterfly::butterfly2_f32(u4.v, u5);
        let (zy6, zy7) = NeonButterfly::butterfly2_f32(u6.v, u7);

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
        let [u0, u2, u4, u6] = self.bf4.exech([store[0], store[2], store[4], store[6]]);
        let [u1, u3, u5, u7] = self.bf4.exech([store[1], store[3], store[5], store[7]]);

        let u3 = vmul_n_f32(vcadd_rot90_f32(u3.v, u3.v), self.root2);
        let u5 = vcadd_rot90_f32(vdup_n_f32(0.), u5.v);
        let u7 = vmul_n_f32(
            vsub_f32(vcadd_rot90_f32(vdup_n_f32(0.), u7.v), u7.v),
            self.root2,
        );

        let (zy0, zy1) = NeonButterfly::butterfly2h_f32(u0.v, u1.v);
        let (zy2, zy3) = NeonButterfly::butterfly2h_f32(u2.v, u3);
        let (zy4, zy5) = NeonButterfly::butterfly2h_f32(u4.v, u5);
        let (zy6, zy7) = NeonButterfly::butterfly2h_f32(u6.v, u7);

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
