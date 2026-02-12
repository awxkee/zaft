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
use crate::neon::mixed::{ColumnButterfly5d, ColumnButterfly5f};
#[cfg(feature = "fcma")]
use crate::neon::mixed::{ColumnFcmaButterfly5d, ColumnFcmaButterfly5f};
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly10d {
    bf5: ColumnButterfly5d,
}

impl ColumnButterfly10d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            bf5: ColumnButterfly5d::new(fft_direction),
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 10]) -> [NeonStoreD; 10] {
        let mid0 = self
            .bf5
            .exec([store[0], store[2], store[4], store[6], store[8]]);
        let mid1 = self
            .bf5
            .exec([store[5], store[7], store[9], store[1], store[3]]);

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) = NeonButterfly::butterfly2_f64(mid0[0].v, mid1[0].v);
        let (y2, y3) = NeonButterfly::butterfly2_f64(mid0[1].v, mid1[1].v);
        let (y4, y5) = NeonButterfly::butterfly2_f64(mid0[2].v, mid1[2].v);
        let (y6, y7) = NeonButterfly::butterfly2_f64(mid0[3].v, mid1[3].v);
        let (y8, y9) = NeonButterfly::butterfly2_f64(mid0[4].v, mid1[4].v);

        [
            NeonStoreD::raw(y0),
            NeonStoreD::raw(y3),
            NeonStoreD::raw(y4),
            NeonStoreD::raw(y7),
            NeonStoreD::raw(y8),
            NeonStoreD::raw(y1),
            NeonStoreD::raw(y2),
            NeonStoreD::raw(y5),
            NeonStoreD::raw(y6),
            NeonStoreD::raw(y9),
        ]
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly10d {
    bf5: ColumnFcmaButterfly5d,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly10d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            bf5: ColumnFcmaButterfly5d::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 10]) -> [NeonStoreD; 10] {
        let mid0 = self
            .bf5
            .exec([store[0], store[2], store[4], store[6], store[8]]);
        let mid1 = self
            .bf5
            .exec([store[5], store[7], store[9], store[1], store[3]]);

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) = NeonButterfly::butterfly2_f64(mid0[0].v, mid1[0].v);
        let (y2, y3) = NeonButterfly::butterfly2_f64(mid0[1].v, mid1[1].v);
        let (y4, y5) = NeonButterfly::butterfly2_f64(mid0[2].v, mid1[2].v);
        let (y6, y7) = NeonButterfly::butterfly2_f64(mid0[3].v, mid1[3].v);
        let (y8, y9) = NeonButterfly::butterfly2_f64(mid0[4].v, mid1[4].v);

        [
            NeonStoreD::raw(y0),
            NeonStoreD::raw(y3),
            NeonStoreD::raw(y4),
            NeonStoreD::raw(y7),
            NeonStoreD::raw(y8),
            NeonStoreD::raw(y1),
            NeonStoreD::raw(y2),
            NeonStoreD::raw(y5),
            NeonStoreD::raw(y6),
            NeonStoreD::raw(y9),
        ]
    }
}

pub(crate) struct ColumnButterfly10f {
    bf5: ColumnButterfly5f,
}

impl ColumnButterfly10f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            bf5: ColumnButterfly5f::new(fft_direction),
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 10]) -> [NeonStoreF; 10] {
        let mid0 = self
            .bf5
            .exec([store[0], store[2], store[4], store[6], store[8]]);
        let mid1 = self
            .bf5
            .exec([store[5], store[7], store[9], store[1], store[3]]);

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) = NeonButterfly::butterfly2_f32(mid0[0].v, mid1[0].v);
        let (y2, y3) = NeonButterfly::butterfly2_f32(mid0[1].v, mid1[1].v);
        let (y4, y5) = NeonButterfly::butterfly2_f32(mid0[2].v, mid1[2].v);
        let (y6, y7) = NeonButterfly::butterfly2_f32(mid0[3].v, mid1[3].v);
        let (y8, y9) = NeonButterfly::butterfly2_f32(mid0[4].v, mid1[4].v);

        [
            NeonStoreF::raw(y0),
            NeonStoreF::raw(y3),
            NeonStoreF::raw(y4),
            NeonStoreF::raw(y7),
            NeonStoreF::raw(y8),
            NeonStoreF::raw(y1),
            NeonStoreF::raw(y2),
            NeonStoreF::raw(y5),
            NeonStoreF::raw(y6),
            NeonStoreF::raw(y9),
        ]
    }

    #[inline(always)]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 10]) -> [NeonStoreFh; 10] {
        unsafe {
            let mid0 = self.bf5.exec([
                NeonStoreF::raw(vcombine_f32(store[0].v, store[5].v)),
                NeonStoreF::raw(vcombine_f32(store[2].v, store[7].v)),
                NeonStoreF::raw(vcombine_f32(store[4].v, store[9].v)),
                NeonStoreF::raw(vcombine_f32(store[6].v, store[1].v)),
                NeonStoreF::raw(vcombine_f32(store[8].v, store[3].v)),
            ]);

            // Since this is good-thomas algorithm, we don't need twiddle factors
            let (y0, y1) =
                NeonButterfly::butterfly2h_f32(vget_low_f32(mid0[0].v), vget_high_f32(mid0[0].v));
            let (y2, y3) =
                NeonButterfly::butterfly2h_f32(vget_low_f32(mid0[1].v), vget_high_f32(mid0[1].v));
            let (y4, y5) =
                NeonButterfly::butterfly2h_f32(vget_low_f32(mid0[2].v), vget_high_f32(mid0[2].v));
            let (y6, y7) =
                NeonButterfly::butterfly2h_f32(vget_low_f32(mid0[3].v), vget_high_f32(mid0[3].v));
            let (y8, y9) =
                NeonButterfly::butterfly2h_f32(vget_low_f32(mid0[4].v), vget_high_f32(mid0[4].v));

            [
                NeonStoreFh::raw(y0),
                NeonStoreFh::raw(y3),
                NeonStoreFh::raw(y4),
                NeonStoreFh::raw(y7),
                NeonStoreFh::raw(y8),
                NeonStoreFh::raw(y1),
                NeonStoreFh::raw(y2),
                NeonStoreFh::raw(y5),
                NeonStoreFh::raw(y6),
                NeonStoreFh::raw(y9),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly10f {
    bf5: ColumnFcmaButterfly5f,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly10f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            bf5: ColumnFcmaButterfly5f::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 10]) -> [NeonStoreF; 10] {
        let mid0 = self
            .bf5
            .exec([store[0], store[2], store[4], store[6], store[8]]);
        let mid1 = self
            .bf5
            .exec([store[5], store[7], store[9], store[1], store[3]]);

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) = NeonButterfly::butterfly2_f32(mid0[0].v, mid1[0].v);
        let (y2, y3) = NeonButterfly::butterfly2_f32(mid0[1].v, mid1[1].v);
        let (y4, y5) = NeonButterfly::butterfly2_f32(mid0[2].v, mid1[2].v);
        let (y6, y7) = NeonButterfly::butterfly2_f32(mid0[3].v, mid1[3].v);
        let (y8, y9) = NeonButterfly::butterfly2_f32(mid0[4].v, mid1[4].v);

        [
            NeonStoreF::raw(y0),
            NeonStoreF::raw(y3),
            NeonStoreF::raw(y4),
            NeonStoreF::raw(y7),
            NeonStoreF::raw(y8),
            NeonStoreF::raw(y1),
            NeonStoreF::raw(y2),
            NeonStoreF::raw(y5),
            NeonStoreF::raw(y6),
            NeonStoreF::raw(y9),
        ]
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 10]) -> [NeonStoreFh; 10] {
        let mid0 = self.bf5.exec([
            NeonStoreF::raw(vcombine_f32(store[0].v, store[5].v)),
            NeonStoreF::raw(vcombine_f32(store[2].v, store[7].v)),
            NeonStoreF::raw(vcombine_f32(store[4].v, store[9].v)),
            NeonStoreF::raw(vcombine_f32(store[6].v, store[1].v)),
            NeonStoreF::raw(vcombine_f32(store[8].v, store[3].v)),
        ]);

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) =
            NeonButterfly::butterfly2h_f32(vget_low_f32(mid0[0].v), vget_high_f32(mid0[0].v));
        let (y2, y3) =
            NeonButterfly::butterfly2h_f32(vget_low_f32(mid0[1].v), vget_high_f32(mid0[1].v));
        let (y4, y5) =
            NeonButterfly::butterfly2h_f32(vget_low_f32(mid0[2].v), vget_high_f32(mid0[2].v));
        let (y6, y7) =
            NeonButterfly::butterfly2h_f32(vget_low_f32(mid0[3].v), vget_high_f32(mid0[3].v));
        let (y8, y9) =
            NeonButterfly::butterfly2h_f32(vget_low_f32(mid0[4].v), vget_high_f32(mid0[4].v));

        [
            NeonStoreFh::raw(y0),
            NeonStoreFh::raw(y3),
            NeonStoreFh::raw(y4),
            NeonStoreFh::raw(y7),
            NeonStoreFh::raw(y8),
            NeonStoreFh::raw(y1),
            NeonStoreFh::raw(y2),
            NeonStoreFh::raw(y5),
            NeonStoreFh::raw(y6),
            NeonStoreFh::raw(y9),
        ]
    }
}
