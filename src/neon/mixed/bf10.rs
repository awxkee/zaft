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
use crate::neon::butterflies::{NeonButterfly, NeonFastButterfly5};
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF, NeonStoreFh};
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly10d {
    bf5: NeonFastButterfly5<f64>,
    rotate: float64x2_t,
}

impl ColumnButterfly10d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                bf5: NeonFastButterfly5::new(fft_direction),
                rotate: vld1q_f64([-0.0f64, 0.0, -0.0f64, 0.0].as_ptr().cast()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 10]) -> [NeonStoreD; 10] {
        let mid0 = self.bf5.exec(
            store[0].v,
            store[2].v,
            store[4].v,
            store[6].v,
            store[8].v,
            self.rotate,
        );
        let mid1 = self.bf5.exec(
            store[5].v,
            store[7].v,
            store[9].v,
            store[1].v,
            store[3].v,
            self.rotate,
        );

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) = NeonButterfly::butterfly2_f64(mid0.0, mid1.0);
        let (y2, y3) = NeonButterfly::butterfly2_f64(mid0.1, mid1.1);
        let (y4, y5) = NeonButterfly::butterfly2_f64(mid0.2, mid1.2);
        let (y6, y7) = NeonButterfly::butterfly2_f64(mid0.3, mid1.3);
        let (y8, y9) = NeonButterfly::butterfly2_f64(mid0.4, mid1.4);

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
    bf5: NeonFastButterfly5<f64>,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly10d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            bf5: NeonFastButterfly5::new(fft_direction),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 10]) -> [NeonStoreD; 10] {
        let mid0 = self
            .bf5
            .exec_fcma(store[0].v, store[2].v, store[4].v, store[6].v, store[8].v);
        let mid1 = self
            .bf5
            .exec_fcma(store[5].v, store[7].v, store[9].v, store[1].v, store[3].v);

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) = NeonButterfly::butterfly2_f64(mid0.0, mid1.0);
        let (y2, y3) = NeonButterfly::butterfly2_f64(mid0.1, mid1.1);
        let (y4, y5) = NeonButterfly::butterfly2_f64(mid0.2, mid1.2);
        let (y6, y7) = NeonButterfly::butterfly2_f64(mid0.3, mid1.3);
        let (y8, y9) = NeonButterfly::butterfly2_f64(mid0.4, mid1.4);

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
    bf5: NeonFastButterfly5<f32>,
    rotate: float32x4_t,
}

impl ColumnButterfly10f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                bf5: NeonFastButterfly5::new(fft_direction),
                rotate: vld1q_f32([-0.0f32, 0.0, -0.0f32, 0.0].as_ptr().cast()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 10]) -> [NeonStoreF; 10] {
        let mid0 = self.bf5.exec(
            store[0].v,
            store[2].v,
            store[4].v,
            store[6].v,
            store[8].v,
            self.rotate,
        );
        let mid1 = self.bf5.exec(
            store[5].v,
            store[7].v,
            store[9].v,
            store[1].v,
            store[3].v,
            self.rotate,
        );

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) = NeonButterfly::butterfly2_f32(mid0.0, mid1.0);
        let (y2, y3) = NeonButterfly::butterfly2_f32(mid0.1, mid1.1);
        let (y4, y5) = NeonButterfly::butterfly2_f32(mid0.2, mid1.2);
        let (y6, y7) = NeonButterfly::butterfly2_f32(mid0.3, mid1.3);
        let (y8, y9) = NeonButterfly::butterfly2_f32(mid0.4, mid1.4);

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
            let mid0 = self.bf5.exec(
                vcombine_f32(store[0].v, store[5].v),
                vcombine_f32(store[2].v, store[7].v),
                vcombine_f32(store[4].v, store[9].v),
                vcombine_f32(store[6].v, store[1].v),
                vcombine_f32(store[8].v, store[3].v),
                self.rotate,
            );

            // Since this is good-thomas algorithm, we don't need twiddle factors
            let (y0, y1) =
                NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.0), vget_high_f32(mid0.0));
            let (y2, y3) =
                NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.1), vget_high_f32(mid0.1));
            let (y4, y5) =
                NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.2), vget_high_f32(mid0.2));
            let (y6, y7) =
                NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.3), vget_high_f32(mid0.3));
            let (y8, y9) =
                NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.4), vget_high_f32(mid0.4));

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
    bf5: NeonFastButterfly5<f32>,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly10f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            bf5: NeonFastButterfly5::new(fft_direction),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 10]) -> [NeonStoreF; 10] {
        let mid0 = self
            .bf5
            .exec_fcma(store[0].v, store[2].v, store[4].v, store[6].v, store[8].v);
        let mid1 = self
            .bf5
            .exec_fcma(store[5].v, store[7].v, store[9].v, store[1].v, store[3].v);

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) = NeonButterfly::butterfly2_f32(mid0.0, mid1.0);
        let (y2, y3) = NeonButterfly::butterfly2_f32(mid0.1, mid1.1);
        let (y4, y5) = NeonButterfly::butterfly2_f32(mid0.2, mid1.2);
        let (y6, y7) = NeonButterfly::butterfly2_f32(mid0.3, mid1.3);
        let (y8, y9) = NeonButterfly::butterfly2_f32(mid0.4, mid1.4);

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

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 10]) -> [NeonStoreFh; 10] {
        let mid0 = self.bf5.exec_fcma(
            vcombine_f32(store[0].v, store[5].v),
            vcombine_f32(store[2].v, store[7].v),
            vcombine_f32(store[4].v, store[9].v),
            vcombine_f32(store[6].v, store[1].v),
            vcombine_f32(store[8].v, store[3].v),
        );

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) = NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.0), vget_high_f32(mid0.0));
        let (y2, y3) = NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.1), vget_high_f32(mid0.1));
        let (y4, y5) = NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.2), vget_high_f32(mid0.2));
        let (y6, y7) = NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.3), vget_high_f32(mid0.3));
        let (y8, y9) = NeonButterfly::butterfly2h_f32(vget_low_f32(mid0.4), vget_high_f32(mid0.4));

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
