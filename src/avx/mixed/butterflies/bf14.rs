/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
use crate::avx::butterflies::AvxButterfly;
use crate::avx::mixed::{AvxStoreD, AvxStoreF, ColumnButterfly7d, ColumnButterfly7f};

pub(crate) struct ColumnButterfly14d {
    bf7: ColumnButterfly7d,
}

impl ColumnButterfly14d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> Self {
        Self {
            bf7: ColumnButterfly7d::new(direction),
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, u: [AvxStoreD; 14]) -> [AvxStoreD; 14] {
        unsafe {
            let (u0, u1) = AvxButterfly::butterfly2_f64(u[0].v, u[7].v);
            let (u2, u3) = AvxButterfly::butterfly2_f64(u[8].v, u[1].v);
            let (u4, u5) = AvxButterfly::butterfly2_f64(u[2].v, u[9].v);
            let (u6, u7) = AvxButterfly::butterfly2_f64(u[10].v, u[3].v);
            let (u8, u9) = AvxButterfly::butterfly2_f64(u[4].v, u[11].v);
            let (u10, u11) = AvxButterfly::butterfly2_f64(u[12].v, u[5].v);
            let (u12, u13) = AvxButterfly::butterfly2_f64(u[6].v, u[13].v);

            // Outer 7-point butterflies
            let [y0, y2, y4, y6, y8, y10, y12] = self.bf7.exec([
                AvxStoreD::raw(u0),
                AvxStoreD::raw(u2),
                AvxStoreD::raw(u4),
                AvxStoreD::raw(u6),
                AvxStoreD::raw(u8),
                AvxStoreD::raw(u10),
                AvxStoreD::raw(u12),
            ]); // (v0, v1, v2, v3, v4, v5, v6)
            let [y7, y9, y11, y13, y1, y3, y5] = self.bf7.exec([
                AvxStoreD::raw(u1),
                AvxStoreD::raw(u3),
                AvxStoreD::raw(u5),
                AvxStoreD::raw(u7),
                AvxStoreD::raw(u9),
                AvxStoreD::raw(u11),
                AvxStoreD::raw(u13),
            ]); // (v7, v8, v9, v10, v11, v12, v13)

            [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13]
        }
    }
}

pub(crate) struct ColumnButterfly14f {
    bf7: ColumnButterfly7f,
}

impl ColumnButterfly14f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> Self {
        Self {
            bf7: ColumnButterfly7f::new(direction),
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, u: [AvxStoreF; 14]) -> [AvxStoreF; 14] {
        let (u0, u1) = AvxButterfly::butterfly2_f32(u[0].v, u[7].v);
        let (u2, u3) = AvxButterfly::butterfly2_f32(u[8].v, u[1].v);
        let (u4, u5) = AvxButterfly::butterfly2_f32(u[2].v, u[9].v);
        let (u6, u7) = AvxButterfly::butterfly2_f32(u[10].v, u[3].v);
        let (u8, u9) = AvxButterfly::butterfly2_f32(u[4].v, u[11].v);
        let (u10, u11) = AvxButterfly::butterfly2_f32(u[12].v, u[5].v);
        let (u12, u13) = AvxButterfly::butterfly2_f32(u[6].v, u[13].v);

        // Outer 7-point butterflies
        let [y0, y2, y4, y6, y8, y10, y12] = self.bf7.exec([
            AvxStoreF::raw(u0),
            AvxStoreF::raw(u2),
            AvxStoreF::raw(u4),
            AvxStoreF::raw(u6),
            AvxStoreF::raw(u8),
            AvxStoreF::raw(u10),
            AvxStoreF::raw(u12),
        ]); // (v0, v1, v2, v3, v4, v5, v6)
        let [y7, y9, y11, y13, y1, y3, y5] = self.bf7.exec([
            AvxStoreF::raw(u1),
            AvxStoreF::raw(u3),
            AvxStoreF::raw(u5),
            AvxStoreF::raw(u7),
            AvxStoreF::raw(u9),
            AvxStoreF::raw(u11),
            AvxStoreF::raw(u13),
        ]); // (v7, v8, v9, v10, v11, v12, v13)

        [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13]
    }
}
