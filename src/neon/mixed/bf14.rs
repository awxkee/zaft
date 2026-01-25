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
use crate::neon::butterflies::NeonButterfly;
use crate::neon::mixed::neon_store::NeonStoreFh;
use crate::neon::mixed::*;

macro_rules! gen_bf14d {
    ($name: ident, $feature: literal, $bf: ident) => {
        pub(crate) struct $name {
            bf7: $bf,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    bf7: $bf::new(fft_direction),
                }
            }

            #[inline]
            #[target_feature(enable = $feature)]
            pub(crate) fn exec(&self, u: [NeonStoreD; 14]) -> [NeonStoreD; 14] {
                let (u0, u1) = NeonButterfly::butterfly2_f64(u[0].v, u[7].v);
                let (u2, u3) = NeonButterfly::butterfly2_f64(u[8].v, u[1].v);
                let (u4, u5) = NeonButterfly::butterfly2_f64(u[2].v, u[9].v);
                let (u6, u7) = NeonButterfly::butterfly2_f64(u[10].v, u[3].v);
                let (u8, u9) = NeonButterfly::butterfly2_f64(u[4].v, u[11].v);
                let (u10, u11) = NeonButterfly::butterfly2_f64(u[12].v, u[5].v);
                let (u12, u13) = NeonButterfly::butterfly2_f64(u[6].v, u[13].v);

                // Outer 7-point butterflies
                let [y0, y2, y4, y6, y8, y10, y12] = self.bf7.exec([
                    NeonStoreD::raw(u0),
                    NeonStoreD::raw(u2),
                    NeonStoreD::raw(u4),
                    NeonStoreD::raw(u6),
                    NeonStoreD::raw(u8),
                    NeonStoreD::raw(u10),
                    NeonStoreD::raw(u12),
                ]); // (v0, v1, v2, v3, v4, v5, v6)
                let [y7, y9, y11, y13, y1, y3, y5] = self.bf7.exec([
                    NeonStoreD::raw(u1),
                    NeonStoreD::raw(u3),
                    NeonStoreD::raw(u5),
                    NeonStoreD::raw(u7),
                    NeonStoreD::raw(u9),
                    NeonStoreD::raw(u11),
                    NeonStoreD::raw(u13),
                ]); // (v7, v8, v9, v10, v11, v12, v13)

                [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13]
            }
        }
    };
}

gen_bf14d!(ColumnButterfly14d, "neon", ColumnButterfly7d);
#[cfg(feature = "fcma")]
gen_bf14d!(ColumnFcmaButterfly14d, "fcma", ColumnFcmaButterfly7d);

macro_rules! gen_bf14 {
    ($name: ident, $feature: literal, $bf: ident) => {
        pub(crate) struct $name {
            bf7: $bf,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    bf7: $bf::new(fft_direction),
                }
            }

            #[inline]
            #[target_feature(enable = $feature)]
            pub(crate) fn exec(&self, u: [NeonStoreF; 14]) -> [NeonStoreF; 14] {
                let (u0, u1) = NeonButterfly::butterfly2_f32(u[0].v, u[7].v);
                let (u2, u3) = NeonButterfly::butterfly2_f32(u[8].v, u[1].v);
                let (u4, u5) = NeonButterfly::butterfly2_f32(u[2].v, u[9].v);
                let (u6, u7) = NeonButterfly::butterfly2_f32(u[10].v, u[3].v);
                let (u8, u9) = NeonButterfly::butterfly2_f32(u[4].v, u[11].v);
                let (u10, u11) = NeonButterfly::butterfly2_f32(u[12].v, u[5].v);
                let (u12, u13) = NeonButterfly::butterfly2_f32(u[6].v, u[13].v);

                // Outer 7-point butterflies
                let [y0, y2, y4, y6, y8, y10, y12] = self.bf7.exec([
                    NeonStoreF::raw(u0),
                    NeonStoreF::raw(u2),
                    NeonStoreF::raw(u4),
                    NeonStoreF::raw(u6),
                    NeonStoreF::raw(u8),
                    NeonStoreF::raw(u10),
                    NeonStoreF::raw(u12),
                ]); // (v0, v1, v2, v3, v4, v5, v6)
                let [y7, y9, y11, y13, y1, y3, y5] = self.bf7.exec([
                    NeonStoreF::raw(u1),
                    NeonStoreF::raw(u3),
                    NeonStoreF::raw(u5),
                    NeonStoreF::raw(u7),
                    NeonStoreF::raw(u9),
                    NeonStoreF::raw(u11),
                    NeonStoreF::raw(u13),
                ]); // (v7, v8, v9, v10, v11, v12, v13)

                [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13]
            }

            #[inline]
            #[target_feature(enable = $feature)]
            pub(crate) fn exech(&self, u: [NeonStoreFh; 14]) -> [NeonStoreFh; 14] {
                let (u0, u1) = NeonButterfly::butterfly2h_f32(u[0].v, u[7].v);
                let (u2, u3) = NeonButterfly::butterfly2h_f32(u[8].v, u[1].v);
                let (u4, u5) = NeonButterfly::butterfly2h_f32(u[2].v, u[9].v);
                let (u6, u7) = NeonButterfly::butterfly2h_f32(u[10].v, u[3].v);
                let (u8, u9) = NeonButterfly::butterfly2h_f32(u[4].v, u[11].v);
                let (u10, u11) = NeonButterfly::butterfly2h_f32(u[12].v, u[5].v);
                let (u12, u13) = NeonButterfly::butterfly2h_f32(u[6].v, u[13].v);

                // Outer 7-point butterflies
                let [y0, y2, y4, y6, y8, y10, y12] = self.bf7.exech([
                    NeonStoreFh::raw(u0),
                    NeonStoreFh::raw(u2),
                    NeonStoreFh::raw(u4),
                    NeonStoreFh::raw(u6),
                    NeonStoreFh::raw(u8),
                    NeonStoreFh::raw(u10),
                    NeonStoreFh::raw(u12),
                ]); // (v0, v1, v2, v3, v4, v5, v6)
                let [y7, y9, y11, y13, y1, y3, y5] = self.bf7.exech([
                    NeonStoreFh::raw(u1),
                    NeonStoreFh::raw(u3),
                    NeonStoreFh::raw(u5),
                    NeonStoreFh::raw(u7),
                    NeonStoreFh::raw(u9),
                    NeonStoreFh::raw(u11),
                    NeonStoreFh::raw(u13),
                ]); // (v7, v8, v9, v10, v11, v12, v13)

                [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13]
            }
        }
    };
}

gen_bf14!(ColumnButterfly14f, "neon", ColumnButterfly7f);
#[cfg(feature = "fcma")]
gen_bf14!(ColumnFcmaButterfly14f, "fcma", ColumnFcmaButterfly7f);
