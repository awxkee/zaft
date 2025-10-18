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
use crate::avx::butterflies::{AvxFastButterfly3, AvxFastButterfly4};
use crate::avx::mixed::avx_stored::AvxStoreD;
use crate::avx::mixed::avx_storef::AvxStoreF;

pub(crate) struct ColumnButterfly12d {
    bf4: AvxFastButterfly4<f64>,
    bf3: AvxFastButterfly3<f64>,
}

impl ColumnButterfly12d {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> ColumnButterfly12d {
        unsafe {
            Self {
                bf3: AvxFastButterfly3::<f64>::new(direction),
                bf4: AvxFastButterfly4::<f64>::new(direction),
            }
        }
    }
}

impl ColumnButterfly12d {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn exec(&self, v: [AvxStoreD; 12]) -> [AvxStoreD; 12] {
        unsafe {
            let (u0, u1, u2, u3) = self.bf4.exec(v[0].v, v[3].v, v[6].v, v[9].v);
            let (u4, u5, u6, u7) = self.bf4.exec(v[4].v, v[7].v, v[10].v, v[1].v);
            let (u8, u9, u10, u11) = self.bf4.exec(v[8].v, v[11].v, v[2].v, v[5].v);

            let (v0, v4, v8) = self.bf3.exec(u0, u4, u8); // (v0, v4, v8)
            let (v9, v1, v5) = self.bf3.exec(u1, u5, u9); // (v9, v1, v5)
            let (v6, v10, v2) = self.bf3.exec(u2, u6, u10); // (v6, v10, v2)
            let (v3, v7, v11) = self.bf3.exec(u3, u7, u11); // (v3, v7, v11)

            [
                AvxStoreD::raw(v0),
                AvxStoreD::raw(v1),
                AvxStoreD::raw(v2),
                AvxStoreD::raw(v3),
                AvxStoreD::raw(v4),
                AvxStoreD::raw(v5),
                AvxStoreD::raw(v6),
                AvxStoreD::raw(v7),
                AvxStoreD::raw(v8),
                AvxStoreD::raw(v9),
                AvxStoreD::raw(v10),
                AvxStoreD::raw(v11),
            ]
        }
    }
}

pub(crate) struct ColumnButterfly12f {
    bf4: AvxFastButterfly4<f32>,
    bf3: AvxFastButterfly3<f32>,
}

impl ColumnButterfly12f {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> ColumnButterfly12f {
        unsafe {
            Self {
                bf3: AvxFastButterfly3::<f32>::new(direction),
                bf4: AvxFastButterfly4::<f32>::new(direction),
            }
        }
    }
}

impl ColumnButterfly12f {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn exec(&self, v: [AvxStoreF; 12]) -> [AvxStoreF; 12] {
        unsafe {
            let (u0, u1, u2, u3) = self.bf4.exec(v[0].v, v[3].v, v[6].v, v[9].v);
            let (u4, u5, u6, u7) = self.bf4.exec(v[4].v, v[7].v, v[10].v, v[1].v);
            let (u8, u9, u10, u11) = self.bf4.exec(v[8].v, v[11].v, v[2].v, v[5].v);

            let (v0, v4, v8) = self.bf3.exec(u0, u4, u8); // (v0, v4, v8)
            let (v9, v1, v5) = self.bf3.exec(u1, u5, u9); // (v9, v1, v5)
            let (v6, v10, v2) = self.bf3.exec(u2, u6, u10); // (v6, v10, v2)
            let (v3, v7, v11) = self.bf3.exec(u3, u7, u11); // (v3, v7, v11)

            [
                AvxStoreF::raw(v0),
                AvxStoreF::raw(v1),
                AvxStoreF::raw(v2),
                AvxStoreF::raw(v3),
                AvxStoreF::raw(v4),
                AvxStoreF::raw(v5),
                AvxStoreF::raw(v6),
                AvxStoreF::raw(v7),
                AvxStoreF::raw(v8),
                AvxStoreF::raw(v9),
                AvxStoreF::raw(v10),
                AvxStoreF::raw(v11),
            ]
        }
    }
}
