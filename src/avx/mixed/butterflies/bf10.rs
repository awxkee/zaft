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
use crate::avx::butterflies::{AvxButterfly, AvxFastButterfly5d, AvxFastButterfly5f};
use crate::avx::mixed::avx_stored::AvxStoreD;
use crate::avx::mixed::avx_storef::AvxStoreF;

pub(crate) struct ColumnButterfly10d {
    bf5: AvxFastButterfly5d,
}

impl ColumnButterfly10d {
    #[target_feature(enable = "avx")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly10d {
        unsafe {
            Self {
                bf5: AvxFastButterfly5d::new(direction),
            }
        }
    }
}

impl ColumnButterfly10d {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) fn exec(&self, v: [AvxStoreD; 10]) -> [AvxStoreD; 10] {
        let mid0 = self.bf5.exec(v[0].v, v[2].v, v[4].v, v[6].v, v[8].v);
        let mid1 = self.bf5.exec(v[5].v, v[7].v, v[9].v, v[1].v, v[3].v);

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) = AvxButterfly::butterfly2_f64(mid0.0, mid1.0);
        let (y2, y3) = AvxButterfly::butterfly2_f64(mid0.1, mid1.1);
        let (y4, y5) = AvxButterfly::butterfly2_f64(mid0.2, mid1.2);
        let (y6, y7) = AvxButterfly::butterfly2_f64(mid0.3, mid1.3);
        let (y8, y9) = AvxButterfly::butterfly2_f64(mid0.4, mid1.4);
        [
            AvxStoreD::raw(y0),
            AvxStoreD::raw(y3),
            AvxStoreD::raw(y4),
            AvxStoreD::raw(y7),
            AvxStoreD::raw(y8),
            AvxStoreD::raw(y1),
            AvxStoreD::raw(y2),
            AvxStoreD::raw(y5),
            AvxStoreD::raw(y6),
            AvxStoreD::raw(y9),
        ]
    }
}

pub(crate) struct ColumnButterfly10f {
    bf5: AvxFastButterfly5f,
}

impl ColumnButterfly10f {
    #[target_feature(enable = "avx")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly10f {
        unsafe {
            Self {
                bf5: AvxFastButterfly5f::new(direction),
            }
        }
    }
}

impl ColumnButterfly10f {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn exec(&self, v: [AvxStoreF; 10]) -> [AvxStoreF; 10] {
        let mid0 = self.bf5._m256_exec(v[0].v, v[2].v, v[4].v, v[6].v, v[8].v);
        let mid1 = self.bf5._m256_exec(v[5].v, v[7].v, v[9].v, v[1].v, v[3].v);

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) = AvxButterfly::butterfly2_f32(mid0.0, mid1.0);
        let (y2, y3) = AvxButterfly::butterfly2_f32(mid0.1, mid1.1);
        let (y4, y5) = AvxButterfly::butterfly2_f32(mid0.2, mid1.2);
        let (y6, y7) = AvxButterfly::butterfly2_f32(mid0.3, mid1.3);
        let (y8, y9) = AvxButterfly::butterfly2_f32(mid0.4, mid1.4);
        [
            AvxStoreF::raw(y0),
            AvxStoreF::raw(y3),
            AvxStoreF::raw(y4),
            AvxStoreF::raw(y7),
            AvxStoreF::raw(y8),
            AvxStoreF::raw(y1),
            AvxStoreF::raw(y2),
            AvxStoreF::raw(y5),
            AvxStoreF::raw(y6),
            AvxStoreF::raw(y9),
        ]
    }
}
