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
use crate::avx::mixed::avx_stored::AvxStoreD;
use crate::avx::mixed::avx_storef::AvxStoreF;
use crate::avx::mixed::{
    ColumnButterfly2d, ColumnButterfly2f, ColumnButterfly5d, ColumnButterfly5f,
};

pub(crate) struct ColumnButterfly10d {
    bf5: ColumnButterfly5d,
    bf2: ColumnButterfly2d,
}

impl ColumnButterfly10d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly10d {
        Self {
            bf5: ColumnButterfly5d::new(direction),
            bf2: ColumnButterfly2d::new(direction),
        }
    }
}

impl ColumnButterfly10d {
    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreD; 10]) -> [AvxStoreD; 10] {
        let mid0 = self.bf5.exec([v[0], v[2], v[4], v[6], v[8]]);
        let mid1 = self.bf5.exec([v[5], v[7], v[9], v[1], v[3]]);

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let [y0, y1] = self.bf2.exec([mid0[0], mid1[0]]);
        let [y2, y3] = self.bf2.exec([mid0[1], mid1[1]]);
        let [y4, y5] = self.bf2.exec([mid0[2], mid1[2]]);
        let [y6, y7] = self.bf2.exec([mid0[3], mid1[3]]);
        let [y8, y9] = self.bf2.exec([mid0[4], mid1[4]]);
        [y0, y3, y4, y7, y8, y1, y2, y5, y6, y9]
    }
}

pub(crate) struct ColumnButterfly10f {
    bf5: ColumnButterfly5f,
    bf2: ColumnButterfly2f,
}

impl ColumnButterfly10f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly10f {
        Self {
            bf5: ColumnButterfly5f::new(direction),
            bf2: ColumnButterfly2f::new(direction),
        }
    }
}

impl ColumnButterfly10f {
    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 10]) -> [AvxStoreF; 10] {
        let mid0 = self.bf5.exec([v[0], v[2], v[4], v[6], v[8]]);
        let mid1 = self.bf5.exec([v[5], v[7], v[9], v[1], v[3]]);

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let [y0, y1] = self.bf2.exec([mid0[0], mid1[0]]);
        let [y2, y3] = self.bf2.exec([mid0[1], mid1[1]]);
        let [y4, y5] = self.bf2.exec([mid0[2], mid1[2]]);
        let [y6, y7] = self.bf2.exec([mid0[3], mid1[3]]);
        let [y8, y9] = self.bf2.exec([mid0[4], mid1[4]]);
        [y0, y3, y4, y7, y8, y1, y2, y5, y6, y9]
    }
}
