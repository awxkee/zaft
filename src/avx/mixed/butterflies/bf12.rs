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
    ColumnButterfly3d, ColumnButterfly3f, ColumnButterfly4d, ColumnButterfly4f,
};

pub(crate) struct ColumnButterfly12d {
    pub(crate) bf4: ColumnButterfly4d,
    bf3: ColumnButterfly3d,
}

impl ColumnButterfly12d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly12d {
        Self {
            bf3: ColumnButterfly3d::new(direction),
            bf4: ColumnButterfly4d::new(direction),
        }
    }
}

impl ColumnButterfly12d {
    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreD; 12]) -> [AvxStoreD; 12] {
        let [u0, u1, u2, u3] = self.bf4.exec([v[0], v[3], v[6], v[9]]);
        let [u4, u5, u6, u7] = self.bf4.exec([v[4], v[7], v[10], v[1]]);
        let [u8, u9, u10, u11] = self.bf4.exec([v[8], v[11], v[2], v[5]]);

        let [v0, v4, v8] = self.bf3.exec([u0, u4, u8]); // (v0, v4, v8)
        let [v9, v1, v5] = self.bf3.exec([u1, u5, u9]); // (v9, v1, v5)
        let [v6, v10, v2] = self.bf3.exec([u2, u6, u10]); // (v6, v10, v2)
        let [v3, v7, v11] = self.bf3.exec([u3, u7, u11]); // (v3, v7, v11)

        [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11]
    }
}

pub(crate) struct ColumnButterfly12f {
    pub(crate) bf4: ColumnButterfly4f,
    bf3: ColumnButterfly3f,
}

impl ColumnButterfly12f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly12f {
        Self {
            bf3: ColumnButterfly3f::new(direction),
            bf4: ColumnButterfly4f::new(direction),
        }
    }
}

impl ColumnButterfly12f {
    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 12]) -> [AvxStoreF; 12] {
        let [u0, u1, u2, u3] = self.bf4.exec([v[0], v[3], v[6], v[9]]);
        let [u4, u5, u6, u7] = self.bf4.exec([v[4], v[7], v[10], v[1]]);
        let [u8, u9, u10, u11] = self.bf4.exec([v[8], v[11], v[2], v[5]]);

        let [v0, v4, v8] = self.bf3.exec([u0, u4, u8]); // (v0, v4, v8)
        let [v9, v1, v5] = self.bf3.exec([u1, u5, u9]); // (v9, v1, v5)
        let [v6, v10, v2] = self.bf3.exec([u2, u6, u10]); // (v6, v10, v2)
        let [v3, v7, v11] = self.bf3.exec([u3, u7, u11]); // (v3, v7, v11)

        [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11]
    }
}
