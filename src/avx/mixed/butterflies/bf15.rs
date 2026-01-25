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
use crate::avx::mixed::{
    AvxStoreD, AvxStoreF, ColumnButterfly3d, ColumnButterfly3f, ColumnButterfly5d,
    ColumnButterfly5f,
};

pub(crate) struct ColumnButterfly15d {
    bf3: ColumnButterfly3d,
    bf5: ColumnButterfly5d,
}

impl ColumnButterfly15d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            bf3: ColumnButterfly3d::new(fft_direction),
            bf5: ColumnButterfly5d::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec(&self, v: [AvxStoreD; 15]) -> [AvxStoreD; 15] {
        let mid0 = self.bf5.exec([v[0], v[3], v[6], v[9], v[12]]);
        let mid1 = self.bf5.exec([v[5], v[8], v[11], v[14], v[2]]);
        let mid2 = self.bf5.exec([v[10], v[13], v[1], v[4], v[7]]);

        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-3 FFTs down the columns
        let [y0, y1, y2] = self.bf3.exec([mid0[0], mid1[0], mid2[0]]);
        let [y3, y4, y5] = self.bf3.exec([mid0[1], mid1[1], mid2[1]]);
        let [y6, y7, y8] = self.bf3.exec([mid0[2], mid1[2], mid2[2]]);
        let [y9, y10, y11] = self.bf3.exec([mid0[3], mid1[3], mid2[3]]);
        let [y12, y13, y14] = self.bf3.exec([mid0[4], mid1[4], mid2[4]]);
        [
            y0, y4, y8, y9, y13, y2, y3, y7, y11, y12, y1, y5, y6, y10, y14,
        ]
    }
}

pub(crate) struct ColumnButterfly15f {
    bf3: ColumnButterfly3f,
    bf5: ColumnButterfly5f,
}

impl ColumnButterfly15f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            bf3: ColumnButterfly3f::new(fft_direction),
            bf5: ColumnButterfly5f::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec(&self, v: [AvxStoreF; 15]) -> [AvxStoreF; 15] {
        let mid0 = self.bf5.exec([v[0], v[3], v[6], v[9], v[12]]);
        let mid1 = self.bf5.exec([v[5], v[8], v[11], v[14], v[2]]);
        let mid2 = self.bf5.exec([v[10], v[13], v[1], v[4], v[7]]);

        // Since this is good-thomas algorithm, we don't need twiddle factors

        // Transpose the data and do size-3 FFTs down the columns
        let [y0, y1, y2] = self.bf3.exec([mid0[0], mid1[0], mid2[0]]);
        let [y3, y4, y5] = self.bf3.exec([mid0[1], mid1[1], mid2[1]]);
        let [y6, y7, y8] = self.bf3.exec([mid0[2], mid1[2], mid2[2]]);
        let [y9, y10, y11] = self.bf3.exec([mid0[3], mid1[3], mid2[3]]);
        let [y12, y13, y14] = self.bf3.exec([mid0[4], mid1[4], mid2[4]]);
        [
            y0, y4, y8, y9, y13, y2, y3, y7, y11, y12, y1, y5, y6, y10, y14,
        ]
    }
}
