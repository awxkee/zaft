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
use crate::avx::butterflies::AvxButterfly;
use crate::avx::mixed::avx_stored::AvxStoreD;
use crate::avx::mixed::avx_storef::AvxStoreF;
use crate::avx::mixed::{ColumnButterfly3d, ColumnButterfly3f};

pub(crate) struct ColumnButterfly6d {
    bf3: ColumnButterfly3d,
}

impl ColumnButterfly6d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly6d {
        Self {
            bf3: ColumnButterfly3d::new(direction),
        }
    }
}

impl ColumnButterfly6d {
    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreD; 6]) -> [AvxStoreD; 6] {
        unsafe {
            let [t0, t2, t4] = self.bf3.exec([v[0], v[2], v[4]]);
            let [t1, t3, t5] = self.bf3.exec([v[3], v[5], v[1]]);
            let (y0, y3) = AvxButterfly::butterfly2_f64(t0.v, t1.v);
            let (y4, y1) = AvxButterfly::butterfly2_f64(t2.v, t3.v);
            let (y2, y5) = AvxButterfly::butterfly2_f64(t4.v, t5.v);

            [
                AvxStoreD::raw(y0),
                AvxStoreD::raw(y1),
                AvxStoreD::raw(y2),
                AvxStoreD::raw(y3),
                AvxStoreD::raw(y4),
                AvxStoreD::raw(y5),
            ]
        }
    }
}

pub(crate) struct ColumnButterfly6f {
    bf3: ColumnButterfly3f,
}

impl ColumnButterfly6f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly6f {
        Self {
            bf3: ColumnButterfly3f::new(direction),
        }
    }
}

impl ColumnButterfly6f {
    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 6]) -> [AvxStoreF; 6] {
        let [t0, t2, t4] = self.bf3.exec([v[0], v[2], v[4]]);
        let [t1, t3, t5] = self.bf3.exec([v[3], v[5], v[1]]);
        let (y0, y3) = AvxButterfly::butterfly2_f32(t0.v, t1.v);
        let (y4, y1) = AvxButterfly::butterfly2_f32(t2.v, t3.v);
        let (y2, y5) = AvxButterfly::butterfly2_f32(t4.v, t5.v);

        [
            AvxStoreF::raw(y0),
            AvxStoreF::raw(y1),
            AvxStoreF::raw(y2),
            AvxStoreF::raw(y3),
            AvxStoreF::raw(y4),
            AvxStoreF::raw(y5),
        ]
    }
}
