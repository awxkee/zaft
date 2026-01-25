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
use crate::avx::butterflies::{AvxButterfly, AvxFastButterfly3};
use crate::avx::mixed::avx_stored::AvxStoreD;
use crate::avx::mixed::avx_storef::AvxStoreF;

pub(crate) struct ColumnButterfly6d {
    bf3: AvxFastButterfly3<f64>,
}

impl ColumnButterfly6d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly6d {
        unsafe {
            Self {
                bf3: AvxFastButterfly3::<f64>::new(direction),
            }
        }
    }
}

impl ColumnButterfly6d {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn exec(&self, v: [AvxStoreD; 6]) -> [AvxStoreD; 6] {
        let (t0, t2, t4) = self.bf3.exec(v[0].v, v[2].v, v[4].v);
        let (t1, t3, t5) = self.bf3.exec(v[3].v, v[5].v, v[1].v);
        let (y0, y3) = AvxButterfly::butterfly2_f64(t0, t1);
        let (y4, y1) = AvxButterfly::butterfly2_f64(t2, t3);
        let (y2, y5) = AvxButterfly::butterfly2_f64(t4, t5);

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

pub(crate) struct ColumnButterfly6f {
    bf3: AvxFastButterfly3<f32>,
}

impl ColumnButterfly6f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly6f {
        unsafe {
            Self {
                bf3: AvxFastButterfly3::<f32>::new(direction),
            }
        }
    }
}

impl ColumnButterfly6f {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 6]) -> [AvxStoreF; 6] {
        let (t0, t2, t4) = self.bf3.exec(v[0].v, v[2].v, v[4].v);
        let (t1, t3, t5) = self.bf3.exec(v[3].v, v[5].v, v[1].v);
        let (y0, y3) = AvxButterfly::butterfly2_f32(t0, t1);
        let (y4, y1) = AvxButterfly::butterfly2_f32(t2, t3);
        let (y2, y5) = AvxButterfly::butterfly2_f32(t4, t5);

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
