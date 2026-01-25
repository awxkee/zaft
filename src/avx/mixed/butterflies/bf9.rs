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
use crate::avx::mixed::{ColumnButterfly3d, ColumnButterfly3f};
use crate::util::compute_twiddle;

pub(crate) struct ColumnButterfly9d {
    pub(crate) bf3: ColumnButterfly3d,
    twiddle1: AvxStoreD,
    twiddle2: AvxStoreD,
    twiddle4: AvxStoreD,
}

impl ColumnButterfly9d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly9d {
        let tw1 = compute_twiddle::<f64>(1, 9, direction);
        let tw2 = compute_twiddle::<f64>(2, 9, direction);
        let tw4 = compute_twiddle::<f64>(4, 9, direction);
        Self {
            twiddle1: AvxStoreD::set_complex(&tw1),
            twiddle2: AvxStoreD::set_complex(&tw2),
            twiddle4: AvxStoreD::set_complex(&tw4),
            bf3: ColumnButterfly3d::new(direction),
        }
    }
}

impl ColumnButterfly9d {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn exec(&self, v: [AvxStoreD; 9]) -> [AvxStoreD; 9] {
        let [u0, u3, u6] = self.bf3.exec([v[0], v[3], v[6]]);
        let [u1, mut u4, mut u7] = self.bf3.exec([v[1], v[4], v[7]]);
        let [u2, mut u5, mut u8] = self.bf3.exec([v[2], v[5], v[8]]);

        u4 = AvxStoreD::mul_by_complex(u4, self.twiddle1);
        u7 = AvxStoreD::mul_by_complex(u7, self.twiddle2);
        u5 = AvxStoreD::mul_by_complex(u5, self.twiddle2);
        u8 = AvxStoreD::mul_by_complex(u8, self.twiddle4);

        let [y0, y3, y6] = self.bf3.exec([u0, u1, u2]);
        let [y1, y4, y7] = self.bf3.exec([u3, u4, u5]);
        let [y2, y5, y8] = self.bf3.exec([u6, u7, u8]);
        [y0, y1, y2, y3, y4, y5, y6, y7, y8]
    }
}

pub(crate) struct ColumnButterfly9f {
    pub(crate) bf3: ColumnButterfly3f,
    twiddle1: AvxStoreF,
    twiddle2: AvxStoreF,
    twiddle4: AvxStoreF,
}

impl ColumnButterfly9f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly9f {
        let tw1 = compute_twiddle::<f32>(1, 9, direction);
        let tw2 = compute_twiddle::<f32>(2, 9, direction);
        let tw4 = compute_twiddle::<f32>(4, 9, direction);
        Self {
            twiddle1: AvxStoreF::set_complex(tw1),
            twiddle2: AvxStoreF::set_complex(tw2),
            twiddle4: AvxStoreF::set_complex(tw4),
            bf3: ColumnButterfly3f::new(direction),
        }
    }
}

impl ColumnButterfly9f {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 9]) -> [AvxStoreF; 9] {
        let [u0, u3, u6] = self.bf3.exec([v[0], v[3], v[6]]);
        let [u1, mut u4, mut u7] = self.bf3.exec([v[1], v[4], v[7]]);
        let [u2, mut u5, mut u8] = self.bf3.exec([v[2], v[5], v[8]]);

        u4 = AvxStoreF::mul_by_complex(u4, self.twiddle1);
        u7 = AvxStoreF::mul_by_complex(u7, self.twiddle2);
        u5 = AvxStoreF::mul_by_complex(u5, self.twiddle2);
        u8 = AvxStoreF::mul_by_complex(u8, self.twiddle4);

        let [y0, y3, y6] = self.bf3.exec([u0, u1, u2]);
        let [y1, y4, y7] = self.bf3.exec([u3, u4, u5]);
        let [y2, y5, y8] = self.bf3.exec([u6, u7, u8]);
        [y0, y1, y2, y3, y4, y5, y6, y7, y8]
    }
}
