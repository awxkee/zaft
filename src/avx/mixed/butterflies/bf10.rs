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
use crate::avx::butterflies::{AvxButterfly, AvxFastButterfly3, AvxFastButterfly5d};
use crate::avx::mixed::avx_store::AvxStoreD;
use crate::avx::util::_mm256_fcmul_pd;
use crate::util::compute_twiddle;
use std::arch::x86_64::*;

pub(crate) struct ColumnButterfly10d {
    bf5: AvxFastButterfly5d,
}

impl ColumnButterfly10d {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> ColumnButterfly10d {
        unsafe {
            Self {
                bf5: AvxFastButterfly5d::new(direction),
            }
        }
    }
}

impl ColumnButterfly10d {
    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn exec(&self, v: [AvxStoreD; 10]) -> [AvxStoreD; 10] {
        unsafe {
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
}
