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
use std::arch::x86_64::*;

pub(crate) struct ColumnButterfly2d {}

impl ColumnButterfly2d {
    pub(crate) fn new(_: FftDirection) -> ColumnButterfly2d {
        Self {}
    }
}

impl ColumnButterfly2d {
    #[target_feature(enable = "avx2")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn exec(&self, v: [AvxStoreD; 2]) -> [AvxStoreD; 2] {
        let t = _mm256_add_pd(v[0].v, v[1].v);
        let y1 = _mm256_sub_pd(v[0].v, v[1].v);
        let y0 = t;
        [AvxStoreD::raw(y0), AvxStoreD::raw(y1)]
    }
}

pub(crate) struct ColumnButterfly2f {}

impl ColumnButterfly2f {
    pub(crate) fn new(_: FftDirection) -> ColumnButterfly2f {
        Self {}
    }
}

impl ColumnButterfly2f {
    #[target_feature(enable = "avx2")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 2]) -> [AvxStoreF; 2] {
        let t = _mm256_add_ps(v[0].v, v[1].v);
        let y1 = _mm256_sub_ps(v[0].v, v[1].v);
        let y0 = t;
        [AvxStoreF::raw(y0), AvxStoreF::raw(y1)]
    }
}
