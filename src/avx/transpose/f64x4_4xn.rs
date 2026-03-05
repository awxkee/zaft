/*
 * // Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::avx::mixed::AvxStoreD;
use std::arch::x86_64::{_mm256_permute2f128_pd, _mm256_shuffle_pd};

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_4x4_f64(store: [AvxStoreD; 4]) -> [AvxStoreD; 4] {
    let tmp0 = _mm256_shuffle_pd::<0x0>(store[0].v, store[1].v);
    let tmp2 = _mm256_shuffle_pd::<0xF>(store[0].v, store[1].v);
    let tmp1 = _mm256_shuffle_pd::<0x0>(store[2].v, store[3].v);
    let tmp3 = _mm256_shuffle_pd::<0xF>(store[2].v, store[3].v);

    let row0 = _mm256_permute2f128_pd::<0x20>(tmp0, tmp1);
    let row1 = _mm256_permute2f128_pd::<0x20>(tmp2, tmp3);
    let row2 = _mm256_permute2f128_pd::<0x31>(tmp0, tmp1);
    let row3 = _mm256_permute2f128_pd::<0x31>(tmp2, tmp3);

    [
        AvxStoreD::raw(row0),
        AvxStoreD::raw(row1),
        AvxStoreD::raw(row2),
        AvxStoreD::raw(row3),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x4_4x3(rows: [AvxStoreD; 3]) -> [AvxStoreD; 4] {
    transpose_4x4_f64([rows[0], rows[1], rows[2], AvxStoreD::zero()])
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x4_4x5(rows: [AvxStoreD; 5]) -> [AvxStoreD; 8] {
    let a0 = transpose_4x4_f64([rows[0], rows[1], rows[2], rows[3]]);
    let b0 = transpose_4x4_f64([
        rows[4],
        AvxStoreD::zero(),
        AvxStoreD::zero(),
        AvxStoreD::zero(),
    ]);
    [a0[0], a0[1], a0[2], a0[3], b0[0], b0[1], b0[2], b0[3]]
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x4_4x7(rows: [AvxStoreD; 7]) -> [AvxStoreD; 8] {
    let a0 = transpose_4x4_f64([rows[0], rows[1], rows[2], rows[3]]);
    let b0 = transpose_4x4_f64([rows[4], rows[5], rows[6], AvxStoreD::zero()]);
    [a0[0], a0[1], a0[2], a0[3], b0[0], b0[1], b0[2], b0[3]]
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x4_4x9(rows: [AvxStoreD; 9]) -> [AvxStoreD; 12] {
    let a0 = transpose_4x4_f64([rows[0], rows[1], rows[2], rows[3]]);
    let b0 = transpose_4x4_f64([rows[4], rows[5], rows[6], rows[7]]);
    let c0 = transpose_4x4_f64([
        rows[8],
        AvxStoreD::zero(),
        AvxStoreD::zero(),
        AvxStoreD::zero(),
    ]);
    [
        a0[0], a0[1], a0[2], a0[3], b0[0], b0[1], b0[2], b0[3], c0[0], c0[1], c0[2], c0[3],
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x4_4x11(rows: [AvxStoreD; 11]) -> [AvxStoreD; 12] {
    let a0 = transpose_4x4_f64([rows[0], rows[1], rows[2], rows[3]]);
    let b0 = transpose_4x4_f64([rows[4], rows[5], rows[6], rows[7]]);
    let c0 = transpose_4x4_f64([rows[8], rows[9], rows[10], AvxStoreD::zero()]);
    [
        a0[0], a0[1], a0[2], a0[3], b0[0], b0[1], b0[2], b0[3], c0[0], c0[1], c0[2], c0[3],
    ]
}
