/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use std::arch::aarch64::float32x4x2_t;

#[inline]
pub(crate) fn neon_transpose_f32x2_8x2(rows: [NeonStoreF; 8]) -> [NeonStoreF; 8] {
    // matrix transpose implementation (8x2 -> 2x8):
    // [ A B ]^T => [ A^T C^T E^T G^T I^T K^T M^T O^T ]
    // [ C D ]      [ B^T D^T F^T H^T J^T L^T N^T P^T ]
    // [ E F ]
    // [ G H ]
    // [ I J ]
    // [ K L ]
    // [ M N ]
    // [ O P ]
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, rows[5].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[6].v, rows[7].v));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
    ]
}

#[inline]
pub(crate) fn transpose_8x8_f32(
    rows0: [NeonStoreF; 8],
    rows1: [NeonStoreF; 8],
    rows2: [NeonStoreF; 8],
    rows3: [NeonStoreF; 8],
) -> (
    [NeonStoreF; 8],
    [NeonStoreF; 8],
    [NeonStoreF; 8],
    [NeonStoreF; 8],
) {
    // matrix transpose implementation (8x2 -> 2x8):
    // [ A B ]^T => [ A^T C^T E^T G^T I^T K^T M^T O^T ]
    // [ C D ]      [ B^T D^T F^T H^T J^T L^T N^T P^T ]
    // [ E F ]
    // [ G H ]
    // [ I J ]
    // [ K L ]
    // [ M N ]
    // [ O P ]

    let transposed00 = neon_transpose_f32x2_8x2(rows0);
    let transposed01 = neon_transpose_f32x2_8x2(rows1);
    let transposed10 = neon_transpose_f32x2_8x2(rows2);
    let transposed11 = neon_transpose_f32x2_8x2(rows3);

    (
        [
            transposed00[0],
            transposed00[1],
            transposed01[0],
            transposed01[1],
            transposed10[0],
            transposed10[1],
            transposed11[0],
            transposed11[1],
        ],
        [
            transposed00[2],
            transposed00[3],
            transposed01[2],
            transposed01[3],
            transposed10[2],
            transposed10[3],
            transposed11[2],
            transposed11[3],
        ],
        [
            transposed00[4],
            transposed00[5],
            transposed01[4],
            transposed01[5],
            transposed10[4],
            transposed10[5],
            transposed11[4],
            transposed11[5],
        ],
        [
            transposed00[6],
            transposed00[7],
            transposed01[6],
            transposed01[7],
            transposed10[6],
            transposed10[7],
            transposed11[6],
            transposed11[7],
        ],
    )
}
