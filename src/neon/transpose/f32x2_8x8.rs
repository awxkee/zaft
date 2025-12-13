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
use crate::neon::transpose::{neon_transpose_f32x2_2x2_impl, transpose_f32x2_4x4};
use std::arch::aarch64::{float32x4x2_t, vdupq_n_f32};

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

#[inline(always)]
pub(crate) fn neon_transpose_f32x2_8x5_aos(
    rows1: [NeonStoreF; 5],
    rows2: [NeonStoreF; 5],
    rows3: [NeonStoreF; 5],
    rows4: [NeonStoreF; 5],
) -> ([NeonStoreF; 8], [NeonStoreF; 8], [NeonStoreF; 8]) {
    unsafe {
        let tl = transpose_f32x2_4x4(
            float32x4x2_t(rows1[0].v, rows2[0].v),
            float32x4x2_t(rows1[1].v, rows2[1].v),
            float32x4x2_t(rows1[2].v, rows2[2].v),
            float32x4x2_t(rows1[3].v, rows2[3].v),
        );
        // Bottom-left 1x4 complex block (pad 3 rows with zeros)
        let bl = transpose_f32x2_4x4(
            float32x4x2_t(rows1[4].v, rows2[4].v),
            float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
            float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
            float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
        );

        // Top-right 4x2 complex block (pad 2 columns with zeros to form 4x4)
        let tr = transpose_f32x2_4x4(
            float32x4x2_t(rows3[0].v, rows4[0].v),
            float32x4x2_t(rows3[1].v, rows4[1].v),
            float32x4x2_t(rows3[2].v, rows4[2].v),
            float32x4x2_t(rows3[3].v, rows4[3].v),
        );
        // Bottom-right 2x2 complex block
        let br = transpose_f32x2_4x4(
            float32x4x2_t(rows3[4].v, rows4[4].v),
            float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
            float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
            float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
        );

        // Reassemble left 6 rows (first 4 columns)
        let output_left = [
            tl.0, tl.1, tl.2, tl.3, // top 4 rows
            tr.0, tr.1, tr.2, tr.3, // bottom 2 rows
        ];

        // Reassemble right 6 rows (last 2 columns)
        let output_right = [
            bl.0, bl.1, bl.2, bl.3, // top 4 rows
            br.0, br.1, br.2, br.3, // bottom 2 rows
        ];

        (
            [
                NeonStoreF::raw(output_left[0].0),
                NeonStoreF::raw(output_left[1].0),
                NeonStoreF::raw(output_left[2].0),
                NeonStoreF::raw(output_left[3].0),
                NeonStoreF::raw(output_left[4].0),
                NeonStoreF::raw(output_left[5].0),
                NeonStoreF::raw(output_left[6].0),
                NeonStoreF::raw(output_left[7].0),
            ],
            [
                NeonStoreF::raw(output_left[0].1),
                NeonStoreF::raw(output_left[1].1),
                NeonStoreF::raw(output_left[2].1),
                NeonStoreF::raw(output_left[3].1),
                NeonStoreF::raw(output_left[4].1),
                NeonStoreF::raw(output_left[5].1),
                NeonStoreF::raw(output_left[6].1),
                NeonStoreF::raw(output_left[7].1),
            ],
            [
                NeonStoreF::raw(output_right[0].0),
                NeonStoreF::raw(output_right[1].0),
                NeonStoreF::raw(output_right[2].0),
                NeonStoreF::raw(output_right[3].0),
                NeonStoreF::raw(output_right[4].0),
                NeonStoreF::raw(output_right[5].0),
                NeonStoreF::raw(output_right[6].0),
                NeonStoreF::raw(output_right[7].0),
            ],
        )
    }
}
