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
use crate::neon::f32x2_4x4::transpose_f32x2_4x4;
use crate::neon::mixed::NeonStoreF;
use std::arch::aarch64::{float32x4x2_t, vdupq_n_f32};

#[inline]
pub(crate) fn neon_transpose_f32x2_7x7_aos(
    rows1: [NeonStoreF; 7],
    rows2: [NeonStoreF; 7],
    rows3: [NeonStoreF; 7],
    rows4: [NeonStoreF; 7],
) -> (
    [NeonStoreF; 7],
    [NeonStoreF; 7],
    [NeonStoreF; 7],
    [NeonStoreF; 7],
) {
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
            float32x4x2_t(rows1[5].v, rows2[5].v),
            float32x4x2_t(rows1[6].v, rows2[6].v),
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
            float32x4x2_t(rows3[5].v, rows4[5].v),
            float32x4x2_t(rows3[6].v, rows4[6].v),
            float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
        );

        // Reassemble left 6 rows (first 4 columns)
        let output_left = [
            tl.0, tl.1, tl.2, tl.3, // top 4 rows
            tr.0, tr.1, tr.2, // bottom 2 rows
        ];

        // Reassemble right 6 rows (last 2 columns)
        let output_right = [
            bl.0, bl.1, bl.2, bl.3, // top 4 rows
            br.0, br.1, br.2, // bottom 2 rows
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
            ],
            [
                NeonStoreF::raw(output_left[0].1),
                NeonStoreF::raw(output_left[1].1),
                NeonStoreF::raw(output_left[2].1),
                NeonStoreF::raw(output_left[3].1),
                NeonStoreF::raw(output_left[4].1),
                NeonStoreF::raw(output_left[5].1),
                NeonStoreF::raw(output_left[6].1),
            ],
            [
                NeonStoreF::raw(output_right[0].0),
                NeonStoreF::raw(output_right[1].0),
                NeonStoreF::raw(output_right[2].0),
                NeonStoreF::raw(output_right[3].0),
                NeonStoreF::raw(output_right[4].0),
                NeonStoreF::raw(output_right[5].0),
                NeonStoreF::raw(output_right[6].0),
            ],
            [
                NeonStoreF::raw(output_right[0].1),
                NeonStoreF::raw(output_right[1].1),
                NeonStoreF::raw(output_right[2].1),
                NeonStoreF::raw(output_right[3].1),
                NeonStoreF::raw(output_right[4].1),
                NeonStoreF::raw(output_right[5].1),
                NeonStoreF::raw(output_right[6].1),
            ],
        )
    }
}
