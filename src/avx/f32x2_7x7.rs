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
use crate::avx::f32x2_4x4::avx_transpose_f32x2_4x4_impl;
use crate::avx::f64x2_4x4::avx_transpose_f64x2_4x4_impl;
use crate::avx::mixed::AvxStoreD;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_7x7_f32(
    left: [__m256; 7],
    right: [__m256; 7],
) -> ([__m256; 7], [__m256; 7]) {
    // --- Top-left 4x4 complex block ---
    let tl = avx_transpose_f32x2_4x4_impl(left[0], left[1], left[2], left[3]);
    // Bottom-left 2x4 complex block (pad 2 rows with zeros)
    let bl = avx_transpose_f32x2_4x4_impl(left[4], left[5], left[6], _mm256_setzero_ps());

    // Top-right 4x2 complex block (pad 2 columns with zeros to form 4x4)
    let tr = avx_transpose_f32x2_4x4_impl(right[0], right[1], right[2], right[3]);
    // Bottom-right 2x2 complex block
    let br = avx_transpose_f32x2_4x4_impl(right[4], right[5], right[6], _mm256_setzero_ps());

    // --- Reassemble left output (first 4 columns) ---
    let output_left = [
        tl.0, tl.1, tl.2, tl.3, // top 4 rows
        tr.0, tr.1, tr.2, // bottom 2 rows
    ];

    // Reassemble right 6 rows (last 2 columns)
    let output_right = [
        bl.0, bl.1, bl.2, bl.3, // top 4 rows
        br.0, br.1, br.2, // bottom 2 rows
    ];

    (output_left, output_right)
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_7x7_f64(
    rows1: [AvxStoreD; 7],
    rows2: [AvxStoreD; 7],
    rows3: [AvxStoreD; 7],
    rows4: [AvxStoreD; 7],
) -> (
    [AvxStoreD; 7],
    [AvxStoreD; 7],
    [AvxStoreD; 7],
    [AvxStoreD; 7],
) {
    let tl = avx_transpose_f64x2_4x4_impl(
        (rows1[0].v, rows2[0].v),
        (rows1[1].v, rows2[1].v),
        (rows1[2].v, rows2[2].v),
        (rows1[3].v, rows2[3].v),
    );
    // Bottom-left 1x4 complex block (pad 3 rows with zeros)
    let bl = avx_transpose_f64x2_4x4_impl(
        (rows1[4].v, rows2[4].v),
        (rows1[5].v, rows2[5].v),
        (rows1[6].v, rows2[6].v),
        (_mm256_setzero_pd(), _mm256_setzero_pd()),
    );

    // Top-right 4x2 complex block (pad 2 columns with zeros to form 4x4)
    let tr = avx_transpose_f64x2_4x4_impl(
        (rows3[0].v, rows4[0].v),
        (rows3[1].v, rows4[1].v),
        (rows3[2].v, rows4[2].v),
        (rows3[3].v, rows4[3].v),
    );
    // Bottom-right 2x2 complex block
    let br = avx_transpose_f64x2_4x4_impl(
        (rows3[4].v, rows4[4].v),
        (rows3[5].v, rows4[5].v),
        (rows3[6].v, rows4[6].v),
        (_mm256_setzero_pd(), _mm256_setzero_pd()),
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
            AvxStoreD::raw(output_left[0].0),
            AvxStoreD::raw(output_left[1].0),
            AvxStoreD::raw(output_left[2].0),
            AvxStoreD::raw(output_left[3].0),
            AvxStoreD::raw(output_left[4].0),
            AvxStoreD::raw(output_left[5].0),
            AvxStoreD::raw(output_left[6].0),
        ],
        [
            AvxStoreD::raw(output_left[0].1),
            AvxStoreD::raw(output_left[1].1),
            AvxStoreD::raw(output_left[2].1),
            AvxStoreD::raw(output_left[3].1),
            AvxStoreD::raw(output_left[4].1),
            AvxStoreD::raw(output_left[5].1),
            AvxStoreD::raw(output_left[6].1),
        ],
        [
            AvxStoreD::raw(output_right[0].0),
            AvxStoreD::raw(output_right[1].0),
            AvxStoreD::raw(output_right[2].0),
            AvxStoreD::raw(output_right[3].0),
            AvxStoreD::raw(output_right[4].0),
            AvxStoreD::raw(output_right[5].0),
            AvxStoreD::raw(output_right[6].0),
        ],
        [
            AvxStoreD::raw(output_right[0].1),
            AvxStoreD::raw(output_right[1].1),
            AvxStoreD::raw(output_right[2].1),
            AvxStoreD::raw(output_right[3].1),
            AvxStoreD::raw(output_right[4].1),
            AvxStoreD::raw(output_right[5].1),
            AvxStoreD::raw(output_right[6].1),
        ],
    )
}
