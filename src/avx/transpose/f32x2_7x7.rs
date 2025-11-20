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
use crate::avx::mixed::{AvxStoreD, AvxStoreF};
use num_complex::Complex;
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn store_transpose_7x7_f32(
    left: [AvxStoreF; 7],
    right: [AvxStoreF; 7],
) -> ([AvxStoreF; 7], [AvxStoreF; 7]) {
    let (q0, q1) = transpose_7x7_f32(
        [
            left[0].v, left[1].v, left[2].v, left[3].v, left[4].v, left[5].v, left[6].v,
        ],
        [
            right[0].v, right[1].v, right[2].v, right[3].v, right[4].v, right[5].v, right[6].v,
        ],
    );
    (
        [
            AvxStoreF::raw(q0[0]),
            AvxStoreF::raw(q0[1]),
            AvxStoreF::raw(q0[2]),
            AvxStoreF::raw(q0[3]),
            AvxStoreF::raw(q0[4]),
            AvxStoreF::raw(q0[5]),
            AvxStoreF::raw(q0[6]),
        ],
        [
            AvxStoreF::raw(q1[0]),
            AvxStoreF::raw(q1[1]),
            AvxStoreF::raw(q1[2]),
            AvxStoreF::raw(q1[3]),
            AvxStoreF::raw(q1[4]),
            AvxStoreF::raw(q1[5]),
            AvxStoreF::raw(q1[6]),
        ],
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn block_transpose_f32x2_7x7(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let rows0: [AvxStoreF; 7] = std::array::from_fn(|x| {
            AvxStoreF::from_complex_ref(src.get_unchecked(x * src_stride..))
        });
        let rows1: [AvxStoreF; 7] = std::array::from_fn(|x| {
            AvxStoreF::from_complex3(src.get_unchecked(x * src_stride + 4..))
        });

        let (v0, v1) = store_transpose_7x7_f32(rows0, rows1);

        for i in 0..7 {
            v0[i].write(dst.get_unchecked_mut(i * dst_stride..));
            v1[i].write_lo3(dst.get_unchecked_mut(i * dst_stride + 4..));
        }
    }
}
