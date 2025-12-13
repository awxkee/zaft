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
use crate::neon::transpose::f32x2_4x4::transpose_f32x2_4x4;
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use num_complex::Complex;
use std::arch::aarch64::{float32x4x2_t, vdupq_n_f32};

#[inline]
pub(crate) fn neon_transpose_f32x2_7x5_aos(
    rows1: [NeonStoreF; 5],
    rows2: [NeonStoreF; 5],
    rows3: [NeonStoreF; 5],
    rows4: [NeonStoreF; 5],
) -> ([NeonStoreF; 7], [NeonStoreF; 7], [NeonStoreF; 7]) {
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
        )
    }
}

#[inline(always)]
pub(crate) fn neon_transpose_f32x2_7x6_aos(
    rows1: [NeonStoreF; 6],
    rows2: [NeonStoreF; 6],
    rows3: [NeonStoreF; 6],
    rows4: [NeonStoreF; 6],
) -> ([NeonStoreF; 7], [NeonStoreF; 7], [NeonStoreF; 7]) {
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
            float32x4x2_t(rows3[5].v, rows4[5].v),
            float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
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
        )
    }
}

#[inline]
pub(crate) fn block_transpose_f32x2_7x5(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let rows0: [NeonStoreF; 5] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride..))
        });
        let rows1: [NeonStoreF; 5] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 2..))
        });
        let rows2: [NeonStoreF; 5] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 4..))
        });
        let rows3: [NeonStoreF; 5] = std::array::from_fn(|x| {
            NeonStoreF::from_complex(src.get_unchecked(x * src_stride + 6))
        });

        let (v0, v1, v2) = neon_transpose_f32x2_7x5_aos(rows0, rows1, rows2, rows3);

        for i in 0..7 {
            v0[i].write(dst.get_unchecked_mut(i * dst_stride..));
            v1[i].write(dst.get_unchecked_mut(i * dst_stride + 2..));
            v2[i].write_lo(dst.get_unchecked_mut(i * dst_stride + 4..));
        }
    }
}

#[inline(always)]
pub(crate) fn transpose_2x7(rows: [NeonStoreF; 7]) -> [NeonStoreF; 8] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, rows[5].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[6].v, unsafe { vdupq_n_f32(0.) }));
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
pub(crate) fn block_transpose_f32x2_7x6(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let rows0: [NeonStoreF; 6] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride..))
        });
        let rows1: [NeonStoreF; 6] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 2..))
        });
        let rows2: [NeonStoreF; 6] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 4..))
        });
        let rows3: [NeonStoreF; 6] = std::array::from_fn(|x| {
            NeonStoreF::from_complex(src.get_unchecked(x * src_stride + 6))
        });

        let (v0, v1, v2) = neon_transpose_f32x2_7x6_aos(rows0, rows1, rows2, rows3);

        for i in 0..7 {
            v0[i].write(dst.get_unchecked_mut(i * dst_stride..));
            v1[i].write(dst.get_unchecked_mut(i * dst_stride + 2..));
            v2[i].write(dst.get_unchecked_mut(i * dst_stride + 4..));
        }
    }
}

#[inline]
pub(crate) fn block_transpose_f32x2_7x3(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let rows0: [NeonStoreF; 3] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride..))
        });
        let rows1: [NeonStoreF; 3] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 2..))
        });
        let rows2: [NeonStoreF; 3] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 4..))
        });
        let rows3: [NeonStoreF; 3] = std::array::from_fn(|x| {
            NeonStoreF::from_complex(src.get_unchecked(x * src_stride + 6))
        });

        let (v0, v1, _) = neon_transpose_f32x2_7x5_aos(
            [
                rows0[0],
                rows0[1],
                rows0[2],
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
            [
                rows1[0],
                rows1[1],
                rows1[2],
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
            [
                rows2[0],
                rows2[1],
                rows2[2],
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
            [
                rows3[0],
                rows3[1],
                rows3[2],
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
        );

        for i in 0..7 {
            v0[i].write(dst.get_unchecked_mut(i * dst_stride..));
            v1[i].write_lo(dst.get_unchecked_mut(i * dst_stride + 2..));
        }
    }
}

#[inline]
pub(crate) fn block_transpose_f32x2_7x2(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let rows0: [NeonStoreF; 2] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride..))
        });
        let rows1: [NeonStoreF; 2] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 2..))
        });
        let rows2: [NeonStoreF; 2] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 4..))
        });
        let rows3: [NeonStoreF; 2] = std::array::from_fn(|x| {
            NeonStoreF::from_complex(src.get_unchecked(x * src_stride + 6))
        });

        let (v0, _, _) = neon_transpose_f32x2_7x5_aos(
            [
                rows0[0],
                rows0[1],
                NeonStoreF::default(),
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
            [
                rows1[0],
                rows1[1],
                NeonStoreF::default(),
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
            [
                rows2[0],
                rows2[1],
                NeonStoreF::default(),
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
            [
                rows3[0],
                rows3[1],
                NeonStoreF::default(),
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
        );

        for i in 0..7 {
            v0[i].write(dst.get_unchecked_mut(i * dst_stride..));
        }
    }
}

#[inline]
pub(crate) fn block_transpose_f32x2_4x7(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let r0: [NeonStoreF; 7] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride..))
        });
        let r1: [NeonStoreF; 7] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 2..))
        });

        let q0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r0[0].v, r0[1].v));
        let q1 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r0[2].v, r0[3].v));
        let q2 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r0[4].v, r0[5].v));
        let q3 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r0[6].v, vdupq_n_f32(0.)));

        let q4 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r1[0].v, r1[1].v));
        let q5 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r1[2].v, r1[3].v));
        let q6 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r1[4].v, r1[5].v));
        let q7 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r1[6].v, vdupq_n_f32(0.)));

        NeonStoreF::raw(q0.0).write(dst);
        NeonStoreF::raw(q0.1).write(dst.get_unchecked_mut(dst_stride..));

        NeonStoreF::raw(q1.0).write(dst.get_unchecked_mut(2..));
        NeonStoreF::raw(q1.1).write(dst.get_unchecked_mut(2 + dst_stride..));

        NeonStoreF::raw(q2.0).write(dst.get_unchecked_mut(4..));
        NeonStoreF::raw(q2.1).write(dst.get_unchecked_mut(4 + dst_stride..));

        NeonStoreF::raw(q3.0).write_lo(dst.get_unchecked_mut(6..));
        NeonStoreF::raw(q3.1).write_lo(dst.get_unchecked_mut(6 + dst_stride..));

        NeonStoreF::raw(q4.0).write(dst.get_unchecked_mut(dst_stride * 2..));
        NeonStoreF::raw(q4.1).write(dst.get_unchecked_mut(dst_stride * 3..));

        NeonStoreF::raw(q5.0).write(dst.get_unchecked_mut(2 + dst_stride * 2..));
        NeonStoreF::raw(q5.1).write(dst.get_unchecked_mut(2 + dst_stride * 3..));

        NeonStoreF::raw(q6.0).write(dst.get_unchecked_mut(4 + dst_stride * 2..));
        NeonStoreF::raw(q6.1).write(dst.get_unchecked_mut(4 + dst_stride * 3..));

        NeonStoreF::raw(q7.0).write_lo(dst.get_unchecked_mut(6 + dst_stride * 2..));
        NeonStoreF::raw(q7.1).write_lo(dst.get_unchecked_mut(6 + dst_stride * 3..));
    }
}

#[inline]
pub(crate) fn block_transpose_f32x2_5x7(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let r0: [NeonStoreF; 7] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride..))
        });
        let r1: [NeonStoreF; 7] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 2..))
        });
        let r2: [NeonStoreF; 7] = std::array::from_fn(|x| {
            NeonStoreF::from_complex(src.get_unchecked(x * src_stride + 4))
        });

        let q0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r0[0].v, r0[1].v));
        let q1 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r0[2].v, r0[3].v));
        let q2 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r0[4].v, r0[5].v));
        let q3 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r0[6].v, vdupq_n_f32(0.)));

        let q4 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r1[0].v, r1[1].v));
        let q5 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r1[2].v, r1[3].v));
        let q6 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r1[4].v, r1[5].v));
        let q7 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r1[6].v, vdupq_n_f32(0.)));

        let q8 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r2[0].v, r2[1].v));
        let q9 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r2[2].v, r2[3].v));
        let q10 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r2[4].v, r2[5].v));
        let q11 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r2[6].v, vdupq_n_f32(0.)));

        NeonStoreF::raw(q0.0).write(dst);
        NeonStoreF::raw(q0.1).write(dst.get_unchecked_mut(dst_stride..));

        NeonStoreF::raw(q1.0).write(dst.get_unchecked_mut(2..));
        NeonStoreF::raw(q1.1).write(dst.get_unchecked_mut(2 + dst_stride..));

        NeonStoreF::raw(q2.0).write(dst.get_unchecked_mut(4..));
        NeonStoreF::raw(q2.1).write(dst.get_unchecked_mut(4 + dst_stride..));

        NeonStoreF::raw(q3.0).write_lo(dst.get_unchecked_mut(6..));
        NeonStoreF::raw(q3.1).write_lo(dst.get_unchecked_mut(6 + dst_stride..));

        NeonStoreF::raw(q4.0).write(dst.get_unchecked_mut(dst_stride * 2..));
        NeonStoreF::raw(q4.1).write(dst.get_unchecked_mut(dst_stride * 3..));

        NeonStoreF::raw(q5.0).write(dst.get_unchecked_mut(2 + dst_stride * 2..));
        NeonStoreF::raw(q5.1).write(dst.get_unchecked_mut(2 + dst_stride * 3..));

        NeonStoreF::raw(q6.0).write(dst.get_unchecked_mut(4 + dst_stride * 2..));
        NeonStoreF::raw(q6.1).write(dst.get_unchecked_mut(4 + dst_stride * 3..));

        NeonStoreF::raw(q7.0).write_lo(dst.get_unchecked_mut(6 + dst_stride * 2..));
        NeonStoreF::raw(q7.1).write_lo(dst.get_unchecked_mut(6 + dst_stride * 3..));

        NeonStoreF::raw(q8.0).write(dst.get_unchecked_mut(dst_stride * 4..));
        NeonStoreF::raw(q9.0).write(dst.get_unchecked_mut(2 + dst_stride * 4..));
        NeonStoreF::raw(q10.0).write(dst.get_unchecked_mut(4 + dst_stride * 4..));
        NeonStoreF::raw(q11.0).write_lo(dst.get_unchecked_mut(6 + dst_stride * 4..));
    }
}
