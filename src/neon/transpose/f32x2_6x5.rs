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
use num_complex::Complex;
use std::arch::aarch64::{float32x4x2_t, vdupq_n_f32};

#[inline(always)]
pub(crate) fn transpose_2x5(rows: [NeonStoreF; 5]) -> [NeonStoreF; 6] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, unsafe { vdupq_n_f32(0.) }));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
    ]
}

#[inline(always)]
pub(crate) fn transpose_6x5(
    rows0: [NeonStoreF; 5],
    rows1: [NeonStoreF; 5],
    rows2: [NeonStoreF; 5],
) -> [NeonStoreF; 18] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[2].v, rows0[3].v));
    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[4].v, unsafe { vdupq_n_f32(0.) }));

    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    let e0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[2].v, rows1[3].v));
    let h0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[4].v, unsafe { vdupq_n_f32(0.) }));

    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[0].v, rows2[1].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[2].v, rows2[3].v));
    let i0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[4].v, unsafe { vdupq_n_f32(0.) }));
    [
        // row 0
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
        NeonStoreF::raw(e0.0),
        NeonStoreF::raw(e0.1),
        NeonStoreF::raw(f0.0),
        NeonStoreF::raw(f0.1),
        NeonStoreF::raw(g0.0),
        NeonStoreF::raw(g0.1),
        NeonStoreF::raw(h0.0),
        NeonStoreF::raw(h0.1),
        NeonStoreF::raw(i0.0),
        NeonStoreF::raw(i0.1),
    ]
}

#[inline]
pub(crate) fn block_transpose_f32x2_6x5(
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

        let q = transpose_6x5(rows0, rows1, rows2);

        for i in 0..6 {
            q[i].write(dst.get_unchecked_mut(i * dst_stride..));
            q[i + 6].write(dst.get_unchecked_mut(i * dst_stride + 2..));
            q[i + 12].write_lo(dst.get_unchecked_mut(i * dst_stride + 4..));
        }
    }
}
