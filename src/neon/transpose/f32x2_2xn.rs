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
use std::arch::aarch64::{float32x4x2_t, vdupq_n_f32};

#[inline(always)]
pub(crate) fn transpose_2x2(rows: [NeonStoreF; 2]) -> [NeonStoreF; 2] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    [NeonStoreF::raw(a0.0), NeonStoreF::raw(a0.1)]
}

#[inline(always)]
pub(crate) fn transpose_2x3(rows: [NeonStoreF; 3]) -> [NeonStoreF; 4] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, unsafe { vdupq_n_f32(0.) }));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
    ]
}

#[inline(always)]
pub(crate) fn transpose_2x4(rows: [NeonStoreF; 4]) -> [NeonStoreF; 4] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
    ]
}

#[inline(always)]
pub(crate) fn transpose_2x12(rows: [NeonStoreF; 12]) -> [NeonStoreF; 12] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, rows[5].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[6].v, rows[7].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[8].v, rows[9].v));
    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[10].v, rows[11].v));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
        NeonStoreF::raw(f0.0),
        NeonStoreF::raw(f0.1),
        NeonStoreF::raw(g0.0),
        NeonStoreF::raw(g0.1),
    ]
}

#[inline(always)]
pub(crate) fn transpose_2x13(rows: [NeonStoreF; 13]) -> [NeonStoreF; 14] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, rows[5].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[6].v, rows[7].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[8].v, rows[9].v));
    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[10].v, rows[11].v));
    let h0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[12].v, unsafe { vdupq_n_f32(0.) }));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
        NeonStoreF::raw(f0.0),
        NeonStoreF::raw(f0.1),
        NeonStoreF::raw(g0.0),
        NeonStoreF::raw(g0.1),
        NeonStoreF::raw(h0.0),
        NeonStoreF::raw(h0.1),
    ]
}

#[inline(always)]
pub(crate) fn transpose_2x14(rows: [NeonStoreF; 14]) -> [NeonStoreF; 14] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, rows[5].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[6].v, rows[7].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[8].v, rows[9].v));
    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[10].v, rows[11].v));
    let h0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[12].v, rows[13].v));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
        NeonStoreF::raw(f0.0),
        NeonStoreF::raw(f0.1),
        NeonStoreF::raw(g0.0),
        NeonStoreF::raw(g0.1),
        NeonStoreF::raw(h0.0),
        NeonStoreF::raw(h0.1),
    ]
}

#[inline(always)]
pub(crate) fn transpose_2x15(rows: [NeonStoreF; 15]) -> [NeonStoreF; 16] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, rows[5].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[6].v, rows[7].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[8].v, rows[9].v));
    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[10].v, rows[11].v));
    let h0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[12].v, rows[13].v));
    let i0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[14].v, unsafe { vdupq_n_f32(0.) }));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
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

#[inline(always)]
pub(crate) fn transpose_2x16(rows: [NeonStoreF; 16]) -> [NeonStoreF; 16] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, rows[5].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[6].v, rows[7].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[8].v, rows[9].v));
    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[10].v, rows[11].v));
    let h0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[12].v, rows[13].v));
    let i0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[14].v, rows[15].v));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
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

#[inline(always)]
pub(crate) fn transpose_2x17(rows: [NeonStoreF; 17]) -> [NeonStoreF; 18] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, rows[5].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[6].v, rows[7].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[8].v, rows[9].v));
    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[10].v, rows[11].v));
    let h0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[12].v, rows[13].v));
    let i0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[14].v, rows[15].v));
    let j0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[16].v, unsafe { vdupq_n_f32(0.) }));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
        NeonStoreF::raw(f0.0),
        NeonStoreF::raw(f0.1),
        NeonStoreF::raw(g0.0),
        NeonStoreF::raw(g0.1),
        NeonStoreF::raw(h0.0),
        NeonStoreF::raw(h0.1),
        NeonStoreF::raw(i0.0),
        NeonStoreF::raw(i0.1),
        NeonStoreF::raw(j0.0),
        NeonStoreF::raw(j0.1),
    ]
}

#[inline(always)]
pub(crate) fn transpose_2x18(rows: [NeonStoreF; 18]) -> [NeonStoreF; 18] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, rows[5].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[6].v, rows[7].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[8].v, rows[9].v));
    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[10].v, rows[11].v));
    let h0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[12].v, rows[13].v));
    let i0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[14].v, rows[15].v));
    let j0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[16].v, rows[17].v));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
        NeonStoreF::raw(f0.0),
        NeonStoreF::raw(f0.1),
        NeonStoreF::raw(g0.0),
        NeonStoreF::raw(g0.1),
        NeonStoreF::raw(h0.0),
        NeonStoreF::raw(h0.1),
        NeonStoreF::raw(i0.0),
        NeonStoreF::raw(i0.1),
        NeonStoreF::raw(j0.0),
        NeonStoreF::raw(j0.1),
    ]
}

#[inline(always)]
pub(crate) fn transpose_2x19(rows: [NeonStoreF; 19]) -> [NeonStoreF; 20] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, rows[5].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[6].v, rows[7].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[8].v, rows[9].v));
    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[10].v, rows[11].v));
    let h0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[12].v, rows[13].v));
    let i0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[14].v, rows[15].v));
    let j0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[16].v, rows[17].v));
    let k0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[18].v, unsafe { vdupq_n_f32(0.) }));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
        NeonStoreF::raw(f0.0),
        NeonStoreF::raw(f0.1),
        NeonStoreF::raw(g0.0),
        NeonStoreF::raw(g0.1),
        NeonStoreF::raw(h0.0),
        NeonStoreF::raw(h0.1),
        NeonStoreF::raw(i0.0),
        NeonStoreF::raw(i0.1),
        NeonStoreF::raw(j0.0),
        NeonStoreF::raw(j0.1),
        NeonStoreF::raw(k0.0),
        NeonStoreF::raw(k0.1),
    ]
}

#[inline(always)]
pub(crate) fn transpose_2x20(rows: [NeonStoreF; 20]) -> [NeonStoreF; 20] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, rows[5].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[6].v, rows[7].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[8].v, rows[9].v));
    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[10].v, rows[11].v));
    let h0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[12].v, rows[13].v));
    let i0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[14].v, rows[15].v));
    let j0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[16].v, rows[17].v));
    let k0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[18].v, rows[19].v));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
        NeonStoreF::raw(f0.0),
        NeonStoreF::raw(f0.1),
        NeonStoreF::raw(g0.0),
        NeonStoreF::raw(g0.1),
        NeonStoreF::raw(h0.0),
        NeonStoreF::raw(h0.1),
        NeonStoreF::raw(i0.0),
        NeonStoreF::raw(i0.1),
        NeonStoreF::raw(j0.0),
        NeonStoreF::raw(j0.1),
        NeonStoreF::raw(k0.0),
        NeonStoreF::raw(k0.1),
    ]
}
