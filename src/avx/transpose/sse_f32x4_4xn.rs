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
use crate::avx::mixed::SseStoreF;
use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64};
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_sse_f32x4_4x4(store: [SseStoreF; 4]) -> [SseStoreF; 4] {
    let a0 = _mm_unpacklo_ps(store[0].v, store[1].v);
    let a1 = _mm_unpacklo_ps(store[2].v, store[3].v);
    let a2 = _mm_unpackhi_ps(store[0].v, store[1].v);
    let a3 = _mm_unpackhi_ps(store[2].v, store[3].v);

    // Unpack 64 bit elements resulting in:
    // out[0]: 00 10 20 30
    // out[1]: 01 11 21 31
    // out[2]: 02 12 22 32
    // out[3]: 03 13 23 33
    let r0 = _mm_unpacklo_ps64(a0, a1);
    let r1 = _mm_unpackhi_ps64(a0, a1);
    let r2 = _mm_unpacklo_ps64(a2, a3);
    let r3 = _mm_unpackhi_ps64(a2, a3);
    [
        SseStoreF::raw(r0),
        SseStoreF::raw(r1),
        SseStoreF::raw(r2),
        SseStoreF::raw(r3),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_sse_f32x4_4x3(rows: [SseStoreF; 3]) -> [SseStoreF; 4] {
    transpose_sse_f32x4_4x4([rows[0], rows[1], rows[2], SseStoreF::undefined()])
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_sse_f32x4_4x5(rows: [SseStoreF; 5]) -> [SseStoreF; 8] {
    let a0 = transpose_sse_f32x4_4x4([rows[0], rows[1], rows[2], rows[3]]);
    let b0 = transpose_sse_f32x4_4x4([
        rows[4],
        SseStoreF::undefined(),
        SseStoreF::undefined(),
        SseStoreF::undefined(),
    ]);
    [a0[0], a0[1], a0[2], a0[3], b0[0], b0[1], b0[2], b0[3]]
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_sse_f32x4_4x7(rows: [SseStoreF; 7]) -> [SseStoreF; 8] {
    let a0 = transpose_sse_f32x4_4x4([rows[0], rows[1], rows[2], rows[3]]);
    let b0 = transpose_sse_f32x4_4x4([rows[4], rows[5], rows[6], SseStoreF::undefined()]);
    [a0[0], a0[1], a0[2], a0[3], b0[0], b0[1], b0[2], b0[3]]
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_sse_f32x4_4x9(rows: [SseStoreF; 9]) -> [SseStoreF; 12] {
    let a0 = transpose_sse_f32x4_4x4([rows[0], rows[1], rows[2], rows[3]]);
    let b0 = transpose_sse_f32x4_4x4([rows[4], rows[5], rows[6], rows[7]]);
    let c0 = transpose_sse_f32x4_4x4([
        rows[8],
        SseStoreF::undefined(),
        SseStoreF::undefined(),
        SseStoreF::undefined(),
    ]);
    [
        a0[0], a0[1], a0[2], a0[3], b0[0], b0[1], b0[2], b0[3], c0[0], c0[1], c0[2], c0[3],
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_sse_f32x4_4x11(rows: [SseStoreF; 11]) -> [SseStoreF; 12] {
    let a0 = transpose_sse_f32x4_4x4([rows[0], rows[1], rows[2], rows[3]]);
    let b0 = transpose_sse_f32x4_4x4([rows[4], rows[5], rows[6], rows[7]]);
    let c0 = transpose_sse_f32x4_4x4([rows[8], rows[9], rows[10], SseStoreF::undefined()]);
    [
        a0[0], a0[1], a0[2], a0[3], b0[0], b0[1], b0[2], b0[3], c0[0], c0[1], c0[2], c0[3],
    ]
}
