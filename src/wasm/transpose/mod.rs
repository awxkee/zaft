/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
mod fixed_block;

use crate::wasm::store::WasmStoreF;
pub(crate) use fixed_block::{
    WasmTransposeNx2F32, WasmTransposeNx3F32, WasmTransposeNx4F32, WasmTransposeNx5F32,
    WasmTransposeNx7F32, WasmTransposeNx8F32, WasmTransposeNx9F32,
};
use std::arch::wasm32::i64x2_shuffle;

#[inline(always)]
pub(crate) fn transpose_f32x2_2x2(v: [WasmStoreF; 2]) -> [WasmStoreF; 2] {
    let col0 = i64x2_shuffle::<0, 2>(v[0].v, v[1].v);
    let col1 = i64x2_shuffle::<1, 3>(v[0].v, v[1].v);
    [WasmStoreF::raw(col0), WasmStoreF::raw(col1)]
}

#[inline(always)]
pub(crate) fn transpose_f32x2_4x2(
    rows0: [WasmStoreF; 2],
    rows1: [WasmStoreF; 2],
) -> [WasmStoreF; 4] {
    let [a0, a1] = transpose_f32x2_2x2([rows0[0], rows0[1]]);
    let [b0, b1] = transpose_f32x2_2x2([rows1[0], rows1[1]]);
    [a0, a1, b0, b1]
}

#[inline(always)]
pub(crate) fn transpose_f32x2_4x4(
    rows0: [WasmStoreF; 4],
    rows1: [WasmStoreF; 4],
) -> ([WasmStoreF; 4], [WasmStoreF; 4]) {
    let a0 = transpose_f32x2_2x2([rows0[0], rows0[1]]);
    let d0 = transpose_f32x2_2x2([rows0[2], rows0[3]]);

    let b0 = transpose_f32x2_2x2([rows1[0], rows1[1]]);
    let e0 = transpose_f32x2_2x2([rows1[2], rows1[3]]);
    ([a0[0], a0[1], b0[0], b0[1]], [d0[0], d0[1], e0[0], e0[1]])
}

#[inline(always)]
pub(crate) fn transpose_f32x2_2x8(rows: [WasmStoreF; 8]) -> [WasmStoreF; 8] {
    let a0 = transpose_f32x2_2x2([rows[0], rows[1]]);
    let b0 = transpose_f32x2_2x2([rows[2], rows[3]]);
    let c0 = transpose_f32x2_2x2([rows[4], rows[5]]);
    let d0 = transpose_f32x2_2x2([rows[6], rows[7]]);
    [a0[0], a0[1], b0[0], b0[1], c0[0], c0[1], d0[0], d0[1]]
}

#[inline(always)]
pub(crate) fn transpose_f32x2_2x9(rows: [WasmStoreF; 9]) -> [WasmStoreF; 10] {
    let a0 = transpose_f32x2_2x2([rows[0], rows[1]]);
    let b0 = transpose_f32x2_2x2([rows[2], rows[3]]);
    let c0 = transpose_f32x2_2x2([rows[4], rows[5]]);
    let d0 = transpose_f32x2_2x2([rows[6], rows[7]]);
    let e0 = transpose_f32x2_2x2([rows[8], WasmStoreF::zero()]);
    [
        a0[0], a0[1], b0[0], b0[1], c0[0], c0[1], d0[0], d0[1], e0[0], e0[1],
    ]
}

#[inline(always)]
pub(crate) fn transpose_f32x2_2x7(rows: [WasmStoreF; 7]) -> [WasmStoreF; 8] {
    let a0 = transpose_f32x2_2x2([rows[0], rows[1]]);
    let b0 = transpose_f32x2_2x2([rows[2], rows[3]]);
    let c0 = transpose_f32x2_2x2([rows[4], rows[5]]);
    let d0 = transpose_f32x2_2x2([rows[6], WasmStoreF::zero()]);
    [a0[0], a0[1], b0[0], b0[1], c0[0], c0[1], d0[0], d0[1]]
}

#[inline(always)]
pub(crate) fn transpose_f32x2_2x5(rows: [WasmStoreF; 5]) -> [WasmStoreF; 6] {
    let a0 = transpose_f32x2_2x2([rows[0], rows[1]]);
    let b0 = transpose_f32x2_2x2([rows[2], rows[3]]);
    let c0 = transpose_f32x2_2x2([rows[4], WasmStoreF::zero()]);
    [a0[0], a0[1], b0[0], b0[1], c0[0], c0[1]]
}

#[inline(always)]
pub(crate) fn transpose_f32x2_2x4(rows: [WasmStoreF; 4]) -> [WasmStoreF; 4] {
    let a0 = transpose_f32x2_2x2([rows[0], rows[1]]);
    let b0 = transpose_f32x2_2x2([rows[2], rows[3]]);
    [a0[0], a0[1], b0[0], b0[1]]
}

#[inline(always)]
pub(crate) fn transpose_f32x2_2x3(rows: [WasmStoreF; 3]) -> [WasmStoreF; 4] {
    let a0 = transpose_f32x2_2x2([rows[0], rows[1]]);
    let b0 = transpose_f32x2_2x2([rows[2], WasmStoreF::zero()]);
    [a0[0], a0[1], b0[0], b0[1]]
}

#[inline(always)]
pub(crate) fn transpose_2x16(rows: [WasmStoreF; 16]) -> [WasmStoreF; 16] {
    let a0 = transpose_f32x2_2x2([rows[0], rows[1]]);
    let b0 = transpose_f32x2_2x2([rows[2], rows[3]]);
    let c0 = transpose_f32x2_2x2([rows[4], rows[5]]);
    let d0 = transpose_f32x2_2x2([rows[6], rows[7]]);
    let f0 = transpose_f32x2_2x2([rows[8], rows[9]]);
    let g0 = transpose_f32x2_2x2([rows[10], rows[11]]);
    let h0 = transpose_f32x2_2x2([rows[12], rows[13]]);
    let i0 = transpose_f32x2_2x2([rows[14], rows[15]]);
    [
        a0[0], a0[1], b0[0], b0[1], c0[0], c0[1], d0[0], d0[1], f0[0], f0[1], g0[0], g0[1], h0[0],
        h0[1], i0[0], i0[1],
    ]
}
