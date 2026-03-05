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
use crate::neon::mixed::NeonStoreF;
use std::arch::aarch64::{
    float32x4_t, float32x4x2_t, float32x4x4_t, vreinterpretq_f32_f64, vreinterpretq_f64_f32,
    vtrn1q_f64, vtrn2q_f64, vtrnq_f32,
};

#[inline(always)]
pub(crate) fn vtrnq_f64_to_f32(a0: float32x4_t, a1: float32x4_t) -> float32x4x2_t {
    unsafe {
        let b0 = vreinterpretq_f32_f64(vtrn1q_f64(
            vreinterpretq_f64_f32(a0),
            vreinterpretq_f64_f32(a1),
        ));
        let b1 = vreinterpretq_f32_f64(vtrn2q_f64(
            vreinterpretq_f64_f32(a0),
            vreinterpretq_f64_f32(a1),
        ));
        float32x4x2_t(b0, b1)
    }
}

#[inline(always)]
pub(crate) fn neon_transpose_4x4(v0: float32x4x4_t) -> float32x4x4_t {
    unsafe {
        // Swap 32 bit elements. Goes from:
        // a0: 00 01 02 03
        // a1: 10 11 12 13
        // a2: 20 21 22 23
        // a3: 30 31 32 33
        // to:
        // b0.0: 00 10 02 12
        // b0.1: 01 11 03 13
        // b1.0: 20 30 22 32
        // b1.1: 21 31 23 33

        let b0 = vtrnq_f32(v0.0, v0.1);
        let b1 = vtrnq_f32(v0.2, v0.3);

        // Swap 64 bit elements resulting in:
        // c0.0: 00 10 20 30
        // c0.1: 02 12 22 32
        // c1.0: 01 11 21 31
        // c1.1: 03 13 23 33

        let c0 = vtrnq_f64_to_f32(b0.0, b1.0);
        let c1 = vtrnq_f64_to_f32(b0.1, b1.1);

        float32x4x4_t(c0.0, c1.0, c0.1, c1.1)
    }
}

#[inline(always)]
pub(crate) fn transpose_4x4_f32(store: [NeonStoreF; 4]) -> [NeonStoreF; 4] {
    let q = neon_transpose_4x4(float32x4x4_t(
        store[0].v, store[1].v, store[2].v, store[3].v,
    ));
    [
        NeonStoreF::raw(q.0),
        NeonStoreF::raw(q.1),
        NeonStoreF::raw(q.2),
        NeonStoreF::raw(q.3),
    ]
}

#[inline(always)]
pub(crate) fn transpose_f32x4_4x3(rows: [NeonStoreF; 3]) -> [NeonStoreF; 4] {
    let a0 = transpose_4x4_f32([rows[0], rows[1], rows[2], NeonStoreF::default()]);
    [a0[0], a0[1], a0[2], a0[3]]
}

#[inline(always)]
pub(crate) fn transpose_f32x4_4x5(rows: [NeonStoreF; 5]) -> [NeonStoreF; 8] {
    let a0 = transpose_4x4_f32([rows[0], rows[1], rows[2], rows[3]]);
    let b0 = transpose_4x4_f32([
        rows[4],
        NeonStoreF::default(),
        NeonStoreF::default(),
        NeonStoreF::default(),
    ]);
    [a0[0], a0[1], a0[2], a0[3], b0[0], b0[1], b0[2], b0[3]]
}

#[inline(always)]
pub(crate) fn transpose_f32x4_4x7(rows: [NeonStoreF; 7]) -> [NeonStoreF; 8] {
    let a0 = transpose_4x4_f32([rows[0], rows[1], rows[2], rows[3]]);
    let b0 = transpose_4x4_f32([rows[4], rows[5], rows[6], NeonStoreF::default()]);
    [a0[0], a0[1], a0[2], a0[3], b0[0], b0[1], b0[2], b0[3]]
}

#[inline(always)]
pub(crate) fn transpose_f32x4_4x9(rows: [NeonStoreF; 9]) -> [NeonStoreF; 12] {
    let a0 = transpose_4x4_f32([rows[0], rows[1], rows[2], rows[3]]);
    let b0 = transpose_4x4_f32([rows[4], rows[5], rows[6], rows[7]]);
    let c0 = transpose_4x4_f32([
        rows[8],
        NeonStoreF::default(),
        NeonStoreF::default(),
        NeonStoreF::default(),
    ]);
    [
        a0[0], a0[1], a0[2], a0[3], b0[0], b0[1], b0[2], b0[3], c0[0], c0[1], c0[2], c0[3],
    ]
}

#[inline(always)]
pub(crate) fn transpose_f32x4_4x11(rows: [NeonStoreF; 11]) -> [NeonStoreF; 12] {
    let a0 = transpose_4x4_f32([rows[0], rows[1], rows[2], rows[3]]);
    let b0 = transpose_4x4_f32([rows[4], rows[5], rows[6], rows[7]]);
    let c0 = transpose_4x4_f32([rows[8], rows[9], rows[10], NeonStoreF::default()]);
    [
        a0[0], a0[1], a0[2], a0[3], b0[0], b0[1], b0[2], b0[3], c0[0], c0[1], c0[2], c0[3],
    ]
}
