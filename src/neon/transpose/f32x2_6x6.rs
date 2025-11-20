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
use crate::neon::transpose::f32x2_2x2::neon_transpose_f32x2_2x2_impl;
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) fn transpose_2x6(rows: [NeonStoreF; 6]) -> [NeonStoreF; 6] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[0].v, rows[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[2].v, rows[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows[4].v, rows[5].v));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
    ]
}

#[inline]
pub(crate) fn neon_transpose_f32x2_6x6(
    r0: float32x4x3_t,
    r1: float32x4x3_t,
    r2: float32x4x3_t,
    r3: float32x4x3_t,
    r4: float32x4x3_t,
    r5: float32x4x3_t,
) -> (
    float32x4x3_t,
    float32x4x3_t,
    float32x4x3_t,
    float32x4x3_t,
    float32x4x3_t,
    float32x4x3_t,
) {
    // Perform an 6 x 4 matrix transpose by building on top of the existing 2 x 2
    // matrix transpose implementation:
    // [ A B C ]^T => [ A^T D^T G^T ]
    // [ D E F ]      [ B^T E^T H^T ]
    // [ G H I ]      [ C^T F^T I^T ]

    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r0.0, r1.0));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r0.1, r1.1));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r0.2, r1.2));

    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r2.0, r3.0));
    let e0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r2.1, r3.1));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r2.2, r3.2));

    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r4.0, r5.0));
    let h0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r4.1, r5.1));
    let i0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r4.2, r5.2));

    (
        float32x4x3_t(a0.0, d0.0, g0.0),
        float32x4x3_t(a0.1, d0.1, g0.1),
        float32x4x3_t(b0.0, e0.0, h0.0),
        float32x4x3_t(b0.1, e0.1, h0.1),
        float32x4x3_t(c0.0, f0.0, i0.0),
        float32x4x3_t(c0.1, f0.1, i0.1),
    )
}

#[inline]
pub(crate) fn neon_transpose_f32x2_6x6_aos(
    v0: [NeonStoreF; 6],
    v1: [NeonStoreF; 6],
    v2: [NeonStoreF; 6],
) -> ([NeonStoreF; 6], [NeonStoreF; 6], [NeonStoreF; 6]) {
    // Perform an 6 x 4 matrix transpose by building on top of the existing 2 x 2
    // matrix transpose implementation:
    // [ A B C ]^T => [ A^T D^T G^T ]
    // [ D E F ]      [ B^T E^T H^T ]
    // [ G H I ]      [ C^T F^T I^T ]

    let chunk00 = float32x4x2_t(v0[0].v, v0[1].v);
    let chunk01 = float32x4x2_t(v0[2].v, v0[3].v);
    let chunk02 = float32x4x2_t(v0[4].v, v0[5].v);
    let chunk10 = float32x4x2_t(v1[0].v, v1[1].v);
    let chunk11 = float32x4x2_t(v1[2].v, v1[3].v);
    let chunk12 = float32x4x2_t(v1[4].v, v1[5].v);
    let chunk20 = float32x4x2_t(v2[0].v, v2[1].v);
    let chunk21 = float32x4x2_t(v2[2].v, v2[3].v);
    let chunk22 = float32x4x2_t(v2[4].v, v2[5].v);

    let output00 = neon_transpose_f32x2_2x2_impl(chunk00);
    let output01 = neon_transpose_f32x2_2x2_impl(chunk10);
    let output02 = neon_transpose_f32x2_2x2_impl(chunk20);
    let output10 = neon_transpose_f32x2_2x2_impl(chunk01);
    let output11 = neon_transpose_f32x2_2x2_impl(chunk11);
    let output12 = neon_transpose_f32x2_2x2_impl(chunk21);
    let output20 = neon_transpose_f32x2_2x2_impl(chunk02);
    let output21 = neon_transpose_f32x2_2x2_impl(chunk12);
    let output22 = neon_transpose_f32x2_2x2_impl(chunk22);

    (
        [
            NeonStoreF::raw(output00.0),
            NeonStoreF::raw(output00.1),
            NeonStoreF::raw(output01.0),
            NeonStoreF::raw(output01.1),
            NeonStoreF::raw(output02.0),
            NeonStoreF::raw(output02.1),
        ],
        [
            NeonStoreF::raw(output10.0),
            NeonStoreF::raw(output10.1),
            NeonStoreF::raw(output11.0),
            NeonStoreF::raw(output11.1),
            NeonStoreF::raw(output12.0),
            NeonStoreF::raw(output12.1),
        ],
        [
            NeonStoreF::raw(output20.0),
            NeonStoreF::raw(output20.1),
            NeonStoreF::raw(output21.0),
            NeonStoreF::raw(output21.1),
            NeonStoreF::raw(output22.0),
            NeonStoreF::raw(output22.1),
        ],
    )
}
