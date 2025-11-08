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
use crate::neon::f32x2_2x2::neon_transpose_f32x2_2x2_impl;
use std::arch::aarch64::*;

#[inline]
pub(crate) fn transpose_f32x2_6x6(
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
    unsafe {
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
}
