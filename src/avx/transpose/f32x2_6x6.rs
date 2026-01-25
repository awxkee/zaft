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
use crate::avx::transpose::avx_transpose_f32x2_4x4_impl;
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) fn transpose_6x6_f32(
    left: [__m256; 6],
    right: [__m256; 6],
) -> ([__m256; 6], [__m256; 6]) {
    unsafe {
        let tl = avx_transpose_f32x2_4x4_impl(left[0], left[1], left[2], left[3]);
        // Bottom-left 2x4 complex block (pad 2 rows with zeros)
        let bl = avx_transpose_f32x2_4x4_impl(
            left[4],
            left[5],
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );

        // Top-right 4x2 complex block (pad 2 columns with zeros to form 4x4)
        let tr = avx_transpose_f32x2_4x4_impl(right[0], right[1], right[2], right[3]);
        // Bottom-right 2x2 complex block
        let br = avx_transpose_f32x2_4x4_impl(
            right[4],
            right[5],
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );

        // Reassemble left 6 rows (first 4 columns)
        let output_left = [
            tl.0, tl.1, tl.2, tl.3, // top 4 rows
            tr.0, tr.1, // bottom 2 rows
        ];

        // Reassemble right 6 rows (last 2 columns)
        let output_right = [
            bl.0, bl.1, bl.2, bl.3, // top 4 rows
            br.0, br.1, // bottom 2 rows
        ];

        (output_left, output_right)
    }
}
