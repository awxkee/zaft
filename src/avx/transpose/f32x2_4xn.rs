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

use crate::avx::mixed::AvxStoreF;
use crate::avx::transpose::f32x2_4x4::avx_transpose_f32x2_4x4_impl;
use std::arch::x86_64::_mm256_setzero_ps;

#[inline(always)]
pub(crate) fn transpose_4x2(rows: [AvxStoreF; 2]) -> [AvxStoreF; 4] {
    unsafe {
        let a0 = avx_transpose_f32x2_4x4_impl(
            rows[0].v,
            rows[1].v,
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        [
            AvxStoreF::raw(a0.0),
            AvxStoreF::raw(a0.1),
            AvxStoreF::raw(a0.2),
            AvxStoreF::raw(a0.3),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_4x3(rows: [AvxStoreF; 3]) -> [AvxStoreF; 4] {
    unsafe {
        let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, _mm256_setzero_ps());
        [
            AvxStoreF::raw(a0.0),
            AvxStoreF::raw(a0.1),
            AvxStoreF::raw(a0.2),
            AvxStoreF::raw(a0.3),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_4x4(rows: [AvxStoreF; 4]) -> [AvxStoreF; 4] {
    let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
    ]
}

#[inline(always)]
pub(crate) fn transpose_4x5(rows: [AvxStoreF; 5]) -> [AvxStoreF; 8] {
    unsafe {
        let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
        let b0 = avx_transpose_f32x2_4x4_impl(
            rows[4].v,
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        [
            AvxStoreF::raw(a0.0),
            AvxStoreF::raw(a0.1),
            AvxStoreF::raw(a0.2),
            AvxStoreF::raw(a0.3),
            AvxStoreF::raw(b0.0),
            AvxStoreF::raw(b0.1),
            AvxStoreF::raw(b0.2),
            AvxStoreF::raw(b0.3),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_4x6(rows: [AvxStoreF; 6]) -> [AvxStoreF; 8] {
    unsafe {
        let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
        let b0 = avx_transpose_f32x2_4x4_impl(
            rows[4].v,
            rows[5].v,
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        [
            AvxStoreF::raw(a0.0),
            AvxStoreF::raw(a0.1),
            AvxStoreF::raw(a0.2),
            AvxStoreF::raw(a0.3),
            AvxStoreF::raw(b0.0),
            AvxStoreF::raw(b0.1),
            AvxStoreF::raw(b0.2),
            AvxStoreF::raw(b0.3),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_4x7(rows: [AvxStoreF; 7]) -> [AvxStoreF; 8] {
    unsafe {
        let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
        let b0 = avx_transpose_f32x2_4x4_impl(rows[4].v, rows[5].v, rows[6].v, _mm256_setzero_ps());
        [
            AvxStoreF::raw(a0.0),
            AvxStoreF::raw(a0.1),
            AvxStoreF::raw(a0.2),
            AvxStoreF::raw(a0.3),
            AvxStoreF::raw(b0.0),
            AvxStoreF::raw(b0.1),
            AvxStoreF::raw(b0.2),
            AvxStoreF::raw(b0.3),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_4x8(rows: [AvxStoreF; 8]) -> [AvxStoreF; 8] {
    let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
    let b0 = avx_transpose_f32x2_4x4_impl(rows[4].v, rows[5].v, rows[6].v, rows[7].v);
    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
        AvxStoreF::raw(b0.0),
        AvxStoreF::raw(b0.1),
        AvxStoreF::raw(b0.2),
        AvxStoreF::raw(b0.3),
    ]
}

#[inline(always)]
pub(crate) fn transpose_4x9(rows: [AvxStoreF; 9]) -> [AvxStoreF; 12] {
    unsafe {
        let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
        let b0 = avx_transpose_f32x2_4x4_impl(rows[4].v, rows[5].v, rows[6].v, rows[7].v);
        let c0 = avx_transpose_f32x2_4x4_impl(
            rows[8].v,
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        [
            AvxStoreF::raw(a0.0),
            AvxStoreF::raw(a0.1),
            AvxStoreF::raw(a0.2),
            AvxStoreF::raw(a0.3),
            AvxStoreF::raw(b0.0),
            AvxStoreF::raw(b0.1),
            AvxStoreF::raw(b0.2),
            AvxStoreF::raw(b0.3),
            AvxStoreF::raw(c0.0),
            AvxStoreF::raw(c0.1),
            AvxStoreF::raw(c0.2),
            AvxStoreF::raw(c0.3),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_4x10(rows: [AvxStoreF; 10]) -> [AvxStoreF; 12] {
    unsafe {
        let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
        let b0 = avx_transpose_f32x2_4x4_impl(rows[4].v, rows[5].v, rows[6].v, rows[7].v);
        let c0 = avx_transpose_f32x2_4x4_impl(
            rows[8].v,
            rows[9].v,
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        [
            AvxStoreF::raw(a0.0),
            AvxStoreF::raw(a0.1),
            AvxStoreF::raw(a0.2),
            AvxStoreF::raw(a0.3),
            AvxStoreF::raw(b0.0),
            AvxStoreF::raw(b0.1),
            AvxStoreF::raw(b0.2),
            AvxStoreF::raw(b0.3),
            AvxStoreF::raw(c0.0),
            AvxStoreF::raw(c0.1),
            AvxStoreF::raw(c0.2),
            AvxStoreF::raw(c0.3),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_4x11(rows: [AvxStoreF; 11]) -> [AvxStoreF; 12] {
    unsafe {
        let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
        let b0 = avx_transpose_f32x2_4x4_impl(rows[4].v, rows[5].v, rows[6].v, rows[7].v);
        let c0 =
            avx_transpose_f32x2_4x4_impl(rows[8].v, rows[9].v, rows[10].v, _mm256_setzero_ps());
        [
            AvxStoreF::raw(a0.0),
            AvxStoreF::raw(a0.1),
            AvxStoreF::raw(a0.2),
            AvxStoreF::raw(a0.3),
            AvxStoreF::raw(b0.0),
            AvxStoreF::raw(b0.1),
            AvxStoreF::raw(b0.2),
            AvxStoreF::raw(b0.3),
            AvxStoreF::raw(c0.0),
            AvxStoreF::raw(c0.1),
            AvxStoreF::raw(c0.2),
            AvxStoreF::raw(c0.3),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_4x12(rows: [AvxStoreF; 12]) -> [AvxStoreF; 12] {
    let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
    let b0 = avx_transpose_f32x2_4x4_impl(rows[4].v, rows[5].v, rows[6].v, rows[7].v);
    let c0 = avx_transpose_f32x2_4x4_impl(rows[8].v, rows[9].v, rows[10].v, rows[11].v);
    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
        AvxStoreF::raw(b0.0),
        AvxStoreF::raw(b0.1),
        AvxStoreF::raw(b0.2),
        AvxStoreF::raw(b0.3),
        AvxStoreF::raw(c0.0),
        AvxStoreF::raw(c0.1),
        AvxStoreF::raw(c0.2),
        AvxStoreF::raw(c0.3),
    ]
}

#[inline(always)]
pub(crate) fn transpose_4x13(rows: [AvxStoreF; 13]) -> [AvxStoreF; 16] {
    unsafe {
        let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
        let b0 = avx_transpose_f32x2_4x4_impl(rows[4].v, rows[5].v, rows[6].v, rows[7].v);
        let c0 = avx_transpose_f32x2_4x4_impl(rows[8].v, rows[9].v, rows[10].v, rows[11].v);
        let d0 = avx_transpose_f32x2_4x4_impl(
            rows[12].v,
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        [
            AvxStoreF::raw(a0.0),
            AvxStoreF::raw(a0.1),
            AvxStoreF::raw(a0.2),
            AvxStoreF::raw(a0.3),
            AvxStoreF::raw(b0.0),
            AvxStoreF::raw(b0.1),
            AvxStoreF::raw(b0.2),
            AvxStoreF::raw(b0.3),
            AvxStoreF::raw(c0.0),
            AvxStoreF::raw(c0.1),
            AvxStoreF::raw(c0.2),
            AvxStoreF::raw(c0.3),
            AvxStoreF::raw(d0.0),
            AvxStoreF::raw(d0.1),
            AvxStoreF::raw(d0.2),
            AvxStoreF::raw(d0.3),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_4x14(rows: [AvxStoreF; 14]) -> [AvxStoreF; 16] {
    unsafe {
        let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
        let b0 = avx_transpose_f32x2_4x4_impl(rows[4].v, rows[5].v, rows[6].v, rows[7].v);
        let c0 = avx_transpose_f32x2_4x4_impl(rows[8].v, rows[9].v, rows[10].v, rows[11].v);
        let d0 = avx_transpose_f32x2_4x4_impl(
            rows[12].v,
            rows[13].v,
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        [
            AvxStoreF::raw(a0.0),
            AvxStoreF::raw(a0.1),
            AvxStoreF::raw(a0.2),
            AvxStoreF::raw(a0.3),
            AvxStoreF::raw(b0.0),
            AvxStoreF::raw(b0.1),
            AvxStoreF::raw(b0.2),
            AvxStoreF::raw(b0.3),
            AvxStoreF::raw(c0.0),
            AvxStoreF::raw(c0.1),
            AvxStoreF::raw(c0.2),
            AvxStoreF::raw(c0.3),
            AvxStoreF::raw(d0.0),
            AvxStoreF::raw(d0.1),
            AvxStoreF::raw(d0.2),
            AvxStoreF::raw(d0.3),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_4x15(rows: [AvxStoreF; 15]) -> [AvxStoreF; 16] {
    unsafe {
        let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
        let b0 = avx_transpose_f32x2_4x4_impl(rows[4].v, rows[5].v, rows[6].v, rows[7].v);
        let c0 = avx_transpose_f32x2_4x4_impl(rows[8].v, rows[9].v, rows[10].v, rows[11].v);
        let d0 =
            avx_transpose_f32x2_4x4_impl(rows[12].v, rows[13].v, rows[14].v, _mm256_setzero_ps());
        [
            AvxStoreF::raw(a0.0),
            AvxStoreF::raw(a0.1),
            AvxStoreF::raw(a0.2),
            AvxStoreF::raw(a0.3),
            AvxStoreF::raw(b0.0),
            AvxStoreF::raw(b0.1),
            AvxStoreF::raw(b0.2),
            AvxStoreF::raw(b0.3),
            AvxStoreF::raw(c0.0),
            AvxStoreF::raw(c0.1),
            AvxStoreF::raw(c0.2),
            AvxStoreF::raw(c0.3),
            AvxStoreF::raw(d0.0),
            AvxStoreF::raw(d0.1),
            AvxStoreF::raw(d0.2),
            AvxStoreF::raw(d0.3),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_4x16(rows: [AvxStoreF; 16]) -> [AvxStoreF; 16] {
    let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
    let b0 = avx_transpose_f32x2_4x4_impl(rows[4].v, rows[5].v, rows[6].v, rows[7].v);
    let c0 = avx_transpose_f32x2_4x4_impl(rows[8].v, rows[9].v, rows[10].v, rows[11].v);
    let d0 = avx_transpose_f32x2_4x4_impl(rows[12].v, rows[13].v, rows[14].v, rows[15].v);
    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
        AvxStoreF::raw(b0.0),
        AvxStoreF::raw(b0.1),
        AvxStoreF::raw(b0.2),
        AvxStoreF::raw(b0.3),
        AvxStoreF::raw(c0.0),
        AvxStoreF::raw(c0.1),
        AvxStoreF::raw(c0.2),
        AvxStoreF::raw(c0.3),
        AvxStoreF::raw(d0.0),
        AvxStoreF::raw(d0.1),
        AvxStoreF::raw(d0.2),
        AvxStoreF::raw(d0.3),
    ]
}
