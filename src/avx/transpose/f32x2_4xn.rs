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
use crate::avx::transpose::transpose_f32x2_4x4_aos;

#[inline(always)]
pub(crate) fn transpose_4x2(rows: [AvxStoreF; 2]) -> [AvxStoreF; 4] {
    transpose_f32x2_4x4_aos([
        rows[0],
        rows[1],
        AvxStoreF::undefined(),
        AvxStoreF::undefined(),
    ])
}

#[inline(always)]
pub(crate) fn transpose_4x3(rows: [AvxStoreF; 3]) -> [AvxStoreF; 4] {
    transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], AvxStoreF::undefined()])
}

#[inline(always)]
pub(crate) fn transpose_4x4(rows: [AvxStoreF; 4]) -> [AvxStoreF; 4] {
    transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]])
}

#[inline(always)]
pub(crate) fn transpose_4x5(rows: [AvxStoreF; 5]) -> [AvxStoreF; 8] {
    let a = transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]]);
    let b = transpose_f32x2_4x4_aos([
        rows[4],
        AvxStoreF::undefined(),
        AvxStoreF::undefined(),
        AvxStoreF::undefined(),
    ]);
    [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
}

#[inline(always)]
pub(crate) fn transpose_4x6(rows: [AvxStoreF; 6]) -> [AvxStoreF; 8] {
    let a = transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]]);
    let b = transpose_f32x2_4x4_aos([
        rows[4],
        rows[5],
        AvxStoreF::undefined(),
        AvxStoreF::undefined(),
    ]);
    [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
}

#[inline(always)]
pub(crate) fn transpose_4x7(rows: [AvxStoreF; 7]) -> [AvxStoreF; 8] {
    let a = transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]]);
    let b = transpose_f32x2_4x4_aos([rows[4], rows[5], rows[6], AvxStoreF::undefined()]);
    [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
}

#[inline(always)]
pub(crate) fn transpose_4x8(rows: [AvxStoreF; 8]) -> [AvxStoreF; 8] {
    let a = transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]]);
    let b = transpose_f32x2_4x4_aos([rows[4], rows[5], rows[6], rows[7]]);
    [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
}

#[inline(always)]
pub(crate) fn transpose_4x9(rows: [AvxStoreF; 9]) -> [AvxStoreF; 12] {
    let a = transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]]);
    let b = transpose_f32x2_4x4_aos([rows[4], rows[5], rows[6], rows[7]]);
    let c = transpose_f32x2_4x4_aos([
        rows[8],
        AvxStoreF::undefined(),
        AvxStoreF::undefined(),
        AvxStoreF::undefined(),
    ]);
    [
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3],
    ]
}

#[inline(always)]
pub(crate) fn transpose_4x10(rows: [AvxStoreF; 10]) -> [AvxStoreF; 12] {
    let a = transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]]);
    let b = transpose_f32x2_4x4_aos([rows[4], rows[5], rows[6], rows[7]]);
    let c = transpose_f32x2_4x4_aos([
        rows[8],
        rows[9],
        AvxStoreF::undefined(),
        AvxStoreF::undefined(),
    ]);
    [
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3],
    ]
}

#[inline(always)]
pub(crate) fn transpose_4x11(rows: [AvxStoreF; 11]) -> [AvxStoreF; 12] {
    let a = transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]]);
    let b = transpose_f32x2_4x4_aos([rows[4], rows[5], rows[6], rows[7]]);
    let c = transpose_f32x2_4x4_aos([rows[8], rows[9], rows[10], AvxStoreF::undefined()]);
    [
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3],
    ]
}

#[inline(always)]
pub(crate) fn transpose_4x12(rows: [AvxStoreF; 12]) -> [AvxStoreF; 12] {
    let a = transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]]);
    let b = transpose_f32x2_4x4_aos([rows[4], rows[5], rows[6], rows[7]]);
    let c = transpose_f32x2_4x4_aos([rows[8], rows[9], rows[10], rows[11]]);
    [
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3],
    ]
}

#[inline(always)]
pub(crate) fn transpose_4x13(rows: [AvxStoreF; 13]) -> [AvxStoreF; 16] {
    let a = transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]]);
    let b = transpose_f32x2_4x4_aos([rows[4], rows[5], rows[6], rows[7]]);
    let c = transpose_f32x2_4x4_aos([rows[8], rows[9], rows[10], rows[11]]);
    let d = transpose_f32x2_4x4_aos([
        rows[12],
        AvxStoreF::undefined(),
        AvxStoreF::undefined(),
        AvxStoreF::undefined(),
    ]);
    [
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3], d[0], d[1], d[2],
        d[3],
    ]
}

#[inline(always)]
pub(crate) fn transpose_4x14(rows: [AvxStoreF; 14]) -> [AvxStoreF; 16] {
    let a = transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]]);
    let b = transpose_f32x2_4x4_aos([rows[4], rows[5], rows[6], rows[7]]);
    let c = transpose_f32x2_4x4_aos([rows[8], rows[9], rows[10], rows[11]]);
    let d = transpose_f32x2_4x4_aos([
        rows[12],
        rows[13],
        AvxStoreF::undefined(),
        AvxStoreF::undefined(),
    ]);
    [
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3], d[0], d[1], d[2],
        d[3],
    ]
}

#[inline(always)]
pub(crate) fn transpose_4x15(rows: [AvxStoreF; 15]) -> [AvxStoreF; 16] {
    let a = transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]]);
    let b = transpose_f32x2_4x4_aos([rows[4], rows[5], rows[6], rows[7]]);
    let c = transpose_f32x2_4x4_aos([rows[8], rows[9], rows[10], rows[11]]);
    let d = transpose_f32x2_4x4_aos([rows[12], rows[13], rows[14], AvxStoreF::undefined()]);
    [
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3], d[0], d[1], d[2],
        d[3],
    ]
}

#[inline(always)]
pub(crate) fn transpose_4x16(rows: [AvxStoreF; 16]) -> [AvxStoreF; 16] {
    let a = transpose_f32x2_4x4_aos([rows[0], rows[1], rows[2], rows[3]]);
    let b = transpose_f32x2_4x4_aos([rows[4], rows[5], rows[6], rows[7]]);
    let c = transpose_f32x2_4x4_aos([rows[8], rows[9], rows[10], rows[11]]);
    let d = transpose_f32x2_4x4_aos([rows[12], rows[13], rows[14], rows[15]]);
    [
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3], d[0], d[1], d[2],
        d[3],
    ]
}
