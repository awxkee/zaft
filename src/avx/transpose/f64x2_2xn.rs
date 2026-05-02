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
use crate::avx::mixed::AvxStoreD;
use std::arch::x86_64::_mm256_permute2f128_pd;

#[inline(always)]
pub(crate) fn transpose_f64x2_2x5(rows: [AvxStoreD; 5]) -> [AvxStoreD; 6] {
    let a = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b = transpose_f64x2_2x2d([rows[2], rows[3]]);
    let c = transpose_f64x2_2x2d([rows[4], AvxStoreD::undefined()]);
    [a[0], a[1], b[0], b[1], c[0], c[1]]
}

#[inline(always)]
pub(crate) fn transpose_2x4d(rows: [AvxStoreD; 4]) -> [AvxStoreD; 4] {
    let a0 = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b0 = transpose_f64x2_2x2d([rows[2], rows[3]]);
    [a0[0], a0[1], b0[0], b0[1]]
}

#[inline(always)]
pub(crate) fn transpose_2x3d(rows: [AvxStoreD; 3]) -> [AvxStoreD; 4] {
    let a0 = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b0 = transpose_f64x2_2x2d([rows[2], AvxStoreD::undefined()]);
    [a0[0], a0[1], b0[0], b0[1]]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x6(rows: [AvxStoreD; 6]) -> [AvxStoreD; 6] {
    let a = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b = transpose_f64x2_2x2d([rows[2], rows[3]]);
    let c = transpose_f64x2_2x2d([rows[4], rows[5]]);
    [a[0], a[1], b[0], b[1], c[0], c[1]]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x7(rows: [AvxStoreD; 7]) -> [AvxStoreD; 8] {
    let a = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b = transpose_f64x2_2x2d([rows[2], rows[3]]);
    let c = transpose_f64x2_2x2d([rows[4], rows[5]]);
    let d = transpose_f64x2_2x2d([rows[6], AvxStoreD::undefined()]);
    [a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x8(rows: [AvxStoreD; 8]) -> [AvxStoreD; 8] {
    let a = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b = transpose_f64x2_2x2d([rows[2], rows[3]]);
    let c = transpose_f64x2_2x2d([rows[4], rows[5]]);
    let d = transpose_f64x2_2x2d([rows[6], rows[7]]);
    [a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x9(rows: [AvxStoreD; 9]) -> [AvxStoreD; 10] {
    let a = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b = transpose_f64x2_2x2d([rows[2], rows[3]]);
    let c = transpose_f64x2_2x2d([rows[4], rows[5]]);
    let d = transpose_f64x2_2x2d([rows[6], rows[7]]);
    let e = transpose_f64x2_2x2d([rows[8], AvxStoreD::undefined()]);
    [a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1], e[0], e[1]]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x10(rows: [AvxStoreD; 10]) -> [AvxStoreD; 10] {
    let a = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b = transpose_f64x2_2x2d([rows[2], rows[3]]);
    let c = transpose_f64x2_2x2d([rows[4], rows[5]]);
    let d = transpose_f64x2_2x2d([rows[6], rows[7]]);
    let e = transpose_f64x2_2x2d([rows[8], rows[9]]);
    [a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1], e[0], e[1]]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x11(rows: [AvxStoreD; 11]) -> [AvxStoreD; 12] {
    let a = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b = transpose_f64x2_2x2d([rows[2], rows[3]]);
    let c = transpose_f64x2_2x2d([rows[4], rows[5]]);
    let d = transpose_f64x2_2x2d([rows[6], rows[7]]);
    let e = transpose_f64x2_2x2d([rows[8], rows[9]]);
    let f = transpose_f64x2_2x2d([rows[10], AvxStoreD::undefined()]);
    [
        a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1], e[0], e[1], f[0], f[1],
    ]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x12(rows: [AvxStoreD; 12]) -> [AvxStoreD; 12] {
    let a = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b = transpose_f64x2_2x2d([rows[2], rows[3]]);
    let c = transpose_f64x2_2x2d([rows[4], rows[5]]);
    let d = transpose_f64x2_2x2d([rows[6], rows[7]]);
    let e = transpose_f64x2_2x2d([rows[8], rows[9]]);
    let f = transpose_f64x2_2x2d([rows[10], rows[11]]);
    [
        a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1], e[0], e[1], f[0], f[1],
    ]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x13(rows: [AvxStoreD; 13]) -> [AvxStoreD; 14] {
    let a = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b = transpose_f64x2_2x2d([rows[2], rows[3]]);
    let c = transpose_f64x2_2x2d([rows[4], rows[5]]);
    let d = transpose_f64x2_2x2d([rows[6], rows[7]]);
    let e = transpose_f64x2_2x2d([rows[8], rows[9]]);
    let f = transpose_f64x2_2x2d([rows[10], rows[11]]);
    let g = transpose_f64x2_2x2d([rows[12], AvxStoreD::undefined()]);
    [
        a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1], e[0], e[1], f[0], f[1], g[0], g[1],
    ]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x14(rows: [AvxStoreD; 14]) -> [AvxStoreD; 14] {
    let a = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b = transpose_f64x2_2x2d([rows[2], rows[3]]);
    let c = transpose_f64x2_2x2d([rows[4], rows[5]]);
    let d = transpose_f64x2_2x2d([rows[6], rows[7]]);
    let e = transpose_f64x2_2x2d([rows[8], rows[9]]);
    let f = transpose_f64x2_2x2d([rows[10], rows[11]]);
    let g = transpose_f64x2_2x2d([rows[12], rows[13]]);
    [
        a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1], e[0], e[1], f[0], f[1], g[0], g[1],
    ]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x15(rows: [AvxStoreD; 15]) -> [AvxStoreD; 16] {
    let a = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b = transpose_f64x2_2x2d([rows[2], rows[3]]);
    let c = transpose_f64x2_2x2d([rows[4], rows[5]]);
    let d = transpose_f64x2_2x2d([rows[6], rows[7]]);
    let e = transpose_f64x2_2x2d([rows[8], rows[9]]);
    let f = transpose_f64x2_2x2d([rows[10], rows[11]]);
    let g = transpose_f64x2_2x2d([rows[12], rows[13]]);
    let h = transpose_f64x2_2x2d([rows[14], AvxStoreD::undefined()]);
    [
        a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1], e[0], e[1], f[0], f[1], g[0], g[1], h[0],
        h[1],
    ]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x16(rows: [AvxStoreD; 16]) -> [AvxStoreD; 16] {
    let a = transpose_f64x2_2x2d([rows[0], rows[1]]);
    let b = transpose_f64x2_2x2d([rows[2], rows[3]]);
    let c = transpose_f64x2_2x2d([rows[4], rows[5]]);
    let d = transpose_f64x2_2x2d([rows[6], rows[7]]);
    let e = transpose_f64x2_2x2d([rows[8], rows[9]]);
    let f = transpose_f64x2_2x2d([rows[10], rows[11]]);
    let g = transpose_f64x2_2x2d([rows[12], rows[13]]);
    let h = transpose_f64x2_2x2d([rows[14], rows[15]]);
    [
        a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1], e[0], e[1], f[0], f[1], g[0], g[1], h[0],
        h[1],
    ]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x2d(v: [AvxStoreD; 2]) -> [AvxStoreD; 2] {
    const HI_HI: i32 = 0b0011_0001;
    const LO_LO: i32 = 0b0010_0000;
    unsafe {
        // a0 a1
        // a2 a3
        // --->
        // a1 a3
        // a0 a2

        let q0 = _mm256_permute2f128_pd::<LO_LO>(v[0].v, v[1].v);
        let q1 = _mm256_permute2f128_pd::<HI_HI>(v[0].v, v[1].v);
        [AvxStoreD::raw(q0), AvxStoreD::raw(q1)]
    }
}
