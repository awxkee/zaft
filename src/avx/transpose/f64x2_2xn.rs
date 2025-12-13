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
use crate::avx::transpose::transpose_f64x2_2x2;
use std::arch::x86_64::{_mm256_permute2f128_pd, _mm256_setzero_pd};

#[inline(always)]
pub(crate) fn transpose_f64x2_2x5(rows: [AvxStoreD; 5]) -> [AvxStoreD; 6] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        let c0 = transpose_f64x2_2x2(rows[4].v, _mm256_setzero_pd());
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
            AvxStoreD::raw(c0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_2x4d(rows: [AvxStoreD; 4]) -> [AvxStoreD; 4] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_2x3d(rows: [AvxStoreD; 3]) -> [AvxStoreD; 4] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, _mm256_setzero_pd());
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x6(rows: [AvxStoreD; 6]) -> [AvxStoreD; 6] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        let c0 = transpose_f64x2_2x2(rows[4].v, rows[5].v);
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
            AvxStoreD::raw(c0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x7(rows: [AvxStoreD; 7]) -> [AvxStoreD; 8] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        let c0 = transpose_f64x2_2x2(rows[4].v, rows[5].v);
        let d0 = transpose_f64x2_2x2(rows[6].v, _mm256_setzero_pd());
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
            AvxStoreD::raw(c0.1),
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x8(rows: [AvxStoreD; 8]) -> [AvxStoreD; 8] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        let c0 = transpose_f64x2_2x2(rows[4].v, rows[5].v);
        let d0 = transpose_f64x2_2x2(rows[6].v, rows[7].v);
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
            AvxStoreD::raw(c0.1),
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x9(rows: [AvxStoreD; 9]) -> [AvxStoreD; 10] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        let c0 = transpose_f64x2_2x2(rows[4].v, rows[5].v);
        let d0 = transpose_f64x2_2x2(rows[6].v, rows[7].v);
        let e0 = transpose_f64x2_2x2(rows[8].v, _mm256_setzero_pd());
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
            AvxStoreD::raw(c0.1),
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
            AvxStoreD::raw(e0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x10(rows: [AvxStoreD; 10]) -> [AvxStoreD; 10] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        let c0 = transpose_f64x2_2x2(rows[4].v, rows[5].v);
        let d0 = transpose_f64x2_2x2(rows[6].v, rows[7].v);
        let e0 = transpose_f64x2_2x2(rows[8].v, rows[9].v);
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
            AvxStoreD::raw(c0.1),
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
            AvxStoreD::raw(e0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x11(rows: [AvxStoreD; 11]) -> [AvxStoreD; 12] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        let c0 = transpose_f64x2_2x2(rows[4].v, rows[5].v);
        let d0 = transpose_f64x2_2x2(rows[6].v, rows[7].v);
        let e0 = transpose_f64x2_2x2(rows[8].v, rows[9].v);
        let f0 = transpose_f64x2_2x2(rows[10].v, _mm256_setzero_pd());
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
            AvxStoreD::raw(c0.1),
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
            AvxStoreD::raw(e0.1),
            AvxStoreD::raw(f0.0),
            AvxStoreD::raw(f0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x12(rows: [AvxStoreD; 12]) -> [AvxStoreD; 12] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        let c0 = transpose_f64x2_2x2(rows[4].v, rows[5].v);
        let d0 = transpose_f64x2_2x2(rows[6].v, rows[7].v);
        let e0 = transpose_f64x2_2x2(rows[8].v, rows[9].v);
        let f0 = transpose_f64x2_2x2(rows[10].v, rows[11].v);
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
            AvxStoreD::raw(c0.1),
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
            AvxStoreD::raw(e0.1),
            AvxStoreD::raw(f0.0),
            AvxStoreD::raw(f0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x13(rows: [AvxStoreD; 13]) -> [AvxStoreD; 14] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        let c0 = transpose_f64x2_2x2(rows[4].v, rows[5].v);
        let d0 = transpose_f64x2_2x2(rows[6].v, rows[7].v);
        let e0 = transpose_f64x2_2x2(rows[8].v, rows[9].v);
        let f0 = transpose_f64x2_2x2(rows[10].v, rows[11].v);
        let g0 = transpose_f64x2_2x2(rows[12].v, _mm256_setzero_pd());
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
            AvxStoreD::raw(c0.1),
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
            AvxStoreD::raw(e0.1),
            AvxStoreD::raw(f0.0),
            AvxStoreD::raw(f0.1),
            AvxStoreD::raw(g0.0),
            AvxStoreD::raw(g0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x14(rows: [AvxStoreD; 14]) -> [AvxStoreD; 14] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        let c0 = transpose_f64x2_2x2(rows[4].v, rows[5].v);
        let d0 = transpose_f64x2_2x2(rows[6].v, rows[7].v);
        let e0 = transpose_f64x2_2x2(rows[8].v, rows[9].v);
        let f0 = transpose_f64x2_2x2(rows[10].v, rows[11].v);
        let g0 = transpose_f64x2_2x2(rows[12].v, rows[13].v);
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
            AvxStoreD::raw(c0.1),
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
            AvxStoreD::raw(e0.1),
            AvxStoreD::raw(f0.0),
            AvxStoreD::raw(f0.1),
            AvxStoreD::raw(g0.0),
            AvxStoreD::raw(g0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x15(rows: [AvxStoreD; 15]) -> [AvxStoreD; 16] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        let c0 = transpose_f64x2_2x2(rows[4].v, rows[5].v);
        let d0 = transpose_f64x2_2x2(rows[6].v, rows[7].v);
        let e0 = transpose_f64x2_2x2(rows[8].v, rows[9].v);
        let f0 = transpose_f64x2_2x2(rows[10].v, rows[11].v);
        let g0 = transpose_f64x2_2x2(rows[12].v, rows[13].v);
        let h0 = transpose_f64x2_2x2(rows[14].v, _mm256_setzero_pd());
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
            AvxStoreD::raw(c0.1),
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
            AvxStoreD::raw(e0.1),
            AvxStoreD::raw(f0.0),
            AvxStoreD::raw(f0.1),
            AvxStoreD::raw(g0.0),
            AvxStoreD::raw(g0.1),
            AvxStoreD::raw(h0.0),
            AvxStoreD::raw(h0.1),
        ]
    }
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x16(rows: [AvxStoreD; 16]) -> [AvxStoreD; 16] {
    unsafe {
        let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
        let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
        let c0 = transpose_f64x2_2x2(rows[4].v, rows[5].v);
        let d0 = transpose_f64x2_2x2(rows[6].v, rows[7].v);
        let e0 = transpose_f64x2_2x2(rows[8].v, rows[9].v);
        let f0 = transpose_f64x2_2x2(rows[10].v, rows[11].v);
        let g0 = transpose_f64x2_2x2(rows[12].v, rows[13].v);
        let h0 = transpose_f64x2_2x2(rows[14].v, rows[15].v);
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
            AvxStoreD::raw(c0.1),
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
            AvxStoreD::raw(e0.1),
            AvxStoreD::raw(f0.0),
            AvxStoreD::raw(f0.1),
            AvxStoreD::raw(g0.0),
            AvxStoreD::raw(g0.1),
            AvxStoreD::raw(h0.0),
            AvxStoreD::raw(h0.1),
        ]
    }
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
