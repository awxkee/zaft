/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
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
use std::arch::x86_64::*;

#[inline(always)]
fn transpose_f64x2_2x2(v0: (__m256d, __m256d)) -> (__m256d, __m256d) {
    unsafe {
        const HI_HI: i32 = 0b0011_0001;
        const LO_LO: i32 = 0b0010_0000;

        // a0 a1
        // a2 a3
        // --->
        // a1 a3
        // a0 a2

        let q0 = _mm256_permute2f128_pd::<LO_LO>(v0.0, v0.1);
        let q1 = _mm256_permute2f128_pd::<HI_HI>(v0.0, v0.1);
        (q0, q1)
    }
}

#[inline(always)]
pub(crate) fn avx_transpose_f64x2_6x6_impl(
    v0: [AvxStoreD; 6],
    v1: [AvxStoreD; 6],
    v2: [AvxStoreD; 6],
) -> ([AvxStoreD; 6], [AvxStoreD; 6], [AvxStoreD; 6]) {
    unsafe {
        // Perform an 6 x 4 matrix transpose by building on top of the existing 2 x 2
        // matrix transpose implementation:
        // [ A B C ]^T => [ A^T D^T G^T ]
        // [ D E F ]      [ B^T E^T H^T ]
        // [ G H I ]      [ C^T F^T I^T ]

        let chunk00 = (v0[0].v, v0[1].v);
        let chunk01 = (v0[2].v, v0[3].v);
        let chunk02 = (v0[4].v, v0[5].v);
        let chunk10 = (v1[0].v, v1[1].v);
        let chunk11 = (v1[2].v, v1[3].v);
        let chunk12 = (v1[4].v, v1[5].v);
        let chunk20 = (v2[0].v, v2[1].v);
        let chunk21 = (v2[2].v, v2[3].v);
        let chunk22 = (v2[4].v, v2[5].v);

        let output00 = transpose_f64x2_2x2(chunk00);
        let output01 = transpose_f64x2_2x2(chunk10);
        let output02 = transpose_f64x2_2x2(chunk20);
        let output10 = transpose_f64x2_2x2(chunk01);
        let output11 = transpose_f64x2_2x2(chunk11);
        let output12 = transpose_f64x2_2x2(chunk21);
        let output20 = transpose_f64x2_2x2(chunk02);
        let output21 = transpose_f64x2_2x2(chunk12);
        let output22 = transpose_f64x2_2x2(chunk22);

        (
            [
                AvxStoreD::raw(output00.0),
                AvxStoreD::raw(output00.1),
                AvxStoreD::raw(output01.0),
                AvxStoreD::raw(output01.1),
                AvxStoreD::raw(output02.0),
                AvxStoreD::raw(output02.1),
            ],
            [
                AvxStoreD::raw(output10.0),
                AvxStoreD::raw(output10.1),
                AvxStoreD::raw(output11.0),
                AvxStoreD::raw(output11.1),
                AvxStoreD::raw(output12.0),
                AvxStoreD::raw(output12.1),
            ],
            [
                AvxStoreD::raw(output20.0),
                AvxStoreD::raw(output20.1),
                AvxStoreD::raw(output21.0),
                AvxStoreD::raw(output21.1),
                AvxStoreD::raw(output22.0),
                AvxStoreD::raw(output22.1),
            ],
        )
    }
}
