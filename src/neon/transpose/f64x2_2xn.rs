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
use crate::neon::mixed::NeonStoreD;

#[inline(always)]
pub(crate) fn transpose_f64x2_2x3(rows: [NeonStoreD; 3]) -> [NeonStoreD; 4] {
    let a0 = rows[0].transpose_2x2(rows[1]);
    let b0 = rows[2].transpose_2x2(NeonStoreD::default());
    [a0[0], a0[1], b0[0], b0[1]]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x9(rows: [NeonStoreD; 9]) -> [NeonStoreD; 10] {
    let a0 = rows[0].transpose_2x2(rows[1]);
    let b0 = rows[2].transpose_2x2(rows[3]);
    let c0 = rows[4].transpose_2x2(rows[5]);
    let d0 = rows[6].transpose_2x2(rows[7]);
    let f0 = rows[8].transpose_2x2(NeonStoreD::default());
    [
        a0[0], a0[1], b0[0], b0[1], c0[0], c0[1], d0[0], d0[1], f0[0], f0[1],
    ]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x11(rows: [NeonStoreD; 11]) -> [NeonStoreD; 12] {
    let a0 = rows[0].transpose_2x2(rows[1]);
    let b0 = rows[2].transpose_2x2(rows[3]);
    let c0 = rows[4].transpose_2x2(rows[5]);
    let d0 = rows[6].transpose_2x2(rows[7]);
    let f0 = rows[8].transpose_2x2(rows[9]);
    let e0 = rows[10].transpose_2x2(NeonStoreD::default());
    [
        a0[0], a0[1], b0[0], b0[1], c0[0], c0[1], d0[0], d0[1], f0[0], f0[1], e0[0], e0[1],
    ]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x7(rows: [NeonStoreD; 7]) -> [NeonStoreD; 8] {
    let a0 = rows[0].transpose_2x2(rows[1]);
    let b0 = rows[2].transpose_2x2(rows[3]);
    let c0 = rows[4].transpose_2x2(rows[5]);
    let d0 = rows[6].transpose_2x2(NeonStoreD::default());
    [a0[0], a0[1], b0[0], b0[1], c0[0], c0[1], d0[0], d0[1]]
}

#[inline(always)]
pub(crate) fn transpose_f64x2_2x5(rows: [NeonStoreD; 5]) -> [NeonStoreD; 6] {
    let a0 = rows[0].transpose_2x2(rows[1]);
    let b0 = rows[2].transpose_2x2(rows[3]);
    let c0 = rows[4].transpose_2x2(NeonStoreD::default());
    [a0[0], a0[1], b0[0], b0[1], c0[0], c0[1]]
}
