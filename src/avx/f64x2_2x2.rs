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

use num_complex::Complex;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx")]
pub(crate) fn transpose_f64x2_2x2(v0: __m256d, v1: __m256d) -> (__m256d, __m256d) {
    const HI_HI: i32 = 0b0011_0001;
    const LO_LO: i32 = 0b0010_0000;

    // a0 a1
    // a2 a3
    // --->
    // a1 a3
    // a0 a2

    let q0 = _mm256_permute2f128_pd::<LO_LO>(v0, v1);
    let q1 = _mm256_permute2f128_pd::<HI_HI>(v0, v1);
    (q0, q1)
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) fn avx_transpose_f64x2_2x2(
    src: &[Complex<f64>],
    src_stride: usize,
    dst: &mut [Complex<f64>],
    dst_stride: usize,
) {
    unsafe {
        let row0 = _mm256_loadu_pd(src.as_ptr().cast());
        let row1 = _mm256_loadu_pd(src.get_unchecked(src_stride..).as_ptr().cast());

        let (v0, v1) = transpose_f64x2_2x2(row0, row1);

        _mm256_storeu_pd(dst.as_mut_ptr().cast(), v0);
        _mm256_storeu_pd(dst.get_unchecked_mut(dst_stride..).as_mut_ptr().cast(), v1);
    }
}
