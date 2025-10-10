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
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx_transpose_u64_4x4_impl(
    v0: (__m256i, __m256i, __m256i, __m256i),
) -> (__m256i, __m256i, __m256i, __m256i) {
    // Unpack 32 bit elements. Goes from:
    // in[0]: 00 01 02 03
    // in[1]: 10 11 12 13
    // in[2]: 20 21 22 23
    // in[3]: 30 31 32 33
    // to:
    // a0:    00 10 02 12
    // a1:    20 30 22 32
    // a2:    01 11 03 13
    // a3:    21 31 23 33
    let a0 = _mm256_unpacklo_epi64(v0.0, v0.1);
    let a1 = _mm256_unpacklo_epi64(v0.2, v0.3);
    let a2 = _mm256_unpackhi_epi64(v0.0, v0.1);
    let a3 = _mm256_unpackhi_epi64(v0.2, v0.3);

    // Unpack 64 bit elements resulting in:
    // out[0]: 00 10 20 30
    // out[1]: 01 11 21 31
    // out[2]: 02 12 22 32
    // out[3]: 03 13 23 33

    const HI_HI: i32 = 0b0011_0001;
    const LO_LO: i32 = 0b0010_0000;

    let o0 = _mm256_permute2f128_si256::<LO_LO>(a0, a1);
    let o1 = _mm256_permute2f128_si256::<LO_LO>(a2, a3);
    let o2 = _mm256_permute2f128_si256::<HI_HI>(a0, a1);
    let o3 = _mm256_permute2f128_si256::<HI_HI>(a2, a3);
    (o0, o1, o2, o3)
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_transpose_f32x2_4x4(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let row0 = _mm256_loadu_si256(src.as_ptr().cast());
        let row1 = _mm256_loadu_si256(src.get_unchecked(src_stride..).as_ptr().cast());
        let row2 = _mm256_loadu_si256(src.get_unchecked(2 * src_stride..).as_ptr().cast());
        let row3 = _mm256_loadu_si256(src.get_unchecked(3 * src_stride..).as_ptr().cast());

        let v0 = avx_transpose_u64_4x4_impl((row0, row1, row2, row3));

        _mm256_storeu_si256(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), v0.0);
        _mm256_storeu_si256(
            dst.get_unchecked_mut(dst_stride..).as_mut_ptr().cast(),
            v0.1,
        );
        _mm256_storeu_si256(
            dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr().cast(),
            v0.2,
        );
        _mm256_storeu_si256(
            dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr().cast(),
            v0.3,
        );
    }
}
