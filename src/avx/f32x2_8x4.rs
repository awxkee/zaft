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

use crate::avx::f32x2_4x4::avx_transpose_u64_4x4_impl;
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_transpose_f32x2_8x4(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let a0 = _mm256_loadu_si256(src.as_ptr().cast());
        let a1 = _mm256_loadu_si256(src.get_unchecked(src_stride..).as_ptr().cast());
        let a2 = _mm256_loadu_si256(src.get_unchecked(2 * src_stride..).as_ptr().cast());
        let a3 = _mm256_loadu_si256(src.get_unchecked(3 * src_stride..).as_ptr().cast());

        let b0 = _mm256_loadu_si256(src.get_unchecked(4..).as_ptr().cast());
        let b1 = _mm256_loadu_si256(src.get_unchecked(4 + src_stride..).as_ptr().cast());
        let b2 = _mm256_loadu_si256(src.get_unchecked(4 + 2 * src_stride..).as_ptr().cast());
        let b3 = _mm256_loadu_si256(src.get_unchecked(4 + 3 * src_stride..).as_ptr().cast());

        let v0 = avx_transpose_u64_4x4_impl((a0, a1, a2, a3));
        let v1 = avx_transpose_u64_4x4_impl((b0, b1, b2, b3));

        _mm256_storeu_si256(dst.as_mut_ptr().cast(), v0.0);
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

        _mm256_storeu_si256(
            dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr().cast(),
            v1.0,
        );
        _mm256_storeu_si256(
            dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr().cast(),
            v1.1,
        );
        _mm256_storeu_si256(
            dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr().cast(),
            v1.2,
        );
        _mm256_storeu_si256(
            dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr().cast(),
            v1.3,
        );
    }
}
