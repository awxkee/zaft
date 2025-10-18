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

use crate::avx::f64x2_2x2::transpose_f64x2_2x2;
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn avx_transpose_f64x2_4x4(
    src: &[Complex<f64>],
    src_stride: usize,
    dst: &mut [Complex<f64>],
    dst_stride: usize,
) {
    unsafe {
        let a0 = _mm256_loadu_pd(src.as_ptr().cast());
        let a1 = _mm256_loadu_pd(src.get_unchecked(src_stride..).as_ptr().cast());

        let b0 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
        let b1 = _mm256_loadu_pd(src.get_unchecked(src_stride + 2..).as_ptr().cast());

        let c0 = _mm256_loadu_pd(src.get_unchecked(src_stride * 2..).as_ptr().cast());
        let c1 = _mm256_loadu_pd(src.get_unchecked(src_stride * 3..).as_ptr().cast());

        let d0 = _mm256_loadu_pd(src.get_unchecked(src_stride * 2 + 2..).as_ptr().cast());
        let d1 = _mm256_loadu_pd(src.get_unchecked(src_stride * 3 + 2..).as_ptr().cast());

        // Perform an 4 x 4 matrix transpose by building on top of the existing 2 x 2
        // matrix transpose implementation:
        // [ A B ]^T => [ A^T C^T ]
        // [ C D ]      [ B^T D^T ]

        let a = transpose_f64x2_2x2(a0, a1);
        let b = transpose_f64x2_2x2(b0, b1);
        let c = transpose_f64x2_2x2(c0, c1);
        let d = transpose_f64x2_2x2(d0, d1);

        _mm256_storeu_pd(dst.as_mut_ptr().cast(), a.0);
        _mm256_storeu_pd(dst.get_unchecked_mut(dst_stride..).as_mut_ptr().cast(), a.1);

        _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), c.0);
        _mm256_storeu_pd(
            dst.get_unchecked_mut(2 + dst_stride..).as_mut_ptr().cast(),
            c.1,
        );

        _mm256_storeu_pd(
            dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr().cast(),
            b.0,
        );
        _mm256_storeu_pd(
            dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr().cast(),
            b.1,
        );

        _mm256_storeu_pd(
            dst.get_unchecked_mut(2 + 2 * dst_stride..)
                .as_mut_ptr()
                .cast(),
            d.0,
        );
        _mm256_storeu_pd(
            dst.get_unchecked_mut(2 + 3 * dst_stride..)
                .as_mut_ptr()
                .cast(),
            d.1,
        );
    }
}
