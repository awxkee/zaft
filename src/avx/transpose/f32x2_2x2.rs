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
pub(crate) unsafe fn transpose_u64_2x2_impl(v0: __m128, v1: __m128) -> (__m128, __m128) {
    let l = _mm_shuffle_pd::<0b00>(_mm_castps_pd(v0), _mm_castps_pd(v1));
    let h = _mm_shuffle_pd::<0b11>(_mm_castps_pd(v0), _mm_castps_pd(v1));

    (_mm_castpd_ps(l), _mm_castpd_ps(h))
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn avx_transpose_f32x2_2x2(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let row0 = _mm_loadu_ps(src.as_ptr().cast());
        let row1 = _mm_loadu_ps(src.get_unchecked(src_stride..).as_ptr().cast());

        let v0 = transpose_u64_2x2_impl(row0, row1);

        _mm_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), v0.0);
        _mm_storeu_ps(
            dst.get_unchecked_mut(dst_stride..).as_mut_ptr().cast(),
            v0.1,
        );
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f32_2x2_impl(v0: (__m256, __m256)) -> (__m256, __m256) {
    let l = _mm_shuffle_pd::<0b00>(
        _mm_castps_pd(_mm256_castps256_ps128(v0.0)),
        _mm_castps_pd(_mm256_castps256_ps128(v0.1)),
    );
    let h = _mm_shuffle_pd::<0b11>(
        _mm_castps_pd(_mm256_castps256_ps128(v0.0)),
        _mm_castps_pd(_mm256_castps256_ps128(v0.1)),
    );

    (
        _mm256_castps128_ps256(_mm_castpd_ps(l)),
        _mm256_castps128_ps256(_mm_castpd_ps(h)),
    )
}
