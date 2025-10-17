/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use crate::FftDirection;
use crate::avx::rotate::AvxRotate;
use crate::util::compute_twiddle;
use std::arch::x86_64::*;

pub(crate) struct AvxFastButterfly5d {
    rotate: AvxRotate<f64>,
    tw1_re: __m256d,
    tw1_im: __m256d,
    tw2_re: __m256d,
    tw2_im: __m256d,
}

pub(crate) struct AvxFastButterfly5f {
    rotate: AvxRotate<f32>,
    tw1_re: __m256,
    tw1_im: __m256,
    tw2_re: __m256,
    tw2_im: __m256,
}

impl AvxFastButterfly5d {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 5, direction);
        let tw2 = compute_twiddle(2, 5, direction);
        Self {
            rotate: AvxRotate::new(FftDirection::Inverse),
            tw1_re: _mm256_set1_pd(tw1.re),
            tw1_im: _mm256_set1_pd(tw1.im),
            tw2_re: _mm256_set1_pd(tw2.re),
            tw2_im: _mm256_set1_pd(tw2.im),
        }
    }
}

impl AvxFastButterfly5d {
    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn exec(
        &self,
        u0: __m256d,
        u1: __m256d,
        u2: __m256d,
        u3: __m256d,
        u4: __m256d,
    ) -> (__m256d, __m256d, __m256d, __m256d, __m256d) {
        unsafe {
            let x14p = _mm256_add_pd(u1, u4);
            let x14n = _mm256_sub_pd(u1, u4);
            let x23p = _mm256_add_pd(u2, u3);
            let x23n = _mm256_sub_pd(u2, u3);
            let y0 = _mm256_add_pd(_mm256_add_pd(u0, x14p), x23p);

            let temp_b1_1 = _mm256_mul_pd(self.tw1_im, x14n);
            let temp_b2_1 = _mm256_mul_pd(self.tw2_im, x14n);

            let temp_a1 =
                _mm256_fmadd_pd(self.tw2_re, x23p, _mm256_fmadd_pd(self.tw1_re, x14p, u0));
            let temp_a2 =
                _mm256_fmadd_pd(self.tw1_re, x23p, _mm256_fmadd_pd(self.tw2_re, x14p, u0));

            let temp_b1 = _mm256_fmadd_pd(self.tw2_im, x23n, temp_b1_1);
            let temp_b2 = _mm256_fnmadd_pd(self.tw1_im, x23n, temp_b2_1);

            let temp_b1_rot = self.rotate.rotate_m256d(temp_b1);
            let temp_b2_rot = self.rotate.rotate_m256d(temp_b2);

            let y1 = _mm256_add_pd(temp_a1, temp_b1_rot);
            let y2 = _mm256_add_pd(temp_a2, temp_b2_rot);
            let y3 = _mm256_sub_pd(temp_a2, temp_b2_rot);
            let y4 = _mm256_sub_pd(temp_a1, temp_b1_rot);
            (y0, y1, y2, y3, y4)
        }
    }
}

impl AvxFastButterfly5f {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 5, direction);
        let tw2 = compute_twiddle(2, 5, direction);
        Self {
            rotate: AvxRotate::new(FftDirection::Inverse),
            tw1_re: _mm256_set1_ps(tw1.re),
            tw1_im: _mm256_set1_ps(tw1.im),
            tw2_re: _mm256_set1_ps(tw2.re),
            tw2_im: _mm256_set1_ps(tw2.im),
        }
    }
}

impl AvxFastButterfly5f {
    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn _m256_exec(
        &self,
        u0: __m256,
        u1: __m256,
        u2: __m256,
        u3: __m256,
        u4: __m256,
    ) -> (__m256, __m256, __m256, __m256, __m256) {
        unsafe {
            let x14p = _mm256_add_ps(u1, u4);
            let x14n = _mm256_sub_ps(u1, u4);
            let x23p = _mm256_add_ps(u2, u3);
            let x23n = _mm256_sub_ps(u2, u3);
            let y0 = _mm256_add_ps(_mm256_add_ps(u0, x14p), x23p);

            let temp_b1_1 = _mm256_mul_ps(self.tw1_im, x14n);
            let temp_b2_1 = _mm256_mul_ps(self.tw2_im, x14n);

            let temp_a1 =
                _mm256_fmadd_ps(self.tw2_re, x23p, _mm256_fmadd_ps(self.tw1_re, x14p, u0));
            let temp_a2 =
                _mm256_fmadd_ps(self.tw1_re, x23p, _mm256_fmadd_ps(self.tw2_re, x14p, u0));

            let temp_b1 = _mm256_fmadd_ps(self.tw2_im, x23n, temp_b1_1);
            let temp_b2 = _mm256_fnmadd_ps(self.tw1_im, x23n, temp_b2_1);

            let temp_b1_rot = self.rotate.rotate_m256(temp_b1);
            let temp_b2_rot = self.rotate.rotate_m256(temp_b2);

            let y1 = _mm256_add_ps(temp_a1, temp_b1_rot);
            let y2 = _mm256_add_ps(temp_a2, temp_b2_rot);
            let y3 = _mm256_sub_ps(temp_a2, temp_b2_rot);
            let y4 = _mm256_sub_ps(temp_a1, temp_b1_rot);
            (y0, y1, y2, y3, y4)
        }
    }

    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn exec(
        &self,
        u0: __m128,
        u1: __m128,
        u2: __m128,
        u3: __m128,
        u4: __m128,
    ) -> (__m128, __m128, __m128, __m128, __m128) {
        unsafe {
            let x14p = _mm_add_ps(u1, u4);
            let x14n = _mm_sub_ps(u1, u4);
            let x23p = _mm_add_ps(u2, u3);
            let x23n = _mm_sub_ps(u2, u3);
            let y0 = _mm_add_ps(_mm_add_ps(u0, x14p), x23p);

            let temp_b1_1 = _mm_mul_ps(_mm256_castps256_ps128(self.tw1_im), x14n);
            let temp_b2_1 = _mm_mul_ps(_mm256_castps256_ps128(self.tw2_im), x14n);

            let temp_a1 = _mm_fmadd_ps(
                _mm256_castps256_ps128(self.tw2_re),
                x23p,
                _mm_fmadd_ps(_mm256_castps256_ps128(self.tw1_re), x14p, u0),
            );
            let temp_a2 = _mm_fmadd_ps(
                _mm256_castps256_ps128(self.tw1_re),
                x23p,
                _mm_fmadd_ps(_mm256_castps256_ps128(self.tw2_re), x14p, u0),
            );

            let temp_b1 = _mm_fmadd_ps(_mm256_castps256_ps128(self.tw2_im), x23n, temp_b1_1);
            let temp_b2 = _mm_fnmadd_ps(_mm256_castps256_ps128(self.tw1_im), x23n, temp_b2_1);

            let temp_b1_rot = self.rotate.rotate_m128(temp_b1);
            let temp_b2_rot = self.rotate.rotate_m128(temp_b2);

            let y1 = _mm_add_ps(temp_a1, temp_b1_rot);
            let y2 = _mm_add_ps(temp_a2, temp_b2_rot);
            let y3 = _mm_sub_ps(temp_a2, temp_b2_rot);
            let y4 = _mm_sub_ps(temp_a1, temp_b1_rot);
            (y0, y1, y2, y3, y4)
        }
    }
}
