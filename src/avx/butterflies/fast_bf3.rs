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
use crate::avx::util::shuffle;
use crate::util::compute_twiddle;
use std::arch::x86_64::*;
use std::marker::PhantomData;

pub(crate) struct AvxFastButterfly3<T> {
    twiddle_re: __m256d,
    twiddle_im: __m256d,
    phantom_data: PhantomData<T>,
}

impl AvxFastButterfly3<f64> {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<f64>(1, 3, direction);
        unsafe {
            Self {
                twiddle_re: _mm256_set1_pd(twiddle.re),
                twiddle_im: _mm256_loadu_pd(
                    [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im].as_ptr(),
                ),
                phantom_data: PhantomData,
            }
        }
    }
}

impl AvxFastButterfly3<f32> {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<f32>(1, 3, direction);
        unsafe {
            Self {
                twiddle_re: _mm256_castps_pd(_mm256_set1_ps(twiddle.re)),
                twiddle_im: _mm256_castps_pd(_mm256_loadu_ps(
                    [
                        -twiddle.im,
                        twiddle.im,
                        -twiddle.im,
                        twiddle.im,
                        -twiddle.im,
                        twiddle.im,
                        -twiddle.im,
                        twiddle.im,
                    ]
                    .as_ptr(),
                )),
                phantom_data: PhantomData,
            }
        }
    }
}

impl AvxFastButterfly3<f64> {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn exec_short(
        &self,
        u0: __m128d,
        u1: __m128d,
        u2: __m128d,
    ) -> (__m128d, __m128d, __m128d) {
        let xp = _mm_add_pd(u1, u2);
        let xn = _mm_sub_pd(u1, u2);
        let sum = _mm_add_pd(u0, xp);

        let w_1 = _mm_fmadd_pd(_mm256_castpd256_pd128(self.twiddle_re), xp, u0);
        let xn_rot = _mm_shuffle_pd::<0b01>(xn, xn);

        let y0 = sum;
        let y1 = _mm_fmadd_pd(_mm256_castpd256_pd128(self.twiddle_im), xn_rot, w_1);
        let y2 = _mm_fnmadd_pd(_mm256_castpd256_pd128(self.twiddle_im), xn_rot, w_1);
        (y0, y1, y2)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn exec(
        &self,
        u0: __m256d,
        u1: __m256d,
        u2: __m256d,
    ) -> (__m256d, __m256d, __m256d) {
        let xp = _mm256_add_pd(u1, u2);
        let xn = _mm256_sub_pd(u1, u2);
        let sum = _mm256_add_pd(u0, xp);

        let w_1 = _mm256_fmadd_pd(self.twiddle_re, xp, u0);
        let xn_rot = _mm256_permute_pd::<0b0101>(xn);

        let y0 = sum;
        let y1 = _mm256_fmadd_pd(self.twiddle_im, xn_rot, w_1);
        let y2 = _mm256_fnmadd_pd(self.twiddle_im, xn_rot, w_1);
        (y0, y1, y2)
    }
}

impl AvxFastButterfly3<f32> {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn exec_short(
        &self,
        u0: __m128,
        u1: __m128,
        u2: __m128,
    ) -> (__m128, __m128, __m128) {
        let xp = _mm_add_ps(u1, u2);
        let xn = _mm_sub_ps(u1, u2);
        let sum = _mm_add_ps(u0, xp);

        const SH: i32 = shuffle(2, 3, 0, 1);
        let w_1 = _mm_fmadd_ps(
            _mm256_castps256_ps128(_mm256_castpd_ps(self.twiddle_re)),
            xp,
            u0,
        );
        let xn_rot = _mm_shuffle_ps::<SH>(xn, xn);

        let y0 = sum;
        let y1 = _mm_fmadd_ps(
            _mm256_castps256_ps128(_mm256_castpd_ps(self.twiddle_im)),
            xn_rot,
            w_1,
        );
        let y2 = _mm_fnmadd_ps(
            _mm256_castps256_ps128(_mm256_castpd_ps(self.twiddle_im)),
            xn_rot,
            w_1,
        );
        (y0, y1, y2)
    }

    // #[target_feature(enable = "avx", enable = "fma")]
    // #[inline]
    // pub(crate) unsafe fn exec(
    //     &self,
    //     u0: __m256,
    //     u1: __m256,
    //     u2: __m256,
    // ) -> (__m256, __m256, __m256) {
    //     let xp = _mm256_add_ps(u1, u2);
    //     let xn = _mm256_sub_ps(u1, u2);
    //     let sum = _mm256_add_ps(u0, xp);
    //
    //     const SH: i32 = shuffle(2, 3, 0, 1);
    //     let w_1 = _mm256_fmadd_ps(_mm256_castpd_ps(self.twiddle_re), xp, u0);
    //     let xn_rot = _mm256_shuffle_ps::<SH>(xn, xn);
    //
    //     let y0 = sum;
    //     let y1 = _mm256_fmadd_ps(xn_rot, _mm256_castpd_ps(self.twiddle_im), w_1);
    //     let y2 = _mm256_fnmadd_ps(xn_rot, _mm256_castpd_ps(self.twiddle_im), w_1);
    //     (y0, y1, y2)
    // }
}
