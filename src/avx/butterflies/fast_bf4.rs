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
use std::arch::x86_64::*;
use std::marker::PhantomData;

pub(crate) struct AvxFastButterfly4<T> {
    rotate: __m256d,
    phantom_data: PhantomData<T>,
}

impl AvxFastButterfly4<f64> {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> Self {
        unsafe {
            Self {
                rotate: _mm256_loadu_pd(match direction {
                    FftDirection::Inverse => {
                        [-0.0f64, 0.0, -0.0, 0.0, -0.0f64, 0.0, -0.0, 0.0].as_ptr()
                    }
                    FftDirection::Forward => {
                        [0.0f64, -0.0, 0.0, -0.0, 0.0f64, -0.0, 0.0, -0.0].as_ptr()
                    }
                }),
                phantom_data: PhantomData,
            }
        }
    }
}

impl AvxFastButterfly4<f32> {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> Self {
        unsafe {
            Self {
                rotate: _mm256_castps_pd(_mm256_loadu_ps(match direction {
                    FftDirection::Inverse => {
                        [-0.0f32, 0.0, -0.0, 0.0, -0.0f32, 0.0, -0.0, 0.0].as_ptr()
                    }
                    FftDirection::Forward => {
                        [0.0f32, -0.0, 0.0, -0.0, 0.0f32, -0.0, 0.0, -0.0].as_ptr()
                    }
                })),
                phantom_data: PhantomData,
            }
        }
    }
}

impl AvxFastButterfly4<f32> {
    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn exec_short(
        &self,
        a: __m128,
        b: __m128,
        c: __m128,
        d: __m128,
    ) -> (__m128, __m128, __m128, __m128) {
        let t0 = _mm_add_ps(a, c);
        let t1 = _mm_sub_ps(a, c);
        let t2 = _mm_add_ps(b, d);
        let mut t3 = _mm_sub_ps(b, d);
        const SH: i32 = shuffle(2, 3, 0, 1);
        t3 = _mm_xor_ps(
            _mm_shuffle_ps::<SH>(t3, t3),
            _mm256_castps256_ps128(_mm256_castpd_ps(self.rotate)),
        );
        (
            _mm_add_ps(t0, t2),
            _mm_add_ps(t1, t3),
            _mm_sub_ps(t0, t2),
            _mm_sub_ps(t1, t3),
        )
    }
}

impl AvxFastButterfly4<f64> {
    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn exec_short(
        &self,
        a: __m128d,
        b: __m128d,
        c: __m128d,
        d: __m128d,
    ) -> (__m128d, __m128d, __m128d, __m128d) {
        let t0 = _mm_add_pd(a, c);
        let t1 = _mm_sub_pd(a, c);
        let t2 = _mm_add_pd(b, d);
        let mut t3 = _mm_sub_pd(b, d);
        t3 = _mm_xor_pd(
            _mm_shuffle_pd::<0b01>(t3, t3),
            _mm256_castpd256_pd128(self.rotate),
        );
        (
            _mm_add_pd(t0, t2),
            _mm_add_pd(t1, t3),
            _mm_sub_pd(t0, t2),
            _mm_sub_pd(t1, t3),
        )
    }
}
