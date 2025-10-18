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
use crate::avx::util::shuffle;
use num_traits::AsPrimitive;
use std::arch::x86_64::*;
use std::marker::PhantomData;
use std::ops::Neg;

pub(crate) struct AvxFastButterfly4<T> {
    rotate: AvxRotate<T>,
    phantom_data: PhantomData<T>,
}

impl<T: Copy + 'static + Neg<Output = T>> AvxFastButterfly4<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> Self {
        unsafe {
            Self {
                rotate: AvxRotate::<T>::new(direction),
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
        unsafe {
            let t0 = _mm_add_ps(a, c);
            let t1 = _mm_sub_ps(a, c);
            let t2 = _mm_add_ps(b, d);
            let mut t3 = _mm_sub_ps(b, d);
            const SH: i32 = shuffle(2, 3, 0, 1);
            t3 = self.rotate.rotate_m128(t3);
            (
                _mm_add_ps(t0, t2),
                _mm_add_ps(t1, t3),
                _mm_sub_ps(t0, t2),
                _mm_sub_ps(t1, t3),
            )
        }
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
        unsafe {
            let t0 = _mm_add_pd(a, c);
            let t1 = _mm_sub_pd(a, c);
            let t2 = _mm_add_pd(b, d);
            let mut t3 = _mm_sub_pd(b, d);
            t3 = self.rotate.rotate_m128d(t3);
            (
                _mm_add_pd(t0, t2),
                _mm_add_pd(t1, t3),
                _mm_sub_pd(t0, t2),
                _mm_sub_pd(t1, t3),
            )
        }
    }

    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn exec(
        &self,
        a: __m256d,
        b: __m256d,
        c: __m256d,
        d: __m256d,
    ) -> (__m256d, __m256d, __m256d, __m256d) {
        unsafe {
            let t0 = _mm256_add_pd(a, c);
            let t1 = _mm256_sub_pd(a, c);
            let t2 = _mm256_add_pd(b, d);
            let mut t3 = _mm256_sub_pd(b, d);
            t3 = self.rotate.rotate_m256d(t3);
            (
                _mm256_add_pd(t0, t2),
                _mm256_add_pd(t1, t3),
                _mm256_sub_pd(t0, t2),
                _mm256_sub_pd(t1, t3),
            )
        }
    }
}
