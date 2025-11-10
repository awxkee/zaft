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
use crate::avx::butterflies::AvxFastButterfly3;
use crate::avx::util::{_mm256_create_pd, _mm256_fcmul_pd, _mm256_set_complexd};
use crate::util::compute_twiddle;
use std::arch::x86_64::*;

pub(crate) struct AvxFastButterfly9d {
    tw1: __m256d,
    tw2: __m256d,
    tw4: __m256d,
    pub(crate) bf3: AvxFastButterfly3<f64>,
}

impl AvxFastButterfly9d {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> Self {
        let tw1 = compute_twiddle::<f64>(1, 9, direction);
        let tw2 = compute_twiddle::<f64>(2, 9, direction);
        let tw4 = compute_twiddle::<f64>(4, 9, direction);
        unsafe {
            Self {
                tw1: _mm256_set_complexd(tw1),
                tw2: _mm256_set_complexd(tw2),
                tw4: _mm256_set_complexd(tw4),
                bf3: AvxFastButterfly3::<f64>::new(direction),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx", enable = "fma")]
    pub(crate) fn exec(
        &self,
        u0: __m256d,
        u1: __m256d,
        u2: __m256d,
        u3: __m256d,
        u4: __m256d,
        u5: __m256d,
        u6: __m256d,
        u7: __m256d,
        u8: __m256d,
    ) -> (
        __m256d,
        __m256d,
        __m256d,
        __m256d,
        __m256d,
        __m256d,
        __m256d,
        __m256d,
        __m256d,
    ) {
        let (u0, u3, u6) = self.bf3.exec(u0, u3, u6);
        let (u1, mut u4, mut u7) = self.bf3.exec(u1, u4, u7);
        let (u2, mut u5, mut u8) = self.bf3.exec(u2, u5, u8);

        u4 = _mm256_fcmul_pd(u4, self.tw1);
        u7 = _mm256_fcmul_pd(u7, self.tw2);
        u5 = _mm256_fcmul_pd(u5, self.tw2);
        u8 = _mm256_fcmul_pd(u8, self.tw4);

        let (y0, y3, y6) = self.bf3.exec(u0, u1, u2);
        let (y1, y4, y7) = self.bf3.exec(u3, u4, u5);
        let (y2, y5, y8) = self.bf3.exec(u6, u7, u8);
        (y0, y1, y2, y3, y4, y5, y6, y7, y8)
    }

    #[inline]
    #[target_feature(enable = "avx", enable = "fma")]
    pub(crate) fn exec_m128d(
        &self,
        u0: __m128d,
        u1: __m128d,
        u2: __m128d,
        u3: __m128d,
        u4: __m128d,
        u5: __m128d,
        u6: __m128d,
        u7: __m128d,
        u8: __m128d,
    ) -> (
        __m128d,
        __m128d,
        __m128d,
        __m128d,
        __m128d,
        __m128d,
        __m128d,
        __m128d,
        __m128d,
    ) {
        unsafe {
            let (u0, u3, u6) = self.bf3.exec_m128(u0, u3, u6);
            let (u1, mut u4, mut u7) = self.bf3.exec_m128(u1, u4, u7);
            let (u2, mut u5, mut u8) = self.bf3.exec_m128(u2, u5, u8);

            const LO_LO: i32 = 0b0010_0000;

            let u4u7 = _mm256_fcmul_pd(
                _mm256_create_pd(u4, u7),
                _mm256_permute2f128_pd::<LO_LO>(self.tw1, self.tw2),
            );
            u4 = _mm256_castpd256_pd128(u4u7);
            u7 = _mm256_extractf128_pd::<1>(u4u7);
            let u5u8 = _mm256_fcmul_pd(
                _mm256_create_pd(u5, u8),
                _mm256_permute2f128_pd::<LO_LO>(self.tw2, self.tw4),
            );
            u5 = _mm256_castpd256_pd128(u5u8);
            u8 = _mm256_extractf128_pd::<1>(u5u8);

            let (y0, y3, y6) = self.bf3.exec_m128(u0, u1, u2);
            let (y1, y4, y7) = self.bf3.exec_m128(u3, u4, u5);
            let (y2, y5, y8) = self.bf3.exec_m128(u6, u7, u8);
            (y0, y1, y2, y3, y4, y5, y6, y7, y8)
        }
    }
}
