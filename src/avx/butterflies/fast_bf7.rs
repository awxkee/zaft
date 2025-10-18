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
use crate::avx::butterflies::AvxButterfly;
use crate::avx::rotate::AvxRotate;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxFastButterfly7<T> {
    rotate: AvxRotate<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}

impl<T: Copy + FftTrigonometry + Float + 'static> AvxFastButterfly7<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> Self {
        Self {
            rotate: AvxRotate::new(FftDirection::Inverse),
            twiddle1: compute_twiddle(1, 7, direction),
            twiddle2: compute_twiddle(2, 7, direction),
            twiddle3: compute_twiddle(3, 7, direction),
        }
    }
}

impl AvxFastButterfly7<f64> {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn exec(
        &self,
        u0: __m256d,
        u1: __m256d,
        u2: __m256d,
        u3: __m256d,
        u4: __m256d,
        u5: __m256d,
        u6: __m256d,
    ) -> (
        __m256d,
        __m256d,
        __m256d,
        __m256d,
        __m256d,
        __m256d,
        __m256d,
    ) {
        unsafe {
            let (x1p6, x1m6) = AvxButterfly::butterfly2_f64(u1, u6);
            let x1m6 = self.rotate.rotate_m256d(x1m6);
            let y00 = _mm256_add_pd(u0, x1p6);
            let (x2p5, x2m5) = AvxButterfly::butterfly2_f64(u2, u5);
            let x2m5 = self.rotate.rotate_m256d(x2m5);
            let y00 = _mm256_add_pd(y00, x2p5);
            let (x3p4, x3m4) = AvxButterfly::butterfly2_f64(u3, u4);
            let x3m4 = self.rotate.rotate_m256d(x3m4);
            let y00 = _mm256_add_pd(y00, x3p4);

            let m0106a = _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle1.re), u0);
            let m0106a = _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle2.re), m0106a);
            let m0106a = _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle3.re), m0106a);
            let m0106b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle1.im));
            let m0106b = _mm256_fmadd_pd(x2m5, _mm256_set1_pd(self.twiddle2.im), m0106b);
            let m0106b = _mm256_fmadd_pd(x3m4, _mm256_set1_pd(self.twiddle3.im), m0106b);
            let (y01, y06) = AvxButterfly::butterfly2_f64(m0106a, m0106b);

            let m0205a = _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle2.re), u0);
            let m0205a = _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle3.re), m0205a);
            let m0205a = _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle1.re), m0205a);
            let m0205b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle2.im));
            let m0205b = _mm256_fnmadd_pd(x2m5, _mm256_set1_pd(self.twiddle3.im), m0205b);
            let m0205b = _mm256_fnmadd_pd(x3m4, _mm256_set1_pd(self.twiddle1.im), m0205b);
            let (y02, y05) = AvxButterfly::butterfly2_f64(m0205a, m0205b);

            let m0304a = _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle3.re), u0);
            let m0304a = _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle1.re), m0304a);
            let m0304a = _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle2.re), m0304a);
            let m0304b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle3.im));
            let m0304b = _mm256_fnmadd_pd(x2m5, _mm256_set1_pd(self.twiddle1.im), m0304b);
            let m0304b = _mm256_fmadd_pd(x3m4, _mm256_set1_pd(self.twiddle2.im), m0304b);
            let (y03, y04) = AvxButterfly::butterfly2_f64(m0304a, m0304b);
            (y00, y01, y02, y03, y04, y05, y06)
        }
    }
}

impl AvxFastButterfly7<f32> {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn exec_short(
        &self,
        u0: __m128,
        u1: __m128,
        u2: __m128,
        u3: __m128,
        u4: __m128,
        u5: __m128,
        u6: __m128,
    ) -> (__m128, __m128, __m128, __m128, __m128, __m128, __m128) {
        unsafe {
            let (x1p6, x1m6) = AvxButterfly::butterfly2_f32_m128(u1, u6);
            let x1m6 = self.rotate.rotate_m128(x1m6);
            let y00 = _mm_add_ps(u0, x1p6);
            let (x2p5, x2m5) = AvxButterfly::butterfly2_f32_m128(u2, u5);
            let x2m5 = self.rotate.rotate_m128(x2m5);
            let y00 = _mm_add_ps(y00, x2p5);
            let (x3p4, x3m4) = AvxButterfly::butterfly2_f32_m128(u3, u4);
            let x3m4 = self.rotate.rotate_m128(x3m4);
            let y00 = _mm_add_ps(y00, x3p4);

            let m0106a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle1.re), u0);
            let m0106a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle2.re), m0106a);
            let m0106a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle3.re), m0106a);
            let m0106b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle1.im));
            let m0106b = _mm_fmadd_ps(x2m5, _mm_set1_ps(self.twiddle2.im), m0106b);
            let m0106b = _mm_fmadd_ps(x3m4, _mm_set1_ps(self.twiddle3.im), m0106b);
            let (y01, y06) = AvxButterfly::butterfly2_f32_m128(m0106a, m0106b);

            let m0205a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle2.re), u0);
            let m0205a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle3.re), m0205a);
            let m0205a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle1.re), m0205a);
            let m0205b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle2.im));
            let m0205b = _mm_fnmadd_ps(x2m5, _mm_set1_ps(self.twiddle3.im), m0205b);
            let m0205b = _mm_fnmadd_ps(x3m4, _mm_set1_ps(self.twiddle1.im), m0205b);
            let (y02, y05) = AvxButterfly::butterfly2_f32_m128(m0205a, m0205b);

            let m0304a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle3.re), u0);
            let m0304a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle1.re), m0304a);
            let m0304a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle2.re), m0304a);
            let m0304b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle3.im));
            let m0304b = _mm_fnmadd_ps(x2m5, _mm_set1_ps(self.twiddle1.im), m0304b);
            let m0304b = _mm_fmadd_ps(x3m4, _mm_set1_ps(self.twiddle2.im), m0304b);
            let (y03, y04) = AvxButterfly::butterfly2_f32_m128(m0304a, m0304b);
            (y00, y01, y02, y03, y04, y05, y06)
        }
    }
}
