/*
 * // Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly {}

impl AvxButterfly {
    // #[inline]
    // pub(crate) fn butterfly3_f32(
    //     u0: float32x4_t,
    //     u1: float32x4_t,
    //     u2: float32x4_t,
    //     tw_re: float32x4_t,
    //     tw_w_2: float32x4_t,
    // ) -> (float32x4_t, float32x4_t, float32x4_t) {
    //     unsafe {
    //         let xp = vaddq_f32(u1, u2);
    //         let xn = vsubq_f32(u1, u2);
    //         let sum = vaddq_f32(u0, xp);
    //
    //         let w_1 = vfmaq_f32(u0, tw_re, xp);
    //         let vw_2 = vmulq_f32(tw_w_2, vrev64q_f32(xn));
    //
    //         let y0 = sum;
    //         let y1 = vaddq_f32(w_1, vw_2);
    //         let y2 = vsubq_f32(w_1, vw_2);
    //         (y0, y1, y2)
    //     }
    // }
    //
    // #[inline]
    // pub(crate) fn butterfly2_f32(u0: float32x4_t, u1: float32x4_t) -> (float32x4_t, float32x4_t) {
    //     unsafe {
    //         let t = vaddq_f32(u0, u1);
    //
    //         let y1 = vsubq_f32(u0, u1);
    //         let y0 = t;
    //         (y0, y1)
    //     }
    // }

    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn butterfly3_f64(
        u0: __m256d,
        u1: __m256d,
        u2: __m256d,
        tw_re: __m256d,
        tw_w_2: __m256d,
    ) -> (__m256d, __m256d, __m256d) {
        let xp = _mm256_add_pd(u1, u2);
        let xn = _mm256_sub_pd(u1, u2);
        let sum = _mm256_add_pd(u0, xp);

        let w_1 = _mm256_fmadd_pd(tw_re, xp, u0);
        let vw_2 = _mm256_mul_pd(tw_w_2, _mm256_permute_pd::<0b0101>(xn));

        let y0 = sum;
        let y1 = _mm256_add_pd(w_1, vw_2);
        let y2 = _mm256_sub_pd(w_1, vw_2);
        (y0, y1, y2)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn butterfly3_f64_m128(
        u0: __m128d,
        u1: __m128d,
        u2: __m128d,
        tw_re: __m128d,
        tw_w_2: __m128d,
    ) -> (__m128d, __m128d, __m128d) {
        let xp = _mm_add_pd(u1, u2);
        let xn = _mm_sub_pd(u1, u2);
        let sum = _mm_add_pd(u0, xp);

        let w_1 = _mm_fmadd_pd(tw_re, xp, u0);
        let vw_2 = _mm_mul_pd(tw_w_2, _mm_shuffle_pd::<0b01>(xn, xn));

        let y0 = sum;
        let y1 = _mm_add_pd(w_1, vw_2);
        let y2 = _mm_sub_pd(w_1, vw_2);
        (y0, y1, y2)
    }

    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn butterfly2_f64_m128(u0: __m128d, u1: __m128d) -> (__m128d, __m128d) {
        let t = _mm_add_pd(u0, u1);
        let y1 = _mm_sub_pd(u0, u1);
        let y0 = t;
        (y0, y1)
    }

    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn butterfly2_f64(u0: __m256d, u1: __m256d) -> (__m256d, __m256d) {
        let t = _mm256_add_pd(u0, u1);
        let y1 = _mm256_sub_pd(u0, u1);
        let y0 = t;
        (y0, y1)
    }
}
