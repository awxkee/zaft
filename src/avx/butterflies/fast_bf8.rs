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
use num_traits::{AsPrimitive, Float};
use std::any::TypeId;
use std::arch::x86_64::*;

pub(crate) struct AvxFastButterfly8<T> {
    pub(crate) rotate: AvxRotate<T>,
    root2: __m256d,
}

impl<T: Copy + FftTrigonometry + Float + 'static> AvxFastButterfly8<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> Self {
        Self {
            rotate: AvxRotate::new(direction),
            root2: if TypeId::of::<T>() == TypeId::of::<f64>() {
                _mm256_set1_pd(0.5f64.sqrt())
            } else {
                _mm256_castps_pd(_mm256_set1_ps(0.5f32.sqrt()))
            },
        }
    }
}

impl AvxFastButterfly8<f64> {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn exec_short(
        &self,
        u0: __m128d,
        u1: __m128d,
        u2: __m128d,
        u3: __m128d,
        u4: __m128d,
        u5: __m128d,
        u6: __m128d,
        u7: __m128d,
    ) -> (
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
            let (u0, u2, u4, u6) = AvxButterfly::butterfly4h_f64(
                u0,
                u2,
                u4,
                u6,
                _mm256_castpd256_pd128(self.rotate.rot_flag),
            );
            let (u1, mut u3, mut u5, mut u7) = AvxButterfly::butterfly4h_f64(
                u1,
                u3,
                u5,
                u7,
                _mm256_castpd256_pd128(self.rotate.rot_flag),
            );

            u3 = _mm_mul_pd(
                _mm_add_pd(
                    _mm_xor_pd(
                        _mm_shuffle_pd::<0b01>(u3, u3),
                        _mm256_castpd256_pd128(self.rotate.rot_flag),
                    ),
                    u3,
                ),
                _mm256_castpd256_pd128(self.root2),
            );
            u5 = _mm_xor_pd(
                _mm_shuffle_pd::<0b01>(u5, u5),
                _mm256_castpd256_pd128(self.rotate.rot_flag),
            );
            u7 = _mm_mul_pd(
                _mm_sub_pd(
                    _mm_xor_pd(
                        _mm_shuffle_pd::<0b01>(u7, u7),
                        _mm256_castpd256_pd128(self.rotate.rot_flag),
                    ),
                    u7,
                ),
                _mm256_castpd256_pd128(self.root2),
            );

            let (y0, y1) = AvxButterfly::butterfly2_f64_m128(u0, u1);
            let (y2, y3) = AvxButterfly::butterfly2_f64_m128(u2, u3);
            let (y4, y5) = AvxButterfly::butterfly2_f64_m128(u4, u5);
            let (y6, y7) = AvxButterfly::butterfly2_f64_m128(u6, u7);
            (y0, y2, y4, y6, y1, y3, y5, y7)
        }
    }
}

impl AvxFastButterfly8<f32> {
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
        u7: __m128,
    ) -> (
        __m128,
        __m128,
        __m128,
        __m128,
        __m128,
        __m128,
        __m128,
        __m128,
    ) {
        unsafe {
            let (u0, u2, u4, u6) = AvxButterfly::butterfly4h_f32(
                u0,
                u2,
                u4,
                u6,
                _mm_castpd_ps(_mm256_castpd256_pd128(self.rotate.rot_flag)),
            );
            let (u1, mut u3, mut u5, mut u7) = AvxButterfly::butterfly4h_f32(
                u1,
                u3,
                u5,
                u7,
                _mm_castpd_ps(_mm256_castpd256_pd128(self.rotate.rot_flag)),
            );

            u3 = _mm_mul_ps(
                _mm_add_ps(self.rotate.rotate_m128(u3), u3),
                _mm_castpd_ps(_mm256_castpd256_pd128(self.root2)),
            );
            u5 = self.rotate.rotate_m128(u5);
            u7 = _mm_mul_ps(
                _mm_sub_ps(self.rotate.rotate_m128(u7), u7),
                _mm_castpd_ps(_mm256_castpd256_pd128(self.root2)),
            );

            let (y0, y1) = AvxButterfly::butterfly2_f32_m128(u0, u1);
            let (y2, y3) = AvxButterfly::butterfly2_f32_m128(u2, u3);
            let (y4, y5) = AvxButterfly::butterfly2_f32_m128(u4, u5);
            let (y6, y7) = AvxButterfly::butterfly2_f32_m128(u6, u7);
            (y0, y2, y4, y6, y1, y3, y5, y7)
        }
    }
}
