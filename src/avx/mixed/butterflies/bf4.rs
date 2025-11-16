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
use crate::avx::mixed::SseStoreF;
use crate::avx::mixed::avx_stored::AvxStoreD;
use crate::avx::mixed::avx_storef::AvxStoreF;
use crate::avx::rotate::AvxRotate;
use std::arch::x86_64::*;

pub(crate) struct ColumnButterfly4d {
    rotate: __m256d,
}

impl ColumnButterfly4d {
    #[target_feature(enable = "avx")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly4d {
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
            }
        }
    }
}

impl ColumnButterfly4d {
    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) fn exec(&self, v: [AvxStoreD; 4]) -> [AvxStoreD; 4] {
        let t0 = _mm256_add_pd(v[0].v, v[2].v);
        let t1 = _mm256_sub_pd(v[0].v, v[2].v);
        let t2 = _mm256_add_pd(v[1].v, v[3].v);
        let mut t3 = _mm256_sub_pd(v[1].v, v[3].v);
        t3 = _mm256_xor_pd(_mm256_permute_pd::<0b0101>(t3), self.rotate);

        let y0 = _mm256_add_pd(t0, t2);
        let y1 = _mm256_add_pd(t1, t3);
        let y2 = _mm256_sub_pd(t0, t2);
        let y3 = _mm256_sub_pd(t1, t3);
        [
            AvxStoreD::raw(y0),
            AvxStoreD::raw(y1),
            AvxStoreD::raw(y2),
            AvxStoreD::raw(y3),
        ]
    }
}

pub(crate) struct ColumnButterfly4f {
    rotate: AvxRotate<f32>,
}

impl ColumnButterfly4f {
    #[target_feature(enable = "avx")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly4f {
        Self {
            rotate: AvxRotate::new(direction),
        }
    }
}

impl ColumnButterfly4f {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) fn exec(&self, v: [AvxStoreF; 4]) -> [AvxStoreF; 4] {
        let t0 = _mm256_add_ps(v[0].v, v[2].v);
        let t1 = _mm256_sub_ps(v[0].v, v[2].v);
        let t2 = _mm256_add_ps(v[1].v, v[3].v);
        let mut t3 = _mm256_sub_ps(v[1].v, v[3].v);
        t3 = self.rotate.rotate_m256(t3);

        let y0 = _mm256_add_ps(t0, t2);
        let y1 = _mm256_add_ps(t1, t3);
        let y2 = _mm256_sub_ps(t0, t2);
        let y3 = _mm256_sub_ps(t1, t3);
        [
            AvxStoreF::raw(y0),
            AvxStoreF::raw(y1),
            AvxStoreF::raw(y2),
            AvxStoreF::raw(y3),
        ]
    }

    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) fn exech(&self, v: [SseStoreF; 4]) -> [SseStoreF; 4] {
        let t0 = _mm_add_ps(v[0].v, v[2].v);
        let t1 = _mm_sub_ps(v[0].v, v[2].v);
        let t2 = _mm_add_ps(v[1].v, v[3].v);
        let mut t3 = _mm_sub_ps(v[1].v, v[3].v);
        t3 = self.rotate.rotate_m128(t3);

        let y0 = _mm_add_ps(t0, t2);
        let y1 = _mm_add_ps(t1, t3);
        let y2 = _mm_sub_ps(t0, t2);
        let y3 = _mm_sub_ps(t1, t3);
        [
            SseStoreF::raw(y0),
            SseStoreF::raw(y1),
            SseStoreF::raw(y2),
            SseStoreF::raw(y3),
        ]
    }
}
