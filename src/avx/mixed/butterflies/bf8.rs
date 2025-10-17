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
use crate::avx::mixed::avx_store::AvxStoreD;
use crate::avx::rotate::AvxRotate;
use std::arch::x86_64::*;

pub(crate) struct ColumnButterfly8d {
    rotate: AvxRotate<f64>,
    root2: __m256d,
}

impl ColumnButterfly8d {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> ColumnButterfly8d {
        unsafe {
            Self {
                rotate: AvxRotate::new(direction),
                root2: _mm256_set1_pd(0.5f64.sqrt()),
            }
        }
    }
}

impl ColumnButterfly8d {
    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn exec(&self, v: [AvxStoreD; 8]) -> [AvxStoreD; 8] {
        unsafe {
            let (u0, u2, u4, u6) =
                AvxButterfly::butterfly4_f64(v[0].v, v[2].v, v[4].v, v[6].v, self.rotate.rot_flag);
            let (u1, mut u3, mut u5, mut u7) =
                AvxButterfly::butterfly4_f64(v[1].v, v[3].v, v[5].v, v[7].v, self.rotate.rot_flag);

            u3 = _mm256_mul_pd(_mm256_add_pd(self.rotate.rotate_m256d(u3), u3), self.root2);
            u5 = self.rotate.rotate_m256d(u5);
            u7 = _mm256_mul_pd(_mm256_sub_pd(self.rotate.rotate_m256d(u7), u7), self.root2);

            let (y0, y1) = AvxButterfly::butterfly2_f64(u0, u1);
            let (y2, y3) = AvxButterfly::butterfly2_f64(u2, u3);
            let (y4, y5) = AvxButterfly::butterfly2_f64(u4, u5);
            let (y6, y7) = AvxButterfly::butterfly2_f64(u6, u7);
            [
                AvxStoreD::raw(y0),
                AvxStoreD::raw(y2),
                AvxStoreD::raw(y4),
                AvxStoreD::raw(y6),
                AvxStoreD::raw(y1),
                AvxStoreD::raw(y3),
                AvxStoreD::raw(y5),
                AvxStoreD::raw(y7),
            ]
        }
    }
}
