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
use crate::avx::mixed::avx_stored::AvxStoreD;
use crate::avx::util::_mm256_fcmul_pd;
use crate::util::compute_twiddle;
use std::arch::x86_64::*;

pub(crate) struct ColumnButterfly9d {
    bf3: AvxFastButterfly3<f64>,
    twiddle1: __m256d,
    twiddle2: __m256d,
    twiddle4: __m256d,
}

impl ColumnButterfly9d {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> ColumnButterfly9d {
        let tw1 = compute_twiddle::<f64>(1, 9, direction);
        let tw2 = compute_twiddle::<f64>(2, 9, direction);
        let tw4 = compute_twiddle::<f64>(4, 9, direction);
        unsafe {
            Self {
                twiddle1: _mm256_loadu_pd([tw1.re, tw1.im, tw1.re, tw1.im].as_ptr().cast()),
                twiddle2: _mm256_loadu_pd([tw2.re, tw2.im, tw2.re, tw2.im].as_ptr().cast()),
                twiddle4: _mm256_loadu_pd([tw4.re, tw4.im, tw4.re, tw4.im].as_ptr().cast()),
                bf3: AvxFastButterfly3::<f64>::new(direction),
            }
        }
    }
}

impl ColumnButterfly9d {
    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn exec(&self, v: [AvxStoreD; 9]) -> [AvxStoreD; 9] {
        unsafe {
            let (u0, u3, u6) = self.bf3.exec(v[0].v, v[3].v, v[6].v);
            let (u1, mut u4, mut u7) = self.bf3.exec(v[1].v, v[4].v, v[7].v);
            let (u2, mut u5, mut u8) = self.bf3.exec(v[2].v, v[5].v, v[8].v);

            u4 = _mm256_fcmul_pd(u4, self.twiddle1);
            u7 = _mm256_fcmul_pd(u7, self.twiddle2);
            u5 = _mm256_fcmul_pd(u5, self.twiddle2);
            u8 = _mm256_fcmul_pd(u8, self.twiddle4);

            let (y0, y3, y6) = self.bf3.exec(u0, u1, u2);
            let (y1, y4, y7) = self.bf3.exec(u3, u4, u5);
            let (y2, y5, y8) = self.bf3.exec(u6, u7, u8);
            [
                AvxStoreD::raw(y0),
                AvxStoreD::raw(y1),
                AvxStoreD::raw(y2),
                AvxStoreD::raw(y3),
                AvxStoreD::raw(y4),
                AvxStoreD::raw(y5),
                AvxStoreD::raw(y6),
                AvxStoreD::raw(y7),
                AvxStoreD::raw(y8),
            ]
        }
    }
}
