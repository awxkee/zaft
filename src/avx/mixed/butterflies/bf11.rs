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
use crate::avx::mixed::avx_stored::AvxStoreD;
use crate::avx::mixed::avx_storef::AvxStoreF;
use crate::avx::rotate::AvxRotate;
use crate::util::compute_twiddle;
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct ColumnButterfly11d {
    rotate: AvxRotate<f64>,
    twiddle1: Complex<f64>,
    twiddle2: Complex<f64>,
    twiddle3: Complex<f64>,
    twiddle4: Complex<f64>,
    twiddle5: Complex<f64>,
}

impl ColumnButterfly11d {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> ColumnButterfly11d {
        Self {
            rotate: AvxRotate::<f64>::new(FftDirection::Inverse),
            twiddle1: compute_twiddle(1, 11, direction),
            twiddle2: compute_twiddle(2, 11, direction),
            twiddle3: compute_twiddle(3, 11, direction),
            twiddle4: compute_twiddle(4, 11, direction),
            twiddle5: compute_twiddle(5, 11, direction),
        }
    }
}

impl ColumnButterfly11d {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) fn exec(&self, v: [AvxStoreD; 11]) -> [AvxStoreD; 11] {
        let u0 = v[0].v;
        let y00 = u0;
        let (x1p10, x1m10) = AvxButterfly::butterfly2_f64(v[1].v, v[10].v);
        let x1m10 = self.rotate.rotate_m256d(x1m10);
        let y00 = _mm256_add_pd(y00, x1p10);
        let (x2p9, x2m9) = AvxButterfly::butterfly2_f64(v[2].v, v[9].v);
        let x2m9 = self.rotate.rotate_m256d(x2m9);
        let y00 = _mm256_add_pd(y00, x2p9);
        let (x3p8, x3m8) = AvxButterfly::butterfly2_f64(v[3].v, v[8].v);
        let x3m8 = self.rotate.rotate_m256d(x3m8);
        let y00 = _mm256_add_pd(y00, x3p8);
        let (x4p7, x4m7) = AvxButterfly::butterfly2_f64(v[4].v, v[7].v);
        let x4m7 = self.rotate.rotate_m256d(x4m7);
        let y00 = _mm256_add_pd(y00, x4p7);
        let (x5p6, x5m6) = AvxButterfly::butterfly2_f64(v[5].v, v[6].v);
        let x5m6 = self.rotate.rotate_m256d(x5m6);
        let y00 = _mm256_add_pd(y00, x5p6);

        let m0110a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle1.re), u0);
        let m0110a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle2.re), x2p9, m0110a);
        let m0110a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle3.re), x3p8, m0110a);
        let m0110a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle4.re), x4p7, m0110a);
        let m0110a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle5.re), x5p6, m0110a);
        let m0110b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle1.im));
        let m0110b = _mm256_fmadd_pd(x2m9, _mm256_set1_pd(self.twiddle2.im), m0110b);
        let m0110b = _mm256_fmadd_pd(x3m8, _mm256_set1_pd(self.twiddle3.im), m0110b);
        let m0110b = _mm256_fmadd_pd(x4m7, _mm256_set1_pd(self.twiddle4.im), m0110b);
        let m0110b = _mm256_fmadd_pd(x5m6, _mm256_set1_pd(self.twiddle5.im), m0110b);
        let (y01, y10) = AvxButterfly::butterfly2_f64(m0110a, m0110b);

        let m0209a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle2.re), u0);
        let m0209a = _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle4.re), m0209a);
        let m0209a = _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle5.re), m0209a);
        let m0209a = _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle3.re), m0209a);
        let m0209a = _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle1.re), m0209a);
        let m0209b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle2.im));
        let m0209b = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle4.im), x2m9, m0209b);
        let m0209b = _mm256_fnmadd_pd(x3m8, _mm256_set1_pd(self.twiddle5.im), m0209b);
        let m0209b = _mm256_fnmadd_pd(x4m7, _mm256_set1_pd(self.twiddle3.im), m0209b);
        let m0209b = _mm256_fnmadd_pd(x5m6, _mm256_set1_pd(self.twiddle1.im), m0209b);
        let (y02, y09) = AvxButterfly::butterfly2_f64(m0209a, m0209b);

        let m0308a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle3.re), u0);
        let m0308a = _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle5.re), m0308a);
        let m0308a = _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle2.re), m0308a);
        let m0308a = _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle1.re), m0308a);
        let m0308a = _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle4.re), m0308a);
        let m0308b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle3.im));
        let m0308b = _mm256_fnmadd_pd(x2m9, _mm256_set1_pd(self.twiddle5.im), m0308b);
        let m0308b = _mm256_fnmadd_pd(x3m8, _mm256_set1_pd(self.twiddle2.im), m0308b);
        let m0308b = _mm256_fmadd_pd(x4m7, _mm256_set1_pd(self.twiddle1.im), m0308b);
        let m0308b = _mm256_fmadd_pd(x5m6, _mm256_set1_pd(self.twiddle4.im), m0308b);
        let (y03, y08) = AvxButterfly::butterfly2_f64(m0308a, m0308b);

        let m0407a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle4.re), u0);
        let m0407a = _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle3.re), m0407a);
        let m0407a = _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle1.re), m0407a);
        let m0407a = _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle5.re), m0407a);
        let m0407a = _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle2.re), m0407a);
        let m0407b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle4.im));
        let m0407b = _mm256_fnmadd_pd(x2m9, _mm256_set1_pd(self.twiddle3.im), m0407b);
        let m0407b = _mm256_fmadd_pd(x3m8, _mm256_set1_pd(self.twiddle1.im), m0407b);
        let m0407b = _mm256_fmadd_pd(x4m7, _mm256_set1_pd(self.twiddle5.im), m0407b);
        let m0407b = _mm256_fnmadd_pd(x5m6, _mm256_set1_pd(self.twiddle2.im), m0407b);
        let (y04, y07) = AvxButterfly::butterfly2_f64(m0407a, m0407b);

        let m0506a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle5.re), u0);
        let m0506a = _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle1.re), m0506a);
        let m0506a = _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle4.re), m0506a);
        let m0506a = _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle2.re), m0506a);
        let m0506a = _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle3.re), m0506a);
        let m0506b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle5.im));
        let m0506b = _mm256_fnmadd_pd(x2m9, _mm256_set1_pd(self.twiddle1.im), m0506b);
        let m0506b = _mm256_fmadd_pd(x3m8, _mm256_set1_pd(self.twiddle4.im), m0506b);
        let m0506b = _mm256_fnmadd_pd(x4m7, _mm256_set1_pd(self.twiddle2.im), m0506b);
        let m0506b = _mm256_fmadd_pd(x5m6, _mm256_set1_pd(self.twiddle3.im), m0506b);
        let (y05, y06) = AvxButterfly::butterfly2_f64(m0506a, m0506b);

        [
            AvxStoreD::raw(y00),
            AvxStoreD::raw(y01),
            AvxStoreD::raw(y02),
            AvxStoreD::raw(y03),
            AvxStoreD::raw(y04),
            AvxStoreD::raw(y05),
            AvxStoreD::raw(y06),
            AvxStoreD::raw(y07),
            AvxStoreD::raw(y08),
            AvxStoreD::raw(y09),
            AvxStoreD::raw(y10),
        ]
    }
}

pub(crate) struct ColumnButterfly11f {
    rotate: AvxRotate<f32>,
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
    twiddle3: Complex<f32>,
    twiddle4: Complex<f32>,
    twiddle5: Complex<f32>,
}

impl ColumnButterfly11f {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> ColumnButterfly11f {
        Self {
            rotate: AvxRotate::<f32>::new(FftDirection::Inverse),
            twiddle1: compute_twiddle(1, 11, direction),
            twiddle2: compute_twiddle(2, 11, direction),
            twiddle3: compute_twiddle(3, 11, direction),
            twiddle4: compute_twiddle(4, 11, direction),
            twiddle5: compute_twiddle(5, 11, direction),
        }
    }
}

impl ColumnButterfly11f {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn exec(&self, v: [AvxStoreF; 11]) -> [AvxStoreF; 11] {
        let u0 = v[0].v;
        let y00 = u0;
        let (x1p10, x1m10) = AvxButterfly::butterfly2_f32(v[1].v, v[10].v);
        let x1m10 = self.rotate.rotate_m256(x1m10);
        let y00 = _mm256_add_ps(y00, x1p10);
        let (x2p9, x2m9) = AvxButterfly::butterfly2_f32(v[2].v, v[9].v);
        let x2m9 = self.rotate.rotate_m256(x2m9);
        let y00 = _mm256_add_ps(y00, x2p9);
        let (x3p8, x3m8) = AvxButterfly::butterfly2_f32(v[3].v, v[8].v);
        let x3m8 = self.rotate.rotate_m256(x3m8);
        let y00 = _mm256_add_ps(y00, x3p8);
        let (x4p7, x4m7) = AvxButterfly::butterfly2_f32(v[4].v, v[7].v);
        let x4m7 = self.rotate.rotate_m256(x4m7);
        let y00 = _mm256_add_ps(y00, x4p7);
        let (x5p6, x5m6) = AvxButterfly::butterfly2_f32(v[5].v, v[6].v);
        let x5m6 = self.rotate.rotate_m256(x5m6);
        let y00 = _mm256_add_ps(y00, x5p6);

        let m0110a = _mm256_fmadd_ps(x1p10, _mm256_set1_ps(self.twiddle1.re), u0);
        let m0110a = _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle2.re), x2p9, m0110a);
        let m0110a = _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle3.re), x3p8, m0110a);
        let m0110a = _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle4.re), x4p7, m0110a);
        let m0110a = _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle5.re), x5p6, m0110a);
        let m0110b = _mm256_mul_ps(x1m10, _mm256_set1_ps(self.twiddle1.im));
        let m0110b = _mm256_fmadd_ps(x2m9, _mm256_set1_ps(self.twiddle2.im), m0110b);
        let m0110b = _mm256_fmadd_ps(x3m8, _mm256_set1_ps(self.twiddle3.im), m0110b);
        let m0110b = _mm256_fmadd_ps(x4m7, _mm256_set1_ps(self.twiddle4.im), m0110b);
        let m0110b = _mm256_fmadd_ps(x5m6, _mm256_set1_ps(self.twiddle5.im), m0110b);
        let (y01, y10) = AvxButterfly::butterfly2_f32(m0110a, m0110b);

        let m0209a = _mm256_fmadd_ps(x1p10, _mm256_set1_ps(self.twiddle2.re), u0);
        let m0209a = _mm256_fmadd_ps(x2p9, _mm256_set1_ps(self.twiddle4.re), m0209a);
        let m0209a = _mm256_fmadd_ps(x3p8, _mm256_set1_ps(self.twiddle5.re), m0209a);
        let m0209a = _mm256_fmadd_ps(x4p7, _mm256_set1_ps(self.twiddle3.re), m0209a);
        let m0209a = _mm256_fmadd_ps(x5p6, _mm256_set1_ps(self.twiddle1.re), m0209a);
        let m0209b = _mm256_mul_ps(x1m10, _mm256_set1_ps(self.twiddle2.im));
        let m0209b = _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle4.im), x2m9, m0209b);
        let m0209b = _mm256_fnmadd_ps(x3m8, _mm256_set1_ps(self.twiddle5.im), m0209b);
        let m0209b = _mm256_fnmadd_ps(x4m7, _mm256_set1_ps(self.twiddle3.im), m0209b);
        let m0209b = _mm256_fnmadd_ps(x5m6, _mm256_set1_ps(self.twiddle1.im), m0209b);
        let (y02, y09) = AvxButterfly::butterfly2_f32(m0209a, m0209b);

        let m0308a = _mm256_fmadd_ps(x1p10, _mm256_set1_ps(self.twiddle3.re), u0);
        let m0308a = _mm256_fmadd_ps(x2p9, _mm256_set1_ps(self.twiddle5.re), m0308a);
        let m0308a = _mm256_fmadd_ps(x3p8, _mm256_set1_ps(self.twiddle2.re), m0308a);
        let m0308a = _mm256_fmadd_ps(x4p7, _mm256_set1_ps(self.twiddle1.re), m0308a);
        let m0308a = _mm256_fmadd_ps(x5p6, _mm256_set1_ps(self.twiddle4.re), m0308a);
        let m0308b = _mm256_mul_ps(x1m10, _mm256_set1_ps(self.twiddle3.im));
        let m0308b = _mm256_fnmadd_ps(x2m9, _mm256_set1_ps(self.twiddle5.im), m0308b);
        let m0308b = _mm256_fnmadd_ps(x3m8, _mm256_set1_ps(self.twiddle2.im), m0308b);
        let m0308b = _mm256_fmadd_ps(x4m7, _mm256_set1_ps(self.twiddle1.im), m0308b);
        let m0308b = _mm256_fmadd_ps(x5m6, _mm256_set1_ps(self.twiddle4.im), m0308b);
        let (y03, y08) = AvxButterfly::butterfly2_f32(m0308a, m0308b);

        let m0407a = _mm256_fmadd_ps(x1p10, _mm256_set1_ps(self.twiddle4.re), u0);
        let m0407a = _mm256_fmadd_ps(x2p9, _mm256_set1_ps(self.twiddle3.re), m0407a);
        let m0407a = _mm256_fmadd_ps(x3p8, _mm256_set1_ps(self.twiddle1.re), m0407a);
        let m0407a = _mm256_fmadd_ps(x4p7, _mm256_set1_ps(self.twiddle5.re), m0407a);
        let m0407a = _mm256_fmadd_ps(x5p6, _mm256_set1_ps(self.twiddle2.re), m0407a);
        let m0407b = _mm256_mul_ps(x1m10, _mm256_set1_ps(self.twiddle4.im));
        let m0407b = _mm256_fnmadd_ps(x2m9, _mm256_set1_ps(self.twiddle3.im), m0407b);
        let m0407b = _mm256_fmadd_ps(x3m8, _mm256_set1_ps(self.twiddle1.im), m0407b);
        let m0407b = _mm256_fmadd_ps(x4m7, _mm256_set1_ps(self.twiddle5.im), m0407b);
        let m0407b = _mm256_fnmadd_ps(x5m6, _mm256_set1_ps(self.twiddle2.im), m0407b);
        let (y04, y07) = AvxButterfly::butterfly2_f32(m0407a, m0407b);

        let m0506a = _mm256_fmadd_ps(x1p10, _mm256_set1_ps(self.twiddle5.re), u0);
        let m0506a = _mm256_fmadd_ps(x2p9, _mm256_set1_ps(self.twiddle1.re), m0506a);
        let m0506a = _mm256_fmadd_ps(x3p8, _mm256_set1_ps(self.twiddle4.re), m0506a);
        let m0506a = _mm256_fmadd_ps(x4p7, _mm256_set1_ps(self.twiddle2.re), m0506a);
        let m0506a = _mm256_fmadd_ps(x5p6, _mm256_set1_ps(self.twiddle3.re), m0506a);
        let m0506b = _mm256_mul_ps(x1m10, _mm256_set1_ps(self.twiddle5.im));
        let m0506b = _mm256_fnmadd_ps(x2m9, _mm256_set1_ps(self.twiddle1.im), m0506b);
        let m0506b = _mm256_fmadd_ps(x3m8, _mm256_set1_ps(self.twiddle4.im), m0506b);
        let m0506b = _mm256_fnmadd_ps(x4m7, _mm256_set1_ps(self.twiddle2.im), m0506b);
        let m0506b = _mm256_fmadd_ps(x5m6, _mm256_set1_ps(self.twiddle3.im), m0506b);
        let (y05, y06) = AvxButterfly::butterfly2_f32(m0506a, m0506b);

        [
            AvxStoreF::raw(y00),
            AvxStoreF::raw(y01),
            AvxStoreF::raw(y02),
            AvxStoreF::raw(y03),
            AvxStoreF::raw(y04),
            AvxStoreF::raw(y05),
            AvxStoreF::raw(y06),
            AvxStoreF::raw(y07),
            AvxStoreF::raw(y08),
            AvxStoreF::raw(y09),
            AvxStoreF::raw(y10),
        ]
    }
}
