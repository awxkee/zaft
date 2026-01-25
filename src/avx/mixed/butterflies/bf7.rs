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

pub(crate) struct ColumnButterfly7d {
    rotate: AvxRotate<f64>,
    twiddle1: Complex<f64>,
    twiddle2: Complex<f64>,
    twiddle3: Complex<f64>,
}

impl ColumnButterfly7d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly7d {
        Self {
            rotate: AvxRotate::new(FftDirection::Inverse),
            twiddle1: compute_twiddle(1, 7, direction),
            twiddle2: compute_twiddle(2, 7, direction),
            twiddle3: compute_twiddle(3, 7, direction),
        }
    }
}

impl ColumnButterfly7d {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn exec(&self, v: [AvxStoreD; 7]) -> [AvxStoreD; 7] {
        let (x1p6, x1m6) = AvxButterfly::butterfly2_f64(v[1].v, v[6].v);
        let x1m6 = self.rotate.rotate_m256d(x1m6);
        let y00 = _mm256_add_pd(v[0].v, x1p6);
        let (x2p5, x2m5) = AvxButterfly::butterfly2_f64(v[2].v, v[5].v);
        let x2m5 = self.rotate.rotate_m256d(x2m5);
        let y00 = _mm256_add_pd(y00, x2p5);
        let (x3p4, x3m4) = AvxButterfly::butterfly2_f64(v[3].v, v[4].v);
        let x3m4 = self.rotate.rotate_m256d(x3m4);
        let y00 = _mm256_add_pd(y00, x3p4);

        let m0106a = _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle1.re), v[0].v);
        let m0106a = _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle2.re), m0106a);
        let m0106a = _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle3.re), m0106a);
        let m0106b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle1.im));
        let m0106b = _mm256_fmadd_pd(x2m5, _mm256_set1_pd(self.twiddle2.im), m0106b);
        let m0106b = _mm256_fmadd_pd(x3m4, _mm256_set1_pd(self.twiddle3.im), m0106b);
        let (y01, y06) = AvxButterfly::butterfly2_f64(m0106a, m0106b);

        let m0205a = _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle2.re), v[0].v);
        let m0205a = _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle3.re), m0205a);
        let m0205a = _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle1.re), m0205a);
        let m0205b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle2.im));
        let m0205b = _mm256_fnmadd_pd(x2m5, _mm256_set1_pd(self.twiddle3.im), m0205b);
        let m0205b = _mm256_fnmadd_pd(x3m4, _mm256_set1_pd(self.twiddle1.im), m0205b);
        let (y02, y05) = AvxButterfly::butterfly2_f64(m0205a, m0205b);

        let m0304a = _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle3.re), v[0].v);
        let m0304a = _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle1.re), m0304a);
        let m0304a = _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle2.re), m0304a);
        let m0304b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle3.im));
        let m0304b = _mm256_fnmadd_pd(x2m5, _mm256_set1_pd(self.twiddle1.im), m0304b);
        let m0304b = _mm256_fmadd_pd(x3m4, _mm256_set1_pd(self.twiddle2.im), m0304b);
        let (y03, y04) = AvxButterfly::butterfly2_f64(m0304a, m0304b);
        [
            AvxStoreD::raw(y00),
            AvxStoreD::raw(y01),
            AvxStoreD::raw(y02),
            AvxStoreD::raw(y03),
            AvxStoreD::raw(y04),
            AvxStoreD::raw(y05),
            AvxStoreD::raw(y06),
        ]
    }
}

pub(crate) struct ColumnButterfly7f {
    rotate: AvxRotate<f32>,
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
    twiddle3: Complex<f32>,
}

impl ColumnButterfly7f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly7f {
        Self {
            rotate: AvxRotate::new(FftDirection::Inverse),
            twiddle1: compute_twiddle(1, 7, direction),
            twiddle2: compute_twiddle(2, 7, direction),
            twiddle3: compute_twiddle(3, 7, direction),
        }
    }
}

impl ColumnButterfly7f {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 7]) -> [AvxStoreF; 7] {
        let (x1p6, x1m6) = AvxButterfly::butterfly2_f32(v[1].v, v[6].v);
        let x1m6 = self.rotate.rotate_m256(x1m6);
        let y00 = _mm256_add_ps(v[0].v, x1p6);
        let (x2p5, x2m5) = AvxButterfly::butterfly2_f32(v[2].v, v[5].v);
        let x2m5 = self.rotate.rotate_m256(x2m5);
        let y00 = _mm256_add_ps(y00, x2p5);
        let (x3p4, x3m4) = AvxButterfly::butterfly2_f32(v[3].v, v[4].v);
        let x3m4 = self.rotate.rotate_m256(x3m4);
        let y00 = _mm256_add_ps(y00, x3p4);

        let m0106a = _mm256_fmadd_ps(x1p6, _mm256_set1_ps(self.twiddle1.re), v[0].v);
        let m0106a = _mm256_fmadd_ps(x2p5, _mm256_set1_ps(self.twiddle2.re), m0106a);
        let m0106a = _mm256_fmadd_ps(x3p4, _mm256_set1_ps(self.twiddle3.re), m0106a);
        let m0106b = _mm256_mul_ps(x1m6, _mm256_set1_ps(self.twiddle1.im));
        let m0106b = _mm256_fmadd_ps(x2m5, _mm256_set1_ps(self.twiddle2.im), m0106b);
        let m0106b = _mm256_fmadd_ps(x3m4, _mm256_set1_ps(self.twiddle3.im), m0106b);
        let (y01, y06) = AvxButterfly::butterfly2_f32(m0106a, m0106b);

        let m0205a = _mm256_fmadd_ps(x1p6, _mm256_set1_ps(self.twiddle2.re), v[0].v);
        let m0205a = _mm256_fmadd_ps(x2p5, _mm256_set1_ps(self.twiddle3.re), m0205a);
        let m0205a = _mm256_fmadd_ps(x3p4, _mm256_set1_ps(self.twiddle1.re), m0205a);
        let m0205b = _mm256_mul_ps(x1m6, _mm256_set1_ps(self.twiddle2.im));
        let m0205b = _mm256_fnmadd_ps(x2m5, _mm256_set1_ps(self.twiddle3.im), m0205b);
        let m0205b = _mm256_fnmadd_ps(x3m4, _mm256_set1_ps(self.twiddle1.im), m0205b);
        let (y02, y05) = AvxButterfly::butterfly2_f32(m0205a, m0205b);

        let m0304a = _mm256_fmadd_ps(x1p6, _mm256_set1_ps(self.twiddle3.re), v[0].v);
        let m0304a = _mm256_fmadd_ps(x2p5, _mm256_set1_ps(self.twiddle1.re), m0304a);
        let m0304a = _mm256_fmadd_ps(x3p4, _mm256_set1_ps(self.twiddle2.re), m0304a);
        let m0304b = _mm256_mul_ps(x1m6, _mm256_set1_ps(self.twiddle3.im));
        let m0304b = _mm256_fnmadd_ps(x2m5, _mm256_set1_ps(self.twiddle1.im), m0304b);
        let m0304b = _mm256_fmadd_ps(x3m4, _mm256_set1_ps(self.twiddle2.im), m0304b);
        let (y03, y04) = AvxButterfly::butterfly2_f32(m0304a, m0304b);
        [
            AvxStoreF::raw(y00),
            AvxStoreF::raw(y01),
            AvxStoreF::raw(y02),
            AvxStoreF::raw(y03),
            AvxStoreF::raw(y04),
            AvxStoreF::raw(y05),
            AvxStoreF::raw(y06),
        ]
    }
}
