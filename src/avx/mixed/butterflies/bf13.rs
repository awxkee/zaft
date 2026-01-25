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

pub(crate) struct ColumnButterfly13d {
    rotate: AvxRotate<f64>,
    twiddle1: Complex<f64>,
    twiddle2: Complex<f64>,
    twiddle3: Complex<f64>,
    twiddle4: Complex<f64>,
    twiddle5: Complex<f64>,
    twiddle6: Complex<f64>,
}

impl ColumnButterfly13d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly13d {
        Self {
            rotate: AvxRotate::new(FftDirection::Inverse),
            twiddle1: compute_twiddle(1, 13, direction),
            twiddle2: compute_twiddle(2, 13, direction),
            twiddle3: compute_twiddle(3, 13, direction),
            twiddle4: compute_twiddle(4, 13, direction),
            twiddle5: compute_twiddle(5, 13, direction),
            twiddle6: compute_twiddle(6, 13, direction),
        }
    }
}

impl ColumnButterfly13d {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn exec(&self, v: [AvxStoreD; 13]) -> [AvxStoreD; 13] {
        let u0 = v[0].v;
        let y00 = v[0].v;
        let (x1p12, x1m12) = AvxButterfly::butterfly2_f64(v[1].v, v[12].v);
        let x1m12 = self.rotate.rotate_m256d(x1m12);
        let y00 = _mm256_add_pd(y00, x1p12);
        let (x2p11, x2m11) = AvxButterfly::butterfly2_f64(v[2].v, v[11].v);
        let x2m11 = self.rotate.rotate_m256d(x2m11);
        let y00 = _mm256_add_pd(y00, x2p11);
        let (x3p10, x3m10) = AvxButterfly::butterfly2_f64(v[3].v, v[10].v);
        let x3m10 = self.rotate.rotate_m256d(x3m10);
        let y00 = _mm256_add_pd(y00, x3p10);
        let (x4p9, x4m9) = AvxButterfly::butterfly2_f64(v[4].v, v[9].v);
        let x4m9 = self.rotate.rotate_m256d(x4m9);
        let y00 = _mm256_add_pd(y00, x4p9);
        let (x5p8, x5m8) = AvxButterfly::butterfly2_f64(v[5].v, v[8].v);
        let x5m8 = self.rotate.rotate_m256d(x5m8);
        let y00 = _mm256_add_pd(y00, x5p8);
        let (x6p7, x6m7) = AvxButterfly::butterfly2_f64(v[6].v, v[7].v);
        let x6m7 = self.rotate.rotate_m256d(x6m7);
        let y00 = _mm256_add_pd(y00, x6p7);

        let m0112a = _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle1.re), u0);
        let m0112a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle2.re), x2p11, m0112a);
        let m0112a = _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle3.re), x3p10, m0112a);
        let m0112a = _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle4.re), m0112a);
        let m0112a = _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle5.re), m0112a);
        let m0112a = _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle6.re), m0112a);
        let m0112b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle1.im));
        let m0112b = _mm256_fmadd_pd(x2m11, _mm256_set1_pd(self.twiddle2.im), m0112b);
        let m0112b = _mm256_fmadd_pd(x3m10, _mm256_set1_pd(self.twiddle3.im), m0112b);
        let m0112b = _mm256_fmadd_pd(x4m9, _mm256_set1_pd(self.twiddle4.im), m0112b);
        let m0112b = _mm256_fmadd_pd(x5m8, _mm256_set1_pd(self.twiddle5.im), m0112b);
        let m0112b = _mm256_fmadd_pd(x6m7, _mm256_set1_pd(self.twiddle6.im), m0112b);
        let (y01, y12) = AvxButterfly::butterfly2_f64(m0112a, m0112b);

        let m0211a = _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle2.re), u0);
        let m0211a = _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle4.re), m0211a);
        let m0211a = _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle6.re), m0211a);
        let m0211a = _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle5.re), m0211a);
        let m0211a = _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle3.re), m0211a);
        let m0211a = _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle1.re), m0211a);
        let m0211b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle2.im));
        let m0211b = _mm256_fmadd_pd(x2m11, _mm256_set1_pd(self.twiddle4.im), m0211b);
        let m0211b = _mm256_fmadd_pd(x3m10, _mm256_set1_pd(self.twiddle6.im), m0211b);
        let m0211b = _mm256_fnmadd_pd(x4m9, _mm256_set1_pd(self.twiddle5.im), m0211b);
        let m0211b = _mm256_fnmadd_pd(x5m8, _mm256_set1_pd(self.twiddle3.im), m0211b);
        let m0211b = _mm256_fnmadd_pd(x6m7, _mm256_set1_pd(self.twiddle1.im), m0211b);
        let (y02, y11) = AvxButterfly::butterfly2_f64(m0211a, m0211b);

        let m0310a = _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle3.re), u0);
        let m0310a = _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle6.re), m0310a);
        let m0310a = _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle4.re), m0310a);
        let m0310a = _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle1.re), m0310a);
        let m0310a = _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle2.re), m0310a);
        let m0310a = _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle5.re), m0310a);
        let m0310b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle3.im));
        let m0310b = _mm256_fmadd_pd(x2m11, _mm256_set1_pd(self.twiddle6.im), m0310b);
        let m0310b = _mm256_fnmadd_pd(x3m10, _mm256_set1_pd(self.twiddle4.im), m0310b);
        let m0310b = _mm256_fnmadd_pd(x4m9, _mm256_set1_pd(self.twiddle1.im), m0310b);
        let m0310b = _mm256_fmadd_pd(x5m8, _mm256_set1_pd(self.twiddle2.im), m0310b);
        let m0310b = _mm256_fmadd_pd(x6m7, _mm256_set1_pd(self.twiddle5.im), m0310b);
        let (y03, y10) = AvxButterfly::butterfly2_f64(m0310a, m0310b);

        let m0409a = _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle4.re), u0);
        let m0409a = _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle5.re), m0409a);
        let m0409a = _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle1.re), m0409a);
        let m0409a = _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle3.re), m0409a);
        let m0409a = _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle6.re), m0409a);
        let m0409a = _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle2.re), m0409a);
        let m0409b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle4.im));
        let m0409b = _mm256_fnmadd_pd(x2m11, _mm256_set1_pd(self.twiddle5.im), m0409b);
        let m0409b = _mm256_fnmadd_pd(x3m10, _mm256_set1_pd(self.twiddle1.im), m0409b);
        let m0409b = _mm256_fmadd_pd(x4m9, _mm256_set1_pd(self.twiddle3.im), m0409b);
        let m0409b = _mm256_fnmadd_pd(x5m8, _mm256_set1_pd(self.twiddle6.im), m0409b);
        let m0409b = _mm256_fnmadd_pd(x6m7, _mm256_set1_pd(self.twiddle2.im), m0409b);
        let (y04, y09) = AvxButterfly::butterfly2_f64(m0409a, m0409b);

        let m0508a = _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle5.re), u0);
        let m0508a = _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle3.re), m0508a);
        let m0508a = _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle2.re), m0508a);
        let m0508a = _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle6.re), m0508a);
        let m0508a = _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle1.re), m0508a);
        let m0508a = _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle4.re), m0508a);
        let m0508b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle5.im));
        let m0508b = _mm256_fnmadd_pd(x2m11, _mm256_set1_pd(self.twiddle3.im), m0508b);
        let m0508b = _mm256_fmadd_pd(x3m10, _mm256_set1_pd(self.twiddle2.im), m0508b);
        let m0508b = _mm256_fnmadd_pd(x4m9, _mm256_set1_pd(self.twiddle6.im), m0508b);
        let m0508b = _mm256_fnmadd_pd(x5m8, _mm256_set1_pd(self.twiddle1.im), m0508b);
        let m0508b = _mm256_fmadd_pd(x6m7, _mm256_set1_pd(self.twiddle4.im), m0508b);
        let (y05, y08) = AvxButterfly::butterfly2_f64(m0508a, m0508b);

        let m0607a = _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle6.re), u0);
        let m0607a = _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle1.re), m0607a);
        let m0607a = _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle5.re), m0607a);
        let m0607a = _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle2.re), m0607a);
        let m0607a = _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle4.re), m0607a);
        let m0607a = _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle3.re), m0607a);
        let m0607b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle6.im));
        let m0607b = _mm256_fnmadd_pd(x2m11, _mm256_set1_pd(self.twiddle1.im), m0607b);
        let m0607b = _mm256_fmadd_pd(x3m10, _mm256_set1_pd(self.twiddle5.im), m0607b);
        let m0607b = _mm256_fnmadd_pd(x4m9, _mm256_set1_pd(self.twiddle2.im), m0607b);
        let m0607b = _mm256_fmadd_pd(x5m8, _mm256_set1_pd(self.twiddle4.im), m0607b);
        let m0607b = _mm256_fnmadd_pd(x6m7, _mm256_set1_pd(self.twiddle3.im), m0607b);
        let (y06, y07) = AvxButterfly::butterfly2_f64(m0607a, m0607b);

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
            AvxStoreD::raw(y11),
            AvxStoreD::raw(y12),
        ]
    }
}

pub(crate) struct ColumnButterfly13f {
    rotate: AvxRotate<f32>,
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
    twiddle3: Complex<f32>,
    twiddle4: Complex<f32>,
    twiddle5: Complex<f32>,
    twiddle6: Complex<f32>,
}

impl ColumnButterfly13f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly13f {
        Self {
            rotate: AvxRotate::new(FftDirection::Inverse),
            twiddle1: compute_twiddle(1, 13, direction),
            twiddle2: compute_twiddle(2, 13, direction),
            twiddle3: compute_twiddle(3, 13, direction),
            twiddle4: compute_twiddle(4, 13, direction),
            twiddle5: compute_twiddle(5, 13, direction),
            twiddle6: compute_twiddle(6, 13, direction),
        }
    }
}

impl ColumnButterfly13f {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 13]) -> [AvxStoreF; 13] {
        let u0 = v[0].v;
        let y00 = v[0].v;
        let (x1p12, x1m12) = AvxButterfly::butterfly2_f32(v[1].v, v[12].v);
        let x1m12 = self.rotate.rotate_m256(x1m12);
        let y00 = _mm256_add_ps(y00, x1p12);
        let (x2p11, x2m11) = AvxButterfly::butterfly2_f32(v[2].v, v[11].v);
        let x2m11 = self.rotate.rotate_m256(x2m11);
        let y00 = _mm256_add_ps(y00, x2p11);
        let (x3p10, x3m10) = AvxButterfly::butterfly2_f32(v[3].v, v[10].v);
        let x3m10 = self.rotate.rotate_m256(x3m10);
        let y00 = _mm256_add_ps(y00, x3p10);
        let (x4p9, x4m9) = AvxButterfly::butterfly2_f32(v[4].v, v[9].v);
        let x4m9 = self.rotate.rotate_m256(x4m9);
        let y00 = _mm256_add_ps(y00, x4p9);
        let (x5p8, x5m8) = AvxButterfly::butterfly2_f32(v[5].v, v[8].v);
        let x5m8 = self.rotate.rotate_m256(x5m8);
        let y00 = _mm256_add_ps(y00, x5p8);
        let (x6p7, x6m7) = AvxButterfly::butterfly2_f32(v[6].v, v[7].v);
        let x6m7 = self.rotate.rotate_m256(x6m7);
        let y00 = _mm256_add_ps(y00, x6p7);

        let m0112a = _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle1.re), u0);
        let m0112a = _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle2.re), x2p11, m0112a);
        let m0112a = _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle3.re), x3p10, m0112a);
        let m0112a = _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle4.re), m0112a);
        let m0112a = _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle5.re), m0112a);
        let m0112a = _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle6.re), m0112a);
        let m0112b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle1.im));
        let m0112b = _mm256_fmadd_ps(x2m11, _mm256_set1_ps(self.twiddle2.im), m0112b);
        let m0112b = _mm256_fmadd_ps(x3m10, _mm256_set1_ps(self.twiddle3.im), m0112b);
        let m0112b = _mm256_fmadd_ps(x4m9, _mm256_set1_ps(self.twiddle4.im), m0112b);
        let m0112b = _mm256_fmadd_ps(x5m8, _mm256_set1_ps(self.twiddle5.im), m0112b);
        let m0112b = _mm256_fmadd_ps(x6m7, _mm256_set1_ps(self.twiddle6.im), m0112b);
        let (y01, y12) = AvxButterfly::butterfly2_f32(m0112a, m0112b);

        let m0211a = _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle2.re), u0);
        let m0211a = _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle4.re), m0211a);
        let m0211a = _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle6.re), m0211a);
        let m0211a = _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle5.re), m0211a);
        let m0211a = _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle3.re), m0211a);
        let m0211a = _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle1.re), m0211a);
        let m0211b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle2.im));
        let m0211b = _mm256_fmadd_ps(x2m11, _mm256_set1_ps(self.twiddle4.im), m0211b);
        let m0211b = _mm256_fmadd_ps(x3m10, _mm256_set1_ps(self.twiddle6.im), m0211b);
        let m0211b = _mm256_fnmadd_ps(x4m9, _mm256_set1_ps(self.twiddle5.im), m0211b);
        let m0211b = _mm256_fnmadd_ps(x5m8, _mm256_set1_ps(self.twiddle3.im), m0211b);
        let m0211b = _mm256_fnmadd_ps(x6m7, _mm256_set1_ps(self.twiddle1.im), m0211b);
        let (y02, y11) = AvxButterfly::butterfly2_f32(m0211a, m0211b);

        let m0310a = _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle3.re), u0);
        let m0310a = _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle6.re), m0310a);
        let m0310a = _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle4.re), m0310a);
        let m0310a = _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle1.re), m0310a);
        let m0310a = _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle2.re), m0310a);
        let m0310a = _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle5.re), m0310a);
        let m0310b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle3.im));
        let m0310b = _mm256_fmadd_ps(x2m11, _mm256_set1_ps(self.twiddle6.im), m0310b);
        let m0310b = _mm256_fnmadd_ps(x3m10, _mm256_set1_ps(self.twiddle4.im), m0310b);
        let m0310b = _mm256_fnmadd_ps(x4m9, _mm256_set1_ps(self.twiddle1.im), m0310b);
        let m0310b = _mm256_fmadd_ps(x5m8, _mm256_set1_ps(self.twiddle2.im), m0310b);
        let m0310b = _mm256_fmadd_ps(x6m7, _mm256_set1_ps(self.twiddle5.im), m0310b);
        let (y03, y10) = AvxButterfly::butterfly2_f32(m0310a, m0310b);

        let m0409a = _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle4.re), u0);
        let m0409a = _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle5.re), m0409a);
        let m0409a = _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle1.re), m0409a);
        let m0409a = _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle3.re), m0409a);
        let m0409a = _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle6.re), m0409a);
        let m0409a = _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle2.re), m0409a);
        let m0409b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle4.im));
        let m0409b = _mm256_fnmadd_ps(x2m11, _mm256_set1_ps(self.twiddle5.im), m0409b);
        let m0409b = _mm256_fnmadd_ps(x3m10, _mm256_set1_ps(self.twiddle1.im), m0409b);
        let m0409b = _mm256_fmadd_ps(x4m9, _mm256_set1_ps(self.twiddle3.im), m0409b);
        let m0409b = _mm256_fnmadd_ps(x5m8, _mm256_set1_ps(self.twiddle6.im), m0409b);
        let m0409b = _mm256_fnmadd_ps(x6m7, _mm256_set1_ps(self.twiddle2.im), m0409b);
        let (y04, y09) = AvxButterfly::butterfly2_f32(m0409a, m0409b);

        let m0508a = _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle5.re), u0);
        let m0508a = _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle3.re), m0508a);
        let m0508a = _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle2.re), m0508a);
        let m0508a = _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle6.re), m0508a);
        let m0508a = _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle1.re), m0508a);
        let m0508a = _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle4.re), m0508a);
        let m0508b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle5.im));
        let m0508b = _mm256_fnmadd_ps(x2m11, _mm256_set1_ps(self.twiddle3.im), m0508b);
        let m0508b = _mm256_fmadd_ps(x3m10, _mm256_set1_ps(self.twiddle2.im), m0508b);
        let m0508b = _mm256_fnmadd_ps(x4m9, _mm256_set1_ps(self.twiddle6.im), m0508b);
        let m0508b = _mm256_fnmadd_ps(x5m8, _mm256_set1_ps(self.twiddle1.im), m0508b);
        let m0508b = _mm256_fmadd_ps(x6m7, _mm256_set1_ps(self.twiddle4.im), m0508b);
        let (y05, y08) = AvxButterfly::butterfly2_f32(m0508a, m0508b);

        let m0607a = _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle6.re), u0);
        let m0607a = _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle1.re), m0607a);
        let m0607a = _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle5.re), m0607a);
        let m0607a = _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle2.re), m0607a);
        let m0607a = _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle4.re), m0607a);
        let m0607a = _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle3.re), m0607a);
        let m0607b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle6.im));
        let m0607b = _mm256_fnmadd_ps(x2m11, _mm256_set1_ps(self.twiddle1.im), m0607b);
        let m0607b = _mm256_fmadd_ps(x3m10, _mm256_set1_ps(self.twiddle5.im), m0607b);
        let m0607b = _mm256_fnmadd_ps(x4m9, _mm256_set1_ps(self.twiddle2.im), m0607b);
        let m0607b = _mm256_fmadd_ps(x5m8, _mm256_set1_ps(self.twiddle4.im), m0607b);
        let m0607b = _mm256_fnmadd_ps(x6m7, _mm256_set1_ps(self.twiddle3.im), m0607b);
        let (y06, y07) = AvxButterfly::butterfly2_f32(m0607a, m0607b);

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
            AvxStoreF::raw(y11),
            AvxStoreF::raw(y12),
        ]
    }
}
