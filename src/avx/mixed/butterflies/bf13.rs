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
use num_traits::MulAdd;
use std::arch::x86_64::*;

pub(crate) struct ColumnButterfly13d {
    rotate: AvxRotate<f64>,
    twiddle1_re: __m256d,
    twiddle2_re: __m256d,
    twiddle3_re: __m256d,
    twiddle4_re: __m256d,
    twiddle5_re: __m256d,
    twiddle6_re: __m256d,
    twiddle1_im: __m256d,
    twiddle2_im: __m256d,
    twiddle3_im: __m256d,
    twiddle4_im: __m256d,
    twiddle5_im: __m256d,
    twiddle6_im: __m256d,
}

impl ColumnButterfly13d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly13d {
        let twiddle1 = compute_twiddle(1, 13, direction);
        let twiddle2 = compute_twiddle(2, 13, direction);
        let twiddle3 = compute_twiddle(3, 13, direction);
        let twiddle4 = compute_twiddle(4, 13, direction);
        let twiddle5 = compute_twiddle(5, 13, direction);
        let twiddle6 = compute_twiddle(6, 13, direction);
        Self {
            rotate: AvxRotate::new(FftDirection::Inverse),
            twiddle1_re: _mm256_set1_pd(twiddle1.re),
            twiddle1_im: _mm256_set1_pd(twiddle1.im),
            twiddle2_re: _mm256_set1_pd(twiddle2.re),
            twiddle2_im: _mm256_set1_pd(twiddle2.im),
            twiddle3_re: _mm256_set1_pd(twiddle3.re),
            twiddle3_im: _mm256_set1_pd(twiddle3.im),
            twiddle4_re: _mm256_set1_pd(twiddle4.re),
            twiddle4_im: _mm256_set1_pd(twiddle4.im),
            twiddle5_re: _mm256_set1_pd(twiddle5.re),
            twiddle5_im: _mm256_set1_pd(twiddle5.im),
            twiddle6_re: _mm256_set1_pd(twiddle6.re),
            twiddle6_im: _mm256_set1_pd(twiddle6.im),
        }
    }
}

impl ColumnButterfly13d {
    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreD; 13]) -> [AvxStoreD; 13] {
        unsafe {
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

            let m0112a = _mm256_fmadd_pd(x1p12, self.twiddle1_re, u0);
            let m0112a = _mm256_fmadd_pd(self.twiddle2_re, x2p11, m0112a);
            let m0112a = _mm256_fmadd_pd(self.twiddle3_re, x3p10, m0112a);
            let m0112a = _mm256_fmadd_pd(x4p9, self.twiddle4_re, m0112a);
            let m0112a = _mm256_fmadd_pd(x5p8, self.twiddle5_re, m0112a);
            let m0112a = _mm256_fmadd_pd(x6p7, self.twiddle6_re, m0112a);
            let m0112b = _mm256_mul_pd(x1m12, self.twiddle1_im);
            let m0112b = _mm256_fmadd_pd(x2m11, self.twiddle2_im, m0112b);
            let m0112b = _mm256_fmadd_pd(x3m10, self.twiddle3_im, m0112b);
            let m0112b = _mm256_fmadd_pd(x4m9, self.twiddle4_im, m0112b);
            let m0112b = _mm256_fmadd_pd(x5m8, self.twiddle5_im, m0112b);
            let m0112b = _mm256_fmadd_pd(x6m7, self.twiddle6_im, m0112b);
            let (y01, y12) = AvxButterfly::butterfly2_f64(m0112a, m0112b);

            let m0211a = _mm256_fmadd_pd(x1p12, self.twiddle2_re, u0);
            let m0211a = _mm256_fmadd_pd(x2p11, self.twiddle4_re, m0211a);
            let m0211a = _mm256_fmadd_pd(x3p10, self.twiddle6_re, m0211a);
            let m0211a = _mm256_fmadd_pd(x4p9, self.twiddle5_re, m0211a);
            let m0211a = _mm256_fmadd_pd(x5p8, self.twiddle3_re, m0211a);
            let m0211a = _mm256_fmadd_pd(x6p7, self.twiddle1_re, m0211a);
            let m0211b = _mm256_mul_pd(x1m12, self.twiddle2_im);
            let m0211b = _mm256_fmadd_pd(x2m11, self.twiddle4_im, m0211b);
            let m0211b = _mm256_fmadd_pd(x3m10, self.twiddle6_im, m0211b);
            let m0211b = _mm256_fnmadd_pd(x4m9, self.twiddle5_im, m0211b);
            let m0211b = _mm256_fnmadd_pd(x5m8, self.twiddle3_im, m0211b);
            let m0211b = _mm256_fnmadd_pd(x6m7, self.twiddle1_im, m0211b);
            let (y02, y11) = AvxButterfly::butterfly2_f64(m0211a, m0211b);

            let m0310a = _mm256_fmadd_pd(x1p12, self.twiddle3_re, u0);
            let m0310a = _mm256_fmadd_pd(x2p11, self.twiddle6_re, m0310a);
            let m0310a = _mm256_fmadd_pd(x3p10, self.twiddle4_re, m0310a);
            let m0310a = _mm256_fmadd_pd(x4p9, self.twiddle1_re, m0310a);
            let m0310a = _mm256_fmadd_pd(x5p8, self.twiddle2_re, m0310a);
            let m0310a = _mm256_fmadd_pd(x6p7, self.twiddle5_re, m0310a);
            let m0310b = _mm256_mul_pd(x1m12, self.twiddle3_im);
            let m0310b = _mm256_fmadd_pd(x2m11, self.twiddle6_im, m0310b);
            let m0310b = _mm256_fnmadd_pd(x3m10, self.twiddle4_im, m0310b);
            let m0310b = _mm256_fnmadd_pd(x4m9, self.twiddle1_im, m0310b);
            let m0310b = _mm256_fmadd_pd(x5m8, self.twiddle2_im, m0310b);
            let m0310b = _mm256_fmadd_pd(x6m7, self.twiddle5_im, m0310b);
            let (y03, y10) = AvxButterfly::butterfly2_f64(m0310a, m0310b);

            let m0409a = _mm256_fmadd_pd(x1p12, self.twiddle4_re, u0);
            let m0409a = _mm256_fmadd_pd(x2p11, self.twiddle5_re, m0409a);
            let m0409a = _mm256_fmadd_pd(x3p10, self.twiddle1_re, m0409a);
            let m0409a = _mm256_fmadd_pd(x4p9, self.twiddle3_re, m0409a);
            let m0409a = _mm256_fmadd_pd(x5p8, self.twiddle6_re, m0409a);
            let m0409a = _mm256_fmadd_pd(x6p7, self.twiddle2_re, m0409a);
            let m0409b = _mm256_mul_pd(x1m12, self.twiddle4_im);
            let m0409b = _mm256_fnmadd_pd(x2m11, self.twiddle5_im, m0409b);
            let m0409b = _mm256_fnmadd_pd(x3m10, self.twiddle1_im, m0409b);
            let m0409b = _mm256_fmadd_pd(x4m9, self.twiddle3_im, m0409b);
            let m0409b = _mm256_fnmadd_pd(x5m8, self.twiddle6_im, m0409b);
            let m0409b = _mm256_fnmadd_pd(x6m7, self.twiddle2_im, m0409b);
            let (y04, y09) = AvxButterfly::butterfly2_f64(m0409a, m0409b);

            let m0508a = _mm256_fmadd_pd(x1p12, self.twiddle5_re, u0);
            let m0508a = _mm256_fmadd_pd(x2p11, self.twiddle3_re, m0508a);
            let m0508a = _mm256_fmadd_pd(x3p10, self.twiddle2_re, m0508a);
            let m0508a = _mm256_fmadd_pd(x4p9, self.twiddle6_re, m0508a);
            let m0508a = _mm256_fmadd_pd(x5p8, self.twiddle1_re, m0508a);
            let m0508a = _mm256_fmadd_pd(x6p7, self.twiddle4_re, m0508a);
            let m0508b = _mm256_mul_pd(x1m12, self.twiddle5_im);
            let m0508b = _mm256_fnmadd_pd(x2m11, self.twiddle3_im, m0508b);
            let m0508b = _mm256_fmadd_pd(x3m10, self.twiddle2_im, m0508b);
            let m0508b = _mm256_fnmadd_pd(x4m9, self.twiddle6_im, m0508b);
            let m0508b = _mm256_fnmadd_pd(x5m8, self.twiddle1_im, m0508b);
            let m0508b = _mm256_fmadd_pd(x6m7, self.twiddle4_im, m0508b);
            let (y05, y08) = AvxButterfly::butterfly2_f64(m0508a, m0508b);

            let m0607a = _mm256_fmadd_pd(x1p12, self.twiddle6_re, u0);
            let m0607a = _mm256_fmadd_pd(x2p11, self.twiddle1_re, m0607a);
            let m0607a = _mm256_fmadd_pd(x3p10, self.twiddle5_re, m0607a);
            let m0607a = _mm256_fmadd_pd(x4p9, self.twiddle2_re, m0607a);
            let m0607a = _mm256_fmadd_pd(x5p8, self.twiddle4_re, m0607a);
            let m0607a = _mm256_fmadd_pd(x6p7, self.twiddle3_re, m0607a);
            let m0607b = _mm256_mul_pd(x1m12, self.twiddle6_im);
            let m0607b = _mm256_fnmadd_pd(x2m11, self.twiddle1_im, m0607b);
            let m0607b = _mm256_fmadd_pd(x3m10, self.twiddle5_im, m0607b);
            let m0607b = _mm256_fnmadd_pd(x4m9, self.twiddle2_im, m0607b);
            let m0607b = _mm256_fmadd_pd(x5m8, self.twiddle4_im, m0607b);
            let m0607b = _mm256_fnmadd_pd(x6m7, self.twiddle3_im, m0607b);
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
}

pub(crate) struct ColumnButterfly13f {
    rotate: AvxRotate<f32>,
    twiddle1_re: __m256,
    twiddle2_re: __m256,
    twiddle3_re: __m256,
    twiddle4_re: __m256,
    twiddle5_re: __m256,
    twiddle6_re: __m256,
    twiddle1_im: __m256,
    twiddle2_im: __m256,
    twiddle3_im: __m256,
    twiddle4_im: __m256,
    twiddle5_im: __m256,
    twiddle6_im: __m256,
}

impl ColumnButterfly13f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly13f {
        let twiddle1 = compute_twiddle(1, 13, direction);
        let twiddle2 = compute_twiddle(2, 13, direction);
        let twiddle3 = compute_twiddle(3, 13, direction);
        let twiddle4 = compute_twiddle(4, 13, direction);
        let twiddle5 = compute_twiddle(5, 13, direction);
        let twiddle6 = compute_twiddle(6, 13, direction);
        Self {
            rotate: AvxRotate::new(FftDirection::Inverse),
            twiddle1_re: _mm256_set1_ps(twiddle1.re),
            twiddle1_im: _mm256_set1_ps(twiddle1.im),
            twiddle2_re: _mm256_set1_ps(twiddle2.re),
            twiddle2_im: _mm256_set1_ps(twiddle2.im),
            twiddle3_re: _mm256_set1_ps(twiddle3.re),
            twiddle3_im: _mm256_set1_ps(twiddle3.im),
            twiddle4_re: _mm256_set1_ps(twiddle4.re),
            twiddle4_im: _mm256_set1_ps(twiddle4.im),
            twiddle5_re: _mm256_set1_ps(twiddle5.re),
            twiddle5_im: _mm256_set1_ps(twiddle5.im),
            twiddle6_re: _mm256_set1_ps(twiddle6.re),
            twiddle6_im: _mm256_set1_ps(twiddle6.im),
        }
    }
}

impl ColumnButterfly13f {
    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 13]) -> [AvxStoreF; 13] {
        unsafe {
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

            let m0112a = _mm256_fmadd_ps(x1p12, self.twiddle1_re, u0);
            let m0112a = _mm256_fmadd_ps(self.twiddle2_re, x2p11, m0112a);
            let m0112a = _mm256_fmadd_ps(self.twiddle3_re, x3p10, m0112a);
            let m0112a = _mm256_fmadd_ps(x4p9, self.twiddle4_re, m0112a);
            let m0112a = _mm256_fmadd_ps(x5p8, self.twiddle5_re, m0112a);
            let m0112a = _mm256_fmadd_ps(x6p7, self.twiddle6_re, m0112a);
            let m0112b = _mm256_mul_ps(x1m12, self.twiddle1_im);
            let m0112b = _mm256_fmadd_ps(x2m11, self.twiddle2_im, m0112b);
            let m0112b = _mm256_fmadd_ps(x3m10, self.twiddle3_im, m0112b);
            let m0112b = _mm256_fmadd_ps(x4m9, self.twiddle4_im, m0112b);
            let m0112b = _mm256_fmadd_ps(x5m8, self.twiddle5_im, m0112b);
            let m0112b = _mm256_fmadd_ps(x6m7, self.twiddle6_im, m0112b);
            let (y01, y12) = AvxButterfly::butterfly2_f32(m0112a, m0112b);

            let m0211a = _mm256_fmadd_ps(x1p12, self.twiddle2_re, u0);
            let m0211a = _mm256_fmadd_ps(x2p11, self.twiddle4_re, m0211a);
            let m0211a = _mm256_fmadd_ps(x3p10, self.twiddle6_re, m0211a);
            let m0211a = _mm256_fmadd_ps(x4p9, self.twiddle5_re, m0211a);
            let m0211a = _mm256_fmadd_ps(x5p8, self.twiddle3_re, m0211a);
            let m0211a = _mm256_fmadd_ps(x6p7, self.twiddle1_re, m0211a);
            let m0211b = _mm256_mul_ps(x1m12, self.twiddle2_im);
            let m0211b = _mm256_fmadd_ps(x2m11, self.twiddle4_im, m0211b);
            let m0211b = _mm256_fmadd_ps(x3m10, self.twiddle6_im, m0211b);
            let m0211b = _mm256_fnmadd_ps(x4m9, self.twiddle5_im, m0211b);
            let m0211b = _mm256_fnmadd_ps(x5m8, self.twiddle3_im, m0211b);
            let m0211b = _mm256_fnmadd_ps(x6m7, self.twiddle1_im, m0211b);
            let (y02, y11) = AvxButterfly::butterfly2_f32(m0211a, m0211b);

            let m0310a = _mm256_fmadd_ps(x1p12, self.twiddle3_re, u0);
            let m0310a = _mm256_fmadd_ps(x2p11, self.twiddle6_re, m0310a);
            let m0310a = _mm256_fmadd_ps(x3p10, self.twiddle4_re, m0310a);
            let m0310a = _mm256_fmadd_ps(x4p9, self.twiddle1_re, m0310a);
            let m0310a = _mm256_fmadd_ps(x5p8, self.twiddle2_re, m0310a);
            let m0310a = _mm256_fmadd_ps(x6p7, self.twiddle5_re, m0310a);
            let m0310b = _mm256_mul_ps(x1m12, self.twiddle3_im);
            let m0310b = _mm256_fmadd_ps(x2m11, self.twiddle6_im, m0310b);
            let m0310b = _mm256_fnmadd_ps(x3m10, self.twiddle4_im, m0310b);
            let m0310b = _mm256_fnmadd_ps(x4m9, self.twiddle1_im, m0310b);
            let m0310b = _mm256_fmadd_ps(x5m8, self.twiddle2_im, m0310b);
            let m0310b = _mm256_fmadd_ps(x6m7, self.twiddle5_im, m0310b);
            let (y03, y10) = AvxButterfly::butterfly2_f32(m0310a, m0310b);

            let m0409a = _mm256_fmadd_ps(x1p12, self.twiddle4_re, u0);
            let m0409a = _mm256_fmadd_ps(x2p11, self.twiddle5_re, m0409a);
            let m0409a = _mm256_fmadd_ps(x3p10, self.twiddle1_re, m0409a);
            let m0409a = _mm256_fmadd_ps(x4p9, self.twiddle3_re, m0409a);
            let m0409a = _mm256_fmadd_ps(x5p8, self.twiddle6_re, m0409a);
            let m0409a = _mm256_fmadd_ps(x6p7, self.twiddle2_re, m0409a);
            let m0409b = _mm256_mul_ps(x1m12, self.twiddle4_im);
            let m0409b = _mm256_fnmadd_ps(x2m11, self.twiddle5_im, m0409b);
            let m0409b = _mm256_fnmadd_ps(x3m10, self.twiddle1_im, m0409b);
            let m0409b = _mm256_fmadd_ps(x4m9, self.twiddle3_im, m0409b);
            let m0409b = _mm256_fnmadd_ps(x5m8, self.twiddle6_im, m0409b);
            let m0409b = _mm256_fnmadd_ps(x6m7, self.twiddle2_im, m0409b);
            let (y04, y09) = AvxButterfly::butterfly2_f32(m0409a, m0409b);

            let m0508a = _mm256_fmadd_ps(x1p12, self.twiddle5_re, u0);
            let m0508a = _mm256_fmadd_ps(x2p11, self.twiddle3_re, m0508a);
            let m0508a = _mm256_fmadd_ps(x3p10, self.twiddle2_re, m0508a);
            let m0508a = _mm256_fmadd_ps(x4p9, self.twiddle6_re, m0508a);
            let m0508a = _mm256_fmadd_ps(x5p8, self.twiddle1_re, m0508a);
            let m0508a = _mm256_fmadd_ps(x6p7, self.twiddle4_re, m0508a);
            let m0508b = _mm256_mul_ps(x1m12, self.twiddle5_im);
            let m0508b = _mm256_fnmadd_ps(x2m11, self.twiddle3_im, m0508b);
            let m0508b = _mm256_fmadd_ps(x3m10, self.twiddle2_im, m0508b);
            let m0508b = _mm256_fnmadd_ps(x4m9, self.twiddle6_im, m0508b);
            let m0508b = _mm256_fnmadd_ps(x5m8, self.twiddle1_im, m0508b);
            let m0508b = _mm256_fmadd_ps(x6m7, self.twiddle4_im, m0508b);
            let (y05, y08) = AvxButterfly::butterfly2_f32(m0508a, m0508b);

            let m0607a = _mm256_fmadd_ps(x1p12, self.twiddle6_re, u0);
            let m0607a = _mm256_fmadd_ps(x2p11, self.twiddle1_re, m0607a);
            let m0607a = _mm256_fmadd_ps(x3p10, self.twiddle5_re, m0607a);
            let m0607a = _mm256_fmadd_ps(x4p9, self.twiddle2_re, m0607a);
            let m0607a = _mm256_fmadd_ps(x5p8, self.twiddle4_re, m0607a);
            let m0607a = _mm256_fmadd_ps(x6p7, self.twiddle3_re, m0607a);
            let m0607b = _mm256_mul_ps(x1m12, self.twiddle6_im);
            let m0607b = _mm256_fnmadd_ps(x2m11, self.twiddle1_im, m0607b);
            let m0607b = _mm256_fmadd_ps(x3m10, self.twiddle5_im, m0607b);
            let m0607b = _mm256_fnmadd_ps(x4m9, self.twiddle2_im, m0607b);
            let m0607b = _mm256_fmadd_ps(x5m8, self.twiddle4_im, m0607b);
            let m0607b = _mm256_fnmadd_ps(x6m7, self.twiddle3_im, m0607b);
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
}

pub(crate) struct ColumnRdftButterfly13f {
    twiddle1_re: AvxStoreF,
    twiddle2_re: AvxStoreF,
    twiddle3_re: AvxStoreF,
    twiddle4_re: AvxStoreF,
    twiddle5_re: AvxStoreF,
    twiddle6_re: AvxStoreF,
    twiddle1_im: AvxStoreF,
    twiddle2_im: AvxStoreF,
    twiddle3_im: AvxStoreF,
    twiddle4_im: AvxStoreF,
    twiddle5_im: AvxStoreF,
    twiddle6_im: AvxStoreF,
}

impl ColumnRdftButterfly13f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new() -> Self {
        let twiddle1 = compute_twiddle(1, 13, FftDirection::Forward);
        let twiddle2 = compute_twiddle(2, 13, FftDirection::Forward);
        let twiddle3 = compute_twiddle(3, 13, FftDirection::Forward);
        let twiddle4 = compute_twiddle(4, 13, FftDirection::Forward);
        let twiddle5 = compute_twiddle(5, 13, FftDirection::Forward);
        let twiddle6 = compute_twiddle(6, 13, FftDirection::Forward);
        Self {
            twiddle1_re: AvxStoreF::dup(twiddle1.re),
            twiddle1_im: AvxStoreF::dup(twiddle1.im),
            twiddle2_re: AvxStoreF::dup(twiddle2.re),
            twiddle2_im: AvxStoreF::dup(twiddle2.im),
            twiddle3_re: AvxStoreF::dup(twiddle3.re),
            twiddle3_im: AvxStoreF::dup(twiddle3.im),
            twiddle4_re: AvxStoreF::dup(twiddle4.re),
            twiddle4_im: AvxStoreF::dup(twiddle4.im),
            twiddle5_re: AvxStoreF::dup(twiddle5.re),
            twiddle5_im: AvxStoreF::dup(twiddle5.im),
            twiddle6_re: AvxStoreF::dup(twiddle6.re),
            twiddle6_im: AvxStoreF::dup(twiddle6.im),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec(&self, store: [AvxStoreF; 13]) -> [[AvxStoreF; 7]; 2] {
        let x112p = store[1] + store[12];
        let x112n = store[1] - store[12];
        let x211p = store[2] + store[11];
        let x211n = store[2] - store[11];
        let x310p = store[3] + store[10];
        let x310n = store[3] - store[10];
        let x49p = store[4] + store[9];
        let x49n = store[4] - store[9];
        let x58p = store[5] + store[8];
        let x58n = store[5] - store[8];
        let x67p = store[6] + store[7];
        let x67n = store[6] - store[7];

        // DC
        let y0 = store[0] + x112p + x211p + x310p + x49p + x58p + x67p;

        let b112re = x112p.mul_add(
            self.twiddle1_re,
            x211p.mul_add(
                self.twiddle2_re,
                x310p.mul_add(
                    self.twiddle3_re,
                    x49p.mul_add(
                        self.twiddle4_re,
                        x58p.mul_add(self.twiddle5_re, x67p.mul_add(self.twiddle6_re, store[0])),
                    ),
                ),
            ),
        );

        let b211re = x112p.mul_add(
            self.twiddle2_re,
            x211p.mul_add(
                self.twiddle4_re,
                x310p.mul_add(
                    self.twiddle6_re,
                    x49p.mul_add(
                        self.twiddle5_re,
                        x58p.mul_add(self.twiddle3_re, x67p.mul_add(self.twiddle1_re, store[0])),
                    ),
                ),
            ),
        );

        let b310re = x112p.mul_add(
            self.twiddle3_re,
            x211p.mul_add(
                self.twiddle6_re,
                x310p.mul_add(
                    self.twiddle4_re,
                    x49p.mul_add(
                        self.twiddle1_re,
                        x58p.mul_add(self.twiddle2_re, x67p.mul_add(self.twiddle5_re, store[0])),
                    ),
                ),
            ),
        );

        let b49re = x112p.mul_add(
            self.twiddle4_re,
            x211p.mul_add(
                self.twiddle5_re,
                x310p.mul_add(
                    self.twiddle1_re,
                    x49p.mul_add(
                        self.twiddle3_re,
                        x58p.mul_add(self.twiddle6_re, x67p.mul_add(self.twiddle2_re, store[0])),
                    ),
                ),
            ),
        );

        let b58re = x112p.mul_add(
            self.twiddle5_re,
            x211p.mul_add(
                self.twiddle3_re,
                x310p.mul_add(
                    self.twiddle2_re,
                    x49p.mul_add(
                        self.twiddle6_re,
                        x58p.mul_add(self.twiddle1_re, x67p.mul_add(self.twiddle4_re, store[0])),
                    ),
                ),
            ),
        );

        let b67re = x112p.mul_add(
            self.twiddle6_re,
            x211p.mul_add(
                self.twiddle1_re,
                x310p.mul_add(
                    self.twiddle5_re,
                    x49p.mul_add(
                        self.twiddle2_re,
                        x58p.mul_add(self.twiddle4_re, x67p.mul_add(self.twiddle3_re, store[0])),
                    ),
                ),
            ),
        );

        let b112im = x112n.mul_add(
            self.twiddle1_im,
            x211n.mul_add(
                self.twiddle2_im,
                x310n.mul_add(
                    self.twiddle3_im,
                    x49n.mul_add(
                        self.twiddle4_im,
                        x58n.mul_add(self.twiddle5_im, x67n * self.twiddle6_im),
                    ),
                ),
            ),
        );

        let b211im = x112n.mul_add(
            self.twiddle2_im,
            x211n.mul_add(
                self.twiddle4_im,
                x310n.mul_add(
                    self.twiddle6_im,
                    x49n.mul_add(
                        -self.twiddle5_im,
                        x58n.mul_add(-self.twiddle3_im, x67n * -self.twiddle1_im),
                    ),
                ),
            ),
        );

        let b310im = x112n.mul_add(
            self.twiddle3_im,
            x211n.mul_add(
                self.twiddle6_im,
                x310n.mul_add(
                    -self.twiddle4_im,
                    x49n.mul_add(
                        -self.twiddle1_im,
                        x58n.mul_add(self.twiddle2_im, x67n * self.twiddle5_im),
                    ),
                ),
            ),
        );

        let b49im = x112n.mul_add(
            self.twiddle4_im,
            x211n.mul_add(
                -self.twiddle5_im,
                x310n.mul_add(
                    -self.twiddle1_im,
                    x49n.mul_add(
                        self.twiddle3_im,
                        x58n.mul_add(-self.twiddle6_im, x67n * -self.twiddle2_im),
                    ),
                ),
            ),
        );

        let b58im = x112n.mul_add(
            self.twiddle5_im,
            x211n.mul_add(
                -self.twiddle3_im,
                x310n.mul_add(
                    self.twiddle2_im,
                    x49n.mul_add(
                        -self.twiddle6_im,
                        x58n.mul_add(-self.twiddle1_im, x67n * self.twiddle4_im),
                    ),
                ),
            ),
        );

        let b67im = x112n.mul_add(
            self.twiddle6_im,
            x211n.mul_add(
                -self.twiddle1_im,
                x310n.mul_add(
                    self.twiddle5_im,
                    x49n.mul_add(
                        -self.twiddle2_im,
                        x58n.mul_add(self.twiddle4_im, x67n * -self.twiddle3_im),
                    ),
                ),
            ),
        );

        let v0 = y0.zip(AvxStoreF::zero());
        let v1 = b112re.zip(b112im);
        let v2 = b211re.zip(b211im);
        let v3 = b310re.zip(b310im);
        let v4 = b49re.zip(b49im);
        let v5 = b58re.zip(b58im);
        let v6 = b67re.zip(b67im);

        [
            [v0[0], v1[0], v2[0], v3[0], v4[0], v5[0], v6[0]],
            [v0[1], v1[1], v2[1], v3[1], v4[1], v5[1], v6[1]],
        ]
    }
}

pub(crate) struct ColumnRdftButterfly13d {
    twiddle1_re: AvxStoreD,
    twiddle2_re: AvxStoreD,
    twiddle3_re: AvxStoreD,
    twiddle4_re: AvxStoreD,
    twiddle5_re: AvxStoreD,
    twiddle6_re: AvxStoreD,
    twiddle1_im: AvxStoreD,
    twiddle2_im: AvxStoreD,
    twiddle3_im: AvxStoreD,
    twiddle4_im: AvxStoreD,
    twiddle5_im: AvxStoreD,
    twiddle6_im: AvxStoreD,
}

impl ColumnRdftButterfly13d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new() -> Self {
        let twiddle1 = compute_twiddle(1, 13, FftDirection::Forward);
        let twiddle2 = compute_twiddle(2, 13, FftDirection::Forward);
        let twiddle3 = compute_twiddle(3, 13, FftDirection::Forward);
        let twiddle4 = compute_twiddle(4, 13, FftDirection::Forward);
        let twiddle5 = compute_twiddle(5, 13, FftDirection::Forward);
        let twiddle6 = compute_twiddle(6, 13, FftDirection::Forward);
        Self {
            twiddle1_re: AvxStoreD::dup(twiddle1.re),
            twiddle1_im: AvxStoreD::dup(twiddle1.im),
            twiddle2_re: AvxStoreD::dup(twiddle2.re),
            twiddle2_im: AvxStoreD::dup(twiddle2.im),
            twiddle3_re: AvxStoreD::dup(twiddle3.re),
            twiddle3_im: AvxStoreD::dup(twiddle3.im),
            twiddle4_re: AvxStoreD::dup(twiddle4.re),
            twiddle4_im: AvxStoreD::dup(twiddle4.im),
            twiddle5_re: AvxStoreD::dup(twiddle5.re),
            twiddle5_im: AvxStoreD::dup(twiddle5.im),
            twiddle6_re: AvxStoreD::dup(twiddle6.re),
            twiddle6_im: AvxStoreD::dup(twiddle6.im),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec(&self, store: [AvxStoreD; 13]) -> [[AvxStoreD; 7]; 2] {
        let x112p = store[1] + store[12];
        let x112n = store[1] - store[12];
        let x211p = store[2] + store[11];
        let x211n = store[2] - store[11];
        let x310p = store[3] + store[10];
        let x310n = store[3] - store[10];
        let x49p = store[4] + store[9];
        let x49n = store[4] - store[9];
        let x58p = store[5] + store[8];
        let x58n = store[5] - store[8];
        let x67p = store[6] + store[7];
        let x67n = store[6] - store[7];

        // DC
        let y0 = store[0] + x112p + x211p + x310p + x49p + x58p + x67p;

        let b112re = x112p.mul_add(
            self.twiddle1_re,
            x211p.mul_add(
                self.twiddle2_re,
                x310p.mul_add(
                    self.twiddle3_re,
                    x49p.mul_add(
                        self.twiddle4_re,
                        x58p.mul_add(self.twiddle5_re, x67p.mul_add(self.twiddle6_re, store[0])),
                    ),
                ),
            ),
        );

        let b211re = x112p.mul_add(
            self.twiddle2_re,
            x211p.mul_add(
                self.twiddle4_re,
                x310p.mul_add(
                    self.twiddle6_re,
                    x49p.mul_add(
                        self.twiddle5_re,
                        x58p.mul_add(self.twiddle3_re, x67p.mul_add(self.twiddle1_re, store[0])),
                    ),
                ),
            ),
        );

        let b310re = x112p.mul_add(
            self.twiddle3_re,
            x211p.mul_add(
                self.twiddle6_re,
                x310p.mul_add(
                    self.twiddle4_re,
                    x49p.mul_add(
                        self.twiddle1_re,
                        x58p.mul_add(self.twiddle2_re, x67p.mul_add(self.twiddle5_re, store[0])),
                    ),
                ),
            ),
        );

        let b49re = x112p.mul_add(
            self.twiddle4_re,
            x211p.mul_add(
                self.twiddle5_re,
                x310p.mul_add(
                    self.twiddle1_re,
                    x49p.mul_add(
                        self.twiddle3_re,
                        x58p.mul_add(self.twiddle6_re, x67p.mul_add(self.twiddle2_re, store[0])),
                    ),
                ),
            ),
        );

        let b58re = x112p.mul_add(
            self.twiddle5_re,
            x211p.mul_add(
                self.twiddle3_re,
                x310p.mul_add(
                    self.twiddle2_re,
                    x49p.mul_add(
                        self.twiddle6_re,
                        x58p.mul_add(self.twiddle1_re, x67p.mul_add(self.twiddle4_re, store[0])),
                    ),
                ),
            ),
        );

        let b67re = x112p.mul_add(
            self.twiddle6_re,
            x211p.mul_add(
                self.twiddle1_re,
                x310p.mul_add(
                    self.twiddle5_re,
                    x49p.mul_add(
                        self.twiddle2_re,
                        x58p.mul_add(self.twiddle4_re, x67p.mul_add(self.twiddle3_re, store[0])),
                    ),
                ),
            ),
        );

        let b112im = x112n.mul_add(
            self.twiddle1_im,
            x211n.mul_add(
                self.twiddle2_im,
                x310n.mul_add(
                    self.twiddle3_im,
                    x49n.mul_add(
                        self.twiddle4_im,
                        x58n.mul_add(self.twiddle5_im, x67n * self.twiddle6_im),
                    ),
                ),
            ),
        );

        let b211im = x112n.mul_add(
            self.twiddle2_im,
            x211n.mul_add(
                self.twiddle4_im,
                x310n.mul_add(
                    self.twiddle6_im,
                    x49n.mul_add(
                        -self.twiddle5_im,
                        x58n.mul_add(-self.twiddle3_im, x67n * -self.twiddle1_im),
                    ),
                ),
            ),
        );

        let b310im = x112n.mul_add(
            self.twiddle3_im,
            x211n.mul_add(
                self.twiddle6_im,
                x310n.mul_add(
                    -self.twiddle4_im,
                    x49n.mul_add(
                        -self.twiddle1_im,
                        x58n.mul_add(self.twiddle2_im, x67n * self.twiddle5_im),
                    ),
                ),
            ),
        );

        let b49im = x112n.mul_add(
            self.twiddle4_im,
            x211n.mul_add(
                -self.twiddle5_im,
                x310n.mul_add(
                    -self.twiddle1_im,
                    x49n.mul_add(
                        self.twiddle3_im,
                        x58n.mul_add(-self.twiddle6_im, x67n * -self.twiddle2_im),
                    ),
                ),
            ),
        );

        let b58im = x112n.mul_add(
            self.twiddle5_im,
            x211n.mul_add(
                -self.twiddle3_im,
                x310n.mul_add(
                    self.twiddle2_im,
                    x49n.mul_add(
                        -self.twiddle6_im,
                        x58n.mul_add(-self.twiddle1_im, x67n * self.twiddle4_im),
                    ),
                ),
            ),
        );

        let b67im = x112n.mul_add(
            self.twiddle6_im,
            x211n.mul_add(
                -self.twiddle1_im,
                x310n.mul_add(
                    self.twiddle5_im,
                    x49n.mul_add(
                        -self.twiddle2_im,
                        x58n.mul_add(self.twiddle4_im, x67n * -self.twiddle3_im),
                    ),
                ),
            ),
        );

        let v0 = y0.zip(AvxStoreD::zero());
        let v1 = b112re.zip(b112im);
        let v2 = b211re.zip(b211im);
        let v3 = b310re.zip(b310im);
        let v4 = b49re.zip(b49im);
        let v5 = b58re.zip(b58im);
        let v6 = b67re.zip(b67im);

        [
            [v0[0], v1[0], v2[0], v3[0], v4[0], v5[0], v6[0]],
            [v0[1], v1[1], v2[1], v3[1], v4[1], v5[1], v6[1]],
        ]
    }
}
