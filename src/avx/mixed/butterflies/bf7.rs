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

pub(crate) struct ColumnButterfly7d {
    rotate: AvxRotate<f64>,
    twiddle1_re: __m256d,
    twiddle2_re: __m256d,
    twiddle3_re: __m256d,
    twiddle1_im: __m256d,
    twiddle2_im: __m256d,
    twiddle3_im: __m256d,
}

impl ColumnButterfly7d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly7d {
        let twiddle1 = compute_twiddle(1, 7, direction);
        let twiddle2 = compute_twiddle(2, 7, direction);
        let twiddle3 = compute_twiddle(3, 7, direction);
        Self {
            rotate: AvxRotate::new(FftDirection::Inverse),
            twiddle1_re: _mm256_set1_pd(twiddle1.re),
            twiddle1_im: _mm256_set1_pd(twiddle1.im),
            twiddle2_re: _mm256_set1_pd(twiddle2.re),
            twiddle2_im: _mm256_set1_pd(twiddle2.im),
            twiddle3_re: _mm256_set1_pd(twiddle3.re),
            twiddle3_im: _mm256_set1_pd(twiddle3.im),
        }
    }
}

impl ColumnButterfly7d {
    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreD; 7]) -> [AvxStoreD; 7] {
        unsafe {
            let (x1p6, x1m6) = AvxButterfly::butterfly2_f64(v[1].v, v[6].v);
            let x1m6 = self.rotate.rotate_m256d(x1m6);
            let y00 = _mm256_add_pd(v[0].v, x1p6);
            let (x2p5, x2m5) = AvxButterfly::butterfly2_f64(v[2].v, v[5].v);
            let x2m5 = self.rotate.rotate_m256d(x2m5);
            let y00 = _mm256_add_pd(y00, x2p5);
            let (x3p4, x3m4) = AvxButterfly::butterfly2_f64(v[3].v, v[4].v);
            let x3m4 = self.rotate.rotate_m256d(x3m4);
            let y00 = _mm256_add_pd(y00, x3p4);

            let m0106a = _mm256_fmadd_pd(x1p6, self.twiddle1_re, v[0].v);
            let m0106a = _mm256_fmadd_pd(x2p5, self.twiddle2_re, m0106a);
            let m0106a = _mm256_fmadd_pd(x3p4, self.twiddle3_re, m0106a);
            let m0106b = _mm256_mul_pd(x1m6, self.twiddle1_im);
            let m0106b = _mm256_fmadd_pd(x2m5, self.twiddle2_im, m0106b);
            let m0106b = _mm256_fmadd_pd(x3m4, self.twiddle3_im, m0106b);
            let (y01, y06) = AvxButterfly::butterfly2_f64(m0106a, m0106b);

            let m0205a = _mm256_fmadd_pd(x1p6, self.twiddle2_re, v[0].v);
            let m0205a = _mm256_fmadd_pd(x2p5, self.twiddle3_re, m0205a);
            let m0205a = _mm256_fmadd_pd(x3p4, self.twiddle1_re, m0205a);
            let m0205b = _mm256_mul_pd(x1m6, self.twiddle2_im);
            let m0205b = _mm256_fnmadd_pd(x2m5, self.twiddle3_im, m0205b);
            let m0205b = _mm256_fnmadd_pd(x3m4, self.twiddle1_im, m0205b);
            let (y02, y05) = AvxButterfly::butterfly2_f64(m0205a, m0205b);

            let m0304a = _mm256_fmadd_pd(x1p6, self.twiddle3_re, v[0].v);
            let m0304a = _mm256_fmadd_pd(x2p5, self.twiddle1_re, m0304a);
            let m0304a = _mm256_fmadd_pd(x3p4, self.twiddle2_re, m0304a);
            let m0304b = _mm256_mul_pd(x1m6, self.twiddle3_im);
            let m0304b = _mm256_fnmadd_pd(x2m5, self.twiddle1_im, m0304b);
            let m0304b = _mm256_fmadd_pd(x3m4, self.twiddle2_im, m0304b);
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
}
pub(crate) struct ColumnButterfly7f {
    rotate: AvxRotate<f32>,
    twiddle1_re: __m256,
    twiddle2_re: __m256,
    twiddle3_re: __m256,
    twiddle1_im: __m256,
    twiddle2_im: __m256,
    twiddle3_im: __m256,
}

impl ColumnButterfly7f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly7f {
        let twiddle1 = compute_twiddle(1, 7, direction);
        let twiddle2 = compute_twiddle(2, 7, direction);
        let twiddle3 = compute_twiddle(3, 7, direction);
        Self {
            rotate: AvxRotate::new(FftDirection::Inverse),
            twiddle1_re: _mm256_set1_ps(twiddle1.re),
            twiddle1_im: _mm256_set1_ps(twiddle1.im),
            twiddle2_re: _mm256_set1_ps(twiddle2.re),
            twiddle2_im: _mm256_set1_ps(twiddle2.im),
            twiddle3_re: _mm256_set1_ps(twiddle3.re),
            twiddle3_im: _mm256_set1_ps(twiddle3.im),
        }
    }
}

impl ColumnButterfly7f {
    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 7]) -> [AvxStoreF; 7] {
        unsafe {
            let (x1p6, x1m6) = AvxButterfly::butterfly2_f32(v[1].v, v[6].v);
            let x1m6 = self.rotate.rotate_m256(x1m6);
            let y00 = _mm256_add_ps(v[0].v, x1p6);
            let (x2p5, x2m5) = AvxButterfly::butterfly2_f32(v[2].v, v[5].v);
            let x2m5 = self.rotate.rotate_m256(x2m5);
            let y00 = _mm256_add_ps(y00, x2p5);
            let (x3p4, x3m4) = AvxButterfly::butterfly2_f32(v[3].v, v[4].v);
            let x3m4 = self.rotate.rotate_m256(x3m4);
            let y00 = _mm256_add_ps(y00, x3p4);

            let m0106a = _mm256_fmadd_ps(x1p6, self.twiddle1_re, v[0].v);
            let m0106a = _mm256_fmadd_ps(x2p5, self.twiddle2_re, m0106a);
            let m0106a = _mm256_fmadd_ps(x3p4, self.twiddle3_re, m0106a);
            let m0106b = _mm256_mul_ps(x1m6, self.twiddle1_im);
            let m0106b = _mm256_fmadd_ps(x2m5, self.twiddle2_im, m0106b);
            let m0106b = _mm256_fmadd_ps(x3m4, self.twiddle3_im, m0106b);
            let (y01, y06) = AvxButterfly::butterfly2_f32(m0106a, m0106b);

            let m0205a = _mm256_fmadd_ps(x1p6, self.twiddle2_re, v[0].v);
            let m0205a = _mm256_fmadd_ps(x2p5, self.twiddle3_re, m0205a);
            let m0205a = _mm256_fmadd_ps(x3p4, self.twiddle1_re, m0205a);
            let m0205b = _mm256_mul_ps(x1m6, self.twiddle2_im);
            let m0205b = _mm256_fnmadd_ps(x2m5, self.twiddle3_im, m0205b);
            let m0205b = _mm256_fnmadd_ps(x3m4, self.twiddle1_im, m0205b);
            let (y02, y05) = AvxButterfly::butterfly2_f32(m0205a, m0205b);

            let m0304a = _mm256_fmadd_ps(x1p6, self.twiddle3_re, v[0].v);
            let m0304a = _mm256_fmadd_ps(x2p5, self.twiddle1_re, m0304a);
            let m0304a = _mm256_fmadd_ps(x3p4, self.twiddle2_re, m0304a);
            let m0304b = _mm256_mul_ps(x1m6, self.twiddle3_im);
            let m0304b = _mm256_fnmadd_ps(x2m5, self.twiddle1_im, m0304b);
            let m0304b = _mm256_fmadd_ps(x3m4, self.twiddle2_im, m0304b);
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

    #[inline(always)]
    pub(crate) fn exec_streaming<A: Fn(usize) -> AvxStoreF, J: FnMut(usize, AvxStoreF)>(
        &self,
        v: A,
        mut store: J,
    ) {
        unsafe {
            let u0 = v(0).v;

            let (x1p6, x1m6) = AvxButterfly::butterfly2_f32(v(1).v, v(6).v);
            let (x2p5, x2m5) = AvxButterfly::butterfly2_f32(v(2).v, v(5).v);
            let (x3p4, x3m4) = AvxButterfly::butterfly2_f32(v(3).v, v(4).v);

            let x1m6 = self.rotate.rotate_m256(x1m6);
            let x2m5 = self.rotate.rotate_m256(x2m5);
            let x3m4 = self.rotate.rotate_m256(x3m4);

            // y00
            let y00 = _mm256_add_ps(u0, x1p6);
            let y00 = _mm256_add_ps(y00, x2p5);
            let y00 = _mm256_add_ps(y00, x3p4);
            store(0, AvxStoreF::raw(y00));

            // (y01, y06)
            let m0106a = _mm256_fmadd_ps(x1p6, self.twiddle1_re, u0);
            let m0106b = _mm256_mul_ps(x1m6, self.twiddle1_im);
            let m0106a = _mm256_fmadd_ps(x2p5, self.twiddle2_re, m0106a);
            let m0106b = _mm256_fmadd_ps(x2m5, self.twiddle2_im, m0106b);
            let m0106a = _mm256_fmadd_ps(x3p4, self.twiddle3_re, m0106a);
            let m0106b = _mm256_fmadd_ps(x3m4, self.twiddle3_im, m0106b);
            let (y01, y06) = AvxButterfly::butterfly2_f32(m0106a, m0106b);
            store(1, AvxStoreF::raw(y01));
            store(6, AvxStoreF::raw(y06));

            // (y02, y05)
            let m0205a = _mm256_fmadd_ps(x1p6, self.twiddle2_re, u0);
            let m0205b = _mm256_mul_ps(x1m6, self.twiddle2_im);
            let m0205a = _mm256_fmadd_ps(x2p5, self.twiddle3_re, m0205a);
            let m0205b = _mm256_fnmadd_ps(x2m5, self.twiddle3_im, m0205b);
            let m0205a = _mm256_fmadd_ps(x3p4, self.twiddle1_re, m0205a);
            let m0205b = _mm256_fnmadd_ps(x3m4, self.twiddle1_im, m0205b);
            let (y02, y05) = AvxButterfly::butterfly2_f32(m0205a, m0205b);
            store(2, AvxStoreF::raw(y02));
            store(5, AvxStoreF::raw(y05));

            // (y03, y04)
            let m0304a = _mm256_fmadd_ps(x1p6, self.twiddle3_re, u0);
            let m0304b = _mm256_mul_ps(x1m6, self.twiddle3_im);
            let m0304a = _mm256_fmadd_ps(x2p5, self.twiddle1_re, m0304a);
            let m0304b = _mm256_fnmadd_ps(x2m5, self.twiddle1_im, m0304b);
            let m0304a = _mm256_fmadd_ps(x3p4, self.twiddle2_re, m0304a);
            let m0304b = _mm256_fmadd_ps(x3m4, self.twiddle2_im, m0304b);
            let (y03, y04) = AvxButterfly::butterfly2_f32(m0304a, m0304b);
            store(3, AvxStoreF::raw(y03));
            store(4, AvxStoreF::raw(y04));
        }
    }
}

pub(crate) struct ColumnRdftButterfly7f {
    twiddle1_re: AvxStoreF,
    twiddle2_re: AvxStoreF,
    twiddle3_re: AvxStoreF,
    twiddle1_im: AvxStoreF,
    twiddle2_im: AvxStoreF,
    twiddle3_im: AvxStoreF,
}

impl ColumnRdftButterfly7f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new() -> Self {
        let twiddle1 = compute_twiddle(1, 7, FftDirection::Forward);
        let twiddle2 = compute_twiddle(2, 7, FftDirection::Forward);
        let twiddle3 = compute_twiddle(3, 7, FftDirection::Forward);
        Self {
            twiddle1_re: AvxStoreF::dup(twiddle1.re),
            twiddle1_im: AvxStoreF::dup(twiddle1.im),
            twiddle2_re: AvxStoreF::dup(twiddle2.re),
            twiddle2_im: AvxStoreF::dup(twiddle2.im),
            twiddle3_re: AvxStoreF::dup(twiddle3.re),
            twiddle3_im: AvxStoreF::dup(twiddle3.im),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec(&self, store: [AvxStoreF; 7]) -> [[AvxStoreF; 4]; 2] {
        let x16p = store[1] + store[6];
        let x16n = store[1] - store[6];
        let x25p = store[2] + store[5];
        let x25n = store[2] - store[5];
        let x34p = store[3] + store[4];
        let x34n = store[3] - store[4];
        let y0 = store[0] + x16p + x25p + x34p;

        let x16re_a = x16p.mul_add(
            self.twiddle1_re,
            x25p.mul_add(self.twiddle2_re, x34p.mul_add(self.twiddle3_re, store[0])),
        );
        let x25re_a = x34p.mul_add(
            self.twiddle1_re,
            x16p.mul_add(self.twiddle2_re, x25p.mul_add(self.twiddle3_re, store[0])),
        );
        let x34re_a = x25p.mul_add(
            self.twiddle1_re,
            x34p.mul_add(self.twiddle2_re, x16p.mul_add(self.twiddle3_re, store[0])),
        );
        let x16im_b = x16n.mul_add(
            self.twiddle1_im,
            x25n.mul_add(self.twiddle2_im, x34n * self.twiddle3_im),
        );
        let x25im_b = x34n.mul_add(
            -self.twiddle1_im,
            x16n.mul_add(self.twiddle2_im, x25n * -self.twiddle3_im),
        );
        let x34im_b = x25n.mul_add(
            self.twiddle1_im,
            x34n.mul_add(-self.twiddle2_im, x16n * -self.twiddle3_im),
        );
        let v0 = y0.zip(AvxStoreF::zero());
        let v1 = x16re_a.zip(x16im_b);
        let v2 = x25re_a.zip(x25im_b);
        let v3 = x34re_a.zip(-x34im_b);
        [[v0[0], v1[0], v2[0], v3[0]], [v0[1], v1[1], v2[1], v3[1]]]
    }
}

pub(crate) struct ColumnRdftButterfly7d {
    twiddle1_re: AvxStoreD,
    twiddle2_re: AvxStoreD,
    twiddle3_re: AvxStoreD,
    twiddle1_im: AvxStoreD,
    twiddle2_im: AvxStoreD,
    twiddle3_im: AvxStoreD,
}

impl ColumnRdftButterfly7d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new() -> Self {
        let twiddle1 = compute_twiddle(1, 7, FftDirection::Forward);
        let twiddle2 = compute_twiddle(2, 7, FftDirection::Forward);
        let twiddle3 = compute_twiddle(3, 7, FftDirection::Forward);
        Self {
            twiddle1_re: AvxStoreD::dup(twiddle1.re),
            twiddle1_im: AvxStoreD::dup(twiddle1.im),
            twiddle2_re: AvxStoreD::dup(twiddle2.re),
            twiddle2_im: AvxStoreD::dup(twiddle2.im),
            twiddle3_re: AvxStoreD::dup(twiddle3.re),
            twiddle3_im: AvxStoreD::dup(twiddle3.im),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec(&self, store: [AvxStoreD; 7]) -> [[AvxStoreD; 4]; 2] {
        let x16p = store[1] + store[6];
        let x16n = store[1] - store[6];
        let x25p = store[2] + store[5];
        let x25n = store[2] - store[5];
        let x34p = store[3] + store[4];
        let x34n = store[3] - store[4];
        let y0 = store[0] + x16p + x25p + x34p;

        let x16re_a = x16p.mul_add(
            self.twiddle1_re,
            x25p.mul_add(self.twiddle2_re, x34p.mul_add(self.twiddle3_re, store[0])),
        );
        let x25re_a = x34p.mul_add(
            self.twiddle1_re,
            x16p.mul_add(self.twiddle2_re, x25p.mul_add(self.twiddle3_re, store[0])),
        );
        let x34re_a = x25p.mul_add(
            self.twiddle1_re,
            x34p.mul_add(self.twiddle2_re, x16p.mul_add(self.twiddle3_re, store[0])),
        );
        let x16im_b = x16n.mul_add(
            self.twiddle1_im,
            x25n.mul_add(self.twiddle2_im, x34n * self.twiddle3_im),
        );
        let x25im_b = x34n.mul_add(
            -self.twiddle1_im,
            x16n.mul_add(self.twiddle2_im, x25n * -self.twiddle3_im),
        );
        let x34im_b = x25n.mul_add(
            self.twiddle1_im,
            x34n.mul_add(-self.twiddle2_im, x16n * -self.twiddle3_im),
        );
        let v0 = y0.zip(AvxStoreD::zero());
        let v1 = x16re_a.zip(x16im_b);
        let v2 = x25re_a.zip(x25im_b);
        let v3 = x34re_a.zip(-x34im_b);
        [[v0[0], v1[0], v2[0], v3[0]], [v0[1], v1[1], v2[1], v3[1]]]
    }
}
