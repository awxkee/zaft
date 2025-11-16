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
use crate::avx::butterflies::{AvxButterfly, AvxFastButterfly8};
use crate::avx::mixed::avx_stored::AvxStoreD;
use crate::avx::mixed::avx_storef::AvxStoreF;
use crate::avx::rotate::AvxRotate;
use crate::avx::util::{
    _mm256_fcmul_pd, _mm256_fcmul_pd_conj_b, _mm256_fcmul_ps, _mm256_fcmul_ps_conj_b,
};
use crate::util::compute_twiddle;
use std::arch::x86_64::*;

pub(crate) struct ColumnButterfly16d {
    pub(crate) rotate: AvxRotate<f64>,
    pub(crate) bf8: AvxFastButterfly8<f64>,
    twiddle1: __m256d,
    twiddle2: __m256d,
    twiddle3: __m256d,
}

impl ColumnButterfly16d {
    #[target_feature(enable = "avx")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly16d {
        let tw1 = compute_twiddle(1, 16, direction);
        let tw2 = compute_twiddle(2, 16, direction);
        let tw3 = compute_twiddle(3, 16, direction);
        unsafe {
            Self {
                rotate: AvxRotate::new(direction),
                bf8: AvxFastButterfly8::new(direction),
                twiddle1: _mm256_loadu_pd([tw1.re, tw1.im, tw1.re, tw1.im].as_ptr()),
                twiddle2: _mm256_loadu_pd([tw2.re, tw2.im, tw2.re, tw2.im].as_ptr()),
                twiddle3: _mm256_loadu_pd([tw3.re, tw3.im, tw3.re, tw3.im].as_ptr()),
            }
        }
    }
}

impl ColumnButterfly16d {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) fn exec(&self, v: [AvxStoreD; 16]) -> [AvxStoreD; 16] {
        let evens = self.bf8.exec(
            v[0].v, v[2].v, v[4].v, v[6].v, v[8].v, v[10].v, v[12].v, v[14].v,
        );

        let mut odds_1 =
            AvxButterfly::butterfly4_f64(v[1].v, v[5].v, v[9].v, v[13].v, self.rotate.rot_flag);
        let mut odds_2 =
            AvxButterfly::butterfly4_f64(v[15].v, v[3].v, v[7].v, v[11].v, self.rotate.rot_flag);

        odds_1.1 = _mm256_fcmul_pd(odds_1.1, self.twiddle1);
        odds_2.1 = _mm256_fcmul_pd_conj_b(odds_2.1, self.twiddle1);

        odds_1.2 = _mm256_fcmul_pd(odds_1.2, self.twiddle2);
        odds_2.2 = _mm256_fcmul_pd_conj_b(odds_2.2, self.twiddle2);

        odds_1.3 = _mm256_fcmul_pd(odds_1.3, self.twiddle3);
        odds_2.3 = _mm256_fcmul_pd_conj_b(odds_2.3, self.twiddle3);

        // step 4: cross FFTs
        let (o01, o02) = AvxButterfly::butterfly2_f64(odds_1.0, odds_2.0);
        odds_1.0 = o01;
        odds_2.0 = o02;

        let (o03, o04) = AvxButterfly::butterfly2_f64(odds_1.1, odds_2.1);
        odds_1.1 = o03;
        odds_2.1 = o04;
        let (o05, o06) = AvxButterfly::butterfly2_f64(odds_1.2, odds_2.2);
        odds_1.2 = o05;
        odds_2.2 = o06;
        let (o07, o08) = AvxButterfly::butterfly2_f64(odds_1.3, odds_2.3);
        odds_1.3 = o07;
        odds_2.3 = o08;

        // apply the butterfly 4 twiddle factor, which is just a rotation
        odds_2.0 = self.rotate.rotate_m256d(odds_2.0);
        odds_2.1 = self.rotate.rotate_m256d(odds_2.1);
        odds_2.2 = self.rotate.rotate_m256d(odds_2.2);
        odds_2.3 = self.rotate.rotate_m256d(odds_2.3);

        [
            AvxStoreD::raw(_mm256_add_pd(evens.0, odds_1.0)),
            AvxStoreD::raw(_mm256_add_pd(evens.1, odds_1.1)),
            AvxStoreD::raw(_mm256_add_pd(evens.2, odds_1.2)),
            AvxStoreD::raw(_mm256_add_pd(evens.3, odds_1.3)),
            AvxStoreD::raw(_mm256_add_pd(evens.4, odds_2.0)),
            AvxStoreD::raw(_mm256_add_pd(evens.5, odds_2.1)),
            AvxStoreD::raw(_mm256_add_pd(evens.6, odds_2.2)),
            AvxStoreD::raw(_mm256_add_pd(evens.7, odds_2.3)),
            AvxStoreD::raw(_mm256_sub_pd(evens.0, odds_1.0)),
            AvxStoreD::raw(_mm256_sub_pd(evens.1, odds_1.1)),
            AvxStoreD::raw(_mm256_sub_pd(evens.2, odds_1.2)),
            AvxStoreD::raw(_mm256_sub_pd(evens.3, odds_1.3)),
            AvxStoreD::raw(_mm256_sub_pd(evens.4, odds_2.0)),
            AvxStoreD::raw(_mm256_sub_pd(evens.5, odds_2.1)),
            AvxStoreD::raw(_mm256_sub_pd(evens.6, odds_2.2)),
            AvxStoreD::raw(_mm256_sub_pd(evens.7, odds_2.3)),
        ]
    }
}

pub(crate) struct ColumnButterfly16f {
    pub(crate) rotate: AvxRotate<f32>,
    pub(crate) bf8: AvxFastButterfly8<f32>,
    twiddle1: __m256,
    twiddle2: __m256,
    twiddle3: __m256,
}

impl ColumnButterfly16f {
    #[target_feature(enable = "avx")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly16f {
        let tw1 = compute_twiddle(1, 16, direction);
        let tw2 = compute_twiddle(2, 16, direction);
        let tw3 = compute_twiddle(3, 16, direction);
        unsafe {
            Self {
                rotate: AvxRotate::new(direction),
                bf8: AvxFastButterfly8::new(direction),
                twiddle1: _mm256_loadu_ps(
                    [
                        tw1.re, tw1.im, tw1.re, tw1.im, tw1.re, tw1.im, tw1.re, tw1.im,
                    ]
                    .as_ptr(),
                ),
                twiddle2: _mm256_loadu_ps(
                    [
                        tw2.re, tw2.im, tw2.re, tw2.im, tw2.re, tw2.im, tw2.re, tw2.im,
                    ]
                    .as_ptr(),
                ),
                twiddle3: _mm256_loadu_ps(
                    [
                        tw3.re, tw3.im, tw3.re, tw3.im, tw3.re, tw3.im, tw3.re, tw3.im,
                    ]
                    .as_ptr(),
                ),
            }
        }
    }
}

impl ColumnButterfly16f {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) fn exec(&self, v: [AvxStoreF; 16]) -> [AvxStoreF; 16] {
        let evens = self.bf8.exec(
            v[0].v, v[2].v, v[4].v, v[6].v, v[8].v, v[10].v, v[12].v, v[14].v,
        );

        let mut odds_1 = AvxButterfly::butterfly4_f32(
            v[1].v,
            v[5].v,
            v[9].v,
            v[13].v,
            _mm256_castpd_ps(self.rotate.rot_flag),
        );
        let mut odds_2 = AvxButterfly::butterfly4_f32(
            v[15].v,
            v[3].v,
            v[7].v,
            v[11].v,
            _mm256_castpd_ps(self.rotate.rot_flag),
        );

        odds_1.1 = _mm256_fcmul_ps(odds_1.1, self.twiddle1);
        odds_2.1 = _mm256_fcmul_ps_conj_b(odds_2.1, self.twiddle1);

        odds_1.2 = _mm256_fcmul_ps(odds_1.2, self.twiddle2);
        odds_2.2 = _mm256_fcmul_ps_conj_b(odds_2.2, self.twiddle2);

        odds_1.3 = _mm256_fcmul_ps(odds_1.3, self.twiddle3);
        odds_2.3 = _mm256_fcmul_ps_conj_b(odds_2.3, self.twiddle3);

        // step 4: cross FFTs
        let (o01, o02) = AvxButterfly::butterfly2_f32(odds_1.0, odds_2.0);
        odds_1.0 = o01;
        odds_2.0 = o02;

        let (o03, o04) = AvxButterfly::butterfly2_f32(odds_1.1, odds_2.1);
        odds_1.1 = o03;
        odds_2.1 = o04;
        let (o05, o06) = AvxButterfly::butterfly2_f32(odds_1.2, odds_2.2);
        odds_1.2 = o05;
        odds_2.2 = o06;
        let (o07, o08) = AvxButterfly::butterfly2_f32(odds_1.3, odds_2.3);
        odds_1.3 = o07;
        odds_2.3 = o08;

        // apply the butterfly 4 twiddle factor, which is just a rotation
        odds_2.0 = self.rotate.rotate_m256(odds_2.0);
        odds_2.1 = self.rotate.rotate_m256(odds_2.1);
        odds_2.2 = self.rotate.rotate_m256(odds_2.2);
        odds_2.3 = self.rotate.rotate_m256(odds_2.3);

        [
            AvxStoreF::raw(_mm256_add_ps(evens.0, odds_1.0)),
            AvxStoreF::raw(_mm256_add_ps(evens.1, odds_1.1)),
            AvxStoreF::raw(_mm256_add_ps(evens.2, odds_1.2)),
            AvxStoreF::raw(_mm256_add_ps(evens.3, odds_1.3)),
            AvxStoreF::raw(_mm256_add_ps(evens.4, odds_2.0)),
            AvxStoreF::raw(_mm256_add_ps(evens.5, odds_2.1)),
            AvxStoreF::raw(_mm256_add_ps(evens.6, odds_2.2)),
            AvxStoreF::raw(_mm256_add_ps(evens.7, odds_2.3)),
            AvxStoreF::raw(_mm256_sub_ps(evens.0, odds_1.0)),
            AvxStoreF::raw(_mm256_sub_ps(evens.1, odds_1.1)),
            AvxStoreF::raw(_mm256_sub_ps(evens.2, odds_1.2)),
            AvxStoreF::raw(_mm256_sub_ps(evens.3, odds_1.3)),
            AvxStoreF::raw(_mm256_sub_ps(evens.4, odds_2.0)),
            AvxStoreF::raw(_mm256_sub_ps(evens.5, odds_2.1)),
            AvxStoreF::raw(_mm256_sub_ps(evens.6, odds_2.2)),
            AvxStoreF::raw(_mm256_sub_ps(evens.7, odds_2.3)),
        ]
    }
}
