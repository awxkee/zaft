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
use crate::avx::mixed::{ColumnButterfly8d, ColumnButterfly8f};
use crate::util::compute_twiddle;
use std::arch::x86_64::*;

pub(crate) struct ColumnButterfly16d {
    pub(crate) bf8: ColumnButterfly8d,
    twiddle1: AvxStoreD,
    twiddle2: AvxStoreD,
    twiddle3: AvxStoreD,
}

impl ColumnButterfly16d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly16d {
        let tw1 = compute_twiddle(1, 16, direction);
        let tw2 = compute_twiddle(2, 16, direction);
        let tw3 = compute_twiddle(3, 16, direction);
        Self {
            bf8: ColumnButterfly8d::new(direction),
            twiddle1: AvxStoreD::set_values(tw1.re, tw1.im, tw1.re, tw1.im),
            twiddle2: AvxStoreD::set_values(tw2.re, tw2.im, tw2.re, tw2.im),
            twiddle3: AvxStoreD::set_values(tw3.re, tw3.im, tw3.re, tw3.im),
        }
    }
}

impl ColumnButterfly16d {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub(crate) fn exec(&self, v: [AvxStoreD; 16]) -> [AvxStoreD; 16] {
        let evens = self
            .bf8
            .exec([v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]]);

        let mut odds_1 = self.bf8.bf4.exec([v[1], v[5], v[9], v[13]]);
        let mut odds_2 = self.bf8.bf4.exec([v[15], v[3], v[7], v[11]]);

        odds_1[1] = AvxStoreD::mul_by_complex(odds_1[1], self.twiddle1);
        odds_2[1] = AvxStoreD::mul_by_complex_conj_b(odds_2[1], self.twiddle1);

        odds_1[2] = AvxStoreD::mul_by_complex(odds_1[2], self.twiddle2);
        odds_2[2] = AvxStoreD::mul_by_complex_conj_b(odds_2[2], self.twiddle2);

        odds_1[3] = AvxStoreD::mul_by_complex(odds_1[3], self.twiddle3);
        odds_2[3] = AvxStoreD::mul_by_complex_conj_b(odds_2[3], self.twiddle3);

        // step 4: cross FFTs
        let (o01, o02) = AvxButterfly::butterfly2_f64(odds_1[0].v, odds_2[0].v);
        odds_1[0] = AvxStoreD::raw(o01);
        odds_2[0] = AvxStoreD::raw(o02);

        let (o03, o04) = AvxButterfly::butterfly2_f64(odds_1[1].v, odds_2[1].v);
        odds_1[1] = AvxStoreD::raw(o03);
        odds_2[1] = AvxStoreD::raw(o04);
        let (o05, o06) = AvxButterfly::butterfly2_f64(odds_1[2].v, odds_2[2].v);
        odds_1[2] = AvxStoreD::raw(o05);
        odds_2[2] = AvxStoreD::raw(o06);
        let (o07, o08) = AvxButterfly::butterfly2_f64(odds_1[3].v, odds_2[3].v);
        odds_1[3] = AvxStoreD::raw(o07);
        odds_2[3] = AvxStoreD::raw(o08);

        // apply the butterfly 4 twiddle factor, which is just a rotation
        odds_2[0] = AvxStoreD::raw(self.bf8.bf4.rotate.rotate_m256d(odds_2[0].v));
        odds_2[1] = AvxStoreD::raw(self.bf8.bf4.rotate.rotate_m256d(odds_2[1].v));
        odds_2[2] = AvxStoreD::raw(self.bf8.bf4.rotate.rotate_m256d(odds_2[2].v));
        odds_2[3] = AvxStoreD::raw(self.bf8.bf4.rotate.rotate_m256d(odds_2[3].v));

        [
            AvxStoreD::raw(_mm256_add_pd(evens[0].v, odds_1[0].v)),
            AvxStoreD::raw(_mm256_add_pd(evens[1].v, odds_1[1].v)),
            AvxStoreD::raw(_mm256_add_pd(evens[2].v, odds_1[2].v)),
            AvxStoreD::raw(_mm256_add_pd(evens[3].v, odds_1[3].v)),
            AvxStoreD::raw(_mm256_add_pd(evens[4].v, odds_2[0].v)),
            AvxStoreD::raw(_mm256_add_pd(evens[5].v, odds_2[1].v)),
            AvxStoreD::raw(_mm256_add_pd(evens[6].v, odds_2[2].v)),
            AvxStoreD::raw(_mm256_add_pd(evens[7].v, odds_2[3].v)),
            AvxStoreD::raw(_mm256_sub_pd(evens[0].v, odds_1[0].v)),
            AvxStoreD::raw(_mm256_sub_pd(evens[1].v, odds_1[1].v)),
            AvxStoreD::raw(_mm256_sub_pd(evens[2].v, odds_1[2].v)),
            AvxStoreD::raw(_mm256_sub_pd(evens[3].v, odds_1[3].v)),
            AvxStoreD::raw(_mm256_sub_pd(evens[4].v, odds_2[0].v)),
            AvxStoreD::raw(_mm256_sub_pd(evens[5].v, odds_2[1].v)),
            AvxStoreD::raw(_mm256_sub_pd(evens[6].v, odds_2[2].v)),
            AvxStoreD::raw(_mm256_sub_pd(evens[7].v, odds_2[3].v)),
        ]
    }
}

pub(crate) struct ColumnButterfly16f {
    pub(crate) bf8: ColumnButterfly8f,
    twiddle1: AvxStoreF,
    twiddle2: AvxStoreF,
    twiddle3: AvxStoreF,
}

impl ColumnButterfly16f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly16f {
        let tw1 = compute_twiddle(1, 16, direction);
        let tw2 = compute_twiddle(2, 16, direction);
        let tw3 = compute_twiddle(3, 16, direction);
        Self {
            bf8: ColumnButterfly8f::new(direction),
            twiddle1: AvxStoreF::set_values8(
                tw1.re, tw1.im, tw1.re, tw1.im, tw1.re, tw1.im, tw1.re, tw1.im,
            ),
            twiddle2: AvxStoreF::set_values8(
                tw2.re, tw2.im, tw2.re, tw2.im, tw2.re, tw2.im, tw2.re, tw2.im,
            ),
            twiddle3: AvxStoreF::set_values8(
                tw3.re, tw3.im, tw3.re, tw3.im, tw3.re, tw3.im, tw3.re, tw3.im,
            ),
        }
    }
}

impl ColumnButterfly16f {
    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 16]) -> [AvxStoreF; 16] {
        unsafe {
            let evens = self
                .bf8
                .exec([v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]]);

            let mut odds_1 = self.bf8.bf4.exec([v[1], v[5], v[9], v[13]]);
            let mut odds_2 = self.bf8.bf4.exec([v[15], v[3], v[7], v[11]]);

            odds_1[1] = AvxStoreF::mul_by_complex(odds_1[1], self.twiddle1);
            odds_2[1] = AvxStoreF::mul_by_complex_conj_b(odds_2[1], self.twiddle1);

            odds_1[2] = AvxStoreF::mul_by_complex(odds_1[2], self.twiddle2);
            odds_2[2] = AvxStoreF::mul_by_complex_conj_b(odds_2[2], self.twiddle2);

            odds_1[3] = AvxStoreF::mul_by_complex(odds_1[3], self.twiddle3);
            odds_2[3] = AvxStoreF::mul_by_complex_conj_b(odds_2[3], self.twiddle3);

            let (o01, o02) = AvxButterfly::butterfly2_f32(odds_1[0].v, odds_2[0].v);
            odds_1[0] = AvxStoreF::raw(o01);
            odds_2[0] = AvxStoreF::raw(o02);

            let (o03, o04) = AvxButterfly::butterfly2_f32(odds_1[1].v, odds_2[1].v);
            odds_1[1] = AvxStoreF::raw(o03);
            odds_2[1] = AvxStoreF::raw(o04);
            let (o05, o06) = AvxButterfly::butterfly2_f32(odds_1[2].v, odds_2[2].v);
            odds_1[2] = AvxStoreF::raw(o05);
            odds_2[2] = AvxStoreF::raw(o06);
            let (o07, o08) = AvxButterfly::butterfly2_f32(odds_1[3].v, odds_2[3].v);
            odds_1[3] = AvxStoreF::raw(o07);
            odds_2[3] = AvxStoreF::raw(o08);

            // apply the butterfly 4 twiddle factor, which is just a rotation
            odds_2[0] = AvxStoreF::raw(self.bf8.bf4.rotate.rotate_m256(odds_2[0].v));
            odds_2[1] = AvxStoreF::raw(self.bf8.bf4.rotate.rotate_m256(odds_2[1].v));
            odds_2[2] = AvxStoreF::raw(self.bf8.bf4.rotate.rotate_m256(odds_2[2].v));
            odds_2[3] = AvxStoreF::raw(self.bf8.bf4.rotate.rotate_m256(odds_2[3].v));

            [
                AvxStoreF::raw(_mm256_add_ps(evens[0].v, odds_1[0].v)),
                AvxStoreF::raw(_mm256_add_ps(evens[1].v, odds_1[1].v)),
                AvxStoreF::raw(_mm256_add_ps(evens[2].v, odds_1[2].v)),
                AvxStoreF::raw(_mm256_add_ps(evens[3].v, odds_1[3].v)),
                AvxStoreF::raw(_mm256_add_ps(evens[4].v, odds_2[0].v)),
                AvxStoreF::raw(_mm256_add_ps(evens[5].v, odds_2[1].v)),
                AvxStoreF::raw(_mm256_add_ps(evens[6].v, odds_2[2].v)),
                AvxStoreF::raw(_mm256_add_ps(evens[7].v, odds_2[3].v)),
                AvxStoreF::raw(_mm256_sub_ps(evens[0].v, odds_1[0].v)),
                AvxStoreF::raw(_mm256_sub_ps(evens[1].v, odds_1[1].v)),
                AvxStoreF::raw(_mm256_sub_ps(evens[2].v, odds_1[2].v)),
                AvxStoreF::raw(_mm256_sub_ps(evens[3].v, odds_1[3].v)),
                AvxStoreF::raw(_mm256_sub_ps(evens[4].v, odds_2[0].v)),
                AvxStoreF::raw(_mm256_sub_ps(evens[5].v, odds_2[1].v)),
                AvxStoreF::raw(_mm256_sub_ps(evens[6].v, odds_2[2].v)),
                AvxStoreF::raw(_mm256_sub_ps(evens[7].v, odds_2[3].v)),
            ]
        }
    }
}
