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
#[cfg(feature = "fcma")]
use crate::neon::butterflies::{FastFcmaBf4d, FastFcmaBf4f};
use crate::neon::butterflies::{NeonButterfly, NeonFastButterfly8};
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF, NeonStoreFh};
use crate::neon::util::*;
use crate::util::compute_twiddle;
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly16d {
    rotate: float64x2_t,
    bf8: NeonFastButterfly8<f64>,
    twiddle1: float64x2_t,
    twiddle2: float64x2_t,
    twiddle3: float64x2_t,
}

impl ColumnButterfly16d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let tw1 = compute_twiddle(1, 16, fft_direction);
            let tw2 = compute_twiddle(2, 16, fft_direction);
            let tw3 = compute_twiddle(3, 16, fft_direction);
            Self {
                rotate: vld1q_f64(match fft_direction {
                    FftDirection::Inverse => [-0.0, 0.0].as_ptr(),
                    FftDirection::Forward => [0.0, -0.0].as_ptr(),
                }),
                bf8: NeonFastButterfly8::new(fft_direction),
                twiddle1: vld1q_f64([tw1.re, tw1.im].as_ptr()),
                twiddle2: vld1q_f64([tw2.re, tw2.im].as_ptr()),
                twiddle3: vld1q_f64([tw3.re, tw3.im].as_ptr()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, v: [NeonStoreD; 16]) -> [NeonStoreD; 16] {
        unsafe {
            let evens = self.bf8.exec(
                v[0].v,
                v[2].v,
                v[4].v,
                v[6].v,
                v[8].v,
                v[10].v,
                v[12].v,
                v[14].v,
                self.rotate,
            );

            let mut odds_1 =
                NeonButterfly::butterfly4_f64(v[1].v, v[5].v, v[9].v, v[13].v, self.rotate);
            let mut odds_2 =
                NeonButterfly::butterfly4_f64(v[15].v, v[3].v, v[7].v, v[11].v, self.rotate);

            odds_1.1 = vfcmulq_f64(odds_1.1, self.twiddle1);
            odds_2.1 = vfcmulq_conj_b_f64(odds_2.1, self.twiddle1);

            odds_1.2 = vfcmulq_f64(odds_1.2, self.twiddle2);
            odds_2.2 = vfcmulq_conj_b_f64(odds_2.2, self.twiddle2);

            odds_1.3 = vfcmulq_f64(odds_1.3, self.twiddle3);
            odds_2.3 = vfcmulq_conj_b_f64(odds_2.3, self.twiddle3);

            // step 4: cross FFTs
            let (o01, o02) = NeonButterfly::butterfly2_f64(odds_1.0, odds_2.0);
            odds_1.0 = o01;
            odds_2.0 = o02;

            let (o03, o04) = NeonButterfly::butterfly2_f64(odds_1.1, odds_2.1);
            odds_1.1 = o03;
            odds_2.1 = o04;
            let (o05, o06) = NeonButterfly::butterfly2_f64(odds_1.2, odds_2.2);
            odds_1.2 = o05;
            odds_2.2 = o06;
            let (o07, o08) = NeonButterfly::butterfly2_f64(odds_1.3, odds_2.3);
            odds_1.3 = o07;
            odds_2.3 = o08;

            // apply the butterfly 4 twiddle factor, which is just a rotation
            odds_2.0 = v_rotate90_f64(odds_2.0, self.rotate);
            odds_2.1 = v_rotate90_f64(odds_2.1, self.rotate);
            odds_2.2 = v_rotate90_f64(odds_2.2, self.rotate);
            odds_2.3 = v_rotate90_f64(odds_2.3, self.rotate);

            [
                NeonStoreD::raw(vaddq_f64(evens.0, odds_1.0)),
                NeonStoreD::raw(vaddq_f64(evens.1, odds_1.1)),
                NeonStoreD::raw(vaddq_f64(evens.2, odds_1.2)),
                NeonStoreD::raw(vaddq_f64(evens.3, odds_1.3)),
                NeonStoreD::raw(vaddq_f64(evens.4, odds_2.0)),
                NeonStoreD::raw(vaddq_f64(evens.5, odds_2.1)),
                NeonStoreD::raw(vaddq_f64(evens.6, odds_2.2)),
                NeonStoreD::raw(vaddq_f64(evens.7, odds_2.3)),
                NeonStoreD::raw(vsubq_f64(evens.0, odds_1.0)),
                NeonStoreD::raw(vsubq_f64(evens.1, odds_1.1)),
                NeonStoreD::raw(vsubq_f64(evens.2, odds_1.2)),
                NeonStoreD::raw(vsubq_f64(evens.3, odds_1.3)),
                NeonStoreD::raw(vsubq_f64(evens.4, odds_2.0)),
                NeonStoreD::raw(vsubq_f64(evens.5, odds_2.1)),
                NeonStoreD::raw(vsubq_f64(evens.6, odds_2.2)),
                NeonStoreD::raw(vsubq_f64(evens.7, odds_2.3)),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly16d {
    bf8: NeonFastButterfly8<f64>,
    twiddle1: float64x2_t,
    twiddle2: float64x2_t,
    twiddle3: float64x2_t,
    bf4: FastFcmaBf4d,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly16d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let tw1 = compute_twiddle(1, 16, fft_direction);
            let tw2 = compute_twiddle(2, 16, fft_direction);
            let tw3 = compute_twiddle(3, 16, fft_direction);
            Self {
                bf8: NeonFastButterfly8::new(fft_direction),
                twiddle1: vld1q_f64([tw1.re, tw1.im].as_ptr()),
                twiddle2: vld1q_f64([tw2.re, tw2.im].as_ptr()),
                twiddle3: vld1q_f64([tw3.re, tw3.im].as_ptr()),
                bf4: FastFcmaBf4d::new(fft_direction),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, v: [NeonStoreD; 16]) -> [NeonStoreD; 16] {
        let evens = self.bf8.exec_fcma(
            v[0].v, v[2].v, v[4].v, v[6].v, v[8].v, v[10].v, v[12].v, v[14].v, &self.bf4,
        );

        let mut odds_1 = self.bf4.exec(v[1].v, v[5].v, v[9].v, v[13].v);
        let mut odds_2 = self.bf4.exec(v[15].v, v[3].v, v[7].v, v[11].v);

        odds_1.1 = vfcmulq_fcma_f64(odds_1.1, self.twiddle1);
        odds_2.1 = vfcmulq_conj_b_fcma_f64(odds_2.1, self.twiddle1);

        odds_1.2 = vfcmulq_fcma_f64(odds_1.2, self.twiddle2);
        odds_2.2 = vfcmulq_conj_b_fcma_f64(odds_2.2, self.twiddle2);

        odds_1.3 = vfcmulq_fcma_f64(odds_1.3, self.twiddle3);
        odds_2.3 = vfcmulq_conj_b_fcma_f64(odds_2.3, self.twiddle3);

        // step 4: cross FFTs
        let (o01, o02) = NeonButterfly::butterfly2_f64(odds_1.0, odds_2.0);
        odds_1.0 = o01;
        odds_2.0 = o02;

        let (o03, o04) = NeonButterfly::butterfly2_f64(odds_1.1, odds_2.1);
        odds_1.1 = o03;
        odds_2.1 = o04;
        let (o05, o06) = NeonButterfly::butterfly2_f64(odds_1.2, odds_2.2);
        odds_1.2 = o05;
        odds_2.2 = o06;
        let (o07, o08) = NeonButterfly::butterfly2_f64(odds_1.3, odds_2.3);
        odds_1.3 = o07;
        odds_2.3 = o08;

        // apply the butterfly 4 twiddle factor, which is just a rotation
        odds_2.0 = vcmlaq_rot90_f64(vdupq_n_f64(0.), self.bf4.rot_sign, odds_2.0);
        odds_2.1 = vcmlaq_rot90_f64(vdupq_n_f64(0.), self.bf4.rot_sign, odds_2.1);
        odds_2.2 = vcmlaq_rot90_f64(vdupq_n_f64(0.), self.bf4.rot_sign, odds_2.2);
        odds_2.3 = vcmlaq_rot90_f64(vdupq_n_f64(0.), self.bf4.rot_sign, odds_2.3);

        [
            NeonStoreD::raw(vaddq_f64(evens.0, odds_1.0)),
            NeonStoreD::raw(vaddq_f64(evens.1, odds_1.1)),
            NeonStoreD::raw(vaddq_f64(evens.2, odds_1.2)),
            NeonStoreD::raw(vaddq_f64(evens.3, odds_1.3)),
            NeonStoreD::raw(vaddq_f64(evens.4, odds_2.0)),
            NeonStoreD::raw(vaddq_f64(evens.5, odds_2.1)),
            NeonStoreD::raw(vaddq_f64(evens.6, odds_2.2)),
            NeonStoreD::raw(vaddq_f64(evens.7, odds_2.3)),
            NeonStoreD::raw(vsubq_f64(evens.0, odds_1.0)),
            NeonStoreD::raw(vsubq_f64(evens.1, odds_1.1)),
            NeonStoreD::raw(vsubq_f64(evens.2, odds_1.2)),
            NeonStoreD::raw(vsubq_f64(evens.3, odds_1.3)),
            NeonStoreD::raw(vsubq_f64(evens.4, odds_2.0)),
            NeonStoreD::raw(vsubq_f64(evens.5, odds_2.1)),
            NeonStoreD::raw(vsubq_f64(evens.6, odds_2.2)),
            NeonStoreD::raw(vsubq_f64(evens.7, odds_2.3)),
        ]
    }
}

pub(crate) struct ColumnButterfly16f {
    rotate: float32x4_t,
    bf8: NeonFastButterfly8<f32>,
    twiddle1: float32x4_t,
    twiddle2: float32x4_t,
    twiddle3: float32x4_t,
}

impl ColumnButterfly16f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let tw1 = compute_twiddle(1, 16, fft_direction);
            let tw2 = compute_twiddle(2, 16, fft_direction);
            let tw3 = compute_twiddle(3, 16, fft_direction);
            Self {
                rotate: vld1q_f32(match fft_direction {
                    FftDirection::Inverse => [-0.0, 0.0, -0.0, 0.0].as_ptr(),
                    FftDirection::Forward => [0.0, -0.0, 0.0, -0.0].as_ptr(),
                }),
                bf8: NeonFastButterfly8::new(fft_direction),
                twiddle1: vld1q_f32([tw1.re, tw1.im, tw1.re, tw1.im].as_ptr()),
                twiddle2: vld1q_f32([tw2.re, tw2.im, tw2.re, tw2.im].as_ptr()),
                twiddle3: vld1q_f32([tw3.re, tw3.im, tw3.re, tw3.im].as_ptr()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, v: [NeonStoreF; 16]) -> [NeonStoreF; 16] {
        unsafe {
            let evens = self.bf8.exec(
                v[0].v,
                v[2].v,
                v[4].v,
                v[6].v,
                v[8].v,
                v[10].v,
                v[12].v,
                v[14].v,
                self.rotate,
            );

            let mut odds_1 =
                NeonButterfly::butterfly4_f32(v[1].v, v[5].v, v[9].v, v[13].v, self.rotate);
            let mut odds_2 =
                NeonButterfly::butterfly4_f32(v[15].v, v[3].v, v[7].v, v[11].v, self.rotate);

            odds_1.1 = vfcmulq_f32(odds_1.1, self.twiddle1);
            odds_2.1 = vfcmulq_conj_b_f32(odds_2.1, self.twiddle1);

            odds_1.2 = vfcmulq_f32(odds_1.2, self.twiddle2);
            odds_2.2 = vfcmulq_conj_b_f32(odds_2.2, self.twiddle2);

            odds_1.3 = vfcmulq_f32(odds_1.3, self.twiddle3);
            odds_2.3 = vfcmulq_conj_b_f32(odds_2.3, self.twiddle3);

            // step 4: cross FFTs
            let (o01, o02) = NeonButterfly::butterfly2_f32(odds_1.0, odds_2.0);
            odds_1.0 = o01;
            odds_2.0 = o02;

            let (o03, o04) = NeonButterfly::butterfly2_f32(odds_1.1, odds_2.1);
            odds_1.1 = o03;
            odds_2.1 = o04;
            let (o05, o06) = NeonButterfly::butterfly2_f32(odds_1.2, odds_2.2);
            odds_1.2 = o05;
            odds_2.2 = o06;
            let (o07, o08) = NeonButterfly::butterfly2_f32(odds_1.3, odds_2.3);
            odds_1.3 = o07;
            odds_2.3 = o08;

            // apply the butterfly 4 twiddle factor, which is just a rotation
            odds_2.0 = v_rotate90_f32(odds_2.0, self.rotate);
            odds_2.1 = v_rotate90_f32(odds_2.1, self.rotate);
            odds_2.2 = v_rotate90_f32(odds_2.2, self.rotate);
            odds_2.3 = v_rotate90_f32(odds_2.3, self.rotate);

            [
                NeonStoreF::raw(vaddq_f32(evens.0, odds_1.0)),
                NeonStoreF::raw(vaddq_f32(evens.1, odds_1.1)),
                NeonStoreF::raw(vaddq_f32(evens.2, odds_1.2)),
                NeonStoreF::raw(vaddq_f32(evens.3, odds_1.3)),
                NeonStoreF::raw(vaddq_f32(evens.4, odds_2.0)),
                NeonStoreF::raw(vaddq_f32(evens.5, odds_2.1)),
                NeonStoreF::raw(vaddq_f32(evens.6, odds_2.2)),
                NeonStoreF::raw(vaddq_f32(evens.7, odds_2.3)),
                NeonStoreF::raw(vsubq_f32(evens.0, odds_1.0)),
                NeonStoreF::raw(vsubq_f32(evens.1, odds_1.1)),
                NeonStoreF::raw(vsubq_f32(evens.2, odds_1.2)),
                NeonStoreF::raw(vsubq_f32(evens.3, odds_1.3)),
                NeonStoreF::raw(vsubq_f32(evens.4, odds_2.0)),
                NeonStoreF::raw(vsubq_f32(evens.5, odds_2.1)),
                NeonStoreF::raw(vsubq_f32(evens.6, odds_2.2)),
                NeonStoreF::raw(vsubq_f32(evens.7, odds_2.3)),
            ]
        }
    }

    #[inline(always)]
    pub(crate) fn exech(&self, v: [NeonStoreFh; 16]) -> [NeonStoreFh; 16] {
        unsafe {
            let evens = self.bf8.exech(
                v[0].v,
                v[2].v,
                v[4].v,
                v[6].v,
                v[8].v,
                v[10].v,
                v[12].v,
                v[14].v,
                vget_low_f32(self.rotate),
            );

            let mut odds_1 = NeonButterfly::butterfly4h_f32(
                v[1].v,
                v[5].v,
                v[9].v,
                v[13].v,
                vget_low_f32(self.rotate),
            );
            let mut odds_2 = NeonButterfly::butterfly4h_f32(
                v[15].v,
                v[3].v,
                v[7].v,
                v[11].v,
                vget_low_f32(self.rotate),
            );

            odds_1.1 = vfcmul_f32(odds_1.1, vget_low_f32(self.twiddle1));
            odds_2.1 = vfcmul_conj_b_f32(odds_2.1, vget_low_f32(self.twiddle1));

            odds_1.2 = vfcmul_f32(odds_1.2, vget_low_f32(self.twiddle2));
            odds_2.2 = vfcmul_conj_b_f32(odds_2.2, vget_low_f32(self.twiddle2));

            odds_1.3 = vfcmul_f32(odds_1.3, vget_low_f32(self.twiddle3));
            odds_2.3 = vfcmul_conj_b_f32(odds_2.3, vget_low_f32(self.twiddle3));

            // step 4: cross FFTs
            let (o01, o02) = NeonButterfly::butterfly2h_f32(odds_1.0, odds_2.0);
            odds_1.0 = o01;
            odds_2.0 = o02;

            let (o03, o04) = NeonButterfly::butterfly2h_f32(odds_1.1, odds_2.1);
            odds_1.1 = o03;
            odds_2.1 = o04;
            let (o05, o06) = NeonButterfly::butterfly2h_f32(odds_1.2, odds_2.2);
            odds_1.2 = o05;
            odds_2.2 = o06;
            let (o07, o08) = NeonButterfly::butterfly2h_f32(odds_1.3, odds_2.3);
            odds_1.3 = o07;
            odds_2.3 = o08;

            // apply the butterfly 4 twiddle factor, which is just a rotation
            odds_2.0 = vh_rotate90_f32(odds_2.0, vget_low_f32(self.rotate));
            odds_2.1 = vh_rotate90_f32(odds_2.1, vget_low_f32(self.rotate));
            odds_2.2 = vh_rotate90_f32(odds_2.2, vget_low_f32(self.rotate));
            odds_2.3 = vh_rotate90_f32(odds_2.3, vget_low_f32(self.rotate));

            [
                NeonStoreFh::raw(vadd_f32(evens.0, odds_1.0)),
                NeonStoreFh::raw(vadd_f32(evens.1, odds_1.1)),
                NeonStoreFh::raw(vadd_f32(evens.2, odds_1.2)),
                NeonStoreFh::raw(vadd_f32(evens.3, odds_1.3)),
                NeonStoreFh::raw(vadd_f32(evens.4, odds_2.0)),
                NeonStoreFh::raw(vadd_f32(evens.5, odds_2.1)),
                NeonStoreFh::raw(vadd_f32(evens.6, odds_2.2)),
                NeonStoreFh::raw(vadd_f32(evens.7, odds_2.3)),
                NeonStoreFh::raw(vsub_f32(evens.0, odds_1.0)),
                NeonStoreFh::raw(vsub_f32(evens.1, odds_1.1)),
                NeonStoreFh::raw(vsub_f32(evens.2, odds_1.2)),
                NeonStoreFh::raw(vsub_f32(evens.3, odds_1.3)),
                NeonStoreFh::raw(vsub_f32(evens.4, odds_2.0)),
                NeonStoreFh::raw(vsub_f32(evens.5, odds_2.1)),
                NeonStoreFh::raw(vsub_f32(evens.6, odds_2.2)),
                NeonStoreFh::raw(vsub_f32(evens.7, odds_2.3)),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly16f {
    bf8: NeonFastButterfly8<f32>,
    twiddle1: float32x4_t,
    twiddle2: float32x4_t,
    twiddle3: float32x4_t,
    pub(super) bf4: FastFcmaBf4f,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly16f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let tw1 = compute_twiddle(1, 16, fft_direction);
            let tw2 = compute_twiddle(2, 16, fft_direction);
            let tw3 = compute_twiddle(3, 16, fft_direction);
            Self {
                bf8: NeonFastButterfly8::new(fft_direction),
                twiddle1: vld1q_f32([tw1.re, tw1.im, tw1.re, tw1.im].as_ptr()),
                twiddle2: vld1q_f32([tw2.re, tw2.im, tw2.re, tw2.im].as_ptr()),
                twiddle3: vld1q_f32([tw3.re, tw3.im, tw3.re, tw3.im].as_ptr()),
                bf4: FastFcmaBf4f::new(fft_direction),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, v: [NeonStoreF; 16]) -> [NeonStoreF; 16] {
        let evens = self.bf8.exec_fcma(
            v[0].v, v[2].v, v[4].v, v[6].v, v[8].v, v[10].v, v[12].v, v[14].v, &self.bf4,
        );

        let mut odds_1 = self.bf4.exec(v[1].v, v[5].v, v[9].v, v[13].v);
        let mut odds_2 = self.bf4.exec(v[15].v, v[3].v, v[7].v, v[11].v);

        odds_1.1 = vfcmulq_fcma_f32(odds_1.1, self.twiddle1);
        odds_2.1 = vfcmulq_b_conj_fcma_f32(odds_2.1, self.twiddle1);

        odds_1.2 = vfcmulq_fcma_f32(odds_1.2, self.twiddle2);
        odds_2.2 = vfcmulq_b_conj_fcma_f32(odds_2.2, self.twiddle2);

        odds_1.3 = vfcmulq_fcma_f32(odds_1.3, self.twiddle3);
        odds_2.3 = vfcmulq_b_conj_fcma_f32(odds_2.3, self.twiddle3);

        // step 4: cross FFTs
        let (o01, o02) = NeonButterfly::butterfly2_f32(odds_1.0, odds_2.0);
        odds_1.0 = o01;
        odds_2.0 = o02;

        let (o03, o04) = NeonButterfly::butterfly2_f32(odds_1.1, odds_2.1);
        odds_1.1 = o03;
        odds_2.1 = o04;
        let (o05, o06) = NeonButterfly::butterfly2_f32(odds_1.2, odds_2.2);
        odds_1.2 = o05;
        odds_2.2 = o06;
        let (o07, o08) = NeonButterfly::butterfly2_f32(odds_1.3, odds_2.3);
        odds_1.3 = o07;
        odds_2.3 = o08;

        // apply the butterfly 4 twiddle factor, which is just a rotation
        odds_2.0 = vcmlaq_rot90_f32(vdupq_n_f32(0.), self.bf4.rot_sign, odds_2.0);
        odds_2.1 = vcmlaq_rot90_f32(vdupq_n_f32(0.), self.bf4.rot_sign, odds_2.1);
        odds_2.2 = vcmlaq_rot90_f32(vdupq_n_f32(0.), self.bf4.rot_sign, odds_2.2);
        odds_2.3 = vcmlaq_rot90_f32(vdupq_n_f32(0.), self.bf4.rot_sign, odds_2.3);

        [
            NeonStoreF::raw(vaddq_f32(evens.0, odds_1.0)),
            NeonStoreF::raw(vaddq_f32(evens.1, odds_1.1)),
            NeonStoreF::raw(vaddq_f32(evens.2, odds_1.2)),
            NeonStoreF::raw(vaddq_f32(evens.3, odds_1.3)),
            NeonStoreF::raw(vaddq_f32(evens.4, odds_2.0)),
            NeonStoreF::raw(vaddq_f32(evens.5, odds_2.1)),
            NeonStoreF::raw(vaddq_f32(evens.6, odds_2.2)),
            NeonStoreF::raw(vaddq_f32(evens.7, odds_2.3)),
            NeonStoreF::raw(vsubq_f32(evens.0, odds_1.0)),
            NeonStoreF::raw(vsubq_f32(evens.1, odds_1.1)),
            NeonStoreF::raw(vsubq_f32(evens.2, odds_1.2)),
            NeonStoreF::raw(vsubq_f32(evens.3, odds_1.3)),
            NeonStoreF::raw(vsubq_f32(evens.4, odds_2.0)),
            NeonStoreF::raw(vsubq_f32(evens.5, odds_2.1)),
            NeonStoreF::raw(vsubq_f32(evens.6, odds_2.2)),
            NeonStoreF::raw(vsubq_f32(evens.7, odds_2.3)),
        ]
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, v: [NeonStoreFh; 16]) -> [NeonStoreFh; 16] {
        let evens = self.bf8.exech_fcma(
            v[0].v, v[2].v, v[4].v, v[6].v, v[8].v, v[10].v, v[12].v, v[14].v, &self.bf4,
        );

        let mut odds_1 = self.bf4.exech(v[1].v, v[5].v, v[9].v, v[13].v);
        let mut odds_2 = self.bf4.exech(v[15].v, v[3].v, v[7].v, v[11].v);

        odds_1.1 = vfcmul_fcma_f32(odds_1.1, vget_low_f32(self.twiddle1));
        odds_2.1 = vfcmul_b_conj_fcma_f32(odds_2.1, vget_low_f32(self.twiddle1));

        odds_1.2 = vfcmul_fcma_f32(odds_1.2, vget_low_f32(self.twiddle2));
        odds_2.2 = vfcmul_b_conj_fcma_f32(odds_2.2, vget_low_f32(self.twiddle2));

        odds_1.3 = vfcmul_fcma_f32(odds_1.3, vget_low_f32(self.twiddle3));
        odds_2.3 = vfcmul_b_conj_fcma_f32(odds_2.3, vget_low_f32(self.twiddle3));

        // step 4: cross FFTs
        let (o01, o02) = NeonButterfly::butterfly2h_f32(odds_1.0, odds_2.0);
        odds_1.0 = o01;
        odds_2.0 = o02;

        let (o03, o04) = NeonButterfly::butterfly2h_f32(odds_1.1, odds_2.1);
        odds_1.1 = o03;
        odds_2.1 = o04;
        let (o05, o06) = NeonButterfly::butterfly2h_f32(odds_1.2, odds_2.2);
        odds_1.2 = o05;
        odds_2.2 = o06;
        let (o07, o08) = NeonButterfly::butterfly2h_f32(odds_1.3, odds_2.3);
        odds_1.3 = o07;
        odds_2.3 = o08;

        // apply the butterfly 4 twiddle factor, which is just a rotation
        odds_2.0 = vcmla_rot90_f32(vdup_n_f32(0.), vget_low_f32(self.bf4.rot_sign), odds_2.0);
        odds_2.1 = vcmla_rot90_f32(vdup_n_f32(0.), vget_low_f32(self.bf4.rot_sign), odds_2.1);
        odds_2.2 = vcmla_rot90_f32(vdup_n_f32(0.), vget_low_f32(self.bf4.rot_sign), odds_2.2);
        odds_2.3 = vcmla_rot90_f32(vdup_n_f32(0.), vget_low_f32(self.bf4.rot_sign), odds_2.3);

        [
            NeonStoreFh::raw(vadd_f32(evens.0, odds_1.0)),
            NeonStoreFh::raw(vadd_f32(evens.1, odds_1.1)),
            NeonStoreFh::raw(vadd_f32(evens.2, odds_1.2)),
            NeonStoreFh::raw(vadd_f32(evens.3, odds_1.3)),
            NeonStoreFh::raw(vadd_f32(evens.4, odds_2.0)),
            NeonStoreFh::raw(vadd_f32(evens.5, odds_2.1)),
            NeonStoreFh::raw(vadd_f32(evens.6, odds_2.2)),
            NeonStoreFh::raw(vadd_f32(evens.7, odds_2.3)),
            NeonStoreFh::raw(vsub_f32(evens.0, odds_1.0)),
            NeonStoreFh::raw(vsub_f32(evens.1, odds_1.1)),
            NeonStoreFh::raw(vsub_f32(evens.2, odds_1.2)),
            NeonStoreFh::raw(vsub_f32(evens.3, odds_1.3)),
            NeonStoreFh::raw(vsub_f32(evens.4, odds_2.0)),
            NeonStoreFh::raw(vsub_f32(evens.5, odds_2.1)),
            NeonStoreFh::raw(vsub_f32(evens.6, odds_2.2)),
            NeonStoreFh::raw(vsub_f32(evens.7, odds_2.3)),
        ]
    }
}
