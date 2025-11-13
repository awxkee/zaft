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
use crate::neon::butterflies::{NeonButterfly, NeonFastButterfly8};
use crate::neon::util::*;
use crate::util::compute_twiddle;
use std::arch::aarch64::*;

pub(crate) struct NeonFastButterfly16d {
    pub(crate) rot: float64x2_t,
    twiddle1: float64x2_t,
    twiddle2: float64x2_t,
    twiddle3: float64x2_t,
    pub(crate) bf8: NeonFastButterfly8<f64>,
}

impl NeonFastButterfly16d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle::<f64>(1, 16, fft_direction);
        let tw2 = compute_twiddle::<f64>(2, 16, fft_direction);
        let tw3 = compute_twiddle::<f64>(3, 16, fft_direction);
        unsafe {
            Self {
                twiddle1: vdup_complex_f64(tw1),
                twiddle2: vdup_complex_f64(tw2),
                twiddle3: vdup_complex_f64(tw3),
                rot: vld1q_f64(match fft_direction {
                    FftDirection::Inverse => [-0.0, 0.0].as_ptr(),
                    FftDirection::Forward => [0.0, -0.0].as_ptr(),
                }),
                bf8: NeonFastButterfly8::new(fft_direction),
            }
        }
    }
}

impl NeonFastButterfly16d {
    #[inline(always)]
    pub(crate) fn exec(
        &self,
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        u3: float64x2_t,
        u4: float64x2_t,
        u5: float64x2_t,
        u6: float64x2_t,
        u7: float64x2_t,
        u8: float64x2_t,
        u9: float64x2_t,
        u10: float64x2_t,
        u11: float64x2_t,
        u12: float64x2_t,
        u13: float64x2_t,
        u14: float64x2_t,
        u15: float64x2_t,
    ) -> (
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
    ) {
        unsafe {
            let evens = self.bf8.exec(u0, u2, u4, u6, u8, u10, u12, u14, self.rot);

            let mut odds_1 = NeonButterfly::butterfly4_f64(u1, u5, u9, u13, self.rot);
            let mut odds_2 = NeonButterfly::butterfly4_f64(u15, u3, u7, u11, self.rot);

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
            odds_2.0 = v_rotate90_f64(odds_2.0, self.rot);
            odds_2.1 = v_rotate90_f64(odds_2.1, self.rot);
            odds_2.2 = v_rotate90_f64(odds_2.2, self.rot);
            odds_2.3 = v_rotate90_f64(odds_2.3, self.rot);

            (
                vaddq_f64(evens.0, odds_1.0),
                vaddq_f64(evens.1, odds_1.1),
                vaddq_f64(evens.2, odds_1.2),
                vaddq_f64(evens.3, odds_1.3),
                vaddq_f64(evens.4, odds_2.0),
                vaddq_f64(evens.5, odds_2.1),
                vaddq_f64(evens.6, odds_2.2),
                vaddq_f64(evens.7, odds_2.3),
                vsubq_f64(evens.0, odds_1.0),
                vsubq_f64(evens.1, odds_1.1),
                vsubq_f64(evens.2, odds_1.2),
                vsubq_f64(evens.3, odds_1.3),
                vsubq_f64(evens.4, odds_2.0),
                vsubq_f64(evens.5, odds_2.1),
                vsubq_f64(evens.6, odds_2.2),
                vsubq_f64(evens.7, odds_2.3),
            )
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn forward(
        &self,
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        u3: float64x2_t,
        u4: float64x2_t,
        u5: float64x2_t,
        u6: float64x2_t,
        u7: float64x2_t,
        u8: float64x2_t,
        u9: float64x2_t,
        u10: float64x2_t,
        u11: float64x2_t,
        u12: float64x2_t,
        u13: float64x2_t,
        u14: float64x2_t,
        u15: float64x2_t,
    ) -> (
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
    ) {
        let evens = self.bf8.forward(u0, u2, u4, u6, u8, u10, u12, u14);

        let mut odds_1 = NeonButterfly::bf4_f64_forward(u1, u5, u9, u13);
        let mut odds_2 = NeonButterfly::bf4_f64_forward(u15, u3, u7, u11);

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
        odds_2.0 = vcaddq_rot270_f64(vdupq_n_f64(0.), odds_2.0);
        odds_2.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), odds_2.1);
        odds_2.2 = vcaddq_rot270_f64(vdupq_n_f64(0.), odds_2.2);
        odds_2.3 = vcaddq_rot270_f64(vdupq_n_f64(0.), odds_2.3);

        (
            vaddq_f64(evens.0, odds_1.0),
            vaddq_f64(evens.1, odds_1.1),
            vaddq_f64(evens.2, odds_1.2),
            vaddq_f64(evens.3, odds_1.3),
            vaddq_f64(evens.4, odds_2.0),
            vaddq_f64(evens.5, odds_2.1),
            vaddq_f64(evens.6, odds_2.2),
            vaddq_f64(evens.7, odds_2.3),
            vsubq_f64(evens.0, odds_1.0),
            vsubq_f64(evens.1, odds_1.1),
            vsubq_f64(evens.2, odds_1.2),
            vsubq_f64(evens.3, odds_1.3),
            vsubq_f64(evens.4, odds_2.0),
            vsubq_f64(evens.5, odds_2.1),
            vsubq_f64(evens.6, odds_2.2),
            vsubq_f64(evens.7, odds_2.3),
        )
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn backward(
        &self,
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        u3: float64x2_t,
        u4: float64x2_t,
        u5: float64x2_t,
        u6: float64x2_t,
        u7: float64x2_t,
        u8: float64x2_t,
        u9: float64x2_t,
        u10: float64x2_t,
        u11: float64x2_t,
        u12: float64x2_t,
        u13: float64x2_t,
        u14: float64x2_t,
        u15: float64x2_t,
    ) -> (
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
        float64x2_t,
    ) {
        let evens = self.bf8.backward(u0, u2, u4, u6, u8, u10, u12, u14);

        let mut odds_1 = NeonButterfly::bf4_f64_backward(u1, u5, u9, u13);
        let mut odds_2 = NeonButterfly::bf4_f64_backward(u15, u3, u7, u11);

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
        odds_2.0 = vcaddq_rot90_f64(vdupq_n_f64(0.), odds_2.0);
        odds_2.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), odds_2.1);
        odds_2.2 = vcaddq_rot90_f64(vdupq_n_f64(0.), odds_2.2);
        odds_2.3 = vcaddq_rot90_f64(vdupq_n_f64(0.), odds_2.3);

        (
            vaddq_f64(evens.0, odds_1.0),
            vaddq_f64(evens.1, odds_1.1),
            vaddq_f64(evens.2, odds_1.2),
            vaddq_f64(evens.3, odds_1.3),
            vaddq_f64(evens.4, odds_2.0),
            vaddq_f64(evens.5, odds_2.1),
            vaddq_f64(evens.6, odds_2.2),
            vaddq_f64(evens.7, odds_2.3),
            vsubq_f64(evens.0, odds_1.0),
            vsubq_f64(evens.1, odds_1.1),
            vsubq_f64(evens.2, odds_1.2),
            vsubq_f64(evens.3, odds_1.3),
            vsubq_f64(evens.4, odds_2.0),
            vsubq_f64(evens.5, odds_2.1),
            vsubq_f64(evens.6, odds_2.2),
            vsubq_f64(evens.7, odds_2.3),
        )
    }
}
