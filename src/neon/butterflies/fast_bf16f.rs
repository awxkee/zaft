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

pub(crate) struct NeonFastButterfly16f {
    pub(crate) rot: float32x4_t,
    tw1tw2: float32x4_t,
    twiddle3: float32x4_t,
    pub(crate) bf8: NeonFastButterfly8<f32>,
}

impl NeonFastButterfly16f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle::<f32>(1, 16, fft_direction);
        let tw2 = compute_twiddle::<f32>(2, 16, fft_direction);
        let tw3 = compute_twiddle::<f32>(3, 16, fft_direction);
        unsafe {
            Self {
                tw1tw2: vcombine_f32(vdup_complex_f32(tw1), vdup_complex_f32(tw2)),
                twiddle3: vdupq_complex_f32(tw3),
                rot: vld1q_f32(match fft_direction {
                    FftDirection::Inverse => [-0.0, 0.0, -0.0, 0.0].as_ptr(),
                    FftDirection::Forward => [0.0, -0.0, 0.0, -0.0].as_ptr(),
                }),
                bf8: NeonFastButterfly8::new(fft_direction),
            }
        }
    }
}

impl NeonFastButterfly16f {
    #[inline(always)]
    pub(crate) fn exec(
        &self,
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
        u3: float32x2_t,
        u4: float32x2_t,
        u5: float32x2_t,
        u6: float32x2_t,
        u7: float32x2_t,
        u8: float32x2_t,
        u9: float32x2_t,
        u10: float32x2_t,
        u11: float32x2_t,
        u12: float32x2_t,
        u13: float32x2_t,
        u14: float32x2_t,
        u15: float32x2_t,
    ) -> (
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
    ) {
        unsafe {
            let evens = self
                .bf8
                .exech(u0, u2, u4, u6, u8, u10, u12, u14, vget_low_f32(self.rot));

            let mut odds_1 =
                NeonButterfly::butterfly4h_f32(u1, u5, u9, u13, vget_low_f32(self.rot));
            let mut odds_2 =
                NeonButterfly::butterfly4h_f32(u15, u3, u7, u11, vget_low_f32(self.rot));

            let o1_1_o1_2 = vfcmulq_f32(vcombine_f32(odds_1.1, odds_1.2), self.tw1tw2);
            let o2_1_o2_2 = vfcmulq_conj_b_f32(vcombine_f32(odds_2.1, odds_2.2), self.tw1tw2);

            odds_1.1 = vget_low_f32(o1_1_o1_2);
            odds_1.2 = vget_high_f32(o1_1_o1_2);

            odds_2.1 = vget_low_f32(o2_1_o2_2);
            odds_2.2 = vget_high_f32(o2_1_o2_2);

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
            odds_2.0 = vh_rotate90_f32(odds_2.0, vget_low_f32(self.rot));
            odds_2.1 = vh_rotate90_f32(odds_2.1, vget_low_f32(self.rot));
            odds_2.2 = vh_rotate90_f32(odds_2.2, vget_low_f32(self.rot));
            odds_2.3 = vh_rotate90_f32(odds_2.3, vget_low_f32(self.rot));

            (
                vadd_f32(evens.0, odds_1.0),
                vadd_f32(evens.1, odds_1.1),
                vadd_f32(evens.2, odds_1.2),
                vadd_f32(evens.3, odds_1.3),
                vadd_f32(evens.4, odds_2.0),
                vadd_f32(evens.5, odds_2.1),
                vadd_f32(evens.6, odds_2.2),
                vadd_f32(evens.7, odds_2.3),
                vsub_f32(evens.0, odds_1.0),
                vsub_f32(evens.1, odds_1.1),
                vsub_f32(evens.2, odds_1.2),
                vsub_f32(evens.3, odds_1.3),
                vsub_f32(evens.4, odds_2.0),
                vsub_f32(evens.5, odds_2.1),
                vsub_f32(evens.6, odds_2.2),
                vsub_f32(evens.7, odds_2.3),
            )
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn forwardh(
        &self,
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
        u3: float32x2_t,
        u4: float32x2_t,
        u5: float32x2_t,
        u6: float32x2_t,
        u7: float32x2_t,
        u8: float32x2_t,
        u9: float32x2_t,
        u10: float32x2_t,
        u11: float32x2_t,
        u12: float32x2_t,
        u13: float32x2_t,
        u14: float32x2_t,
        u15: float32x2_t,
    ) -> (
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
    ) {
        unsafe {
            let evens = self.bf8.forwardh(u0, u2, u4, u6, u8, u10, u12, u14);

            let mut odds_1 = NeonButterfly::bf4h_forward_f32(u1, u5, u9, u13);
            let mut odds_2 = NeonButterfly::bf4h_forward_f32(u15, u3, u7, u11);

            let o1_1_o1_2 = vfcmulq_fcma_f32(vcombine_f32(odds_1.1, odds_1.2), self.tw1tw2);
            let o2_1_o2_2 = vfcmulq_b_conj_fcma_f32(vcombine_f32(odds_2.1, odds_2.2), self.tw1tw2);

            odds_1.1 = vget_low_f32(o1_1_o1_2);
            odds_1.2 = vget_high_f32(o1_1_o1_2);

            odds_2.1 = vget_low_f32(o2_1_o2_2);
            odds_2.2 = vget_high_f32(o2_1_o2_2);

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
            odds_2.0 = vcadd_rot270_f32(vdup_n_f32(0.), odds_2.0);
            odds_2.1 = vcadd_rot270_f32(vdup_n_f32(0.), odds_2.1);
            odds_2.2 = vcadd_rot270_f32(vdup_n_f32(0.), odds_2.2);
            odds_2.3 = vcadd_rot270_f32(vdup_n_f32(0.), odds_2.3);

            (
                vadd_f32(evens.0, odds_1.0),
                vadd_f32(evens.1, odds_1.1),
                vadd_f32(evens.2, odds_1.2),
                vadd_f32(evens.3, odds_1.3),
                vadd_f32(evens.4, odds_2.0),
                vadd_f32(evens.5, odds_2.1),
                vadd_f32(evens.6, odds_2.2),
                vadd_f32(evens.7, odds_2.3),
                vsub_f32(evens.0, odds_1.0),
                vsub_f32(evens.1, odds_1.1),
                vsub_f32(evens.2, odds_1.2),
                vsub_f32(evens.3, odds_1.3),
                vsub_f32(evens.4, odds_2.0),
                vsub_f32(evens.5, odds_2.1),
                vsub_f32(evens.6, odds_2.2),
                vsub_f32(evens.7, odds_2.3),
            )
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn backwardh(
        &self,
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
        u3: float32x2_t,
        u4: float32x2_t,
        u5: float32x2_t,
        u6: float32x2_t,
        u7: float32x2_t,
        u8: float32x2_t,
        u9: float32x2_t,
        u10: float32x2_t,
        u11: float32x2_t,
        u12: float32x2_t,
        u13: float32x2_t,
        u14: float32x2_t,
        u15: float32x2_t,
    ) -> (
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
        float32x2_t,
    ) {
        unsafe {
            let evens = self.bf8.backwardh(u0, u2, u4, u6, u8, u10, u12, u14);

            let mut odds_1 = NeonButterfly::bf4h_backward_f32(u1, u5, u9, u13);
            let mut odds_2 = NeonButterfly::bf4h_backward_f32(u15, u3, u7, u11);

            let o1_1_o1_2 = vfcmulq_fcma_f32(vcombine_f32(odds_1.1, odds_1.2), self.tw1tw2);
            let o2_1_o2_2 = vfcmulq_b_conj_fcma_f32(vcombine_f32(odds_2.1, odds_2.2), self.tw1tw2);

            odds_1.1 = vget_low_f32(o1_1_o1_2);
            odds_1.2 = vget_high_f32(o1_1_o1_2);

            odds_2.1 = vget_low_f32(o2_1_o2_2);
            odds_2.2 = vget_high_f32(o2_1_o2_2);

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
            odds_2.0 = vcadd_rot90_f32(vdup_n_f32(0.), odds_2.0);
            odds_2.1 = vcadd_rot90_f32(vdup_n_f32(0.), odds_2.1);
            odds_2.2 = vcadd_rot90_f32(vdup_n_f32(0.), odds_2.2);
            odds_2.3 = vcadd_rot90_f32(vdup_n_f32(0.), odds_2.3);

            (
                vadd_f32(evens.0, odds_1.0),
                vadd_f32(evens.1, odds_1.1),
                vadd_f32(evens.2, odds_1.2),
                vadd_f32(evens.3, odds_1.3),
                vadd_f32(evens.4, odds_2.0),
                vadd_f32(evens.5, odds_2.1),
                vadd_f32(evens.6, odds_2.2),
                vadd_f32(evens.7, odds_2.3),
                vsub_f32(evens.0, odds_1.0),
                vsub_f32(evens.1, odds_1.1),
                vsub_f32(evens.2, odds_1.2),
                vsub_f32(evens.3, odds_1.3),
                vsub_f32(evens.4, odds_2.0),
                vsub_f32(evens.5, odds_2.1),
                vsub_f32(evens.6, odds_2.2),
                vsub_f32(evens.7, odds_2.3),
            )
        }
    }
}
