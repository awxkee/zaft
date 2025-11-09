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
use crate::neon::butterflies::NeonButterfly;
use crate::neon::butterflies::fast_bf16d::NeonFastButterfly16d;
use crate::neon::butterflies::fast_bf16f::NeonFastButterfly16f;
use crate::neon::util::{
    vdup_complex_f32, vdup_complex_f64, vfcmulq_conj_b_fcma_f64, vfcmulq_fcma_f32, vfcmulq_fcma_f64,
};
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaButterfly32d {
    direction: FftDirection,
    twiddle1: float64x2_t,
    twiddle2: float64x2_t,
    twiddle3: float64x2_t,
    twiddle4: float64x2_t,
    twiddle5: float64x2_t,
    twiddle6: float64x2_t,
    twiddle7: float64x2_t,
    bf16: NeonFastButterfly16d,
}

impl NeonFcmaButterfly32d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                direction: fft_direction,
                twiddle1: vdup_complex_f64(compute_twiddle(1, 32, fft_direction)),
                twiddle2: vdup_complex_f64(compute_twiddle(2, 32, fft_direction)),
                twiddle3: vdup_complex_f64(compute_twiddle(3, 32, fft_direction)),
                twiddle4: vdup_complex_f64(compute_twiddle(4, 32, fft_direction)),
                twiddle5: vdup_complex_f64(compute_twiddle(5, 32, fft_direction)),
                twiddle6: vdup_complex_f64(compute_twiddle(6, 32, fft_direction)),
                twiddle7: vdup_complex_f64(compute_twiddle(7, 32, fft_direction)),
                bf16: NeonFastButterfly16d::new(fft_direction),
            }
        }
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly32d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            match self.direction {
                FftDirection::Forward => self.execute_forward_f64(in_place),
                FftDirection::Inverse => self.execute_backward_f64(in_place),
            }
        }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        32
    }
}

impl NeonFcmaButterfly32d {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_forward_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(32) {
                let u0 = vld1q_f64(chunk.as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                let u6 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                let u7 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());
                let u8 = vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast());
                let u9 = vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast());
                let u10 = vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast());
                let u11 = vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast());
                let u12 = vld1q_f64(chunk.get_unchecked(12..).as_ptr().cast());
                let u13 = vld1q_f64(chunk.get_unchecked(13..).as_ptr().cast());
                let u14 = vld1q_f64(chunk.get_unchecked(14..).as_ptr().cast());
                let u15 = vld1q_f64(chunk.get_unchecked(15..).as_ptr().cast());
                let u16 = vld1q_f64(chunk.get_unchecked(16..).as_ptr().cast());
                let u17 = vld1q_f64(chunk.get_unchecked(17..).as_ptr().cast());
                let u18 = vld1q_f64(chunk.get_unchecked(18..).as_ptr().cast());
                let u19 = vld1q_f64(chunk.get_unchecked(19..).as_ptr().cast());
                let u20 = vld1q_f64(chunk.get_unchecked(20..).as_ptr().cast());
                let u21 = vld1q_f64(chunk.get_unchecked(21..).as_ptr().cast());
                let u22 = vld1q_f64(chunk.get_unchecked(22..).as_ptr().cast());
                let u23 = vld1q_f64(chunk.get_unchecked(23..).as_ptr().cast());
                let u24 = vld1q_f64(chunk.get_unchecked(24..).as_ptr().cast());
                let u25 = vld1q_f64(chunk.get_unchecked(25..).as_ptr().cast());
                let u26 = vld1q_f64(chunk.get_unchecked(26..).as_ptr().cast());
                let u27 = vld1q_f64(chunk.get_unchecked(27..).as_ptr().cast());
                let u28 = vld1q_f64(chunk.get_unchecked(28..).as_ptr().cast());
                let u29 = vld1q_f64(chunk.get_unchecked(29..).as_ptr().cast());
                let u30 = vld1q_f64(chunk.get_unchecked(30..).as_ptr().cast());
                let u31 = vld1q_f64(chunk.get_unchecked(31..).as_ptr().cast());

                let s_evens = self.bf16.forward(
                    u0, u2, u4, u6, u8, u10, u12, u14, u16, u18, u20, u22, u24, u26, u28, u30,
                );
                let mut odds1 = self.bf16.bf8.forward(u1, u5, u9, u13, u17, u21, u25, u29);
                let mut odds2 = self.bf16.bf8.forward(u31, u3, u7, u11, u15, u19, u23, u27);

                odds1.1 = vfcmulq_fcma_f64(odds1.1, self.twiddle1);
                odds2.1 = vfcmulq_conj_b_fcma_f64(odds2.1, self.twiddle1);

                odds1.2 = vfcmulq_fcma_f64(odds1.2, self.twiddle2);
                odds2.2 = vfcmulq_conj_b_fcma_f64(odds2.2, self.twiddle2);

                odds1.3 = vfcmulq_fcma_f64(odds1.3, self.twiddle3);
                odds2.3 = vfcmulq_conj_b_fcma_f64(odds2.3, self.twiddle3);

                odds1.4 = vfcmulq_fcma_f64(odds1.4, self.twiddle4);
                odds2.4 = vfcmulq_conj_b_fcma_f64(odds2.4, self.twiddle4);

                odds1.5 = vfcmulq_fcma_f64(odds1.5, self.twiddle5);
                odds2.5 = vfcmulq_conj_b_fcma_f64(odds2.5, self.twiddle5);

                odds1.6 = vfcmulq_fcma_f64(odds1.6, self.twiddle6);
                odds2.6 = vfcmulq_conj_b_fcma_f64(odds2.6, self.twiddle6);

                odds1.7 = vfcmulq_fcma_f64(odds1.7, self.twiddle7);
                odds2.7 = vfcmulq_conj_b_fcma_f64(odds2.7, self.twiddle7);

                let mut q0 = NeonButterfly::butterfly2_f64(odds1.0, odds2.0);
                let mut q1 = NeonButterfly::butterfly2_f64(odds1.1, odds2.1);
                let mut q2 = NeonButterfly::butterfly2_f64(odds1.2, odds2.2);
                let mut q3 = NeonButterfly::butterfly2_f64(odds1.3, odds2.3);
                let mut q4 = NeonButterfly::butterfly2_f64(odds1.4, odds2.4);
                let mut q5 = NeonButterfly::butterfly2_f64(odds1.5, odds2.5);
                let mut q6 = NeonButterfly::butterfly2_f64(odds1.6, odds2.6);
                let mut q7 = NeonButterfly::butterfly2_f64(odds1.7, odds2.7);

                q0.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q0.1);
                q1.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q1.1);
                q2.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q2.1);
                q3.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q3.1);
                q4.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q4.1);
                q5.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q5.1);
                q6.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q6.1);
                q7.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q7.1);

                vst1q_f64(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.0, q0.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.1, q1.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.2, q2.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.3, q3.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.4, q4.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.5, q5.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.6, q6.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(7..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.7, q7.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.8, q0.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(9..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.9, q1.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.10, q2.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(11..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.11, q3.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.12, q4.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(13..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.13, q5.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.14, q6.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(15..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.15, q7.1),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.0, q0.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(17..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.1, q1.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.2, q2.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(19..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.3, q3.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.4, q4.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(21..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.5, q5.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.6, q6.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(23..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.7, q7.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.8, q0.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(25..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.9, q1.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.10, q2.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(27..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.11, q3.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.12, q4.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(29..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.13, q5.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(30..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.14, q6.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(31..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.15, q7.1),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_out_of_place_forward_f64(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
                let u0 = vld1q_f64(src.as_ptr().cast());
                let u1 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(src.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(src.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(src.get_unchecked(5..).as_ptr().cast());
                let u6 = vld1q_f64(src.get_unchecked(6..).as_ptr().cast());
                let u7 = vld1q_f64(src.get_unchecked(7..).as_ptr().cast());
                let u8 = vld1q_f64(src.get_unchecked(8..).as_ptr().cast());
                let u9 = vld1q_f64(src.get_unchecked(9..).as_ptr().cast());
                let u10 = vld1q_f64(src.get_unchecked(10..).as_ptr().cast());
                let u11 = vld1q_f64(src.get_unchecked(11..).as_ptr().cast());
                let u12 = vld1q_f64(src.get_unchecked(12..).as_ptr().cast());
                let u13 = vld1q_f64(src.get_unchecked(13..).as_ptr().cast());
                let u14 = vld1q_f64(src.get_unchecked(14..).as_ptr().cast());
                let u15 = vld1q_f64(src.get_unchecked(15..).as_ptr().cast());
                let u16 = vld1q_f64(src.get_unchecked(16..).as_ptr().cast());
                let u17 = vld1q_f64(src.get_unchecked(17..).as_ptr().cast());
                let u18 = vld1q_f64(src.get_unchecked(18..).as_ptr().cast());
                let u19 = vld1q_f64(src.get_unchecked(19..).as_ptr().cast());
                let u20 = vld1q_f64(src.get_unchecked(20..).as_ptr().cast());
                let u21 = vld1q_f64(src.get_unchecked(21..).as_ptr().cast());
                let u22 = vld1q_f64(src.get_unchecked(22..).as_ptr().cast());
                let u23 = vld1q_f64(src.get_unchecked(23..).as_ptr().cast());
                let u24 = vld1q_f64(src.get_unchecked(24..).as_ptr().cast());
                let u25 = vld1q_f64(src.get_unchecked(25..).as_ptr().cast());
                let u26 = vld1q_f64(src.get_unchecked(26..).as_ptr().cast());
                let u27 = vld1q_f64(src.get_unchecked(27..).as_ptr().cast());
                let u28 = vld1q_f64(src.get_unchecked(28..).as_ptr().cast());
                let u29 = vld1q_f64(src.get_unchecked(29..).as_ptr().cast());
                let u30 = vld1q_f64(src.get_unchecked(30..).as_ptr().cast());
                let u31 = vld1q_f64(src.get_unchecked(31..).as_ptr().cast());

                let s_evens = self.bf16.forward(
                    u0, u2, u4, u6, u8, u10, u12, u14, u16, u18, u20, u22, u24, u26, u28, u30,
                );
                let mut odds1 = self.bf16.bf8.forward(u1, u5, u9, u13, u17, u21, u25, u29);
                let mut odds2 = self.bf16.bf8.forward(u31, u3, u7, u11, u15, u19, u23, u27);

                odds1.1 = vfcmulq_fcma_f64(odds1.1, self.twiddle1);
                odds2.1 = vfcmulq_conj_b_fcma_f64(odds2.1, self.twiddle1);

                odds1.2 = vfcmulq_fcma_f64(odds1.2, self.twiddle2);
                odds2.2 = vfcmulq_conj_b_fcma_f64(odds2.2, self.twiddle2);

                odds1.3 = vfcmulq_fcma_f64(odds1.3, self.twiddle3);
                odds2.3 = vfcmulq_conj_b_fcma_f64(odds2.3, self.twiddle3);

                odds1.4 = vfcmulq_fcma_f64(odds1.4, self.twiddle4);
                odds2.4 = vfcmulq_conj_b_fcma_f64(odds2.4, self.twiddle4);

                odds1.5 = vfcmulq_fcma_f64(odds1.5, self.twiddle5);
                odds2.5 = vfcmulq_conj_b_fcma_f64(odds2.5, self.twiddle5);

                odds1.6 = vfcmulq_fcma_f64(odds1.6, self.twiddle6);
                odds2.6 = vfcmulq_conj_b_fcma_f64(odds2.6, self.twiddle6);

                odds1.7 = vfcmulq_fcma_f64(odds1.7, self.twiddle7);
                odds2.7 = vfcmulq_conj_b_fcma_f64(odds2.7, self.twiddle7);

                let mut q0 = NeonButterfly::butterfly2_f64(odds1.0, odds2.0);
                let mut q1 = NeonButterfly::butterfly2_f64(odds1.1, odds2.1);
                let mut q2 = NeonButterfly::butterfly2_f64(odds1.2, odds2.2);
                let mut q3 = NeonButterfly::butterfly2_f64(odds1.3, odds2.3);
                let mut q4 = NeonButterfly::butterfly2_f64(odds1.4, odds2.4);
                let mut q5 = NeonButterfly::butterfly2_f64(odds1.5, odds2.5);
                let mut q6 = NeonButterfly::butterfly2_f64(odds1.6, odds2.6);
                let mut q7 = NeonButterfly::butterfly2_f64(odds1.7, odds2.7);

                q0.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q0.1);
                q1.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q1.1);
                q2.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q2.1);
                q3.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q3.1);
                q4.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q4.1);
                q5.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q5.1);
                q6.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q6.1);
                q7.1 = vcaddq_rot270_f64(vdupq_n_f64(0.), q7.1);

                vst1q_f64(dst.as_mut_ptr().cast(), vaddq_f64(s_evens.0, q0.0));
                vst1q_f64(
                    dst.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.1, q1.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.2, q2.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.3, q3.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.4, q4.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.5, q5.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.6, q6.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(7..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.7, q7.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.8, q0.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(9..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.9, q1.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.10, q2.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(11..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.11, q3.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.12, q4.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(13..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.13, q5.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.14, q6.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(15..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.15, q7.1),
                );

                vst1q_f64(
                    dst.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.0, q0.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(17..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.1, q1.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.2, q2.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(19..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.3, q3.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.4, q4.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(21..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.5, q5.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.6, q6.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(23..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.7, q7.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.8, q0.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(25..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.9, q1.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.10, q2.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(27..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.11, q3.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.12, q4.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(29..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.13, q5.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(30..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.14, q6.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(31..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.15, q7.1),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_backward_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(32) {
                let u0 = vld1q_f64(chunk.as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                let u6 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                let u7 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());
                let u8 = vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast());
                let u9 = vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast());
                let u10 = vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast());
                let u11 = vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast());
                let u12 = vld1q_f64(chunk.get_unchecked(12..).as_ptr().cast());
                let u13 = vld1q_f64(chunk.get_unchecked(13..).as_ptr().cast());
                let u14 = vld1q_f64(chunk.get_unchecked(14..).as_ptr().cast());
                let u15 = vld1q_f64(chunk.get_unchecked(15..).as_ptr().cast());
                let u16 = vld1q_f64(chunk.get_unchecked(16..).as_ptr().cast());
                let u17 = vld1q_f64(chunk.get_unchecked(17..).as_ptr().cast());
                let u18 = vld1q_f64(chunk.get_unchecked(18..).as_ptr().cast());
                let u19 = vld1q_f64(chunk.get_unchecked(19..).as_ptr().cast());
                let u20 = vld1q_f64(chunk.get_unchecked(20..).as_ptr().cast());
                let u21 = vld1q_f64(chunk.get_unchecked(21..).as_ptr().cast());
                let u22 = vld1q_f64(chunk.get_unchecked(22..).as_ptr().cast());
                let u23 = vld1q_f64(chunk.get_unchecked(23..).as_ptr().cast());
                let u24 = vld1q_f64(chunk.get_unchecked(24..).as_ptr().cast());
                let u25 = vld1q_f64(chunk.get_unchecked(25..).as_ptr().cast());
                let u26 = vld1q_f64(chunk.get_unchecked(26..).as_ptr().cast());
                let u27 = vld1q_f64(chunk.get_unchecked(27..).as_ptr().cast());
                let u28 = vld1q_f64(chunk.get_unchecked(28..).as_ptr().cast());
                let u29 = vld1q_f64(chunk.get_unchecked(29..).as_ptr().cast());
                let u30 = vld1q_f64(chunk.get_unchecked(30..).as_ptr().cast());
                let u31 = vld1q_f64(chunk.get_unchecked(31..).as_ptr().cast());

                let s_evens = self.bf16.backward(
                    u0, u2, u4, u6, u8, u10, u12, u14, u16, u18, u20, u22, u24, u26, u28, u30,
                );
                let mut odds1 = self.bf16.bf8.backward(u1, u5, u9, u13, u17, u21, u25, u29);
                let mut odds2 = self.bf16.bf8.backward(u31, u3, u7, u11, u15, u19, u23, u27);

                odds1.1 = vfcmulq_fcma_f64(odds1.1, self.twiddle1);
                odds2.1 = vfcmulq_conj_b_fcma_f64(odds2.1, self.twiddle1);

                odds1.2 = vfcmulq_fcma_f64(odds1.2, self.twiddle2);
                odds2.2 = vfcmulq_conj_b_fcma_f64(odds2.2, self.twiddle2);

                odds1.3 = vfcmulq_fcma_f64(odds1.3, self.twiddle3);
                odds2.3 = vfcmulq_conj_b_fcma_f64(odds2.3, self.twiddle3);

                odds1.4 = vfcmulq_fcma_f64(odds1.4, self.twiddle4);
                odds2.4 = vfcmulq_conj_b_fcma_f64(odds2.4, self.twiddle4);

                odds1.5 = vfcmulq_fcma_f64(odds1.5, self.twiddle5);
                odds2.5 = vfcmulq_conj_b_fcma_f64(odds2.5, self.twiddle5);

                odds1.6 = vfcmulq_fcma_f64(odds1.6, self.twiddle6);
                odds2.6 = vfcmulq_conj_b_fcma_f64(odds2.6, self.twiddle6);

                odds1.7 = vfcmulq_fcma_f64(odds1.7, self.twiddle7);
                odds2.7 = vfcmulq_conj_b_fcma_f64(odds2.7, self.twiddle7);

                let mut q0 = NeonButterfly::butterfly2_f64(odds1.0, odds2.0);
                let mut q1 = NeonButterfly::butterfly2_f64(odds1.1, odds2.1);
                let mut q2 = NeonButterfly::butterfly2_f64(odds1.2, odds2.2);
                let mut q3 = NeonButterfly::butterfly2_f64(odds1.3, odds2.3);
                let mut q4 = NeonButterfly::butterfly2_f64(odds1.4, odds2.4);
                let mut q5 = NeonButterfly::butterfly2_f64(odds1.5, odds2.5);
                let mut q6 = NeonButterfly::butterfly2_f64(odds1.6, odds2.6);
                let mut q7 = NeonButterfly::butterfly2_f64(odds1.7, odds2.7);

                q0.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q0.1);
                q1.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q1.1);
                q2.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q2.1);
                q3.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q3.1);
                q4.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q4.1);
                q5.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q5.1);
                q6.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q6.1);
                q7.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q7.1);

                vst1q_f64(chunk.as_mut_ptr().cast(), vaddq_f64(s_evens.0, q0.0));

                vst1q_f64(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.0, q0.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.1, q1.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.2, q2.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.3, q3.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.4, q4.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.5, q5.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.6, q6.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(7..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.7, q7.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.8, q0.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(9..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.9, q1.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.10, q2.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(11..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.11, q3.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.12, q4.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(13..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.13, q5.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.14, q6.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(15..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.15, q7.1),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.0, q0.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(17..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.1, q1.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.2, q2.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(19..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.3, q3.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.4, q4.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(21..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.5, q5.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.6, q6.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(23..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.7, q7.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.8, q0.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(25..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.9, q1.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.10, q2.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(27..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.11, q3.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.12, q4.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(29..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.13, q5.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(30..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.14, q6.1),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(31..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.15, q7.1),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_out_of_place_backward_f64(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
                let u0 = vld1q_f64(src.as_ptr().cast());
                let u1 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(src.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(src.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(src.get_unchecked(5..).as_ptr().cast());
                let u6 = vld1q_f64(src.get_unchecked(6..).as_ptr().cast());
                let u7 = vld1q_f64(src.get_unchecked(7..).as_ptr().cast());
                let u8 = vld1q_f64(src.get_unchecked(8..).as_ptr().cast());
                let u9 = vld1q_f64(src.get_unchecked(9..).as_ptr().cast());
                let u10 = vld1q_f64(src.get_unchecked(10..).as_ptr().cast());
                let u11 = vld1q_f64(src.get_unchecked(11..).as_ptr().cast());
                let u12 = vld1q_f64(src.get_unchecked(12..).as_ptr().cast());
                let u13 = vld1q_f64(src.get_unchecked(13..).as_ptr().cast());
                let u14 = vld1q_f64(src.get_unchecked(14..).as_ptr().cast());
                let u15 = vld1q_f64(src.get_unchecked(15..).as_ptr().cast());
                let u16 = vld1q_f64(src.get_unchecked(16..).as_ptr().cast());
                let u17 = vld1q_f64(src.get_unchecked(17..).as_ptr().cast());
                let u18 = vld1q_f64(src.get_unchecked(18..).as_ptr().cast());
                let u19 = vld1q_f64(src.get_unchecked(19..).as_ptr().cast());
                let u20 = vld1q_f64(src.get_unchecked(20..).as_ptr().cast());
                let u21 = vld1q_f64(src.get_unchecked(21..).as_ptr().cast());
                let u22 = vld1q_f64(src.get_unchecked(22..).as_ptr().cast());
                let u23 = vld1q_f64(src.get_unchecked(23..).as_ptr().cast());
                let u24 = vld1q_f64(src.get_unchecked(24..).as_ptr().cast());
                let u25 = vld1q_f64(src.get_unchecked(25..).as_ptr().cast());
                let u26 = vld1q_f64(src.get_unchecked(26..).as_ptr().cast());
                let u27 = vld1q_f64(src.get_unchecked(27..).as_ptr().cast());
                let u28 = vld1q_f64(src.get_unchecked(28..).as_ptr().cast());
                let u29 = vld1q_f64(src.get_unchecked(29..).as_ptr().cast());
                let u30 = vld1q_f64(src.get_unchecked(30..).as_ptr().cast());
                let u31 = vld1q_f64(src.get_unchecked(31..).as_ptr().cast());

                let s_evens = self.bf16.backward(
                    u0, u2, u4, u6, u8, u10, u12, u14, u16, u18, u20, u22, u24, u26, u28, u30,
                );
                let mut odds1 = self.bf16.bf8.backward(u1, u5, u9, u13, u17, u21, u25, u29);
                let mut odds2 = self.bf16.bf8.backward(u31, u3, u7, u11, u15, u19, u23, u27);

                odds1.1 = vfcmulq_fcma_f64(odds1.1, self.twiddle1);
                odds2.1 = vfcmulq_conj_b_fcma_f64(odds2.1, self.twiddle1);

                odds1.2 = vfcmulq_fcma_f64(odds1.2, self.twiddle2);
                odds2.2 = vfcmulq_conj_b_fcma_f64(odds2.2, self.twiddle2);

                odds1.3 = vfcmulq_fcma_f64(odds1.3, self.twiddle3);
                odds2.3 = vfcmulq_conj_b_fcma_f64(odds2.3, self.twiddle3);

                odds1.4 = vfcmulq_fcma_f64(odds1.4, self.twiddle4);
                odds2.4 = vfcmulq_conj_b_fcma_f64(odds2.4, self.twiddle4);

                odds1.5 = vfcmulq_fcma_f64(odds1.5, self.twiddle5);
                odds2.5 = vfcmulq_conj_b_fcma_f64(odds2.5, self.twiddle5);

                odds1.6 = vfcmulq_fcma_f64(odds1.6, self.twiddle6);
                odds2.6 = vfcmulq_conj_b_fcma_f64(odds2.6, self.twiddle6);

                odds1.7 = vfcmulq_fcma_f64(odds1.7, self.twiddle7);
                odds2.7 = vfcmulq_conj_b_fcma_f64(odds2.7, self.twiddle7);

                let mut q0 = NeonButterfly::butterfly2_f64(odds1.0, odds2.0);
                let mut q1 = NeonButterfly::butterfly2_f64(odds1.1, odds2.1);
                let mut q2 = NeonButterfly::butterfly2_f64(odds1.2, odds2.2);
                let mut q3 = NeonButterfly::butterfly2_f64(odds1.3, odds2.3);
                let mut q4 = NeonButterfly::butterfly2_f64(odds1.4, odds2.4);
                let mut q5 = NeonButterfly::butterfly2_f64(odds1.5, odds2.5);
                let mut q6 = NeonButterfly::butterfly2_f64(odds1.6, odds2.6);
                let mut q7 = NeonButterfly::butterfly2_f64(odds1.7, odds2.7);

                q0.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q0.1);
                q1.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q1.1);
                q2.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q2.1);
                q3.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q3.1);
                q4.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q4.1);
                q5.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q5.1);
                q6.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q6.1);
                q7.1 = vcaddq_rot90_f64(vdupq_n_f64(0.), q7.1);

                vst1q_f64(dst.as_mut_ptr().cast(), vaddq_f64(s_evens.0, q0.0));
                vst1q_f64(
                    dst.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.1, q1.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.2, q2.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.3, q3.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.4, q4.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.5, q5.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.6, q6.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(7..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.7, q7.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.8, q0.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(9..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.9, q1.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.10, q2.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(11..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.11, q3.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.12, q4.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(13..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.13, q5.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.14, q6.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(15..).as_mut_ptr().cast(),
                    vaddq_f64(s_evens.15, q7.1),
                );

                vst1q_f64(
                    dst.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.0, q0.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(17..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.1, q1.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.2, q2.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(19..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.3, q3.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.4, q4.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(21..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.5, q5.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.6, q6.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(23..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.7, q7.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.8, q0.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(25..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.9, q1.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.10, q2.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(27..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.11, q3.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.12, q4.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(29..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.13, q5.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(30..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.14, q6.1),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(31..).as_mut_ptr().cast(),
                    vsubq_f64(s_evens.15, q7.1),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for NeonFcmaButterfly32d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe {
            match self.direction {
                FftDirection::Forward => self.execute_out_of_place_forward_f64(src, dst),
                FftDirection::Inverse => self.execute_out_of_place_backward_f64(src, dst),
            }
        }
    }
}

impl CompositeFftExecutor<f64> for NeonFcmaButterfly32d {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

pub(crate) struct NeonFcmaButterfly32f {
    direction: FftDirection,
    twiddle1: float32x4_t,
    twiddle2: float32x4_t,
    twiddle3: float32x4_t,
    twiddle4: float32x4_t,
    twiddle5: float32x4_t,
    twiddle6: float32x4_t,
    twiddle7: float32x4_t,
    bf16: NeonFastButterfly16f,
}

impl NeonFcmaButterfly32f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 32, fft_direction);
        let tw2 = compute_twiddle(2, 32, fft_direction);
        let tw3 = compute_twiddle(3, 32, fft_direction);
        let tw4 = compute_twiddle(4, 32, fft_direction);
        let tw5 = compute_twiddle(5, 32, fft_direction);
        let tw6 = compute_twiddle(6, 32, fft_direction);
        let tw7 = compute_twiddle(7, 32, fft_direction);
        unsafe {
            Self {
                direction: fft_direction,
                twiddle1: vcombine_f32(vdup_complex_f32(tw1), vdup_complex_f32(tw1.conj())),
                twiddle2: vcombine_f32(vdup_complex_f32(tw2), vdup_complex_f32(tw2.conj())),
                twiddle3: vcombine_f32(vdup_complex_f32(tw3), vdup_complex_f32(tw3.conj())),
                twiddle4: vcombine_f32(vdup_complex_f32(tw4), vdup_complex_f32(tw4.conj())),
                twiddle5: vcombine_f32(vdup_complex_f32(tw5), vdup_complex_f32(tw5.conj())),
                twiddle6: vcombine_f32(vdup_complex_f32(tw6), vdup_complex_f32(tw6.conj())),
                twiddle7: vcombine_f32(vdup_complex_f32(tw7), vdup_complex_f32(tw7.conj())),
                bf16: NeonFastButterfly16f::new(fft_direction),
            }
        }
    }
}

impl FftExecutor<f32> for NeonFcmaButterfly32f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            match self.direction {
                FftDirection::Forward => self.execute_forward(in_place),
                FftDirection::Inverse => self.execute_backward(in_place),
            }
        }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        32
    }
}

impl NeonFcmaButterfly32f {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_forward(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(32) {
                let u0u1 = vld1q_f32(chunk.as_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());
                let u16u17 = vld1q_f32(chunk.get_unchecked(16..).as_ptr().cast());
                let u18u19 = vld1q_f32(chunk.get_unchecked(18..).as_ptr().cast());
                let u20u21 = vld1q_f32(chunk.get_unchecked(20..).as_ptr().cast());
                let u22u23 = vld1q_f32(chunk.get_unchecked(22..).as_ptr().cast());
                let u24u25 = vld1q_f32(chunk.get_unchecked(24..).as_ptr().cast());
                let u26u27 = vld1q_f32(chunk.get_unchecked(26..).as_ptr().cast());
                let u28u29 = vld1q_f32(chunk.get_unchecked(28..).as_ptr().cast());
                let u30u31 = vld1q_f32(chunk.get_unchecked(30..).as_ptr().cast());

                let s_evens = self.bf16.forwardh(
                    vget_low_f32(u0u1),
                    vget_low_f32(u2u3),
                    vget_low_f32(u4u5),
                    vget_low_f32(u6u7),
                    vget_low_f32(u8u9),
                    vget_low_f32(u10u11),
                    vget_low_f32(u12u13),
                    vget_low_f32(u14u15),
                    vget_low_f32(u16u17),
                    vget_low_f32(u18u19),
                    vget_low_f32(u20u21),
                    vget_low_f32(u22u23),
                    vget_low_f32(u24u25),
                    vget_low_f32(u26u27),
                    vget_low_f32(u28u29),
                    vget_low_f32(u30u31),
                );
                let mut odds1_2 = self.bf16.bf8.forward(
                    vcombine_f32(vget_high_f32(u0u1), vget_high_f32(u30u31)),
                    vcombine_f32(vget_high_f32(u4u5), vget_high_f32(u2u3)), //u5,
                    vcombine_f32(vget_high_f32(u8u9), vget_high_f32(u6u7)), //u9,
                    vcombine_f32(vget_high_f32(u12u13), vget_high_f32(u10u11)), //u13,
                    vcombine_f32(vget_high_f32(u16u17), vget_high_f32(u14u15)), //u17,
                    vcombine_f32(vget_high_f32(u20u21), vget_high_f32(u18u19)), //u21
                    vcombine_f32(vget_high_f32(u24u25), vget_high_f32(u22u23)), //u25,
                    vcombine_f32(vget_high_f32(u28u29), vget_high_f32(u26u27)), //u29,
                );

                odds1_2.1 = vfcmulq_fcma_f32(odds1_2.1, self.twiddle1);
                odds1_2.2 = vfcmulq_fcma_f32(odds1_2.2, self.twiddle2);
                odds1_2.3 = vfcmulq_fcma_f32(odds1_2.3, self.twiddle3);
                odds1_2.4 = vfcmulq_fcma_f32(odds1_2.4, self.twiddle4);
                odds1_2.5 = vfcmulq_fcma_f32(odds1_2.5, self.twiddle5);
                odds1_2.6 = vfcmulq_fcma_f32(odds1_2.6, self.twiddle6);
                odds1_2.7 = vfcmulq_fcma_f32(odds1_2.7, self.twiddle7);

                let mut q0 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.0),
                    vget_high_f32(odds1_2.0),
                );
                let mut q1 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.1),
                    vget_high_f32(odds1_2.1),
                );
                let mut q2 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.2),
                    vget_high_f32(odds1_2.2),
                );
                let mut q3 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.3),
                    vget_high_f32(odds1_2.3),
                );
                let mut q4 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.4),
                    vget_high_f32(odds1_2.4),
                );
                let mut q5 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.5),
                    vget_high_f32(odds1_2.5),
                );
                let mut q6 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.6),
                    vget_high_f32(odds1_2.6),
                );
                let mut q7 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.7),
                    vget_high_f32(odds1_2.7),
                );

                q0.1 = vcadd_rot270_f32(vdup_n_f32(0.), q0.1);
                q1.1 = vcadd_rot270_f32(vdup_n_f32(0.), q1.1);
                q2.1 = vcadd_rot270_f32(vdup_n_f32(0.), q2.1);
                q3.1 = vcadd_rot270_f32(vdup_n_f32(0.), q3.1);
                q4.1 = vcadd_rot270_f32(vdup_n_f32(0.), q4.1);
                q5.1 = vcadd_rot270_f32(vdup_n_f32(0.), q5.1);
                q6.1 = vcadd_rot270_f32(vdup_n_f32(0.), q6.1);
                q7.1 = vcadd_rot270_f32(vdup_n_f32(0.), q7.1);

                let evens01 = vcombine_f32(s_evens.0, s_evens.1);
                let evens23 = vcombine_f32(s_evens.2, s_evens.3);
                let evens45 = vcombine_f32(s_evens.4, s_evens.5);
                let evens67 = vcombine_f32(s_evens.6, s_evens.7);
                let evens89 = vcombine_f32(s_evens.8, s_evens.9);
                let evens1011 = vcombine_f32(s_evens.10, s_evens.11);
                let evens1213 = vcombine_f32(s_evens.12, s_evens.13);
                let evens1415 = vcombine_f32(s_evens.14, s_evens.15);

                let q00 = vcombine_f32(q0.0, q1.0);
                let q01 = vcombine_f32(q2.0, q3.0);
                let q02 = vcombine_f32(q4.0, q5.0);
                let q03 = vcombine_f32(q6.0, q7.0);
                let q04 = vcombine_f32(q0.1, q1.1);
                let q05 = vcombine_f32(q2.1, q3.1);
                let q06 = vcombine_f32(q4.1, q5.1);
                let q07 = vcombine_f32(q6.1, q7.1);

                vst1q_f32(chunk.as_mut_ptr().cast(), vaddq_f32(evens01, q00));
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vaddq_f32(evens23, q01),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vaddq_f32(evens45, q02),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vaddq_f32(evens67, q03),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vaddq_f32(evens89, q04),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vaddq_f32(evens1011, q05),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vaddq_f32(evens1213, q06),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vaddq_f32(evens1415, q07),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vsubq_f32(evens01, q00),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vsubq_f32(evens23, q01),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vsubq_f32(evens45, q02),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    vsubq_f32(evens67, q03),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    vsubq_f32(evens89, q04),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    vsubq_f32(evens1011, q05),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    vsubq_f32(evens1213, q06),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(30..).as_mut_ptr().cast(),
                    vsubq_f32(evens1415, q07),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_out_of_place_forward(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
                let u0u1 = vld1q_f32(src.as_ptr().cast());
                let u2u3 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(src.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(src.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(src.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(src.get_unchecked(14..).as_ptr().cast());
                let u16u17 = vld1q_f32(src.get_unchecked(16..).as_ptr().cast());
                let u18u19 = vld1q_f32(src.get_unchecked(18..).as_ptr().cast());
                let u20u21 = vld1q_f32(src.get_unchecked(20..).as_ptr().cast());
                let u22u23 = vld1q_f32(src.get_unchecked(22..).as_ptr().cast());
                let u24u25 = vld1q_f32(src.get_unchecked(24..).as_ptr().cast());
                let u26u27 = vld1q_f32(src.get_unchecked(26..).as_ptr().cast());
                let u28u29 = vld1q_f32(src.get_unchecked(28..).as_ptr().cast());
                let u30u31 = vld1q_f32(src.get_unchecked(30..).as_ptr().cast());

                let s_evens = self.bf16.forwardh(
                    vget_low_f32(u0u1),
                    vget_low_f32(u2u3),
                    vget_low_f32(u4u5),
                    vget_low_f32(u6u7),
                    vget_low_f32(u8u9),
                    vget_low_f32(u10u11),
                    vget_low_f32(u12u13),
                    vget_low_f32(u14u15),
                    vget_low_f32(u16u17),
                    vget_low_f32(u18u19),
                    vget_low_f32(u20u21),
                    vget_low_f32(u22u23),
                    vget_low_f32(u24u25),
                    vget_low_f32(u26u27),
                    vget_low_f32(u28u29),
                    vget_low_f32(u30u31),
                );
                let mut odds1_2 = self.bf16.bf8.forward(
                    vcombine_f32(vget_high_f32(u0u1), vget_high_f32(u30u31)),
                    vcombine_f32(vget_high_f32(u4u5), vget_high_f32(u2u3)), //u5,
                    vcombine_f32(vget_high_f32(u8u9), vget_high_f32(u6u7)), //u9,
                    vcombine_f32(vget_high_f32(u12u13), vget_high_f32(u10u11)), //u13,
                    vcombine_f32(vget_high_f32(u16u17), vget_high_f32(u14u15)), //u17,
                    vcombine_f32(vget_high_f32(u20u21), vget_high_f32(u18u19)), //u21
                    vcombine_f32(vget_high_f32(u24u25), vget_high_f32(u22u23)), //u25,
                    vcombine_f32(vget_high_f32(u28u29), vget_high_f32(u26u27)), //u29,
                );

                odds1_2.1 = vfcmulq_fcma_f32(odds1_2.1, self.twiddle1);
                odds1_2.2 = vfcmulq_fcma_f32(odds1_2.2, self.twiddle2);
                odds1_2.3 = vfcmulq_fcma_f32(odds1_2.3, self.twiddle3);
                odds1_2.4 = vfcmulq_fcma_f32(odds1_2.4, self.twiddle4);
                odds1_2.5 = vfcmulq_fcma_f32(odds1_2.5, self.twiddle5);
                odds1_2.6 = vfcmulq_fcma_f32(odds1_2.6, self.twiddle6);
                odds1_2.7 = vfcmulq_fcma_f32(odds1_2.7, self.twiddle7);

                let mut q0 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.0),
                    vget_high_f32(odds1_2.0),
                );
                let mut q1 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.1),
                    vget_high_f32(odds1_2.1),
                );
                let mut q2 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.2),
                    vget_high_f32(odds1_2.2),
                );
                let mut q3 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.3),
                    vget_high_f32(odds1_2.3),
                );
                let mut q4 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.4),
                    vget_high_f32(odds1_2.4),
                );
                let mut q5 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.5),
                    vget_high_f32(odds1_2.5),
                );
                let mut q6 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.6),
                    vget_high_f32(odds1_2.6),
                );
                let mut q7 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.7),
                    vget_high_f32(odds1_2.7),
                );

                q0.1 = vcadd_rot270_f32(vdup_n_f32(0.), q0.1);
                q1.1 = vcadd_rot270_f32(vdup_n_f32(0.), q1.1);
                q2.1 = vcadd_rot270_f32(vdup_n_f32(0.), q2.1);
                q3.1 = vcadd_rot270_f32(vdup_n_f32(0.), q3.1);
                q4.1 = vcadd_rot270_f32(vdup_n_f32(0.), q4.1);
                q5.1 = vcadd_rot270_f32(vdup_n_f32(0.), q5.1);
                q6.1 = vcadd_rot270_f32(vdup_n_f32(0.), q6.1);
                q7.1 = vcadd_rot270_f32(vdup_n_f32(0.), q7.1);

                let evens01 = vcombine_f32(s_evens.0, s_evens.1);
                let evens23 = vcombine_f32(s_evens.2, s_evens.3);
                let evens45 = vcombine_f32(s_evens.4, s_evens.5);
                let evens67 = vcombine_f32(s_evens.6, s_evens.7);
                let evens89 = vcombine_f32(s_evens.8, s_evens.9);
                let evens1011 = vcombine_f32(s_evens.10, s_evens.11);
                let evens1213 = vcombine_f32(s_evens.12, s_evens.13);
                let evens1415 = vcombine_f32(s_evens.14, s_evens.15);

                let q00 = vcombine_f32(q0.0, q1.0);
                let q01 = vcombine_f32(q2.0, q3.0);
                let q02 = vcombine_f32(q4.0, q5.0);
                let q03 = vcombine_f32(q6.0, q7.0);
                let q04 = vcombine_f32(q0.1, q1.1);
                let q05 = vcombine_f32(q2.1, q3.1);
                let q06 = vcombine_f32(q4.1, q5.1);
                let q07 = vcombine_f32(q6.1, q7.1);

                vst1q_f32(dst.as_mut_ptr().cast(), vaddq_f32(evens01, q00));
                vst1q_f32(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vaddq_f32(evens23, q01),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vaddq_f32(evens45, q02),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vaddq_f32(evens67, q03),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vaddq_f32(evens89, q04),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vaddq_f32(evens1011, q05),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vaddq_f32(evens1213, q06),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vaddq_f32(evens1415, q07),
                );

                vst1q_f32(
                    dst.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vsubq_f32(evens01, q00),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vsubq_f32(evens23, q01),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vsubq_f32(evens45, q02),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    vsubq_f32(evens67, q03),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    vsubq_f32(evens89, q04),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    vsubq_f32(evens1011, q05),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    vsubq_f32(evens1213, q06),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(30..).as_mut_ptr().cast(),
                    vsubq_f32(evens1415, q07),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_backward(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(32) {
                let u0u1 = vld1q_f32(chunk.as_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());
                let u16u17 = vld1q_f32(chunk.get_unchecked(16..).as_ptr().cast());
                let u18u19 = vld1q_f32(chunk.get_unchecked(18..).as_ptr().cast());
                let u20u21 = vld1q_f32(chunk.get_unchecked(20..).as_ptr().cast());
                let u22u23 = vld1q_f32(chunk.get_unchecked(22..).as_ptr().cast());
                let u24u25 = vld1q_f32(chunk.get_unchecked(24..).as_ptr().cast());
                let u26u27 = vld1q_f32(chunk.get_unchecked(26..).as_ptr().cast());
                let u28u29 = vld1q_f32(chunk.get_unchecked(28..).as_ptr().cast());
                let u30u31 = vld1q_f32(chunk.get_unchecked(30..).as_ptr().cast());

                let s_evens = self.bf16.backwardh(
                    vget_low_f32(u0u1),
                    vget_low_f32(u2u3),
                    vget_low_f32(u4u5),
                    vget_low_f32(u6u7),
                    vget_low_f32(u8u9),
                    vget_low_f32(u10u11),
                    vget_low_f32(u12u13),
                    vget_low_f32(u14u15),
                    vget_low_f32(u16u17),
                    vget_low_f32(u18u19),
                    vget_low_f32(u20u21),
                    vget_low_f32(u22u23),
                    vget_low_f32(u24u25),
                    vget_low_f32(u26u27),
                    vget_low_f32(u28u29),
                    vget_low_f32(u30u31),
                );
                let mut odds1_2 = self.bf16.bf8.backward(
                    vcombine_f32(vget_high_f32(u0u1), vget_high_f32(u30u31)),
                    vcombine_f32(vget_high_f32(u4u5), vget_high_f32(u2u3)), //u5,
                    vcombine_f32(vget_high_f32(u8u9), vget_high_f32(u6u7)), //u9,
                    vcombine_f32(vget_high_f32(u12u13), vget_high_f32(u10u11)), //u13,
                    vcombine_f32(vget_high_f32(u16u17), vget_high_f32(u14u15)), //u17,
                    vcombine_f32(vget_high_f32(u20u21), vget_high_f32(u18u19)), //u21
                    vcombine_f32(vget_high_f32(u24u25), vget_high_f32(u22u23)), //u25,
                    vcombine_f32(vget_high_f32(u28u29), vget_high_f32(u26u27)), //u29,
                );
                // let mut odds2 =
                //     self.bf16
                //         .bf8
                //         .exec(u31, u3, u7, u11, u15, u19, u23, u27, self.bf16.rot);

                odds1_2.1 = vfcmulq_fcma_f32(odds1_2.1, self.twiddle1);
                odds1_2.2 = vfcmulq_fcma_f32(odds1_2.2, self.twiddle2);
                odds1_2.3 = vfcmulq_fcma_f32(odds1_2.3, self.twiddle3);
                odds1_2.4 = vfcmulq_fcma_f32(odds1_2.4, self.twiddle4);
                odds1_2.5 = vfcmulq_fcma_f32(odds1_2.5, self.twiddle5);
                odds1_2.6 = vfcmulq_fcma_f32(odds1_2.6, self.twiddle6);
                odds1_2.7 = vfcmulq_fcma_f32(odds1_2.7, self.twiddle7);

                let mut q0 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.0),
                    vget_high_f32(odds1_2.0),
                );
                let mut q1 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.1),
                    vget_high_f32(odds1_2.1),
                );
                let mut q2 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.2),
                    vget_high_f32(odds1_2.2),
                );
                let mut q3 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.3),
                    vget_high_f32(odds1_2.3),
                );
                let mut q4 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.4),
                    vget_high_f32(odds1_2.4),
                );
                let mut q5 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.5),
                    vget_high_f32(odds1_2.5),
                );
                let mut q6 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.6),
                    vget_high_f32(odds1_2.6),
                );
                let mut q7 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.7),
                    vget_high_f32(odds1_2.7),
                );

                q0.1 = vcadd_rot90_f32(vdup_n_f32(0.), q0.1);
                q1.1 = vcadd_rot90_f32(vdup_n_f32(0.), q1.1);
                q2.1 = vcadd_rot90_f32(vdup_n_f32(0.), q2.1);
                q3.1 = vcadd_rot90_f32(vdup_n_f32(0.), q3.1);
                q4.1 = vcadd_rot90_f32(vdup_n_f32(0.), q4.1);
                q5.1 = vcadd_rot90_f32(vdup_n_f32(0.), q5.1);
                q6.1 = vcadd_rot90_f32(vdup_n_f32(0.), q6.1);
                q7.1 = vcadd_rot90_f32(vdup_n_f32(0.), q7.1);

                let evens01 = vcombine_f32(s_evens.0, s_evens.1);
                let evens23 = vcombine_f32(s_evens.2, s_evens.3);
                let evens45 = vcombine_f32(s_evens.4, s_evens.5);
                let evens67 = vcombine_f32(s_evens.6, s_evens.7);
                let evens89 = vcombine_f32(s_evens.8, s_evens.9);
                let evens1011 = vcombine_f32(s_evens.10, s_evens.11);
                let evens1213 = vcombine_f32(s_evens.12, s_evens.13);
                let evens1415 = vcombine_f32(s_evens.14, s_evens.15);

                let q00 = vcombine_f32(q0.0, q1.0);
                let q01 = vcombine_f32(q2.0, q3.0);
                let q02 = vcombine_f32(q4.0, q5.0);
                let q03 = vcombine_f32(q6.0, q7.0);
                let q04 = vcombine_f32(q0.1, q1.1);
                let q05 = vcombine_f32(q2.1, q3.1);
                let q06 = vcombine_f32(q4.1, q5.1);
                let q07 = vcombine_f32(q6.1, q7.1);

                vst1q_f32(chunk.as_mut_ptr().cast(), vaddq_f32(evens01, q00));
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vaddq_f32(evens23, q01),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vaddq_f32(evens45, q02),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vaddq_f32(evens67, q03),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vaddq_f32(evens89, q04),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vaddq_f32(evens1011, q05),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vaddq_f32(evens1213, q06),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vaddq_f32(evens1415, q07),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vsubq_f32(evens01, q00),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vsubq_f32(evens23, q01),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vsubq_f32(evens45, q02),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    vsubq_f32(evens67, q03),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    vsubq_f32(evens89, q04),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    vsubq_f32(evens1011, q05),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    vsubq_f32(evens1213, q06),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(30..).as_mut_ptr().cast(),
                    vsubq_f32(evens1415, q07),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_out_of_place_backward(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
                let u0u1 = vld1q_f32(src.as_ptr().cast());
                let u2u3 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(src.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(src.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(src.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(src.get_unchecked(14..).as_ptr().cast());
                let u16u17 = vld1q_f32(src.get_unchecked(16..).as_ptr().cast());
                let u18u19 = vld1q_f32(src.get_unchecked(18..).as_ptr().cast());
                let u20u21 = vld1q_f32(src.get_unchecked(20..).as_ptr().cast());
                let u22u23 = vld1q_f32(src.get_unchecked(22..).as_ptr().cast());
                let u24u25 = vld1q_f32(src.get_unchecked(24..).as_ptr().cast());
                let u26u27 = vld1q_f32(src.get_unchecked(26..).as_ptr().cast());
                let u28u29 = vld1q_f32(src.get_unchecked(28..).as_ptr().cast());
                let u30u31 = vld1q_f32(src.get_unchecked(30..).as_ptr().cast());

                let s_evens = self.bf16.backwardh(
                    vget_low_f32(u0u1),
                    vget_low_f32(u2u3),
                    vget_low_f32(u4u5),
                    vget_low_f32(u6u7),
                    vget_low_f32(u8u9),
                    vget_low_f32(u10u11),
                    vget_low_f32(u12u13),
                    vget_low_f32(u14u15),
                    vget_low_f32(u16u17),
                    vget_low_f32(u18u19),
                    vget_low_f32(u20u21),
                    vget_low_f32(u22u23),
                    vget_low_f32(u24u25),
                    vget_low_f32(u26u27),
                    vget_low_f32(u28u29),
                    vget_low_f32(u30u31),
                );
                let mut odds1_2 = self.bf16.bf8.backward(
                    vcombine_f32(vget_high_f32(u0u1), vget_high_f32(u30u31)),
                    vcombine_f32(vget_high_f32(u4u5), vget_high_f32(u2u3)), //u5,
                    vcombine_f32(vget_high_f32(u8u9), vget_high_f32(u6u7)), //u9,
                    vcombine_f32(vget_high_f32(u12u13), vget_high_f32(u10u11)), //u13,
                    vcombine_f32(vget_high_f32(u16u17), vget_high_f32(u14u15)), //u17,
                    vcombine_f32(vget_high_f32(u20u21), vget_high_f32(u18u19)), //u21
                    vcombine_f32(vget_high_f32(u24u25), vget_high_f32(u22u23)), //u25,
                    vcombine_f32(vget_high_f32(u28u29), vget_high_f32(u26u27)), //u29,
                );
                // let mut odds2 =
                //     self.bf16
                //         .bf8
                //         .exec(u31, u3, u7, u11, u15, u19, u23, u27, self.bf16.rot);

                odds1_2.1 = vfcmulq_fcma_f32(odds1_2.1, self.twiddle1);
                odds1_2.2 = vfcmulq_fcma_f32(odds1_2.2, self.twiddle2);
                odds1_2.3 = vfcmulq_fcma_f32(odds1_2.3, self.twiddle3);
                odds1_2.4 = vfcmulq_fcma_f32(odds1_2.4, self.twiddle4);
                odds1_2.5 = vfcmulq_fcma_f32(odds1_2.5, self.twiddle5);
                odds1_2.6 = vfcmulq_fcma_f32(odds1_2.6, self.twiddle6);
                odds1_2.7 = vfcmulq_fcma_f32(odds1_2.7, self.twiddle7);

                let mut q0 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.0),
                    vget_high_f32(odds1_2.0),
                );
                let mut q1 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.1),
                    vget_high_f32(odds1_2.1),
                );
                let mut q2 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.2),
                    vget_high_f32(odds1_2.2),
                );
                let mut q3 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.3),
                    vget_high_f32(odds1_2.3),
                );
                let mut q4 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.4),
                    vget_high_f32(odds1_2.4),
                );
                let mut q5 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.5),
                    vget_high_f32(odds1_2.5),
                );
                let mut q6 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.6),
                    vget_high_f32(odds1_2.6),
                );
                let mut q7 = NeonButterfly::butterfly2h_f32(
                    vget_low_f32(odds1_2.7),
                    vget_high_f32(odds1_2.7),
                );

                q0.1 = vcadd_rot90_f32(vdup_n_f32(0.), q0.1);
                q1.1 = vcadd_rot90_f32(vdup_n_f32(0.), q1.1);
                q2.1 = vcadd_rot90_f32(vdup_n_f32(0.), q2.1);
                q3.1 = vcadd_rot90_f32(vdup_n_f32(0.), q3.1);
                q4.1 = vcadd_rot90_f32(vdup_n_f32(0.), q4.1);
                q5.1 = vcadd_rot90_f32(vdup_n_f32(0.), q5.1);
                q6.1 = vcadd_rot90_f32(vdup_n_f32(0.), q6.1);
                q7.1 = vcadd_rot90_f32(vdup_n_f32(0.), q7.1);

                let evens01 = vcombine_f32(s_evens.0, s_evens.1);
                let evens23 = vcombine_f32(s_evens.2, s_evens.3);
                let evens45 = vcombine_f32(s_evens.4, s_evens.5);
                let evens67 = vcombine_f32(s_evens.6, s_evens.7);
                let evens89 = vcombine_f32(s_evens.8, s_evens.9);
                let evens1011 = vcombine_f32(s_evens.10, s_evens.11);
                let evens1213 = vcombine_f32(s_evens.12, s_evens.13);
                let evens1415 = vcombine_f32(s_evens.14, s_evens.15);

                let q00 = vcombine_f32(q0.0, q1.0);
                let q01 = vcombine_f32(q2.0, q3.0);
                let q02 = vcombine_f32(q4.0, q5.0);
                let q03 = vcombine_f32(q6.0, q7.0);
                let q04 = vcombine_f32(q0.1, q1.1);
                let q05 = vcombine_f32(q2.1, q3.1);
                let q06 = vcombine_f32(q4.1, q5.1);
                let q07 = vcombine_f32(q6.1, q7.1);

                vst1q_f32(dst.as_mut_ptr().cast(), vaddq_f32(evens01, q00));
                vst1q_f32(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vaddq_f32(evens23, q01),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vaddq_f32(evens45, q02),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vaddq_f32(evens67, q03),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vaddq_f32(evens89, q04),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vaddq_f32(evens1011, q05),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vaddq_f32(evens1213, q06),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vaddq_f32(evens1415, q07),
                );

                vst1q_f32(
                    dst.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    vsubq_f32(evens01, q00),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    vsubq_f32(evens23, q01),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    vsubq_f32(evens45, q02),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    vsubq_f32(evens67, q03),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    vsubq_f32(evens89, q04),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    vsubq_f32(evens1011, q05),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    vsubq_f32(evens1213, q06),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(30..).as_mut_ptr().cast(),
                    vsubq_f32(evens1415, q07),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for NeonFcmaButterfly32f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe {
            match self.direction {
                FftDirection::Forward => self.execute_out_of_place_forward(src, dst),
                FftDirection::Inverse => self.execute_out_of_place_backward(src, dst),
            }
        }
    }
}

impl CompositeFftExecutor<f32> for NeonFcmaButterfly32f {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_fcma_butterfly!(test_fcma_butterfly32, f32, NeonFcmaButterfly32f, 32, 1e-5);
    test_fcma_butterfly!(
        test_fcma_butterfly32_f64,
        f64,
        NeonFcmaButterfly32d,
        32,
        1e-7
    );

    test_oof_fcma_butterfly!(test_oof_butterfly32, f32, NeonFcmaButterfly32f, 32, 1e-5);
    test_oof_fcma_butterfly!(
        test_oof_butterfly32_f64,
        f64,
        NeonFcmaButterfly32d,
        32,
        1e-9
    );
}
