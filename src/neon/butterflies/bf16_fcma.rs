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
use crate::neon::butterflies::fast_bf8::NeonFastButterfly8;
use crate::neon::util::{
    vfcmul_b_conj_fcma_f32, vfcmul_fcma_f32, vfcmulq_b_conj_fcma_f32, vfcmulq_conj_b_fcma_f64,
    vfcmulq_fcma_f32, vfcmulq_fcma_f64, vqtrnq_f32,
};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaButterfly16<T> {
    direction: FftDirection,
    bf8: NeonFastButterfly8<T>,
    twiddle1: [T; 4],
    twiddle2: [T; 4],
    twiddle3: [T; 4],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonFcmaButterfly16<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 16, fft_direction);
        let tw2 = compute_twiddle(2, 16, fft_direction);
        let tw3 = compute_twiddle(3, 16, fft_direction);
        Self {
            direction: fft_direction,
            bf8: NeonFastButterfly8::new(fft_direction),
            twiddle1: [tw1.re, tw1.im, tw1.re, tw1.im],
            twiddle2: [tw2.re, tw2.im, tw2.re, tw2.im],
            twiddle3: [tw3.re, tw3.im, tw3.re, tw3.im],
        }
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly16<f64> {
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
        16
    }
}

impl NeonFcmaButterfly16<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_forward_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 16 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let tw1 = vld1q_f64(self.twiddle1.as_ptr());
            let tw2 = vld1q_f64(self.twiddle2.as_ptr());
            let tw3 = vld1q_f64(self.twiddle3.as_ptr());

            for chunk in in_place.chunks_exact_mut(16) {
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

                let evens = self.bf8.forward(u0, u2, u4, u6, u8, u10, u12, u14);

                let mut odds_1 = NeonButterfly::bf4_f64_forward(u1, u5, u9, u13);
                let mut odds_2 = NeonButterfly::bf4_f64_forward(u15, u3, u7, u11);

                odds_1.1 = vfcmulq_fcma_f64(odds_1.1, tw1);
                odds_2.1 = vfcmulq_conj_b_fcma_f64(odds_2.1, tw1);

                odds_1.2 = vfcmulq_fcma_f64(odds_1.2, tw2);
                odds_2.2 = vfcmulq_conj_b_fcma_f64(odds_2.2, tw2);

                odds_1.3 = vfcmulq_fcma_f64(odds_1.3, tw3);
                odds_2.3 = vfcmulq_conj_b_fcma_f64(odds_2.3, tw3);

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

                vst1q_f64(chunk.as_mut_ptr().cast(), vaddq_f64(evens.0, odds_1.0));
                vst1q_f64(
                    chunk.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vaddq_f64(evens.1, odds_1.1),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vaddq_f64(evens.2, odds_1.2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vaddq_f64(evens.3, odds_1.3),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vaddq_f64(evens.4, odds_2.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    vaddq_f64(evens.5, odds_2.1),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vaddq_f64(evens.6, odds_2.2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(7..).as_mut_ptr().cast(),
                    vaddq_f64(evens.7, odds_2.3),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vsubq_f64(evens.0, odds_1.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(9..).as_mut_ptr().cast(),
                    vsubq_f64(evens.1, odds_1.1),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vsubq_f64(evens.2, odds_1.2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(11..).as_mut_ptr().cast(),
                    vsubq_f64(evens.3, odds_1.3),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vsubq_f64(evens.4, odds_2.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(13..).as_mut_ptr().cast(),
                    vsubq_f64(evens.5, odds_2.1),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vsubq_f64(evens.6, odds_2.2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(15..).as_mut_ptr().cast(),
                    vsubq_f64(evens.7, odds_2.3),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_backward_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 16 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let tw1 = vld1q_f64(self.twiddle1.as_ptr());
            let tw2 = vld1q_f64(self.twiddle2.as_ptr());
            let tw3 = vld1q_f64(self.twiddle3.as_ptr());

            for chunk in in_place.chunks_exact_mut(16) {
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

                let evens = self.bf8.backward(u0, u2, u4, u6, u8, u10, u12, u14);

                let mut odds_1 = NeonButterfly::bf4_f64_backward(u1, u5, u9, u13);
                let mut odds_2 = NeonButterfly::bf4_f64_backward(u15, u3, u7, u11);

                odds_1.1 = vfcmulq_fcma_f64(odds_1.1, tw1);
                odds_2.1 = vfcmulq_conj_b_fcma_f64(odds_2.1, tw1);

                odds_1.2 = vfcmulq_fcma_f64(odds_1.2, tw2);
                odds_2.2 = vfcmulq_conj_b_fcma_f64(odds_2.2, tw2);

                odds_1.3 = vfcmulq_fcma_f64(odds_1.3, tw3);
                odds_2.3 = vfcmulq_conj_b_fcma_f64(odds_2.3, tw3);

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

                vst1q_f64(chunk.as_mut_ptr().cast(), vaddq_f64(evens.0, odds_1.0));
                vst1q_f64(
                    chunk.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vaddq_f64(evens.1, odds_1.1),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vaddq_f64(evens.2, odds_1.2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vaddq_f64(evens.3, odds_1.3),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vaddq_f64(evens.4, odds_2.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    vaddq_f64(evens.5, odds_2.1),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vaddq_f64(evens.6, odds_2.2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(7..).as_mut_ptr().cast(),
                    vaddq_f64(evens.7, odds_2.3),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vsubq_f64(evens.0, odds_1.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(9..).as_mut_ptr().cast(),
                    vsubq_f64(evens.1, odds_1.1),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vsubq_f64(evens.2, odds_1.2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(11..).as_mut_ptr().cast(),
                    vsubq_f64(evens.3, odds_1.3),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vsubq_f64(evens.4, odds_2.0),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(13..).as_mut_ptr().cast(),
                    vsubq_f64(evens.5, odds_2.1),
                );

                vst1q_f64(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vsubq_f64(evens.6, odds_2.2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(15..).as_mut_ptr().cast(),
                    vsubq_f64(evens.7, odds_2.3),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaButterfly16<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            match self.direction {
                FftDirection::Forward => self.execute_forward(in_place),
                FftDirection::Inverse => self.execute_backwards(in_place),
            }
        }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        16
    }
}

impl NeonFcmaButterfly16<f32> {
    #[target_feature(enable = "neon")]
    unsafe fn execute_forward(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 16 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let tw1 = vld1q_f32(self.twiddle1.as_ptr());
            let tw2 = vld1q_f32(self.twiddle2.as_ptr());
            let tw3 = vld1q_f32(self.twiddle3.as_ptr());

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

                let (u0, u1) = vqtrnq_f32(u0u1, u16u17);
                let (u2, u3) = vqtrnq_f32(u2u3, u18u19);
                let (u4, u5) = vqtrnq_f32(u4u5, u20u21);
                let (u6, u7) = vqtrnq_f32(u6u7, u22u23);
                let (u8, u9) = vqtrnq_f32(u8u9, u24u25);
                let (u10, u11) = vqtrnq_f32(u10u11, u26u27);
                let (u12, u13) = vqtrnq_f32(u12u13, u28u29);
                let (u14, u15) = vqtrnq_f32(u14u15, u30u31);

                let evens = self.bf8.forward(u0, u2, u4, u6, u8, u10, u12, u14);

                let mut odds_1 = NeonButterfly::bf4_forward_f32(u1, u5, u9, u13);
                let mut odds_2 = NeonButterfly::bf4_forward_f32(u15, u3, u7, u11);

                odds_1.1 = vfcmulq_fcma_f32(odds_1.1, tw1);
                odds_2.1 = vfcmulq_b_conj_fcma_f32(odds_2.1, tw1);

                odds_1.2 = vfcmulq_fcma_f32(odds_1.2, tw2);
                odds_2.2 = vfcmulq_b_conj_fcma_f32(odds_2.2, tw2);

                odds_1.3 = vfcmulq_fcma_f32(odds_1.3, tw3);
                odds_2.3 = vfcmulq_b_conj_fcma_f32(odds_2.3, tw3);

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
                odds_2.0 = vcaddq_rot270_f32(vdupq_n_f32(0.), odds_2.0);
                odds_2.1 = vcaddq_rot270_f32(vdupq_n_f32(0.), odds_2.1);
                odds_2.2 = vcaddq_rot270_f32(vdupq_n_f32(0.), odds_2.2);
                odds_2.3 = vcaddq_rot270_f32(vdupq_n_f32(0.), odds_2.3);

                let zy0 = vaddq_f32(evens.0, odds_1.0);
                let zy1 = vaddq_f32(evens.1, odds_1.1);
                let zy2 = vaddq_f32(evens.2, odds_1.2);
                let zy3 = vaddq_f32(evens.3, odds_1.3);
                let zy4 = vaddq_f32(evens.4, odds_2.0);
                let zy5 = vaddq_f32(evens.5, odds_2.1);
                let zy6 = vaddq_f32(evens.6, odds_2.2);
                let zy7 = vaddq_f32(evens.7, odds_2.3);
                let zy8 = vsubq_f32(evens.0, odds_1.0);
                let zy9 = vsubq_f32(evens.1, odds_1.1);
                let zy10 = vsubq_f32(evens.2, odds_1.2);
                let zy11 = vsubq_f32(evens.3, odds_1.3);
                let zy12 = vsubq_f32(evens.4, odds_2.0);
                let zy13 = vsubq_f32(evens.5, odds_2.1);
                let zy14 = vsubq_f32(evens.6, odds_2.2);
                let zy15 = vsubq_f32(evens.7, odds_2.3);

                let (y0, y8) = vqtrnq_f32(zy0, zy1);
                let (y1, y9) = vqtrnq_f32(zy2, zy3);
                let (y2, y10) = vqtrnq_f32(zy4, zy5);
                let (y3, y11) = vqtrnq_f32(zy6, zy7);
                let (y4, y12) = vqtrnq_f32(zy8, zy9);
                let (y5, y13) = vqtrnq_f32(zy10, zy11);
                let (y6, y14) = vqtrnq_f32(zy12, zy13);
                let (y7, y15) = vqtrnq_f32(zy14, zy15);

                vst1q_f32(chunk.as_mut_ptr().cast(), y0);
                vst1q_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y1);

                vst1q_f32(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y2);
                vst1q_f32(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y3);

                vst1q_f32(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y4);
                vst1q_f32(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y5);

                vst1q_f32(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y6);
                vst1q_f32(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y7);

                vst1q_f32(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), y8);
                vst1q_f32(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), y9);

                vst1q_f32(chunk.get_unchecked_mut(20..).as_mut_ptr().cast(), y10);
                vst1q_f32(chunk.get_unchecked_mut(22..).as_mut_ptr().cast(), y11);

                vst1q_f32(chunk.get_unchecked_mut(24..).as_mut_ptr().cast(), y12);
                vst1q_f32(chunk.get_unchecked_mut(26..).as_mut_ptr().cast(), y13);

                vst1q_f32(chunk.get_unchecked_mut(28..).as_mut_ptr().cast(), y14);
                vst1q_f32(chunk.get_unchecked_mut(30..).as_mut_ptr().cast(), y15);
            }

            let rem = in_place.chunks_exact_mut(32).into_remainder();

            for chunk in rem.chunks_exact_mut(16) {
                let u0u1 = vld1q_f32(chunk.as_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());

                let (u0, u1) = (vget_low_f32(u0u1), vget_high_f32(u0u1));
                let (u2, u3) = (vget_low_f32(u2u3), vget_high_f32(u2u3));
                let (u4, u5) = (vget_low_f32(u4u5), vget_high_f32(u4u5));
                let (u6, u7) = (vget_low_f32(u6u7), vget_high_f32(u6u7));
                let (u8, u9) = (vget_low_f32(u8u9), vget_high_f32(u8u9));
                let (u10, u11) = (vget_low_f32(u10u11), vget_high_f32(u10u11));
                let (u12, u13) = (vget_low_f32(u12u13), vget_high_f32(u12u13));
                let (u14, u15) = (vget_low_f32(u14u15), vget_high_f32(u14u15));

                let evens = self.bf8.forwardh(u0, u2, u4, u6, u8, u10, u12, u14);

                let mut odds_1 = NeonButterfly::bf4h_forward_f32(u1, u5, u9, u13);
                let mut odds_2 = NeonButterfly::bf4h_forward_f32(u15, u3, u7, u11);

                let o1 = vfcmulq_fcma_f32(
                    vcombine_f32(odds_1.1, odds_1.2),
                    vcombine_f32(vget_low_f32(tw1), vget_low_f32(tw2)),
                );

                let o2 = vfcmulq_b_conj_fcma_f32(
                    vcombine_f32(odds_2.1, odds_2.2),
                    vcombine_f32(vget_low_f32(tw1), vget_low_f32(tw2)),
                );

                odds_1.1 = vget_low_f32(o1);
                odds_2.1 = vget_low_f32(o2);

                odds_1.2 = vget_high_f32(o1);
                odds_2.2 = vget_high_f32(o2);

                odds_1.3 = vfcmul_fcma_f32(odds_1.3, vget_low_f32(tw3));
                odds_2.3 = vfcmul_b_conj_fcma_f32(odds_2.3, vget_low_f32(tw3));

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

                let y0 = vadd_f32(evens.0, odds_1.0);
                let y1 = vadd_f32(evens.1, odds_1.1);
                let y2 = vadd_f32(evens.2, odds_1.2);
                let y3 = vadd_f32(evens.3, odds_1.3);
                let y4 = vadd_f32(evens.4, odds_2.0);
                let y5 = vadd_f32(evens.5, odds_2.1);
                let y6 = vadd_f32(evens.6, odds_2.2);
                let y7 = vadd_f32(evens.7, odds_2.3);
                let y8 = vsub_f32(evens.0, odds_1.0);
                let y9 = vsub_f32(evens.1, odds_1.1);
                let y10 = vsub_f32(evens.2, odds_1.2);
                let y11 = vsub_f32(evens.3, odds_1.3);
                let y12 = vsub_f32(evens.4, odds_2.0);
                let y13 = vsub_f32(evens.5, odds_2.1);
                let y14 = vsub_f32(evens.6, odds_2.2);
                let y15 = vsub_f32(evens.7, odds_2.3);

                vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y0, y1));
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y2, y3),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y4, y5),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y6, y7),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y8, y9),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(y10, y11),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vcombine_f32(y12, y13),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vcombine_f32(y14, y15),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "neon")]
    unsafe fn execute_backwards(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 16 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let tw1 = vld1q_f32(self.twiddle1.as_ptr());
            let tw2 = vld1q_f32(self.twiddle2.as_ptr());
            let tw3 = vld1q_f32(self.twiddle3.as_ptr());

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

                let (u0, u1) = vqtrnq_f32(u0u1, u16u17);
                let (u2, u3) = vqtrnq_f32(u2u3, u18u19);
                let (u4, u5) = vqtrnq_f32(u4u5, u20u21);
                let (u6, u7) = vqtrnq_f32(u6u7, u22u23);
                let (u8, u9) = vqtrnq_f32(u8u9, u24u25);
                let (u10, u11) = vqtrnq_f32(u10u11, u26u27);
                let (u12, u13) = vqtrnq_f32(u12u13, u28u29);
                let (u14, u15) = vqtrnq_f32(u14u15, u30u31);

                let evens = self.bf8.backward(u0, u2, u4, u6, u8, u10, u12, u14);

                let mut odds_1 = NeonButterfly::bf4_backward_f32(u1, u5, u9, u13);
                let mut odds_2 = NeonButterfly::bf4_backward_f32(u15, u3, u7, u11);

                odds_1.1 = vfcmulq_fcma_f32(odds_1.1, tw1);
                odds_2.1 = vfcmulq_b_conj_fcma_f32(odds_2.1, tw1);

                odds_1.2 = vfcmulq_fcma_f32(odds_1.2, tw2);
                odds_2.2 = vfcmulq_b_conj_fcma_f32(odds_2.2, tw2);

                odds_1.3 = vfcmulq_fcma_f32(odds_1.3, tw3);
                odds_2.3 = vfcmulq_b_conj_fcma_f32(odds_2.3, tw3);

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
                odds_2.0 = vcaddq_rot90_f32(vdupq_n_f32(0.), odds_2.0);
                odds_2.1 = vcaddq_rot90_f32(vdupq_n_f32(0.), odds_2.1);
                odds_2.2 = vcaddq_rot90_f32(vdupq_n_f32(0.), odds_2.2);
                odds_2.3 = vcaddq_rot90_f32(vdupq_n_f32(0.), odds_2.3);

                let zy0 = vaddq_f32(evens.0, odds_1.0);
                let zy1 = vaddq_f32(evens.1, odds_1.1);
                let zy2 = vaddq_f32(evens.2, odds_1.2);
                let zy3 = vaddq_f32(evens.3, odds_1.3);
                let zy4 = vaddq_f32(evens.4, odds_2.0);
                let zy5 = vaddq_f32(evens.5, odds_2.1);
                let zy6 = vaddq_f32(evens.6, odds_2.2);
                let zy7 = vaddq_f32(evens.7, odds_2.3);
                let zy8 = vsubq_f32(evens.0, odds_1.0);
                let zy9 = vsubq_f32(evens.1, odds_1.1);
                let zy10 = vsubq_f32(evens.2, odds_1.2);
                let zy11 = vsubq_f32(evens.3, odds_1.3);
                let zy12 = vsubq_f32(evens.4, odds_2.0);
                let zy13 = vsubq_f32(evens.5, odds_2.1);
                let zy14 = vsubq_f32(evens.6, odds_2.2);
                let zy15 = vsubq_f32(evens.7, odds_2.3);

                let (y0, y8) = vqtrnq_f32(zy0, zy1);
                let (y1, y9) = vqtrnq_f32(zy2, zy3);
                let (y2, y10) = vqtrnq_f32(zy4, zy5);
                let (y3, y11) = vqtrnq_f32(zy6, zy7);
                let (y4, y12) = vqtrnq_f32(zy8, zy9);
                let (y5, y13) = vqtrnq_f32(zy10, zy11);
                let (y6, y14) = vqtrnq_f32(zy12, zy13);
                let (y7, y15) = vqtrnq_f32(zy14, zy15);

                vst1q_f32(chunk.as_mut_ptr().cast(), y0);
                vst1q_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y1);

                vst1q_f32(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y2);
                vst1q_f32(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y3);

                vst1q_f32(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y4);
                vst1q_f32(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y5);

                vst1q_f32(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y6);
                vst1q_f32(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), y7);

                vst1q_f32(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), y8);
                vst1q_f32(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), y9);

                vst1q_f32(chunk.get_unchecked_mut(20..).as_mut_ptr().cast(), y10);
                vst1q_f32(chunk.get_unchecked_mut(22..).as_mut_ptr().cast(), y11);

                vst1q_f32(chunk.get_unchecked_mut(24..).as_mut_ptr().cast(), y12);
                vst1q_f32(chunk.get_unchecked_mut(26..).as_mut_ptr().cast(), y13);

                vst1q_f32(chunk.get_unchecked_mut(28..).as_mut_ptr().cast(), y14);
                vst1q_f32(chunk.get_unchecked_mut(30..).as_mut_ptr().cast(), y15);
            }

            let rem = in_place.chunks_exact_mut(32).into_remainder();

            for chunk in rem.chunks_exact_mut(16) {
                let u0u1 = vld1q_f32(chunk.as_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());

                let (u0, u1) = (vget_low_f32(u0u1), vget_high_f32(u0u1));
                let (u2, u3) = (vget_low_f32(u2u3), vget_high_f32(u2u3));
                let (u4, u5) = (vget_low_f32(u4u5), vget_high_f32(u4u5));
                let (u6, u7) = (vget_low_f32(u6u7), vget_high_f32(u6u7));
                let (u8, u9) = (vget_low_f32(u8u9), vget_high_f32(u8u9));
                let (u10, u11) = (vget_low_f32(u10u11), vget_high_f32(u10u11));
                let (u12, u13) = (vget_low_f32(u12u13), vget_high_f32(u12u13));
                let (u14, u15) = (vget_low_f32(u14u15), vget_high_f32(u14u15));

                let evens = self.bf8.backwardh(u0, u2, u4, u6, u8, u10, u12, u14);

                let mut odds_1 = NeonButterfly::bf4h_backward_f32(u1, u5, u9, u13);
                let mut odds_2 = NeonButterfly::bf4h_backward_f32(u15, u3, u7, u11);

                let o1 = vfcmulq_fcma_f32(
                    vcombine_f32(odds_1.1, odds_1.2),
                    vcombine_f32(vget_low_f32(tw1), vget_low_f32(tw2)),
                );

                let o2 = vfcmulq_b_conj_fcma_f32(
                    vcombine_f32(odds_2.1, odds_2.2),
                    vcombine_f32(vget_low_f32(tw1), vget_low_f32(tw2)),
                );

                odds_1.1 = vget_low_f32(o1);
                odds_2.1 = vget_low_f32(o2);

                odds_1.2 = vget_high_f32(o1);
                odds_2.2 = vget_high_f32(o2);

                odds_1.3 = vfcmul_fcma_f32(odds_1.3, vget_low_f32(tw3));
                odds_2.3 = vfcmul_b_conj_fcma_f32(odds_2.3, vget_low_f32(tw3));

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

                let y0 = vadd_f32(evens.0, odds_1.0);
                let y1 = vadd_f32(evens.1, odds_1.1);
                let y2 = vadd_f32(evens.2, odds_1.2);
                let y3 = vadd_f32(evens.3, odds_1.3);
                let y4 = vadd_f32(evens.4, odds_2.0);
                let y5 = vadd_f32(evens.5, odds_2.1);
                let y6 = vadd_f32(evens.6, odds_2.2);
                let y7 = vadd_f32(evens.7, odds_2.3);
                let y8 = vsub_f32(evens.0, odds_1.0);
                let y9 = vsub_f32(evens.1, odds_1.1);
                let y10 = vsub_f32(evens.2, odds_1.2);
                let y11 = vsub_f32(evens.3, odds_1.3);
                let y12 = vsub_f32(evens.4, odds_2.0);
                let y13 = vsub_f32(evens.5, odds_2.1);
                let y14 = vsub_f32(evens.6, odds_2.2);
                let y15 = vsub_f32(evens.7, odds_2.3);

                vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y0, y1));
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y2, y3),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y4, y5),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y6, y7),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y8, y9),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(y10, y11),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vcombine_f32(y12, y13),
                );

                vst1q_f32(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vcombine_f32(y14, y15),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Butterfly16;
    use crate::neon::NeonButterfly16;
    use rand::Rng;

    #[test]
    fn test_butterfly16_f32() {
        for i in 1..5 {
            let size = 16usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix16_reference = Butterfly16::new(FftDirection::Forward);
            let radix16_inv_reference = Butterfly16::new(FftDirection::Inverse);

            let radix_forward = NeonButterfly16::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly16::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix16_reference.execute(&mut z_ref).unwrap();

            input.iter().zip(z_ref.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}, reference failed",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}, reference failed",
                    a.im,
                    b.im,
                    size
                );
            });

            radix_inverse.execute(&mut input).unwrap();
            radix16_inv_reference.execute(&mut z_ref).unwrap();

            input.iter().zip(z_ref.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}, reference inv failed",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}, reference inv failed",
                    a.im,
                    b.im,
                    size
                );
            });

            input = input.iter().map(|&x| x * (1.0 / 16f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly16_f64() {
        for i in 1..5 {
            let size = 16usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix10_reference = Butterfly16::new(FftDirection::Forward);

            let radix_forward = NeonFcmaButterfly16::new(FftDirection::Forward);
            let radix_inverse = NeonFcmaButterfly16::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();

            radix10_reference.execute(&mut z_ref).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {}, reference failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {}, reference failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 16f64)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }
}
