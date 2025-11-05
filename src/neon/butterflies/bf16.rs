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
    conj_f32, v_rotate90_f32, v_rotate90_f64, vfcmul_conj_b_f32, vfcmul_f32, vfcmulq_conj_b_f32,
    vfcmulq_conj_b_f64, vfcmulq_f32, vfcmulq_f64, vh_rotate90_f32, vqtrnq_f32,
};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonButterfly16<T> {
    direction: FftDirection,
    bf8: NeonFastButterfly8<T>,
    twiddle1: [T; 4],
    twiddle2: [T; 4],
    twiddle3: [T; 4],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonButterfly16<T>
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

impl FftExecutor<f64> for NeonButterfly16<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 16 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let rot_sign = vld1q_f64(match self.direction {
                FftDirection::Inverse => [-0.0, 0.0].as_ptr(),
                FftDirection::Forward => [0.0, -0.0].as_ptr(),
            });
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

                let evens = self.bf8.exec(u0, u2, u4, u6, u8, u10, u12, u14, rot_sign);

                let mut odds_1 = NeonButterfly::butterfly4_f64(u1, u5, u9, u13, rot_sign);
                let mut odds_2 = NeonButterfly::butterfly4_f64(u15, u3, u7, u11, rot_sign);

                odds_1.1 = vfcmulq_f64(odds_1.1, tw1);
                odds_2.1 = vfcmulq_conj_b_f64(odds_2.1, tw1);

                odds_1.2 = vfcmulq_f64(odds_1.2, tw2);
                odds_2.2 = vfcmulq_conj_b_f64(odds_2.2, tw2);

                odds_1.3 = vfcmulq_f64(odds_1.3, tw3);
                odds_2.3 = vfcmulq_conj_b_f64(odds_2.3, tw3);

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
                odds_2.0 = v_rotate90_f64(odds_2.0, rot_sign);
                odds_2.1 = v_rotate90_f64(odds_2.1, rot_sign);
                odds_2.2 = v_rotate90_f64(odds_2.2, rot_sign);
                odds_2.3 = v_rotate90_f64(odds_2.3, rot_sign);

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

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        16
    }
}

impl FftExecutorOutOfPlace<f64> for NeonButterfly16<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 16 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 16 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let rot_sign = vld1q_f64(match self.direction {
                FftDirection::Inverse => [-0.0, 0.0].as_ptr(),
                FftDirection::Forward => [0.0, -0.0].as_ptr(),
            });
            let tw1 = vld1q_f64(self.twiddle1.as_ptr());
            let tw2 = vld1q_f64(self.twiddle2.as_ptr());
            let tw3 = vld1q_f64(self.twiddle3.as_ptr());

            for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
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

                let evens = self.bf8.exec(u0, u2, u4, u6, u8, u10, u12, u14, rot_sign);

                let mut odds_1 = NeonButterfly::butterfly4_f64(u1, u5, u9, u13, rot_sign);
                let mut odds_2 = NeonButterfly::butterfly4_f64(u15, u3, u7, u11, rot_sign);

                odds_1.1 = vfcmulq_f64(odds_1.1, tw1);
                odds_2.1 = vfcmulq_conj_b_f64(odds_2.1, tw1);

                odds_1.2 = vfcmulq_f64(odds_1.2, tw2);
                odds_2.2 = vfcmulq_conj_b_f64(odds_2.2, tw2);

                odds_1.3 = vfcmulq_f64(odds_1.3, tw3);
                odds_2.3 = vfcmulq_conj_b_f64(odds_2.3, tw3);

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
                odds_2.0 = v_rotate90_f64(odds_2.0, rot_sign);
                odds_2.1 = v_rotate90_f64(odds_2.1, rot_sign);
                odds_2.2 = v_rotate90_f64(odds_2.2, rot_sign);
                odds_2.3 = v_rotate90_f64(odds_2.3, rot_sign);

                vst1q_f64(dst.as_mut_ptr().cast(), vaddq_f64(evens.0, odds_1.0));
                vst1q_f64(
                    dst.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vaddq_f64(evens.1, odds_1.1),
                );

                vst1q_f64(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vaddq_f64(evens.2, odds_1.2),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vaddq_f64(evens.3, odds_1.3),
                );

                vst1q_f64(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vaddq_f64(evens.4, odds_2.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    vaddq_f64(evens.5, odds_2.1),
                );

                vst1q_f64(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vaddq_f64(evens.6, odds_2.2),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(7..).as_mut_ptr().cast(),
                    vaddq_f64(evens.7, odds_2.3),
                );

                vst1q_f64(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vsubq_f64(evens.0, odds_1.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(9..).as_mut_ptr().cast(),
                    vsubq_f64(evens.1, odds_1.1),
                );

                vst1q_f64(
                    dst.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vsubq_f64(evens.2, odds_1.2),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(11..).as_mut_ptr().cast(),
                    vsubq_f64(evens.3, odds_1.3),
                );

                vst1q_f64(
                    dst.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vsubq_f64(evens.4, odds_2.0),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(13..).as_mut_ptr().cast(),
                    vsubq_f64(evens.5, odds_2.1),
                );

                vst1q_f64(
                    dst.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vsubq_f64(evens.6, odds_2.2),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(15..).as_mut_ptr().cast(),
                    vsubq_f64(evens.7, odds_2.3),
                );
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for NeonButterfly16<f64> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for NeonButterfly16<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 16 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let rot_sign = vld1q_f32(match self.direction {
                FftDirection::Inverse => [-0.0, 0.0, -0.0, 0.0].as_ptr(),
                FftDirection::Forward => [0.0, -0.0, 0.0, -0.0].as_ptr(),
            });
            let tw1 = vld1q_f32(self.twiddle1.as_ptr());
            let tw2 = vld1q_f32(self.twiddle2.as_ptr());
            let tw3 = vld1q_f32(self.twiddle3.as_ptr());

            let conj_factors = vld1q_f32([0.0, -0.0, 0.0, -0.0].as_ptr());

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

                let evens = self.bf8.exec(u0, u2, u4, u6, u8, u10, u12, u14, rot_sign);

                let mut odds_1 = NeonButterfly::butterfly4_f32(u1, u5, u9, u13, rot_sign);
                let mut odds_2 = NeonButterfly::butterfly4_f32(u15, u3, u7, u11, rot_sign);

                odds_1.1 = vfcmulq_f32(odds_1.1, tw1);
                odds_2.1 = vfcmulq_conj_b_f32(odds_2.1, tw1);

                odds_1.2 = vfcmulq_f32(odds_1.2, tw2);
                odds_2.2 = vfcmulq_conj_b_f32(odds_2.2, tw2);

                odds_1.3 = vfcmulq_f32(odds_1.3, tw3);
                odds_2.3 = vfcmulq_conj_b_f32(odds_2.3, tw3);

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
                odds_2.0 = v_rotate90_f32(odds_2.0, rot_sign);
                odds_2.1 = v_rotate90_f32(odds_2.1, rot_sign);
                odds_2.2 = v_rotate90_f32(odds_2.2, rot_sign);
                odds_2.3 = v_rotate90_f32(odds_2.3, rot_sign);

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

                let evens =
                    self.bf8
                        .exech(u0, u2, u4, u6, u8, u10, u12, u14, vget_low_f32(rot_sign));

                let mut odds_1 =
                    NeonButterfly::butterfly4h_f32(u1, u5, u9, u13, vget_low_f32(rot_sign));
                let mut odds_2 =
                    NeonButterfly::butterfly4h_f32(u15, u3, u7, u11, vget_low_f32(rot_sign));

                let o0 = vfcmulq_f32(
                    vcombine_f32(odds_1.1, odds_2.1),
                    vcombine_f32(
                        vget_low_f32(tw1),
                        conj_f32(vget_low_f32(tw1), vget_low_f32(conj_factors)),
                    ),
                );

                let o1 = vfcmulq_f32(
                    vcombine_f32(odds_1.2, odds_2.2),
                    vcombine_f32(
                        vget_low_f32(tw2),
                        conj_f32(vget_low_f32(tw2), vget_low_f32(conj_factors)),
                    ),
                );

                odds_1.1 = vget_low_f32(o0);
                odds_2.1 = vget_high_f32(o0);

                odds_1.2 = vget_low_f32(o1);
                odds_2.2 = vget_high_f32(o1);

                odds_1.3 = vfcmul_f32(odds_1.3, vget_low_f32(tw3));
                odds_2.3 = vfcmul_conj_b_f32(odds_2.3, vget_low_f32(tw3));

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
                odds_2.0 = vh_rotate90_f32(odds_2.0, vget_low_f32(rot_sign));
                odds_2.1 = vh_rotate90_f32(odds_2.1, vget_low_f32(rot_sign));
                odds_2.2 = vh_rotate90_f32(odds_2.2, vget_low_f32(rot_sign));
                odds_2.3 = vh_rotate90_f32(odds_2.3, vget_low_f32(rot_sign));

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

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        16
    }
}

impl FftExecutorOutOfPlace<f32> for NeonButterfly16<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 16 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }

        if dst.len() % 16 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let rot_sign = vld1q_f32(match self.direction {
                FftDirection::Inverse => [-0.0, 0.0, -0.0, 0.0].as_ptr(),
                FftDirection::Forward => [0.0, -0.0, 0.0, -0.0].as_ptr(),
            });
            let tw1 = vld1q_f32(self.twiddle1.as_ptr());
            let tw2 = vld1q_f32(self.twiddle2.as_ptr());
            let tw3 = vld1q_f32(self.twiddle3.as_ptr());

            let conj_factors = vld1q_f32([0.0, -0.0, 0.0, -0.0].as_ptr());

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

                let (u0, u1) = vqtrnq_f32(u0u1, u16u17);
                let (u2, u3) = vqtrnq_f32(u2u3, u18u19);
                let (u4, u5) = vqtrnq_f32(u4u5, u20u21);
                let (u6, u7) = vqtrnq_f32(u6u7, u22u23);
                let (u8, u9) = vqtrnq_f32(u8u9, u24u25);
                let (u10, u11) = vqtrnq_f32(u10u11, u26u27);
                let (u12, u13) = vqtrnq_f32(u12u13, u28u29);
                let (u14, u15) = vqtrnq_f32(u14u15, u30u31);

                let evens = self.bf8.exec(u0, u2, u4, u6, u8, u10, u12, u14, rot_sign);

                let mut odds_1 = NeonButterfly::butterfly4_f32(u1, u5, u9, u13, rot_sign);
                let mut odds_2 = NeonButterfly::butterfly4_f32(u15, u3, u7, u11, rot_sign);

                odds_1.1 = vfcmulq_f32(odds_1.1, tw1);
                odds_2.1 = vfcmulq_conj_b_f32(odds_2.1, tw1);

                odds_1.2 = vfcmulq_f32(odds_1.2, tw2);
                odds_2.2 = vfcmulq_conj_b_f32(odds_2.2, tw2);

                odds_1.3 = vfcmulq_f32(odds_1.3, tw3);
                odds_2.3 = vfcmulq_conj_b_f32(odds_2.3, tw3);

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
                odds_2.0 = v_rotate90_f32(odds_2.0, rot_sign);
                odds_2.1 = v_rotate90_f32(odds_2.1, rot_sign);
                odds_2.2 = v_rotate90_f32(odds_2.2, rot_sign);
                odds_2.3 = v_rotate90_f32(odds_2.3, rot_sign);

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

                vst1q_f32(dst.as_mut_ptr().cast(), y0);
                vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y1);

                vst1q_f32(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y2);
                vst1q_f32(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), y3);

                vst1q_f32(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), y4);
                vst1q_f32(dst.get_unchecked_mut(10..).as_mut_ptr().cast(), y5);

                vst1q_f32(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), y6);
                vst1q_f32(dst.get_unchecked_mut(14..).as_mut_ptr().cast(), y7);

                vst1q_f32(dst.get_unchecked_mut(16..).as_mut_ptr().cast(), y8);
                vst1q_f32(dst.get_unchecked_mut(18..).as_mut_ptr().cast(), y9);

                vst1q_f32(dst.get_unchecked_mut(20..).as_mut_ptr().cast(), y10);
                vst1q_f32(dst.get_unchecked_mut(22..).as_mut_ptr().cast(), y11);

                vst1q_f32(dst.get_unchecked_mut(24..).as_mut_ptr().cast(), y12);
                vst1q_f32(dst.get_unchecked_mut(26..).as_mut_ptr().cast(), y13);

                vst1q_f32(dst.get_unchecked_mut(28..).as_mut_ptr().cast(), y14);
                vst1q_f32(dst.get_unchecked_mut(30..).as_mut_ptr().cast(), y15);
            }

            let rem_src = src.chunks_exact(32).remainder();
            let rem_dst = dst.chunks_exact_mut(32).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(16).zip(rem_src.chunks_exact(16)) {
                let u0u1 = vld1q_f32(src.as_ptr().cast());
                let u2u3 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(src.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(src.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(src.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(src.get_unchecked(14..).as_ptr().cast());

                let (u0, u1) = (vget_low_f32(u0u1), vget_high_f32(u0u1));
                let (u2, u3) = (vget_low_f32(u2u3), vget_high_f32(u2u3));
                let (u4, u5) = (vget_low_f32(u4u5), vget_high_f32(u4u5));
                let (u6, u7) = (vget_low_f32(u6u7), vget_high_f32(u6u7));
                let (u8, u9) = (vget_low_f32(u8u9), vget_high_f32(u8u9));
                let (u10, u11) = (vget_low_f32(u10u11), vget_high_f32(u10u11));
                let (u12, u13) = (vget_low_f32(u12u13), vget_high_f32(u12u13));
                let (u14, u15) = (vget_low_f32(u14u15), vget_high_f32(u14u15));

                let evens =
                    self.bf8
                        .exech(u0, u2, u4, u6, u8, u10, u12, u14, vget_low_f32(rot_sign));

                let mut odds_1 =
                    NeonButterfly::butterfly4h_f32(u1, u5, u9, u13, vget_low_f32(rot_sign));
                let mut odds_2 =
                    NeonButterfly::butterfly4h_f32(u15, u3, u7, u11, vget_low_f32(rot_sign));

                let o0 = vfcmulq_f32(
                    vcombine_f32(odds_1.1, odds_2.1),
                    vcombine_f32(
                        vget_low_f32(tw1),
                        conj_f32(vget_low_f32(tw1), vget_low_f32(conj_factors)),
                    ),
                );

                let o1 = vfcmulq_f32(
                    vcombine_f32(odds_1.2, odds_2.2),
                    vcombine_f32(
                        vget_low_f32(tw2),
                        conj_f32(vget_low_f32(tw2), vget_low_f32(conj_factors)),
                    ),
                );

                odds_1.1 = vget_low_f32(o0);
                odds_2.1 = vget_high_f32(o0);

                odds_1.2 = vget_low_f32(o1);
                odds_2.2 = vget_high_f32(o1);

                odds_1.3 = vfcmul_f32(odds_1.3, vget_low_f32(tw3));
                odds_2.3 = vfcmul_conj_b_f32(odds_2.3, vget_low_f32(tw3));

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
                odds_2.0 = vh_rotate90_f32(odds_2.0, vget_low_f32(rot_sign));
                odds_2.1 = vh_rotate90_f32(odds_2.1, vget_low_f32(rot_sign));
                odds_2.2 = vh_rotate90_f32(odds_2.2, vget_low_f32(rot_sign));
                odds_2.3 = vh_rotate90_f32(odds_2.3, vget_low_f32(rot_sign));

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

                vst1q_f32(dst.as_mut_ptr().cast(), vcombine_f32(y0, y1));
                vst1q_f32(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y2, y3),
                );

                vst1q_f32(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y4, y5),
                );

                vst1q_f32(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y6, y7),
                );

                vst1q_f32(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(y8, y9),
                );

                vst1q_f32(
                    dst.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(y10, y11),
                );

                vst1q_f32(
                    dst.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vcombine_f32(y12, y13),
                );

                vst1q_f32(
                    dst.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    vcombine_f32(y14, y15),
                );
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for NeonButterfly16<f32> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    use rand::Rng;

    test_butterfly!(test_neon_butterfly16, f32, NeonButterfly16, 16, 1e-5);
    test_butterfly!(test_neon_butterfly16_f64, f64, NeonButterfly16, 16, 1e-7);
    test_oof_butterfly!(test_oof_butterfly16, f32, NeonButterfly16, 16, 1e-5);
    test_oof_butterfly!(test_oof_butterfly16_f64, f64, NeonButterfly16, 16, 1e-9);
}
