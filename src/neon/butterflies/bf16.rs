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
use crate::neon::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use crate::neon::util::{v_rotate90_f64, vfcmulq_conj_b_f64, vfcmulq_f64};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;
use std::sync::Arc;

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
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

#[inline(always)]
pub(crate) fn transpose_4x4(
    rows0: [NeonStoreF; 4],
    rows1: [NeonStoreF; 4],
) -> ([NeonStoreF; 4], [NeonStoreF; 4]) {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[2].v, rows0[3].v));

    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    let e0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[2].v, rows1[3].v));
    (
        [
            NeonStoreF::raw(a0.0),
            NeonStoreF::raw(a0.1),
            NeonStoreF::raw(b0.0),
            NeonStoreF::raw(b0.1),
        ],
        [
            NeonStoreF::raw(d0.0),
            NeonStoreF::raw(d0.1),
            NeonStoreF::raw(e0.0),
            NeonStoreF::raw(e0.1),
        ],
    )
}

macro_rules! gen_bf16f {
    ($name: ident, $features: literal, $internal_bf4: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf4;
        pub(crate) struct $name {
            direction: FftDirection,
            bf4: $internal_bf4,
            twiddles: [NeonStoreF; 6],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(4, 4, fft_direction, 16),
                    bf4: $internal_bf4::new(fft_direction),
                }
            }
        }

        impl FftExecutor<f32> for $name {
            fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                unsafe { self.execute_impl(in_place) }
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                16
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(16) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows1: [NeonStoreF; 4] = [NeonStoreF::default(); 4];

                    for chunk in in_place.chunks_exact_mut(16) {
                        // columns
                        for i in 0..4 {
                            rows0[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 4..));
                            rows1[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 4 + 2..));
                        }

                        rows0 = self.bf4.exec(rows0);
                        rows1 = self.bf4.exec(rows1);

                        for i in 1..4 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 3]);
                        }

                        let transposed = transpose_4x4(rows0, rows1);

                        let q0 = self.bf4.exec(transposed.0);
                        let q1 = self.bf4.exec(transposed.1);

                        for i in 0..4 {
                            q0[i].write(chunk.get_unchecked_mut(i * 4..));
                            q1[i].write(chunk.get_unchecked_mut(i * 4 + 2..));
                        }
                    }
                }
                Ok(())
            }

            #[target_feature(enable = $features)]
            fn execute_out_of_place_impl(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(16) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(16) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows1: [NeonStoreF; 4] = [NeonStoreF::default(); 4];

                    for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
                        // columns
                        for i in 0..4 {
                            rows0[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 4..));
                            rows1[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 4 + 2..));
                        }

                        rows0 = self.bf4.exec(rows0);
                        rows1 = self.bf4.exec(rows1);

                        for i in 1..4 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 3]);
                        }

                        let transposed = transpose_4x4(rows0, rows1);

                        let q0 = self.bf4.exec(transposed.0);
                        let q1 = self.bf4.exec(transposed.1);

                        for i in 0..4 {
                            q0[i].write(dst.get_unchecked_mut(i * 4..));
                            q1[i].write(dst.get_unchecked_mut(i * 4 + 2..));
                        }
                    }
                }
                Ok(())
            }
        }

        impl FftExecutorOutOfPlace<f32> for $name {
            fn execute_out_of_place(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_out_of_place_impl(src, dst) }
            }
        }

        impl CompositeFftExecutor<f32> for $name {
            fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
                self
            }
        }
    };
}

gen_bf16f!(NeonButterfly16f, "neon", ColumnButterfly4f, mul_by_complex);
#[cfg(feature = "fcma")]
gen_bf16f!(
    NeonFcmaButterfly16f,
    "fcma",
    ColumnFcmaButterfly4f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly16, f32, NeonButterfly16f, 16, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly16, f32, NeonButterfly16f, 16, 1e-5);
    test_butterfly!(test_neon_butterfly16_f64, f64, NeonButterfly16, 16, 1e-7);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(test_oof_fcma_butterfly16, f32, NeonButterfly16f, 16, 1e-5);
    test_oof_butterfly!(test_oof_butterfly16, f32, NeonButterfly16f, 16, 1e-5);
    test_oof_butterfly!(test_oof_butterfly16_f64, f64, NeonButterfly16, 16, 1e-9);
}
