/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
#![allow(clippy::needless_range_loop)]
use crate::neon::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::neon_transpose_f32x2_6x6_aos;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::vst1q_f32;
use std::sync::Arc;

macro_rules! gen_bf36f {
    ($name: ident, $feature: literal, $internal_bf: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            twiddles: [NeonStoreF; 15],
            bf6_column: $internal_bf,
        }

        impl $name {
            pub fn new(direction: FftDirection) -> Self {
                Self {
                    direction,
                    twiddles: gen_butterfly_twiddles_f32(6, 6, direction, 36),
                    bf6_column: $internal_bf::new(direction),
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

            fn length(&self) -> usize {
                36
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(36) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0 = [NeonStoreF::default(); 6];
                    let mut rows1 = [NeonStoreF::default(); 6];
                    let mut rows2 = [NeonStoreF::default(); 6];

                    for chunk in in_place.chunks_exact_mut(36) {
                        // Mixed Radix 6x6
                        for i in 0..6 {
                            rows0[i] =
                                NeonStoreF::load(chunk.get_unchecked(i * 6..).as_ptr().cast());
                        }
                        rows0 = self.bf6_column.exec(rows0);
                        for i in 1..6 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                        }

                        for i in 0..6 {
                            rows1[i] =
                                NeonStoreF::load(chunk.get_unchecked(i * 6 + 2..).as_ptr().cast());
                        }
                        rows1 = self.bf6_column.exec(rows1);
                        for i in 1..6 {
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i + 4]);
                        }

                        for i in 0..6 {
                            rows2[i] =
                                NeonStoreF::load(chunk.get_unchecked(i * 6 + 4..).as_ptr().cast());
                        }
                        rows2 = self.bf6_column.exec(rows2);
                        for i in 1..6 {
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i + 9]);
                        }

                        let (transposed0, transposed1, transposed2) =
                            neon_transpose_f32x2_6x6_aos(rows0, rows1, rows2);

                        let output0 = self.bf6_column.exec(transposed0);
                        for r in 0..3 {
                            vst1q_f32(
                                chunk.get_unchecked_mut(12 * r..).as_mut_ptr().cast(),
                                output0[r * 2].v,
                            );
                            vst1q_f32(
                                chunk.get_unchecked_mut(12 * r + 6..).as_mut_ptr().cast(),
                                output0[r * 2 + 1].v,
                            );
                        }

                        let output1 = self.bf6_column.exec(transposed1);
                        for r in 0..3 {
                            vst1q_f32(
                                chunk.get_unchecked_mut(12 * r + 2..).as_mut_ptr().cast(),
                                output1[r * 2].v,
                            );
                            vst1q_f32(
                                chunk.get_unchecked_mut(12 * r + 8..).as_mut_ptr().cast(),
                                output1[r * 2 + 1].v,
                            );
                        }

                        let output2 = self.bf6_column.exec(transposed2);
                        for r in 0..3 {
                            vst1q_f32(
                                chunk.get_unchecked_mut(12 * r + 4..).as_mut_ptr().cast(),
                                output2[r * 2].v,
                            );
                            vst1q_f32(
                                chunk.get_unchecked_mut(12 * r + 10..).as_mut_ptr().cast(),
                                output2[r * 2 + 1].v,
                            );
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

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_out_of_place_impl(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(36) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(36) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    let mut rows0 = [NeonStoreF::default(); 6];
                    let mut rows1 = [NeonStoreF::default(); 6];
                    let mut rows2 = [NeonStoreF::default(); 6];

                    for (dst, src) in dst.chunks_exact_mut(36).zip(src.chunks_exact(36)) {
                        // Mixed Radix 6x6
                        for i in 0..6 {
                            rows0[i] = NeonStoreF::load(src.get_unchecked(i * 6..).as_ptr().cast());
                        }
                        rows0 = self.bf6_column.exec(rows0);
                        for i in 1..6 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                        }

                        for i in 0..6 {
                            rows1[i] =
                                NeonStoreF::load(src.get_unchecked(i * 6 + 2..).as_ptr().cast());
                        }
                        rows1 = self.bf6_column.exec(rows1);
                        for i in 1..6 {
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i + 4]);
                        }

                        for i in 0..6 {
                            rows2[i] =
                                NeonStoreF::load(src.get_unchecked(i * 6 + 4..).as_ptr().cast());
                        }
                        rows2 = self.bf6_column.exec(rows2);
                        for i in 1..6 {
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i + 9]);
                        }

                        let (transposed0, transposed1, transposed2) =
                            neon_transpose_f32x2_6x6_aos(rows0, rows1, rows2);

                        let output0 = self.bf6_column.exec(transposed0);
                        for r in 0..3 {
                            vst1q_f32(
                                dst.get_unchecked_mut(12 * r..).as_mut_ptr().cast(),
                                output0[r * 2].v,
                            );
                            vst1q_f32(
                                dst.get_unchecked_mut(12 * r + 6..).as_mut_ptr().cast(),
                                output0[r * 2 + 1].v,
                            );
                        }

                        let output1 = self.bf6_column.exec(transposed1);
                        for r in 0..3 {
                            vst1q_f32(
                                dst.get_unchecked_mut(12 * r + 2..).as_mut_ptr().cast(),
                                output1[r * 2].v,
                            );
                            vst1q_f32(
                                dst.get_unchecked_mut(12 * r + 8..).as_mut_ptr().cast(),
                                output1[r * 2 + 1].v,
                            );
                        }

                        let output2 = self.bf6_column.exec(transposed2);
                        for r in 0..3 {
                            vst1q_f32(
                                dst.get_unchecked_mut(12 * r + 4..).as_mut_ptr().cast(),
                                output2[r * 2].v,
                            );
                            vst1q_f32(
                                dst.get_unchecked_mut(12 * r + 10..).as_mut_ptr().cast(),
                                output2[r * 2 + 1].v,
                            );
                        }
                    }
                }
                Ok(())
            }
        }

        impl CompositeFftExecutor<f32> for $name {
            fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
                self
            }
        }
    };
}

gen_bf36f!(NeonButterfly36f, "neon", ColumnButterfly6f, mul_by_complex);
#[cfg(feature = "fcma")]
gen_bf36f!(
    NeonFcmaButterfly36f,
    "fcma",
    ColumnFcmaButterfly6f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly36, f32, NeonButterfly36f, 36, 1e-4);
    test_oof_butterfly!(test_oof_neon_butterfly36, f32, NeonButterfly36f, 36, 1e-4);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly36, f32, NeonFcmaButterfly36f, 36, 1e-4);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly36,
        f32,
        NeonFcmaButterfly36f,
        36,
        1e-4
    );
}
