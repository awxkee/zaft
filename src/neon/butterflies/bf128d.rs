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

use crate::neon::butterflies::shared::gen_butterfly_twiddles_f64;
use crate::neon::mixed::NeonStoreD;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;
use std::sync::Arc;

macro_rules! gen_bf128d {
    ($name: ident, $feature: literal, $internal_bf16: ident, $internal_bf8: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf8, $internal_bf16};
        pub(crate) struct $name {
            direction: FftDirection,
            bf16: $internal_bf16,
            bf8: $internal_bf8,
            twiddles: [NeonStoreD; 112],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(16, 8, fft_direction, 128),
                    bf16: $internal_bf16::new(fft_direction),
                    bf8: $internal_bf8::new(fft_direction),
                }
            }
        }

        impl FftExecutor<f64> for $name {
            fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                unsafe { self.execute_impl(in_place) }
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                128
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(128) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows: [NeonStoreD; 8] = [NeonStoreD::default(); 8];
                    let mut rows16: [NeonStoreD; 16] = [NeonStoreD::default(); 16];

                    let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 128];

                    for chunk in in_place.chunks_exact_mut(128) {
                        // columns
                        for k in 0..16 {
                            for i in 0..8 {
                                rows[i] =
                                    NeonStoreD::from_complex_ref(chunk.get_unchecked(i * 16 + k..));
                            }

                            rows = self.bf8.exec(rows);

                            for i in 1..8 {
                                rows[i] = NeonStoreD::$mul(rows[i], self.twiddles[i - 1 + 7 * k]);
                            }

                            for i in 0..8 {
                                rows[i].write_uninit(scratch.get_unchecked_mut(k * 8 + i..));
                            }
                        }

                        // rows

                        for k in 0..8 {
                            for i in 0..16 {
                                rows16[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 8 + k..),
                                );
                            }
                            rows16 = self.bf16.exec(rows16);
                            for i in 0..16 {
                                rows16[i].write(chunk.get_unchecked_mut(i * 8 + k..));
                            }
                        }
                    }
                }
                Ok(())
            }
        }
        impl FftExecutorOutOfPlace<f64> for $name {
            fn execute_out_of_place(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_out_of_place_impl(src, dst) }
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_out_of_place_impl(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(128) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(128) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                let mut rows: [NeonStoreD; 8] = [NeonStoreD::default(); 8];
                let mut rows16: [NeonStoreD; 16] = [NeonStoreD::default(); 16];

                let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 128];

                unsafe {
                    for (dst, src) in dst.chunks_exact_mut(128).zip(src.chunks_exact(128)) {
                        // columns
                        for k in 0..16 {
                            for i in 0..8 {
                                rows[i] =
                                    NeonStoreD::from_complex_ref(src.get_unchecked(i * 16 + k..));
                            }

                            rows = self.bf8.exec(rows);

                            for i in 1..8 {
                                rows[i] = NeonStoreD::$mul(rows[i], self.twiddles[i - 1 + 7 * k]);
                            }

                            for i in 0..8 {
                                rows[i].write_uninit(scratch.get_unchecked_mut(k * 8 + i..));
                            }
                        }

                        // rows

                        for k in 0..8 {
                            for i in 0..16 {
                                rows16[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 8 + k..),
                                );
                            }
                            rows16 = self.bf16.exec(rows16);
                            for i in 0..16 {
                                rows16[i].write(dst.get_unchecked_mut(i * 8 + k..));
                            }
                        }
                    }
                }
                Ok(())
            }
        }

        impl CompositeFftExecutor<f64> for $name {
            fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
                self
            }
        }
    };
}

gen_bf128d!(
    NeonButterfly128d,
    "neon",
    ColumnButterfly16d,
    ColumnButterfly8d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf128d!(
    NeonFcmaButterfly128d,
    "fcma",
    ColumnFcmaButterfly16d,
    ColumnFcmaButterfly8d,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(
        test_neon_butterfly128_f64,
        f64,
        NeonButterfly128d,
        128,
        1e-7
    );
    test_oof_butterfly!(
        test_oof_neon_butterfly128_f64,
        f64,
        NeonButterfly128d,
        128,
        1e-7
    );

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly128_f64,
        f64,
        NeonFcmaButterfly128d,
        128,
        1e-7
    );
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly128_f64,
        f64,
        NeonFcmaButterfly128d,
        128,
        1e-7
    );
}
