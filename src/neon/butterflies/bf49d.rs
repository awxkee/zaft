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

macro_rules! gen_bf49d {
    ($name: ident, $feature: literal, $internal_bf: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf7: $internal_bf,
            twiddles: [NeonStoreD; 42],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(7, 7, fft_direction, 49),
                    bf7: $internal_bf::new(fft_direction),
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
                49
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if in_place.len() % 49 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows: [NeonStoreD; 7] = [NeonStoreD::default(); 7];
                    let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 49];

                    for chunk in in_place.chunks_exact_mut(49) {
                        // columns
                        for k in 0..7 {
                            for i in 0..7 {
                                rows[i] =
                                    NeonStoreD::from_complex_ref(chunk.get_unchecked(i * 7 + k..));
                            }

                            rows = self.bf7.exec(rows);

                            for i in 1..7 {
                                rows[i] = NeonStoreD::mul_by_complex(
                                    rows[i],
                                    self.twiddles[i - 1 + 6 * k],
                                );
                            }

                            for i in 0..7 {
                                rows[i].write_uninit(scratch.get_unchecked_mut(k * 7 + i..));
                            }
                        }

                        // rows

                        for k in 0..7 {
                            for i in 0..7 {
                                rows[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 7 + k..),
                                );
                            }
                            rows = self.bf7.exec(rows);
                            for i in 0..7 {
                                rows[i].write(chunk.get_unchecked_mut(i * 7 + k..));
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
                if src.len() % 49 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if dst.len() % 49 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                let mut rows: [NeonStoreD; 7] = [NeonStoreD::default(); 7];
                let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 49];

                unsafe {
                    for (dst, src) in dst.chunks_exact_mut(49).zip(src.chunks_exact(49)) {
                        // columns
                        for k in 0..7 {
                            for i in 0..7 {
                                rows[i] =
                                    NeonStoreD::from_complex_ref(src.get_unchecked(i * 7 + k..));
                            }

                            rows = self.bf7.exec(rows);

                            for i in 1..7 {
                                rows[i] = NeonStoreD::$mul(rows[i], self.twiddles[i - 1 + 6 * k]);
                            }

                            for i in 0..7 {
                                rows[i].write_uninit(scratch.get_unchecked_mut(k * 7 + i..));
                            }
                        }

                        // rows

                        for k in 0..7 {
                            for i in 0..7 {
                                rows[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 7 + k..),
                                );
                            }
                            rows = self.bf7.exec(rows);
                            for i in 0..7 {
                                rows[i].write(dst.get_unchecked_mut(i * 7 + k..));
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

gen_bf49d!(NeonButterfly49d, "neon", ColumnButterfly7d, mul_by_complex);
#[cfg(feature = "fcma")]
gen_bf49d!(
    NeonFcmaButterfly49d,
    "fcma",
    ColumnFcmaButterfly7d,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly49_f64, f64, NeonButterfly49d, 49, 1e-7);
    test_oof_butterfly!(
        test_oof_neon_butterfly49_f64,
        f64,
        NeonButterfly49d,
        49,
        1e-7
    );

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly49_f64,
        f64,
        NeonFcmaButterfly49d,
        49,
        1e-7
    );
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly49_f64,
        f64,
        NeonFcmaButterfly49d,
        49,
        1e-7
    );
}
