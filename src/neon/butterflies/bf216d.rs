/*
 * // Copyright (c) Radzivon Bartoshyk 1/2026. All rights reserved.
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

use crate::FftExecutorOutOfPlace;
use crate::neon::butterflies::shared::gen_butterfly_twiddles_f64;
use crate::neon::mixed::NeonStoreD;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;
use std::sync::Arc;

macro_rules! gen_bf216d {
    ($name: ident, $feature: literal, $internal_bf18: ident, $internal_bf12: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf12, $internal_bf18};
        pub(crate) struct $name {
            direction: FftDirection,
            bf12: $internal_bf12,
            bf18: $internal_bf18,
            twiddles: [NeonStoreD; 198],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(18, 12, fft_direction, 216),
                    bf12: $internal_bf12::new(fft_direction),
                    bf18: $internal_bf18::new(fft_direction),
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
                216
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(216) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows12: [NeonStoreD; 12] = [NeonStoreD::default(); 12];
                    let mut rows18: [NeonStoreD; 18] = [NeonStoreD::default(); 18];

                    let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 216];

                    for chunk in in_place.chunks_exact_mut(216) {
                        // columns
                        for k in 0..18 {
                            for i in 0..12 {
                                rows12[i] =
                                    NeonStoreD::from_complex_ref(chunk.get_unchecked(i * 18 + k..));
                            }

                            rows12 = self.bf12.exec(rows12);

                            for i in 1..12 {
                                rows12[i] =
                                    NeonStoreD::$mul(rows12[i], self.twiddles[i - 1 + 11 * k]);
                            }

                            for i in 0..12 {
                                rows12[i].write_uninit(scratch.get_unchecked_mut(k * 12 + i..));
                            }
                        }

                        // rows

                        for k in 0..12 {
                            for i in 0..18 {
                                rows18[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 12 + k..),
                                );
                            }
                            rows18 = self.bf18.exec(rows18);
                            for i in 0..18 {
                                rows18[i].write(chunk.get_unchecked_mut(i * 12 + k..));
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
                if !src.len().is_multiple_of(216) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(216) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    let mut rows12: [NeonStoreD; 12] = [NeonStoreD::default(); 12];
                    let mut rows18: [NeonStoreD; 18] = [NeonStoreD::default(); 18];

                    let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 216];

                    for (dst, src) in dst.chunks_exact_mut(216).zip(src.chunks_exact(216)) {
                        // columns
                        for k in 0..18 {
                            for i in 0..12 {
                                rows12[i] =
                                    NeonStoreD::from_complex_ref(src.get_unchecked(i * 18 + k..));
                            }

                            rows12 = self.bf12.exec(rows12);

                            for i in 1..12 {
                                rows12[i] =
                                    NeonStoreD::$mul(rows12[i], self.twiddles[i - 1 + 11 * k]);
                            }

                            for i in 0..12 {
                                rows12[i].write_uninit(scratch.get_unchecked_mut(k * 12 + i..));
                            }
                        }

                        // rows

                        for k in 0..12 {
                            for i in 0..18 {
                                rows18[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 12 + k..),
                                );
                            }
                            rows18 = self.bf18.exec(rows18);
                            for i in 0..18 {
                                rows18[i].write(dst.get_unchecked_mut(i * 12 + k..));
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

gen_bf216d!(
    NeonButterfly216d,
    "neon",
    ColumnButterfly18d,
    ColumnButterfly12d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf216d!(
    NeonFcmaButterfly216d,
    "fcma",
    ColumnFcmaButterfly18d,
    ColumnFcmaButterfly12d,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(
        test_neon_butterfly216_f64,
        f64,
        NeonButterfly216d,
        216,
        1e-7
    );

    test_oof_butterfly!(
        test_neon_oof_butterfly216_f64,
        f64,
        NeonButterfly216d,
        216,
        1e-7
    );

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly216_f64,
        f64,
        NeonFcmaButterfly216d,
        216,
        1e-7
    );
}
