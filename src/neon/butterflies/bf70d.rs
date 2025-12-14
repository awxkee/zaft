/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf70d {
    ($name: ident, $feature: literal, $internal_bf7: ident, $internal_bf10: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf7;
        use crate::neon::mixed::$internal_bf10;
        pub(crate) struct $name {
            direction: FftDirection,
            bf7: $internal_bf7,
            bf10: $internal_bf10,
            twiddles: [NeonStoreD; 60],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(10, 7, fft_direction, 70),
                    bf7: $internal_bf7::new(fft_direction),
                    bf10: $internal_bf10::new(fft_direction),
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
                70
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(70) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows: [NeonStoreD; 7] = [NeonStoreD::default(); 7];
                    let mut rows10: [NeonStoreD; 10] = [NeonStoreD::default(); 10];
                    let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 70];

                    for chunk in in_place.chunks_exact_mut(70) {
                        // columns
                        for k in 0..10 {
                            for i in 0..7 {
                                rows[i] =
                                    NeonStoreD::from_complex_ref(chunk.get_unchecked(i * 10 + k..));
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
                            for i in 0..10 {
                                rows10[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 7 + k..),
                                );
                            }
                            rows10 = self.bf10.exec(rows10);
                            for i in 0..10 {
                                rows10[i].write(chunk.get_unchecked_mut(i * 7 + k..));
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf70d!(
    NeonButterfly70d,
    "neon",
    ColumnButterfly7d,
    ColumnButterfly10d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf70d!(
    NeonFcmaButterfly70d,
    "fcma",
    ColumnFcmaButterfly7d,
    ColumnFcmaButterfly10d,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly70_f64, f64, NeonButterfly70d, 70, 1e-7);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly70_f64,
        f64,
        NeonFcmaButterfly70d,
        70,
        1e-7
    );
}
