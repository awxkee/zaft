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
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf42d {
    ($name: ident, $feature: literal, $internal_bf6: ident, $internal_bf7: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf6, $internal_bf7};
        pub(crate) struct $name {
            direction: FftDirection,
            bf6: $internal_bf6,
            bf7: $internal_bf7,
            twiddles: [NeonStoreD; 35],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(7, 6, fft_direction, 42),
                    bf7: $internal_bf7::new(fft_direction),
                    bf6: $internal_bf6::new(fft_direction),
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
                42
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if in_place.len() % 42 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreD; 6] = [NeonStoreD::default(); 6];
                    let mut rows7: [NeonStoreD; 7] = [NeonStoreD::default(); 7];

                    let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 42];

                    for chunk in in_place.chunks_exact_mut(42) {
                        // columns
                        for k in 0..7 {
                            for i in 0..6 {
                                rows0[i] =
                                    NeonStoreD::from_complex_ref(chunk.get_unchecked(i * 7 + k..));
                            }

                            rows0 = self.bf6.exec(rows0);

                            for i in 1..6 {
                                rows0[i] = NeonStoreD::$mul(rows0[i], self.twiddles[i - 1 + 5 * k]);
                            }

                            for i in 0..6 {
                                rows0[i].write_uninit(scratch.get_unchecked_mut(k * 6 + i..));
                            }
                        }

                        // rows

                        for k in 0..6 {
                            for i in 0..7 {
                                rows7[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 6 + k..),
                                );
                            }
                            rows7 = self.bf7.exec(rows7);
                            for i in 0..7 {
                                rows7[i].write(chunk.get_unchecked_mut(i * 6 + k..));
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf42d!(
    NeonButterfly42d,
    "neon",
    ColumnButterfly6d,
    ColumnButterfly7d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf42d!(
    NeonFcmaButterfly42d,
    "fcma",
    ColumnFcmaButterfly6d,
    ColumnFcmaButterfly7d,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly48_f64, f64, NeonButterfly42d, 42, 1e-7);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly48_f64,
        f64,
        NeonFcmaButterfly42d,
        42,
        1e-7
    );
}
