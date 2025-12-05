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

macro_rules! gen_bf129d {
    ($name: ident, $feature: literal, $internal_bf16: ident, $internal_bf9: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf9, $internal_bf16};
        pub(crate) struct $name {
            direction: FftDirection,
            bf16: $internal_bf16,
            bf9: $internal_bf9,
            twiddles: [NeonStoreD; 128],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(16, 9, fft_direction, 144),
                    bf16: $internal_bf16::new(fft_direction),
                    bf9: $internal_bf9::new(fft_direction),
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
                144
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if in_place.len() % 144 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows: [NeonStoreD; 9] = [NeonStoreD::default(); 9];
                    let mut rows16: [NeonStoreD; 16] = [NeonStoreD::default(); 16];

                    let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 144];

                    for chunk in in_place.chunks_exact_mut(144) {
                        // columns
                        for k in 0..16 {
                            for i in 0..9 {
                                rows[i] =
                                    NeonStoreD::from_complex_ref(chunk.get_unchecked(i * 16 + k..));
                            }

                            rows = self.bf9.exec(rows);

                            for i in 1..9 {
                                rows[i] = NeonStoreD::$mul(rows[i], self.twiddles[i - 1 + 8 * k]);
                            }

                            for i in 0..9 {
                                rows[i].write_uninit(scratch.get_unchecked_mut(k * 9 + i..));
                            }
                        }

                        // rows

                        for k in 0..9 {
                            for i in 0..16 {
                                rows16[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 9 + k..),
                                );
                            }
                            rows16 = self.bf16.exec(rows16);
                            for i in 0..16 {
                                rows16[i].write(chunk.get_unchecked_mut(i * 9 + k..));
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf129d!(
    NeonButterfly144d,
    "neon",
    ColumnButterfly16d,
    ColumnButterfly9d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf129d!(
    NeonFcmaButterfly144d,
    "fcma",
    ColumnFcmaButterfly16d,
    ColumnFcmaButterfly9d,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(
        test_neon_butterfly144_f64,
        f64,
        NeonButterfly144d,
        144,
        1e-7
    );

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly144_f64,
        f64,
        NeonFcmaButterfly144d,
        144,
        1e-7
    );
}
