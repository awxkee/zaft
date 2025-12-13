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

use crate::neon::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::transpose_2x6;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf66f {
    ($name: ident, $feature: literal, $internal_bf6: ident, $internal_bf11: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf6;
        use crate::neon::mixed::$internal_bf11;
        pub(crate) struct $name {
            direction: FftDirection,
            bf6: $internal_bf6,
            bf11: $internal_bf11,
            twiddles: [NeonStoreF; 30],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(11, 6, fft_direction, 66),
                    bf6: $internal_bf6::new(fft_direction),
                    bf11: $internal_bf11::new(fft_direction),
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
                66
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(66) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows: [NeonStoreF; 6] = [NeonStoreF::default(); 6];
                    let mut rows11: [NeonStoreF; 11] = [NeonStoreF::default(); 11];
                    let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 66];

                    for chunk in in_place.chunks_exact_mut(66) {
                        // columns
                        for k in 0..5 {
                            for i in 0..6 {
                                rows[i] = NeonStoreF::from_complex_ref(
                                    chunk.get_unchecked(i * 11 + k * 2..),
                                );
                            }

                            rows = self.bf6.exec(rows);

                            for i in 1..6 {
                                rows[i] = NeonStoreF::$mul(rows[i], self.twiddles[i - 1 + 5 * k]);
                            }

                            let transposed = transpose_2x6(rows);

                            for i in 0..3 {
                                transposed[i * 2]
                                    .write_uninit(scratch.get_unchecked_mut(k * 2 * 6 + i * 2..));
                                transposed[i * 2 + 1].write_uninit(
                                    scratch.get_unchecked_mut((k * 2 + 1) * 6 + i * 2..),
                                );
                            }
                        }

                        {
                            let k = 5;
                            for i in 0..6 {
                                rows[i] =
                                    NeonStoreF::from_complex(chunk.get_unchecked(i * 11 + k * 2));
                            }

                            rows = self.bf6.exec(rows);

                            for i in 1..6 {
                                rows[i] = NeonStoreF::$mul(rows[i], self.twiddles[i - 1 + 5 * k]);
                            }

                            let transposed = transpose_2x6(rows);

                            for i in 0..3 {
                                transposed[i * 2]
                                    .write_uninit(scratch.get_unchecked_mut(k * 2 * 6 + i * 2..));
                            }
                        }

                        // rows

                        for k in 0..3 {
                            for i in 0..11 {
                                rows11[i] = NeonStoreF::from_complex_refu(
                                    scratch.get_unchecked(i * 6 + k * 2..),
                                );
                            }
                            rows11 = self.bf11.exec(rows11);
                            for i in 0..11 {
                                rows11[i].write(chunk.get_unchecked_mut(i * 6 + k * 2..));
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf66f!(
    NeonButterfly66f,
    "neon",
    ColumnButterfly6f,
    ColumnButterfly11f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf66f!(
    NeonFcmaButterfly66f,
    "fcma",
    ColumnFcmaButterfly6f,
    ColumnFcmaButterfly11f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly66, f32, NeonButterfly66f, 66, 1e-3);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly66, f32, NeonFcmaButterfly66f, 66, 1e-3);
}
