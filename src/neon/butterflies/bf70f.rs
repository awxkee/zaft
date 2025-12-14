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
use crate::neon::transpose::transpose_2x7;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf70f {
    ($name: ident, $feature: literal, $internal_bf7: ident, $internal_bf10: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf7;
        use crate::neon::mixed::$internal_bf10;
        pub(crate) struct $name {
            direction: FftDirection,
            bf7: $internal_bf7,
            bf10: $internal_bf10,
            twiddles: [NeonStoreF; 30],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(10, 7, fft_direction, 70),
                    bf7: $internal_bf7::new(fft_direction),
                    bf10: $internal_bf10::new(fft_direction),
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
                70
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(70) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows: [NeonStoreF; 7] = [NeonStoreF::default(); 7];
                    let mut rows10: [NeonStoreF; 10] = [NeonStoreF::default(); 10];
                    let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 70];

                    for chunk in in_place.chunks_exact_mut(70) {
                        // columns
                        for k in 0..5 {
                            for i in 0..7 {
                                rows[i] = NeonStoreF::from_complex_ref(
                                    chunk.get_unchecked(i * 10 + k * 2..),
                                );
                            }

                            rows = self.bf7.exec(rows);

                            for i in 1..7 {
                                rows[i] = NeonStoreF::$mul(rows[i], self.twiddles[i - 1 + 6 * k]);
                            }

                            let transposed = transpose_2x7(rows);

                            for i in 0..3 {
                                transposed[i * 2]
                                    .write_uninit(scratch.get_unchecked_mut(k * 2 * 7 + i * 2..));
                                transposed[i * 2 + 1].write_uninit(
                                    scratch.get_unchecked_mut((k * 2 + 1) * 7 + i * 2..),
                                );
                            }

                            {
                                let i = 3;
                                transposed[i * 2]
                                    .write_lo_u(scratch.get_unchecked_mut(k * 2 * 7 + i * 2..));
                                transposed[i * 2 + 1].write_lo_u(
                                    scratch.get_unchecked_mut((k * 2 + 1) * 7 + i * 2..),
                                );
                            }
                        }

                        // rows

                        for k in 0..3 {
                            for i in 0..10 {
                                rows10[i] = NeonStoreF::from_complex_refu(
                                    scratch.get_unchecked(i * 7 + k * 2..),
                                );
                            }
                            rows10 = self.bf10.exec(rows10);
                            for i in 0..10 {
                                rows10[i].write(chunk.get_unchecked_mut(i * 7 + k * 2..));
                            }
                        }

                        {
                            let k = 3;
                            for i in 0..10 {
                                rows10[i] =
                                    NeonStoreF::from_complexu(scratch.get_unchecked(i * 7 + k * 2));
                            }
                            rows10 = self.bf10.exec(rows10);
                            for i in 0..10 {
                                rows10[i].write_lo(chunk.get_unchecked_mut(i * 7 + k * 2..));
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf70f!(
    NeonButterfly70f,
    "neon",
    ColumnButterfly7f,
    ColumnButterfly10f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf70f!(
    NeonFcmaButterfly70f,
    "fcma",
    ColumnFcmaButterfly7f,
    ColumnFcmaButterfly10f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly70, f32, NeonButterfly70f, 70, 1e-3);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly70, f32, NeonFcmaButterfly70f, 70, 1e-3);
}
