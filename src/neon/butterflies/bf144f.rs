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
use crate::neon::transpose::transpose_2x9;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf128f {
    ($name: ident, $feature: literal, $internal_bf16: ident, $internal_bf9: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf9, $internal_bf16};

        pub(crate) struct $name {
            direction: FftDirection,
            bf16: $internal_bf16,
            bf9: $internal_bf9,
            twiddles: [NeonStoreF; 64],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(16, 9, fft_direction, 144),
                    bf16: $internal_bf16::new(fft_direction),
                    bf9: $internal_bf9::new(fft_direction),
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
                144
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if in_place.len() % 144 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows: [NeonStoreF; 9] = [NeonStoreF::default(); 9];
                    let mut rows16: [NeonStoreF; 16] = [NeonStoreF::default(); 16];
                    let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 144];

                    for chunk in in_place.chunks_exact_mut(144) {
                        // columns
                        for k in 0..8 {
                            for i in 0..9 {
                                rows[i] = NeonStoreF::from_complex_ref(
                                    chunk.get_unchecked(i * 16 + k * 2..),
                                );
                            }

                            rows = self.bf9.exec(rows);

                            for i in 1..9 {
                                rows[i] = NeonStoreF::$mul(rows[i], self.twiddles[i - 1 + 8 * k]);
                            }

                            let transposed = transpose_2x9(rows);

                            for i in 0..4 {
                                transposed[i * 2]
                                    .write_uninit(scratch.get_unchecked_mut(k * 2 * 9 + i * 2..));
                                transposed[i * 2 + 1].write_uninit(
                                    scratch.get_unchecked_mut((k * 2 + 1) * 9 + i * 2..),
                                );
                            }
                            transposed[8].write_lo_u(scratch.get_unchecked_mut(k * 2 * 9 + 8..));
                            transposed[9]
                                .write_lo_u(scratch.get_unchecked_mut((k * 2 + 1) * 9 + 8..));
                        }

                        // rows

                        for k in 0..4 {
                            for i in 0..16 {
                                rows16[i] = NeonStoreF::from_complex_refu(
                                    scratch.get_unchecked(i * 9 + k * 2..),
                                );
                            }
                            rows16 = self.bf16.exec(rows16);
                            for i in 0..16 {
                                rows16[i].write(chunk.get_unchecked_mut(i * 9 + k * 2..));
                            }
                        }

                        {
                            let k = 4;
                            for i in 0..16 {
                                rows16[i] =
                                    NeonStoreF::from_complexu(scratch.get_unchecked(i * 9 + k * 2));
                            }
                            rows16 = self.bf16.exec(rows16);
                            for i in 0..16 {
                                rows16[i].write_lo(chunk.get_unchecked_mut(i * 9 + k * 2..));
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf128f!(
    NeonButterfly144f,
    "neon",
    ColumnButterfly16f,
    ColumnButterfly9f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf128f!(
    NeonFcmaButterfly144f,
    "fcma",
    ColumnFcmaButterfly16f,
    ColumnFcmaButterfly9f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly144, f32, NeonButterfly144f, 144, 1e-3);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly144,
        f32,
        NeonFcmaButterfly144f,
        144,
        1e-3
    );
}
