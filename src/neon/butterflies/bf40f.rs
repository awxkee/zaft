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
use crate::neon::transpose::neon_transpose_f32x2_8x5_aos;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

macro_rules! gen_bf40f {
    ($name: ident, $feature: literal, $internal_bf5: ident, $internal_bf8: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf5, $internal_bf8};
        pub(crate) struct $name {
            direction: FftDirection,
            bf5: $internal_bf5,
            bf8: $internal_bf8,
            twiddles: [NeonStoreF; 16],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(8, 5, fft_direction, 40),
                    bf8: $internal_bf8::new(fft_direction),
                    bf5: $internal_bf5::new(fft_direction),
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
                40
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(40) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 5] = [NeonStoreF::default(); 5];
                    let mut rows1: [NeonStoreF; 5] = [NeonStoreF::default(); 5];
                    let mut rows2: [NeonStoreF; 5] = [NeonStoreF::default(); 5];
                    let mut rows3: [NeonStoreF; 5] = [NeonStoreF::default(); 5];

                    for chunk in in_place.chunks_exact_mut(40) {
                        // columns
                        for i in 0..5 {
                            rows0[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 8..));
                            rows1[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 8 + 2..));
                            rows2[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 8 + 4..));
                            rows3[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 8 + 6..));
                        }

                        rows0 = self.bf5.exec(rows0);
                        rows1 = self.bf5.exec(rows1);
                        rows2 = self.bf5.exec(rows2);
                        rows3 = self.bf5.exec(rows3);

                        for i in 1..5 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 4]);
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 8]);
                            rows3[i] = NeonStoreF::$mul(rows3[i], self.twiddles[i - 1 + 12]);
                        }

                        let (mut v0, mut v1, mut v2) =
                            neon_transpose_f32x2_8x5_aos(rows0, rows1, rows2, rows3);

                        // rows
                        v0 = self.bf8.exec(v0);
                        for i in 0..8 {
                            v0[i].write(chunk.get_unchecked_mut(i * 5..));
                        }
                        v1 = self.bf8.exec(v1);
                        for i in 0..8 {
                            v1[i].write(chunk.get_unchecked_mut(i * 5 + 2..));
                        }
                        v2 = self.bf8.exec(v2);
                        for i in 0..8 {
                            v2[i].write_lo(chunk.get_unchecked_mut(i * 5 + 4..));
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf40f!(
    NeonButterfly40f,
    "neon",
    ColumnButterfly5f,
    ColumnButterfly8f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf40f!(
    NeonFcmaButterfly40f,
    "fcma",
    ColumnFcmaButterfly5f,
    ColumnFcmaButterfly8f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly40, f32, NeonButterfly40f, 40, 1e-3);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly40, f32, NeonFcmaButterfly40f, 40, 1e-3);
}
