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

use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::neon_transpose_f32x2_7x6_aos;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf42f {
    ($name: ident, $feature: literal, $internal_bf6: ident, $internal_bf7: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf6, $internal_bf7};
        pub(crate) struct $name {
            direction: FftDirection,
            bf6: $internal_bf6,
            bf7: $internal_bf7,
            twiddles: [NeonStoreF; 20],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                let mut twiddles = [NeonStoreF::default(); 20];
                let mut q = 0usize;
                let len_per_row = 7;
                const COMPLEX_PER_VECTOR: usize = 2;
                let quotient = len_per_row / COMPLEX_PER_VECTOR;
                let remainder = len_per_row % COMPLEX_PER_VECTOR;

                let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
                for x in 0..num_twiddle_columns {
                    for y in 1..6 {
                        twiddles[q] = NeonStoreF::from_complex2(
                            compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 42, fft_direction),
                            compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 42, fft_direction),
                        );
                        q += 1;
                    }
                }
                Self {
                    direction: fft_direction,
                    twiddles,
                    bf7: $internal_bf7::new(fft_direction),
                    bf6: $internal_bf6::new(fft_direction),
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
                42
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if in_place.len() % 42 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 6] = [NeonStoreF::default(); 6];
                    let mut rows1: [NeonStoreF; 6] = [NeonStoreF::default(); 6];
                    let mut rows2: [NeonStoreF; 6] = [NeonStoreF::default(); 6];
                    let mut rows3: [NeonStoreF; 6] = [NeonStoreF::default(); 6];

                    let mut rows7: [NeonStoreF; 7] = [NeonStoreF::default(); 7];

                    let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 42];

                    for chunk in in_place.chunks_exact_mut(42) {
                        // columns
                        for i in 0..6 {
                            rows0[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 7..));
                            rows1[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 7 + 2..));
                            rows2[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 7 + 4..));
                            rows3[i] = NeonStoreF::from_complex(chunk.get_unchecked(i * 7 + 6));
                        }

                        rows0 = self.bf6.exec(rows0);
                        rows1 = self.bf6.exec(rows1);
                        rows2 = self.bf6.exec(rows2);
                        rows3 = self.bf6.exec(rows3);

                        for i in 1..6 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 5]);
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 10]);
                            rows3[i] = NeonStoreF::$mul(rows3[i], self.twiddles[i - 1 + 15]);
                        }

                        let (v0, v1, v2) = neon_transpose_f32x2_7x6_aos(rows0, rows1, rows2, rows3);

                        for i in 0..7 {
                            v0[i].write_uninit(scratch.get_unchecked_mut(i * 6..));
                            v1[i].write_uninit(scratch.get_unchecked_mut(i * 6 + 2..));
                            v2[i].write_uninit(scratch.get_unchecked_mut(i * 6 + 4..));
                        }

                        // rows

                        for k in 0..3 {
                            for i in 0..7 {
                                rows7[i] = NeonStoreF::from_complex_refu(
                                    scratch.get_unchecked(i * 6 + k * 2..),
                                );
                            }
                            rows7 = self.bf7.exec(rows7);
                            for i in 0..7 {
                                rows7[i].write(chunk.get_unchecked_mut(i * 6 + k * 2..));
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf42f!(
    NeonButterfly42f,
    "neon",
    ColumnButterfly6f,
    ColumnButterfly7f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf42f!(
    NeonFcmaButterfly42f,
    "fcma",
    ColumnFcmaButterfly6f,
    ColumnFcmaButterfly7f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly42, f32, NeonButterfly42f, 42, 1e-3);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly42, f32, NeonFcmaButterfly42f, 42, 1e-3);
}
