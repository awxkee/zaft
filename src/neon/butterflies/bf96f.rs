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

use crate::neon::butterflies::shared::{boring_neon_butterfly, gen_butterfly_twiddles_f32};
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::transpose_2x8;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf96f {
    ($name: ident, $features: literal, $internal_bf8: ident, $internal_bf12: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf8;
        use crate::neon::mixed::$internal_bf12;
        pub(crate) struct $name {
            direction: FftDirection,
            bf8: $internal_bf8,
            bf12: $internal_bf12,
            twiddles: [NeonStoreF; 42],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(12, 8, fft_direction, 96),
                    bf8: $internal_bf8::new(fft_direction),
                    bf12: $internal_bf12::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 96);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows: [NeonStoreF; 8] = [NeonStoreF::default(); 8];
                let mut rows12: [NeonStoreF; 12] = [NeonStoreF::default(); 12];
                let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 96];
                unsafe {
                    // columns
                    for k in 0..6 {
                        for i in 0..8 {
                            rows[i] =
                                NeonStoreF::from_complex_ref(chunk.slice_from(i * 12 + k * 2..));
                        }

                        rows = self.bf8.exec(rows);

                        for i in 1..8 {
                            rows[i] = NeonStoreF::$mul(rows[i], self.twiddles[i - 1 + 7 * k]);
                        }

                        let transposed = transpose_2x8(rows);

                        for i in 0..4 {
                            transposed[i * 2]
                                .write_uninit(scratch.get_unchecked_mut(k * 2 * 8 + i * 2..));
                            transposed[i * 2 + 1]
                                .write_uninit(scratch.get_unchecked_mut((k * 2 + 1) * 8 + i * 2..));
                        }
                    }

                    // rows

                    for k in 0..4 {
                        for i in 0..12 {
                            rows12[i] = NeonStoreF::from_complex_refu(
                                scratch.get_unchecked(i * 8 + k * 2..),
                            );
                        }
                        rows12 = self.bf12.exec(rows12);
                        for i in 0..12 {
                            rows12[i].write(chunk.slice_from_mut(i * 8 + k * 2..));
                        }
                    }
                }
            }
        }
    };
}

gen_bf96f!(
    NeonButterfly96f,
    "neon",
    ColumnButterfly8f,
    ColumnButterfly12f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf96f!(
    NeonFcmaButterfly96f,
    "fcma",
    ColumnFcmaButterfly8f,
    ColumnFcmaButterfly12f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly96, f32, NeonButterfly96f, 96, 1e-3);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly96, f32, NeonFcmaButterfly96f, 96, 1e-3);
}
