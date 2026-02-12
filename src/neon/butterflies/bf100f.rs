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

use crate::neon::butterflies::shared::{boring_neon_butterfly, gen_butterfly_twiddles_f32};
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::transpose_2x10;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf100f {
    ($name: ident, $features: literal, $internal_bf: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf10: $internal_bf,
            twiddles: [NeonStoreF; 45],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(10, 10, fft_direction, 100),
                    bf10: $internal_bf::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 100);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows: [NeonStoreF; 10] = [NeonStoreF::default(); 10];
                let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 100];
                unsafe {
                    // columns
                    for k in 0..5 {
                        for i in 0..10 {
                            rows[i] =
                                NeonStoreF::from_complex_ref(chunk.slice_from(i * 10 + k * 2..));
                        }

                        rows = self.bf10.exec(rows);

                        for i in 1..10 {
                            rows[i] = NeonStoreF::$mul(rows[i], self.twiddles[i - 1 + 9 * k]);
                        }

                        let transposed = transpose_2x10(rows);

                        for i in 0..5 {
                            transposed[i * 2]
                                .write_uninit(scratch.get_unchecked_mut(k * 2 * 10 + i * 2..));
                            transposed[i * 2 + 1].write_uninit(
                                scratch.get_unchecked_mut((k * 2 + 1) * 10 + i * 2..),
                            );
                        }
                    }

                    // rows

                    for k in 0..5 {
                        for i in 0..10 {
                            rows[i] = NeonStoreF::from_complex_refu(
                                scratch.get_unchecked(i * 10 + k * 2..),
                            );
                        }
                        rows = self.bf10.exec(rows);
                        for i in 0..10 {
                            rows[i].write(chunk.slice_from_mut(i * 10 + k * 2..));
                        }
                    }
                }
            }
        }
    };
}

gen_bf100f!(
    NeonButterfly100f,
    "neon",
    ColumnButterfly10f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf100f!(
    NeonFcmaButterfly100f,
    "fcma",
    ColumnFcmaButterfly10f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly100, f32, NeonButterfly100f, 100, 1e-3);
    test_oof_butterfly!(
        test_oof_neon_butterfly100,
        f32,
        NeonButterfly100f,
        100,
        1e-3
    );

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly100,
        f32,
        NeonFcmaButterfly100f,
        100,
        1e-3
    );
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly100,
        f32,
        NeonFcmaButterfly100f,
        100,
        1e-3
    );
}
