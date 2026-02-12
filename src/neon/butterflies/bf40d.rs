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

use crate::neon::butterflies::shared::{boring_neon_butterfly, gen_butterfly_twiddles_f64};
use crate::neon::mixed::NeonStoreD;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf40d {
    ($name: ident, $features: literal, $internal_bf5: ident, $internal_bf8: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf5, $internal_bf8};
        pub(crate) struct $name {
            direction: FftDirection,
            bf5: $internal_bf5,
            bf8: $internal_bf8,
            twiddles: [NeonStoreD; 32],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(8, 5, fft_direction, 40),
                    bf8: $internal_bf8::new(fft_direction),
                    bf5: $internal_bf5::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f64, 40);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
                let mut rows0: [NeonStoreD; 5] = [NeonStoreD::default(); 5];
                let mut rows8: [NeonStoreD; 8] = [NeonStoreD::default(); 8];

                let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 40];

                unsafe {
                    // columns
                    for k in 0..8 {
                        for i in 0..5 {
                            rows0[i] = NeonStoreD::from_complex_ref(chunk.slice_from(i * 8 + k..));
                        }

                        rows0 = self.bf5.exec(rows0);

                        for i in 1..5 {
                            rows0[i] = NeonStoreD::$mul(rows0[i], self.twiddles[i - 1 + 4 * k]);
                        }

                        for i in 0..5 {
                            rows0[i].write_uninit(scratch.get_unchecked_mut(k * 5 + i..));
                        }
                    }

                    // rows

                    for k in 0..5 {
                        for i in 0..8 {
                            rows8[i] =
                                NeonStoreD::from_complex_refu(scratch.get_unchecked(i * 5 + k..));
                        }
                        rows8 = self.bf8.exec(rows8);
                        for i in 0..8 {
                            rows8[i].write(chunk.slice_from_mut(i * 5 + k..));
                        }
                    }
                }
            }
        }
    };
}

gen_bf40d!(
    NeonButterfly40d,
    "neon",
    ColumnButterfly5d,
    ColumnButterfly8d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf40d!(
    NeonFcmaButterfly40d,
    "fcma",
    ColumnFcmaButterfly5d,
    ColumnFcmaButterfly8d,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly40_f64, f64, NeonButterfly40d, 40, 1e-7);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly40_f64,
        f64,
        NeonFcmaButterfly40d,
        40,
        1e-7
    );
}
