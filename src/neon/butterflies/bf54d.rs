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

macro_rules! gen_bf54d {
    ($name: ident, $features: literal, $internal_bf6: ident, $internal_bf9: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf6;
        use crate::neon::mixed::$internal_bf9;
        pub(crate) struct $name {
            direction: FftDirection,
            bf6: $internal_bf6,
            bf9: $internal_bf9,
            twiddles: [NeonStoreD; 45],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(9, 6, fft_direction, 54),
                    bf6: $internal_bf6::new(fft_direction),
                    bf9: $internal_bf9::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f64, 54);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
                let mut rows: [NeonStoreD; 6] = [NeonStoreD::default(); 6];
                let mut rows9: [NeonStoreD; 9] = [NeonStoreD::default(); 9];
                let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 54];

                unsafe {
                    // columns
                    for k in 0..9 {
                        for i in 0..6 {
                            rows[i] = NeonStoreD::from_complex_ref(chunk.slice_from(i * 9 + k..));
                        }

                        rows = self.bf6.exec(rows);

                        for i in 1..6 {
                            rows[i] = NeonStoreD::$mul(rows[i], self.twiddles[i - 1 + 5 * k]);
                        }

                        for i in 0..6 {
                            rows[i].write_uninit(scratch.get_unchecked_mut(k * 6 + i..));
                        }
                    }

                    // rows

                    for k in 0..6 {
                        for i in 0..9 {
                            rows9[i] =
                                NeonStoreD::from_complex_refu(scratch.get_unchecked(i * 6 + k..));
                        }
                        rows9 = self.bf9.exec(rows9);
                        for i in 0..9 {
                            rows9[i].write(chunk.slice_from_mut(i * 6 + k..));
                        }
                    }
                }
            }
        }
    };
}

gen_bf54d!(
    NeonButterfly54d,
    "neon",
    ColumnButterfly6d,
    ColumnButterfly9d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf54d!(
    NeonFcmaButterfly54d,
    "fcma",
    ColumnFcmaButterfly6d,
    ColumnFcmaButterfly9d,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly54_f64, f64, NeonButterfly54d, 54, 1e-7);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly54_f64,
        f64,
        NeonFcmaButterfly54d,
        54,
        1e-7
    );
}
