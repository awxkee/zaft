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

macro_rules! gen_bf30d {
    ($name: ident, $feature: literal, $internal_bf5: ident, $internal_bf6: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf5, $internal_bf6};
        pub(crate) struct $name {
            direction: FftDirection,
            bf5: $internal_bf5,
            bf6: $internal_bf6,
            twiddles: [NeonStoreD; 24],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(6, 5, fft_direction, 30),
                    bf6: $internal_bf6::new(fft_direction),
                    bf5: $internal_bf5::new(fft_direction),
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
                30
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if in_place.len() % 30 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreD; 5] = [NeonStoreD::default(); 5];
                    let mut rows6: [NeonStoreD; 6] = [NeonStoreD::default(); 6];

                    let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 30];

                    for chunk in in_place.chunks_exact_mut(30) {
                        // columns
                        for k in 0..6 {
                            rows0[0] = NeonStoreD::from_complex_ref(chunk.get_unchecked(k..));
                            rows0[1] = NeonStoreD::from_complex_ref(chunk.get_unchecked(6 + k..));
                            rows0[2] =
                                NeonStoreD::from_complex_ref(chunk.get_unchecked(2 * 6 + k..));
                            rows0[3] =
                                NeonStoreD::from_complex_ref(chunk.get_unchecked(3 * 6 + k..));
                            rows0[4] =
                                NeonStoreD::from_complex_ref(chunk.get_unchecked(4 * 6 + k..));

                            rows0 = self.bf5.exec(rows0);

                            for i in 1..5 {
                                rows0[i] = NeonStoreD::$mul(rows0[i], self.twiddles[i - 1 + 4 * k]);
                            }

                            rows0[0].write_uninit(scratch.get_unchecked_mut(k * 5..));
                            rows0[1].write_uninit(scratch.get_unchecked_mut(k * 5 + 1..));
                            rows0[2].write_uninit(scratch.get_unchecked_mut(k * 5 + 2..));
                            rows0[3].write_uninit(scratch.get_unchecked_mut(k * 5 + 3..));
                            rows0[4].write_uninit(scratch.get_unchecked_mut(k * 5 + 4..));
                        }

                        // rows

                        for k in 0..5 {
                            for i in 0..6 {
                                rows6[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 5 + k..),
                                );
                            }
                            rows6 = self.bf6.exec(rows6);
                            for i in 0..6 {
                                rows6[i].write(chunk.get_unchecked_mut(i * 5 + k..));
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf30d!(
    NeonButterfly30d,
    "neon",
    ColumnButterfly5d,
    ColumnButterfly6d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf30d!(
    NeonFcmaButterfly30d,
    "fcma",
    ColumnFcmaButterfly5d,
    ColumnFcmaButterfly6d,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly35_f64, f64, NeonButterfly30d, 30, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly35_f64,
        f64,
        NeonFcmaButterfly30d,
        30,
        1e-7
    );
}
