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

use crate::avx::butterflies::shared::{
    boring_avx_butterfly, boring_avx512vl_butterfly, gen_butterfly_twiddles_f64,
};
use crate::avx::mixed::{AvxStoreD, ColumnButterfly16d};
use crate::avx::transpose::transpose_f64x2_2x16;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! define_bf256 {
    ($bf_name: ident, $features: literal) => {
        pub(crate) struct $bf_name {
            direction: FftDirection,
            bf16: ColumnButterfly16d,
            twiddles: [AvxStoreD; 120],
        }

        impl $bf_name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                unsafe { Self::new_init(fft_direction) }
            }

            #[target_feature(enable = $features)]
            fn new_init(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(16, 16, fft_direction, 256),
                    bf16: ColumnButterfly16d::new(fft_direction),
                }
            }
        }

        impl $bf_name {
            #[target_feature(enable = $features)]
            fn exec_bf16(&self, src: &[Complex<f64>], dst: &mut [MaybeUninit<Complex<f64>>; 256]) {
                let mut rows: [AvxStoreD; 16] = [AvxStoreD::zero(); 16];
                // columns
                for k in 0..8 {
                    for i in 0..16 {
                        rows[i] = unsafe {
                            AvxStoreD::from_complex_ref(src.get_unchecked(i * 16 + k * 2..))
                        };
                    }

                    rows = self.bf16.exec(rows);

                    for i in 1..16 {
                        rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 15 * k]);
                    }

                    let transposed = transpose_f64x2_2x16(rows);

                    for i in 0..8 {
                        unsafe {
                            transposed[i * 2].write_u(dst.get_unchecked_mut(k * 2 * 16 + i * 2..));
                            transposed[i * 2 + 1]
                                .write_u(dst.get_unchecked_mut((k * 2 + 1) * 16 + i * 2..));
                        }
                    }
                }
            }

            #[target_feature(enable = $features)]
            fn exec_bf16_2(
                &self,
                src: &[MaybeUninit<Complex<f64>>; 256],
                dst: &mut [Complex<f64>],
            ) {
                let mut rows: [AvxStoreD; 16] = [AvxStoreD::zero(); 16];
                for k in 0..8 {
                    for i in 0..16 {
                        unsafe {
                            rows[i] =
                                AvxStoreD::from_complex_refu(src.get_unchecked(i * 16 + k * 2..));
                        }
                    }
                    rows = self.bf16.exec(rows);
                    for i in 0..16 {
                        unsafe {
                            rows[i].write(dst.get_unchecked_mut(i * 16 + k * 2..));
                        }
                    }
                }
            }

            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
                let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 256];
                self.exec_bf16(chunk.slice_from(0..), &mut scratch);
                // rows
                self.exec_bf16_2(&scratch, chunk.slice_from_mut(0..));
            }
        }
    };
}

define_bf256!(AvxButterfly256d, "avx2,fma");
define_bf256!(Avx512vlButterfly256d, "avx2,fma,avx512f,avx512vl");

boring_avx_butterfly!(AvxButterfly256d, f64, 256);
boring_avx512vl_butterfly!(Avx512vlButterfly256d, f64, 256);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly256_f64, f64, AvxButterfly256d, 256, 1e-8);
    test_oof_avx_butterfly!(
        test_oof_avx_butterfly256_f64,
        f64,
        AvxButterfly256d,
        256,
        1e-8
    );
}
