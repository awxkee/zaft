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

use crate::avx::butterflies::shared::{boring_avx_butterfly, gen_butterfly_twiddles_f64};
use crate::avx::mixed::{AvxStoreD, ColumnButterfly8d, ColumnButterfly11d};
use crate::avx::transpose::transpose_f64x2_2x8;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly88d {
    direction: FftDirection,
    bf8: ColumnButterfly8d,
    bf11: ColumnButterfly11d,
    twiddles: [AvxStoreD; 42],
}

impl AvxButterfly88d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(11, 8, fft_direction, 88),
            bf8: ColumnButterfly8d::new(fft_direction),
            bf11: ColumnButterfly11d::new(fft_direction),
        }
    }
}

boring_avx_butterfly!(AvxButterfly88d, f64, 88);

impl AvxButterfly88d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows: [AvxStoreD; 8] = [AvxStoreD::zero(); 8];
        let mut rows11: [AvxStoreD; 11] = [AvxStoreD::zero(); 11];
        let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 88];
        unsafe {
            // columns
            for k in 0..5 {
                for i in 0..8 {
                    rows[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 11 + k * 2..));
                }

                rows = self.bf8.exec(rows);

                for i in 1..8 {
                    rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 7 * k]);
                }

                let transposed = transpose_f64x2_2x8(rows);

                for i in 0..4 {
                    transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 8 + i * 2..));
                    transposed[i * 2 + 1]
                        .write_u(scratch.get_unchecked_mut((k * 2 + 1) * 8 + i * 2..));
                }
            }

            {
                let k = 5;
                for i in 0..8 {
                    rows[i] = AvxStoreD::from_complex(chunk.index(i * 11 + k * 2));
                }

                rows = self.bf8.exec(rows);

                for i in 1..8 {
                    rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 7 * k]);
                }

                let transposed = transpose_f64x2_2x8(rows);

                for i in 0..4 {
                    transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 8 + i * 2..));
                }
            }

            // rows

            for k in 0..4 {
                for i in 0..11 {
                    rows11[i] =
                        AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 8 + k * 2..));
                }
                rows11 = self.bf11.exec(rows11);
                for i in 0..11 {
                    rows11[i].write(chunk.slice_from_mut(i * 8 + k * 2..));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly88_f64, f64, AvxButterfly88d, 88, 1e-3);
}
