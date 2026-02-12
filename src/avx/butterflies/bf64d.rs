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

use crate::avx::butterflies::shared::boring_avx_butterfly;
use crate::avx::mixed::{AvxStoreD, ColumnButterfly8d};
use crate::avx::transpose::transpose_f64x2_2x8;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly64d {
    direction: FftDirection,
    bf8: ColumnButterfly8d,
    twiddles: [AvxStoreD; 28],
}

impl AvxButterfly64d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        let mut twiddles = [AvxStoreD::zero(); 28];
        let mut q = 0usize;
        let len_per_row = 8;
        const COMPLEX_PER_VECTOR: usize = 2;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..8 {
                twiddles[q] = AvxStoreD::set_complex2(
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 64, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 64, fft_direction),
                );
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf8: ColumnButterfly8d::new(fft_direction),
        }
    }
}

boring_avx_butterfly!(AvxButterfly64d, f64, 64);

impl AvxButterfly64d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows: [AvxStoreD; 8] = [AvxStoreD::zero(); 8];
        let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 64];
        unsafe {
            // columns
            for k in 0..4 {
                for i in 0..8 {
                    rows[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 8 + k * 2..));
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

            // rows

            for k in 0..4 {
                for i in 0..8 {
                    rows[i] = AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 8 + k * 2..));
                }
                rows = self.bf8.exec(rows);
                for i in 0..8 {
                    rows[i].write(chunk.slice_from_mut(i * 8 + k * 2..));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly64, f64, AvxButterfly64d, 64, 1e-7);
    test_oof_avx_butterfly!(test_oof_avx_butterfly64, f64, AvxButterfly64d, 64, 1e-7);
}
