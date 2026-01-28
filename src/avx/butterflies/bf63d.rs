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
use crate::avx::mixed::{AvxStoreD, ColumnButterfly7d, ColumnButterfly9d};
use crate::avx::transpose::transpose_f64x2_2x7;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly63d {
    direction: FftDirection,
    bf7: ColumnButterfly7d,
    bf9: ColumnButterfly9d,
    twiddles: [AvxStoreD; 30],
}

impl AvxButterfly63d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(9, 7, fft_direction, 63),
            bf9: ColumnButterfly9d::new(fft_direction),
            bf7: ColumnButterfly7d::new(fft_direction),
        }
    }
}

boring_avx_butterfly!(AvxButterfly63d, f64, 63);

impl AvxButterfly63d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];
        let mut rows9: [AvxStoreD; 9] = [AvxStoreD::zero(); 9];
        let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 63];

        unsafe {
            // columns
            for k in 0..4 {
                for i in 0..7 {
                    rows[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 9 + k * 2..));
                }

                rows = self.bf7.exec(rows);

                for i in 1..7 {
                    rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 6 * k]);
                }

                let transposed = transpose_f64x2_2x7(rows);

                for i in 0..3 {
                    transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 7 + i * 2..));
                    transposed[i * 2 + 1]
                        .write_u(scratch.get_unchecked_mut((k * 2 + 1) * 7 + i * 2..));
                }

                transposed[6].write_lou(scratch.get_unchecked_mut(k * 2 * 7 + 6..));
                transposed[7].write_lou(scratch.get_unchecked_mut((k * 2 + 1) * 7 + 6..));
            }

            {
                let k = 4;
                for i in 0..7 {
                    rows[i] = AvxStoreD::from_complex(chunk.index(i * 9 + k * 2));
                }

                rows = self.bf7.exec(rows);

                for i in 1..7 {
                    rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 6 * k]);
                }

                let transposed = transpose_f64x2_2x7(rows);

                for i in 0..3 {
                    transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 7 + i * 2..));
                }
                transposed[6].write_lou(scratch.get_unchecked_mut(k * 2 * 7 + 6..));
            }

            // rows

            for k in 0..3 {
                for i in 0..9 {
                    rows9[i] = AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 7 + k * 2..));
                }
                rows9 = self.bf9.exec(rows9);
                for i in 0..9 {
                    rows9[i].write(chunk.slice_from_mut(i * 7 + k * 2..));
                }
            }

            {
                let k = 3;
                for i in 0..9 {
                    rows9[i] = AvxStoreD::from_complexu(scratch.get_unchecked(i * 7 + k * 2));
                }
                rows9 = self.bf9.exec(rows9);
                for i in 0..9 {
                    rows9[i].write_lo(chunk.slice_from_mut(i * 7 + k * 2..));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly63_f64, f64, AvxButterfly63d, 63, 1e-3);
}
