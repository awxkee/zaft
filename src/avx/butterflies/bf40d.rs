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
use crate::avx::mixed::{AvxStoreD, ColumnButterfly5d, ColumnButterfly8d};
use crate::avx::transpose::transpose_f64x2_2x5;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly40d {
    direction: FftDirection,
    bf5: ColumnButterfly5d,
    bf8: ColumnButterfly8d,
    twiddles: [AvxStoreD; 16],
}

impl AvxButterfly40d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(8, 5, fft_direction, 40),
            bf8: ColumnButterfly8d::new(fft_direction),
            bf5: ColumnButterfly5d::new(fft_direction),
        }
    }
}

boring_avx_butterfly!(AvxButterfly40d, f64, 40);

impl AvxButterfly40d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows0: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];
        let mut rows8: [AvxStoreD; 8] = [AvxStoreD::zero(); 8];
        let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 40];
        unsafe {
            // columns
            for k in 0..4 {
                for i in 0..5 {
                    rows0[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 8 + k * 2..));
                }

                rows0 = self.bf5.exec(rows0);

                for i in 1..5 {
                    rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1 + 4 * k]);
                }

                let transposed = transpose_f64x2_2x5(rows0);

                for i in 0..2 {
                    transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 5 + i * 2..));
                    transposed[i * 2 + 1]
                        .write_u(scratch.get_unchecked_mut((k * 2 + 1) * 5 + i * 2..));
                }

                {
                    let i = 2;
                    transposed[4].write_lou(scratch.get_unchecked_mut(k * 2 * 5 + i * 2..));
                    transposed[5].write_lou(scratch.get_unchecked_mut((k * 2 + 1) * 5 + i * 2..));
                }
            }

            // rows

            for k in 0..2 {
                for i in 0..8 {
                    rows8[i] = AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 5 + k * 2..));
                }
                rows8 = self.bf8.exec(rows8);
                for i in 0..8 {
                    rows8[i].write(chunk.slice_from_mut(i * 5 + k * 2..));
                }
            }

            {
                let k = 2;
                for i in 0..8 {
                    rows8[i] = AvxStoreD::from_complexu(scratch.get_unchecked(i * 5 + k * 2));
                }
                rows8 = self.bf8.exec(rows8);
                for i in 0..8 {
                    rows8[i].write_lo(chunk.slice_from_mut(i * 5 + k * 2..));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly40, f64, AvxButterfly40d, 40, 1e-3);
}
