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

use crate::neon::mixed::{ColumnButterfly5d, ColumnButterfly7d, NeonStoreD};
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct NeonButterfly35d {
    direction: FftDirection,
    bf5: ColumnButterfly5d,
    bf7: ColumnButterfly7d,
    twiddles: [NeonStoreD; 28],
}

impl NeonButterfly35d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let mut twiddles = [NeonStoreD::default(); 28];
        let mut q = 0usize;
        let len_per_row = 7;
        const COMPLEX_PER_VECTOR: usize = 1;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..5 {
                twiddles[q] = NeonStoreD::from_complex(&compute_twiddle(
                    y * (x * COMPLEX_PER_VECTOR),
                    35,
                    fft_direction,
                ));
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf7: ColumnButterfly7d::new(fft_direction),
            bf5: ColumnButterfly5d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for NeonButterfly35d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        self.execute_impl(in_place)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        35
    }
}

impl NeonButterfly35d {
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 35 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [NeonStoreD; 5] = [NeonStoreD::default(); 5];
            let mut rows7: [NeonStoreD; 7] = [NeonStoreD::default(); 7];

            let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 42];

            for chunk in in_place.chunks_exact_mut(35) {
                // columns
                for k in 0..7 {
                    rows0[0] = NeonStoreD::from_complex_ref(chunk.get_unchecked(k..));
                    rows0[1] = NeonStoreD::from_complex_ref(chunk.get_unchecked(7 + k..));
                    rows0[2] = NeonStoreD::from_complex_ref(chunk.get_unchecked(2 * 7 + k..));
                    rows0[3] = NeonStoreD::from_complex_ref(chunk.get_unchecked(3 * 7 + k..));
                    rows0[4] = NeonStoreD::from_complex_ref(chunk.get_unchecked(4 * 7 + k..));

                    rows0 = self.bf5.exec(rows0);

                    for i in 1..5 {
                        rows0[i] =
                            NeonStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1 + 4 * k]);
                    }

                    rows0[0].write_uninit(scratch.get_unchecked_mut(k * 5..));
                    rows0[1].write_uninit(scratch.get_unchecked_mut(k * 5 + 1..));
                    rows0[2].write_uninit(scratch.get_unchecked_mut(k * 5 + 2..));
                    rows0[3].write_uninit(scratch.get_unchecked_mut(k * 5 + 3..));
                    rows0[4].write_uninit(scratch.get_unchecked_mut(k * 5 + 4..));
                }

                // rows

                for k in 0..5 {
                    for i in 0..7 {
                        rows7[i] =
                            NeonStoreD::from_complex_refu(scratch.get_unchecked(i * 5 + k..));
                    }
                    rows7 = self.bf7.exec(rows7);
                    for i in 0..7 {
                        rows7[i].write(chunk.get_unchecked_mut(i * 5 + k..));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;

    test_butterfly!(test_neon_butterfly35_f64, f64, NeonButterfly35d, 35, 1e-7);
}
