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

use crate::neon::mixed::{ColumnButterfly4d, ColumnButterfly12d, NeonStoreD};
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

pub(crate) struct NeonButterfly48d {
    direction: FftDirection,
    bf4: ColumnButterfly4d,
    bf12: ColumnButterfly12d,
    twiddles: [NeonStoreD; 36],
}

impl NeonButterfly48d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let mut twiddles = [NeonStoreD::default(); 36];
        let mut q = 0usize;
        let len_per_row = 12;
        const COMPLEX_PER_VECTOR: usize = 1;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..4 {
                twiddles[q] = NeonStoreD::from_complex(&compute_twiddle(
                    y * (x * COMPLEX_PER_VECTOR),
                    48,
                    fft_direction,
                ));
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf12: ColumnButterfly12d::new(fft_direction),
            bf4: ColumnButterfly4d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for NeonButterfly48d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        self.execute_impl(in_place)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        48
    }
}

impl NeonButterfly48d {
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 48 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [NeonStoreD; 4] = [NeonStoreD::default(); 4];
            let mut rows1: [NeonStoreD; 4] = [NeonStoreD::default(); 4];
            let mut rows2: [NeonStoreD; 4] = [NeonStoreD::default(); 4];
            let mut rows12: [NeonStoreD; 12] = [NeonStoreD::default(); 12];

            let mut scratch = [Complex::<f64>::default(); 48];

            for chunk in in_place.chunks_exact_mut(48) {
                // columns
                for k in 0..4 {
                    for i in 0..4 {
                        rows0[i] = NeonStoreD::from_complex_ref(chunk.get_unchecked(i * 12 + k..));
                        rows1[i] =
                            NeonStoreD::from_complex_ref(chunk.get_unchecked(i * 12 + k + 4..));
                        rows2[i] =
                            NeonStoreD::from_complex_ref(chunk.get_unchecked(i * 12 + k + 8..));
                    }

                    rows0 = self.bf4.exec(rows0);
                    rows1 = self.bf4.exec(rows1);
                    rows2 = self.bf4.exec(rows2);

                    for i in 1..4 {
                        rows0[i] =
                            NeonStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1 + 3 * k]);
                        rows1[i] =
                            NeonStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 3 * k + 12]);
                        rows2[i] =
                            NeonStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 3 * k + 24]);
                    }

                    for i in 0..4 {
                        rows0[i].write(scratch.get_unchecked_mut(k * 4 + i..));
                        rows1[i].write(scratch.get_unchecked_mut((k + 4) * 4 + i..));
                        rows2[i].write(scratch.get_unchecked_mut((k + 8) * 4 + i..));
                    }
                }

                // rows

                for k in 0..4 {
                    for i in 0..12 {
                        rows12[i] =
                            NeonStoreD::from_complex_ref(scratch.get_unchecked(i * 4 + k..));
                    }
                    rows12 = self.bf12.exec(rows12);
                    for i in 0..12 {
                        rows12[i].write(chunk.get_unchecked_mut(i * 4 + k..));
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

    test_butterfly!(test_neon_butterfly48_f64, f64, NeonButterfly48d, 48, 1e-7);
}
