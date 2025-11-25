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

use crate::avx::mixed::{AvxStoreD, ColumnButterfly6d, ColumnButterfly7d};
use crate::avx::transpose::transpose_f64x2_2x6;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

pub(crate) struct AvxButterfly42d {
    direction: FftDirection,
    bf6: ColumnButterfly6d,
    bf7: ColumnButterfly7d,
    twiddles: [AvxStoreD; 20],
}

impl AvxButterfly42d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        let mut twiddles = [AvxStoreD::zero(); 20];
        let mut q = 0usize;
        let len_per_row = 7;
        const COMPLEX_PER_VECTOR: usize = 2;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..6 {
                twiddles[q] = AvxStoreD::set_complex2(
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 42, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 42, fft_direction),
                );
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf7: ColumnButterfly7d::new(fft_direction),
            bf6: ColumnButterfly6d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly42d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        42
    }
}

impl AvxButterfly42d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 42 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 6] = [AvxStoreD::zero(); 6];

            let mut rows7: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];

            let mut scratch = [Complex::<f64>::default(); 42];

            for chunk in in_place.chunks_exact_mut(42) {
                // columns
                for k in 0..3 {
                    for i in 0..6 {
                        rows0[i] =
                            AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7 + k * 2..));
                    }

                    rows0 = self.bf6.exec(rows0);

                    for i in 1..6 {
                        rows0[i] =
                            AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1 + 5 * k]);
                    }

                    let transposed = transpose_f64x2_2x6(rows0);

                    for i in 0..3 {
                        transposed[i * 2].write(scratch.get_unchecked_mut(k * 2 * 6 + i * 2..));
                        transposed[i * 2 + 1]
                            .write(scratch.get_unchecked_mut((k * 2 + 1) * 6 + i * 2..));
                    }
                }

                {
                    let k = 3;
                    for i in 0..6 {
                        rows0[i] = AvxStoreD::from_complex(chunk.get_unchecked(i * 7 + k * 2));
                    }

                    rows0 = self.bf6.exec(rows0);

                    for i in 1..6 {
                        rows0[i] =
                            AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1 + 5 * k]);
                    }

                    let transposed = transpose_f64x2_2x6(rows0);

                    for i in 0..3 {
                        transposed[i * 2].write(scratch.get_unchecked_mut(k * 2 * 6 + i * 2..));
                    }
                }

                // rows

                for k in 0..3 {
                    for i in 0..7 {
                        rows7[i] =
                            AvxStoreD::from_complex_ref(scratch.get_unchecked(i * 6 + k * 2..));
                    }
                    rows7 = self.bf7.exec(rows7);
                    for i in 0..7 {
                        rows7[i].write(chunk.get_unchecked_mut(i * 6 + k * 2..));
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
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly42, f64, AvxButterfly42d, 42, 1e-3);
}
