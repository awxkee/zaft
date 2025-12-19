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

use crate::avx::butterflies::shared::gen_butterfly_twiddles_f64;
use crate::avx::mixed::{AvxStoreD, ColumnButterfly6d, ColumnButterfly13d};
use crate::avx::transpose::transpose_f64x2_2x6;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly78d {
    direction: FftDirection,
    bf6: ColumnButterfly6d,
    bf13: ColumnButterfly13d,
    twiddles: [AvxStoreD; 35],
}

impl AvxButterfly78d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(13, 6, fft_direction, 78),
            bf13: ColumnButterfly13d::new(fft_direction),
            bf6: ColumnButterfly6d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly78d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        78
    }
}

impl AvxButterfly78d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(78) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [AvxStoreD; 6] = [AvxStoreD::zero(); 6];
            let mut rows13: [AvxStoreD; 13] = [AvxStoreD::zero(); 13];
            let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 78];

            for chunk in in_place.chunks_exact_mut(78) {
                // columns
                for k in 0..6 {
                    for i in 0..6 {
                        rows[i] =
                            AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 13 + k * 2..));
                    }

                    rows = self.bf6.exec(rows);

                    for i in 1..6 {
                        rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 5 * k]);
                    }

                    let transposed = transpose_f64x2_2x6(rows);

                    for i in 0..3 {
                        transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 6 + i * 2..));
                        transposed[i * 2 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 2 + 1) * 6 + i * 2..));
                    }
                }

                {
                    let k = 6;
                    for i in 0..6 {
                        rows[i] = AvxStoreD::from_complex(chunk.get_unchecked(i * 13 + k * 2));
                    }

                    rows = self.bf6.exec(rows);

                    for i in 1..6 {
                        rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 5 * k]);
                    }

                    let transposed = transpose_f64x2_2x6(rows);

                    for i in 0..3 {
                        transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 6 + i * 2..));
                    }
                }

                // rows

                for k in 0..3 {
                    for i in 0..13 {
                        rows13[i] =
                            AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 6 + k * 2..));
                    }
                    rows13 = self.bf13.exec(rows13);
                    for i in 0..13 {
                        rows13[i].write(chunk.get_unchecked_mut(i * 6 + k * 2..));
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

    test_avx_butterfly!(test_avx_butterfly78_f64, f64, AvxButterfly78d, 78, 1e-3);
}
