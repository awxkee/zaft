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
use crate::avx::mixed::{AvxStoreD, ColumnButterfly7d, ColumnButterfly10d};
use crate::avx::transpose::transpose_f64x2_2x7;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly70d {
    direction: FftDirection,
    bf7: ColumnButterfly7d,
    bf10: ColumnButterfly10d,
    twiddles: [AvxStoreD; 30],
}

impl AvxButterfly70d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(10, 7, fft_direction, 70),
            bf10: ColumnButterfly10d::new(fft_direction),
            bf7: ColumnButterfly7d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly70d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        70
    }
}

impl AvxButterfly70d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(70) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];
            let mut rows10: [AvxStoreD; 10] = [AvxStoreD::zero(); 10];
            let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 70];

            for chunk in in_place.chunks_exact_mut(70) {
                // columns
                for k in 0..5 {
                    for i in 0..7 {
                        rows[i] =
                            AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 10 + k * 2..));
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

                    {
                        let i = 3;
                        transposed[i * 2].write_lou(scratch.get_unchecked_mut(k * 2 * 7 + i * 2..));
                        transposed[i * 2 + 1]
                            .write_lou(scratch.get_unchecked_mut((k * 2 + 1) * 7 + i * 2..));
                    }
                }

                // rows

                for k in 0..3 {
                    for i in 0..10 {
                        rows10[i] =
                            AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 7 + k * 2..));
                    }
                    rows10 = self.bf10.exec(rows10);
                    for i in 0..10 {
                        rows10[i].write(chunk.get_unchecked_mut(i * 7 + k * 2..));
                    }
                }

                {
                    let k = 3;
                    for i in 0..10 {
                        rows10[i] = AvxStoreD::from_complexu(scratch.get_unchecked(i * 7 + k * 2));
                    }
                    rows10 = self.bf10.exec(rows10);
                    for i in 0..10 {
                        rows10[i].write_lo(chunk.get_unchecked_mut(i * 7 + k * 2..));
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

    test_avx_butterfly!(test_avx_butterfly70_f64, f64, AvxButterfly70d, 70, 1e-3);
}
