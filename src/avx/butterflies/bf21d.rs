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
use crate::avx::mixed::{AvxStoreD, ColumnButterfly3d, ColumnButterfly7d};
use crate::avx::transpose::avx_transpose_f64x2_4x4_impl;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::_mm256_setzero_pd;

pub(crate) struct AvxButterfly21d {
    direction: FftDirection,
    bf3: ColumnButterfly3d,
    bf7: ColumnButterfly7d,
    twiddles: [AvxStoreD; 8],
}

impl AvxButterfly21d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(7, 3, fft_direction, 21),
            bf3: ColumnButterfly3d::new(fft_direction),
            bf7: ColumnButterfly7d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly21d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        21
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_7x3(
    rows0: [AvxStoreD; 3],
    rows1: [AvxStoreD; 3],
    rows2: [AvxStoreD; 3],
    rows3: [AvxStoreD; 3],
) -> ([AvxStoreD; 7], [AvxStoreD; 7]) {
    let a0 = avx_transpose_f64x2_4x4_impl(
        (rows0[0].v, rows1[0].v),
        (rows0[1].v, rows1[1].v),
        (rows0[2].v, rows1[2].v),
        (_mm256_setzero_pd(), _mm256_setzero_pd()),
    );
    let b0 = avx_transpose_f64x2_4x4_impl(
        (rows2[0].v, rows3[0].v),
        (rows2[1].v, rows3[1].v),
        (rows2[2].v, rows3[2].v),
        (_mm256_setzero_pd(), _mm256_setzero_pd()),
    );
    (
        [
            // row 0
            AvxStoreD::raw(a0.0.0),
            AvxStoreD::raw(a0.1.0),
            AvxStoreD::raw(a0.2.0),
            AvxStoreD::raw(a0.3.0),
            AvxStoreD::raw(b0.0.0),
            AvxStoreD::raw(b0.1.0),
            AvxStoreD::raw(b0.2.0),
        ],
        [
            // row 0
            AvxStoreD::raw(a0.0.1),
            AvxStoreD::raw(a0.1.1),
            AvxStoreD::raw(a0.2.1),
            AvxStoreD::raw(a0.3.1),
            AvxStoreD::raw(b0.0.1),
            AvxStoreD::raw(b0.1.1),
            AvxStoreD::raw(b0.2.1),
        ],
    )
}

impl AvxButterfly21d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(21) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
            let mut rows1: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
            let mut rows2: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
            let mut rows3: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];

            for chunk in in_place.chunks_exact_mut(21) {
                // columns
                for i in 0..3 {
                    rows0[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7..));
                    rows1[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7 + 2..));
                    rows2[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7 + 4..));
                    rows3[i] = AvxStoreD::from_complex(chunk.get_unchecked(i * 7 + 6));
                }

                rows0 = self.bf3.exec(rows0);
                rows1 = self.bf3.exec(rows1);
                rows2 = self.bf3.exec(rows2);
                rows3 = self.bf3.exec(rows3);

                for i in 1..3 {
                    rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 2]);
                    rows2[i] = AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 4]);
                    rows3[i] = AvxStoreD::mul_by_complex(rows3[i], self.twiddles[i - 1 + 6]);
                }

                let transposed = transpose_7x3(rows0, rows1, rows2, rows3);

                // rows

                let q0 = self.bf7.exec(transposed.0);
                let q1 = self.bf7.exec(transposed.1);

                for i in 0..7 {
                    q0[i].write(chunk.get_unchecked_mut(i * 3..));
                    q1[i].write_lo(chunk.get_unchecked_mut(i * 3 + 2..));
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

    test_avx_butterfly!(test_avx_butterfly21_f64, f64, AvxButterfly21d, 21, 1e-7);
}
