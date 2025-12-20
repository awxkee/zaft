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

use crate::avx::butterflies::shared::gen_butterfly_twiddles_f64;
use crate::avx::mixed::{AvxStoreD, ColumnButterfly4d, ColumnButterfly12d};
use crate::avx::transpose::transpose_f64x2_2x2;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly48d {
    direction: FftDirection,
    bf4: ColumnButterfly4d,
    bf12: ColumnButterfly12d,
    twiddles: [AvxStoreD; 18],
}

impl AvxButterfly48d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(12, 4, fft_direction, 48),
            bf12: ColumnButterfly12d::new(fft_direction),
            bf4: ColumnButterfly4d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly48d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        48
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_6x4(
    rows0: [AvxStoreD; 4],
    rows1: [AvxStoreD; 4],
    rows2: [AvxStoreD; 4],
) -> [AvxStoreD; 12] {
    let a0 = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);
    let d0 = transpose_f64x2_2x2(rows0[2].v, rows0[3].v);
    let b0 = transpose_f64x2_2x2(rows1[0].v, rows1[1].v);
    let f0 = transpose_f64x2_2x2(rows1[2].v, rows1[3].v);
    let c0 = transpose_f64x2_2x2(rows2[0].v, rows2[1].v);
    let g0 = transpose_f64x2_2x2(rows2[2].v, rows2[3].v);
    [
        // row 0
        AvxStoreD::raw(a0.0),
        AvxStoreD::raw(a0.1),
        AvxStoreD::raw(b0.0),
        AvxStoreD::raw(b0.1),
        AvxStoreD::raw(c0.0),
        AvxStoreD::raw(c0.1),
        AvxStoreD::raw(d0.0),
        AvxStoreD::raw(d0.1),
        AvxStoreD::raw(f0.0),
        AvxStoreD::raw(f0.1),
        AvxStoreD::raw(g0.0),
        AvxStoreD::raw(g0.1),
    ]
}

impl AvxButterfly48d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(48) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
            let mut rows1: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
            let mut rows2: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
            let mut rows12: [AvxStoreD; 12] = [AvxStoreD::zero(); 12];

            let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 48];

            for chunk in in_place.chunks_exact_mut(48) {
                // columns
                for k in 0..2 {
                    for i in 0..4 {
                        rows0[i] =
                            AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 12 + k * 6..));
                        rows1[i] =
                            AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 12 + k * 6 + 2..));
                        rows2[i] =
                            AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 12 + k * 6 + 4..));
                    }

                    rows0 = self.bf4.exec(rows0);
                    rows1 = self.bf4.exec(rows1);
                    rows2 = self.bf4.exec(rows2);

                    for i in 1..4 {
                        rows0[i] =
                            AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1 + 9 * k]);
                        rows1[i] =
                            AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 9 * k + 3]);
                        rows2[i] =
                            AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 9 * k + 6]);
                    }

                    let transposed = transpose_6x4(rows0, rows1, rows2);

                    for i in 0..6 {
                        transposed[i].write_u(scratch.get_unchecked_mut(i * 4 + k * 24..));
                        transposed[i + 6].write_u(scratch.get_unchecked_mut(i * 4 + k * 24 + 2..));
                    }
                }

                // rows

                for k in 0..2 {
                    for i in 0..12 {
                        rows12[i] =
                            AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 4 + k * 2..));
                    }
                    rows12 = self.bf12.exec(rows12);
                    for i in 0..12 {
                        rows12[i].write(chunk.get_unchecked_mut(i * 4 + k * 2..));
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

    test_avx_butterfly!(test_avx_butterfly48_f64, f64, AvxButterfly48d, 48, 1e-3);
}
