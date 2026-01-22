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

use crate::avx::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::avx::mixed::{AvxStoreF, ColumnButterfly5f, ColumnButterfly6f};
use crate::avx::transpose::avx_transpose_f32x2_4x4_impl;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::_mm256_setzero_ps;

pub(crate) struct AvxButterfly30f {
    direction: FftDirection,
    bf5: ColumnButterfly5f,
    bf6: ColumnButterfly6f,
    twiddles: [AvxStoreF; 8],
}

impl AvxButterfly30f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(6, 5, fft_direction, 30),
            bf6: ColumnButterfly6f::new(fft_direction),
            bf5: ColumnButterfly5f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly30f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        30
    }
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) fn transpose_6x5(
    left: [AvxStoreF; 5],
    right: [AvxStoreF; 5],
) -> ([AvxStoreF; 6], [AvxStoreF; 6]) {
    // Perform a 6 x 5 matrix transpose by building on top of an existing 4 x 4
    // matrix transpose implementation:
    //
    // Original 6x5 matrix:
    //
    // [ r0c0 r0c1 r0c2 r0c3 r0c4 ]
    // [ r1c0 r1c1 r1c2 r1c3 r1c4 ]
    // [ r2c0 r2c1 r2c2 r2c3 r2c4 ]
    // [ r3c0 r3c1 r3c2 r3c3 r3c4 ]
    // [ r4c0 r4c1 r4c2 r4c3 r4c4 ]
    // [ r5c0 r5c1 r5c2 r5c3 r5c4 ]
    //
    // Split into blocks for transpose:
    //
    // [ A B ]   where A = 4x4 top-left, B = 4x1 top-right
    // [ C D ]         C = 2x4 bottom-left, D = 2x1 bottom-right
    //
    // Transpose:
    //
    // [ A B ]^T => [ A^T  C^T ]
    // [ C D ]      [ B^T  D^T ]
    //
    // In terms of rows/columns:
    //
    // A^T -> top-left 4x4 of output
    // B^T -> becomes a row (length 4) at output row 0..3, col 4
    // C^T -> becomes a column (length 2) at output col 0..3, row 4..5
    // D^T -> bottom-right 2x1 block at output row 4..5, col 4

    let tl = avx_transpose_f32x2_4x4_impl(left[0].v, left[1].v, left[2].v, left[3].v);
    // Bottom-left 2x4 complex block (pad 2 rows with zeros)
    let bl = avx_transpose_f32x2_4x4_impl(
        left[4].v,
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );

    // Top-right 4x2 complex block (pad 2 columns with zeros to form 4x4)
    let tr = avx_transpose_f32x2_4x4_impl(right[0].v, right[1].v, right[2].v, right[3].v);
    // Bottom-right 2x2 complex block
    let br = avx_transpose_f32x2_4x4_impl(
        right[4].v,
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );

    // Reassemble left 6 rows (first 4 columns)
    let output_left = [
        AvxStoreF::raw(tl.0),
        AvxStoreF::raw(tl.1),
        AvxStoreF::raw(tl.2),
        AvxStoreF::raw(tl.3), // top 4 rows
        AvxStoreF::raw(tr.0),
        AvxStoreF::raw(tr.1),
    ];

    // Reassemble right 6 rows (last 2 columns)
    let output_right = [
        AvxStoreF::raw(bl.0),
        AvxStoreF::raw(bl.1),
        AvxStoreF::raw(bl.2),
        AvxStoreF::raw(bl.3), // top 4 rows
        AvxStoreF::raw(br.0),
        AvxStoreF::raw(br.1),
    ];

    (output_left, output_right)
}

impl AvxButterfly30f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(30) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 5] = [AvxStoreF::zero(); 5];
            let mut rows1: [AvxStoreF; 5] = [AvxStoreF::zero(); 5];

            for chunk in in_place.chunks_exact_mut(30) {
                // columns
                {
                    for i in 0..5 {
                        rows0[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 6..));
                        rows1[i] = AvxStoreF::from_complex2(chunk.get_unchecked(i * 6 + 4..));
                    }

                    rows0 = self.bf5.exec(rows0);
                    rows1 = self.bf5.exec(rows1);

                    for i in 1..5 {
                        rows0[i] = AvxStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                        rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1 + 4]);
                    }

                    let (mut t0, mut t1) = transpose_6x5(rows0, rows1);

                    t0 = self.bf6.exec(t0);

                    for i in 0..6 {
                        t0[i].write(chunk.get_unchecked_mut(i * 5..));
                    }

                    t1 = self.bf6.exec(t1);

                    for i in 0..6 {
                        t1[i].write_lo1(chunk.get_unchecked_mut(i * 5 + 4..));
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

    test_avx_butterfly!(test_avx_butterfly30, f32, AvxButterfly30f, 30, 1e-3);
}
