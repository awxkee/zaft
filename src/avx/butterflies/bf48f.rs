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

use crate::avx::butterflies::gen_butterfly_twiddles_interleaved_columns_f32;
use crate::avx::f32x2_4x4::avx_transpose_f32x2_4x4_impl;
use crate::avx::mixed::{AvxStoreF, ColumnButterfly4f, ColumnButterfly12f};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

pub(crate) struct AvxButterfly48f {
    direction: FftDirection,
    bf4: ColumnButterfly4f,
    bf12: ColumnButterfly12f,
    twiddles: [AvxStoreF; 9],
}

impl AvxButterfly48f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: unsafe {
                gen_butterfly_twiddles_interleaved_columns_f32!(4, 12, 0, fft_direction)
            },
            bf12: ColumnButterfly12f::new(fft_direction),
            bf4: ColumnButterfly4f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly48f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
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
fn transpose_12x4_to_4x12_f32(
    rows0: [AvxStoreF; 4],
    rows1: [AvxStoreF; 4],
    rows2: [AvxStoreF; 4],
) -> [AvxStoreF; 12] {
    let transposed0 = avx_transpose_f32x2_4x4_impl(rows0[0].v, rows0[1].v, rows0[2].v, rows0[3].v);
    let transposed1 = avx_transpose_f32x2_4x4_impl(rows1[0].v, rows1[1].v, rows1[2].v, rows1[3].v);
    let transposed2 = avx_transpose_f32x2_4x4_impl(rows2[0].v, rows2[1].v, rows2[2].v, rows2[3].v);

    [
        AvxStoreF::raw(transposed0.0),
        AvxStoreF::raw(transposed0.1),
        AvxStoreF::raw(transposed0.2),
        AvxStoreF::raw(transposed0.3),
        AvxStoreF::raw(transposed1.0),
        AvxStoreF::raw(transposed1.1),
        AvxStoreF::raw(transposed1.2),
        AvxStoreF::raw(transposed1.3),
        AvxStoreF::raw(transposed2.0),
        AvxStoreF::raw(transposed2.1),
        AvxStoreF::raw(transposed2.2),
        AvxStoreF::raw(transposed2.3),
    ]
}

impl AvxButterfly48f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 48 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0 = [AvxStoreF::zero(); 4];
            let mut rows1 = [AvxStoreF::zero(); 4];
            let mut rows2 = [AvxStoreF::zero(); 4];

            for chunk in in_place.chunks_exact_mut(48) {
                // columns
                for r in 0..4 {
                    rows0[r] = AvxStoreF::from_complex_ref(chunk.get_unchecked(12 * r..));
                    rows1[r] = AvxStoreF::from_complex_ref(chunk.get_unchecked(12 * r + 4..));
                    rows2[r] = AvxStoreF::from_complex_ref(chunk.get_unchecked(12 * r + 8..));
                }

                let mut mid0 = self.bf4.exec(rows0);
                let mut mid1 = self.bf4.exec(rows1);
                let mut mid2 = self.bf4.exec(rows2);

                for r in 1..4 {
                    mid0[r] = AvxStoreF::mul_by_complex(mid0[r], self.twiddles[3 * r - 3]);
                    mid1[r] = AvxStoreF::mul_by_complex(mid1[r], self.twiddles[3 * r - 2]);
                    mid2[r] = AvxStoreF::mul_by_complex(mid2[r], self.twiddles[3 * r - 1]);
                }

                let transposed = transpose_12x4_to_4x12_f32(mid0, mid1, mid2);

                let output_rows = self.bf12.exec(transposed);

                output_rows[0].write(chunk);
                output_rows[1].write(chunk.get_unchecked_mut(4..));
                output_rows[2].write(chunk.get_unchecked_mut(8..));
                output_rows[3].write(chunk.get_unchecked_mut(12..));
                output_rows[4].write(chunk.get_unchecked_mut(16..));
                output_rows[5].write(chunk.get_unchecked_mut(20..));
                output_rows[6].write(chunk.get_unchecked_mut(24..));
                output_rows[7].write(chunk.get_unchecked_mut(28..));
                output_rows[8].write(chunk.get_unchecked_mut(32..));
                output_rows[9].write(chunk.get_unchecked_mut(36..));
                output_rows[10].write(chunk.get_unchecked_mut(40..));
                output_rows[11].write(chunk.get_unchecked_mut(44..));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly48, f32, AvxButterfly48f, 48, 1e-3);
}
