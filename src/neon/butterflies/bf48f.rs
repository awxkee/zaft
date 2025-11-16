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

use crate::neon::f32x2_2x2::neon_transpose_f32x2_2x2_impl;
use crate::neon::mixed::{ColumnButterfly4f, ColumnButterfly12f, NeonStoreF};
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::float32x4x2_t;

pub(crate) struct NeonButterfly48f {
    direction: FftDirection,
    bf4: ColumnButterfly4f,
    bf12: ColumnButterfly12f,
    twiddles: [NeonStoreF; 18],
}

impl NeonButterfly48f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let mut twiddles = [NeonStoreF::default(); 18];
        let mut q = 0usize;
        let len_per_row = 12;
        const COMPLEX_PER_VECTOR: usize = 2;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..4 {
                twiddles[q] = NeonStoreF::from_complex2(
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 48, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 48, fft_direction),
                );
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf12: ColumnButterfly12f::new(fft_direction),
            bf4: ColumnButterfly4f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for NeonButterfly48f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
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

#[inline(always)]
pub(crate) fn transpose_6x4(
    rows0: [NeonStoreF; 4],
    rows1: [NeonStoreF; 4],
    rows2: [NeonStoreF; 4],
) -> [NeonStoreF; 12] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[2].v, rows0[3].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[2].v, rows1[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[0].v, rows2[1].v));
    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[2].v, rows2[3].v));
    [
        // row 0
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
        NeonStoreF::raw(f0.0),
        NeonStoreF::raw(f0.1),
        NeonStoreF::raw(g0.0),
        NeonStoreF::raw(g0.1),
    ]
}

impl NeonButterfly48f {
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 48 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
            let mut rows1: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
            let mut rows2: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
            let mut rows12: [NeonStoreF; 12] = [NeonStoreF::default(); 12];

            let mut scratch = [Complex::<f32>::default(); 48];

            for chunk in in_place.chunks_exact_mut(48) {
                // columns
                for k in 0..2 {
                    for i in 0..4 {
                        rows0[i] =
                            NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 12 + k * 6..));
                        rows1[i] =
                            NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 12 + k * 6 + 2..));
                        rows2[i] =
                            NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 12 + k * 6 + 4..));
                    }

                    rows0 = self.bf4.exec(rows0);
                    rows1 = self.bf4.exec(rows1);
                    rows2 = self.bf4.exec(rows2);

                    for i in 1..4 {
                        rows0[i] =
                            NeonStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1 + 9 * k]);
                        rows1[i] =
                            NeonStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1 + 9 * k + 3]);
                        rows2[i] =
                            NeonStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 9 * k + 6]);
                    }

                    let transposed = transpose_6x4(rows0, rows1, rows2);

                    for i in 0..6 {
                        transposed[i].write(scratch.get_unchecked_mut(i * 4 + k * 24..));
                        transposed[i + 6].write(scratch.get_unchecked_mut(i * 4 + k * 24 + 2..));
                    }
                }

                // rows

                for k in 0..2 {
                    for i in 0..12 {
                        rows12[i] =
                            NeonStoreF::from_complex_ref(scratch.get_unchecked(i * 4 + k * 2..));
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
    use crate::butterflies::test_butterfly_small;

    test_butterfly_small!(test_neon_butterfly48, f32, NeonButterfly48f, 48, 1e-3);
}
