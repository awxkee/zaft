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

use crate::neon::mixed::{ColumnButterfly5f, ColumnButterfly7f, NeonStoreF};
use crate::neon::transpose::{neon_transpose_f32x2_2x2_impl, transpose_6x5};
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::{float32x4x2_t, vdupq_n_f32};
use std::mem::MaybeUninit;

pub(crate) struct NeonButterfly35f {
    direction: FftDirection,
    bf5: ColumnButterfly5f,
    bf7: ColumnButterfly7f,
    twiddles: [NeonStoreF; 16],
}

impl NeonButterfly35f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let mut twiddles = [NeonStoreF::default(); 16];
        let mut q = 0usize;
        let len_per_row = 7;
        const COMPLEX_PER_VECTOR: usize = 2;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..5 {
                twiddles[q] = NeonStoreF::from_complex2(
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 35, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 35, fft_direction),
                );
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf7: ColumnButterfly7f::new(fft_direction),
            bf5: ColumnButterfly5f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for NeonButterfly35f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
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

impl NeonButterfly35f {
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 35 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [NeonStoreF; 5] = [NeonStoreF::default(); 5];
            let mut rows1: [NeonStoreF; 5] = [NeonStoreF::default(); 5];
            let mut rows2: [NeonStoreF; 5] = [NeonStoreF::default(); 5];
            let mut rows7: [NeonStoreF; 7] = [NeonStoreF::default(); 7];

            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 35];
            
            for chunk in in_place.chunks_exact_mut(35) {
                // columns
                {
                    let k = 0;
                    for i in 0..5 {
                        rows0[i] =
                            NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 7 + k * 6..));
                        rows1[i] =
                            NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 7 + k * 6 + 2..));
                        rows2[i] =
                            NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 7 + k * 6 + 4..));
                    }

                    rows0 = self.bf5.exec(rows0);
                    rows1 = self.bf5.exec(rows1);
                    rows2 = self.bf5.exec(rows2);

                    for i in 1..5 {
                        rows0[i] = NeonStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                        rows1[i] = NeonStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1 + 4]);
                        rows2[i] = NeonStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 8]);
                    }

                    let transposed = transpose_6x5(rows0, rows1, rows2);

                    for i in 0..6 {
                        transposed[i].write_uninit(scratch.get_unchecked_mut(i * 5..));
                        transposed[i + 6].write_uninit(scratch.get_unchecked_mut(i * 5 + 2..));
                        transposed[i + 12].write_lo_u(scratch.get_unchecked_mut(i * 5 + 4..));
                    }
                }

                {
                    let k = 6;
                    for i in 0..5 {
                        rows0[i] = NeonStoreF::from_complex(chunk.get_unchecked(i * 7 + k));
                    }

                    rows0 = self.bf5.exec(rows0);

                    for i in 1..5 {
                        rows0[i] = NeonStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1 + 12]);
                    }

                    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
                    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[2].v, rows0[3].v));
                    let g0 =
                        neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[4].v, vdupq_n_f32(0.)));
                    NeonStoreF::raw(a0.0).write_uninit(scratch.get_unchecked_mut(6 * 5..));
                    NeonStoreF::raw(d0.0).write_uninit(scratch.get_unchecked_mut(6 * 5 + 2..));
                    NeonStoreF::raw(g0.0).write_lo_u(scratch.get_unchecked_mut(6 * 5 + 4..));
                }

                // rows

                for k in 0..2 {
                    for i in 0..7 {
                        rows7[i] =
                            NeonStoreF::from_complex_refu(scratch.get_unchecked(i * 5 + k * 2..));
                    }
                    rows7 = self.bf7.exec(rows7);
                    for i in 0..7 {
                        rows7[i].write(chunk.get_unchecked_mut(i * 5 + k * 2..));
                    }
                }
                {
                    let k = 2;
                    for i in 0..7 {
                        rows7[i] = NeonStoreF::from_complexu(scratch.get_unchecked(i * 5 + k * 2));
                    }
                    rows7 = self.bf7.exec(rows7);
                    for i in 0..7 {
                        rows7[i].write_lo(chunk.get_unchecked_mut(i * 5 + k * 2..));
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

    test_butterfly!(test_neon_butterfly35, f32, NeonButterfly35f, 35, 1e-3);
}
