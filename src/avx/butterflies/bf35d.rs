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

use crate::avx::mixed::{AvxStoreD, ColumnButterfly5d, ColumnButterfly7d};
use crate::avx::transpose::transpose_f64x2_2x2;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::_mm256_setzero_pd;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly35d {
    direction: FftDirection,
    bf5: ColumnButterfly5d,
    bf7: ColumnButterfly7d,
    twiddles: [AvxStoreD; 16],
}

impl AvxButterfly35d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        let mut twiddles = [AvxStoreD::zero(); 16];
        let mut q = 0usize;
        let len_per_row = 7;
        const COMPLEX_PER_VECTOR: usize = 2;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..5 {
                twiddles[q] = AvxStoreD::set_complex2(
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 35, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 35, fft_direction),
                );
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf7: ColumnButterfly7d::new(fft_direction),
            bf5: ColumnButterfly5d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly35d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        35
    }
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) fn transpose_6x5(
    rows0: [AvxStoreD; 5],
    rows1: [AvxStoreD; 5],
    rows2: [AvxStoreD; 5],
) -> [AvxStoreD; 18] {
    let a0 = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);
    let d0 = transpose_f64x2_2x2(rows0[2].v, rows0[3].v);
    let g0 = transpose_f64x2_2x2(rows0[4].v, _mm256_setzero_pd());

    let b0 = transpose_f64x2_2x2(rows1[0].v, rows1[1].v);
    let e0 = transpose_f64x2_2x2(rows1[2].v, rows1[3].v);
    let h0 = transpose_f64x2_2x2(rows1[4].v, _mm256_setzero_pd());

    let c0 = transpose_f64x2_2x2(rows2[0].v, rows2[1].v);
    let f0 = transpose_f64x2_2x2(rows2[2].v, rows2[3].v);
    let i0 = transpose_f64x2_2x2(rows2[4].v, _mm256_setzero_pd());
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
        AvxStoreD::raw(e0.0),
        AvxStoreD::raw(e0.1),
        AvxStoreD::raw(f0.0),
        AvxStoreD::raw(f0.1),
        AvxStoreD::raw(g0.0),
        AvxStoreD::raw(g0.1),
        AvxStoreD::raw(h0.0),
        AvxStoreD::raw(h0.1),
        AvxStoreD::raw(i0.0),
        AvxStoreD::raw(i0.1),
    ]
}

impl AvxButterfly35d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 35 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];
            let mut rows1: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];
            let mut rows2: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];
            let mut rows7: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];

            let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 35];

            for chunk in in_place.chunks_exact_mut(35) {
                // columns
                {
                    let k = 0;
                    for i in 0..5 {
                        rows0[i] =
                            AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7 + k * 6..));
                        rows1[i] =
                            AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7 + k * 6 + 2..));
                        rows2[i] =
                            AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7 + k * 6 + 4..));
                    }

                    rows0 = self.bf5.exec(rows0);
                    rows1 = self.bf5.exec(rows1);
                    rows2 = self.bf5.exec(rows2);

                    for i in 1..5 {
                        rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                        rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 4]);
                        rows2[i] = AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 8]);
                    }

                    let transposed = transpose_6x5(rows0, rows1, rows2);

                    for i in 0..6 {
                        transposed[i].write_u(scratch.get_unchecked_mut(i * 5..));
                        transposed[i + 6].write_u(scratch.get_unchecked_mut(i * 5 + 2..));
                        transposed[i + 12].write_lou(scratch.get_unchecked_mut(i * 5 + 4..));
                    }
                }

                {
                    let k = 6;
                    for i in 0..5 {
                        rows0[i] = AvxStoreD::from_complex(chunk.get_unchecked(i * 7 + k));
                    }

                    rows0 = self.bf5.exec(rows0);

                    for i in 1..5 {
                        rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1 + 12]);
                    }

                    let a0 = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);
                    let d0 = transpose_f64x2_2x2(rows0[2].v, rows0[3].v);
                    let g0 = transpose_f64x2_2x2(rows0[4].v, _mm256_setzero_pd());

                    AvxStoreD::raw(a0.0).write_u(scratch.get_unchecked_mut(6 * 5..));
                    AvxStoreD::raw(d0.0).write_u(scratch.get_unchecked_mut(6 * 5 + 2..));
                    AvxStoreD::raw(g0.0).write_lou(scratch.get_unchecked_mut(6 * 5 + 4..));
                }

                // rows

                for k in 0..2 {
                    for i in 0..7 {
                        rows7[i] =
                            AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 5 + k * 2..));
                    }
                    rows7 = self.bf7.exec(rows7);
                    for i in 0..7 {
                        rows7[i].write(chunk.get_unchecked_mut(i * 5 + k * 2..));
                    }
                }
                {
                    let k = 2;
                    for i in 0..7 {
                        rows7[i] = AvxStoreD::from_complexu(scratch.get_unchecked(i * 5 + k * 2));
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
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly35_f64, f64, AvxButterfly35d, 35, 1e-3);
}
