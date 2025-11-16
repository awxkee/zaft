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

use crate::avx::f32x2_4x4::avx_transpose_f32x2_4x4_impl;
use crate::avx::mixed::{AvxStoreF, ColumnButterfly11f};
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::_mm256_setzero_ps;

pub(crate) struct AvxButterfly121f {
    direction: FftDirection,
    bf11: ColumnButterfly11f,
    twiddles: [AvxStoreF; 30],
}

impl AvxButterfly121f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        let mut twiddles = [AvxStoreF::zero(); 30];
        let mut q = 0usize;
        let len_per_row = 11;
        const COMPLEX_PER_VECTOR: usize = 4;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..11 {
                twiddles[q] = AvxStoreF::set_complex4(
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 121, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 121, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 2), 121, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 3), 121, fft_direction),
                );
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf11: ColumnButterfly11f::new(fft_direction),
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_11x4(rows: [AvxStoreF; 11]) -> [AvxStoreF; 12] {
    let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
    let b0 = avx_transpose_f32x2_4x4_impl(rows[4].v, rows[5].v, rows[6].v, rows[7].v);
    let c0 = avx_transpose_f32x2_4x4_impl(rows[8].v, rows[9].v, rows[10].v, _mm256_setzero_ps());
    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
        AvxStoreF::raw(b0.0),
        AvxStoreF::raw(b0.1),
        AvxStoreF::raw(b0.2),
        AvxStoreF::raw(b0.3),
        AvxStoreF::raw(c0.0),
        AvxStoreF::raw(c0.1),
        AvxStoreF::raw(c0.2),
        AvxStoreF::raw(c0.3),
    ]
}

impl FftExecutor<f32> for AvxButterfly121f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        121
    }
}

impl AvxButterfly121f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 121 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [AvxStoreF; 11] = [AvxStoreF::zero(); 11];
            let mut scratch = [Complex::<f32>::default(); 121];

            for chunk in in_place.chunks_exact_mut(121) {
                // columns
                for k in 0..2 {
                    for i in 0..11 {
                        rows[i] =
                            AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 11 + k * 4..));
                    }

                    rows = self.bf11.exec(rows);

                    for i in 1..11 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 10 * k]);
                    }

                    let transposed = transpose_11x4(rows);

                    for i in 0..2 {
                        transposed[i * 4].write(scratch.get_unchecked_mut(k * 4 * 11 + i * 4..));
                        transposed[i * 4 + 1]
                            .write(scratch.get_unchecked_mut((k * 4 + 1) * 11 + i * 4..));
                        transposed[i * 4 + 2]
                            .write(scratch.get_unchecked_mut((k * 4 + 2) * 11 + i * 4..));
                        transposed[i * 4 + 3]
                            .write(scratch.get_unchecked_mut((k * 4 + 3) * 11 + i * 4..));
                    }

                    transposed[4 * 2].write_lo3(scratch.get_unchecked_mut(k * 4 * 11 + 8..));
                    transposed[4 * 2 + 1]
                        .write_lo3(scratch.get_unchecked_mut((k * 4 + 1) * 11 + 8..));
                    transposed[4 * 2 + 2]
                        .write_lo3(scratch.get_unchecked_mut((k * 4 + 2) * 11 + 8..));
                    transposed[4 * 2 + 3]
                        .write_lo3(scratch.get_unchecked_mut((k * 4 + 3) * 11 + 8..));
                }

                {
                    for i in 0..11 {
                        rows[i] = AvxStoreF::from_complex3(chunk.get_unchecked(i * 11 + 8..));
                    }

                    rows = self.bf11.exec(rows);

                    for i in 1..11 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 10 * 2]);
                    }

                    let transposed = transpose_11x4(rows);

                    for i in 0..2 {
                        transposed[i * 4].write(scratch.get_unchecked_mut(8 * 11 + i * 4..));
                        transposed[i * 4 + 1]
                            .write(scratch.get_unchecked_mut((8 + 1) * 11 + i * 4..));
                        transposed[i * 4 + 2]
                            .write(scratch.get_unchecked_mut((8 + 2) * 11 + i * 4..));
                    }

                    transposed[4 * 2].write_lo3(scratch.get_unchecked_mut(8 * 11 + 8..));
                    transposed[4 * 2 + 1].write_lo3(scratch.get_unchecked_mut((8 + 1) * 11 + 8..));
                    transposed[4 * 2 + 2].write_lo3(scratch.get_unchecked_mut((8 + 2) * 11 + 8..));
                }

                // rows

                for k in 0..2 {
                    for i in 0..11 {
                        rows[i] =
                            AvxStoreF::from_complex_ref(scratch.get_unchecked(i * 11 + k * 4..));
                    }
                    rows = self.bf11.exec(rows);
                    for i in 0..11 {
                        rows[i].write(chunk.get_unchecked_mut(i * 11 + k * 4..));
                    }
                }

                {
                    for i in 0..11 {
                        rows[i] = AvxStoreF::from_complex3(scratch.get_unchecked(i * 11 + 8..));
                    }
                    rows = self.bf11.exec(rows);
                    for i in 0..11 {
                        rows[i].write_lo3(chunk.get_unchecked_mut(i * 11 + 8..));
                    }
                }
            }
        }
        Ok(())
    }
}
impl FftExecutorOutOfPlace<f32> for AvxButterfly121f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly121f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 121 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 121 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows: [AvxStoreF; 11] = [AvxStoreF::zero(); 11];
            let mut scratch = [Complex::<f32>::default(); 121];

            for (dst, src) in dst.chunks_exact_mut(121).zip(src.chunks_exact(121)) {
                // columns
                for k in 0..2 {
                    for i in 0..11 {
                        rows[i] = AvxStoreF::from_complex_ref(src.get_unchecked(i * 11 + k * 4..));
                    }

                    rows = self.bf11.exec(rows);

                    for i in 1..11 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 10 * k]);
                    }

                    let transposed = transpose_11x4(rows);

                    for i in 0..2 {
                        transposed[i * 4].write(scratch.get_unchecked_mut(k * 4 * 11 + i * 4..));
                        transposed[i * 4 + 1]
                            .write(scratch.get_unchecked_mut((k * 4 + 1) * 11 + i * 4..));
                        transposed[i * 4 + 2]
                            .write(scratch.get_unchecked_mut((k * 4 + 2) * 11 + i * 4..));
                        transposed[i * 4 + 3]
                            .write(scratch.get_unchecked_mut((k * 4 + 3) * 11 + i * 4..));
                    }

                    transposed[4 * 2].write_lo3(scratch.get_unchecked_mut(k * 4 * 11 + 8..));
                    transposed[4 * 2 + 1]
                        .write_lo3(scratch.get_unchecked_mut((k * 4 + 1) * 11 + 8..));
                    transposed[4 * 2 + 2]
                        .write_lo3(scratch.get_unchecked_mut((k * 4 + 2) * 11 + 8..));
                    transposed[4 * 2 + 3]
                        .write_lo3(scratch.get_unchecked_mut((k * 4 + 3) * 11 + 8..));
                }

                {
                    for i in 0..11 {
                        rows[i] = AvxStoreF::from_complex3(src.get_unchecked(i * 11 + 8..));
                    }

                    rows = self.bf11.exec(rows);

                    for i in 1..11 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 10 * 2]);
                    }

                    let transposed = transpose_11x4(rows);

                    for i in 0..2 {
                        transposed[i * 4].write(scratch.get_unchecked_mut(8 * 11 + i * 4..));
                        transposed[i * 4 + 1]
                            .write(scratch.get_unchecked_mut((8 + 1) * 11 + i * 4..));
                        transposed[i * 4 + 2]
                            .write(scratch.get_unchecked_mut((8 + 2) * 11 + i * 4..));
                    }

                    transposed[4 * 2].write_lo3(scratch.get_unchecked_mut(8 * 11 + 8..));
                    transposed[4 * 2 + 1].write_lo3(scratch.get_unchecked_mut((8 + 1) * 11 + 8..));
                    transposed[4 * 2 + 2].write_lo3(scratch.get_unchecked_mut((8 + 2) * 11 + 8..));
                }

                // rows

                for k in 0..2 {
                    for i in 0..11 {
                        rows[i] =
                            AvxStoreF::from_complex_ref(scratch.get_unchecked(i * 11 + k * 4..));
                    }
                    rows = self.bf11.exec(rows);
                    for i in 0..11 {
                        rows[i].write(dst.get_unchecked_mut(i * 11 + k * 4..));
                    }
                }

                {
                    for i in 0..11 {
                        rows[i] = AvxStoreF::from_complex3(scratch.get_unchecked(i * 11 + 8..));
                    }
                    rows = self.bf11.exec(rows);
                    for i in 0..11 {
                        rows[i].write_lo3(dst.get_unchecked_mut(i * 11 + 8..));
                    }
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly121f {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly121, f32, AvxButterfly121f, 121, 1e-3);
    test_oof_avx_butterfly!(test_oof_avx_butterfly121, f32, AvxButterfly121f, 121, 1e-3);
}
