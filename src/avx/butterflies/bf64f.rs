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

use crate::avx::butterflies::gen_butterfly_twiddles_separated_columns_f32;
use crate::avx::mixed::{AvxStoreF, ColumnButterfly8f};
use crate::avx::transpose::avx_transpose_f32x2_4x4_impl;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::sync::Arc;

#[inline]
#[target_feature(enable = "avx")]
fn transpose_8x8_f32(
    rows0: [AvxStoreF; 8],
    rows1: [AvxStoreF; 8],
) -> ([AvxStoreF; 8], [AvxStoreF; 8]) {
    let transposed00 = avx_transpose_f32x2_4x4_impl(rows0[0].v, rows0[1].v, rows0[2].v, rows0[3].v);
    let transposed01 = avx_transpose_f32x2_4x4_impl(rows1[0].v, rows1[1].v, rows1[2].v, rows1[3].v);
    let transposed10 = avx_transpose_f32x2_4x4_impl(rows0[4].v, rows0[5].v, rows0[6].v, rows0[7].v);
    let transposed11 = avx_transpose_f32x2_4x4_impl(rows1[4].v, rows1[5].v, rows1[6].v, rows1[7].v);

    let output0 = [
        AvxStoreF::raw(transposed00.0),
        AvxStoreF::raw(transposed00.1),
        AvxStoreF::raw(transposed00.2),
        AvxStoreF::raw(transposed00.3),
        AvxStoreF::raw(transposed01.0),
        AvxStoreF::raw(transposed01.1),
        AvxStoreF::raw(transposed01.2),
        AvxStoreF::raw(transposed01.3),
    ];
    let output1 = [
        AvxStoreF::raw(transposed10.0),
        AvxStoreF::raw(transposed10.1),
        AvxStoreF::raw(transposed10.2),
        AvxStoreF::raw(transposed10.3),
        AvxStoreF::raw(transposed11.0),
        AvxStoreF::raw(transposed11.1),
        AvxStoreF::raw(transposed11.2),
        AvxStoreF::raw(transposed11.3),
    ];

    (output0, output1)
}

pub(crate) struct AvxButterfly64f {
    direction: FftDirection,
    twiddles: [AvxStoreF; 14],
    bf8_column: ColumnButterfly8f,
}

impl AvxButterfly64f {
    pub(crate) fn new(direction: FftDirection) -> Self {
        unsafe { Self::new_init(direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(direction: FftDirection) -> Self {
        AvxButterfly64f {
            direction,
            twiddles: unsafe { gen_butterfly_twiddles_separated_columns_f32!(8, 8, 0, direction) },
            bf8_column: ColumnButterfly8f::new(direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly64f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        64
    }
}

impl AvxButterfly64f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(64) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0 = [AvxStoreF::zero(); 8];
            let mut rows1 = [AvxStoreF::zero(); 8];
            for chunk in in_place.chunks_exact_mut(64) {
                for r in 0..8 {
                    rows0[r] = AvxStoreF::from_complex_ref(chunk.get_unchecked(8 * r..));
                }
                let mut mid0 = self.bf8_column.exec(rows0);
                for r in 1..8 {
                    mid0[r] = AvxStoreF::mul_by_complex(mid0[r], self.twiddles[r - 1]);
                }

                for r in 0..8 {
                    rows1[r] = AvxStoreF::from_complex_ref(chunk.get_unchecked(8 * r + 4..));
                }
                let mut mid1 = self.bf8_column.exec(rows1);
                for r in 1..8 {
                    mid1[r] = AvxStoreF::mul_by_complex(mid1[r], self.twiddles[r - 1 + 7]);
                }

                let (transposed0, transposed1) = transpose_8x8_f32(mid0, mid1);

                let output0 = self.bf8_column.exec(transposed0);
                for r in 0..8 {
                    output0[r].write(chunk.get_unchecked_mut(8 * r..));
                }

                let output1 = self.bf8_column.exec(transposed1);
                for r in 0..8 {
                    output1[r].write(chunk.get_unchecked_mut(8 * r + 4..));
                }
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly64f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly64f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(64) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(64) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows0 = [AvxStoreF::zero(); 8];
            let mut rows1 = [AvxStoreF::zero(); 8];

            for (dst, src) in dst.chunks_exact_mut(64).zip(src.chunks_exact(64)) {
                for r in 0..8 {
                    rows0[r] = AvxStoreF::from_complex_ref(src.get_unchecked(8 * r..));
                }
                let mut mid0 = self.bf8_column.exec(rows0);
                for r in 1..8 {
                    mid0[r] = AvxStoreF::mul_by_complex(mid0[r], self.twiddles[r - 1]);
                }

                for r in 0..8 {
                    rows1[r] = AvxStoreF::from_complex_ref(src.get_unchecked(8 * r + 4..));
                }
                let mut mid1 = self.bf8_column.exec(rows1);
                for r in 1..8 {
                    mid1[r] = AvxStoreF::mul_by_complex(mid1[r], self.twiddles[r - 1 + 7]);
                }

                let (transposed0, transposed1) = transpose_8x8_f32(mid0, mid1);

                let output0 = self.bf8_column.exec(transposed0);
                for r in 0..8 {
                    output0[r].write(dst.get_unchecked_mut(8 * r..));
                }

                let output1 = self.bf8_column.exec(transposed1);
                for r in 0..8 {
                    output1[r].write(dst.get_unchecked_mut(8 * r + 4..));
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly64f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly64, f32, AvxButterfly64f, 64, 1e-4);
    test_oof_avx_butterfly!(test_avx_butterfly36, f32, AvxButterfly64f, 64, 1e-4);
}
