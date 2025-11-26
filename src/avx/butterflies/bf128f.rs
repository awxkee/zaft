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

use crate::avx::mixed::{AvxStoreF, ColumnButterfly16f, ColumnButterfly8f};
use crate::avx::transpose::transpose_4x8;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;
use std::sync::Arc;

pub(crate) struct AvxButterfly128f {
    direction: FftDirection,
    bf16: ColumnButterfly16f,
    bf8: ColumnButterfly8f,
    twiddles: [AvxStoreF; 28],
}

impl AvxButterfly128f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        let mut twiddles = [AvxStoreF::zero(); 28];
        let mut q = 0usize;
        let len_per_row = 16;
        const COMPLEX_PER_VECTOR: usize = 4;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..8 {
                twiddles[q] = AvxStoreF::set_complex4(
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 128, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 128, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 2), 128, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 3), 128, fft_direction),
                );
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf16: ColumnButterfly16f::new(fft_direction),
            bf8: ColumnButterfly8f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly128f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        128
    }
}

impl AvxButterfly128f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 128 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows16: [AvxStoreF; 16] = [AvxStoreF::zero(); 16];
            let mut rows: [AvxStoreF; 8] = [AvxStoreF::zero(); 8];
            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 128];

            for chunk in in_place.chunks_exact_mut(128) {
                // columns
                for k in 0..4 {
                    for i in 0..8 {
                        rows[i] =
                            AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 16 + k * 4..));
                    }

                    rows = self.bf8.exec(rows);

                    for i in 1..8 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 7 * k]);
                    }

                    let transposed = transpose_4x8(rows);

                    for i in 0..2 {
                        transposed[i * 4].write_u(scratch.get_unchecked_mut(k * 4 * 8 + i * 4..));
                        transposed[i * 4 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 8 + i * 4..));
                        transposed[i * 4 + 2]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 2) * 8 + i * 4..));
                        transposed[i * 4 + 3]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 3) * 8 + i * 4..));
                    }
                }

                // rows

                for k in 0..2 {
                    for i in 0..16 {
                        rows16[i] =
                            AvxStoreF::from_complex_refu(scratch.get_unchecked(i * 8 + k * 4..));
                    }
                    rows16 = self.bf16.exec(rows16);
                    for i in 0..16 {
                        rows16[i].write(chunk.get_unchecked_mut(i * 8 + k * 4..));
                    }
                }
            }
        }
        Ok(())
    }
}
impl FftExecutorOutOfPlace<f32> for AvxButterfly128f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly128f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 128 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 128 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows16: [AvxStoreF; 16] = [AvxStoreF::zero(); 16];
            let mut rows: [AvxStoreF; 8] = [AvxStoreF::zero(); 8];
            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 128];

            for (dst, src) in dst.chunks_exact_mut(128).zip(src.chunks_exact(128)) {
                // columns
                for k in 0..4 {
                    for i in 0..8 {
                        rows[i] = AvxStoreF::from_complex_ref(src.get_unchecked(i * 16 + k * 4..));
                    }

                    rows = self.bf8.exec(rows);

                    for i in 1..8 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 7 * k]);
                    }

                    let transposed = transpose_4x8(rows);

                    for i in 0..2 {
                        transposed[i * 4].write_u(scratch.get_unchecked_mut(k * 4 * 8 + i * 4..));
                        transposed[i * 4 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 8 + i * 4..));
                        transposed[i * 4 + 2]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 2) * 8 + i * 4..));
                        transposed[i * 4 + 3]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 3) * 8 + i * 4..));
                    }
                }

                // rows

                for k in 0..2 {
                    for i in 0..16 {
                        rows16[i] =
                            AvxStoreF::from_complex_refu(scratch.get_unchecked(i * 8 + k * 4..));
                    }
                    rows16 = self.bf16.exec(rows16);
                    for i in 0..16 {
                        rows16[i].write(dst.get_unchecked_mut(i * 8 + k * 4..));
                    }
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly128f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly128, f32, AvxButterfly128f, 128, 1e-3);
    // test_oof_avx_butterfly!(test_oof_avx_butterfly128, f32, AvxButterfly128f, 128, 1e-3);
}
