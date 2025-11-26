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

use crate::neon::mixed::{ColumnButterfly10f, NeonStoreF};
use crate::neon::transpose::transpose_2x10;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;
use std::sync::Arc;

pub(crate) struct NeonButterfly100f {
    direction: FftDirection,
    bf10: ColumnButterfly10f,
    twiddles: [NeonStoreF; 45],
}

impl NeonButterfly100f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let mut twiddles = [NeonStoreF::default(); 45];
        let mut q = 0usize;
        let len_per_row = 10;
        const COMPLEX_PER_VECTOR: usize = 2;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..10 {
                twiddles[q] = NeonStoreF::from_complex2(
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 100, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 100, fft_direction),
                );
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf10: ColumnButterfly10f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for NeonButterfly100f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        self.execute_impl(in_place)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        100
    }
}

impl NeonButterfly100f {
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 100 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [NeonStoreF; 10] = [NeonStoreF::default(); 10];
            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 100];

            for chunk in in_place.chunks_exact_mut(100) {
                // columns
                for k in 0..5 {
                    for i in 0..10 {
                        rows[i] =
                            NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 10 + k * 2..));
                    }

                    rows = self.bf10.exec(rows);

                    for i in 1..10 {
                        rows[i] = NeonStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 9 * k]);
                    }

                    let transposed = transpose_2x10(rows);

                    for i in 0..5 {
                        transposed[i * 2]
                            .write_uninit(scratch.get_unchecked_mut(k * 2 * 10 + i * 2..));
                        transposed[i * 2 + 1]
                            .write_uninit(scratch.get_unchecked_mut((k * 2 + 1) * 10 + i * 2..));
                    }
                }

                // rows

                for k in 0..5 {
                    for i in 0..10 {
                        rows[i] =
                            NeonStoreF::from_complex_refu(scratch.get_unchecked(i * 10 + k * 2..));
                    }
                    rows = self.bf10.exec(rows);
                    for i in 0..10 {
                        rows[i].write(chunk.get_unchecked_mut(i * 10 + k * 2..));
                    }
                }
            }
        }
        Ok(())
    }
}
impl FftExecutorOutOfPlace<f32> for NeonButterfly100f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_impl(src, dst)
    }
}

impl NeonButterfly100f {
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 100 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 100 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows: [NeonStoreF; 10] = [NeonStoreF::default(); 10];
            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 100];

            for (dst, src) in dst.chunks_exact_mut(100).zip(src.chunks_exact(100)) {
                // columns
                for k in 0..5 {
                    for i in 0..10 {
                        rows[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 10 + k * 2..));
                    }

                    rows = self.bf10.exec(rows);

                    for i in 1..10 {
                        rows[i] = NeonStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 9 * k]);
                    }

                    let transposed = transpose_2x10(rows);

                    for i in 0..5 {
                        transposed[i * 2]
                            .write_uninit(scratch.get_unchecked_mut(k * 2 * 10 + i * 2..));
                        transposed[i * 2 + 1]
                            .write_uninit(scratch.get_unchecked_mut((k * 2 + 1) * 10 + i * 2..));
                    }
                }

                // rows

                for k in 0..5 {
                    for i in 0..10 {
                        rows[i] =
                            NeonStoreF::from_complex_refu(scratch.get_unchecked(i * 10 + k * 2..));
                    }
                    rows = self.bf10.exec(rows);
                    for i in 0..10 {
                        rows[i].write(dst.get_unchecked_mut(i * 10 + k * 2..));
                    }
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for NeonButterfly100f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_neon_butterfly100, f32, NeonButterfly100f, 100, 1e-3);
    test_oof_butterfly!(
        test_oof_neon_butterfly100,
        f32,
        NeonButterfly100f,
        100,
        1e-3
    );
}
