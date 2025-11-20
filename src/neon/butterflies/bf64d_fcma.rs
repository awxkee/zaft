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

use crate::neon::mixed::{ColumnFcmaButterfly8d, NeonStoreD};
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::sync::Arc;

pub(crate) struct NeonFcmaButterfly64d {
    direction: FftDirection,
    bf8: ColumnFcmaButterfly8d,
    twiddles: [NeonStoreD; 64],
}

impl NeonFcmaButterfly64d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let mut twiddles = [NeonStoreD::default(); 64];
        let mut q = 0usize;
        let len_per_row = 8;
        const COMPLEX_PER_VECTOR: usize = 1;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..8 {
                twiddles[q] = NeonStoreD::from_complex(&compute_twiddle(
                    y * (x * COMPLEX_PER_VECTOR),
                    64,
                    fft_direction,
                ));
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf8: ColumnFcmaButterfly8d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly64d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
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

impl NeonFcmaButterfly64d {
    #[target_feature(enable = "fcma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 64 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [NeonStoreD; 8] = [NeonStoreD::default(); 8];
            let mut scratch = [Complex::<f64>::default(); 64];

            for chunk in in_place.chunks_exact_mut(64) {
                // columns
                for k in 0..8 {
                    for i in 0..8 {
                        rows[i] = NeonStoreD::from_complex_ref(chunk.get_unchecked(i * 8 + k..));
                    }

                    rows = self.bf8.exec(rows);

                    for i in 1..8 {
                        rows[i] = NeonStoreD::fcmul_fcma(rows[i], self.twiddles[i - 1 + 7 * k]);
                    }

                    for i in 0..8 {
                        rows[i].write(scratch.get_unchecked_mut(k * 8 + i..));
                    }
                }

                // rows

                for k in 0..8 {
                    for i in 0..8 {
                        rows[i] = NeonStoreD::from_complex_ref(scratch.get_unchecked(i * 8 + k..));
                    }
                    rows = self.bf8.exec(rows);
                    for i in 0..8 {
                        rows[i].write(chunk.get_unchecked_mut(i * 8 + k..));
                    }
                }
            }
        }
        Ok(())
    }
}
impl FftExecutorOutOfPlace<f64> for NeonFcmaButterfly64d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl NeonFcmaButterfly64d {
    #[target_feature(enable = "fcma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 64 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 64 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        let mut rows: [NeonStoreD; 8] = [NeonStoreD::default(); 8];
        let mut scratch = [Complex::<f64>::default(); 64];

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(64).zip(src.chunks_exact(64)) {
                // columns
                for k in 0..8 {
                    for i in 0..8 {
                        rows[i] = NeonStoreD::from_complex_ref(src.get_unchecked(i * 8 + k..));
                    }

                    rows = self.bf8.exec(rows);

                    for i in 1..8 {
                        rows[i] = NeonStoreD::fcmul_fcma(rows[i], self.twiddles[i - 1 + 7 * k]);
                    }

                    for i in 0..8 {
                        rows[i].write(scratch.get_unchecked_mut(k * 8 + i..));
                    }
                }

                // rows

                for k in 0..8 {
                    for i in 0..8 {
                        rows[i] = NeonStoreD::from_complex_ref(scratch.get_unchecked(i * 8 + k..));
                    }
                    rows = self.bf8.exec(rows);
                    for i in 0..8 {
                        rows[i].write(dst.get_unchecked_mut(i * 8 + k..));
                    }
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for NeonFcmaButterfly64d {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_fcma_butterfly!(
        test_neon_butterfly64_f64,
        f64,
        NeonFcmaButterfly64d,
        64,
        1e-7
    );
    test_oof_fcma_butterfly!(
        test_oof_neon_butterfly64_f64,
        f64,
        NeonFcmaButterfly64d,
        64,
        1e-7
    );
}
