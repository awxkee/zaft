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

use crate::avx::butterflies::shared::gen_butterfly_twiddles_f64;
use crate::avx::mixed::{AvxStoreD, ColumnButterfly16d};
use crate::avx::transpose::transpose_f64x2_2x16;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;
use std::sync::Arc;

pub(crate) struct AvxButterfly256d {
    direction: FftDirection,
    bf16: ColumnButterfly16d,
    twiddles: [AvxStoreD; 120],
}

impl AvxButterfly256d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(16, 16, fft_direction, 256),
            bf16: ColumnButterfly16d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly256d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        256
    }
}

impl AvxButterfly256d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(256) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [AvxStoreD; 16] = [AvxStoreD::zero(); 16];
            let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 256];

            for chunk in in_place.chunks_exact_mut(256) {
                // columns
                for k in 0..8 {
                    for i in 0..16 {
                        rows[i] =
                            AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 16 + k * 2..));
                    }

                    rows = self.bf16.exec(rows);

                    for i in 1..16 {
                        rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 15 * k]);
                    }

                    let transposed = transpose_f64x2_2x16(rows);

                    for i in 0..8 {
                        transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 16 + i * 2..));
                        transposed[i * 2 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 2 + 1) * 16 + i * 2..));
                    }
                }

                // rows

                for k in 0..8 {
                    for i in 0..16 {
                        rows[i] =
                            AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 16 + k * 2..));
                    }
                    rows = self.bf16.exec(rows);
                    for i in 0..16 {
                        rows[i].write(chunk.get_unchecked_mut(i * 16 + k * 2..));
                    }
                }
            }
        }
        Ok(())
    }
}
impl FftExecutorOutOfPlace<f64> for AvxButterfly256d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly256d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(256) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(256) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows: [AvxStoreD; 16] = [AvxStoreD::zero(); 16];
            let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 256];

            for (dst, src) in dst.chunks_exact_mut(256).zip(src.chunks_exact(256)) {
                // columns
                for k in 0..8 {
                    for i in 0..16 {
                        rows[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 16 + k * 2..));
                    }

                    rows = self.bf16.exec(rows);

                    for i in 1..16 {
                        rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 15 * k]);
                    }

                    let transposed = transpose_f64x2_2x16(rows);

                    for i in 0..8 {
                        transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 16 + i * 2..));
                        transposed[i * 2 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 2 + 1) * 16 + i * 2..));
                    }
                }

                // rows

                for k in 0..8 {
                    for i in 0..16 {
                        rows[i] =
                            AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 16 + k * 2..));
                    }
                    rows = self.bf16.exec(rows);
                    for i in 0..16 {
                        rows[i].write(dst.get_unchecked_mut(i * 16 + k * 2..));
                    }
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly256d {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly256_f64, f64, AvxButterfly256d, 256, 1e-8);
    test_oof_avx_butterfly!(
        test_oof_avx_butterfly256_f64,
        f64,
        AvxButterfly256d,
        256,
        1e-8
    );
}
