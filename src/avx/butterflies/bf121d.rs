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

use crate::avx::mixed::{AvxStoreD, ColumnButterfly11d};
use crate::avx::transpose::transpose_f64x2_2x11;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;
use std::sync::Arc;

pub(crate) struct AvxButterfly121d {
    direction: FftDirection,
    bf11: ColumnButterfly11d,
    twiddles: [AvxStoreD; 60],
}

impl AvxButterfly121d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        let mut twiddles = [AvxStoreD::zero(); 60];
        let mut q = 0usize;
        let len_per_row = 11;
        const COMPLEX_PER_VECTOR: usize = 2;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..11 {
                twiddles[q] = AvxStoreD::set_complex2(
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 121, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 121, fft_direction),
                );
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf11: ColumnButterfly11d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly121d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
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

impl AvxButterfly121d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 121 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [AvxStoreD; 11] = [AvxStoreD::zero(); 11];
            let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 121];

            for chunk in in_place.chunks_exact_mut(121) {
                // columns
                for k in 0..5 {
                    for i in 0..11 {
                        rows[i] =
                            AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 11 + k * 2..));
                    }

                    rows = self.bf11.exec(rows);

                    for i in 1..11 {
                        rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 10 * k]);
                    }

                    let transposed = transpose_f64x2_2x11(rows);

                    for i in 0..5 {
                        transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                        transposed[i * 2 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 2 + 1) * 11 + i * 2..));
                    }
                    {
                        let i = 5;
                        transposed[i * 2]
                            .write_lou(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                        transposed[i * 2 + 1]
                            .write_lou(scratch.get_unchecked_mut((k * 2 + 1) * 11 + i * 2..));
                    }
                }

                {
                    let k = 5;
                    for i in 0..11 {
                        rows[i] = AvxStoreD::from_complex(chunk.get_unchecked(i * 11 + k * 2));
                    }

                    rows = self.bf11.exec(rows);

                    for i in 1..11 {
                        rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 10 * k]);
                    }

                    let transposed = transpose_f64x2_2x11(rows);

                    for i in 0..5 {
                        transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                    }
                    {
                        let i = 5;
                        transposed[i * 2]
                            .write_lou(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                    }
                }

                // rows

                for k in 0..5 {
                    for i in 0..11 {
                        rows[i] =
                            AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 11 + k * 2..));
                    }
                    rows = self.bf11.exec(rows);
                    for i in 0..11 {
                        rows[i].write(chunk.get_unchecked_mut(i * 11 + k * 2..));
                    }
                }
                {
                    let k = 5;
                    for i in 0..11 {
                        rows[i] = AvxStoreD::from_complexu(scratch.get_unchecked(i * 11 + k * 2));
                    }
                    rows = self.bf11.exec(rows);
                    for i in 0..11 {
                        rows[i].write_lo(chunk.get_unchecked_mut(i * 11 + k * 2..));
                    }
                }
            }
        }
        Ok(())
    }
}
impl FftExecutorOutOfPlace<f64> for AvxButterfly121d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly121d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 121 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 121 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows: [AvxStoreD; 11] = [AvxStoreD::zero(); 11];
            let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 121];

            for (dst, src) in dst.chunks_exact_mut(121).zip(src.chunks_exact(121)) {
                // columns
                for k in 0..5 {
                    for i in 0..11 {
                        rows[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 11 + k * 2..));
                    }

                    rows = self.bf11.exec(rows);

                    for i in 1..11 {
                        rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 10 * k]);
                    }

                    let transposed = transpose_f64x2_2x11(rows);

                    for i in 0..5 {
                        transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                        transposed[i * 2 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 2 + 1) * 11 + i * 2..));
                    }
                    {
                        let i = 5;
                        transposed[i * 2]
                            .write_lou(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                        transposed[i * 2 + 1]
                            .write_lou(scratch.get_unchecked_mut((k * 2 + 1) * 11 + i * 2..));
                    }
                }

                {
                    let k = 5;
                    for i in 0..11 {
                        rows[i] = AvxStoreD::from_complex(src.get_unchecked(i * 11 + k * 2));
                    }

                    rows = self.bf11.exec(rows);

                    for i in 1..11 {
                        rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 10 * k]);
                    }

                    let transposed = transpose_f64x2_2x11(rows);

                    for i in 0..5 {
                        transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                    }
                    {
                        let i = 5;
                        transposed[i * 2]
                            .write_lou(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                    }
                }

                // rows

                for k in 0..5 {
                    for i in 0..11 {
                        rows[i] =
                            AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 11 + k * 2..));
                    }
                    rows = self.bf11.exec(rows);
                    for i in 0..11 {
                        rows[i].write(dst.get_unchecked_mut(i * 11 + k * 2..));
                    }
                }
                {
                    let k = 5;
                    for i in 0..11 {
                        rows[i] = AvxStoreD::from_complexu(scratch.get_unchecked(i * 11 + k * 2));
                    }
                    rows = self.bf11.exec(rows);
                    for i in 0..11 {
                        rows[i].write_lo(dst.get_unchecked_mut(i * 11 + k * 2..));
                    }
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly121d {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly121, f64, AvxButterfly121d, 121, 1e-3);
    test_oof_avx_butterfly!(test_oof_avx_butterfly121, f64, AvxButterfly121d, 121, 1e-3);
}
