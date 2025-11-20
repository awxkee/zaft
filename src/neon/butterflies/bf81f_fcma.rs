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

use crate::neon::mixed::{ColumnFcmaButterfly9f, NeonStoreF};
use crate::neon::transpose::transpose_2x9;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;

pub(crate) struct NeonFcmaButterfly81f {
    direction: FftDirection,
    bf9: ColumnFcmaButterfly9f,
    twiddles: [NeonStoreF; 40],
}

impl NeonFcmaButterfly81f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let mut twiddles = [NeonStoreF::default(); 40];
        let mut q = 0usize;
        let len_per_row = 9;
        const COMPLEX_PER_VECTOR: usize = 2;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..9 {
                twiddles[q] = NeonStoreF::from_complex2(
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 81, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 81, fft_direction),
                );
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf9: ColumnFcmaButterfly9f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for NeonFcmaButterfly81f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        81
    }
}

impl NeonFcmaButterfly81f {
    #[target_feature(enable = "fcma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 81 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [NeonStoreF; 9] = [NeonStoreF::default(); 9];
            let mut scratch = [Complex::<f32>::default(); 81];

            for chunk in in_place.chunks_exact_mut(81) {
                // columns
                for k in 0..4 {
                    for i in 0..9 {
                        rows[i] =
                            NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 9 + k * 2..));
                    }

                    rows = self.bf9.exec(rows);

                    for i in 1..9 {
                        rows[i] = NeonStoreF::fcmul_fcma(rows[i], self.twiddles[i - 1 + 8 * k]);
                    }

                    let transposed = transpose_2x9(rows);

                    for i in 0..4 {
                        transposed[i * 2].write(scratch.get_unchecked_mut(k * 2 * 9 + i * 2..));
                        transposed[i * 2 + 1]
                            .write(scratch.get_unchecked_mut((k * 2 + 1) * 9 + i * 2..));
                    }

                    transposed[8].write_lo(scratch.get_unchecked_mut(k * 2 * 9 + 4 * 2..));
                    transposed[8 + 1]
                        .write_lo(scratch.get_unchecked_mut((k * 2 + 1) * 9 + 4 * 2..));
                }

                {
                    let k = 8;
                    for i in 0..9 {
                        rows[i] = NeonStoreF::from_complex(chunk.get_unchecked(i * 9 + k));
                    }

                    rows = self.bf9.exec(rows);

                    for i in 1..9 {
                        rows[i] = NeonStoreF::fcmul_fcma(rows[i], self.twiddles[i - 1 + 32]);
                    }

                    let transposed = transpose_2x9(rows);

                    for i in 0..4 {
                        transposed[i * 2].write(scratch.get_unchecked_mut(8 * 9 + i * 2..));
                    }

                    transposed[8].write_lo(scratch.get_unchecked_mut(8 * 9 + 4 * 2..));
                }

                // rows

                for k in 0..4 {
                    for i in 0..9 {
                        rows[i] =
                            NeonStoreF::from_complex_ref(scratch.get_unchecked(i * 9 + k * 2..));
                    }
                    rows = self.bf9.exec(rows);
                    for i in 0..9 {
                        rows[i].write(chunk.get_unchecked_mut(i * 9 + k * 2..));
                    }
                }

                {
                    let k = 8;
                    for i in 0..9 {
                        rows[i] = NeonStoreF::from_complex(scratch.get_unchecked(i * 9 + k));
                    }
                    rows = self.bf9.exec(rows);
                    for i in 0..9 {
                        rows[i].write_lo(chunk.get_unchecked_mut(i * 9 + k..));
                    }
                }
            }
        }
        Ok(())
    }
}
impl FftExecutorOutOfPlace<f32> for NeonFcmaButterfly81f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl NeonFcmaButterfly81f {
    #[target_feature(enable = "fcma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 81 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 81 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows: [NeonStoreF; 9] = [NeonStoreF::default(); 9];
            let mut scratch = [Complex::<f32>::default(); 81];

            for (dst, src) in dst.chunks_exact_mut(81).zip(src.chunks_exact(81)) {
                // columns
                for k in 0..4 {
                    for i in 0..9 {
                        rows[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 9 + k * 2..));
                    }

                    rows = self.bf9.exec(rows);

                    for i in 1..9 {
                        rows[i] = NeonStoreF::fcmul_fcma(rows[i], self.twiddles[i - 1 + 8 * k]);
                    }

                    let transposed = transpose_2x9(rows);

                    for i in 0..4 {
                        transposed[i * 2].write(scratch.get_unchecked_mut(k * 2 * 9 + i * 2..));
                        transposed[i * 2 + 1]
                            .write(scratch.get_unchecked_mut((k * 2 + 1) * 9 + i * 2..));
                    }

                    transposed[8].write_lo(scratch.get_unchecked_mut(k * 2 * 9 + 4 * 2..));
                    transposed[8 + 1]
                        .write_lo(scratch.get_unchecked_mut((k * 2 + 1) * 9 + 4 * 2..));
                }

                {
                    let k = 8;
                    for i in 0..9 {
                        rows[i] = NeonStoreF::from_complex(src.get_unchecked(i * 9 + k));
                    }

                    rows = self.bf9.exec(rows);

                    for i in 1..9 {
                        rows[i] = NeonStoreF::fcmul_fcma(rows[i], self.twiddles[i - 1 + 32]);
                    }

                    let transposed = transpose_2x9(rows);

                    for i in 0..4 {
                        transposed[i * 2].write(scratch.get_unchecked_mut(8 * 9 + i * 2..));
                    }

                    transposed[8].write_lo(scratch.get_unchecked_mut(8 * 9 + 4 * 2..));
                }

                // rows

                for k in 0..4 {
                    for i in 0..9 {
                        rows[i] =
                            NeonStoreF::from_complex_ref(scratch.get_unchecked(i * 9 + k * 2..));
                    }
                    rows = self.bf9.exec(rows);
                    for i in 0..9 {
                        rows[i].write(dst.get_unchecked_mut(i * 9 + k * 2..));
                    }
                }

                {
                    let k = 8;
                    for i in 0..9 {
                        rows[i] = NeonStoreF::from_complex(scratch.get_unchecked(i * 9 + k));
                    }
                    rows = self.bf9.exec(rows);
                    for i in 0..9 {
                        rows[i].write_lo(dst.get_unchecked_mut(i * 9 + k..));
                    }
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for NeonFcmaButterfly81f {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_fcma_butterfly!(test_neon_butterfly81, f32, NeonFcmaButterfly81f, 81, 1e-3);
    test_oof_fcma_butterfly!(
        test_oof_neon_butterfly81,
        f32,
        NeonFcmaButterfly81f,
        81,
        1e-3
    );
}
