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

use crate::neon::butterflies::gen_butterfly_twiddles_separated_columns_f32;
use crate::neon::mixed::{ColumnButterfly8f, NeonStoreF};
use crate::neon::transpose::transpose_8x8_f32;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::sync::Arc;

pub(crate) struct NeonButterfly64f {
    direction: FftDirection,
    twiddles: [NeonStoreF; 28],
    bf8_column: ColumnButterfly8f,
}

impl NeonButterfly64f {
    pub(crate) fn new(direction: FftDirection) -> Self {
        NeonButterfly64f {
            direction,
            twiddles: gen_butterfly_twiddles_separated_columns_f32!(8, 8, 0, direction),
            bf8_column: ColumnButterfly8f::new(direction),
        }
    }
}

impl FftExecutor<f32> for NeonButterfly64f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 64 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0 = [NeonStoreF::default(); 8];
            let mut rows1 = [NeonStoreF::default(); 8];
            let mut rows2 = [NeonStoreF::default(); 8];
            let mut rows3 = [NeonStoreF::default(); 8];

            for chunk in in_place.chunks_exact_mut(64) {
                for r in 0..8 {
                    rows0[r] = NeonStoreF::from_complex_ref(chunk.get_unchecked(8 * r..));
                }
                let mut mid0 = self.bf8_column.exec(rows0);

                for r in 1..8 {
                    mid0[r] = NeonStoreF::mul_by_complex(mid0[r], self.twiddles[r - 1]);
                }

                for r in 0..8 {
                    rows1[r] = NeonStoreF::from_complex_ref(chunk.get_unchecked(8 * r + 2..));
                }

                let mut mid1 = self.bf8_column.exec(rows1);

                for r in 1..8 {
                    mid1[r] = NeonStoreF::mul_by_complex(mid1[r], self.twiddles[r - 1 + 7]);
                }

                for r in 0..8 {
                    rows2[r] = NeonStoreF::from_complex_ref(chunk.get_unchecked(8 * r + 4..));
                }

                let mut mid2 = self.bf8_column.exec(rows2);

                for r in 1..8 {
                    mid2[r] = NeonStoreF::mul_by_complex(mid2[r], self.twiddles[r - 1 + 7 * 2]);
                }

                for r in 0..8 {
                    rows3[r] = NeonStoreF::from_complex_ref(chunk.get_unchecked(8 * r + 6..));
                }

                let mut mid3 = self.bf8_column.exec(rows3);

                for r in 1..8 {
                    mid3[r] = NeonStoreF::mul_by_complex(mid3[r], self.twiddles[r - 1 + 7 * 3]);
                }

                let (transposed0, transposed1, transposed2, transposed3) =
                    transpose_8x8_f32(mid0, mid1, mid2, mid3);

                let output0 = self.bf8_column.exec(transposed0);
                for r in 0..8 {
                    output0[r].write(chunk.get_unchecked_mut(8 * r..));
                }

                let output1 = self.bf8_column.exec(transposed1);
                for r in 0..8 {
                    output1[r].write(chunk.get_unchecked_mut(8 * r + 2..));
                }

                let output2 = self.bf8_column.exec(transposed2);
                for r in 0..8 {
                    output2[r].write(chunk.get_unchecked_mut(8 * r + 4..));
                }

                let output3 = self.bf8_column.exec(transposed3);
                for r in 0..8 {
                    output3[r].write(chunk.get_unchecked_mut(8 * r + 6..));
                }
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        64
    }
}

impl FftExecutorOutOfPlace<f32> for NeonButterfly64f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_impl(src, dst)
    }
}

impl NeonButterfly64f {
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 64 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 64 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows0 = [NeonStoreF::default(); 8];
            let mut rows1 = [NeonStoreF::default(); 8];
            let mut rows2 = [NeonStoreF::default(); 8];
            let mut rows3 = [NeonStoreF::default(); 8];

            for (dst, src) in dst.chunks_exact_mut(64).zip(src.chunks_exact(64)) {
                for r in 0..8 {
                    rows0[r] = NeonStoreF::from_complex_ref(src.get_unchecked(8 * r..));
                }
                let mut mid0 = self.bf8_column.exec(rows0);

                for r in 1..8 {
                    mid0[r] = NeonStoreF::mul_by_complex(mid0[r], self.twiddles[r - 1]);
                }

                for r in 0..8 {
                    rows1[r] = NeonStoreF::from_complex_ref(src.get_unchecked(8 * r + 2..));
                }

                let mut mid1 = self.bf8_column.exec(rows1);

                for r in 1..8 {
                    mid1[r] = NeonStoreF::mul_by_complex(mid1[r], self.twiddles[r - 1 + 7]);
                }

                for r in 0..8 {
                    rows2[r] = NeonStoreF::from_complex_ref(src.get_unchecked(8 * r + 4..));
                }

                let mut mid2 = self.bf8_column.exec(rows2);

                for r in 1..8 {
                    mid2[r] = NeonStoreF::mul_by_complex(mid2[r], self.twiddles[r - 1 + 7 * 2]);
                }

                for r in 0..8 {
                    rows3[r] = NeonStoreF::from_complex_ref(src.get_unchecked(8 * r + 6..));
                }

                let mut mid3 = self.bf8_column.exec(rows3);

                for r in 1..8 {
                    mid3[r] = NeonStoreF::mul_by_complex(mid3[r], self.twiddles[r - 1 + 7 * 3]);
                }

                let (transposed0, transposed1, transposed2, transposed3) =
                    transpose_8x8_f32(mid0, mid1, mid2, mid3);

                let output0 = self.bf8_column.exec(transposed0);
                for r in 0..8 {
                    output0[r].write(dst.get_unchecked_mut(8 * r..));
                }

                let output1 = self.bf8_column.exec(transposed1);
                for r in 0..8 {
                    output1[r].write(dst.get_unchecked_mut(8 * r + 2..));
                }

                let output2 = self.bf8_column.exec(transposed2);
                for r in 0..8 {
                    output2[r].write(dst.get_unchecked_mut(8 * r + 4..));
                }

                let output3 = self.bf8_column.exec(transposed3);
                for r in 0..8 {
                    output3[r].write(dst.get_unchecked_mut(8 * r + 6..));
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for NeonButterfly64f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_avx_butterfly64, f32, NeonButterfly64f, 64, 1e-4);
    test_oof_butterfly!(test_avx_butterfly36, f32, NeonButterfly64f, 64, 1e-4);
}
