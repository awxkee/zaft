/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use crate::avx::butterflies::{AvxButterfly, gen_butterfly_twiddles_interleaved_columns_f64};
use crate::avx::mixed::{AvxStoreD, ColumnButterfly8d};
use crate::avx::transpose::transpose_f64x2_2x2;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::sync::Arc;

pub(crate) struct AvxButterfly32d {
    direction: FftDirection,
    twiddles: [AvxStoreD; 12],
    bf8_column: ColumnButterfly8d,
}

#[inline]
#[target_feature(enable = "avx")]
fn transpose_8x4_to_4x8_f64(
    rows0: [AvxStoreD; 4],
    rows1: [AvxStoreD; 4],
    rows2: [AvxStoreD; 4],
    rows3: [AvxStoreD; 4],
) -> ([AvxStoreD; 8], [AvxStoreD; 8]) {
    let output00 = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);
    let output01 = transpose_f64x2_2x2(rows1[0].v, rows1[1].v);
    let output02 = transpose_f64x2_2x2(rows2[0].v, rows2[1].v);
    let output03 = transpose_f64x2_2x2(rows3[0].v, rows3[1].v);
    let output10 = transpose_f64x2_2x2(rows0[2].v, rows0[3].v);
    let output11 = transpose_f64x2_2x2(rows1[2].v, rows1[3].v);
    let output12 = transpose_f64x2_2x2(rows2[2].v, rows2[3].v);
    let output13 = transpose_f64x2_2x2(rows3[2].v, rows3[3].v);

    (
        [
            AvxStoreD::raw(output00.0),
            AvxStoreD::raw(output00.1),
            AvxStoreD::raw(output01.0),
            AvxStoreD::raw(output01.1),
            AvxStoreD::raw(output02.0),
            AvxStoreD::raw(output02.1),
            AvxStoreD::raw(output03.0),
            AvxStoreD::raw(output03.1),
        ],
        [
            AvxStoreD::raw(output10.0),
            AvxStoreD::raw(output10.1),
            AvxStoreD::raw(output11.0),
            AvxStoreD::raw(output11.1),
            AvxStoreD::raw(output12.0),
            AvxStoreD::raw(output12.1),
            AvxStoreD::raw(output13.0),
            AvxStoreD::raw(output13.1),
        ],
    )
}

impl AvxButterfly32d {
    pub(crate) fn new(direction: FftDirection) -> AvxButterfly32d {
        Self {
            direction,
            twiddles: unsafe {
                gen_butterfly_twiddles_interleaved_columns_f64!(4, 8, 0, direction)
            },
            bf8_column: unsafe { ColumnButterfly8d::new(direction) },
        }
    }
}

impl AvxButterfly32d {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0 = [AvxStoreD::zero(); 4];
            let mut rows1 = [AvxStoreD::zero(); 4];
            for chunk in in_place.chunks_exact_mut(32) {
                for r in 0..4 {
                    rows0[r] = AvxStoreD::from_complex_ref(chunk.get_unchecked(8 * r..));
                    rows1[r] = AvxStoreD::from_complex_ref(chunk.get_unchecked(8 * r + 2..));
                }
                let mut mid0 =
                    AvxButterfly::qbutterfly4_f64(rows0, self.bf8_column.rotate.rot_flag);
                let mut mid1 =
                    AvxButterfly::qbutterfly4_f64(rows1, self.bf8_column.rotate.rot_flag);
                for r in 1..4 {
                    mid0[r] =
                        AvxStoreD::mul_by_complex(mid0[r], *self.twiddles.get_unchecked(4 * r - 4));
                    mid1[r] =
                        AvxStoreD::mul_by_complex(mid1[r], *self.twiddles.get_unchecked(4 * r - 3));
                }

                // One half is done, so the compiler can spill everything above this. Now do the other set of columns
                let mut rows2 = [AvxStoreD::zero(); 4];
                let mut rows3 = [AvxStoreD::zero(); 4];
                for r in 0..4 {
                    rows2[r] = AvxStoreD::from_complex_ref(chunk.get_unchecked(8 * r + 4..));
                    rows3[r] = AvxStoreD::from_complex_ref(chunk.get_unchecked(8 * r + 6..));
                }
                let mut mid2 =
                    AvxButterfly::qbutterfly4_f64(rows2, self.bf8_column.rotate.rot_flag);
                let mut mid3 =
                    AvxButterfly::qbutterfly4_f64(rows3, self.bf8_column.rotate.rot_flag);
                for r in 1..4 {
                    mid2[r] = AvxStoreD::mul_by_complex(mid2[r], self.twiddles[4 * r - 2]);
                    mid3[r] = AvxStoreD::mul_by_complex(mid3[r], self.twiddles[4 * r - 1]);
                }

                // Transpose our 8x4 array to a 4x8 array
                let (transposed0, transposed1) = transpose_8x4_to_4x8_f64(mid0, mid1, mid2, mid3);

                // Do 4 butterfly 8's down columns of the transposed array
                let output0 = self.bf8_column.exec(transposed0);
                #[allow(clippy::needless_range_loop)]
                for r in 0..8 {
                    output0[r].write(chunk.get_unchecked_mut(4 * r..));
                }
                let output1 = self.bf8_column.exec(transposed1);
                #[allow(clippy::needless_range_loop)]
                for r in 0..8 {
                    output1[r].write(chunk.get_unchecked_mut(4 * r + 2..));
                }
            }

            Ok(())
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_out_of_place_f64(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows0 = [AvxStoreD::zero(); 4];
            let mut rows1 = [AvxStoreD::zero(); 4];
            for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
                for r in 0..4 {
                    rows0[r] = AvxStoreD::from_complex_ref(src.get_unchecked(8 * r..));
                    rows1[r] = AvxStoreD::from_complex_ref(src.get_unchecked(8 * r + 2..));
                }
                let mut mid0 =
                    AvxButterfly::qbutterfly4_f64(rows0, self.bf8_column.rotate.rot_flag);
                let mut mid1 =
                    AvxButterfly::qbutterfly4_f64(rows1, self.bf8_column.rotate.rot_flag);
                for r in 1..4 {
                    mid0[r] =
                        AvxStoreD::mul_by_complex(mid0[r], *self.twiddles.get_unchecked(4 * r - 4));
                    mid1[r] =
                        AvxStoreD::mul_by_complex(mid1[r], *self.twiddles.get_unchecked(4 * r - 3));
                }

                let mut rows2 = [AvxStoreD::zero(); 4];
                let mut rows3 = [AvxStoreD::zero(); 4];
                for r in 0..4 {
                    rows2[r] = AvxStoreD::from_complex_ref(src.get_unchecked(8 * r + 4..));
                    rows3[r] = AvxStoreD::from_complex_ref(src.get_unchecked(8 * r + 6..));
                }
                let mut mid2 =
                    AvxButterfly::qbutterfly4_f64(rows2, self.bf8_column.rotate.rot_flag);
                let mut mid3 =
                    AvxButterfly::qbutterfly4_f64(rows3, self.bf8_column.rotate.rot_flag);
                for r in 1..4 {
                    mid2[r] = AvxStoreD::mul_by_complex(mid2[r], self.twiddles[4 * r - 2]);
                    mid3[r] = AvxStoreD::mul_by_complex(mid3[r], self.twiddles[4 * r - 1]);
                }

                // Transpose our 8x4 array to a 4x8 array
                let (transposed0, transposed1) = transpose_8x4_to_4x8_f64(mid0, mid1, mid2, mid3);

                // Do 4 butterfly 8's down columns of the transposed array
                let output0 = self.bf8_column.exec(transposed0);
                #[allow(clippy::needless_range_loop)]
                for r in 0..8 {
                    output0[r].write(dst.get_unchecked_mut(4 * r..));
                }
                let output1 = self.bf8_column.exec(transposed1);
                #[allow(clippy::needless_range_loop)]
                for r in 0..8 {
                    output1[r].write(dst.get_unchecked_mut(4 * r + 2..));
                }
            }

            Ok(())
        }
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly32d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly32d {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f64> for AvxButterfly32d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly32_f64, f64, AvxButterfly32d, 32, 1e-7);
    test_oof_avx_butterfly!(test_oof_avx_butterfly32_f64, f64, AvxButterfly32d, 32, 1e-7);
}
