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
use crate::avx::butterflies::gen_butterfly_twiddles_interleaved_columns_f32;
use crate::avx::butterflies::shared::boring_avx_butterfly;
use crate::avx::mixed::{AvxStoreF, ColumnButterfly8f};
use crate::avx::transpose::avx_transpose_f32x2_4x4_impl;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, R2CFftExecutor, ZaftError};
use num_complex::Complex;

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_8x4_to_4x8_f32(rows0: [AvxStoreF; 4], rows1: [AvxStoreF; 4]) -> [AvxStoreF; 8] {
    let transposed0 = avx_transpose_f32x2_4x4_impl(rows0[0].v, rows0[1].v, rows0[2].v, rows0[3].v);
    let transposed1 = avx_transpose_f32x2_4x4_impl(rows1[0].v, rows1[1].v, rows1[2].v, rows1[3].v);

    [
        AvxStoreF::raw(transposed0.0),
        AvxStoreF::raw(transposed0.1),
        AvxStoreF::raw(transposed0.2),
        AvxStoreF::raw(transposed0.3),
        AvxStoreF::raw(transposed1.0),
        AvxStoreF::raw(transposed1.1),
        AvxStoreF::raw(transposed1.2),
        AvxStoreF::raw(transposed1.3),
    ]
}

pub(crate) struct AvxButterfly32f {
    direction: FftDirection,
    twiddles: [AvxStoreF; 6],
    bf8: ColumnButterfly8f,
}

impl AvxButterfly32f {
    pub(crate) fn new(direction: FftDirection) -> AvxButterfly32f {
        Self {
            direction,
            twiddles: unsafe {
                gen_butterfly_twiddles_interleaved_columns_f32!(4, 8, 0, direction)
            },
            bf8: unsafe { ColumnButterfly8f::new(direction) },
        }
    }
}

boring_avx_butterfly!(AvxButterfly32f, f32, 32);

impl AvxButterfly32f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows0 = [AvxStoreF::zero(); 4];
        let mut rows1 = [AvxStoreF::zero(); 4];
        for r in 0..4 {
            rows0[r] = AvxStoreF::from_complex_ref(chunk.slice_from(8 * r..));
            rows1[r] = AvxStoreF::from_complex_ref(chunk.slice_from(8 * r + 4..));
        }

        let mut mid0 = self.bf8.bf4.exec(rows0);
        let mut mid1 = self.bf8.bf4.exec(rows1);

        // apply twiddle factors
        for r in 1..4 {
            mid0[r] = AvxStoreF::mul_by_complex(mid0[r], self.twiddles[2 * r - 2]);
            mid1[r] = AvxStoreF::mul_by_complex(mid1[r], self.twiddles[2 * r - 1]);
        }

        // Transpose our 8x4 array to an 4x8 array
        let transposed = transpose_8x4_to_4x8_f32(mid0, mid1);

        let output_rows = self.bf8.exec(transposed);

        output_rows[0].write(chunk.slice_from_mut(0..));
        output_rows[1].write(chunk.slice_from_mut(4..));
        output_rows[2].write(chunk.slice_from_mut(8..));
        output_rows[3].write(chunk.slice_from_mut(12..));
        output_rows[4].write(chunk.slice_from_mut(16..));
        output_rows[5].write(chunk.slice_from_mut(20..));
        output_rows[6].write(chunk.slice_from_mut(24..));
        output_rows[7].write(chunk.slice_from_mut(28..));
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_r2c(&self, src: &[f32], dst: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(32) {
            return Err(ZaftError::InvalidSizeMultiplier(
                src.len(),
                self.real_length(),
            ));
        }
        if !dst.len().is_multiple_of(17) {
            return Err(ZaftError::InvalidSizeMultiplier(
                dst.len(),
                self.complex_length(),
            ));
        }
        if src.len() / 32 != dst.len() / 17 {
            return Err(ZaftError::InvalidSamplesCount(
                src.len() / 32,
                dst.len() / 17,
            ));
        }

        unsafe {
            let mut rows0 = [AvxStoreF::zero(); 4];
            let mut rows1 = [AvxStoreF::zero(); 4];
            for (dst, src) in dst.chunks_exact_mut(17).zip(src.chunks_exact(32)) {
                for r in 0..4 {
                    let [u0, u1] = AvxStoreF::load(src.get_unchecked(8 * r..)).to_complex();
                    rows0[r] = u0;
                    rows1[r] = u1;
                }

                let mut mid0 = self.bf8.bf4.exec(rows0);
                let mut mid1 = self.bf8.bf4.exec(rows1);

                // apply twiddle factors
                for r in 1..4 {
                    mid0[r] =
                        AvxStoreF::mul_by_complex(mid0[r], *self.twiddles.get_unchecked(2 * r - 2));
                    mid1[r] =
                        AvxStoreF::mul_by_complex(mid1[r], *self.twiddles.get_unchecked(2 * r - 1));
                }

                // Transpose our 8x4 array to an 4x8 array
                let transposed = transpose_8x4_to_4x8_f32(mid0, mid1);

                let output_rows = self.bf8.exec(transposed);

                output_rows[0].write(dst);
                output_rows[1].write(dst.get_unchecked_mut(4..));
                output_rows[2].write(dst.get_unchecked_mut(8..));
                output_rows[3].write(dst.get_unchecked_mut(12..));
                output_rows[4].write_lo1(dst.get_unchecked_mut(16..));
            }
        }
        Ok(())
    }
}

impl R2CFftExecutor<f32> for AvxButterfly32f {
    fn execute(&self, input: &[f32], output: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_r2c(input, output) }
    }

    fn execute_with_scratch(
        &self,
        input: &[f32],
        output: &mut [Complex<f32>],
        _: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_r2c(input, output) }
    }

    fn real_length(&self) -> usize {
        32
    }

    fn complex_length(&self) -> usize {
        17
    }

    fn complex_scratch_length(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{
        test_avx_butterfly, test_oof_avx_butterfly, test_r2c_avx_butterfly,
    };

    test_r2c_avx_butterfly!(test_avx_r2c_butterfly32, f32, AvxButterfly32f, 32, 1e-5);

    test_avx_butterfly!(test_avx_butterfly32, f32, AvxButterfly32f, 32, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly32, f32, AvxButterfly32f, 32, 1e-5);
}
