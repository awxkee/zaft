/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
use crate::err::try_vec;
use crate::transpose::TransposeExecutor;
use crate::util::validate_scratch;
use crate::{FftExecutor, ZaftError};
use novtb::{ParallelZonedIterator, TbSliceMut};
use num_complex::Complex;
use std::sync::Arc;

pub trait TwoDimensionalFftExecutor<T> {
    /// Executes a 2D FFT on source and writes the result into output.
    ///
    /// Both source and output must have length equal to width() * height().
    ///
    /// # Errors
    /// Returns a [ZaftError] if the execution fails.
    fn execute(&self, data: &mut [Complex<T>]) -> Result<(), ZaftError>;
    /// Executes a 2D FFT using a pre-allocated `scratch` buffer for temporary storage.
    /// This can reduce allocations and improve performance for repeated calls.
    /// Both `source`, `output`, and `scratch` must have sufficient length.
    ///
    /// # Errors
    /// Returns a [`ZaftError`] if the execution fails.
    fn execute_with_scratch(
        &self,
        data: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError>;
    /// Returns the **width** (number of columns, X-dimension) of the 2D input data grid.
    fn width(&self) -> usize;
    /// Returns the **height** (number of rows, Y-dimension) of the 2D input data grid.
    fn height(&self) -> usize;
    fn scratch_length(&self) -> usize;
}

pub(crate) struct TwoDimensionalC2C<T> {
    pub(crate) width_c2c_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
    pub(crate) height_c2c_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
    pub(crate) transpose_width_to_height: Box<dyn TransposeExecutor<T> + Send + Sync>,
    pub(crate) thread_count: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) oof_width_scratch_size: usize,
    pub(crate) height_scratch_size: usize,
}

impl<T: Copy + Default + Send + Sync> TwoDimensionalFftExecutor<T> for TwoDimensionalC2C<T> {
    fn execute(&self, data: &mut [Complex<T>]) -> Result<(), ZaftError> {
        let mut scratch = try_vec![Complex::<T>::default(); self.scratch_length()];
        self.execute_with_scratch(data, &mut scratch)
    }

    fn execute_with_scratch(
        &self,
        data: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        let full_size = self.width * self.height;
        if !data.len().is_multiple_of(full_size) {
            return Err(ZaftError::InvalidSizeMultiplier(data.len(), full_size));
        }

        let scratch = validate_scratch!(scratch, self.scratch_length());

        let block_size = self.width * self.height;
        let (scratch, r) = scratch.split_at_mut(block_size);
        let (width_scratch, height_scratch) = r.split_at_mut(self.oof_width_scratch_size);

        let pool = novtb::ThreadPool::new(self.thread_count);

        for chunk in data.chunks_exact_mut(full_size) {
            if self.thread_count <= 1 {
                for (row, src_row) in scratch
                    .chunks_exact_mut(self.width)
                    .zip(chunk.chunks_exact_mut(self.width))
                {
                    self.width_c2c_executor.execute_out_of_place_with_scratch(
                        src_row,
                        row,
                        width_scratch,
                    )?;
                }
            } else {
                scratch
                    .tb_par_chunks_exact_mut(self.width)
                    .for_each_enumerated(&pool, |y, row| {
                        let src_row = &chunk[y * self.width..(y + 1) * self.width];
                        _ = self.width_c2c_executor.execute_out_of_place(src_row, row);
                    });
            }

            self.transpose_width_to_height
                .transpose(scratch, chunk, self.width, self.height);

            if self.thread_count <= 1 {
                for column in chunk.chunks_exact_mut(self.height) {
                    self.height_c2c_executor
                        .execute_with_scratch(column, height_scratch)?;
                }
            } else {
                chunk
                    .tb_par_chunks_exact_mut(self.height)
                    .for_each(&pool, |column| {
                        _ = self.height_c2c_executor.execute(column);
                    });
            }
        }
        Ok(())
    }

    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn height(&self) -> usize {
        self.height
    }

    #[inline]
    fn scratch_length(&self) -> usize {
        self.width * self.height
            + if self.thread_count <= 1 {
                self.oof_width_scratch_size
            } else {
                0
            }
            + if self.thread_count <= 1 {
                self.height_scratch_size
            } else {
                0
            }
    }
}
