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
use crate::{C2RFftExecutor, ExecutorWithScratch, FftExecutor, ZaftError};
use novtb::{ParallelZonedIterator, TbSliceMut};
use num_complex::Complex;
use std::sync::Arc;

pub trait TwoDimensionalExecutorC2R<T>: ExecutorWithScratch {
    /// Executes the 2D Real-to-Complex FFT using an internal scratch buffer.
    ///
    /// The size of the `source` slice must be equal to `self.real_size()`, and the size of the
    /// `output` slice must be equal to `self.complex_size()`.
    ///
    /// # Parameters
    /// * `source`: The **real-valued** 2D input array (flattened into a 1D slice).
    /// * `output`: The mutable slice where the **complex-valued** frequency data will be written.
    ///
    /// # Errors
    /// Returns a `ZaftError` if the execution fails (e.g., due to incorrect slice lengths).
    fn execute(&self, source: &mut [Complex<T>], output: &mut [T]) -> Result<(), ZaftError>;
    /// Executes the 2D Real-to-Complex FFT using an externally provided scratch buffer.
    ///
    /// This method allows the caller to manage and reuse the scratch memory, potentially avoiding
    /// internal memory allocations.
    ///
    /// The size requirements for `source` and `output` are the same as `execute`. The `scratch`
    /// slice must meet the size requirement specified by the inherited `ExecutorWithScratch::required_scratch_size()`.
    ///
    /// # Parameters
    /// * `source`: The **real-valued** 2D input array.
    /// * `output`: The mutable slice where the **complex-valued** frequency data will be written.
    /// * `scratch`: The mutable scratch buffer required for intermediate calculations.
    ///
    /// # Errors
    /// Returns a `ZaftError` if the execution fails.
    fn execute_with_scratch(
        &self,
        source: &mut [Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError>;
    /// Returns the **width** (number of columns, X-dimension) of the 2D input data grid.
    fn width(&self) -> usize;
    /// Returns the **height** (number of rows, Y-dimension) of the 2D input data grid.
    fn height(&self) -> usize;
    /// Returns the **total size** (number of real elements) of the 2D input array.
    ///
    /// This is equivalent to `self.width() * self.height()`.
    fn real_size(&self) -> usize;
    /// Returns the **total size** (number of complex elements) of the frequency-domain output array.
    ///
    /// This value accounts for the Hermitian symmetry property of the R2C FFT.
    /// For an input of size `W x H`, the complex size is typically `H * (W/2 + 1)`.
    fn complex_size(&self) -> usize;
}

pub(crate) struct TwoDimensionalC2R<T> {
    pub(crate) width_c2r_executor: Arc<dyn C2RFftExecutor<T> + Send + Sync>,
    pub(crate) height_c2c_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
    pub(crate) transpose_height_to_width: Box<dyn TransposeExecutor<T> + Send + Sync>,
    pub(crate) thread_count: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl<T: Copy + Default + Send + Sync> TwoDimensionalExecutorC2R<T> for TwoDimensionalC2R<T> {
    fn execute(&self, source: &mut [Complex<T>], output: &mut [T]) -> Result<(), ZaftError> {
        let mut scratch = try_vec![Complex::<T>::default(); self.required_scratch_size()];
        self.execute_with_scratch(source, output, &mut scratch)
    }

    fn execute_with_scratch(
        &self,
        source: &mut [Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if scratch.len() < self.required_scratch_size() {
            return Err(ZaftError::ScratchBufferIsTooSmall(
                scratch.len(),
                self.required_scratch_size(),
            ));
        }
        let complex_size = self.complex_size();
        if !source.len().is_multiple_of(complex_size) {
            return Err(ZaftError::InvalidSizeMultiplier(output.len(), complex_size));
        }
        if !output.len().is_multiple_of(self.real_size()) {
            return Err(ZaftError::InvalidSizeMultiplier(
                output.len(),
                self.real_size(),
            ));
        }
        if source.len() / complex_size != output.len() / self.real_size() {
            return Err(ZaftError::InvalidSizeMultiplier(
                output.len() / self.real_size(),
                source.len() / complex_size,
            ));
        }
        let complex_row_size = (self.width / 2) + 1;
        let (scratch, _) = scratch.split_at_mut(self.required_scratch_size());
        let pool = novtb::ThreadPool::new(self.thread_count);

        for (src, dst) in source
            .chunks_exact_mut(self.complex_size())
            .zip(output.chunks_exact_mut(self.real_size()))
        {
            src.tb_par_chunks_exact_mut(self.height)
                .for_each(&pool, |row| {
                    _ = self.height_c2c_executor.execute(row);
                });

            self.transpose_height_to_width
                .transpose(src, scratch, self.height, complex_row_size);

            dst.tb_par_chunks_exact_mut(self.width)
                .for_each_enumerated(&pool, |idx, column| {
                    let arena_src = &scratch[complex_row_size * idx..complex_row_size * (idx + 1)];
                    _ = self.width_c2r_executor.execute(arena_src, column);
                });
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
    fn real_size(&self) -> usize {
        self.width * self.height
    }

    #[inline]
    fn complex_size(&self) -> usize {
        ((self.width / 2) + 1) * self.height
    }
}

impl<T> ExecutorWithScratch for TwoDimensionalC2R<T> {
    #[inline]
    fn required_scratch_size(&self) -> usize {
        ((self.width / 2) + 1) * self.height
    }
}
