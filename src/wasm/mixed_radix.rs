/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
#![allow(clippy::modulo_one)]
use crate::err::try_vec;
use crate::transpose::{TransposeExecutor, TransposeFactory};
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::Zero;
use std::sync::Arc;

macro_rules! define_mixed_radix_neon_d {
    ($radix_name: ident, $bf_name: ident, $row_count: expr, $mul: ident) => {
        use crate::wasm::column::$bf_name;
        pub(crate) struct $radix_name {
            execution_length: usize,
            direction: FftDirection,
            twiddles: Vec<WasmStoreD>,
            width_executor: Arc<dyn FftExecutor<f64> + Send + Sync>,
            width: usize,
            height: usize,
            transpose_executor: Box<dyn TransposeExecutor<f64> + Send + Sync>,
            inner_bf: $bf_name,
            width_scratch_length: usize,
            oof_width_scratch_length: usize,
        }

        impl $radix_name {
            pub(crate) fn new(
                width_executor: Arc<dyn FftExecutor<f64> + Send + Sync>,
            ) -> Result<Self, ZaftError> {
                let direction = width_executor.direction();

                let width = width_executor.length();

                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;

                let len_per_row = width_executor.length();

                let len = len_per_row * ROW_COUNT;
                const COMPLEX_PER_VECTOR: usize = 1;

                let quotient = len_per_row / COMPLEX_PER_VECTOR;
                #[allow(clippy::modulo_one)]
                let remainder = len_per_row % COMPLEX_PER_VECTOR;

                let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
                let mut twiddles = Vec::new();
                twiddles
                    .try_reserve_exact(num_twiddle_columns * TWIDDLES_PER_COLUMN)
                    .map_err(|_| {
                        ZaftError::OutOfMemory(num_twiddle_columns * TWIDDLES_PER_COLUMN)
                    })?;
                for x in 0..num_twiddle_columns {
                    for y in 1..ROW_COUNT {
                        let mut data: [Complex<f64>; COMPLEX_PER_VECTOR] =
                            [Complex::zero(); COMPLEX_PER_VECTOR];
                        for i in 0..COMPLEX_PER_VECTOR {
                            data[i] =
                                compute_twiddle(y * (x * COMPLEX_PER_VECTOR + i), len, direction);
                        }
                        twiddles.push(WasmStoreD::from_complex_ref(data.as_ref()));
                    }
                }

                let width_scratch_length = width_executor.out_of_place_scratch_length();
                let execution_length = width * ROW_COUNT;
                let oof_width_scratch_length = if execution_length >= width_executor.scratch_length() {
                    0
                } else {
                    width_executor.scratch_length()
                };

                Ok($radix_name {
                    execution_length,
                    width_executor,
                    width,
                    height: ROW_COUNT,
                    direction,
                    twiddles,
                    transpose_executor: f64::transpose_strategy(width, ROW_COUNT),
                    inner_bf: $bf_name::new(direction),
                    width_scratch_length,
                    oof_width_scratch_length,
                })
            }
        }

        impl FftExecutor<f64> for $radix_name {
            fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                let mut scratch = try_vec![Complex::zero(); self.scratch_length()];
               self.execute_impl(in_place, scratch.as_mut_slice())
            }

            fn execute_with_scratch(
                &self,
                in_place: &mut [Complex<f64>],
                scratch: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                 self.execute_impl(in_place, scratch)
            }

            fn execute_out_of_place(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                let mut scratch = try_vec![Complex::zero(); self.out_of_place_scratch_length()];
                self.execute_out_of_place_with_scratch(src, dst, scratch.as_mut_slice())
            }

            fn execute_out_of_place_with_scratch(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
                scratch: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                 self.execute_oof_impl(src, dst, scratch)
            }

            fn execute_destructive_with_scratch(
                &self,
                src: &mut [Complex<f64>],
                dst: &mut [Complex<f64>],
                scratch: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                self.execute_d_oof_impl(src, dst, scratch)
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                self.execution_length
            }

            #[inline]
            fn scratch_length(&self) -> usize {
                self.execution_length + self.width_scratch_length
            }

            #[inline]
            fn out_of_place_scratch_length(&self) -> usize {
                self.execution_length + self.oof_width_scratch_length
            }

            #[inline]
            fn destructive_scratch_length(&self) -> usize {
                self.oof_width_scratch_length
            }
        }

        impl $radix_name {
            fn execute_impl(
                &self,
                in_place: &mut [Complex<f64>],
                scratch: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(self.execution_length) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.execution_length,
                    ));
                }

                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.scratch_length());
                let (scratch, width_scratch) = scratch.split_at_mut(self.execution_length);

                for chunk in in_place.chunks_exact_mut(self.execution_length) {
                    self.process_columns_in_place(chunk);

                    self.width_executor.execute_destructive_with_scratch(
                        chunk,
                        scratch,
                        width_scratch,
                    )?;

                    self.transpose_executor
                        .transpose(&scratch, chunk, self.width, self.height);
                }
                Ok(())
            }

            fn process_columns_in_place(&self, chunk: &mut [Complex<f64>]) {
                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
                const COMPLEX_PER_VECTOR: usize = 1;

                let len_per_row = self.length() / ROW_COUNT;
                let chunk_count = len_per_row / COMPLEX_PER_VECTOR;
                // process the column FFTs
                for (c, twiddle_chunk) in self
                    .twiddles
                    .chunks_exact(TWIDDLES_PER_COLUMN)
                    .take(chunk_count)
                    .enumerate()
                {
                    let index_base = c * COMPLEX_PER_VECTOR;

                    let mut columns = [WasmStoreD::default(); ROW_COUNT];
                    for i in 0..ROW_COUNT {
                        unsafe {
                            columns[i] = WasmStoreD::from_complex_ref(
                                chunk.get_unchecked(index_base + len_per_row * i..),
                            );
                        }
                    }

                    let output = self.inner_bf.exec(columns);

                    unsafe {
                        output[0].write(chunk.get_unchecked_mut(index_base..));
                    }

                    // here LLVM doesn't "see" WasmStoreD as the same type returned by output
                    // so we need to force cast it onwards to the same type
                    let mut twiddles = [WasmStoreD::default(); ROW_COUNT - 1];
                    for i in 0..ROW_COUNT - 1 {
                        twiddles[i] = twiddle_chunk[i];
                    }

                    for i in 1..ROW_COUNT {
                        let twiddle = twiddles[i - 1];
                        let output = WasmStoreD::$mul(output[i], twiddle);
                        unsafe {
                            output.write(chunk.get_unchecked_mut(index_base + len_per_row * i..))
                        }
                    }
                }
            }

            fn execute_d_oof_impl(
                &self,
                src: &mut [Complex<f64>],
                dst: &mut [Complex<f64>],
                scratch: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, self.execution_length);

                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.destructive_scratch_length());
                let use_dst_as_scratch = self.execution_length >= self.oof_width_scratch_length;

                for (dst_chunk, src_chunk) in dst
                    .chunks_exact_mut(self.execution_length)
                    .zip(src.chunks_exact_mut(self.execution_length))
                {
                    self.process_columns_in_place(src_chunk);

                    self.width_executor
                        .execute_with_scratch(src_chunk, if use_dst_as_scratch { dst_chunk } else { scratch })?;

                    self.transpose_executor.transpose(
                        src_chunk,
                        dst_chunk,
                        self.width,
                        self.height,
                    );
                }
                Ok(())
            }

            fn process_oof_columns(&self, chunk: &[Complex<f64>], scratch: &mut [Complex<f64>]) {
                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
                const COMPLEX_PER_VECTOR: usize = 1;

                let len_per_row = self.length() / ROW_COUNT;
                let chunk_count = len_per_row / COMPLEX_PER_VECTOR;

                // process the column FFTs
                for (c, twiddle_chunk) in self
                    .twiddles
                    .as_chunks::<TWIDDLES_PER_COLUMN>().0.iter()
                    .take(chunk_count)
                    .enumerate()
                {
                    let index_base = c * COMPLEX_PER_VECTOR;

                    let mut columns = [WasmStoreD::default(); ROW_COUNT];
                    for i in 0..ROW_COUNT {
                        unsafe {
                            columns[i] = WasmStoreD::from_complex_ref(
                                chunk.get_unchecked(index_base + len_per_row * i..),
                            );
                        }
                    }

                    let output = self.inner_bf.exec(columns);

                    unsafe {
                        output[0].write(scratch.get_unchecked_mut(index_base..));
                    }

                    // here LLVM doesn't "see" WasmStoreD as the same type returned by output
                    // so we need to force cast it onwards to the same type
                    let mut twiddles = [WasmStoreD::default(); ROW_COUNT - 1];
                    for i in 0..ROW_COUNT - 1 {
                        twiddles[i] = twiddle_chunk[i];
                    }

                    for i in 1..ROW_COUNT {
                        let twiddle = twiddles[i - 1];
                        let output = WasmStoreD::$mul(output[i], twiddle);
                        unsafe {
                            output.write(scratch.get_unchecked_mut(index_base + len_per_row * i..))
                        }
                    }
                }
            }

            fn execute_oof_impl(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
                scratch: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, self.execution_length);

                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.out_of_place_scratch_length());
                let (scratch, width_scratch) = scratch.split_at_mut(self.execution_length);
                let use_dst_as_scratch = self.execution_length >= self.oof_width_scratch_length;

                for (dst_chunk, chunk) in dst
                    .chunks_exact_mut(self.execution_length)
                    .zip(src.chunks_exact(self.execution_length))
                {
                    self.process_oof_columns(chunk, scratch);

                    self.width_executor
                        .execute_with_scratch(scratch, if use_dst_as_scratch { dst_chunk } else { width_scratch })?;

                    self.transpose_executor
                        .transpose(&scratch, dst_chunk, self.width, self.height);
                }
                Ok(())
            }
        }
    };
}

macro_rules! define_mixed_radix_neon_f {
    ($radix_name: ident, $bf_name: ident, $row_count: expr, $mul: ident) => {
        use crate::wasm::column::$bf_name;
        pub(crate) struct $radix_name {
            execution_length: usize,
            direction: FftDirection,
            twiddles: Vec<WasmStoreF>,
            width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
            width: usize,
            height: usize,
            transpose_executor: Box<dyn TransposeExecutor<f32> + Send + Sync>,
            inner_bf: $bf_name,
            width_scratch_length: usize,
            oof_width_scratch_length: usize,
        }

        impl $radix_name {
            pub(crate) fn new(
                width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
            ) -> Result<Self, ZaftError> {
                let direction = width_executor.direction();

                let width = width_executor.length();

                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;

                // derive some info from our inner FFT
                let len_per_row = width_executor.length();

                let len = len_per_row * ROW_COUNT;
                const COMPLEX_PER_VECTOR: usize = 2;

                let quotient = len_per_row / COMPLEX_PER_VECTOR;
                let remainder = len_per_row % COMPLEX_PER_VECTOR;

                let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
                let mut twiddles = Vec::new();
                twiddles
                    .try_reserve_exact(num_twiddle_columns * TWIDDLES_PER_COLUMN)
                    .map_err(|_| {
                        ZaftError::OutOfMemory(num_twiddle_columns * TWIDDLES_PER_COLUMN)
                    })?;
                for x in 0..num_twiddle_columns {
                    for y in 1..ROW_COUNT {
                        let mut data: [Complex<f32>; COMPLEX_PER_VECTOR] =
                            [Complex::zero(); COMPLEX_PER_VECTOR];
                        for i in 0..COMPLEX_PER_VECTOR {
                            data[i] =
                                compute_twiddle(y * (x * COMPLEX_PER_VECTOR + i), len, direction);
                        }
                        twiddles.push(WasmStoreF::from_complex_ref(data.as_ref()));
                    }
                }

                let width_scratch_length = width_executor.destructive_scratch_length();
                let execution_length = width * ROW_COUNT;
                let oof_width_scratch_length = if execution_length >= width_executor.scratch_length() {
                    0
                } else {
                    width_executor.scratch_length()
                };

                Ok($radix_name {
                    execution_length,
                    width_executor,
                    width,
                    height: ROW_COUNT,
                    direction,
                    twiddles,
                    transpose_executor: f32::transpose_strategy(width, ROW_COUNT),
                    inner_bf: $bf_name::new(direction),
                    width_scratch_length,
                    oof_width_scratch_length,
                })
            }
        }

        impl FftExecutor<f32> for $radix_name {
            fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                let mut scratch = try_vec![Complex::zero(); self.scratch_length()];
                self.execute_impl(in_place, scratch.as_mut_slice())
            }

            fn execute_with_scratch(
                &self,
                in_place: &mut [Complex<f32>],
                scratch: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                self.execute_impl(in_place, scratch)
            }

            fn execute_out_of_place(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                let mut scratch = try_vec![Complex::zero(); self.out_of_place_scratch_length()];
                self.execute_out_of_place_with_scratch(src, dst, scratch.as_mut_slice())
            }

            fn execute_out_of_place_with_scratch(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
                scratch: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                self.execute_oof_impl(src, dst, scratch)
            }

            fn execute_destructive_with_scratch(
                &self,
                src: &mut [Complex<f32>],
                dst: &mut [Complex<f32>],
                scratch: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                self.execute_d_oof_impl(src, dst, scratch)
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                self.execution_length
            }

            #[inline]
            fn scratch_length(&self) -> usize {
                self.execution_length + self.width_scratch_length
            }

            #[inline]
            fn out_of_place_scratch_length(&self) -> usize {
                self.execution_length + self.oof_width_scratch_length
            }

            #[inline]
            fn destructive_scratch_length(&self) -> usize {
                self.oof_width_scratch_length
            }
        }

        impl $radix_name {
            fn execute_impl(
                &self,
                in_place: &mut [Complex<f32>],
                scratch: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(self.execution_length) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.execution_length,
                    ));
                }

                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.scratch_length());
                let (scratch, width_scratch) = scratch.split_at_mut(self.execution_length);

                for chunk in in_place.chunks_exact_mut(self.execution_length) {
                    self.process_columns_in_place(chunk);

                    self.width_executor.execute_destructive_with_scratch(
                        chunk,
                        scratch,
                        width_scratch,
                    )?;

                    self.transpose_executor
                        .transpose(&scratch, chunk, self.width, self.height);
                }
                Ok(())
            }

            fn process_columns_in_place(&self, chunk: &mut [Complex<f32>]) {
                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
                const COMPLEX_PER_VECTOR: usize = 2;

                let len_per_row = self.length() / ROW_COUNT;
                let chunk_count = len_per_row / COMPLEX_PER_VECTOR;

                for (c, twiddle_chunk) in self
                    .twiddles
                    .as_chunks::<TWIDDLES_PER_COLUMN>().0.iter()
                    .take(chunk_count)
                    .enumerate()
                {
                    let index_base = c * COMPLEX_PER_VECTOR;

                    // Load columns from the input into registers
                    let mut columns = [WasmStoreF::default(); ROW_COUNT];
                    for i in 0..ROW_COUNT {
                        unsafe {
                            columns[i] = WasmStoreF::from_complex_ref(
                                chunk.get_unchecked(index_base + len_per_row * i..),
                            );
                        }
                    }

                    let output = self.inner_bf.exec(columns);

                    unsafe {
                        output[0].write(chunk.get_unchecked_mut(index_base..));
                    }

                    // here LLVM doesn't "see" WasmStoreF as the same type returned by output
                    // so we need to force cast it onwards to the same type
                    let mut twiddles = [WasmStoreF::default(); ROW_COUNT - 1];
                    for i in 0..ROW_COUNT - 1 {
                        twiddles[i] = twiddle_chunk[i];
                    }

                    for i in 1..ROW_COUNT {
                        let twiddle = twiddles[i - 1];
                        let output = WasmStoreF::$mul(output[i], twiddle);
                        unsafe {
                            output.write(chunk.get_unchecked_mut(index_base + len_per_row * i..))
                        }
                    }
                }

                let partial_remainder = len_per_row % COMPLEX_PER_VECTOR;
                if partial_remainder > 0 {
                    let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                    let partial_remainder_twiddle_base = self.twiddles.len() - TWIDDLES_PER_COLUMN;
                    let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                    let mut columns = [WasmStoreF::default(); ROW_COUNT];
                    for i in 0..ROW_COUNT {
                        unsafe {
                            columns[i] = WasmStoreF::load_complex(
                                chunk.get_unchecked(partial_remainder_base + len_per_row * i),
                            );
                        }
                    }

                    // apply our butterfly function down the columns
                    let output = self.inner_bf.exec(columns);

                    // always write the first row without twiddles
                    unsafe {
                        output[0].write_lo(chunk.get_unchecked_mut(partial_remainder_base..));
                    }

                    // here LLVM doesn't "see" WasmStoreF as the same type returned by output
                    // so we need to force cast it onwards to the same type
                    let mut twiddles = [WasmStoreF::default(); ROW_COUNT - 1];
                    for i in 0..ROW_COUNT - 1 {
                        twiddles[i] = final_twiddle_chunk[i];
                    }

                    // for the remaining rows, apply twiddle factors and then write back to memory
                    for i in 1..ROW_COUNT {
                        let twiddle = twiddles[i - 1];
                        let output = WasmStoreF::$mul(output[i], twiddle);
                        unsafe {
                            output.write_lo(
                                chunk.get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                            );
                        }
                    }
                }
            }

            fn execute_d_oof_impl(
                &self,
                src: &mut [Complex<f32>],
                dst: &mut [Complex<f32>],
                scratch: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, self.execution_length);

                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.destructive_scratch_length());
                let use_dst_as_scratch = self.execution_length >= self.oof_width_scratch_length;

                for (dst_chunk, src_chunk) in dst
                    .chunks_exact_mut(self.execution_length)
                    .zip(src.chunks_exact_mut(self.execution_length))
                {
                    self.process_columns_in_place(src_chunk);

                    self.width_executor
                        .execute_with_scratch(src_chunk, if use_dst_as_scratch { dst_chunk } else { scratch })?;

                    self.transpose_executor.transpose(
                        src_chunk,
                        dst_chunk,
                        self.width,
                        self.height,
                    );
                }
                Ok(())
            }

            fn process_oof_columns(&self, chunk: &[Complex<f32>], scratch: &mut [Complex<f32>]) {
                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
                const COMPLEX_PER_VECTOR: usize = 2;

                let len_per_row = self.length() / ROW_COUNT;
                let chunk_count = len_per_row / COMPLEX_PER_VECTOR;
                for (c, twiddle_chunk) in self
                    .twiddles
                    .chunks_exact(TWIDDLES_PER_COLUMN)
                    .take(chunk_count)
                    .enumerate()
                {
                    let index_base = c * COMPLEX_PER_VECTOR;

                    // Load columns from the input into registers
                    let mut columns = [WasmStoreF::default(); ROW_COUNT];
                    for i in 0..ROW_COUNT {
                        unsafe {
                            columns[i] = WasmStoreF::from_complex_ref(
                                chunk.get_unchecked(index_base + len_per_row * i..),
                            );
                        }
                    }

                    let output = self.inner_bf.exec(columns);

                    unsafe {
                        output[0].write(scratch.get_unchecked_mut(index_base..));
                    }

                    // here LLVM doesn't "see" WasmStoreF as the same type returned by output
                    // so we need to force cast it onwards to the same type
                    let mut twiddles = [WasmStoreF::default(); ROW_COUNT - 1];
                    for i in 0..ROW_COUNT - 1 {
                        twiddles[i] = twiddle_chunk[i];
                    }

                    for i in 1..ROW_COUNT {
                        let twiddle = twiddles[i - 1];
                        let output = WasmStoreF::$mul(output[i], twiddle);
                        unsafe {
                            output.write(scratch.get_unchecked_mut(index_base + len_per_row * i..))
                        }
                    }
                }

                let partial_remainder = len_per_row % COMPLEX_PER_VECTOR;
                if partial_remainder > 0 {
                    let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                    let partial_remainder_twiddle_base = self.twiddles.len() - TWIDDLES_PER_COLUMN;
                    let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                    let mut columns = [WasmStoreF::default(); ROW_COUNT];
                    for i in 0..ROW_COUNT {
                        unsafe {
                            columns[i] = WasmStoreF::load_complex(
                                chunk.get_unchecked(partial_remainder_base + len_per_row * i),
                            );
                        }
                    }

                    // apply our butterfly function down the columns
                    let output = self.inner_bf.exec(columns);

                    // always write the first row without twiddles
                    unsafe {
                        output[0].write_lo(scratch.get_unchecked_mut(partial_remainder_base..));
                    }

                    // here LLVM doesn't "see" WasmStoreF as the same type returned by output
                    // so we need to force cast it onwards to the same type
                    let mut twiddles = [WasmStoreF::default(); ROW_COUNT - 1];
                    for i in 0..ROW_COUNT - 1 {
                        twiddles[i] = final_twiddle_chunk[i];
                    }

                    // for the remaining rows, apply twiddle factors and then write back to memory
                    for i in 1..ROW_COUNT {
                        let twiddle = twiddles[i - 1];
                        let output = WasmStoreF::$mul(output[i], twiddle);
                        unsafe {
                            output.write_lo(
                                scratch
                                    .get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                            );
                        }
                    }
                }
            }

            fn execute_oof_impl(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
                scratch: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, self.execution_length);

                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.out_of_place_scratch_length());
                let (scratch, width_scratch) = scratch.split_at_mut(self.execution_length);
                let use_dst_as_scratch = self.execution_length >= self.oof_width_scratch_length;

                for (dst_chunk, chunk) in dst
                    .chunks_exact_mut(self.execution_length)
                    .zip(src.chunks_exact(self.execution_length))
                {
                    self.process_oof_columns(chunk, scratch);

                    self.width_executor
                        .execute_with_scratch(scratch, if use_dst_as_scratch { dst_chunk } else { width_scratch })?;

                    self.transpose_executor
                        .transpose(&scratch, dst_chunk, self.width, self.height);
                }
                Ok(())
            }
        }
    };
}

use crate::wasm::store::{WasmStoreD, WasmStoreF};

define_mixed_radix_neon_d!(WasmMixedRadix2, ColumnButterfly2d, 2, mul_by_complex);
define_mixed_radix_neon_d!(WasmMixedRadix3, ColumnButterfly3d, 3, mul_by_complex);
define_mixed_radix_neon_d!(WasmMixedRadix4, ColumnButterfly4d, 4, mul_by_complex);
define_mixed_radix_neon_d!(WasmMixedRadix5, ColumnButterfly5d, 5, mul_by_complex);
define_mixed_radix_neon_d!(WasmMixedRadix7, ColumnButterfly7d, 7, mul_by_complex);
define_mixed_radix_neon_d!(WasmMixedRadix8, ColumnButterfly8d, 8, mul_by_complex);
define_mixed_radix_neon_d!(WasmMixedRadix9, ColumnButterfly9d, 9, mul_by_complex);
define_mixed_radix_neon_f!(WasmMixedRadix2f, ColumnButterfly2f, 2, mul_by_complex);
define_mixed_radix_neon_f!(WasmMixedRadix3f, ColumnButterfly3f, 3, mul_by_complex);
define_mixed_radix_neon_f!(WasmMixedRadix4f, ColumnButterfly4f, 4, mul_by_complex);
define_mixed_radix_neon_f!(WasmMixedRadix5f, ColumnButterfly5f, 5, mul_by_complex);
define_mixed_radix_neon_f!(WasmMixedRadix7f, ColumnButterfly7f, 7, mul_by_complex);
define_mixed_radix_neon_f!(WasmMixedRadix8f, ColumnButterfly8f, 8, mul_by_complex);
define_mixed_radix_neon_f!(WasmMixedRadix9f, ColumnButterfly9f, 9, mul_by_complex);
