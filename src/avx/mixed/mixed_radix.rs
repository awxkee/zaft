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
#![allow(unused_unsafe)]

use crate::avx::mixed::avx_stored::AvxStoreD;
use crate::avx::mixed::avx_storef::AvxStoreF;
use crate::avx::mixed::butterflies::{
    ColumnButterfly2d, ColumnButterfly2f, ColumnButterfly3d, ColumnButterfly3f, ColumnButterfly4d,
    ColumnButterfly4f, ColumnButterfly5d, ColumnButterfly5f, ColumnButterfly6d, ColumnButterfly6f,
    ColumnButterfly7d, ColumnButterfly7f, ColumnButterfly8d, ColumnButterfly8f, ColumnButterfly9d,
    ColumnButterfly9f, ColumnButterfly10d, ColumnButterfly10f, ColumnButterfly11d,
    ColumnButterfly11f, ColumnButterfly12d, ColumnButterfly12f, ColumnButterfly13d,
    ColumnButterfly13f,
};
use crate::err::try_vec;
use crate::transpose::{TransposeExecutor, TransposeFactory};
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::Zero;
use std::sync::Arc;

macro_rules! define_mixed_radixd {
    ($mx_type: ident, $bf_type: ident, $row_count: expr) => {
        pub(crate) struct $mx_type {
            execution_length: usize,
            direction: FftDirection,
            twiddles: Vec<AvxStoreD>,
            width_executor: Arc<dyn FftExecutor<f64> + Send + Sync>,
            width: usize,
            height: usize,
            transpose_executor: Box<dyn TransposeExecutor<f64> + Send + Sync>,
            inner_bf: $bf_type,
            width_scratch_length: usize,
            oof_width_scratch_length: usize,
        }

        impl $mx_type {
            pub fn new(width_executor: Arc<dyn FftExecutor<f64> + Send + Sync>) -> Result<Self, ZaftError> {
                unsafe {
                    Self::new_init(width_executor)
                }
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn new_init(width_executor: Arc<dyn FftExecutor<f64> + Send + Sync>) -> Result<Self, ZaftError> {
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
                    .map_err(|_| ZaftError::OutOfMemory(num_twiddle_columns * TWIDDLES_PER_COLUMN))?;
                for x in 0..num_twiddle_columns {
                    for y in 1..ROW_COUNT {
                        let mut data: [Complex<f64>; 2] = [Complex::zero(); 2];
                        for i in 0..COMPLEX_PER_VECTOR {
                            data[i] = compute_twiddle(
                                y * (x * COMPLEX_PER_VECTOR + i),
                                len,
                                direction,
                            );
                        }
                        twiddles.push(AvxStoreD::from_complex_ref(data.as_ref()));
                    }
                }

                let width_scratch_length = width_executor.destructive_scratch_length();
                let oof_width_scratch_length = width_executor.scratch_length();

                #[allow(unused_unsafe)]
                Ok($mx_type {
                    execution_length: width * ROW_COUNT,
                    width_executor,
                    width,
                    height: ROW_COUNT,
                    direction,
                    twiddles,
                    transpose_executor: f64::transpose_strategy(width, ROW_COUNT),
                    inner_bf: unsafe { $bf_type::new(direction) },
                    width_scratch_length,
                    oof_width_scratch_length,
                })
            }
        }

        impl $mx_type {
            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_f64(&self, in_place: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) -> Result<(), ZaftError> {
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

                    self.width_executor.execute_destructive_with_scratch(chunk, scratch, width_scratch)?;

                    self.transpose_executor
                        .transpose(&scratch, chunk, self.width, self.height);
                }
                Ok(())
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn process_columns_in_place(&self,
                                 chunk: &mut [Complex<f64>]) {
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

                        let mut columns = [AvxStoreD::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreD::from_complex_ref(
                                    chunk.get_unchecked(index_base + len_per_row * i..),
                                );
                            }
                        }

                        let output = unsafe {
                            self.inner_bf.exec(columns)
                        };

                        unsafe {
                            output[0].write(chunk.get_unchecked_mut(index_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreD as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreD::zero(); ROW_COUNT - 1];
                        for i in 0..ROW_COUNT - 1 {
                            twiddles[i] = twiddle_chunk[i];
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let v = AvxStoreD::mul_by_complex(output[i], twiddle);
                            unsafe {
                                v.write(chunk.get_unchecked_mut(index_base + len_per_row * i..))
                            }
                        }
                    }

                    let partial_remainder = len_per_row % COMPLEX_PER_VECTOR;
                    if partial_remainder > 0 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base = self.twiddles.len() - TWIDDLES_PER_COLUMN;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [AvxStoreD::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreD::from_complex(
                                    chunk.get_unchecked(partial_remainder_base + len_per_row * i),
                                );
                            }
                        }

                        // apply our butterfly function down the columns
                        let output = unsafe {
                            self.inner_bf.exec(columns)
                        };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write_lo(chunk.get_unchecked_mut(partial_remainder_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreD as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreD::zero(); ROW_COUNT - 1];
                        for i in 0..ROW_COUNT - 1 {
                            twiddles[i] = final_twiddle_chunk[i];
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let output = AvxStoreD::mul_by_complex(output[i], twiddle);
                            unsafe {
                                output.write_lo(
                                    chunk.get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }
                    }
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_d_oof_impl(&self,
                                 src: &mut [Complex<f64>],
                                 dst: &mut [Complex<f64>],
                                 scratch: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, self.execution_length);

                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.destructive_scratch_length());

                for (dst_chunk, src_chunk) in dst
                    .chunks_exact_mut(self.execution_length)
                    .zip(src.chunks_exact_mut(self.execution_length)) {
                    self.process_columns_in_place(src_chunk);

                    self.width_executor.execute_with_scratch(src_chunk, scratch)?;

                    self.transpose_executor
                        .transpose(src_chunk, dst_chunk, self.width, self.height);
                }
                Ok(())
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn process_columns_oof(&self, src: &[Complex<f64>], dst: &mut [Complex<f64>]) {
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

                        let mut columns = [AvxStoreD::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreD::from_complex_ref(
                                    src.get_unchecked(index_base + len_per_row * i..),
                                );
                            }
                        }

                        let output = unsafe {
                            self.inner_bf.exec(columns)
                        };

                        unsafe {
                            output[0].write(dst.get_unchecked_mut(index_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreD as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreD::zero(); ROW_COUNT - 1];
                        for i in 0..ROW_COUNT - 1 {
                            twiddles[i] = twiddle_chunk[i];
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let v = AvxStoreD::mul_by_complex(output[i], twiddle);
                            unsafe {
                                v.write(dst.get_unchecked_mut(index_base + len_per_row * i..))
                            }
                        }
                    }

                    let partial_remainder = len_per_row % COMPLEX_PER_VECTOR;
                    if partial_remainder > 0 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base = self.twiddles.len() - TWIDDLES_PER_COLUMN;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [AvxStoreD::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreD::from_complex(
                                    src.get_unchecked(partial_remainder_base + len_per_row * i),
                                );
                            }
                        }

                        // apply our butterfly function down the columns
                        let output = unsafe {
                            self.inner_bf.exec(columns)
                        };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write_lo(dst.get_unchecked_mut(partial_remainder_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreD as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreD::zero(); ROW_COUNT - 1];
                        for i in 0..ROW_COUNT - 1 {
                            twiddles[i] = final_twiddle_chunk[i];
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let output = AvxStoreD::mul_by_complex(output[i], twiddle);
                            unsafe {
                                output.write_lo(
                                    dst.get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }
                    }

            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_oof_f64(&self, src: &[Complex<f64>], dst: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, self.execution_length);

                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.out_of_place_scratch_length());
                let (scratch, width_scratch) = scratch.split_at_mut(self.execution_length);

                for (chunk, output_chunk) in src
                    .chunks_exact(self.execution_length)
                    .zip(dst.chunks_exact_mut(self.execution_length)) {
                    self.process_columns_oof(chunk, scratch);

                    self.width_executor.execute_with_scratch(scratch, width_scratch)?;

                    self.transpose_executor
                        .transpose(&scratch, output_chunk, self.width, self.height);
                }
                Ok(())
            }
    }

    impl FftExecutor<f64> for $mx_type {
        fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
            let mut scratch = try_vec![Complex::zero(); self.scratch_length()];
            unsafe { self.execute_f64(in_place, &mut scratch) }
        }

        fn execute_with_scratch(&self, in_place: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) -> Result<(), ZaftError> {
            unsafe { self.execute_f64(in_place, scratch) }
        }

        fn execute_out_of_place(
            &self,
            src: &[Complex<f64>],
            dst: &mut [Complex<f64>],
        ) -> Result<(), ZaftError> {
            let mut scratch = try_vec![Complex::zero(); self.out_of_place_scratch_length()];
            self.execute_out_of_place_with_scratch(src, dst, &mut scratch)
        }

        fn execute_out_of_place_with_scratch(
            &self,
            src: &[Complex<f64>],
            dst: &mut [Complex<f64>],
            scratch: &mut [Complex<f64>],
        ) -> Result<(), ZaftError> {
            unsafe { self.execute_oof_f64(src, dst, scratch) }
        }

        fn execute_destructive_with_scratch(
            &self,
            src: &mut [Complex<f64>],
            dst: &mut [Complex<f64>],
            scratch: &mut [Complex<f64>],
        ) -> Result<(), ZaftError> {
            unsafe {
                self.execute_d_oof_impl(src, dst, scratch)
            }
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
    };
}

macro_rules! define_mixed_radixf {
    ($mx_type: ident, $bf_type: ident, $row_count: expr) => {
        pub(crate) struct $mx_type {
            execution_length: usize,
            direction: FftDirection,
            twiddles: Vec<AvxStoreF>,
            width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
            width: usize,
            height: usize,
            transpose_executor: Box<dyn TransposeExecutor<f32> + Send + Sync>,
            inner_bf: $bf_type,
            width_scratch_length: usize,
            oof_width_scratch_length: usize,
        }

        impl $mx_type {
            pub fn new(width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>) -> Result<Self, ZaftError> {
                unsafe { Self::new_init(width_executor) }
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn new_init(width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>) -> Result<Self, ZaftError> {
                let direction = width_executor.direction();

                let width = width_executor.length();

                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;

                // derive some info from our inner FFT
                let len_per_row = width_executor.length();

                let len = len_per_row * ROW_COUNT;
                const COMPLEX_PER_VECTOR: usize = 4;

                let quotient = len_per_row / COMPLEX_PER_VECTOR;
                let remainder = len_per_row % COMPLEX_PER_VECTOR;

                let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
                let mut twiddles = Vec::new();
                twiddles
                    .try_reserve_exact(num_twiddle_columns * TWIDDLES_PER_COLUMN)
                    .map_err(|_| ZaftError::OutOfMemory(num_twiddle_columns * TWIDDLES_PER_COLUMN))?;
                for x in 0..num_twiddle_columns {
                    for y in 1..ROW_COUNT {
                        let mut data: [Complex<f32>; 4] = [Complex::zero(); 4];
                        for i in 0..COMPLEX_PER_VECTOR {
                            data[i] = compute_twiddle(
                                y * (x * COMPLEX_PER_VECTOR + i),
                                len,
                                direction,
                            );
                        }
                        twiddles.push(AvxStoreF::from_complex_ref(data.as_ref()));
                    }
                }

                let width_scratch_length = width_executor.destructive_scratch_length();
                let oof_width_scratch_length = width_executor.scratch_length();

                #[allow(unused_unsafe)]
                Ok($mx_type {
                    execution_length: width * ROW_COUNT,
                    width_executor,
                    width,
                    height: ROW_COUNT,
                    direction,
                    twiddles,
                    transpose_executor: f32::transpose_strategy(width, ROW_COUNT),
                    inner_bf: unsafe { $bf_type::new(direction) },
                    width_scratch_length,
                    oof_width_scratch_length,
                })
            }
        }

        impl $mx_type {
            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_f32(&self, in_place: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) -> Result<(), ZaftError> {
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

                    self.width_executor.execute_destructive_with_scratch(chunk, scratch, width_scratch)?;

                    self.transpose_executor
                        .transpose(&scratch, chunk, self.width, self.height);
                }
                Ok(())
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn process_columns_in_place(&self, chunk: &mut [Complex<f32>]) {
                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
                const COMPLEX_PER_VECTOR: usize = 4;

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
                        let mut columns = [AvxStoreF::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreF::from_complex_ref(
                                    chunk.get_unchecked(index_base + len_per_row * i..),
                                );
                            }
                        }

                        let output = unsafe { self.inner_bf.exec(columns) };

                        unsafe {
                            output[0].write(chunk.get_unchecked_mut(index_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreF as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreF::zero(); ROW_COUNT - 1];
                        for i in 0..ROW_COUNT - 1 {
                            twiddles[i] = twiddle_chunk[i];
                        }

                        for i in 1..ROW_COUNT {
                            let tw = twiddles[i - 1];
                            let v = AvxStoreF::mul_by_complex(output[i], tw);
                            unsafe {
                                v.write(chunk.get_unchecked_mut(index_base + len_per_row * i..))
                            }
                        }
                    }

                    let partial_remainder = len_per_row % COMPLEX_PER_VECTOR;
                    if partial_remainder == 1 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base =
                            self.twiddles.len() - TWIDDLES_PER_COLUMN;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [AvxStoreF::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreF::from_complex(
                                    chunk.get_unchecked(partial_remainder_base + len_per_row * i),
                                );
                            }
                        }

                        // apply our butterfly function down the columns
                        let output = unsafe { self.inner_bf.exec(columns) };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write_lo1(chunk.get_unchecked_mut(partial_remainder_base..));
                        }

                        let mut twiddles = [AvxStoreF::zero(); ROW_COUNT - 1];
                        for i in 0..ROW_COUNT - 1 {
                            twiddles[i] = final_twiddle_chunk[i];
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let v = AvxStoreF::mul_by_complex(output[i], twiddle);
                            unsafe {
                                v.write_lo1(
                                    chunk.get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }
                    } else if partial_remainder == 2 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base = self.twiddles.len() - TWIDDLES_PER_COLUMN;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [AvxStoreF::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreF::from_complex2(
                                    chunk.get_unchecked(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }

                        // apply our butterfly function down the columns
                        let output = unsafe { self.inner_bf.exec(columns) };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write_lo2(chunk.get_unchecked_mut(partial_remainder_base..));
                        }

                        let mut twiddles = [AvxStoreF::zero(); ROW_COUNT - 1];
                        for i in 0..ROW_COUNT - 1 {
                            twiddles[i] = final_twiddle_chunk[i];
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let v = AvxStoreF::mul_by_complex(output[i], twiddle);
                            unsafe {
                                v.write_lo2(
                                    chunk.get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }
                    } else if partial_remainder == 3 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base = self.twiddles.len() - TWIDDLES_PER_COLUMN;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [AvxStoreF::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreF::from_complex3(
                                    chunk.get_unchecked(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }

                        // apply our butterfly function down the columns
                        let output = unsafe { self.inner_bf.exec(columns) };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write_lo3(chunk.get_unchecked_mut(partial_remainder_base..));
                        }

                        let mut twiddles = [AvxStoreF::zero(); ROW_COUNT - 1];
                        for i in 0..ROW_COUNT - 1 {
                            twiddles[i] = final_twiddle_chunk[i];
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let v = AvxStoreF::mul_by_complex(output[i], twiddle);
                            unsafe {
                                v.write_lo3(
                                    chunk.get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }
                    }
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_d_oof_impl(&self, src: &mut [Complex<f32>],
                                  dst: &mut [Complex<f32>],
                                  scratch: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, self.execution_length);

                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.destructive_scratch_length());

                for (dst_chunk, src_chunk) in dst
                    .chunks_exact_mut(self.execution_length)
                    .zip(src.chunks_exact_mut(self.execution_length)) {
                    self.process_columns_in_place(src_chunk);

                    self.width_executor.execute_with_scratch(src_chunk, scratch)?;

                    self.transpose_executor
                        .transpose(src_chunk, dst_chunk, self.width, self.height);
                }
                Ok(())
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn process_columns_oof(&self, src: &[Complex<f32>], dst: &mut [Complex<f32>]) {
                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
                const COMPLEX_PER_VECTOR: usize = 4;

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
                        let mut columns = [AvxStoreF::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreF::from_complex_ref(
                                    src.get_unchecked(index_base + len_per_row * i..),
                                );
                            }
                        }

                        let output = unsafe { self.inner_bf.exec(columns) };

                        unsafe {
                            output[0].write(dst.get_unchecked_mut(index_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreF as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreF::zero(); ROW_COUNT - 1];
                        for i in 0..ROW_COUNT - 1 {
                            twiddles[i] = twiddle_chunk[i];
                        }

                        for i in 1..ROW_COUNT {
                            let tw = twiddles[i - 1];
                            let v = AvxStoreF::mul_by_complex(output[i], tw);
                            unsafe {
                                v.write(dst.get_unchecked_mut(index_base + len_per_row * i..))
                            }
                        }
                    }

                    let partial_remainder = len_per_row % COMPLEX_PER_VECTOR;
                    if partial_remainder == 1 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base =
                            self.twiddles.len() - TWIDDLES_PER_COLUMN;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [AvxStoreF::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreF::from_complex(
                                    src.get_unchecked(partial_remainder_base + len_per_row * i),
                                );
                            }
                        }

                        // apply our butterfly function down the columns
                        let output = unsafe { self.inner_bf.exec(columns) };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write_lo1(dst.get_unchecked_mut(partial_remainder_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreF as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreF::zero(); ROW_COUNT - 1];
                        for i in 0..ROW_COUNT - 1 {
                            twiddles[i] = final_twiddle_chunk[i];
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let v = AvxStoreF::mul_by_complex(output[i], twiddle);
                            unsafe {
                                v.write_lo1(
                                    dst.get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }
                    } else if partial_remainder == 2 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base = self.twiddles.len() - TWIDDLES_PER_COLUMN;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [AvxStoreF::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreF::from_complex2(
                                    src.get_unchecked(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }

                        // apply our butterfly function down the columns
                        let output = unsafe { self.inner_bf.exec(columns) };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write_lo2(dst.get_unchecked_mut(partial_remainder_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreF as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreF::zero(); ROW_COUNT - 1];
                        for i in 0..ROW_COUNT - 1 {
                            twiddles[i] = final_twiddle_chunk[i];
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let v = AvxStoreF::mul_by_complex(output[i], twiddle);
                            unsafe {
                                v.write_lo2(
                                    dst.get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }
                    } else if partial_remainder == 3 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base = self.twiddles.len() - TWIDDLES_PER_COLUMN;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [AvxStoreF::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreF::from_complex3(
                                    src.get_unchecked(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }

                        // apply our butterfly function down the columns
                        let output = unsafe { self.inner_bf.exec(columns) };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write_lo3(dst.get_unchecked_mut(partial_remainder_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreF as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreF::zero(); ROW_COUNT - 1];
                        for i in 0..ROW_COUNT - 1 {
                            twiddles[i] = final_twiddle_chunk[i];
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let v = AvxStoreF::mul_by_complex(output[i], twiddle);
                            unsafe {
                                v.write_lo3(
                                    dst.get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }
                    }
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_oof_f32(&self, src: &[Complex<f32>], dst: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, self.execution_length);

                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.out_of_place_scratch_length());
                let (scratch, width_scratch) = scratch.split_at_mut(self.execution_length);

                for (chunk, output_chunk) in src
                    .chunks_exact(self.execution_length)
                    .zip(dst.chunks_exact_mut(self.execution_length)) {
                    self.process_columns_oof(chunk, scratch);

                    self.width_executor.execute_with_scratch(scratch, width_scratch)?;

                    self.transpose_executor
                        .transpose(&scratch, output_chunk, self.width, self.height);
                }
                Ok(())
            }
        }

        impl FftExecutor<f32> for $mx_type {
            fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                let mut scratch = try_vec![Complex::zero(); self.scratch_length()];
                unsafe { self.execute_f32(in_place, &mut scratch) }
            }

            fn execute_with_scratch(&self, in_place: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                unsafe { self.execute_f32(in_place, scratch) }
            }

            fn execute_out_of_place(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                let mut scratch = try_vec![Complex::zero(); self.out_of_place_scratch_length()];
                self.execute_out_of_place_with_scratch(src, dst, &mut scratch)
            }

            fn execute_out_of_place_with_scratch(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
                scratch: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_oof_f32(src, dst, scratch) }
            }

            fn execute_destructive_with_scratch(
                &self,
                src: &mut [Complex<f32>],
                dst: &mut [Complex<f32>],
                scratch: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                unsafe {
                    self.execute_d_oof_impl(src, dst, scratch)
                }
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
    };
}

define_mixed_radixd!(AvxMixedRadix2d, ColumnButterfly2d, 2);
define_mixed_radixf!(AvxMixedRadix2f, ColumnButterfly2f, 2);
define_mixed_radixd!(AvxMixedRadix3d, ColumnButterfly3d, 3);
define_mixed_radixf!(AvxMixedRadix3f, ColumnButterfly3f, 3);
define_mixed_radixd!(AvxMixedRadix4d, ColumnButterfly4d, 4);
define_mixed_radixf!(AvxMixedRadix4f, ColumnButterfly4f, 4);
define_mixed_radixd!(AvxMixedRadix5d, ColumnButterfly5d, 5);
define_mixed_radixf!(AvxMixedRadix5f, ColumnButterfly5f, 5);
define_mixed_radixd!(AvxMixedRadix6d, ColumnButterfly6d, 6);
define_mixed_radixf!(AvxMixedRadix6f, ColumnButterfly6f, 6);
define_mixed_radixd!(AvxMixedRadix7d, ColumnButterfly7d, 7);
define_mixed_radixf!(AvxMixedRadix7f, ColumnButterfly7f, 7);
define_mixed_radixd!(AvxMixedRadix8d, ColumnButterfly8d, 8);
define_mixed_radixf!(AvxMixedRadix8f, ColumnButterfly8f, 8);
define_mixed_radixd!(AvxMixedRadix9d, ColumnButterfly9d, 9);
define_mixed_radixf!(AvxMixedRadix9f, ColumnButterfly9f, 9);
define_mixed_radixd!(AvxMixedRadix10d, ColumnButterfly10d, 10);
define_mixed_radixf!(AvxMixedRadix10f, ColumnButterfly10f, 10);
define_mixed_radixd!(AvxMixedRadix11d, ColumnButterfly11d, 11);
define_mixed_radixf!(AvxMixedRadix11f, ColumnButterfly11f, 11);
define_mixed_radixd!(AvxMixedRadix12d, ColumnButterfly12d, 12);
define_mixed_radixf!(AvxMixedRadix12f, ColumnButterfly12f, 12);
define_mixed_radixd!(AvxMixedRadix13d, ColumnButterfly13d, 13);
define_mixed_radixf!(AvxMixedRadix13f, ColumnButterfly13f, 13);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Zaft;
    use crate::mixed_radix::MixedRadix;
    use crate::util::has_valid_avx;

    #[test]
    fn test_avx_mixed_radix_f64() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f64>; 8] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
        ];
        let neon_mixed_rust =
            AvxMixedRadix2d::new(Zaft::strategy(4, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(8, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix3_f64() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f64>; 9] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
        ];
        let neon_mixed_rust =
            AvxMixedRadix3d::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(9, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix4_f64() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f64>; 8] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
        ];
        let neon_mixed_rust =
            AvxMixedRadix4d::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(8, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix5_f64() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f64>; 10] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
        ];
        let neon_mixed_rust =
            AvxMixedRadix5d::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(10, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix6_f64() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f64>; 12] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            AvxMixedRadix6d::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(12, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix7_f64() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f64>; 14] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            AvxMixedRadix7d::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(14, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix8_f64() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f64>; 16] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
        ];
        let neon_mixed_rust =
            AvxMixedRadix8d::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = MixedRadix::new(
            Zaft::strategy(8, FftDirection::Forward).unwrap(),
            Zaft::strategy(2, FftDirection::Forward).unwrap(),
        )
        .unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix9_f64() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f64>; 18] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
        ];
        let neon_mixed_rust =
            AvxMixedRadix9d::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = MixedRadix::new(
            Zaft::strategy(9, FftDirection::Forward).unwrap(),
            Zaft::strategy(2, FftDirection::Forward).unwrap(),
        )
        .unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix10_f64() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f64>; 20] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
        ];
        let neon_mixed_rust =
            AvxMixedRadix10d::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(20, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix11_f64() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f64>; 22] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            AvxMixedRadix11d::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(22, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix12_f64() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f64>; 24] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
        ];
        let neon_mixed_rust =
            AvxMixedRadix12d::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = MixedRadix::new(
            Zaft::strategy(12, FftDirection::Forward).unwrap(),
            Zaft::strategy(2, FftDirection::Forward).unwrap(),
        )
        .unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix_f32() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f32>; 8] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
        ];
        let neon_mixed_rust =
            AvxMixedRadix2f::new(Zaft::strategy(4, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(8, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix3_f32() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f32>; 12] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            AvxMixedRadix3f::new(Zaft::strategy(4, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(12, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix4_f32() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f32>; 20] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(7.54, -0.6534),
        ];
        let neon_mixed_rust =
            AvxMixedRadix4f::new(Zaft::strategy(5, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(20, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix5_f32() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f32>; 25] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(7.54, -0.6534),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
        ];
        let neon_mixed_rust =
            AvxMixedRadix5f::new(Zaft::strategy(5, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(25, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix6_f32() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f32>; 30] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(7.54, -0.6534),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
        ];
        let neon_mixed_rust =
            AvxMixedRadix6f::new(Zaft::strategy(5, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(30, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix7_f32() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f32>; 21] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(7.54, -0.6534),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
        ];
        let neon_mixed_rust =
            AvxMixedRadix7f::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(21, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix8_f32() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f32>; 24] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(7.54, -0.6534),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
        ];
        let neon_mixed_rust =
            AvxMixedRadix8f::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(24, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix9_f32() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f32>; 27] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(7.54, -0.6534),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
        ];
        let neon_mixed_rust =
            AvxMixedRadix9f::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(27, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix10_f32() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f32>; 30] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(7.54, -0.6534),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
        ];
        let neon_mixed_rust =
            AvxMixedRadix10f::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(30, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix11_f32() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f32>; 33] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(7.54, -0.6534),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
        ];
        let neon_mixed_rust =
            AvxMixedRadix11f::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(33, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix12_f32() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f32>; 36] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(7.54, -0.6534),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(0.9, 0.13),
        ];
        let neon_mixed_rust =
            AvxMixedRadix12f::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(36, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_avx_mixed_radix13_f32() {
        if !has_valid_avx() {
            return;
        }
        let src: [Complex<f32>; 39] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(7.54, -0.6534),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(0.9, 0.13),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
        ];
        let neon_mixed_rust =
            AvxMixedRadix13f::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(39, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }
}
