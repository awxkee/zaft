/*
 * // Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::avx::mixed::{AvxStoreD, AvxStoreF};
use crate::transpose::{TransposeExecutor, TransposeFactory};
use crate::util::compute_twiddle;
use crate::{FftExecutor, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::Zero;
use std::sync::Arc;

macro_rules! define_mixed_radix_avx_d {
    ($radix_name: ident, $bf_name: ident, $row_count: expr, $complex_row_count: expr, $mul: ident) => {
        use crate::avx::mixed::butterflies::$bf_name;
        pub(crate) struct $radix_name {
            execution_length: usize,
            twiddles: Vec<AvxStoreD>,
            width_executor: Arc<dyn FftExecutor<f64> + Send + Sync>,
            width: usize,
            height: usize,
            transpose_executor: Box<dyn TransposeExecutor<f64> + Send + Sync>,
            inner_bf: $bf_name,
            width_scratch_length: usize,
            second_stage_len: usize,
        }

        impl $radix_name {
            pub(crate) fn new(
                width_executor: Arc<dyn FftExecutor<f64> + Send + Sync>,
            ) -> Result<Self, ZaftError> {
                unsafe {
                    Self::new_impl(width_executor)
                }
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) fn new_impl(
                width_executor: Arc<dyn FftExecutor<f64> + Send + Sync>,
            ) -> Result<Self, ZaftError> {
                let direction = width_executor.direction();

                let width = width_executor.length();
                assert!(
                    !width.is_multiple_of(2),
                    "This is an UB to call Odd Mixed-Radix R2C with even `width`"
                );

                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = $complex_row_count - 1;

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
                    for y in 1..$complex_row_count {
                        let mut data: [Complex<f64>; COMPLEX_PER_VECTOR] =
                            [Complex::zero(); COMPLEX_PER_VECTOR];
                        for i in 0..COMPLEX_PER_VECTOR {
                            data[i] =
                                compute_twiddle(y * (x * COMPLEX_PER_VECTOR + i), len, direction);
                        }
                        twiddles.push(AvxStoreD::from_complex_ref(data.as_ref()));
                    }
                }

                let width_scratch_length = width_executor.scratch_length();

                let to_remove_second_stage = (width - 1) / 2;

                let second_stage_len = $complex_row_count * width;

                Ok($radix_name {
                    execution_length: width * ROW_COUNT,
                    width_executor,
                    width,
                    height: ROW_COUNT,
                    twiddles,
                    transpose_executor: f64::transpose_strategy(width - to_remove_second_stage, $complex_row_count),
                    inner_bf: $bf_name::new(direction),
                    width_scratch_length,
                    second_stage_len,
                })
            }
        }

        impl R2CFftExecutor<f64> for $radix_name {
            fn execute(&self, input: &[f64], output: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                use crate::err::try_vec;
                let mut scratch = try_vec![Complex::zero(); self.complex_scratch_length()];
                self.execute_with_scratch(input, output, &mut scratch)
            }

            fn execute_with_scratch(&self, input: &[f64], output: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                unsafe {
                    self.execute_oof_impl(input, output, scratch)
                }
            }

            fn complex_length(&self) -> usize {
                 self.execution_length / 2 + 1
            }

            fn complex_scratch_length(&self) -> usize {
                self.second_stage_len + self.width_scratch_length
            }

            fn real_length(&self) -> usize {
                self.execution_length
            }
        }

        impl $radix_name {
            #[target_feature(enable = "avx2", enable = "fma")]
            fn process_columns(
                &self,
                src: &[f64],
                complex: &mut [Complex<f64>],
            ) {
                const ROW_COUNT: usize = $row_count;
                const COMPLEX_ROW_COUNT: usize = $complex_row_count;
                const TWIDDLES_PER_COLUMN: usize = $complex_row_count - 1;
                const COMPLEX_PER_VECTOR: usize = 2;

                let len_per_row = self.real_length() / ROW_COUNT;
                let chunk_count = len_per_row / COMPLEX_PER_VECTOR;

                for (c, twiddle_chunk) in self
                        .twiddles
                        .chunks_exact(TWIDDLES_PER_COLUMN)
                        .take(chunk_count)
                        .enumerate()
                    {
                        let index_base = c * COMPLEX_PER_VECTOR;

                        // Load columns from the input into registers
                        let mut columns = [AvxStoreD::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                let q = AvxStoreD::load2(
                                    src.get_unchecked(index_base + len_per_row * i..),
                                );
                                columns[i] = q.to_complex()[0];
                            }
                        }

                        #[allow(unused_unsafe)]
                        let output = unsafe { self.inner_bf.exec(columns) };

                        unsafe {
                            output[0].write(complex.get_unchecked_mut(index_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreD as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreD::zero(); COMPLEX_ROW_COUNT - 1];
                        for i in 0..COMPLEX_ROW_COUNT - 1 {
                            twiddles[i] = twiddle_chunk[i];
                        }

                        for i in 1..COMPLEX_ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let output = AvxStoreD::$mul(output[i], twiddle);
                            unsafe {
                                output.write(
                                    complex.get_unchecked_mut(index_base + len_per_row * i..),
                                )
                            }
                        }
                    }

                    let partial_remainder = len_per_row % COMPLEX_PER_VECTOR;
                    if partial_remainder > 0 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base =
                            self.twiddles.len() - TWIDDLES_PER_COLUMN;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [AvxStoreD::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreD::load1(
                                    src
                                        .get_unchecked(partial_remainder_base + len_per_row * i..)
                                );
                            }
                        }

                        // apply our butterfly function down the columns
                        #[allow(unused_unsafe)]
                        let output = unsafe { self.inner_bf.exec_r2c(columns) };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write_lo(complex.get_unchecked_mut(partial_remainder_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreD as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreD::zero(); COMPLEX_ROW_COUNT - 1];
                        for i in 0..COMPLEX_ROW_COUNT - 1 {
                            twiddles[i] = final_twiddle_chunk[i];
                        }

                        // for the remaining rows, apply twiddle factors and then write back to memory
                        for i in 1..COMPLEX_ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let output = AvxStoreD::$mul(output[i], twiddle);
                            unsafe {
                                output.write_lo(
                                    complex.get_unchecked_mut(
                                        partial_remainder_base + len_per_row * i..,
                                    ),
                                );
                            }
                        }
                    }

            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_oof_impl(
                &self,
                src: &[f64],
                dst: &mut [Complex<f64>],
                scratch: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
              if !src.len().is_multiple_of(self.execution_length) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        src.len(),
                        self.execution_length,
                    ));
                }
                if !dst.len().is_multiple_of(self.complex_length()) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        src.len(),
                        self.complex_length(),
                    ));
                }
                if src.len() / self.execution_length != dst.len() / self.complex_length() {
                    return Err(ZaftError::InvalidSamplesCount(
                        src.len() / self.execution_length,
                        dst.len() / self.complex_length(),
                    ));
                }

                let to_remove_second_stage = (self.width - 1) / 2;

                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.complex_scratch_length());
                let (scratch_complex1, rem_scratch) = scratch.split_at_mut(self.second_stage_len);

                for (dst_chunk, chunk) in dst
                    .chunks_exact_mut(self.complex_length())
                    .zip(src.chunks_exact(self.execution_length)) {
                    self.process_columns(chunk, scratch_complex1);

                    let (width_scratch, _) = rem_scratch.split_at_mut(self.width_scratch_length);
                    self.width_executor
                        .execute_with_scratch(scratch_complex1, width_scratch)?;

                    self.transpose_executor.transpose_strided(
                        scratch_complex1,
                        self.width,
                        dst_chunk,
                        self.height,
                        self.width - to_remove_second_stage,
                        $complex_row_count,
                    );

                    // conjugated tail
                    for x in (self.width - to_remove_second_stage)..self.width {
                        for y in 1..$complex_row_count {
                            let input_index = x + y * self.width;
                            let output_index = self.execution_length - (y + x * self.height);

                            unsafe {
                                *dst_chunk.get_unchecked_mut(output_index) =
                                    scratch_complex1.get_unchecked(input_index).conj();
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

macro_rules! define_mixed_radix_avx_f {
    ($radix_name: ident, $bf_name: ident, $row_count: expr, $complex_row_count: expr, $mul: ident) => {
        use crate::avx::mixed::butterflies::$bf_name;
        pub(crate) struct $radix_name {
            execution_length: usize,
            twiddles: Vec<AvxStoreF>,
            width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
            width: usize,
            height: usize,
            transpose_executor: Box<dyn TransposeExecutor<f32> + Send + Sync>,
            inner_bf: $bf_name,
            width_scratch_length: usize,
            second_stage_len: usize,
        }

        impl $radix_name {
            pub(crate) fn new(
                width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
            ) -> Result<Self, ZaftError> {
                unsafe {
                    Self::new_impl(width_executor)
                }
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            pub(crate) fn new_impl(
                width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
            ) -> Result<Self, ZaftError> {
                let direction = width_executor.direction();

                let width = width_executor.length();
                assert!(
                    !width.is_multiple_of(2),
                    "This is an UB to call Odd Mixed-Radix R2C with even `width`"
                );

                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = $complex_row_count - 1;

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
                    .map_err(|_| {
                        ZaftError::OutOfMemory(num_twiddle_columns * TWIDDLES_PER_COLUMN)
                    })?;
                for x in 0..num_twiddle_columns {
                    for y in 1..$complex_row_count {
                        let mut data: [Complex<f32>; COMPLEX_PER_VECTOR] =
                            [Complex::zero(); COMPLEX_PER_VECTOR];
                        for i in 0..COMPLEX_PER_VECTOR {
                            data[i] =
                                compute_twiddle(y * (x * COMPLEX_PER_VECTOR + i), len, direction);
                        }
                        twiddles.push(AvxStoreF::from_complex_ref(data.as_ref()));
                    }
                }

                let width_scratch_length = width_executor.scratch_length();

                let to_remove_second_stage = (width - 1) / 2;

                let second_stage_len = $complex_row_count * width;

                Ok($radix_name {
                    execution_length: width * ROW_COUNT,
                    width_executor,
                    width,
                    height: ROW_COUNT,
                    twiddles,
                    transpose_executor: f32::transpose_strategy(width - to_remove_second_stage, $complex_row_count),
                    inner_bf: $bf_name::new(direction),
                    width_scratch_length,
                    second_stage_len,
                })
            }
        }

        impl R2CFftExecutor<f32> for $radix_name {
            fn execute(&self, input: &[f32], output: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                use crate::err::try_vec;
                let mut scratch = try_vec![Complex::zero(); self.complex_scratch_length()];
                self.execute_with_scratch(input, output, &mut scratch)
            }

            fn execute_with_scratch(&self, input: &[f32], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                unsafe {
                    self.execute_oof_impl(input, output, scratch)
                }
            }

            fn complex_length(&self) -> usize {
                 self.execution_length / 2 + 1
            }

            fn complex_scratch_length(&self) -> usize {
                self.second_stage_len + self.width_scratch_length
            }

            fn real_length(&self) -> usize {
                self.execution_length
            }
        }

        impl $radix_name {
            #[target_feature(enable = "avx2", enable = "fma")]
            fn process_columns(
                &self,
                src: &[f32],
                complex: &mut [Complex<f32>],
            ) {
                const ROW_COUNT: usize = $row_count;
                const COMPLEX_ROW_COUNT: usize = $complex_row_count;
                const TWIDDLES_PER_COLUMN: usize = $complex_row_count - 1;
                const COMPLEX_PER_VECTOR: usize = 4;

                let len_per_row = self.real_length() / ROW_COUNT;
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
                                let q = AvxStoreF::load4(
                                    src.get_unchecked(index_base + len_per_row * i..),
                                );
                                columns[i] = q.to_complex()[0];
                            }
                        }

                        #[allow(unused_unsafe)]
                        let output = unsafe { self.inner_bf.exec(columns) };

                        unsafe {
                            output[0].write(complex.get_unchecked_mut(index_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreF as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreF::zero(); COMPLEX_ROW_COUNT - 1];
                        for i in 0..COMPLEX_ROW_COUNT - 1 {
                            twiddles[i] = twiddle_chunk[i];
                        }

                        for i in 1..COMPLEX_ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let output = AvxStoreF::$mul(output[i], twiddle);
                            unsafe {
                                output.write(
                                    complex.get_unchecked_mut(index_base + len_per_row * i..),
                                )
                            }
                        }
                    }

                    let partial_remainder = len_per_row % COMPLEX_PER_VECTOR;
                    if partial_remainder == 3 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base =
                            self.twiddles.len() - TWIDDLES_PER_COLUMN;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [AvxStoreF::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreF::load3(
                                    src
                                        .get_unchecked(partial_remainder_base + len_per_row * i..)
                                ).to_complex()[0];
                            }
                        }

                        // apply our butterfly function down the columns
                        #[allow(unused_unsafe)]
                        let output = unsafe { self.inner_bf.exec_r2c(columns) };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write_lo3(complex.get_unchecked_mut(partial_remainder_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreF as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreF::zero(); COMPLEX_ROW_COUNT - 1];
                        for i in 0..COMPLEX_ROW_COUNT - 1 {
                            twiddles[i] = final_twiddle_chunk[i];
                        }

                        // for the remaining rows, apply twiddle factors and then write back to memory
                        for i in 1..COMPLEX_ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let output = AvxStoreF::$mul(output[i], twiddle);
                            unsafe {
                                output.write_lo3(
                                    complex.get_unchecked_mut(
                                        partial_remainder_base + len_per_row * i..,
                                    ),
                                );
                            }
                        }
                    } else if partial_remainder == 2 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base =
                            self.twiddles.len() - TWIDDLES_PER_COLUMN;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [AvxStoreF::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreF::load2(
                                    src
                                        .get_unchecked(partial_remainder_base + len_per_row * i..)
                                ).to_complex()[0];
                            }
                        }

                        // apply our butterfly function down the columns
                        #[allow(unused_unsafe)]
                        let output = unsafe { self.inner_bf.exec_r2c(columns) };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write_lo2(complex.get_unchecked_mut(partial_remainder_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreF as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreF::zero(); COMPLEX_ROW_COUNT - 1];
                        for i in 0..COMPLEX_ROW_COUNT - 1 {
                            twiddles[i] = final_twiddle_chunk[i];
                        }

                        // for the remaining rows, apply twiddle factors and then write back to memory
                        for i in 1..COMPLEX_ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let output = AvxStoreF::$mul(output[i], twiddle);
                            unsafe {
                                output.write_lo2(
                                    complex.get_unchecked_mut(
                                        partial_remainder_base + len_per_row * i..,
                                    ),
                                );
                            }
                        }
                    } else if partial_remainder == 1 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base =
                            self.twiddles.len() - TWIDDLES_PER_COLUMN;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [AvxStoreF::zero(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = AvxStoreF::load1(
                                    src
                                        .get_unchecked(partial_remainder_base + len_per_row * i..)
                                );
                            }
                        }

                        // apply our butterfly function down the columns
                        #[allow(unused_unsafe)]
                        let output = unsafe { self.inner_bf.exec_r2c(columns) };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write_lo1(complex.get_unchecked_mut(partial_remainder_base..));
                        }

                        // here LLVM doesn't "see" AvxStoreF as the same type returned by output
                        // so we need to force cast it onwards to the same type
                        let mut twiddles = [AvxStoreF::zero(); COMPLEX_ROW_COUNT - 1];
                        for i in 0..COMPLEX_ROW_COUNT - 1 {
                            twiddles[i] = final_twiddle_chunk[i];
                        }

                        // for the remaining rows, apply twiddle factors and then write back to memory
                        for i in 1..COMPLEX_ROW_COUNT {
                            let twiddle = twiddles[i - 1];
                            let output = AvxStoreF::$mul(output[i], twiddle);
                            unsafe {
                                output.write_lo1(
                                    complex.get_unchecked_mut(
                                        partial_remainder_base + len_per_row * i..,
                                    ),
                                );
                            }
                        }
                    }

            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_oof_impl(
                &self,
                src: &[f32],
                dst: &mut [Complex<f32>],
                scratch: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
              if !src.len().is_multiple_of(self.execution_length) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        src.len(),
                        self.execution_length,
                    ));
                }
                if !dst.len().is_multiple_of(self.complex_length()) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        src.len(),
                        self.complex_length(),
                    ));
                }
                if src.len() / self.execution_length != dst.len() / self.complex_length() {
                    return Err(ZaftError::InvalidSamplesCount(
                        src.len() / self.execution_length,
                        dst.len() / self.complex_length(),
                    ));
                }

                let to_remove_second_stage = (self.width - 1) / 2;

                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.complex_scratch_length());
                let (scratch_complex1, rem_scratch) = scratch.split_at_mut(self.second_stage_len);

                for (dst_chunk, chunk) in dst
                    .chunks_exact_mut(self.complex_length())
                    .zip(src.chunks_exact(self.execution_length)) {
                    self.process_columns(chunk, scratch_complex1);

                    let (width_scratch, _) = rem_scratch.split_at_mut(self.width_scratch_length);
                    self.width_executor
                        .execute_with_scratch(scratch_complex1, width_scratch)?;

                    self.transpose_executor.transpose_strided(
                        scratch_complex1,
                        self.width,
                        dst_chunk,
                        self.height,
                        self.width - to_remove_second_stage,
                        $complex_row_count,
                    );

                    // conjugated tail
                    for x in (self.width - to_remove_second_stage)..self.width {
                        for y in 1..$complex_row_count {
                            let input_index = x + y * self.width;
                            let output_index = self.execution_length - (y + x * self.height);

                            unsafe {
                                *dst_chunk.get_unchecked_mut(output_index) =
                                    scratch_complex1.get_unchecked(input_index).conj();
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

define_mixed_radix_avx_d!(AvxR2CMixedRadix3d, ColumnButterfly3d, 3, 2, mul_by_complex);
define_mixed_radix_avx_f!(AvxR2CMixedRadix3f, ColumnButterfly3f, 3, 2, mul_by_complex);
define_mixed_radix_avx_d!(AvxR2CMixedRadix5d, ColumnButterfly5d, 5, 3, mul_by_complex);
define_mixed_radix_avx_f!(AvxR2CMixedRadix5f, ColumnButterfly5f, 5, 3, mul_by_complex);
define_mixed_radix_avx_d!(AvxR2CMixedRadix7d, ColumnButterfly7d, 7, 4, mul_by_complex);
define_mixed_radix_avx_f!(AvxR2CMixedRadix7f, ColumnButterfly7f, 7, 4, mul_by_complex);
define_mixed_radix_avx_d!(AvxR2CMixedRadix9d, ColumnButterfly9d, 9, 5, mul_by_complex);
define_mixed_radix_avx_f!(AvxR2CMixedRadix9f, ColumnButterfly9f, 9, 5, mul_by_complex);
define_mixed_radix_avx_d!(
    AvxR2CMixedRadix11d,
    ColumnButterfly11d,
    11,
    6,
    mul_by_complex
);
define_mixed_radix_avx_f!(
    AvxR2CMixedRadix11f,
    ColumnButterfly11f,
    11,
    6,
    mul_by_complex
);
define_mixed_radix_avx_d!(
    AvxR2CMixedRadix13d,
    ColumnButterfly13d,
    13,
    7,
    mul_by_complex
);
define_mixed_radix_avx_f!(
    AvxR2CMixedRadix13f,
    ColumnButterfly13f,
    13,
    7,
    mul_by_complex
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::Dft;
    use crate::util::has_valid_avx;
    use crate::{FftDirection, FftExecutor, R2CFftExecutor, Zaft};
    use num_complex::Complex;
    use num_traits::Zero;

    #[test]
    fn test_mixed_radixf() {
        if !has_valid_avx() {
            return;
        }
        let src: [f32; 15] = [
            7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.4, 3.4, 2.1, 3.2, 3.3, 9.8, 5.1,
        ];

        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(15, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        println!("DFT -----");

        for chunk in (&reference_value[..10]).chunks_exact(5) {
            println!("{:?}", chunk);
        }

        let local_r2c =
            AvxR2CMixedRadix5f::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let mut complex_output = vec![Complex::zero(); 15 / 2 + 1];
        local_r2c.execute(&src, &mut complex_output).unwrap();

        reference_value
            .iter()
            .zip(complex_output.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-3,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-3,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_mixed_radixd() {
        if !has_valid_avx() {
            return;
        }
        let src: [f64; 15] = [
            7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.4, 3.4, 2.1, 3.2, 3.3, 9.8, 5.1,
        ];

        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(15, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        println!("DFT -----");

        for chunk in (&reference_value[..10]).chunks_exact(5) {
            println!("{:?}", chunk);
        }

        let local_r2c =
            AvxR2CMixedRadix5d::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let mut complex_output = vec![Complex::zero(); 15 / 2 + 1];
        local_r2c.execute(&src, &mut complex_output).unwrap();

        reference_value
            .iter()
            .zip(complex_output.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-3,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-3,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }
}
