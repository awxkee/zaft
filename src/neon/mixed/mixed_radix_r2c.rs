/*
 * // Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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
use crate::neon::mixed::neon_store::NeonStoreF;
use crate::transpose::{TransposeExecutor, TransposeFactory};
use crate::util::{ScratchBuffer, compute_twiddle};
use crate::{FftExecutor, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::Zero;
use std::sync::Arc;

macro_rules! define_mixed_radix_neon_d {
    ($radix_name: ident, $features: literal, $bf_name: ident, $row_count: expr, $complex_row_count: expr, $mul: ident) => {
        use crate::neon::mixed::$bf_name;
        pub(crate) struct $radix_name {
            execution_length: usize,
            twiddles: Vec<NeonStoreD>,
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
                const COMPLEX_PER_VECTOR: usize = 1;

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
                        twiddles.push(NeonStoreD::from_complex_ref(data.as_ref()));
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
                    transpose_executor: f64::transpose_strategy(
                        width - to_remove_second_stage,
                        $complex_row_count,
                    ),
                    inner_bf: $bf_name::new(direction),
                    width_scratch_length,
                    second_stage_len,
                })
            }
        }

        impl R2CFftExecutor<f64> for $radix_name {
            fn execute(&self, input: &[f64], output: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                let mut scratch =
                    ScratchBuffer::<Complex<f64>, 2048>::new(self.complex_scratch_length());
                self.execute_with_scratch(input, output, scratch.as_mut_slice())
            }

            fn execute_with_scratch(
                &self,
                input: &[f64],
                output: &mut [Complex<f64>],
                scratch: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_oof_impl(input, output, scratch) }
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
            #[target_feature(enable = $features)]
            fn process_columns(&self, src: &[f64], complex: &mut [Complex<f64>]) {
                const ROW_COUNT: usize = $row_count;
                const COMPLEX_ROW_COUNT: usize = $complex_row_count;
                const TWIDDLES_PER_COLUMN: usize = $complex_row_count - 1;
                const COMPLEX_PER_VECTOR: usize = 1;

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
                    let mut columns = [NeonStoreD::default(); ROW_COUNT];
                    for i in 0..ROW_COUNT {
                        unsafe {
                            let q = NeonStoreD::load1(
                                src.get_unchecked(index_base + len_per_row * i..),
                            );
                            columns[i] = q;
                        }
                    }

                    #[allow(unused_unsafe)]
                    let output = unsafe { self.inner_bf.exec_r2c(columns) };

                    unsafe {
                        output[0].write(complex.get_unchecked_mut(index_base..));
                    }

                    // here LLVM doesn't "see" NeonStoreD as the same type returned by output
                    // so we need to force cast it onwards to the same type
                    let mut twiddles = [NeonStoreD::default(); COMPLEX_ROW_COUNT - 1];
                    for i in 0..COMPLEX_ROW_COUNT - 1 {
                        twiddles[i] = twiddle_chunk[i];
                    }

                    for i in 1..COMPLEX_ROW_COUNT {
                        let twiddle = twiddles[i - 1];
                        let output = NeonStoreD::$mul(output[i], twiddle);
                        unsafe {
                            output.write(complex.get_unchecked_mut(index_base + len_per_row * i..))
                        }
                    }
                }
            }

            #[target_feature(enable = $features)]
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
                        dst.len(),
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
                    .zip(src.chunks_exact(self.execution_length))
                {
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

macro_rules! define_mixed_radix_neon_f {
    ($radix_name: ident, $features: literal, $bf_name: ident, $row_count: expr, $complex_row_count: expr, $mul: ident) => {
        use crate::neon::mixed::$bf_name;
        pub(crate) struct $radix_name {
            execution_length: usize,
            twiddles: Vec<NeonStoreF>,
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
                        let mut data: [Complex<f32>; COMPLEX_PER_VECTOR] =
                            [Complex::zero(); COMPLEX_PER_VECTOR];
                        for i in 0..COMPLEX_PER_VECTOR {
                            data[i] =
                                compute_twiddle(y * (x * COMPLEX_PER_VECTOR + i), len, direction);
                        }
                        twiddles.push(NeonStoreF::from_complex_ref(data.as_ref()));
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
                    transpose_executor: f32::transpose_strategy(
                        width - to_remove_second_stage,
                        $complex_row_count,
                    ),
                    inner_bf: $bf_name::new(direction),
                    width_scratch_length,
                    second_stage_len,
                })
            }
        }

        impl R2CFftExecutor<f32> for $radix_name {
            fn execute(&self, input: &[f32], output: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                let mut scratch =
                    ScratchBuffer::<Complex<f32>, 2048>::new(self.complex_scratch_length());
                self.execute_with_scratch(input, output, scratch.as_mut_slice())
            }

            fn execute_with_scratch(
                &self,
                input: &[f32],
                output: &mut [Complex<f32>],
                scratch: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_oof_impl(input, output, scratch) }
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
            #[target_feature(enable = $features)]
            fn process_columns(&self, src: &[f32], complex: &mut [Complex<f32>]) {
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
                    let mut columns = [NeonStoreF::default(); ROW_COUNT];
                    for i in 0..ROW_COUNT {
                        unsafe {
                            let q = NeonStoreF::load2(
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

                    // here LLVM doesn't "see" NeonStoreF as the same type returned by output
                    // so we need to force cast it onwards to the same type
                    let mut twiddles = [NeonStoreF::default(); COMPLEX_ROW_COUNT - 1];
                    for i in 0..COMPLEX_ROW_COUNT - 1 {
                        twiddles[i] = twiddle_chunk[i];
                    }

                    for i in 1..COMPLEX_ROW_COUNT {
                        let twiddle = twiddles[i - 1];
                        let output = NeonStoreF::$mul(output[i], twiddle);
                        unsafe {
                            output.write(complex.get_unchecked_mut(index_base + len_per_row * i..))
                        }
                    }
                }

                let partial_remainder = len_per_row % COMPLEX_PER_VECTOR;
                if partial_remainder > 0 {
                    let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                    let partial_remainder_twiddle_base = self.twiddles.len() - TWIDDLES_PER_COLUMN;
                    let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                    let mut columns = [NeonStoreF::default(); ROW_COUNT];
                    for i in 0..ROW_COUNT {
                        unsafe {
                            columns[i] = NeonStoreF::load1(
                                src.get_unchecked(partial_remainder_base + len_per_row * i..),
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

                    // here LLVM doesn't "see" NeonStoreF as the same type returned by output
                    // so we need to force cast it onwards to the same type
                    let mut twiddles = [NeonStoreF::default(); COMPLEX_ROW_COUNT - 1];
                    for i in 0..COMPLEX_ROW_COUNT - 1 {
                        twiddles[i] = final_twiddle_chunk[i];
                    }

                    // for the remaining rows, apply twiddle factors and then write back to memory
                    for i in 1..COMPLEX_ROW_COUNT {
                        let twiddle = twiddles[i - 1];
                        let output = NeonStoreF::$mul(output[i], twiddle);
                        unsafe {
                            output.write_lo(
                                complex
                                    .get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                            );
                        }
                    }
                }
            }

            #[target_feature(enable = $features)]
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
                        dst.len(),
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
                    .zip(src.chunks_exact(self.execution_length))
                {
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

use crate::neon::mixed::NeonStoreD;

define_mixed_radix_neon_f!(
    NeonR2CMixedRadix3f,
    "neon",
    ColumnButterfly3f,
    3,
    2,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f!(
    NeonFcmaR2CMixedRadix3f,
    "fcma",
    ColumnFcmaButterfly3f,
    3,
    2,
    fcmul_fcma
);
define_mixed_radix_neon_d!(
    NeonR2CMixedRadix3d,
    "neon",
    ColumnButterfly3d,
    3,
    2,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d!(
    NeonFcmaR2CMixedRadix3d,
    "fcma",
    ColumnFcmaButterfly3d,
    3,
    2,
    fcmul_fcma
);

define_mixed_radix_neon_f!(
    NeonR2CMixedRadix5f,
    "neon",
    ColumnButterfly5f,
    5,
    3,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f!(
    NeonFcmaR2CMixedRadix5f,
    "fcma",
    ColumnFcmaButterfly5f,
    5,
    3,
    fcmul_fcma
);
define_mixed_radix_neon_d!(
    NeonR2CMixedRadix5d,
    "neon",
    ColumnButterfly5d,
    5,
    3,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d!(
    NeonFcmaR2CMixedRadix5d,
    "fcma",
    ColumnFcmaButterfly5d,
    5,
    3,
    fcmul_fcma
);

define_mixed_radix_neon_f!(
    NeonR2CMixedRadix7f,
    "neon",
    ColumnButterfly7f,
    7,
    4,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f!(
    NeonFcmaR2CMixedRadix7f,
    "fcma",
    ColumnFcmaButterfly7f,
    7,
    4,
    fcmul_fcma
);
define_mixed_radix_neon_d!(
    NeonR2CMixedRadix7d,
    "neon",
    ColumnButterfly7d,
    7,
    4,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d!(
    NeonFcmaR2CMixedRadix7d,
    "fcma",
    ColumnFcmaButterfly7d,
    7,
    4,
    fcmul_fcma
);

define_mixed_radix_neon_f!(
    NeonR2CMixedRadix9f,
    "neon",
    ColumnButterfly9f,
    9,
    5,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f!(
    NeonFcmaR2CMixedRadix9f,
    "fcma",
    ColumnFcmaButterfly9f,
    9,
    5,
    fcmul_fcma
);
define_mixed_radix_neon_d!(
    NeonR2CMixedRadix9d,
    "neon",
    ColumnButterfly9d,
    9,
    5,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d!(
    NeonFcmaR2CMixedRadix9d,
    "fcma",
    ColumnFcmaButterfly9d,
    9,
    5,
    fcmul_fcma
);

define_mixed_radix_neon_f!(
    NeonR2CMixedRadix11f,
    "neon",
    ColumnButterfly11f,
    11,
    6,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f!(
    NeonFcmaR2CMixedRadix11f,
    "fcma",
    ColumnFcmaButterfly11f,
    11,
    6,
    fcmul_fcma
);
define_mixed_radix_neon_d!(
    NeonR2CMixedRadix11d,
    "neon",
    ColumnButterfly11d,
    11,
    6,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d!(
    NeonFcmaR2CMixedRadix11d,
    "fcma",
    ColumnFcmaButterfly11d,
    11,
    6,
    fcmul_fcma
);

define_mixed_radix_neon_f!(
    NeonR2CMixedRadix13f,
    "neon",
    ColumnButterfly13f,
    13,
    7,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f!(
    NeonFcmaR2CMixedRadix13f,
    "fcma",
    ColumnFcmaButterfly13f,
    13,
    7,
    fcmul_fcma
);
define_mixed_radix_neon_d!(
    NeonR2CMixedRadix13d,
    "neon",
    ColumnButterfly13d,
    13,
    7,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d!(
    NeonFcmaR2CMixedRadix13d,
    "fcma",
    ColumnFcmaButterfly13d,
    13,
    7,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use crate::dft::Dft;
    use crate::neon::mixed::mixed_radix_r2c::{NeonR2CMixedRadix5d, NeonR2CMixedRadix5f};
    use crate::{FftDirection, FftExecutor, R2CFftExecutor, Zaft};
    use num_complex::Complex;
    use num_traits::Zero;

    #[test]
    fn test_mixed_radixf() {
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
            NeonR2CMixedRadix5f::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
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
            NeonR2CMixedRadix5d::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
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
