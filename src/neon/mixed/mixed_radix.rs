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
use crate::err::try_vec;
use crate::neon::mixed::bf2::{ColumnButterfly2d, ColumnButterfly2f};
use crate::neon::mixed::bf3::{ColumnButterfly3d, ColumnButterfly3f};
use crate::neon::mixed::bf4::{ColumnButterfly4d, ColumnButterfly4f};
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF, NeonStoreFh};
use crate::transpose::{TransposeExecutor, TransposeFactory};
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::Zero;
use std::sync::Arc;

macro_rules! define_mixed_radix_neon_d {
    ($radix_name: ident, $bf_name: ident, $row_count: expr) => {
        pub(crate) struct $radix_name {
            execution_length: usize,
            direction: FftDirection,
            twiddles: Vec<Complex<f64>>,
            width_executor: Arc<dyn FftExecutor<f64> + Send + Sync>,
            width: usize,
            height: usize,
            transpose_executor: Box<dyn TransposeExecutor<f64> + Send + Sync>,
            inner_bf: $bf_name,
        }

        impl $radix_name {
            pub(crate) fn new(
                width_executor: Arc<dyn FftExecutor<f64> + Send + Sync>,
            ) -> Result<Self, ZaftError> {
                let direction = width_executor.direction();

                let width = width_executor.length();

                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;

                // derive some info from our inner FFT
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
                    .map_err(|_| ZaftError::OutOfMemory(num_twiddle_columns * TWIDDLES_PER_COLUMN))?;
                for x in 0..num_twiddle_columns {
                    for y in 1..ROW_COUNT {
                        for i in 0..COMPLEX_PER_VECTOR {
                            twiddles.push(compute_twiddle(
                                y * (x * COMPLEX_PER_VECTOR + i),
                                len,
                                direction,
                            ));
                        }
                    }
                }

                Ok($radix_name {
                    execution_length: width * ROW_COUNT,
                    width_executor,
                    width,
                    height: ROW_COUNT,
                    direction,
                    twiddles,
                    transpose_executor: f64::transpose_strategy(width, ROW_COUNT),
                    inner_bf: $bf_name::new(direction),
                })
            }
        }

        impl FftExecutor<f64> for $radix_name {
            fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if in_place.len() % self.execution_length != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.execution_length,
                    ));
                }

                let mut scratch = try_vec![Complex::zero(); self.execution_length];

                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
                const COMPLEX_PER_VECTOR: usize = 1;

                let len_per_row = self.length() / ROW_COUNT;
                let chunk_count = len_per_row / COMPLEX_PER_VECTOR;

                for chunk in in_place.chunks_exact_mut(self.execution_length) {
                    // process the column FFTs
                    for (c, twiddle_chunk) in self
                        .twiddles
                        .chunks_exact(TWIDDLES_PER_COLUMN * COMPLEX_PER_VECTOR)
                        .take(chunk_count)
                        .enumerate()
                    {
                        let index_base = c * COMPLEX_PER_VECTOR;

                        let mut columns = [NeonStoreD::default(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = NeonStoreD::from_complex_ref(
                                    chunk.get_unchecked(index_base + len_per_row * i..),
                                );
                            }
                        }

                        let output = self.inner_bf.exec(columns);

                        unsafe {
                            output[0].write(scratch.get_unchecked_mut(index_base..));
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = twiddle_chunk[i - 1];
                            let output =
                                NeonStoreD::mul_by_complex(output[i], NeonStoreD::from_complex(&twiddle));
                            unsafe {
                                output.write(scratch.get_unchecked_mut(index_base + len_per_row * i..))
                            }
                        }
                    }

                    self.width_executor.execute(&mut scratch)?;

                    self.transpose_executor
                        .transpose(&scratch, chunk, self.width, self.height);
                }
                Ok(())
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                self.execution_length
            }
        }
    };
}

#[cfg(feature = "fcma")]
macro_rules! define_mixed_radix_neon_d_fcma {
    ($radix_name: ident, $bf_name: ident, $row_count: expr) => {
        pub(crate) struct $radix_name {
            execution_length: usize,
            direction: FftDirection,
            twiddles: Vec<Complex<f64>>,
            width_executor: Arc<dyn FftExecutor<f64> + Send + Sync>,
            width: usize,
            height: usize,
            transpose_executor: Box<dyn TransposeExecutor<f64> + Send + Sync>,
            inner_bf: $bf_name,
        }

        impl $radix_name {
            pub(crate) fn new(
                width_executor: Arc<dyn FftExecutor<f64> + Send + Sync>,
            ) -> Result<Self, ZaftError> {
                let direction = width_executor.direction();

                let width = width_executor.length();

                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;

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
                    .map_err(|_| ZaftError::OutOfMemory(num_twiddle_columns * TWIDDLES_PER_COLUMN))?;
                for x in 0..num_twiddle_columns {
                    for y in 1..ROW_COUNT {
                        for i in 0..COMPLEX_PER_VECTOR {
                            twiddles.push(compute_twiddle(
                                y * (x * COMPLEX_PER_VECTOR + i),
                                len,
                                direction,
                            ));
                        }
                    }
                }

                Ok($radix_name {
                    execution_length: width * ROW_COUNT,
                    width_executor,
                    width,
                    height: ROW_COUNT,
                    direction,
                    twiddles,
                    transpose_executor: f64::transpose_strategy(width, ROW_COUNT),
                    inner_bf: $bf_name::new(direction),
                })
            }
        }

        impl $radix_name {
            #[target_feature(enable = "fcma")]
            unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if in_place.len() % self.execution_length != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.execution_length,
                    ));
                }

                let mut scratch = try_vec![Complex::zero(); self.execution_length];

                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
                const COMPLEX_PER_VECTOR: usize = 1;

                let len_per_row = self.length() / ROW_COUNT;
                let chunk_count = len_per_row / COMPLEX_PER_VECTOR;

                for chunk in in_place.chunks_exact_mut(self.execution_length) {
                    // process the column FFTs
                    for (c, twiddle_chunk) in self
                        .twiddles
                        .chunks_exact(TWIDDLES_PER_COLUMN * COMPLEX_PER_VECTOR)
                        .take(chunk_count)
                        .enumerate()
                    {
                        let index_base = c * COMPLEX_PER_VECTOR;

                        let mut columns = [NeonStoreD::default(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = NeonStoreD::from_complex_ref(
                                    chunk.get_unchecked(index_base + len_per_row * i..),
                                );
                            }
                        }

                        #[allow(unused_unsafe)]
                        let output = unsafe {
                            self.inner_bf.exec(columns)
                        };

                        unsafe {
                            output[0].write(scratch.get_unchecked_mut(index_base..));
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = twiddle_chunk[i - 1];
                            unsafe {
                                let output =
                                    NeonStoreD::fcmul_fcma(output[i], NeonStoreD::from_complex(&twiddle));
                                output.write(scratch.get_unchecked_mut(index_base + len_per_row * i..))
                            }
                        }
                    }

                    self.width_executor.execute(&mut scratch)?;

                    self.transpose_executor
                        .transpose(&scratch, chunk, self.width, self.height);
                }
                Ok(())
            }
        }

        impl FftExecutor<f64> for $radix_name {
            fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                unsafe {
                    self.execute_f64(in_place)
                }
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                self.execution_length
            }
        }
    };
}

macro_rules! define_mixed_radix_neon_f {
    ($radix_name: ident, $bf_name: ident, $row_count: expr) => {
        pub(crate) struct $radix_name {
            execution_length: usize,
            direction: FftDirection,
            twiddles: Vec<Complex<f32>>,
            width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
            width: usize,
            height: usize,
            transpose_executor: Box<dyn TransposeExecutor<f32> + Send + Sync>,
            inner_bf: $bf_name,
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
                    .map_err(|_| ZaftError::OutOfMemory(num_twiddle_columns * TWIDDLES_PER_COLUMN))?;
                for x in 0..num_twiddle_columns {
                    for y in 1..ROW_COUNT {
                        for i in 0..COMPLEX_PER_VECTOR {
                            twiddles.push(compute_twiddle(
                                y * (x * COMPLEX_PER_VECTOR + i),
                                len,
                                direction,
                            ));
                        }
                    }
                }

                Ok($radix_name {
                    execution_length: width * ROW_COUNT,
                    width_executor,
                    width,
                    height: ROW_COUNT,
                    direction,
                    twiddles,
                    transpose_executor: f32::transpose_strategy(width, ROW_COUNT),
                    inner_bf: $bf_name::new(direction),
                })
            }
        }

        impl FftExecutor<f32> for $radix_name {
            fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if in_place.len() % self.execution_length != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.execution_length,
                    ));
                }

                let mut scratch = try_vec![Complex::default(); self.execution_length];

                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
                const COMPLEX_PER_VECTOR: usize = 2;

                let len_per_row = self.length() / ROW_COUNT;
                let chunk_count = len_per_row / COMPLEX_PER_VECTOR;

                for chunk in in_place.chunks_exact_mut(self.execution_length) {
                    for (c, twiddle_chunk) in self
                        .twiddles
                        .chunks_exact(TWIDDLES_PER_COLUMN * COMPLEX_PER_VECTOR)
                        .take(chunk_count)
                        .enumerate()
                    {
                        let index_base = c * COMPLEX_PER_VECTOR;

                        // Load columns from the input into registers
                        let mut columns = [NeonStoreF::default(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = NeonStoreF::from_complex_ref(
                                    chunk.get_unchecked(index_base + len_per_row * i..),
                                );
                            }
                        }

                        #[allow(unused_unsafe)]
                        let output = unsafe {
                            self.inner_bf.exec(columns)
                        };


                        unsafe {
                            output[0].write(scratch.get_unchecked_mut(index_base..));
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = &twiddle_chunk[i * COMPLEX_PER_VECTOR - COMPLEX_PER_VECTOR..];
                            let output = NeonStoreF::mul_by_complex(
                                output[i],
                                NeonStoreF::load(twiddle.as_ptr().cast()),
                            );
                            unsafe {
                                output.write(scratch.get_unchecked_mut(index_base + len_per_row * i..))
                            }
                        }
                    }

                    let partial_remainder = len_per_row % COMPLEX_PER_VECTOR;
                    if partial_remainder > 0 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base =
                            self.twiddles.len() - TWIDDLES_PER_COLUMN * COMPLEX_PER_VECTOR;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [NeonStoreFh::default(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = NeonStoreFh::load(
                                    chunk
                                        .get_unchecked(partial_remainder_base + len_per_row * i..)
                                        .as_ptr()
                                        .cast(),
                                );
                            }
                        }

                        // apply our butterfly function down the columns
                        #[allow(unused_unsafe)]
                        let output = unsafe {
                            self.inner_bf.exech(columns)
                        };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write(scratch.get_unchecked_mut(partial_remainder_base..));
                        }

                        // for the remaining rows, apply twiddle factors and then write back to memory
                        for i in 1..ROW_COUNT {
                            let twiddle = final_twiddle_chunk[i * COMPLEX_PER_VECTOR - COMPLEX_PER_VECTOR];
                            let output =
                                NeonStoreFh::mul_by_complex(output[i], NeonStoreFh::from_complex(&twiddle));
                            unsafe {
                                output.write(
                                    scratch.get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }
                    }

                    self.width_executor.execute(&mut scratch)?;

                    self.transpose_executor
                        .transpose(&scratch, chunk, self.width, self.height);
                }
                Ok(())
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                self.execution_length
            }
        }

    };
}

#[cfg(feature = "fcma")]
macro_rules! define_mixed_radix_neon_f_fcma {
    ($radix_name: ident, $bf_name: ident, $row_count: expr) => {
        pub(crate) struct $radix_name {
            execution_length: usize,
            direction: FftDirection,
            twiddles: Vec<Complex<f32>>,
            width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
            width: usize,
            height: usize,
            transpose_executor: Box<dyn TransposeExecutor<f32> + Send + Sync>,
            inner_bf: $bf_name,
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
                    .map_err(|_| ZaftError::OutOfMemory(num_twiddle_columns * TWIDDLES_PER_COLUMN))?;
                for x in 0..num_twiddle_columns {
                    for y in 1..ROW_COUNT {
                        for i in 0..COMPLEX_PER_VECTOR {
                            twiddles.push(compute_twiddle(
                                y * (x * COMPLEX_PER_VECTOR + i),
                                len,
                                direction,
                            ));
                        }
                    }
                }

                Ok($radix_name {
                    execution_length: width * ROW_COUNT,
                    width_executor,
                    width,
                    height: ROW_COUNT,
                    direction,
                    twiddles,
                    transpose_executor: f32::transpose_strategy(width, ROW_COUNT),
                    inner_bf: $bf_name::new(direction),
                })
            }
        }


        impl $radix_name {
            #[target_feature(enable = "fcma")]
            unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if in_place.len() % self.execution_length != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.execution_length,
                    ));
                }

                let mut scratch = try_vec![Complex::default(); self.execution_length];

                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
                const COMPLEX_PER_VECTOR: usize = 2;

                let len_per_row = self.length() / ROW_COUNT;
                let chunk_count = len_per_row / COMPLEX_PER_VECTOR;

                for chunk in in_place.chunks_exact_mut(self.execution_length) {
                    for (c, twiddle_chunk) in self
                        .twiddles
                        .chunks_exact(TWIDDLES_PER_COLUMN * COMPLEX_PER_VECTOR)
                        .take(chunk_count)
                        .enumerate()
                    {
                        let index_base = c * COMPLEX_PER_VECTOR;

                        // Load columns from the input into registers
                        let mut columns = [NeonStoreF::default(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = NeonStoreF::from_complex_ref(
                                    chunk.get_unchecked(index_base + len_per_row * i..),
                                );
                            }
                        }

                        #[allow(unused_unsafe)]
                        let output = unsafe {
                            self.inner_bf.exec(columns)
                        };

                        unsafe {
                            output[0].write(scratch.get_unchecked_mut(index_base..));
                        }

                        for i in 1..ROW_COUNT {
                            let twiddle = &twiddle_chunk[i * COMPLEX_PER_VECTOR - COMPLEX_PER_VECTOR..];
                            unsafe {
                                let output = NeonStoreF::fcmul_fcma(
                                    output[i],
                                    NeonStoreF::load(twiddle.as_ptr().cast()),
                                );
                                output.write(scratch.get_unchecked_mut(index_base + len_per_row * i..))
                            }
                        }
                    }

                    let partial_remainder = len_per_row % COMPLEX_PER_VECTOR;
                    if partial_remainder > 0 {
                        let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
                        let partial_remainder_twiddle_base =
                            self.twiddles.len() - TWIDDLES_PER_COLUMN * COMPLEX_PER_VECTOR;
                        let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];

                        let mut columns = [NeonStoreFh::default(); ROW_COUNT];
                        for i in 0..ROW_COUNT {
                            unsafe {
                                columns[i] = NeonStoreFh::load(
                                    chunk
                                        .get_unchecked(partial_remainder_base + len_per_row * i..)
                                        .as_ptr()
                                        .cast(),
                                );
                            }
                        }

                        // apply our butterfly function down the columns
                        #[allow(unused_unsafe)]
                        let output = unsafe {
                            self.inner_bf.exech(columns)
                        };

                        // always write the first row without twiddles
                        unsafe {
                            output[0].write(scratch.get_unchecked_mut(partial_remainder_base..));
                        }

                        // for the remaining rows, apply twiddle factors and then write back to memory
                        for i in 1..ROW_COUNT {
                            let twiddle = final_twiddle_chunk[i * COMPLEX_PER_VECTOR - COMPLEX_PER_VECTOR];
                            unsafe {
                                let output =
                                    NeonStoreFh::fcmul_fcma(output[i], NeonStoreFh::from_complex(&twiddle));
                                output.write(
                                    scratch.get_unchecked_mut(partial_remainder_base + len_per_row * i..),
                                );
                            }
                        }
                    }

                    self.width_executor.execute(&mut scratch)?;

                    self.transpose_executor
                        .transpose(&scratch, chunk, self.width, self.height);
                }
                Ok(())
            }
        }

        impl FftExecutor<f32> for $radix_name {
            fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                unsafe {
                    self.execute_f32(in_place)
                }
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                self.execution_length
            }
        }

    };
}

use crate::neon::mixed::bf5::*;
use crate::neon::mixed::bf6::*;
use crate::neon::mixed::bf7::*;
use crate::neon::mixed::bf8::*;
use crate::neon::mixed::bf9::*;
use crate::neon::mixed::bf10::*;
use crate::neon::mixed::bf11::*;
use crate::neon::mixed::bf12::*;
use crate::neon::mixed::bf13::*;
use crate::neon::mixed::bf16::*;

define_mixed_radix_neon_d!(NeonMixedRadix2, ColumnButterfly2d, 2);
define_mixed_radix_neon_d!(NeonMixedRadix3, ColumnButterfly3d, 3);
define_mixed_radix_neon_d!(NeonMixedRadix4, ColumnButterfly4d, 4);
define_mixed_radix_neon_d!(NeonMixedRadix5, ColumnButterfly5d, 5);
define_mixed_radix_neon_d!(NeonMixedRadix6, ColumnButterfly6d, 6);
define_mixed_radix_neon_d!(NeonMixedRadix7, ColumnButterfly7d, 7);
define_mixed_radix_neon_d!(NeonMixedRadix8, ColumnButterfly8d, 8);
define_mixed_radix_neon_d!(NeonMixedRadix9, ColumnButterfly9d, 9);
define_mixed_radix_neon_d!(NeonMixedRadix10, ColumnButterfly10d, 10);
define_mixed_radix_neon_d!(NeonMixedRadix11, ColumnButterfly11d, 11);
define_mixed_radix_neon_d!(NeonMixedRadix12, ColumnButterfly12d, 12);
define_mixed_radix_neon_d!(NeonMixedRadix13, ColumnButterfly13d, 13);
define_mixed_radix_neon_d!(NeonMixedRadix16, ColumnButterfly16d, 16);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix2, ColumnButterfly2d, 2);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix3, ColumnButterfly3d, 3);
#[cfg(feature = "fcma")]
use crate::neon::mixed::bf4::{ColumnFcmaButterfly4d, ColumnFcmaButterfly4f};

#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix4, ColumnFcmaButterfly4d, 4);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix5, ColumnFcmaButterfly5d, 5);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix6, ColumnFcmaButterfly6d, 6);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix7, ColumnFcmaButterfly7d, 7);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix8, ColumnFcmaButterfly8d, 8);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix9, ColumnFcmaButterfly9d, 9);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix10, ColumnFcmaButterfly10d, 10);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix11, ColumnFcmaButterfly11d, 11);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix12, ColumnFcmaButterfly12d, 12);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix13, ColumnFcmaButterfly13d, 13);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d_fcma!(NeonFcmaMixedRadix16, ColumnFcmaButterfly16d, 16);
define_mixed_radix_neon_f!(NeonMixedRadix2f, ColumnButterfly2f, 2);
define_mixed_radix_neon_f!(NeonMixedRadix3f, ColumnButterfly3f, 3);
define_mixed_radix_neon_f!(NeonMixedRadix4f, ColumnButterfly4f, 4);
define_mixed_radix_neon_f!(NeonMixedRadix5f, ColumnButterfly5f, 5);
define_mixed_radix_neon_f!(NeonMixedRadix6f, ColumnButterfly6f, 6);
define_mixed_radix_neon_f!(NeonMixedRadix7f, ColumnButterfly7f, 7);
define_mixed_radix_neon_f!(NeonMixedRadix8f, ColumnButterfly8f, 8);
define_mixed_radix_neon_f!(NeonMixedRadix9f, ColumnButterfly9f, 9);
define_mixed_radix_neon_f!(NeonMixedRadix10f, ColumnButterfly10f, 10);
define_mixed_radix_neon_f!(NeonMixedRadix11f, ColumnButterfly11f, 11);
define_mixed_radix_neon_f!(NeonMixedRadix12f, ColumnButterfly12f, 12);
define_mixed_radix_neon_f!(NeonMixedRadix13f, ColumnButterfly13f, 13);
define_mixed_radix_neon_f!(NeonMixedRadix16f, ColumnButterfly16f, 16);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix2f, ColumnButterfly2f, 2);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix3f, ColumnButterfly3f, 3);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix4f, ColumnFcmaButterfly4f, 4);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix5f, ColumnFcmaButterfly5f, 5);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix6f, ColumnFcmaButterfly6f, 6);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix7f, ColumnFcmaButterfly7f, 7);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix8f, ColumnFcmaButterfly8f, 8);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix9f, ColumnFcmaButterfly9f, 9);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix10f, ColumnFcmaButterfly10f, 10);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix11f, ColumnFcmaButterfly11f, 11);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix12f, ColumnFcmaButterfly12f, 12);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix13f, ColumnFcmaButterfly13f, 13);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_f_fcma!(NeonFcmaMixedRadix16f, ColumnFcmaButterfly16f, 16);

// pub(crate) struct NeonMixedRadix<T> {
//     execution_length: usize,
//     direction: FftDirection,
//     twiddles: Vec<Complex<T>>,
//     width_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
//     width: usize,
//     height: usize,
//     transpose_executor: Arc<dyn TransposeExecutor<T> + Send + Sync>,
//     inner_bf: ColumnButterfly2<T>,
// }
//
// impl<T: Copy + 'static + FftTrigonometry + Float + TransposeFactory<T>> NeonMixedRadix<T>
// where
//     f64: AsPrimitive<T>,
// {
//     pub fn new(width_executor: Arc<dyn FftExecutor<T> + Send + Sync>) -> Result<Self, ZaftError> {
//         let direction = width_executor.direction();
//
//         let width = width_executor.length();
//
//         const ROW_COUNT: usize = 2;
//         const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
//
//         // derive some info from our inner FFT
//         let len_per_row = width_executor.length();
//
//         let len = len_per_row * ROW_COUNT;
//         const COMPLEX_PER_VECTOR: usize = 2;
//
//         // We're going to process each row of the FFT one AVX register at a time. We need to know how many AVX registers each row can fit,
//         // and if the last register in each row going to have partial data (ie a remainder)
//         let quotient = len_per_row / COMPLEX_PER_VECTOR;
//         let remainder = len_per_row % COMPLEX_PER_VECTOR;
//
//         let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
//         let mut twiddles = Vec::with_capacity(num_twiddle_columns * TWIDDLES_PER_COLUMN);
//         for x in 0..num_twiddle_columns {
//             for y in 1..ROW_COUNT {
//                 for i in 0..COMPLEX_PER_VECTOR {
//                     twiddles.push(compute_twiddle(
//                         y * (x * COMPLEX_PER_VECTOR + i),
//                         len,
//                         direction,
//                     ));
//                 }
//             }
//         }
//
//         Ok(NeonMixedRadix {
//             execution_length: width * ROW_COUNT,
//             width_executor,
//             width,
//             height: ROW_COUNT,
//             direction,
//             twiddles,
//             transpose_executor: T::transpose_strategy(width, ROW_COUNT),
//             inner_bf: ColumnButterfly2::new(direction),
//         })
//     }
// }
//
// impl FftExecutor<f64> for NeonMixedRadix<f64> {
//     fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
//         if in_place.len() % self.execution_length != 0 {
//             return Err(ZaftError::InvalidSizeMultiplier(
//                 in_place.len(),
//                 self.execution_length,
//             ));
//         }
//
//         let mut scratch = try_vec![Complex::zero(); self.execution_length];
//
//         // a0 a1 a2 a3
//         // a4 a5 a6 a7
//         // a8 a9 a10 a11
//         // a12 a13 a14 a15
//
//         // a3 a7 a11 a15
//         // a2 a6 a10 a14
//         // a1 a5 a9 a13
//         // a0 a4 a8 a12
//
//         for chunk in in_place.chunks_exact_mut(self.execution_length) {
//             const ROW_COUNT: usize = 2;
//             const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
//             const COMPLEX_PER_VECTOR: usize = 1;
//
//             let len_per_row = self.length() / ROW_COUNT;
//             let chunk_count = len_per_row / COMPLEX_PER_VECTOR;
//
//             // process the column FFTs
//             for (c, twiddle_chunk) in self
//                 .twiddles
//                 .chunks_exact(TWIDDLES_PER_COLUMN * COMPLEX_PER_VECTOR)
//                 .take(chunk_count)
//                 .enumerate()
//             {
//                 let index_base = c * COMPLEX_PER_VECTOR;
//
//                 // Load columns from the input into registers
//                 let mut columns = [NeonStoreD::default(); ROW_COUNT];
//                 for i in 0..ROW_COUNT {
//                     unsafe {
//                         columns[i] = NeonStoreD::from_complex_ref(
//                             chunk.get_unchecked(index_base + len_per_row * i..),
//                         );
//                     }
//                 }
//
//                 let output = self.inner_bf.exec(columns);
//
//                 unsafe {
//                     output[0].write(scratch.get_unchecked_mut(index_base..));
//                 }
//
//                 for i in 1..ROW_COUNT {
//                     let twiddle = twiddle_chunk[i - 1];
//                     let output =
//                         NeonStoreD::mul_by_complex(output[i], NeonStoreD::from_complex(&twiddle));
//                     unsafe {
//                         output.write(scratch.get_unchecked_mut(index_base + len_per_row * i..))
//                     }
//                 }
//             }
//
//             self.width_executor.execute(&mut scratch)?;
//
//             self.transpose_executor
//                 .transpose(&scratch, chunk, self.width, self.height);
//         }
//         Ok(())
//     }
//
//     fn direction(&self) -> FftDirection {
//         self.direction
//     }
//
//     #[inline]
//     fn length(&self) -> usize {
//         self.execution_length
//     }
// }
//
// impl FftExecutor<f32> for NeonMixedRadix<f32> {
//     fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
//         if in_place.len() % self.execution_length != 0 {
//             return Err(ZaftError::InvalidSizeMultiplier(
//                 in_place.len(),
//                 self.execution_length,
//             ));
//         }
//
//         let mut scratch = try_vec![Complex::zero(); self.execution_length];
//
//         const ROW_COUNT: usize = 2;
//         const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
//         const COMPLEX_PER_VECTOR: usize = 2;
//
//         let len_per_row = self.length() / ROW_COUNT;
//         let chunk_count = len_per_row / COMPLEX_PER_VECTOR;
//
//         for chunk in in_place.chunks_exact_mut(self.execution_length) {
//             for (c, twiddle_chunk) in self
//                 .twiddles
//                 .chunks_exact(TWIDDLES_PER_COLUMN * COMPLEX_PER_VECTOR)
//                 .take(chunk_count)
//                 .enumerate()
//             {
//                 let index_base = c * COMPLEX_PER_VECTOR;
//
//                 // Load columns from the input into registers
//                 let mut columns = [NeonStoreF::default(); ROW_COUNT];
//                 for i in 0..ROW_COUNT {
//                     unsafe {
//                         columns[i] = NeonStoreF::from_complex_ref(
//                             chunk.get_unchecked(index_base + len_per_row * i..),
//                         );
//                     }
//                 }
//
//                 let output = self.inner_bf.exec(columns);
//
//                 unsafe {
//                     output[0].write(scratch.get_unchecked_mut(index_base..));
//                 }
//
//                 for i in 1..ROW_COUNT {
//                     let twiddle = &twiddle_chunk[i * COMPLEX_PER_VECTOR - COMPLEX_PER_VECTOR..];
//                     let output = NeonStoreF::mul_by_complex(
//                         output[i],
//                         NeonStoreF::load(twiddle.as_ptr().cast()),
//                     );
//                     unsafe {
//                         output.write(scratch.get_unchecked_mut(index_base + len_per_row * i..))
//                     }
//                 }
//             }
//
//             let partial_remainder = len_per_row % COMPLEX_PER_VECTOR;
//             if partial_remainder > 0 {
//                 let partial_remainder_base = chunk_count * COMPLEX_PER_VECTOR;
//                 let partial_remainder_twiddle_base =
//                     self.twiddles.len() - TWIDDLES_PER_COLUMN * COMPLEX_PER_VECTOR;
//                 let final_twiddle_chunk = &self.twiddles[partial_remainder_twiddle_base..];
//
//                 let mut columns = [NeonStoreFh::default(); ROW_COUNT];
//                 for i in 0..ROW_COUNT {
//                     unsafe {
//                         columns[i] = NeonStoreFh::load(
//                             chunk
//                                 .get_unchecked(partial_remainder_base + len_per_row * i..)
//                                 .as_ptr()
//                                 .cast(),
//                         );
//                     }
//                 }
//
//                 // apply our butterfly function down the columns
//                 let output = self.inner_bf.exech(columns);
//
//                 // always write the first row without twiddles
//                 unsafe {
//                     output[0].write(scratch.get_unchecked_mut(partial_remainder_base..));
//                 }
//
//                 // for the remaining rows, apply twiddle factors and then write back to memory
//                 for i in 1..ROW_COUNT {
//                     let twiddle = final_twiddle_chunk[i - 1];
//                     let output =
//                         NeonStoreFh::mul_by_complex(output[i], NeonStoreFh::from_complex(&twiddle));
//                     unsafe {
//                         output.write(
//                             scratch.get_unchecked_mut(partial_remainder_base + len_per_row * i..),
//                         );
//                     }
//                 }
//             }
//
//             self.width_executor.execute(&mut scratch)?;
//
//             self.transpose_executor
//                 .transpose(&scratch, chunk, self.width, self.height);
//         }
//         Ok(())
//     }
//
//     fn direction(&self) -> FftDirection {
//         self.direction
//     }
//
//     #[inline]
//     fn length(&self) -> usize {
//         self.execution_length
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Zaft;

    #[test]
    fn test_neon_mixed_radix_f64() {
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
            NeonMixedRadix2::new(Zaft::strategy(4, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(8, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        bf8.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        neon_mixed_rust.execute(&mut test_value).unwrap();
        assert_eq!(reference_value, test_value);
    }

    #[test]
    fn test_neon_mixed_radix3_f64() {
        let src: [Complex<f64>; 12] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
        ];
        let neon_mixed_rust =
            NeonMixedRadix3::new(Zaft::strategy(4, FftDirection::Forward).unwrap()).unwrap();
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
    fn test_neon_mixed_radix4_f64() {
        let src: [Complex<f64>; 12] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
        ];
        let neon_mixed_rust =
            NeonMixedRadix4::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
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

    #[cfg(feature = "fcma")]
    #[test]
    fn test_neon_mixed_radix4_f64_fcma() {
        if std::arch::is_aarch64_feature_detected!("fcma") {
            let src: [Complex<f64>; 12] = [
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(-0.45, -0.4),
                Complex::new(0.45, -0.4),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(-0.45, -0.4),
                Complex::new(0.45, -0.4),
            ];
            let neon_mixed_rust =
                NeonMixedRadix4::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
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
    }

    #[test]
    fn test_neon_mixed_radix5_f64() {
        let src: [Complex<f64>; 20] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
        ];
        let neon_mixed_rust =
            NeonMixedRadix5::new(Zaft::strategy(4, FftDirection::Forward).unwrap()).unwrap();
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
    fn test_neon_mixed_radix6_f64() {
        let src: [Complex<f64>; 18] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            NeonMixedRadix6::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(18, FftDirection::Forward).unwrap();
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
    fn test_neon_mixed_radix7_f64() {
        let src: [Complex<f64>; 14] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
        ];
        let neon_mixed_rust =
            NeonMixedRadix7::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
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
    fn test_neon_mixed_radix9_f64() {
        let src: [Complex<f64>; 18] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            NeonMixedRadix9::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(18, FftDirection::Forward).unwrap();
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
    fn test_neon_mixed_radix10_f64() {
        let src: [Complex<f64>; 20] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            NeonMixedRadix10::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
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
    fn test_neon_mixed_radix11_f64() {
        let src: [Complex<f64>; 22] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
        ];
        let neon_mixed_rust =
            NeonMixedRadix11::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
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
    fn test_neon_mixed_radix13_f64() {
        let src: [Complex<f64>; 26] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
        ];
        let neon_mixed_rust =
            NeonMixedRadix13::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(26, FftDirection::Forward).unwrap();
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
    fn test_neon_mixed_radix16_f64() {
        let src: [Complex<f64>; 32] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            NeonMixedRadix16::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(32, FftDirection::Forward).unwrap();
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
    #[cfg(feature = "fcma")]
    fn test_neon_mixed_radix16_fcma_f64() {
        if std::arch::is_aarch64_feature_detected!("fcma") {
            let src: [Complex<f64>; 32] = [
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(-0.45, -0.4),
                Complex::new(0.45, -0.4),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(-0.45, -0.4),
                Complex::new(0.45, -0.4),
                Complex::new(0.45, -0.4),
                Complex::new(3.25, 2.7),
                Complex::new(-0.45, -0.4),
                Complex::new(0.45, -0.4),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(0.654, 0.324),
                Complex::new(3.25, 2.7),
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(-0.45, -0.4),
                Complex::new(0.45, -0.4),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(-0.45, -0.4),
                Complex::new(0.654, 0.324),
            ];
            let neon_mixed_rust =
                NeonFcmaMixedRadix16::new(Zaft::strategy(2, FftDirection::Forward).unwrap())
                    .unwrap();
            let bf8 = Zaft::strategy(32, FftDirection::Forward).unwrap();
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
    }

    #[test]
    fn test_neon_mixed_radix_f32() {
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
            NeonMixedRadix2f::new(Zaft::strategy(4, FftDirection::Forward).unwrap()).unwrap();
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
    fn test_neon_mixed_radix_rem_f32() {
        let src: [Complex<f32>; 6] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            NeonMixedRadix2f::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(6, FftDirection::Forward).unwrap();
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
    fn test_neon_mixed_radix3_rem_f32() {
        let src: [Complex<f32>; 12] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            NeonMixedRadix3f::new(Zaft::strategy(4, FftDirection::Forward).unwrap()).unwrap();
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
    fn test_neon_mixed_radix4_rem_f32() {
        let src: [Complex<f32>; 12] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            NeonMixedRadix4f::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
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

    #[cfg(feature = "fcma")]
    #[test]
    fn test_neon_mixed_radix4_rem_f32_fcma() {
        if std::arch::is_aarch64_feature_detected!("fcma") {
            let src: [Complex<f32>; 12] = [
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
            ];
            let neon_mixed_rust =
                NeonFcmaMixedRadix4f::new(Zaft::strategy(3, FftDirection::Forward).unwrap())
                    .unwrap();
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
    }

    #[test]
    fn test_neon_mixed_radix5_rem_f32() {
        let src: [Complex<f32>; 20] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            NeonMixedRadix5f::new(Zaft::strategy(4, FftDirection::Forward).unwrap()).unwrap();
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

    #[cfg(feature = "fcma")]
    #[test]
    fn test_neon_mixed_radix20_rem_f32() {
        if std::arch::is_aarch64_feature_detected!("fcma") {
            let src: [Complex<f32>; 20] = [
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
            ];
            let neon_mixed_rust =
                NeonFcmaMixedRadix5f::new(Zaft::strategy(4, FftDirection::Forward).unwrap())
                    .unwrap();
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
    }

    #[test]
    fn test_neon_mixed_radix6_rem_f32() {
        let src: [Complex<f32>; 18] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
        ];
        let neon_mixed_rust =
            NeonMixedRadix6f::new(Zaft::strategy(3, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(18, FftDirection::Forward).unwrap();
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
    fn test_neon_mixed_radix7_rem_f32() {
        let src: [Complex<f32>; 14] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            NeonMixedRadix7f::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
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
    fn test_neon_mixed_radix8_rem_f32() {
        let src: [Complex<f32>; 16] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
        ];
        let neon_mixed_rust =
            NeonMixedRadix8f::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(16, FftDirection::Forward).unwrap();
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
    #[cfg(feature = "fcma")]
    fn test_neon_mixed_radix8_fcma_f32() {
        if std::arch::is_aarch64_feature_detected!("fcma") {
            let src: [Complex<f32>; 16] = [
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
            ];
            let neon_mixed_rust =
                NeonMixedRadix8f::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
            let bf8 = Zaft::strategy(16, FftDirection::Forward).unwrap();
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

    #[test]
    fn test_neon_mixed_radix10_rem_f32() {
        let src: [Complex<f32>; 20] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let neon_mixed_rust =
            NeonMixedRadix10f::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
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
    fn test_neon_mixed_radix11_rem_f32() {
        let src: [Complex<f32>; 22] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
        ];
        let neon_mixed_rust =
            NeonMixedRadix11f::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
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
    fn test_neon_mixed_radix13_rem_f32() {
        let src: [Complex<f32>; 26] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
        ];
        let neon_mixed_rust =
            NeonMixedRadix13f::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(26, FftDirection::Forward).unwrap();
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
    fn test_neon_mixed_radix16_rem_f32() {
        let src: [Complex<f32>; 32] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
        ];
        let neon_mixed_rust =
            NeonMixedRadix16f::new(Zaft::strategy(2, FftDirection::Forward).unwrap()).unwrap();
        let bf8 = Zaft::strategy(32, FftDirection::Forward).unwrap();
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

    #[cfg(feature = "fcma")]
    #[test]
    fn test_neon_mixed_radix16_fcma_f32() {
        if std::arch::is_aarch64_feature_detected!("fcma") {
            let src: [Complex<f32>; 32] = [
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
                Complex::new(0.9, 0.13),
                Complex::new(0.9, 0.13),
                Complex::new(3.25, 2.7),
                Complex::new(0.654, 0.324),
                Complex::new(1.3, 1.6),
                Complex::new(1.7, -0.4),
                Complex::new(8.2, -0.1),
            ];
            let neon_mixed_rust =
                NeonFcmaMixedRadix16f::new(Zaft::strategy(2, FftDirection::Forward).unwrap())
                    .unwrap();
            let bf8 = Zaft::strategy(32, FftDirection::Forward).unwrap();
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
}
