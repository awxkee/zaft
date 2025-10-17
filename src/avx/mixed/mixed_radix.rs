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
use crate::avx::mixed::avx_store::AvxStoreD;
use crate::avx::mixed::butterflies::{
    ColumnButterfly2d, ColumnButterfly3d, ColumnButterfly4d, ColumnButterfly5d,
};
use crate::err::try_vec;
use crate::transpose::{TransposeExecutor, TransposeFactory};
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::Zero;

macro_rules! define_mixed_radixd {
    ($mx_type: ident, $bf_type: ident, $row_count: expr) => {
        pub(crate) struct $mx_type {
            execution_length: usize,
            direction: FftDirection,
            twiddles: Vec<Complex<f64>>,
            width_executor: Box<dyn FftExecutor<f64> + Send + Sync>,
            width: usize,
            height: usize,
            transpose_executor: Box<dyn TransposeExecutor<f64> + Send + Sync>,
            inner_bf: $bf_type,
        }

        impl $mx_type {
            pub fn new(width_executor: Box<dyn FftExecutor<f64> + Send + Sync>) -> Result<Self, ZaftError> {
                let direction = width_executor.direction();

                let width = width_executor.length();

                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;

                // derive some info from our inner FFT
                let len_per_row = width_executor.length();

                let len = len_per_row * ROW_COUNT;
                const COMPLEX_PER_VECTOR: usize = 2;

                // We're going to process each row of the FFT one AVX register at a time. We need to know how many AVX registers each row can fit,
                // and if the last register in each row going to have partial data (ie a remainder)
                let quotient = len_per_row / COMPLEX_PER_VECTOR;
                let remainder = len_per_row % COMPLEX_PER_VECTOR;

                let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
                let mut twiddles = Vec::with_capacity(num_twiddle_columns * TWIDDLES_PER_COLUMN);
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
                })
            }
        }

    impl $mx_type {
        #[target_feature(enable = "avx2", enable = "fma")]
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
                        output[0].write(scratch.get_unchecked_mut(index_base..));
                    }

                    for i in 1..ROW_COUNT {
                        let twiddle = &twiddle_chunk[i * COMPLEX_PER_VECTOR - COMPLEX_PER_VECTOR..];
                        let output =
                            AvxStoreD::mul_by_complex(output[i], AvxStoreD::from_complex_ref(twiddle));
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
                        output[0].write(scratch.get_unchecked_mut(partial_remainder_base..));
                    }

                    // for the remaining rows, apply twiddle factors and then write back to memory
                    for i in 1..ROW_COUNT {
                        let twiddle = final_twiddle_chunk[i*COMPLEX_PER_VECTOR - COMPLEX_PER_VECTOR];
                        let output =
                            AvxStoreD::mul_by_complex(output[i], AvxStoreD::from_complex(&twiddle));
                        unsafe {
                            output.write_lo(
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

    impl FftExecutor<f64> for $mx_type {
        fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
            unsafe { self.execute_f64(in_place) }
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

define_mixed_radixd!(AvxMixedRadix2d, ColumnButterfly2d, 2);
define_mixed_radixd!(AvxMixedRadix3d, ColumnButterfly3d, 3);
define_mixed_radixd!(AvxMixedRadix4d, ColumnButterfly4d, 4);
define_mixed_radixd!(AvxMixedRadix5d, ColumnButterfly5d, 5);

pub(crate) struct AvxMixedRadix {
    execution_length: usize,
    direction: FftDirection,
    twiddles: Vec<Complex<f64>>,
    width_executor: Box<dyn FftExecutor<f64> + Send + Sync>,
    width: usize,
    height: usize,
    transpose_executor: Box<dyn TransposeExecutor<f64> + Send + Sync>,
    inner_bf: ColumnButterfly2d,
}

impl AvxMixedRadix {
    pub fn new(width_executor: Box<dyn FftExecutor<f64> + Send + Sync>) -> Result<Self, ZaftError> {
        let direction = width_executor.direction();

        let width = width_executor.length();

        const ROW_COUNT: usize = 2;
        const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;

        // derive some info from our inner FFT
        let len_per_row = width_executor.length();

        let len = len_per_row * ROW_COUNT;
        const COMPLEX_PER_VECTOR: usize = 2;

        // We're going to process each row of the FFT one AVX register at a time. We need to know how many AVX registers each row can fit,
        // and if the last register in each row going to have partial data (ie a remainder)
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        let mut twiddles = Vec::with_capacity(num_twiddle_columns * TWIDDLES_PER_COLUMN);
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

        Ok(AvxMixedRadix {
            execution_length: width * ROW_COUNT,
            width_executor,
            width,
            height: ROW_COUNT,
            direction,
            twiddles,
            transpose_executor: f64::transpose_strategy(width, ROW_COUNT),
            inner_bf: ColumnButterfly2d::new(direction),
        })
    }
}

impl AvxMixedRadix {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let mut scratch = try_vec![Complex::zero(); self.execution_length];

        const ROW_COUNT: usize = 2;
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
                let mut columns = [AvxStoreD::zero(); ROW_COUNT];
                for i in 0..ROW_COUNT {
                    unsafe {
                        columns[i] = AvxStoreD::from_complex_ref(
                            chunk.get_unchecked(index_base + len_per_row * i..),
                        );
                    }
                }

                let output = unsafe { self.inner_bf.exec(columns) };

                unsafe {
                    output[0].write(scratch.get_unchecked_mut(index_base..));
                }

                for i in 1..ROW_COUNT {
                    let twiddle = &twiddle_chunk[i * COMPLEX_PER_VECTOR - COMPLEX_PER_VECTOR..];
                    let output =
                        AvxStoreD::mul_by_complex(output[i], AvxStoreD::from_complex_ref(twiddle));
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

                let mut columns = [AvxStoreD::zero(); ROW_COUNT];
                for i in 0..ROW_COUNT {
                    unsafe {
                        columns[i] = AvxStoreD::from_complex(
                            chunk.get_unchecked(partial_remainder_base + len_per_row * i),
                        );
                    }
                }

                // apply our butterfly function down the columns
                let output = unsafe { self.inner_bf.exec(columns) };

                // always write the first row without twiddles
                unsafe {
                    output[0].write(scratch.get_unchecked_mut(partial_remainder_base..));
                }

                // for the remaining rows, apply twiddle factors and then write back to memory
                for i in 1..ROW_COUNT {
                    let twiddle = final_twiddle_chunk[i - 1];
                    let output =
                        AvxStoreD::mul_by_complex(output[i], AvxStoreD::from_complex(&twiddle));
                    unsafe {
                        output.write_lo(
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

impl FftExecutor<f64> for AvxMixedRadix {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }
}

/*impl FftExecutor<f32> for AvxMixedRadix<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let mut scratch = try_vec![Complex::zero(); self.execution_length];

        const ROW_COUNT: usize = 2;
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

                let output = self.inner_bf.exec(columns);

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
                let output = self.inner_bf.exech(columns);

                // always write the first row without twiddles
                unsafe {
                    output[0].write(scratch.get_unchecked_mut(partial_remainder_base..));
                }

                // for the remaining rows, apply twiddle factors and then write back to memory
                for i in 1..ROW_COUNT {
                    let twiddle = final_twiddle_chunk[i - 1];
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
}*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Zaft;

    #[test]
    fn test_avx_mixed_radix_f64() {
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
}
