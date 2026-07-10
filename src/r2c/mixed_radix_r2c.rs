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
use crate::err::try_vec;
use crate::spectrum_arithmetic::ComplexArith;
use crate::transpose::{TransposeExecutor, TransposeExecutorReal};
use crate::util::{compute_twiddle, validate_scratch};
use crate::{FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub(crate) struct MixedRadixR2cOdd<T> {
    execution_length: usize,
    twiddles: Vec<Complex<T>>,
    width_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
    width: usize,
    height_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
    height: usize,
    spectrum_ops: Arc<dyn ComplexArith<T> + Send + Sync>,
    width_transpose_real: Box<dyn TransposeExecutorReal<T> + Send + Sync>,
    height_transpose: Box<dyn TransposeExecutor<T> + Send + Sync>,
    width_transpose: Box<dyn TransposeExecutor<T> + Send + Sync>,
    second_stage_len: usize,
    width_scratch_length: usize,
    height_scratch_length: usize,
}

impl<T: FftSample> MixedRadixR2cOdd<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(
        width_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
        height_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Self, ZaftError> {
        let direction = width_executor.direction();

        let width = width_executor.length();

        let height = height_executor.length();

        let len = width.checked_mul(height).ok_or(ZaftError::Overflow)?;

        let first_stage_remove = (width * height - width) / 2;

        let first_stages = first_stage_remove / (len / height);
        let complex_height = height - first_stages;
        let twiddles_len = width * complex_height - complex_height;

        let mut twiddles = try_vec![Complex::zero(); twiddles_len];
        for (x, row) in twiddles.chunks_exact_mut(complex_height).enumerate() {
            let x = x + 1;
            for (y, dst) in row.iter_mut().enumerate() {
                *dst = compute_twiddle(x * y, len, direction);
            }
        }

        let to_remove_second_stage = (width - 1) / 2;

        let second_stage_len = complex_height * width;
        let width_scratch_length = width_executor.scratch_length();
        let height_scratch_length = height_executor.scratch_length();

        Ok(MixedRadixR2cOdd {
            execution_length: width * height,
            width_executor,
            width,
            height_executor,
            height,
            twiddles,
            spectrum_ops: T::make_complex_arith(),
            width_transpose_real: T::transpose_strategy_real(width, height),
            height_transpose: T::transpose_strategy(complex_height, width),
            width_transpose: T::transpose_strategy(width - to_remove_second_stage, complex_height),
            second_stage_len,
            width_scratch_length,
            height_scratch_length,
        })
    }
}

impl<T: FftSample> R2CFftExecutor<T> for MixedRadixR2cOdd<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.real_length(),
            output.len(),
            self.complex_length(),
        )?;
        let mut scratch = try_vec![Complex::zero(); self.complex_scratch_length()];
        self.execute_with_scratch(input, output, &mut scratch)
    }

    fn execute_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.real_length(),
            output.len(),
            self.complex_length(),
        )?;

        let complex_length = self.complex_length();

        let scratch = validate_scratch!(scratch, self.complex_scratch_length());
        let (scratch_initial, rem_scratch) = scratch.split_at_mut(self.execution_length);
        let (scratch_complex1, rem_scratch) = rem_scratch.split_at_mut(self.second_stage_len);

        let to_remove = (self.height - 1) / 2;
        let complex_height = self.height - to_remove;
        let to_remove_second_stage = (self.width - 1) / 2;
        let to_remove_second_stage_tail = if self.width.is_multiple_of(2) {
            ((self.width - 1) / 2) + 1
        } else {
            (self.width - 1) / 2
        };

        for (input, complex) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(complex_length))
        {
            self.width_transpose_real
                .transpose(input, scratch_initial, self.width, self.height);

            let (height_scratch, _) = rem_scratch.split_at_mut(self.height_scratch_length);
            self.height_executor
                .execute_with_scratch(scratch_initial, height_scratch)?;

            scratch_complex1[..complex_height].copy_from_slice(&scratch_initial[..complex_height]);

            self.spectrum_ops.mul_and_cut(
                &scratch_initial[self.height..],
                self.height,
                &self.twiddles,
                complex_height,
                &mut scratch_complex1[complex_height..],
            );

            let (scratch_complex0, _) = scratch_initial.split_at_mut(self.second_stage_len);

            self.height_transpose.transpose(
                scratch_complex1,
                scratch_complex0,
                complex_height,
                self.width,
            );

            let (width_scratch, _) = rem_scratch.split_at_mut(self.width_scratch_length);
            self.width_executor
                .execute_with_scratch(scratch_complex0, width_scratch)?;

            // // first stage with removed redundancy
            // for x in 0..(self.width - to_remove_second_stage) {
            //     let max_y = if x == nyquist_x { 1 } else { complex_height };
            //     for y in 0..max_y {
            //         let input_index = x + y * self.width;
            //         let output_index = y + x * self.height;
            //
            //         unsafe {
            //                *complex.get_unchecked_mut(output_index) =
            //                    *scratch_complex0.get_unchecked(input_index);
            //         }
            //     }
            // }

            // Split into three regions:
            // 1. Regular columns x=0..nyquist_x
            // 2. Nyquist column x=nyquist_x      → only y=0, scalar copy (even width only)
            // 3. Conjugate tail

            let nyquist_x = if self.width.is_multiple_of(2) {
                Some(self.width / 2)
            } else {
                None
            };
            let regular_cols = nyquist_x.unwrap_or(self.width - to_remove_second_stage);
            self.width_transpose.transpose_strided(
                scratch_complex0,
                self.width,
                complex,
                self.height,
                regular_cols, // only regular columns, uniform complex_height rows each
                complex_height,
            );

            if let Some(nx) = nyquist_x {
                let input_index = nx; // x=nyquist_x, y=0
                let output_index = nx * self.height; // y=0 + nyquist_x * height
                unsafe {
                    *complex.get_unchecked_mut(output_index) =
                        *scratch_complex0.get_unchecked(input_index);
                }
            }

            // conjugated tail
            for x in (self.width - to_remove_second_stage_tail)..self.width {
                for y in 1..complex_height {
                    let input_index = x + y * self.width;
                    let output_index = self.execution_length - (y + x * self.height);

                    unsafe {
                        *complex.get_unchecked_mut(output_index) =
                            scratch_complex0.get_unchecked(input_index).conj();
                    }
                }
            }
        }
        Ok(())
    }

    fn real_length(&self) -> usize {
        self.execution_length
    }

    fn complex_length(&self) -> usize {
        self.execution_length / 2 + 1
    }

    fn complex_scratch_length(&self) -> usize {
        self.execution_length
            + self.second_stage_len
            + self.width_scratch_length.max(self.height_scratch_length)
    }
}

#[cfg(test)]
mod tests {
    use crate::dft::Dft;
    use crate::r2c::mixed_radix_r2c::MixedRadixR2cOdd;
    use crate::{FftDirection, FftExecutor, R2CFftExecutor, Zaft};
    use num_complex::Complex;
    use num_traits::Zero;

    #[test]
    fn test_mixed_radixd() {
        let src: [f64; 40] = [
            7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.4, 3.4, 2.1, 3.2, 3.3, 9.8, 5.2, 2.1, 3.2,
            3.3, 9.8, 5.2, 2.1, 3.2, 3.3, 9.8, 5.2, 2.1, 3.2, 3.3, 9.8, 5.2, 2.1, 3.2, 3.3, 9.8,
            5.2, 2.1, 3.2, 3.3, 9.8, 5.2,
        ];
        let mx = MixedRadixR2cOdd::new(
            Zaft::strategy(5, FftDirection::Forward).unwrap(),
            Zaft::strategy(8, FftDirection::Forward).unwrap(),
        )
        .unwrap();
        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(40, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        println!("DFT -----");

        for chunk in (&reference_value[..10]).as_chunks::<5>().0.iter() {
            println!("{:?}", chunk);
        }

        let test_value = src.to_vec();
        let mut complex_output = vec![Complex::zero(); 40 / 2 + 1];
        mx.execute(&test_value, &mut complex_output).unwrap();
        reference_value
            .iter()
            .zip(complex_output.iter())
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
    fn test_mixed_radixf() {
        let src: [f32; 40] = [
            7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.4, 3.4, 2.1, 3.2, 3.3, 9.8, 5.2, 2.1, 3.2,
            3.3, 9.8, 5.2, 2.1, 3.2, 3.3, 9.8, 5.2, 2.1, 3.2, 3.3, 9.8, 5.2, 2.1, 3.2, 3.3, 9.8,
            5.2, 2.1, 3.2, 3.3, 9.8, 5.2,
        ];
        let mx = MixedRadixR2cOdd::new(
            Zaft::strategy(5, FftDirection::Forward).unwrap(),
            Zaft::strategy(8, FftDirection::Forward).unwrap(),
        )
        .unwrap();
        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(40, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        println!("DFT -----");

        for chunk in (&reference_value[..10]).as_chunks::<5>().0.iter() {
            println!("{:?}", chunk);
        }

        let test_value = src.to_vec();
        let mut complex_output = vec![Complex::zero(); 40 / 2 + 1];
        mx.execute(&test_value, &mut complex_output).unwrap();
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
    fn test_mixed_radixf_8f() {
        let src: [f32; 16] = [
            7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4,
        ];
        let mx = MixedRadixR2cOdd::new(
            Zaft::strategy(4, FftDirection::Forward).unwrap(),
            Zaft::strategy(4, FftDirection::Forward).unwrap(),
        )
        .unwrap();
        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(16, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        println!("DFT -----");

        for chunk in (&reference_value[..8]).as_chunks::<4>().0.iter() {
            println!("{:?}", chunk);
        }

        let test_value = src.to_vec();
        let mut complex_output = vec![Complex::zero(); 16 / 2 + 1];
        mx.execute(&test_value, &mut complex_output).unwrap();
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
    fn test_mixed_radixf_22f() {
        let src: [f32; 22] = [
            7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 1.3,
            5.6, 2.6, 6.4, 6.1, 5.1,
        ];
        let mx = MixedRadixR2cOdd::new(
            Zaft::strategy(2, FftDirection::Forward).unwrap(),
            Zaft::strategy(11, FftDirection::Forward).unwrap(),
        )
        .unwrap();
        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(22, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        println!("DFT -----");

        for chunk in (&reference_value[..11]).as_chunks::<2>().0.iter() {
            println!("{:?}", chunk);
        }

        let test_value = src.to_vec();
        let mut complex_output = vec![Complex::zero(); 22 / 2 + 1];
        mx.execute(&test_value, &mut complex_output).unwrap();
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
    fn test_mixed_radixf_24f() {
        let src: [f32; 24] = [
            7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 1.3,
            5.6, 2.6, 6.4, 6.1, 5.1, 7.9, 1.3,
        ];
        let mx = MixedRadixR2cOdd::new(
            Zaft::strategy(3, FftDirection::Forward).unwrap(),
            Zaft::strategy(8, FftDirection::Forward).unwrap(),
        )
        .unwrap();
        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(24, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        println!("DFT -----");

        for chunk in (&reference_value[..8]).as_chunks::<3>().0.iter() {
            println!("{:?}", chunk);
        }

        let test_value = src.to_vec();
        let mut complex_output = vec![Complex::zero(); 24 / 2 + 1];
        mx.execute(&test_value, &mut complex_output).unwrap();
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
