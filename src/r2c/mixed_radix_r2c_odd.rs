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
use crate::util::compute_twiddle;
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

        assert!(
            !width.is_multiple_of(2),
            "This is an UB to call Odd Mixed-Radix R2C with even `width`"
        );

        let height = height_executor.length();

        let len = width * height;

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
        })
    }
}

impl<T: FftSample> R2CFftExecutor<T> for MixedRadixR2cOdd<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !input.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                input.len(),
                self.execution_length,
            ));
        }
        if !output.len().is_multiple_of(self.complex_length()) {
            return Err(ZaftError::InvalidSizeMultiplier(
                input.len(),
                self.complex_length(),
            ));
        }
        if input.len() / self.execution_length != output.len() / self.complex_length() {
            return Err(ZaftError::InvalidSamplesCount(
                input.len() / self.execution_length,
                output.len() / self.complex_length(),
            ));
        }

        let complex_length = self.complex_length();

        let first_stage_remove = (self.width * self.height - self.width) / 2;

        let first_stages = first_stage_remove / (self.execution_length / self.height);
        let complex_height = self.height - first_stages;

        let second_stage_len = complex_height * self.width;

        let mut scratch_initial = try_vec![Complex::<T>::zero(); self.execution_length];
        let mut scratch_complex2 = try_vec![Complex::<T>::zero(); second_stage_len * 2];
        let (scratch_complex0, scratch_complex1) = scratch_complex2.split_at_mut(second_stage_len);

        let to_remove = (self.height - 1) / 2;
        let complex_height = self.height - to_remove;
        let to_remove_second_stage = (self.width - 1) / 2;

        for (input, complex) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(complex_length))
        {
            // STEP 1: transpose
            self.width_transpose_real.transpose(
                input,
                &mut scratch_initial,
                self.width,
                self.height,
            );

            // STEP 2: perform FFTs of size `height`
            self.height_executor.execute(&mut scratch_initial)?;

            // STEP 3: Apply twiddle factors
            for (dst, &src) in scratch_complex1[..complex_height]
                .iter_mut()
                .zip(scratch_initial[..complex_height].iter())
            {
                *dst = src;
            }

            self.spectrum_ops.mul_and_cut(
                &scratch_initial[self.height..],
                self.height,
                &self.twiddles,
                complex_height,
                &mut scratch_complex1[complex_height..],
            );

            // STEP 4: transpose again
            self.height_transpose.transpose(
                scratch_complex1,
                scratch_complex0,
                complex_height,
                self.width,
            );

            // STEP 5: perform FFTs of size `width`
            self.width_executor.execute(scratch_complex0)?;

            // first stage with removed redundancy
            // for x in 0..(self.width - to_remove_second_stage) {
            //     for y in 0..complex_height {
            //         let input_index = x + y * self.width;
            //         let output_index = y + x * self.height;
            //
            //         unsafe {
            //             *complex.get_unchecked_mut(output_index) =
            //                 *scratch_complex0.get_unchecked(input_index);
            //         }
            //     }
            // }
            self.width_transpose.transpose_strided(
                scratch_complex0,
                self.width,
                complex,
                self.height,
                self.width - to_remove_second_stage,
                complex_height,
            );

            // conjugated tail
            for x in (self.width - to_remove_second_stage)..self.width {
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
}

#[cfg(test)]
mod tests {
    use crate::dft::Dft;
    use crate::r2c::mixed_radix_r2c_odd::MixedRadixR2cOdd;
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

        for chunk in (&reference_value[..10]).chunks_exact(5) {
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

        for chunk in (&reference_value[..10]).chunks_exact(5) {
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
}
