/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //2028
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
use crate::fast_divider::DividerU64;
use crate::prime_factors::{PrimeFactors, primitive_root};
use crate::spectrum_arithmetic::ComplexArith;
use crate::util::{compute_twiddle, validate_oof_sizes, validate_scratch};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_integer::Integer;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub(crate) struct RadersFft<T> {
    convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    convolve_fft_twiddles: Vec<Complex<T>>,
    execution_length: usize,
    direction: FftDirection,
    input_indices: Vec<usize>,
    output_indices: Vec<usize>,
    spectrum_ops: Arc<dyn ComplexArith<T> + Send + Sync>,
    convolve_scratch_length: usize,
}

impl<T: FftSample> RadersFft<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        size: usize,
        convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        fft_direction: FftDirection,
    ) -> Result<RadersFft<T>, ZaftError> {
        assert!(
            PrimeFactors::from_number(size as u64).is_prime(),
            "Input length for Rader's must be a prime number"
        );

        let direction = convolve_fft.direction();
        let convolve_fft_len = convolve_fft.length();
        assert_eq!(fft_direction, direction);
        let dividing_len = DividerU64::new(size as u64);

        // compute the primitive root and its inverse for this size
        let primitive_root = primitive_root(size as u64).unwrap();

        // compute the multiplicative inverse of primative_root mod len and vice versa.
        // i64::extended_gcd will compute both the inverse of left mod right, and the inverse of right mod left, but we're only goingto use one of them
        // the primitive root inverse might be negative, if o make it positive by wrapping
        let gcd_data = i64::extended_gcd(&(primitive_root as i64), &(size as i64));
        let primitive_root_inverse = if gcd_data.x >= 0 {
            gcd_data.x
        } else {
            gcd_data.x + size as i64
        } as u64;

        // precompute the coefficients to use inside the process method
        let inner_fft_scale: T = (1f64 / convolve_fft_len as f64).as_();
        let mut inner_fft_input = try_vec![Complex::zero(); convolve_fft_len];
        let mut twiddle_input = 1;
        for dst in &mut inner_fft_input {
            let twiddle = compute_twiddle(twiddle_input, size, direction);
            *dst = twiddle * inner_fft_scale;

            twiddle_input =
                ((twiddle_input as u64 * primitive_root_inverse) % dividing_len) as usize;
        }

        convolve_fft.execute(&mut inner_fft_input)?;

        let mut input_index = 1;
        let mut input_indices = try_vec![0usize; size - 1];
        for indexer in input_indices.iter_mut() {
            input_index = ((input_index as u64 * primitive_root) % dividing_len) as usize;

            *indexer = input_index - 1;
        }

        let mut output_index = 1;
        let mut output_indices = try_vec![0usize; size - 1];
        for indexer in output_indices.iter_mut() {
            output_index = ((output_index as u64 * primitive_root_inverse) % dividing_len) as usize;
            *indexer = output_index - 1;
        }

        let convolve_scratch_length = convolve_fft.scratch_length();

        Ok(RadersFft {
            execution_length: size,
            convolve_fft,
            convolve_fft_twiddles: inner_fft_input,
            direction: fft_direction,
            input_indices,
            output_indices,
            spectrum_ops: T::make_complex_arith(),
            convolve_scratch_length,
        })
    }
}

impl<T: FftSample> FftExecutor<T> for RadersFft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        let mut scratch = try_vec![Complex::zero(); self.scratch_length()];
        self.execute_with_scratch(in_place, &mut scratch)
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let scratch = validate_scratch!(scratch, self.scratch_length());
        let (scratch, convolve_scratch) = scratch.split_at_mut(self.execution_length);

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // The first output element is just the sum of all the input elements, and we need to store off the first input value
            let (buffer_first, buffer) = chunk.split_first_mut().unwrap();
            let buffer_first_val = *buffer_first;

            let (scratch, _) = scratch.split_at_mut(self.length() - 1);

            // copy the buffer into the scratch, reordering as we go. also compute a sum of all elements
            for (scratch_element, &buffer_idx) in scratch.iter_mut().zip(self.input_indices.iter())
            {
                *scratch_element = unsafe { *buffer.get_unchecked(buffer_idx) };
            }

            // perform the first of two inner FFTs

            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            // scratch[0] now contains the sum of elements 1..len. We need the sum of all elements, so all we have to do is add the first input
            *buffer_first = *buffer_first + scratch[0];

            // multiply the inner result with our cached setup data
            // also conjugate every entry. this sets us up to do an inverse FFT
            // (because an inverse FFT is equivalent to a normal FFT where you conjugate both the inputs and outputs)
            self.spectrum_ops
                .mul_conjugate_in_place(scratch, &self.convolve_fft_twiddles);

            // We need to add the first input value to all output values. We can accomplish this by adding it to the DC input of our inner ifft.
            // Of course, we have to conjugate it, just like we conjugated the complex multiplied above
            scratch[0] = scratch[0] + buffer_first_val.conj();

            // execute the second FFT
            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            // copy the final values into the output, reordering as we go
            for (scratch_element, &buffer_idx) in scratch.iter().zip(self.output_indices.iter()) {
                unsafe {
                    *buffer.get_unchecked_mut(buffer_idx) = scratch_element.conj();
                }
            }
        }
        Ok(())
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        let mut scratch = try_vec![Complex::zero(); self.out_of_place_scratch_length()];
        self.execute_out_of_place_with_scratch(src, dst, &mut scratch)
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        validate_oof_sizes!(src, dst, self.execution_length);

        let scratch = validate_scratch!(scratch, self.out_of_place_scratch_length());
        let (scratch, convolve_scratch) = scratch.split_at_mut(self.execution_length);

        for (chunk, output_chunk) in src
            .chunks_exact(self.execution_length)
            .zip(dst.chunks_exact_mut(self.execution_length))
        {
            // The first output element is just the sum of all the input elements, and we need to store off the first input value
            let (buffer_first, buffer) = chunk.split_first().unwrap();
            let buffer_first_val = *buffer_first;

            let (scratch, _) = scratch.split_at_mut(self.length() - 1);

            // copy the buffer into the scratch, reordering as we go. also compute a sum of all elements
            for (scratch_element, &buffer_idx) in scratch.iter_mut().zip(self.input_indices.iter())
            {
                *scratch_element = unsafe { *buffer.get_unchecked(buffer_idx) };
            }

            // perform the first of two inner FFTs

            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            // scratch[0] now contains the sum of elements 1..len. We need the sum of all elements, so all we have to do is add the first input
            unsafe {
                *output_chunk.get_unchecked_mut(0) = *buffer_first + scratch[0];
            }

            // multiply the inner result with our cached setup data
            // also conjugate every entry. this sets us up to do an inverse FFT
            // (because an inverse FFT is equivalent to a normal FFT where you conjugate both the inputs and outputs)
            self.spectrum_ops
                .mul_conjugate_in_place(scratch, &self.convolve_fft_twiddles);

            // We need to add the first input value to all output values. We can accomplish this by adding it to the DC input of our inner ifft.
            // Of course, we have to conjugate it, just like we conjugated the complex multiplied above
            scratch[0] = scratch[0] + buffer_first_val.conj();

            // execute the second FFT
            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            let (_, buffer) = output_chunk.split_first_mut().unwrap();

            // copy the final values into the output, reordering as we go
            for (scratch_element, &buffer_idx) in scratch.iter().zip(self.output_indices.iter()) {
                unsafe {
                    *buffer.get_unchecked_mut(buffer_idx) = scratch_element.conj();
                }
            }
        }
        Ok(())
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_with_scratch(src, dst, scratch)
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
        self.execution_length + self.convolve_scratch_length
    }

    #[inline]
    fn out_of_place_scratch_length(&self) -> usize {
        self.execution_length + self.convolve_scratch_length
    }

    #[inline]
    fn destructive_scratch_length(&self) -> usize {
        self.execution_length + self.convolve_scratch_length
    }
}
