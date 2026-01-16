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
use crate::err::try_vec;
use crate::fast_divider::DividerU64;
use crate::neon::util::{conj_f64, conjq_f32};
use crate::prime_factors::{PrimeFactors, primitive_root};
use crate::spectrum_arithmetic::ComplexArith;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_integer::Integer;
use num_traits::{AsPrimitive, Zero};
use std::arch::aarch64::*;
use std::sync::Arc;

pub(crate) struct NeonRadersFft<T> {
    convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    convolve_fft_twiddles: Vec<Complex<T>>,
    execution_length: usize,
    direction: FftDirection,
    input_indices: Vec<u32>,
    output_indices: Vec<u32>,
    spectrum_ops: Arc<dyn ComplexArith<T> + Send + Sync>,
}

pub(crate) trait RadersIndicer<T> {
    fn index_inputs(buffer: &[Complex<T>], output: &mut [Complex<T>], indices: &[u32]);
    fn output_indices(buffer: &mut [Complex<T>], scratch: &[Complex<T>], indices: &[u32]);
}

impl RadersIndicer<f32> for f32 {
    fn index_inputs(buffer: &[Complex<f32>], output: &mut [Complex<f32>], indices: &[u32]) {
        unsafe {
            for (scratch_element, buffer_idx) in
                output.chunks_exact_mut(6).zip(indices.chunks_exact(6))
            {
                let idx0 = buffer_idx[0] as usize;
                let idx1 = buffer_idx[1] as usize;

                let v0 = vld1_f32(buffer.get_unchecked(idx0..).as_ptr().cast());
                let v1 = vld1_f32(buffer.get_unchecked(idx1..).as_ptr().cast());

                let idx2 = buffer_idx[2] as usize;
                let idx3 = buffer_idx[3] as usize;

                let v2 = vld1_f32(buffer.get_unchecked(idx2..).as_ptr().cast());
                let v3 = vld1_f32(buffer.get_unchecked(idx3..).as_ptr().cast());

                let idx4 = buffer_idx[4] as usize;
                let idx5 = buffer_idx[5] as usize;

                let v4 = vld1_f32(buffer.get_unchecked(idx4..).as_ptr().cast());
                let v5 = vld1_f32(buffer.get_unchecked(idx5..).as_ptr().cast());

                vst1_f32(scratch_element.as_mut_ptr().cast(), v0);
                vst1_f32(
                    scratch_element.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    v1,
                );
                vst1_f32(
                    scratch_element.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    v2,
                );
                vst1_f32(
                    scratch_element.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    v3,
                );
                vst1_f32(
                    scratch_element.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    v4,
                );
                vst1_f32(
                    scratch_element.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    v5,
                );
            }

            let rem = output.chunks_exact_mut(6).into_remainder();
            let rem_indices = indices.chunks_exact(6).remainder();

            for (scratch_element, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                let v0 = vld1_f32(buffer.get_unchecked(buffer_idx as usize..).as_ptr().cast());
                vst1_f32(scratch_element as *mut Complex<f32> as *mut f32, v0);
            }
        }
    }

    fn output_indices(buffer: &mut [Complex<f32>], scratch: &[Complex<f32>], indices: &[u32]) {
        unsafe {
            static CONJ: [f32; 4] = [0.0, -0.0, 0.0, -0.0];
            let conj = vld1q_f32(CONJ.as_ptr());
            for (src, buffer_idx) in scratch.chunks_exact(6).zip(indices.chunks_exact(6)) {
                let mut v0 = vld1q_f32(src.as_ptr().cast());
                let mut v1 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let mut v2 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());

                let idx0 = buffer_idx[0] as usize;
                let idx1 = buffer_idx[1] as usize;
                let idx2 = buffer_idx[2] as usize;

                v0 = conjq_f32(v0, conj);
                v1 = conjq_f32(v1, conj);
                v2 = conjq_f32(v2, conj);

                let idx3 = buffer_idx[3] as usize;
                let idx4 = buffer_idx[4] as usize;
                let idx5 = buffer_idx[5] as usize;

                vst1_f32(
                    buffer.get_unchecked_mut(idx0..).as_mut_ptr().cast(),
                    vget_low_f32(v0),
                );
                vst1_f32(
                    buffer.get_unchecked_mut(idx1..).as_mut_ptr().cast(),
                    vget_high_f32(v0),
                );
                vst1_f32(
                    buffer.get_unchecked_mut(idx2..).as_mut_ptr().cast(),
                    vget_low_f32(v1),
                );
                vst1_f32(
                    buffer.get_unchecked_mut(idx3..).as_mut_ptr().cast(),
                    vget_high_f32(v1),
                );
                vst1_f32(
                    buffer.get_unchecked_mut(idx4..).as_mut_ptr().cast(),
                    vget_low_f32(v2),
                );
                vst1_f32(
                    buffer.get_unchecked_mut(idx5..).as_mut_ptr().cast(),
                    vget_high_f32(v2),
                );
            }

            let rem_scratch = scratch.chunks_exact(6).remainder();
            let rem_indices = indices.chunks_exact(6).remainder();

            for (scratch_element, &buffer_idx) in rem_scratch.iter().zip(rem_indices.iter()) {
                *buffer.get_unchecked_mut(buffer_idx as usize) = scratch_element.conj();
            }
        }
    }
}

impl RadersIndicer<f64> for f64 {
    fn index_inputs(buffer: &[Complex<f64>], output: &mut [Complex<f64>], indices: &[u32]) {
        unsafe {
            for (scratch_element, buffer_idx) in
                output.chunks_exact_mut(6).zip(indices.chunks_exact(6))
            {
                let idx0 = buffer_idx[0] as usize;
                let idx1 = buffer_idx[1] as usize;

                let v0 = vld1q_f64(buffer.get_unchecked(idx0..).as_ptr().cast());
                let v1 = vld1q_f64(buffer.get_unchecked(idx1..).as_ptr().cast());

                let idx2 = buffer_idx[2] as usize;
                let idx3 = buffer_idx[3] as usize;

                let v2 = vld1q_f64(buffer.get_unchecked(idx2..).as_ptr().cast());
                let v3 = vld1q_f64(buffer.get_unchecked(idx3..).as_ptr().cast());

                let idx4 = buffer_idx[4] as usize;
                let idx5 = buffer_idx[5] as usize;

                let v4 = vld1q_f64(buffer.get_unchecked(idx4..).as_ptr().cast());
                let v5 = vld1q_f64(buffer.get_unchecked(idx5..).as_ptr().cast());

                vst1q_f64(scratch_element.as_mut_ptr().cast(), v0);
                vst1q_f64(
                    scratch_element.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    v1,
                );
                vst1q_f64(
                    scratch_element.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    v2,
                );
                vst1q_f64(
                    scratch_element.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    v3,
                );
                vst1q_f64(
                    scratch_element.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    v4,
                );
                vst1q_f64(
                    scratch_element.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    v5,
                );
            }

            let rem = output.chunks_exact_mut(6).into_remainder();
            let rem_indices = indices.chunks_exact(6).remainder();

            for (scratch_element, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                let v0 = vld1q_f64(buffer.get_unchecked(buffer_idx as usize..).as_ptr().cast());
                vst1q_f64(scratch_element as *mut Complex<f64> as *mut f64, v0);
            }
        }
    }

    fn output_indices(buffer: &mut [Complex<f64>], scratch: &[Complex<f64>], indices: &[u32]) {
        unsafe {
            static CONJ: [f64; 2] = [0.0, -0.0];
            let conj = vld1q_f64(CONJ.as_ptr());
            for (src, buffer_idx) in scratch.chunks_exact(6).zip(indices.chunks_exact(6)) {
                let mut v0 = vld1q_f64(src.as_ptr().cast());
                let mut v1 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());

                let idx0 = buffer_idx[0];
                let idx1 = buffer_idx[1];

                v0 = conj_f64(v0, conj);
                v1 = conj_f64(v1, conj);

                let mut v2 = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());
                let mut v3 = vld1q_f64(src.get_unchecked(3..).as_ptr().cast());

                let mut v4 = vld1q_f64(src.get_unchecked(4..).as_ptr().cast());
                let mut v5 = vld1q_f64(src.get_unchecked(5..).as_ptr().cast());

                let idx2 = buffer_idx[2];
                let idx3 = buffer_idx[3];

                v2 = conj_f64(v2, conj);
                v3 = conj_f64(v3, conj);

                v4 = conj_f64(v4, conj);
                v5 = conj_f64(v5, conj);

                let idx4 = buffer_idx[4];
                let idx5 = buffer_idx[5];

                vst1q_f64(
                    buffer
                        .get_unchecked_mut(idx0 as usize..)
                        .as_mut_ptr()
                        .cast(),
                    v0,
                );
                vst1q_f64(
                    buffer
                        .get_unchecked_mut(idx1 as usize..)
                        .as_mut_ptr()
                        .cast(),
                    v1,
                );
                vst1q_f64(
                    buffer
                        .get_unchecked_mut(idx2 as usize..)
                        .as_mut_ptr()
                        .cast(),
                    v2,
                );
                vst1q_f64(
                    buffer
                        .get_unchecked_mut(idx3 as usize..)
                        .as_mut_ptr()
                        .cast(),
                    v3,
                );
                vst1q_f64(
                    buffer
                        .get_unchecked_mut(idx4 as usize..)
                        .as_mut_ptr()
                        .cast(),
                    v4,
                );
                vst1q_f64(
                    buffer
                        .get_unchecked_mut(idx5 as usize..)
                        .as_mut_ptr()
                        .cast(),
                    v5,
                );
            }

            let rem_scratch = scratch.chunks_exact(6).remainder();
            let rem_indices = indices.chunks_exact(6).remainder();

            for (scratch_element, &buffer_idx) in rem_scratch.iter().zip(rem_indices.iter()) {
                *buffer.get_unchecked_mut(buffer_idx as usize) = scratch_element.conj();
            }
        }
    }
}

impl<T: FftSample> NeonRadersFft<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        size: usize,
        convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        fft_direction: FftDirection,
    ) -> Result<NeonRadersFft<T>, ZaftError> {
        assert!(
            PrimeFactors::from_number(size as u64).is_prime(),
            "Input length for Rader's must be a prime number"
        );

        let direction = convolve_fft.direction();
        let convolve_fft_len = convolve_fft.length();
        assert_eq!(fft_direction, direction);
        let reduced_len = DividerU64::new(size as u64);

        // compute the primitive root and its inverse for this size
        let primitive_root = primitive_root(size as u64).unwrap();

        // compute the multiplicative inverse of primative_root mod len and vice versa.
        // i64::extended_gcd will compute both the inverse of left mod right, and the inverse of right mod left, but we're only goingto use one of them
        // the primtive root inverse might be negative, if o make it positive by wrapping
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
                ((twiddle_input as u64 * primitive_root_inverse) % reduced_len) as usize;
        }

        convolve_fft.execute(&mut inner_fft_input)?;

        let mut input_index = 1;
        let mut input_indices = try_vec![0u32; size - 1];
        for indexer in input_indices.iter_mut() {
            input_index = ((input_index as u64 * primitive_root) % reduced_len) as u32;

            *indexer = input_index - 1;
        }

        let mut output_index = 1;
        let mut output_indices = try_vec![0u32; size - 1];
        for indexer in output_indices.iter_mut() {
            output_index = ((output_index as u64 * primitive_root_inverse) % reduced_len) as u32;
            *indexer = output_index - 1;
        }

        Ok(NeonRadersFft {
            execution_length: size,
            convolve_fft,
            convolve_fft_twiddles: inner_fft_input,
            direction: fft_direction,
            input_indices,
            output_indices,
            spectrum_ops: T::make_complex_arith(),
        })
    }
}

impl<T: FftSample + RadersIndicer<T>> FftExecutor<T> for NeonRadersFft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let mut scratch = try_vec![Complex::zero(); self.execution_length];

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // The first output element is just the sum of all the input elements, and we need to store off the first input value
            let (buffer_first, buffer) = chunk.split_first_mut().unwrap();
            let buffer_first_val = *buffer_first;

            let (scratch, _) = scratch.split_at_mut(self.length() - 1);

            // copy the buffer into the scratch, reordering as we go. also compute a sum of all elements
            T::index_inputs(buffer, scratch, &self.input_indices);

            // perform the first of two inner FFTs

            self.convolve_fft.execute(scratch)?;

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
            self.convolve_fft.execute(scratch)?;

            // copy the final values into the output, reordering as we go
            T::output_indices(buffer, scratch, &self.output_indices);
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}
