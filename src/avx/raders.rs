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
use crate::prime_factors::{PrimeFactors, primitive_root};
use crate::spectrum_arithmetic::{SpectrumOps, SpectrumOpsFactory};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_integer::Integer;
use num_traits::{AsPrimitive, Float, MulAdd, Num, Zero};
use std::arch::x86_64::*;
use std::ops::{Add, Mul, Neg, Sub};
use strength_reduce::StrengthReducedU64;

pub(crate) struct AvxRadersFft<T> {
    convolve_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    convolve_fft_twiddles: Vec<Complex<T>>,
    execution_length: usize,
    direction: FftDirection,
    input_indices: Vec<u32>,
    output_indices: Vec<u32>,
    spectrum_ops: Box<dyn SpectrumOps<T> + Send + Sync>,
}

pub(crate) trait RadersIndicer<T> {
    unsafe fn index_inputs(buffer: &[Complex<T>], output: &mut [Complex<T>], indices: &[u32]);
    unsafe fn output_indices(buffer: &mut [Complex<T>], scratch: &[Complex<T>], indices: &[u32]);
}

impl RadersIndicer<f32> for f32 {
    #[target_feature(enable = "avx2")]
    unsafe fn index_inputs(buffer: &[Complex<f32>], output: &mut [Complex<f32>], indices: &[u32]) {
        unsafe {
            let one = _mm_set1_epi32(1); // [1, 1, 1, 1]

            for (scratch_element, buffer_idx) in
                output.chunks_exact_mut(4).zip(indices.chunks_exact(4))
            {
                let idx = _mm_slli_epi32::<1>(_mm_loadu_si128(buffer_idx.as_ptr().cast()));

                let idx_plus_one = _mm_add_epi32(idx, one); // [idx0+1, idx1+1, idx2+1, idx3+1]

                // Interleave: [idx0, idx0+1, idx1, idx1+1]
                let idx0 = _mm_unpacklo_epi32(idx, idx_plus_one); // low 2 elements

                // Interleave: [idx2, idx2+1, idx3, idx3+1]
                let idx1 = _mm_unpackhi_epi32(idx, idx_plus_one); // high 2 elements

                let v0 = _mm256_i32gather_ps::<4>(
                    buffer.as_ptr().cast(),
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(idx0), idx1),
                );

                _mm256_storeu_ps(scratch_element.as_mut_ptr().cast(), v0);
            }

            let rem = output.chunks_exact_mut(4).into_remainder();
            let rem_indices = indices.chunks_exact(4).remainder();

            for (scratch_element, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                *scratch_element = *buffer.get_unchecked(buffer_idx as usize);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn output_indices(
        buffer: &mut [Complex<f32>],
        scratch: &[Complex<f32>],
        indices: &[u32],
    ) {
        unsafe {
            let one = _mm_set1_epi32(1); // [1, 1, 1, 1]
            let conj_factors =
                _mm256_loadu_ps([0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0].as_ptr());

            for (scratch_element, buffer_idx) in
                buffer.chunks_exact_mut(4).zip(indices.chunks_exact(4))
            {
                let idx = _mm_slli_epi32::<1>(_mm_loadu_si128(buffer_idx.as_ptr().cast()));

                let idx_plus_one = _mm_add_epi32(idx, one); // [idx0+1, idx1+1, idx2+1, idx3+1]

                // Interleave: [idx0, idx0+1, idx1, idx1+1]
                let idx0 = _mm_unpacklo_epi32(idx, idx_plus_one); // low 2 elements

                // Interleave: [idx2, idx2+1, idx3, idx3+1]
                let idx1 = _mm_unpackhi_epi32(idx, idx_plus_one); // high 2 elements

                let v0 = _mm256_xor_ps(
                    _mm256_i32gather_ps::<4>(
                        scratch.as_ptr().cast(),
                        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(idx0), idx1),
                    ),
                    conj_factors,
                );

                _mm256_storeu_ps(scratch_element.as_mut_ptr().cast(), v0);
            }

            let rem = buffer.chunks_exact_mut(4).into_remainder();
            let rem_indices = indices.chunks_exact(4).remainder();

            for (dst, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                *dst = scratch.get_unchecked(buffer_idx as usize).conj();
            }
        }
    }
}

impl RadersIndicer<f64> for f64 {
    #[target_feature(enable = "avx2")]
    unsafe fn index_inputs(buffer: &[Complex<f64>], output: &mut [Complex<f64>], indices: &[u32]) {
        unsafe {
            let one = _mm_set1_epi32(1); // [1, 1, 1, 1]

            for (scratch_element, buffer_idx) in
                output.chunks_exact_mut(4).zip(indices.chunks_exact(4))
            {
                let idx = _mm_slli_epi32::<1>(_mm_loadu_si128(buffer_idx.as_ptr().cast()));

                let idx_plus_one = _mm_add_epi32(idx, one); // [idx0+1, idx1+1, idx2+1, idx3+1]

                // Interleave: [idx0, idx0+1, idx1, idx1+1]
                let idx0 = _mm_unpacklo_epi32(idx, idx_plus_one); // low 2 elements

                // Interleave: [idx2, idx2+1, idx3, idx3+1]
                let idx1 = _mm_unpackhi_epi32(idx, idx_plus_one); // high 2 elements

                let v0 = _mm256_i32gather_pd::<8>(buffer.as_ptr().cast(), idx0);
                let v1 = _mm256_i32gather_pd::<8>(buffer.as_ptr().cast(), idx1);

                _mm256_storeu_pd(scratch_element.as_mut_ptr().cast(), v0);
                _mm256_storeu_pd(
                    scratch_element.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    v1,
                );
            }

            let rem = output.chunks_exact_mut(4).into_remainder();
            let rem_indices = indices.chunks_exact(4).remainder();

            for (scratch_element, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                *scratch_element = *buffer.get_unchecked(buffer_idx as usize);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn output_indices(
        buffer: &mut [Complex<f64>],
        scratch: &[Complex<f64>],
        indices: &[u32],
    ) {
        unsafe {
            let one = _mm_set1_epi32(1); // [1, 1, 1, 1]
            let conj_factors = _mm256_loadu_pd([0.0, -0.0, 0.0, -0.0].as_ptr());

            for (scratch_element, buffer_idx) in
                buffer.chunks_exact_mut(4).zip(indices.chunks_exact(4))
            {
                let idx = _mm_slli_epi32::<1>(_mm_loadu_si128(buffer_idx.as_ptr().cast()));

                let idx_plus_one = _mm_add_epi32(idx, one); // [idx0+1, idx1+1, idx2+1, idx3+1]

                // Interleave: [idx0, idx0+1, idx1, idx1+1]
                let idx0 = _mm_unpacklo_epi32(idx, idx_plus_one); // low 2 elements

                // Interleave: [idx2, idx2+1, idx3, idx3+1]
                let idx1 = _mm_unpackhi_epi32(idx, idx_plus_one); // high 2 elements

                let v0 = _mm256_xor_pd(
                    _mm256_i32gather_pd::<8>(scratch.as_ptr().cast(), idx0),
                    conj_factors,
                );
                let v1 = _mm256_xor_pd(
                    _mm256_i32gather_pd::<8>(scratch.as_ptr().cast(), idx1),
                    conj_factors,
                );

                _mm256_storeu_pd(scratch_element.as_mut_ptr().cast(), v0);
                _mm256_storeu_pd(
                    scratch_element.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    v1,
                );
            }

            let rem = buffer.chunks_exact_mut(4).into_remainder();
            let rem_indices = indices.chunks_exact(4).remainder();

            for (dst, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                *dst = scratch.get_unchecked(buffer_idx as usize).conj();
            }
        }
    }
}

impl<
    T: Copy
        + Default
        + Clone
        + FftTrigonometry
        + Float
        + Zero
        + Default
        + SpectrumOpsFactory<T>
        + 'static,
> AvxRadersFft<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn new(
        size: usize,
        convolve_fft: Box<dyn FftExecutor<T> + Send + Sync>,
        fft_direction: FftDirection,
    ) -> Result<AvxRadersFft<T>, ZaftError> {
        assert!(
            PrimeFactors::from_number(size as u64).is_prime(),
            "Input length for Rader's must be a prime number"
        );

        let direction = convolve_fft.direction();
        let convolve_fft_len = convolve_fft.length();
        assert_eq!(fft_direction, direction);
        let reduced_len = StrengthReducedU64::new(size as u64);

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
        let mut z_output = try_vec![0u32; size - 1];
        for (input_idx, &output_idx) in output_indices.iter().enumerate() {
            z_output[output_idx as usize] = input_idx as u32;
        }

        Ok(AvxRadersFft {
            execution_length: size,
            convolve_fft,
            input_indices,
            output_indices: z_output,
            convolve_fft_twiddles: inner_fft_input,
            direction: fft_direction,
            spectrum_ops: T::make_spectrum_arithmetic(),
        })
    }
}

impl<
    T: Copy
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Num
        + 'static
        + Neg<Output = T>
        + MulAdd<T, Output = T>
        + RadersIndicer<T>,
> AvxRadersFft<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_impl(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
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
            unsafe {
                T::index_inputs(buffer, scratch, &self.input_indices);
            }

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
            unsafe {
                T::output_indices(buffer, scratch, &self.output_indices);
            }
        }
        Ok(())
    }
}

impl<
    T: Copy
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Num
        + 'static
        + Neg<Output = T>
        + MulAdd<T, Output = T>
        + RadersIndicer<T>,
> FftExecutor<T> for AvxRadersFft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}
