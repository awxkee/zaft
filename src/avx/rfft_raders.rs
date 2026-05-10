/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use crate::avx::raders::{AvxRadersFactory, RadersIndicer};
use crate::err::try_vec;
use crate::fast_divider::DividerU64;
use crate::prime_factors::{PrimeFactors, primitive_root};
use crate::spectrum_arithmetic::ComplexArith;
use crate::util::{compute_twiddle, validate_scratch};
use crate::{FftDirection, FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_integer::Integer;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub(crate) struct AvxRadersRFft<T> {
    convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    convolve_r2c: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
    convolve_fft_twiddles: Vec<Complex<T>>,
    execution_length: usize,
    input_indices: Vec<u32>,
    output_indices: Vec<u32>,
    spectrum_ops: Arc<dyn ComplexArith<T> + Send + Sync>,
    convolve_fft_scratch_length: usize,
    indicer: Arc<dyn RadersIndicer<T> + Send + Sync>,
}

impl<T: FftSample + AvxRadersFactory<T>> AvxRadersRFft<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        size: usize,
        convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        convolve_r2c: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
        fft_direction: FftDirection,
    ) -> Result<AvxRadersRFft<T>, ZaftError> {
        unsafe { Self::new_init(size, convolve_fft, convolve_r2c, fft_direction) }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn new_init(
        size: usize,
        convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        convolve_r2c: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
        fft_direction: FftDirection,
    ) -> Result<AvxRadersRFft<T>, ZaftError> {
        assert!(
            PrimeFactors::from_number(size as u64).is_prime(),
            "Input length for Rader's must be a prime number"
        );

        let direction = convolve_fft.direction();
        let convolve_fft_len = convolve_fft.length();
        assert_eq!(fft_direction, direction);
        let dividing_len = DividerU64::new(size as u64);

        let primitive_root =
            primitive_root(size as u64).ok_or(ZaftError::CantFindPrimitiveRootFor(size as u64))?;

        let gcd_data = i64::extended_gcd(&(primitive_root as i64), &(size as i64));
        let primitive_root_inverse = if gcd_data.x >= 0 {
            gcd_data.x
        } else {
            gcd_data.x + size as i64
        } as u64;

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
        let input_indices = (0..size - 1)
            .map(|_| {
                input_index = ((input_index as u64 * primitive_root) % dividing_len) as usize;
                (input_index - 1) as u32
            })
            .collect::<Vec<_>>();

        let mut output_index = 1;
        let output_indices = (0..size - 1)
            .map(|_| {
                output_index =
                    ((output_index as u64 * primitive_root_inverse) % dividing_len) as usize;
                (output_index - 1) as u32
            })
            .collect::<Vec<_>>();

        let mut z_output = try_vec![0u32; size - 1];
        for (input_idx, &output_idx) in output_indices.iter().enumerate() {
            z_output[output_idx as usize] = input_idx as u32;
        }

        let convolve_scratch_length = convolve_fft
            .scratch_length()
            .max(convolve_r2c.complex_scratch_length());

        Ok(AvxRadersRFft {
            execution_length: size,
            convolve_fft,
            input_indices,
            output_indices: z_output,
            convolve_fft_twiddles: inner_fft_input,
            spectrum_ops: T::make_complex_arith(),
            convolve_fft_scratch_length: convolve_scratch_length,
            indicer: T::make_raders_indicer(),
            convolve_r2c,
        })
    }
}

impl<T: FftSample> AvxRadersRFft<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_r2c(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !input.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                input.len(),
                self.execution_length,
            ));
        }
        if !output.len().is_multiple_of(self.complex_length()) {
            return Err(ZaftError::InvalidSizeMultiplier(
                output.len(),
                self.complex_length(),
            ));
        }
        if input.len() / self.execution_length != output.len() / self.complex_length() {
            return Err(ZaftError::InvalidSamplesCount(
                input.len() / self.execution_length,
                output.len() / self.complex_length(),
            ));
        }

        assert_eq!(size_of::<T>() * 2, size_of::<Complex<T>>());
        assert_eq!(align_of::<T>(), align_of::<Complex<T>>());

        let scratch = validate_scratch!(scratch, self.complex_scratch_length());
        let (scratch, rm) = scratch.split_at_mut(self.execution_length);
        let (real_scratch_c, convolve_scratch) =
            rm.split_at_mut(self.convolve_fft_twiddles.len() / 2 + 1);
        assert!(real_scratch_c.len() >= self.convolve_fft_twiddles.len() / 2 + 1);
        let real_scratch: &mut [T] = unsafe {
            std::slice::from_raw_parts_mut(
                real_scratch_c.as_mut_ptr().cast(),
                self.convolve_fft_twiddles.len(),
            )
        };

        let convolve_complex_length = self.convolve_fft_twiddles.len() / 2 + 1;

        for (input, complex) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.complex_length()))
        {
            let (buffer_first, buffer) = input.split_first().unwrap();
            let buffer_first_val = *buffer_first;

            let (scratch, _) = scratch.split_at_mut(self.real_length() - 1);

            for (scratch_element, &buffer_idx) in
                real_scratch.iter_mut().zip(self.input_indices.iter())
            {
                *scratch_element = unsafe { *buffer.get_unchecked(buffer_idx as usize) }
            }

            self.convolve_r2c.execute_with_scratch(
                real_scratch,
                &mut scratch[..convolve_complex_length],
                convolve_scratch,
            )?;

            // scratch[0] now contains the sum of elements 1..len. We need the sum of all elements, so all we have to do is add the first input
            complex[0] = Complex::new(buffer_first_val, T::zero()) + scratch[0];

            self.spectrum_ops
                .mul_conjugate_expand_h2c(scratch, &self.convolve_fft_twiddles);

            scratch[0] = scratch[0] + Complex::new(buffer_first_val, T::zero());

            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            let output = &mut complex[1..];
            let out_len = output.len();
            unsafe {
                self.indicer
                    .output_indices(output, scratch, &self.output_indices[..out_len]);
            }
        }
        Ok(())
    }
}

impl<T: FftSample> R2CFftExecutor<T> for AvxRadersRFft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        let mut scratch = vec![Complex::zero(); self.complex_scratch_length()];
        unsafe { self.execute_r2c(input, output, scratch.as_mut_slice()) }
    }

    fn execute_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_r2c(input, output, scratch) }
    }

    #[inline]
    fn real_length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn complex_length(&self) -> usize {
        self.execution_length / 2 + 1
    }

    #[inline]
    fn complex_scratch_length(&self) -> usize {
        self.execution_length
            + self.convolve_fft_twiddles.len() / 2
            + 1
            + self.convolve_fft_scratch_length
    }
}
