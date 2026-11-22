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
use crate::r2c::C2RTwiddlesHandler;
use crate::r2c::c2r_twiddles::C2RTwiddlesFactory;
use crate::util::{compute_twiddle, validate_scratch};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub trait C2RFftExecutor<T> {
    /// Executes the Complex-to-Real Inverse FFT.
    ///
    /// The size of the `input` slice must be equal to `self.complex_length()`, and the size of the
    /// `output` slice must be equal to `self.real_length()`.
    ///
    /// # Parameters
    /// * `input`: The **complex-valued**, frequency-domain input array.
    /// * `output`: The mutable slice where the final **real-valued** time-domain result will be written.
    ///
    /// # Errors
    /// Returns a `ZaftError` if the execution fails (e.g., due to incorrect slice lengths or internal computation errors).
    fn execute(&self, input: &[Complex<T>], output: &mut [T]) -> Result<(), ZaftError>;
    fn execute_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError>;
    /// Returns the **length** of the final **real-valued** output array (N).
    ///
    /// This is the size of the time-domain vector that results from the inverse transform.
    fn real_length(&self) -> usize;
    /// Returns the **length** of the **complex-valued** input array (`N/2 + 1`).
    fn complex_length(&self) -> usize;
    fn complex_scratch_length(&self) -> usize;
}

pub(crate) struct C2RFftEvenInterceptor<T> {
    intercept: Arc<dyn FftExecutor<T> + Send + Sync>,
    twiddles: Vec<Complex<T>>,
    length: usize,
    complex_length: usize,
    twiddles_handler: Arc<dyn C2RTwiddlesHandler<T> + Send + Sync>,
    complex_scratch_len: usize,
}

impl<T: FftSample> C2RFftEvenInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn install(
        length: usize,
        intercept: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Self, ZaftError>
    where
        T: C2RTwiddlesFactory<T>,
    {
        assert_eq!(length % 2, 0, "R2C must be even in even interceptor");
        assert_eq!(
            intercept.length(),
            length / 2,
            "Underlying interceptor must have a half-length of real values"
        );
        assert_eq!(
            intercept.direction(),
            FftDirection::Inverse,
            "Complex to real fft must be inverse"
        );

        let twiddles_count = if length.is_multiple_of(4) {
            length / 4
        } else {
            length / 4 + 1
        };
        let mut twiddles = try_vec![Complex::<T>::zero(); twiddles_count - 1];
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            *twiddle = compute_twiddle(i + 1, length, FftDirection::Inverse);
        }

        let complex_length = length / 2 + 1;
        let complex_scratch_len = if intercept.scratch_length() == 0 {
            0
        } else {
            complex_length + intercept.destructive_scratch_length()
        };

        Ok(Self {
            intercept,
            twiddles,
            length,
            complex_length,
            twiddles_handler: T::make_c2r_twiddles_handler(),
            complex_scratch_len,
        })
    }

    fn prepare_inverse_input(&self, input: &[Complex<T>], packed: &mut [Complex<T>]) {
        let (dc, nyquist) = (input[0], input[input.len() - 1]);
        let first_sum = dc + nyquist;
        let first_diff = dc - nyquist;
        packed[0] = Complex::new(first_sum.re - first_sum.im, first_diff.re - first_diff.im);

        let middle = input.len() / 2;
        let (left_input, right_input) = input.split_at(middle);
        let (left, right) = packed.split_at_mut(middle);
        self.twiddles_handler.handle(
            &self.twiddles,
            &left_input[1..],
            &right_input[..right_input.len() - 1],
            &mut left[1..],
            right,
        );

        if input.len() % 2 == 1 {
            let center = input[middle];
            packed[middle] = (center + center).conj();
        }
    }
}

impl<T: FftSample> C2RFftExecutor<T> for C2RFftEvenInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[Complex<T>], output: &mut [T]) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.complex_length(),
            output.len(),
            self.real_length(),
        )?;
        let mut scratch = try_vec![Complex::zero(); self.complex_scratch_length()];
        self.execute_with_scratch(input, output, scratch.as_mut_slice())
    }

    fn execute_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.complex_length(),
            output.len(),
            self.real_length(),
        )?;

        assert_eq!(align_of::<T>(), align_of::<Complex<T>>());
        assert_eq!(size_of::<T>() * 2, size_of::<Complex<T>>());

        let scratch = validate_scratch!(scratch, self.complex_scratch_length());
        let packed_length = self.length / 2;
        // SAFETY: The output length is even and Complex<T> has two consecutive
        // scalars with the alignment checked above.
        let packed_output =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), output.len() / 2) };
        let blocks = input
            .chunks_exact(self.complex_length)
            .zip(packed_output.chunks_exact_mut(packed_length));

        if scratch.is_empty() {
            for (input, packed) in blocks {
                self.prepare_inverse_input(input, packed);
            }
            if !packed_output.is_empty() {
                self.intercept
                    .execute_with_scratch(packed_output, scratch)?;
            }
        } else {
            let (packed, intercept_scratch) = scratch.split_at_mut(self.complex_length);
            let packed = &mut packed[..packed_length];
            for (input, output) in blocks {
                self.prepare_inverse_input(input, packed);
                self.intercept.execute_destructive_with_scratch(
                    packed,
                    output,
                    intercept_scratch,
                )?;
            }
        }

        Ok(())
    }

    fn real_length(&self) -> usize {
        self.length
    }

    fn complex_length(&self) -> usize {
        self.complex_length
    }

    #[inline]
    fn complex_scratch_length(&self) -> usize {
        self.complex_scratch_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_free_even_inverse_batch() {
        let plan = crate::Zaft::make_c2r_fft_f64(4).unwrap();
        assert_eq!(plan.complex_scratch_length(), 0);

        let input = [
            Complex::new(7.0, 2.0),
            Complex::new(1.5, -0.25),
            Complex::new(-3.0, 5.0),
            Complex::new(-7.0, -2.0),
            Complex::new(-1.5, 0.25),
            Complex::new(3.0, -5.0),
        ];
        let mut output = [100.0; 10];
        plan.execute_with_scratch(&input, &mut output[1..9], &mut [])
            .unwrap();
        assert_eq!(
            output,
            [100.0, 0.0, 13.5, -6.0, 12.5, 0.0, -13.5, 6.0, -12.5, 100.0]
        );
    }
}
