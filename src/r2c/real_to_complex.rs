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
use crate::r2c::r2c_twiddles::R2CTwiddlesHandler;
use crate::util::{compute_twiddle, validate_scratch};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub trait R2CFftExecutor<T> {
    /// Executes the Real-to-Complex Forward FFT.
    ///
    /// The size of the `input` slice must be equal to `self.real_length()`, and the size of the
    /// `output` slice must be equal to `self.complex_length()`.
    ///
    /// # Parameters
    /// * `input`: The **real-valued** time-domain input array.
    /// * `output`: The mutable slice where the **complex-valued, Hermitian symmetric** frequency data will be written.
    ///
    /// # Errors
    /// Returns a `ZaftError` if the execution fails (e.g., due to incorrect slice lengths or internal computation errors).
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError>;
    fn execute_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError>;
    /// Returns the **length** of the **real-valued** input array (N).
    ///
    /// This is the size of the time-domain vector being transformed.
    fn real_length(&self) -> usize;
    /// Returns the **length** of the **complex-valued** output array (`N/2 + 1`).
    ///
    /// This represents the number of complex elements required to store the meaningful, non-redundant
    /// frequency components due to Hermitian symmetry.
    fn complex_length(&self) -> usize;
    fn complex_scratch_length(&self) -> usize;
}

pub(crate) struct R2CFftEvenInterceptor<T> {
    intercept: Arc<dyn FftExecutor<T> + Send + Sync>,
    twiddles: Vec<Complex<T>>,
    length: usize,
    complex_length: usize,
    twiddles_handler: Arc<dyn R2CTwiddlesHandler<T> + Send + Sync>,
    complex_scratch_len: usize,
}

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) struct R2CFftOddInterceptor<T> {
    intercept: Arc<dyn FftExecutor<T> + Send + Sync>,
    length: usize,
    complex_length: usize,
    intercept_scratch_length: usize,
}

#[cfg(any(target_arch = "wasm32", test))]
impl<T: FftSample> R2CFftOddInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn install(
        length: usize,
        intercept: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Self, ZaftError> {
        assert_ne!(length % 2, 0, "R2C must be odd in odd interceptor");
        assert_eq!(
            intercept.length(),
            length,
            "Underlying interceptor must have the full real length"
        );
        assert_eq!(
            intercept.direction(),
            FftDirection::Forward,
            "Real to complex FFT must be forward"
        );

        let intercept_scratch_length = intercept.scratch_length();

        Ok(Self {
            intercept,
            length,
            complex_length: length / 2 + 1,
            intercept_scratch_length,
        })
    }
}

#[cfg(any(target_arch = "wasm32", test))]
impl<T: FftSample> R2CFftExecutor<T> for R2CFftOddInterceptor<T>
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

        let scratch = validate_scratch!(scratch, self.complex_scratch_length());
        let (complex_buffer, intercept_scratch) = scratch.split_at_mut(self.length);

        for (input, output) in input
            .chunks_exact(self.length)
            .zip(output.chunks_exact_mut(self.complex_length))
        {
            for (dst, src) in complex_buffer.iter_mut().zip(input) {
                *dst = Complex::new(*src, T::zero());
            }
            self.intercept
                .execute_with_scratch(complex_buffer, intercept_scratch)?;
            output.copy_from_slice(&complex_buffer[..self.complex_length]);
        }

        Ok(())
    }

    fn real_length(&self) -> usize {
        self.length
    }

    #[inline]
    fn complex_length(&self) -> usize {
        self.complex_length
    }

    #[inline]
    fn complex_scratch_length(&self) -> usize {
        self.length + self.intercept_scratch_length
    }
}

impl<T: FftSample> R2CFftEvenInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn install(
        length: usize,
        intercept: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Self, ZaftError> {
        assert_eq!(
            intercept.direction(),
            FftDirection::Forward,
            "Complex to real fft must be inverse"
        );
        assert_eq!(length % 2, 0, "R2C must be even in even interceptor");
        assert_eq!(
            intercept.length(),
            length / 2,
            "Underlying interceptor must have a half-length of real values"
        );

        let twiddles_count = if length.is_multiple_of(4) {
            length / 4
        } else {
            length / 4 + 1
        };
        let mut twiddles = try_vec![Complex::<T>::zero(); twiddles_count - 1];
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            *twiddle = compute_twiddle(i + 1, length, FftDirection::Forward) * 0.5f64.as_();
        }

        let scratch_len = intercept.scratch_length();

        Ok(Self {
            intercept,
            twiddles,
            length,
            complex_length: length / 2 + 1,
            twiddles_handler: T::make_r2c_twiddles_handler(),
            complex_scratch_len: scratch_len,
        })
    }
}

impl<T: FftSample> R2CFftExecutor<T> for R2CFftEvenInterceptor<T>
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
        self.execute_with_scratch(input, output, scratch.as_mut_slice())
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

        let scratch = validate_scratch!(scratch, self.complex_scratch_length());

        for (input, output) in input
            .chunks_exact(self.length)
            .zip(output.chunks_exact_mut(self.complex_length))
        {
            for (dst, input_pair) in output
                .iter_mut()
                .zip(input.as_chunks::<2>().0.iter())
                .take(self.length / 2)
            {
                *dst = Complex::new(input_pair[0], input_pair[1]);
            }

            self.intercept
                .execute_with_scratch(&mut output[..self.length / 2], scratch)?;

            let (mut output_left, mut output_right) = output.split_at_mut(output.len() / 2);

            match (output_left.first_mut(), output_right.last_mut()) {
                (Some(first_element), Some(last_element)) => {
                    let first_value = *first_element;
                    *first_element = Complex {
                        re: first_value.re + first_value.im,
                        im: T::zero(),
                    };
                    *last_element = Complex {
                        re: first_value.re - first_value.im,
                        im: T::zero(),
                    };

                    output_left = &mut output_left[1..];
                    let right_len = output_right.len();
                    output_right = &mut output_right[..right_len - 1];
                }
                _ => {
                    return Ok(());
                }
            }

            self.twiddles_handler
                .handle(&self.twiddles, output_left, output_right);

            if output.len() % 2 == 1
                && let Some(center_element) = output.get_mut(output.len() / 2)
            {
                center_element.im = -center_element.im;
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

    fn complex_scratch_length(&self) -> usize {
        self.complex_scratch_len
    }
}

#[cfg(test)]
mod tests {
    use super::{R2CFftExecutor, R2CFftOddInterceptor};
    use crate::dft::Dft;
    use crate::{FftDirection, FftExecutor, Zaft};
    use num_complex::Complex;

    #[test]
    fn test_odd_interceptor_prime_lengths() {
        for length in [17, 19, 23, 29] {
            let fft = Zaft::strategy(length, FftDirection::Forward).unwrap();
            let r2c = R2CFftOddInterceptor::install(length, fft).unwrap();
            let input = (0..length)
                .map(|x| ((x * 17 + 3) % 29) as f32 / 29.0)
                .collect::<Vec<_>>();
            let mut output = vec![Complex::<f32>::default(); length / 2 + 1];
            r2c.execute(&input, &mut output).unwrap();

            let dft = Dft::new(length, FftDirection::Forward).unwrap();
            let mut reference = input
                .iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect::<Vec<_>>();
            dft.execute(&mut reference).unwrap();

            for (actual, expected) in output.iter().zip(&reference) {
                assert!((actual.re - expected.re).abs() < 1e-4);
                assert!((actual.im - expected.im).abs() < 1e-4);
            }
        }
    }
}
