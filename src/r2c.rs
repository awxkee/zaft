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
use crate::r2c_twiddles::{R2CTwiddlesFactory, R2CTwiddlesHandler};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num, Zero};
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};

pub trait R2CFftExecutor<T> {
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError>;
    fn real_length(&self) -> usize;
    fn complex_length(&self) -> usize;
}

pub(crate) struct R2CFftEvenInterceptor<T> {
    intercept: Box<dyn FftExecutor<T> + Send + Sync>,
    twiddles: Vec<Complex<T>>,
    length: usize,
    complex_length: usize,
    twiddles_handler: Box<dyn R2CTwiddlesHandler<T> + Send + Sync>,
}

impl<
    T: Copy
        + Clone
        + FftTrigonometry
        + Mul<T, Output = T>
        + 'static
        + Zero
        + Num
        + Float
        + R2CTwiddlesFactory<T>,
> R2CFftEvenInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn install(
        length: usize,
        intercept: Box<dyn FftExecutor<T> + Send + Sync>,
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

        let twiddles_count = if length % 4 == 0 {
            length / 4
        } else {
            length / 4 + 1
        };
        let mut twiddles = try_vec![Complex::<T>::zero(); twiddles_count - 1];
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            *twiddle = compute_twiddle(i + 1, length, FftDirection::Forward) * 0.5f64.as_();
        }
        Ok(Self {
            intercept,
            twiddles,
            length,
            complex_length: length / 2 + 1,
            twiddles_handler: T::make_r2c_twiddles_handler(),
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
        + Debug,
> R2CFftExecutor<T> for R2CFftEvenInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if input.len() % self.length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), self.length));
        }
        if output.len() % self.complex_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                output.len(),
                self.complex_length,
            ));
        }
        for (input, output) in input
            .chunks_exact(self.length)
            .zip(output.chunks_exact_mut(self.complex_length))
        {
            for (dst, input_pair) in output
                .iter_mut()
                .zip(input.chunks_exact(2))
                .take(self.length / 2)
            {
                *dst = Complex::new(input_pair[0], input_pair[1]);
            }

            self.intercept.execute(&mut output[..self.length / 2])?;

            let (mut output_left, mut output_right) = output.split_at_mut(output.len() / 2);

            // The first and last element don't require any twiddle factors, so skip that work
            match (output_left.first_mut(), output_right.last_mut()) {
                (Some(first_element), Some(last_element)) => {
                    // The first and last elements are just a sum and difference of the first value's real and imaginary values
                    let first_value = *first_element;
                    *first_element = Complex {
                        re: first_value.re + first_value.im,
                        im: T::zero(),
                    };
                    *last_element = Complex {
                        re: first_value.re - first_value.im,
                        im: T::zero(),
                    };

                    // Chop the first and last element off of our slices so that the loop below doesn't have to deal with them
                    output_left = &mut output_left[1..];
                    let right_len = output_right.len();
                    output_right = &mut output_right[..right_len - 1];
                }
                _ => {
                    return Ok(());
                }
            }

            self.twiddles_handler
                .handle(&self.twiddles, &mut output_left, &mut output_right);

            if output.len() % 2 == 1 {
                if let Some(center_element) = output.get_mut(output.len() / 2) {
                    center_element.im = -center_element.im;
                }
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
}

pub(crate) struct R2CFftOddInterceptor<T> {
    intercept: Box<dyn FftExecutor<T> + Send + Sync>,
    length: usize,
    complex_length: usize,
}

impl<T: Copy + Clone + FftTrigonometry + Mul<T, Output = T> + 'static + Zero + Num + Float>
    R2CFftOddInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn install(
        length: usize,
        intercept: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Self, ZaftError> {
        assert_eq!(
            intercept.direction(),
            FftDirection::Forward,
            "Complex to real fft must be inverse"
        );
        assert_ne!(length % 2, 0, "R2C must be even in even interceptor");
        assert_eq!(
            intercept.length(),
            length,
            "Underlying interceptor must have full length of real values"
        );

        Ok(Self {
            intercept,
            length,
            complex_length: length / 2 + 1,
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
        + MulAdd<T, Output = T>,
> R2CFftExecutor<T> for R2CFftOddInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if input.len() % self.length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), self.length));
        }
        if output.len() % self.complex_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                output.len(),
                self.complex_length,
            ));
        }

        let mut scratch = try_vec![Complex::<T>::zero(); input.len()];

        for (input, output) in input
            .chunks_exact(self.length)
            .zip(output.chunks_exact_mut(self.complex_length))
        {
            for (dst, input_pair) in output.iter_mut().zip(input.chunks_exact(2)) {
                *dst = Complex::new(input_pair[0], input_pair[1]);
            }

            for (val, buf) in input.iter().zip(scratch.iter_mut()) {
                *buf = Complex::new(*val, T::zero());
            }
            // FFT and store result in buffer_out
            self.intercept.execute(&mut scratch)?;
            output.copy_from_slice(&scratch[..self.complex_length()]);
            if let Some(elem) = output.first_mut() {
                elem.im = T::zero();
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
}
