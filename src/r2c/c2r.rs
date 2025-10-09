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
use crate::r2c::R2CTwiddlesHandler;
use crate::r2c::c2r_twiddles::C2RTwiddlesFactory;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num, Zero};
use std::ops::{Add, Mul, Neg, Sub};

pub trait C2RFftExecutor<T> {
    fn execute(&self, input: &[Complex<T>], output: &mut [T]) -> Result<(), ZaftError>;
    fn real_length(&self) -> usize;
    fn complex_length(&self) -> usize;
}

pub(crate) struct C2RFftEvenInterceptor<T> {
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
        + C2RTwiddlesFactory<T>,
> C2RFftEvenInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn install(
        length: usize,
        intercept: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Self, ZaftError> {
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

        let twiddles_count = if length % 4 == 0 {
            length / 4
        } else {
            length / 4 + 1
        };
        let mut twiddles = try_vec![Complex::<T>::zero(); twiddles_count - 1];
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            *twiddle = compute_twiddle(i + 1, length, FftDirection::Inverse);
        }
        Ok(Self {
            intercept,
            twiddles,
            length,
            complex_length: length / 2 + 1,
            twiddles_handler: T::make_c2r_twiddles_handler(),
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
> C2RFftExecutor<T> for C2RFftEvenInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[Complex<T>], output: &mut [T]) -> Result<(), ZaftError> {
        if output.len() % self.length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), self.length));
        }
        if input.len() % self.complex_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                output.len(),
                self.complex_length,
            ));
        }

        let mut scratch = try_vec![Complex::<T>::zero(); self.complex_length];

        for (input, output) in input
            .chunks_exact(self.complex_length)
            .zip(output.chunks_exact_mut(self.length))
        {
            scratch.copy_from_slice(input);
            scratch[0].im = 0.0f64.as_();
            scratch.last_mut().unwrap().im = 0.0f64.as_();

            let (mut input_left, mut input_right) = scratch.split_at_mut(input.len() / 2);

            // We have to preprocess the input in-place before we send it to the FFT.
            // The first and centermost values have to be preprocessed separately from the rest, so do that now.
            match (input_left.first_mut(), input_right.last_mut()) {
                (Some(first_input), Some(last_input)) => {
                    let first_sum = *first_input + *last_input;
                    let first_diff = *first_input - *last_input;

                    *first_input = Complex {
                        re: first_sum.re - first_sum.im,
                        im: first_diff.re - first_diff.im,
                    };

                    input_left = &mut input_left[1..];
                    let right_len = input_right.len();
                    input_right = &mut input_right[..right_len - 1];
                }
                _ => return Ok(()),
            };

            self.twiddles_handler
                .handle(&self.twiddles, input_left, input_right);

            // If the output len is odd, the loop above can't preprocess the centermost element, so handle that separately
            if scratch.len() % 2 == 1 {
                let center_element = input[input.len() / 2];
                let doubled = center_element + center_element;
                scratch[input.len() / 2] = doubled.conj();
            }

            self.intercept.execute(&mut scratch[..output.len() / 2])?;

            for (dst, src) in output.chunks_exact_mut(2).zip(scratch.iter()) {
                dst[0] = src.re;
                dst[1] = src.im;
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

pub(crate) struct C2RFftOddInterceptor<T> {
    intercept: Box<dyn FftExecutor<T> + Send + Sync>,
    length: usize,
    complex_length: usize,
}

impl<T: Copy + Clone + FftTrigonometry + Mul<T, Output = T> + 'static + Zero + Num + Float>
    C2RFftOddInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn install(
        length: usize,
        intercept: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Self, ZaftError> {
        assert_ne!(length % 2, 0, "R2C must be even in even interceptor");
        assert_eq!(
            intercept.length(),
            length,
            "Underlying interceptor must have full length of real values"
        );
        assert_eq!(
            intercept.direction(),
            FftDirection::Inverse,
            "Complex to real fft must be inverse"
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
> C2RFftExecutor<T> for C2RFftOddInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[Complex<T>], output: &mut [T]) -> Result<(), ZaftError> {
        if output.len() % self.length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), self.length));
        }
        if input.len() % self.complex_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                output.len(),
                self.complex_length,
            ));
        }

        let mut scratch = try_vec![Complex::<T>::zero(); self.length];

        for (input, output) in input
            .chunks_exact(self.complex_length)
            .zip(output.chunks_exact_mut(self.length))
        {
            scratch[..input.len()].copy_from_slice(input);
            scratch[0].im = 0.0.as_();
            for (buf, val) in scratch
                .iter_mut()
                .rev()
                .take(self.length / 2)
                .zip(input.iter().skip(1))
            {
                *buf = val.conj();
            }
            self.intercept.execute(&mut scratch)?;
            for (dst, src) in output.iter_mut().zip(scratch.iter()) {
                *dst = src.re;
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
