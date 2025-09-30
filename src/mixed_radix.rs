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
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num, Zero};
use std::ops::{Add, Mul, Neg, Sub};

pub(crate) struct MixedRadix<T> {
    execution_length: usize,
    direction: FftDirection,
    twiddles: Vec<Complex<T>>,
    width_executor: Box<dyn FftExecutor<T> + Send + Sync>,
    width: usize,
    height_executor: Box<dyn FftExecutor<T> + Send + Sync>,
    height: usize,
}

impl<T: Copy + 'static + FftTrigonometry + Float> MixedRadix<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(
        width_executor: Box<dyn FftExecutor<T> + Send + Sync>,
        height_executor: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Self, ZaftError> {
        assert_eq!(
            width_executor.direction(),
            height_executor.direction(),
            "width_fft and height_fft must have the same direction. got width direction={}, height direction={}",
            width_executor.direction(),
            height_executor.direction()
        );

        let direction = width_executor.direction();

        let width = width_executor.length();
        let height = height_executor.length();

        let len = width * height;

        let mut twiddles = try_vec![Complex::zero(); len];
        for (x, row) in twiddles.chunks_exact_mut(height).enumerate() {
            for (y, dst) in row.iter_mut().enumerate() {
                *dst = compute_twiddle(x * y, len, direction);
            }
        }
        Ok(MixedRadix {
            execution_length: width * height,
            width_executor,
            width,
            height_executor,
            height,
            direction,
            twiddles,
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
> FftExecutor<T> for MixedRadix<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let mut scratch = try_vec![Complex::zero(); self.width * self.height];
        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // SIX STEP FFT:
            // STEP 1: transpose
            transpose_small(self.width, self.height, chunk, &mut scratch);

            // STEP 2: perform FFTs of size `height`
            self.height_executor.execute(&mut scratch)?;

            // STEP 3: Apply twiddle factors
            for ((src, dst), twiddle) in scratch
                .iter()
                .zip(chunk.iter_mut())
                .zip(self.twiddles.iter())
            {
                *dst = *src * twiddle;
            }

            // STEP 4: transpose again
            transpose_small(self.height, self.width, chunk, &mut scratch);

            // STEP 5: perform FFTs of size `width`
            self.width_executor.execute(&mut scratch)?;

            // STEP 6: transpose again
            transpose_small(self.width, self.height, &scratch, chunk);
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

pub(crate) fn transpose_small<T: Copy>(width: usize, height: usize, input: &[T], output: &mut [T]) {
    for x in 0..width {
        for y in 0..height {
            let input_index = x + y * width;
            let output_index = y + x * height;

            unsafe {
                *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
            }
        }
    }
}
