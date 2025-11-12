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
use crate::fast_divider::DividerU16;
use crate::traits::FftTrigonometry;
use crate::transpose::{TransposeExecutor, TransposeFactory};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num, Zero};
use std::ops::{Add, Mul, Neg, Sub};

pub(crate) struct GoodThomasSmallFft<T> {
    width: usize,
    width_size_fft: Box<dyn FftExecutor<T> + Send + Sync>,

    height: usize,
    height_size_fft: Box<dyn FftExecutor<T> + Send + Sync>,

    width_divisor: DividerU16,
    width_divisor_plus_one: DividerU16,
    execution_length: u16,
    direction: FftDirection,
    transpose_ops: Box<dyn TransposeExecutor<T> + Send + Sync>,
}

impl<
    T: Copy
        + Default
        + Clone
        + FftTrigonometry
        + Float
        + Zero
        + Default
        + TransposeFactory<T>
        + 'static,
> GoodThomasSmallFft<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(
        mut width_fft: Box<dyn FftExecutor<T> + Send + Sync>,
        mut height_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<GoodThomasSmallFft<T>, ZaftError> {
        assert_eq!(
            width_fft.direction(),
            height_fft.direction(),
            "width_fft and height_fft must have the same direction. got width direction={}, height direction={}",
            width_fft.direction(),
            height_fft.direction()
        );

        let mut width = width_fft.length();
        let mut height = height_fft.length();
        let direction = width_fft.direction();

        // This algorithm doesn't work if width and height aren't coprime
        let gcd = num_integer::gcd(width as i64, height as i64);
        assert_eq!(
            gcd, 1,
            "Invalid width and height for Good-Thomas Algorithm (width={width}, height={height}): Inputs must be coprime"
        );

        // The trick we're using for our index remapping will only work if width < height, so just swap them if it isn't
        if width > height {
            std::mem::swap(&mut width, &mut height);
            std::mem::swap(&mut width_fft, &mut height_fft);
        }

        let len = width * height;

        Ok(Self {
            width,
            width_size_fft: width_fft,

            height,
            height_size_fft: height_fft,

            width_divisor: DividerU16::new(width as u16),
            width_divisor_plus_one: DividerU16::new(width as u16 + 1),

            execution_length: len as u16,
            direction,
            transpose_ops: T::transpose_strategy(width, height),
        })
    }
}

impl<T: Copy> GoodThomasSmallFft<T> {
    fn reindex_input(&self, source: &[Complex<T>], destination: &mut [Complex<T>]) {
        // A critical part of the good-thomas algorithm is re-indexing the inputs and outputs.
        // To remap the inputs, we will use the CRT mapping, paired with the normal transpose we'd do for mixed radix.
        //
        // The algorithm for the CRT mapping will work like this:
        // 1: Keep an output index, initialized to 0
        // 2: The output index will be incremented by width + 1
        // 3: At the start of the row, compute if we will increment output_index past self.len()
        //      3a: If we will, then compute exactly how many increments it will take,
        //      3b: Increment however many times as we scan over the input row, copying each element to the output index
        //      3c: Subtract self.len() from output_index
        // 4: Scan over the rest of the row, incrementing output_index, and copying each element to output_index, thne incrementing output_index
        // 5: The first index of each row will be the final index of the previous row plus one, but because of our incrementing (width+1) inside the loop, we overshot, so at the end of the row, subtract width from output_index
        //
        // This ends up producing the same result as computing the multiplicative inverse of width mod height and etc by the CRT mapping, but with only one integer division per row, instead of one per element.
        let mut destination_index = 0u16;
        for mut source_row in source.chunks_exact(self.width) {
            let increments_until_cycle = (1
                + (self.execution_length - destination_index) / self.width_divisor_plus_one)
                as usize;

            // If we have to rollover output_index on this row, do it in a separate loop
            if increments_until_cycle < self.width {
                let (pre_cycle_row, post_cycle_row) = source_row.split_at(increments_until_cycle);

                for input_element in pre_cycle_row {
                    unsafe {
                        *destination.get_unchecked_mut(destination_index as usize) = *input_element;
                    }
                    destination_index += self.width_divisor_plus_one.divisor();
                }

                // Store the split slice back to input_row, os that outside the loop, we can finish the job of iterating the row
                source_row = post_cycle_row;
                destination_index -= self.execution_length;
            }

            // Loop over the entire row (if we did not roll over) or what's left of the row (if we did) and keep incrementing output_row
            for input_element in source_row {
                unsafe {
                    *destination.get_unchecked_mut(destination_index as usize) = *input_element;
                }
                destination_index += self.width_divisor_plus_one.divisor();
            }

            // The first index of the next will be the final index this row, plus one.
            // But because of our incrementing (width+1) inside the loop above, we overshot, so subtract width, and we'll get (width + 1) - width = 1
            destination_index -= self.width as u16;
        }
    }

    fn reindex_output(&self, source: &[Complex<T>], destination: &mut [Complex<T>]) {
        // A critical part of the good-thomas algorithm is re-indexing the inputs and outputs.
        // To remap the outputs, we will use the ruritanian mapping, paired with the normal transpose we'd do for mixed radix.
        //
        // The algorithm for the ruritanian mapping will work like this:
        // 1: At the start of every row, compute the output index = (y * self.height) % self.width
        // 2: We will increment this output index by self.width for every element
        // 3: Compute where in the row the output index will wrap around
        // 4: Instead of starting copying from the beginning of the row, start copying from after the rollover point
        // 5: When we hit the end of the row, continue from the beginning of the row, continuing to increment the output index by self.width
        //
        // This achieves the same result as the modular arithmetic of the ruritanian mapping, but with only one integer divison per row, instead of one per element
        for (y, source_chunk) in source.chunks_exact(self.height).enumerate() {
            let (quotient, remainder) =
                DividerU16::div_rem(y as u16 * self.height as u16, self.width_divisor);

            // Compute our base index and starting point in the row
            let mut destination_index = remainder as usize;
            let start_x = self.height - quotient as usize;

            // Process the first part of the row
            for x in start_x..self.height {
                unsafe {
                    *destination.get_unchecked_mut(destination_index) =
                        *source_chunk.get_unchecked(x);
                }
                destination_index += self.width;
            }

            // Wrap back around to the beginning of the row and keep incrementing
            for x in 0..start_x {
                unsafe {
                    *destination.get_unchecked_mut(destination_index) =
                        *source_chunk.get_unchecked(x);
                }
                destination_index += self.width;
            }
        }
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
> FftExecutor<T> for GoodThomasSmallFft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        let length = self.execution_length as usize;
        if in_place.len() % length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), length));
        }

        let mut scratch_full = try_vec![Complex::zero(); length * 2];
        let (scratch_left, scratch_right) = scratch_full.split_at_mut(length);

        for chunk in in_place.chunks_exact_mut(length) {
            // Re-index the input, copying from the buffer to the scratch in the process
            self.reindex_input(chunk, scratch_left);

            // run FFTs of size `width`
            self.width_size_fft.execute(scratch_left)?;

            // transpose
            self.transpose_ops
                .transpose(scratch_left, scratch_right, self.width, self.height);

            // run FFTs of size 'height'
            self.height_size_fft.execute(scratch_right)?;
            // Re-index the output, copying from the scratch to the buffer in the process
            self.reindex_output(scratch_right, chunk);
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length as usize
    }
}
