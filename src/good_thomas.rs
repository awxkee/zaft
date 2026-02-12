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
use crate::fast_divider::DividerUsize;
use crate::transpose::TransposeExecutor;
use crate::util::{validate_oof_sizes, validate_scratch};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub(crate) struct GoodThomasFft<T> {
    width: usize,
    width_size_fft: Arc<dyn FftExecutor<T> + Send + Sync>,

    height: usize,
    height_size_fft: Arc<dyn FftExecutor<T> + Send + Sync>,

    width_divisor: DividerUsize,
    width_divisor_plus_one: DividerUsize,
    execution_length: usize,
    direction: FftDirection,
    transpose_ops: Box<dyn TransposeExecutor<T> + Send + Sync>,
    width_scratch_length: usize,
    height_scratch_length: usize,
    height_destructive_scratch: usize,
}

impl<T: FftSample> GoodThomasFft<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(
        mut width_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        mut height_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<GoodThomasFft<T>, ZaftError> {
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

        let width_scratch_length = width_fft.destructive_scratch_length();
        let height_scratch_length = height_fft.scratch_length();
        let height_destructive_scratch = height_fft.destructive_scratch_length();

        Ok(Self {
            width,
            width_size_fft: width_fft,

            height,
            height_size_fft: height_fft,

            width_divisor: DividerUsize::new(width),
            width_divisor_plus_one: DividerUsize::new(width + 1),

            execution_length: len,
            direction,
            transpose_ops: T::transpose_strategy(width, height),
            width_scratch_length,
            height_scratch_length,
            height_destructive_scratch,
        })
    }
}

impl<T: Copy> GoodThomasFft<T> {
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
        // 4: Scan over the rest of the row, incrementing output_index, and copying each element to output_index, then incrementing output_index
        // 5: The first index of each row will be the final index of the previous row plus one, but because of our incrementing (width+1) inside the loop, we overshot, so at the end of the row, subtract width from output_index
        //
        // This ends up producing the same result as computing the multiplicative inverse of width mod height and etc by the CRT mapping, but with only one integer division per row, instead of one per element.
        let mut destination_index = 0;
        for mut source_row in source.chunks_exact(self.width) {
            let increments_until_cycle =
                1 + (self.execution_length - destination_index) / self.width_divisor_plus_one;

            // If we have to rollover output_index on this row, do it in a separate loop
            if increments_until_cycle < self.width {
                let (pre_cycle_row, post_cycle_row) = source_row.split_at(increments_until_cycle);

                for input_element in pre_cycle_row {
                    unsafe {
                        *destination.get_unchecked_mut(destination_index) = *input_element;
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
                    *destination.get_unchecked_mut(destination_index) = *input_element;
                }
                destination_index += self.width_divisor_plus_one.divisor();
            }

            // The first index of the next will be the final index this row, plus one.
            // But because of our incrementing (width+1) inside the loop above, we overshot, so subtract width, and we'll get (width + 1) - width = 1
            destination_index -= self.width;
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
        // This achieves the same result as the modular arithmetic ofthe ruritanian mapping, but with only one integer divison per row, instead of one per element
        for (y, source_chunk) in source.chunks_exact(self.height).enumerate() {
            let (quotient, remainder) = DividerUsize::div_rem(y * self.height, self.width_divisor);

            // Compute our base index and starting point in the row
            let mut destination_index = remainder;
            let start_x = self.height - quotient;

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

impl<T: FftSample> FftExecutor<T> for GoodThomasFft<T>
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
        let (scratch_left, sr) = scratch.split_at_mut(self.execution_length);

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // Re-index the input, copying from the buffer to the scratch in the process
            self.reindex_input(chunk, scratch_left);

            let (width_scratch, _) = sr.split_at_mut(self.width_scratch_length);
            // run FFTs of size `width`
            self.width_size_fft.execute_destructive_with_scratch(
                scratch_left,
                chunk,
                width_scratch,
            )?;

            // transpose
            self.transpose_ops
                .transpose(chunk, scratch_left, self.width, self.height);

            // run FFTs of size 'height'
            let (height_scratch, _) = sr.split_at_mut(self.height_scratch_length);
            self.height_size_fft
                .execute_with_scratch(scratch_left, height_scratch)?;

            // Re-index the output, copying from the scratch to the buffer in the process
            self.reindex_output(scratch_left, chunk);
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
        let (scratch_left, sr) = scratch.split_at_mut(self.execution_length);

        for (chunk, output_chunk) in src
            .chunks_exact(self.execution_length)
            .zip(dst.chunks_exact_mut(self.execution_length))
        {
            // Re-index the input, copying from the buffer to the scratch in the process
            self.reindex_input(chunk, scratch_left);

            let (width_scratch, _) = sr.split_at_mut(self.width_scratch_length);
            // run FFTs of size `width`
            self.width_size_fft.execute_destructive_with_scratch(
                scratch_left,
                output_chunk,
                width_scratch,
            )?;

            // transpose
            self.transpose_ops
                .transpose(output_chunk, scratch_left, self.width, self.height);

            // run FFTs of size 'height'
            let (height_scratch, _) = sr.split_at_mut(self.height_scratch_length);
            self.height_size_fft
                .execute_with_scratch(scratch_left, height_scratch)?;

            // Re-index the output, copying from the scratch to the buffer in the process
            self.reindex_output(scratch_left, output_chunk);
        }
        Ok(())
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        validate_oof_sizes!(src, dst, self.execution_length);

        let scratch = validate_scratch!(scratch, self.destructive_scratch_length());

        for (src_chunk, output_chunk) in src
            .chunks_exact_mut(self.execution_length)
            .zip(dst.chunks_exact_mut(self.execution_length))
        {
            // Re-index the input, copying from the buffer to the scratch in the process
            self.reindex_input(src_chunk, output_chunk);

            let (width_scratch, _) = scratch.split_at_mut(self.width_scratch_length);
            // run FFTs of size `width`
            self.width_size_fft.execute_destructive_with_scratch(
                output_chunk,
                src_chunk,
                width_scratch,
            )?;

            // transpose
            self.transpose_ops
                .transpose(src_chunk, output_chunk, self.width, self.height);

            // run FFTs of size 'height'
            let (height_scratch, _) = scratch.split_at_mut(self.height_destructive_scratch);
            self.height_size_fft.execute_destructive_with_scratch(
                output_chunk,
                src_chunk,
                height_scratch,
            )?;

            // Re-index the output, copying from the scratch to the buffer in the process
            self.reindex_output(src_chunk, output_chunk);
        }
        Ok(())
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
        self.execution_length + self.width_scratch_length.max(self.height_scratch_length)
    }

    #[inline]
    fn out_of_place_scratch_length(&self) -> usize {
        self.scratch_length()
    }

    #[inline]
    fn destructive_scratch_length(&self) -> usize {
        self.width_scratch_length
            .max(self.height_destructive_scratch)
    }
}
