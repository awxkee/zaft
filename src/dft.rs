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
use crate::complex_fma::c_mul_add_fast;
use crate::err::try_vec;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};

pub(crate) struct Dft<T> {
    execution_length: usize,
    twiddles: Vec<Complex<T>>,
    direction: FftDirection,
}

impl<T: FftSample> Dft<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Dft<T>, ZaftError> {
        Ok(Dft {
            execution_length: size,
            twiddles: generate_twiddles_dft(size, fft_direction)?,
            direction: fft_direction,
        })
    }
}

pub(crate) fn generate_twiddles_dft<T: Copy + FftTrigonometry + 'static + Float + Default>(
    size: usize,
    fft_direction: FftDirection,
) -> Result<Vec<Complex<T>>, ZaftError>
where
    f64: AsPrimitive<T>,
{
    let mut twiddles = try_vec![Complex::<T>::default(); size];
    for (k, dst) in twiddles.iter_mut().enumerate() {
        *dst = compute_twiddle(k, size, fft_direction);
    }
    Ok(twiddles)
}

impl<T: FftSample> FftExecutor<T> for Dft<T>
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

        let mut output = try_vec![Complex::<T>::default(); self.execution_length];

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            for (k, dst) in output.iter_mut().enumerate() {
                let mut sum = Complex::<T>::new(0f64.as_(), 0f64.as_());
                let mut twiddle_idx = 0usize;
                for src in chunk.iter() {
                    let w = unsafe { *self.twiddles.get_unchecked(twiddle_idx) };
                    sum = c_mul_add_fast(*src, w, sum);
                    twiddle_idx += k;
                    if twiddle_idx >= self.twiddles.len() {
                        twiddle_idx -= self.twiddles.len();
                    }
                }
                *dst = sum;
            }

            chunk.copy_from_slice(&output);
        }
        Ok(())
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<T>],
        _: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        self.execute(in_place)
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_with_scratch(src, dst, &mut [])
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
        _: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                src.len(),
                self.execution_length,
            ));
        }
        if !dst.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                dst.len(),
                self.execution_length,
            ));
        }

        let mut output = try_vec![Complex::<T>::default(); self.execution_length];

        for (chunk, output_chunk) in src
            .chunks_exact(self.execution_length)
            .zip(dst.chunks_exact_mut(self.execution_length))
        {
            for (k, dst) in output.iter_mut().enumerate() {
                let mut sum = Complex::<T>::new(0f64.as_(), 0f64.as_());
                let mut twiddle_idx = 0usize;
                for src in chunk.iter() {
                    let w = unsafe { *self.twiddles.get_unchecked(twiddle_idx) };
                    sum = c_mul_add_fast(*src, w, sum);
                    twiddle_idx += k;
                    if twiddle_idx >= self.twiddles.len() {
                        twiddle_idx -= self.twiddles.len();
                    }
                }
                *dst = sum;
            }

            output_chunk.copy_from_slice(&output);
        }
        Ok(())
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_with_scratch(src, dst, scratch)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    fn scratch_length(&self) -> usize {
        0
    }

    fn out_of_place_scratch_length(&self) -> usize {
        0
    }

    fn destructive_scratch_length(&self) -> usize {
        0
    }
}
