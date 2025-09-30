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
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Sub};

pub(crate) struct Dft<T> {
    size: usize,
    twiddles: Vec<Complex<T>>,
    direction: FftDirection,
}

impl<T: Copy + Float + FftTrigonometry + 'static + AsPrimitive<f64>> Dft<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Dft<T>, ZaftError> {
        Ok(Dft {
            size,
            twiddles: generate_twiddles_dft(size, fft_direction)?,
            direction: fft_direction,
        })
    }
}

pub(crate) fn generate_twiddles_dft<T: Copy + FftTrigonometry + 'static>(
    size: usize,
    fft_direction: FftDirection,
) -> Result<Vec<Complex<T>>, ZaftError>
where
    f64: AsPrimitive<T>,
{
    let mut twiddles = Vec::new();
    twiddles
        .try_reserve_exact(size)
        .map_err(|_| ZaftError::OutOfMemory(size))?;
    for t in 0..size {
        let angle = -2.0 * t as f64 / size as f64;
        let angle = match fft_direction {
            FftDirection::Forward => angle,
            FftDirection::Inverse => -angle,
        };
        let (s, c) = angle.as_().sincos_pi();
        twiddles.push(Complex { re: c, im: s });
    }
    Ok(twiddles)
}

impl<
    T: Copy
        + Float
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Num
        + Default
        + FftTrigonometry
        + 'static
        + AsPrimitive<f64>
        + MulAdd<T, Output = T>,
> FftExecutor<T> for Dft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if in_place.len() % self.size != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), self.size));
        }

        let mut output = try_vec![Complex::<T>::default(); self.size];

        for chunk in in_place.chunks_exact_mut(self.size) {
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

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.size
    }
}
