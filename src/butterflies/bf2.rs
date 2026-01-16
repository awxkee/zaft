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
use crate::{
    CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, FftSample,
    R2CFftExecutor, ZaftError,
};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;
use std::sync::Arc;

#[allow(unused)]
pub(crate) struct Butterfly2<T> {
    pub(crate) phantom_data: PhantomData<T>,
    pub(crate) direction: FftDirection,
}

#[allow(unused)]
impl<T> Butterfly2<T> {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            phantom_data: PhantomData,
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Butterfly2<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(2) {
            let u0 = chunk[0];
            let u1 = chunk[1];

            let y0 = u0 + u1;
            let y1 = u0 - u1;

            chunk[0] = y0;
            chunk[1] = y1;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        2
    }
}

impl<T: FftSample> Butterfly2<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn exec(data: &[Complex<T>; 2]) -> [Complex<T>; 2] {
        let u0 = data[0];
        let u1 = data[1];

        let y0 = u0 + u1;
        let y1 = u0 - u1;

        [y0, y1]
    }
}

impl<T: FftSample> FftExecutorOutOfPlace<T> for Butterfly2<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
            let u0 = src[0];
            let u1 = src[1];

            let y0 = u0 + u1;
            let y1 = u0 - u1;

            dst[0] = y0;
            dst[1] = y1;
        }
        Ok(())
    }
}

impl<T: FftSample> CompositeFftExecutor<T> for Butterfly2<T>
where
    f64: AsPrimitive<T>,
{
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<T> + Send + Sync> {
        self
    }
}

impl<T: FftSample> R2CFftExecutor<T> for Butterfly2<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !input.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), 2));
        }
        if !output.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(output.len(), 2));
        }

        for (input, complex) in input.chunks_exact(2).zip(output.chunks_exact_mut(2)) {
            let u0 = input[0];
            let u1 = input[1];

            let y0 = u0 + u1;
            let y1 = u0 - u1;

            complex[0] = Complex::new(y0, T::zero());
            complex[1] = Complex::new(y1, T::zero());
        }
        Ok(())
    }

    fn real_length(&self) -> usize {
        2
    }

    fn complex_length(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    use crate::r2c::test_r2c_butterfly;

    test_r2c_butterfly!(test_r2c_butterfly2, f32, Butterfly2, 2, 1e-5);
    test_butterfly!(test_butterfly2, f32, Butterfly2, 2, 1e-5);
    test_oof_butterfly!(test_oof_butterfly2, f32, Butterfly2, 2, 1e-5);
}
