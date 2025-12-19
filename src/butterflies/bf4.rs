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
use crate::butterflies::rotate_90;
use crate::traits::FftTrigonometry;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

#[allow(unused)]
pub(crate) struct Butterfly4<T> {
    direction: FftDirection,
    twiddle: Complex<T>,
}

#[allow(unused)]
impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle: match fft_direction {
                FftDirection::Inverse => Complex::new(T::zero(), -T::one()),
                FftDirection::Forward => Complex::new(T::zero(), T::one()),
            },
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
> FftExecutor<T> for Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(4) {
            let a = chunk[0];
            let b = chunk[1];
            let c = chunk[2];
            let d = chunk[3];

            let t0 = a + c;
            let t1 = a - c;
            let t2 = b + d;
            let z3 = b - d;
            let t3 = rotate_90(z3, self.direction);

            chunk[0] = t0 + t2;
            chunk[1] = t1 + t3;
            chunk[2] = t0 - t2;
            chunk[3] = t1 - t3;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        4
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
> FftExecutorOutOfPlace<T> for Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            let a = src[0];
            let b = src[1];
            let c = src[2];
            let d = src[3];

            let t0 = a + c;
            let t1 = a - c;
            let t2 = b + d;
            let z3 = b - d;
            let t3 = rotate_90(z3, self.direction);

            dst[0] = t0 + t2;
            dst[1] = t1 + t3;
            dst[2] = t0 - t2;
            dst[3] = t1 - t3;
        }
        Ok(())
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
        + Send
        + Sync,
> CompositeFftExecutor<T> for Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<T> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_butterfly4, f32, Butterfly4, 4, 1e-5);
    test_oof_butterfly!(test_oof_butterfly4, f32, Butterfly4, 4, 1e-5);
}
