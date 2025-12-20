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
use crate::butterflies::short_butterflies::{FastButterfly2, FastButterfly3};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

#[allow(unused)]
pub(crate) struct Butterfly6<T> {
    direction: FftDirection,
    twiddle: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly6 {
            direction: fft_direction,
            twiddle: compute_twiddle(1, 3, fft_direction),
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
        + MulAdd<T, Output = T>
        + FftTrigonometry
        + Float
        + Default,
> FftExecutor<T> for Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.length()) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let fast_butterfly3 = FastButterfly3::new(self.direction);
        let fast_butterfly2 = FastButterfly2::new(self.direction);

        for chunk in in_place.chunks_exact_mut(6) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];
            let u4 = chunk[4];
            let u5 = chunk[5];

            // Radix-6 butterfly

            let (t0, t2, t4) = fast_butterfly3.butterfly3(u0, u2, u4);
            let (t1, t3, t5) = fast_butterfly3.butterfly3(u3, u5, u1);
            let (y0, y3) = fast_butterfly2.butterfly2(t0, t1);
            let (y4, y1) = fast_butterfly2.butterfly2(t2, t3);
            let (y2, y5) = fast_butterfly2.butterfly2(t4, t5);

            chunk[0] = y0;
            chunk[1] = y1;
            chunk[2] = y2;
            chunk[3] = y3;
            chunk[4] = y4;
            chunk[5] = y5;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        6
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
        + FftTrigonometry
        + Float
        + Default,
> FftExecutorOutOfPlace<T> for Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(self.length()) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(self.length()) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        let fast_butterfly3 = FastButterfly3::new(self.direction);
        let fast_butterfly2 = FastButterfly2::new(self.direction);

        for (dst, src) in dst.chunks_exact_mut(6).zip(src.chunks_exact(6)) {
            let u0 = src[0];
            let u1 = src[1];
            let u2 = src[2];
            let u3 = src[3];
            let u4 = src[4];
            let u5 = src[5];

            // Radix-6 butterfly

            let (t0, t2, t4) = fast_butterfly3.butterfly3(u0, u2, u4);
            let (t1, t3, t5) = fast_butterfly3.butterfly3(u3, u5, u1);
            let (y0, y3) = fast_butterfly2.butterfly2(t0, t1);
            let (y4, y1) = fast_butterfly2.butterfly2(t2, t3);
            let (y2, y5) = fast_butterfly2.butterfly2(t4, t5);

            dst[0] = y0;
            dst[1] = y1;
            dst[2] = y2;
            dst[3] = y3;
            dst[4] = y4;
            dst[5] = y5;
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
        + FftTrigonometry
        + Float
        + Default
        + Send
        + Sync,
> CompositeFftExecutor<T> for Butterfly6<T>
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

    test_butterfly!(test_butterfly6, f32, Butterfly6, 6, 1e-5);
    test_oof_butterfly!(test_oof_butterfly6, f32, Butterfly6, 6, 1e-5);
}
