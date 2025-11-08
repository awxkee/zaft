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
use crate::butterflies::short_butterflies::{FastButterfly2, FastButterfly5};
use crate::traits::FftTrigonometry;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Butterfly10<T> {
    direction: FftDirection,
    bf5: FastButterfly5<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> Butterfly10<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly10 {
            direction: fft_direction,
            bf5: FastButterfly5::new(fft_direction),
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
        + Float
        + Default,
> FftExecutor<T> for Butterfly10<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if in_place.len() % self.length() != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let bf2 = FastButterfly2::new(self.direction);

        for chunk in in_place.chunks_exact_mut(10) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];
            let u4 = chunk[4];
            let u5 = chunk[5];
            let u6 = chunk[6];
            let u7 = chunk[7];
            let u8 = chunk[8];
            let u9 = chunk[9];

            // Good-thomas butterfly-10
            let mid0 = self.bf5.bf5(u0, u2, u4, u6, u8);
            let mid1 = self.bf5.bf5(u5, u7, u9, u1, u3);

            // Since this is good-thomas algorithm, we don't need twiddle factors
            let (y0, y1) = bf2.butterfly2(mid0.0, mid1.0);
            let (y2, y3) = bf2.butterfly2(mid0.1, mid1.1);
            let (y4, y5) = bf2.butterfly2(mid0.2, mid1.2);
            let (y6, y7) = bf2.butterfly2(mid0.3, mid1.3);
            let (y8, y9) = bf2.butterfly2(mid0.4, mid1.4);

            chunk[0] = y0;
            chunk[1] = y3;
            chunk[2] = y4;

            chunk[3] = y7;
            chunk[4] = y8;
            chunk[5] = y1;

            chunk[6] = y2;
            chunk[7] = y5;
            chunk[8] = y6;

            chunk[9] = y9;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        10
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
        + Float
        + Default,
> FftExecutorOutOfPlace<T> for Butterfly10<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if src.len() % self.length() != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % self.length() != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        let bf2 = FastButterfly2::<T>::new(self.direction);

        for (dst, src) in dst.chunks_exact_mut(10).zip(src.chunks_exact(10)) {
            let u0 = src[0];
            let u1 = src[1];
            let u2 = src[2];
            let u3 = src[3];
            let u4 = src[4];
            let u5 = src[5];
            let u6 = src[6];
            let u7 = src[7];
            let u8 = src[8];
            let u9 = src[9];

            // Good-thomas butterfly-10
            let mid0 = self.bf5.bf5(u0, u2, u4, u6, u8);
            let mid1 = self.bf5.bf5(u5, u7, u9, u1, u3);

            // Since this is good-thomas algorithm, we don't need twiddle factors
            let (y0, y1) = bf2.butterfly2(mid0.0, mid1.0);
            let (y2, y3) = bf2.butterfly2(mid0.1, mid1.1);
            let (y4, y5) = bf2.butterfly2(mid0.2, mid1.2);
            let (y6, y7) = bf2.butterfly2(mid0.3, mid1.3);
            let (y8, y9) = bf2.butterfly2(mid0.4, mid1.4);

            dst[0] = y0;
            dst[1] = y3;
            dst[2] = y4;

            dst[3] = y7;
            dst[4] = y8;
            dst[5] = y1;

            dst[6] = y2;
            dst[7] = y5;
            dst[8] = y6;

            dst[9] = y9;
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
        + Float
        + Default
        + Send
        + Sync,
> CompositeFftExecutor<T> for Butterfly10<T>
where
    f64: AsPrimitive<T>,
{
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<T> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_butterfly10, f32, Butterfly10, 10, 1e-5);
    test_oof_butterfly!(test_oof_butterfly10, f32, Butterfly10, 10, 1e-5);
}
