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
use crate::butterflies::short_butterflies::FastButterfly3;
use crate::complex_fma::c_mul_fast;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

#[allow(unused)]
pub(crate) struct Butterfly9<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle4: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly9 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 9, fft_direction),
            twiddle2: compute_twiddle(2, 9, fft_direction),
            twiddle4: compute_twiddle(4, 9, fft_direction),
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
        + Default
        + FftTrigonometry,
> FftExecutor<T> for Butterfly9<T>
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

        let bf3 = FastButterfly3::new(self.direction);

        for chunk in in_place.chunks_exact_mut(9) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];
            let u4 = chunk[4];
            let u5 = chunk[5];
            let u6 = chunk[6];
            let u7 = chunk[7];
            let u8 = chunk[8];

            // Radix-9 butterfly

            let (u0, u3, u6) = bf3.butterfly3(u0, u3, u6);
            let (u1, mut u4, mut u7) = bf3.butterfly3(u1, u4, u7);
            let (u2, mut u5, mut u8) = bf3.butterfly3(u2, u5, u8);

            u4 = c_mul_fast(u4, self.twiddle1);
            u7 = c_mul_fast(u7, self.twiddle2);
            u5 = c_mul_fast(u5, self.twiddle2);
            u8 = c_mul_fast(u8, self.twiddle4);

            let (zu0, zu3, zu6) = bf3.butterfly3(u0, u1, u2);
            let (zu1, zu4, zu7) = bf3.butterfly3(u3, u4, u5);
            let (zu2, zu5, zu8) = bf3.butterfly3(u6, u7, u8);

            chunk[0] = zu0;
            chunk[1] = zu1;
            chunk[2] = zu2;

            chunk[3] = zu3;
            chunk[4] = zu4;
            chunk[5] = zu5;

            chunk[6] = zu6;
            chunk[7] = zu7;
            chunk[8] = zu8;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        9
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
        + FftTrigonometry,
> FftExecutorOutOfPlace<T> for Butterfly9<T>
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

        let bf3 = FastButterfly3::new(self.direction);

        for (dst, src) in dst.chunks_exact_mut(9).zip(src.chunks_exact(9)) {
            let u0 = src[0];
            let u1 = src[1];
            let u2 = src[2];
            let u3 = src[3];
            let u4 = src[4];
            let u5 = src[5];
            let u6 = src[6];
            let u7 = src[7];
            let u8 = src[8];

            // Radix-9 butterfly

            let (u0, u3, u6) = bf3.butterfly3(u0, u3, u6);
            let (u1, mut u4, mut u7) = bf3.butterfly3(u1, u4, u7);
            let (u2, mut u5, mut u8) = bf3.butterfly3(u2, u5, u8);

            u4 = c_mul_fast(u4, self.twiddle1);
            u7 = c_mul_fast(u7, self.twiddle2);
            u5 = c_mul_fast(u5, self.twiddle2);
            u8 = c_mul_fast(u8, self.twiddle4);

            let (zu0, zu3, zu6) = bf3.butterfly3(u0, u1, u2);
            let (zu1, zu4, zu7) = bf3.butterfly3(u3, u4, u5);
            let (zu2, zu5, zu8) = bf3.butterfly3(u6, u7, u8);

            dst[0] = zu0;
            dst[1] = zu1;
            dst[2] = zu2;

            dst[3] = zu3;
            dst[4] = zu4;
            dst[5] = zu5;

            dst[6] = zu6;
            dst[7] = zu7;
            dst[8] = zu8;
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
        + FftTrigonometry
        + Send
        + Sync,
> CompositeFftExecutor<T> for Butterfly9<T>
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

    test_butterfly!(test_butterfly9, f32, Butterfly9, 9, 1e-5);
    test_oof_butterfly!(test_oof_butterfly9, f32, Butterfly9, 9, 1e-5);
}
