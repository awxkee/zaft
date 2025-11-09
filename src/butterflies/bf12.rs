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
use crate::butterflies::short_butterflies::{FastButterfly3, FastButterfly4};
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Butterfly12<T> {
    direction: FftDirection,
    phantom_data: PhantomData<T>,
    bf3: FastButterfly3<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly12 {
            direction: fft_direction,
            phantom_data: PhantomData,
            bf3: FastButterfly3::new(fft_direction),
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
> FftExecutor<T> for Butterfly12<T>
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

        let bf4 = FastButterfly4::new(self.direction);

        for chunk in in_place.chunks_exact_mut(12) {
            let u0 = chunk[0];
            let u1 = chunk[3];
            let u2 = chunk[6];
            let u3 = chunk[9];

            let u4 = chunk[4];
            let u5 = chunk[7];
            let u6 = chunk[10];
            let u7 = chunk[1];

            let u8 = chunk[8];
            let u9 = chunk[11];
            let u10 = chunk[2];
            let u11 = chunk[5];

            let (u0, u1, u2, u3) = bf4.butterfly4(u0, u1, u2, u3);
            let (u4, u5, u6, u7) = bf4.butterfly4(u4, u5, u6, u7);
            let (u8, u9, u10, u11) = bf4.butterfly4(u8, u9, u10, u11);

            let (v0, v4, v8) = self.bf3.butterfly3(u0, u4, u8); // (v0, v4, v8)
            let (v9, v1, v5) = self.bf3.butterfly3(u1, u5, u9); // (v9, v1, v5)
            let (v6, v10, v2) = self.bf3.butterfly3(u2, u6, u10); // (v6, v10, v2)
            let (v3, v7, v11) = self.bf3.butterfly3(u3, u7, u11); // (v3, v7, v11)

            chunk[0] = v0;
            chunk[1] = v1;
            chunk[2] = v2;
            chunk[3] = v3;

            chunk[4] = v4;
            chunk[5] = v5;
            chunk[6] = v6;
            chunk[7] = v7;

            chunk[8] = v8;
            chunk[9] = v9;
            chunk[10] = v10;
            chunk[11] = v11;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        12
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;

    test_butterfly!(test_butterfly12, f32, Butterfly12, 12, 1e-5);
}
