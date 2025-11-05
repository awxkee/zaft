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

use crate::butterflies::short_butterflies::{FastButterfly3, FastButterfly5};
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Butterfly15<T> {
    direction: FftDirection,
    bf5: FastButterfly5<T>,
    bf3: FastButterfly3<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> Butterfly15<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly15 {
            direction: fft_direction,
            bf5: FastButterfly5::new(fft_direction),
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
        + Default
        + FftTrigonometry,
> FftExecutor<T> for Butterfly15<T>
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

        for chunk in in_place.chunks_exact_mut(15) {
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
            let u10 = chunk[10];
            let u11 = chunk[11];
            let u12 = chunk[12];

            let u13 = chunk[13];
            let u14 = chunk[14];

            // Size-5 FFTs down the columns of our reordered array
            let mid0 = self.bf5.bf5(u0, u3, u6, u9, u12);
            let mid1 = self.bf5.bf5(u5, u8, u11, u14, u2);
            let mid2 = self.bf5.bf5(u10, u13, u1, u4, u7);

            // Since this is good-thomas algorithm, we don't need twiddle factors

            // Transpose the data and do size-3 FFTs down the columns
            let (y0, y1, y2) = self.bf3.butterfly3(mid0.0, mid1.0, mid2.0);
            let (y3, y4, y5) = self.bf3.butterfly3(mid0.1, mid1.1, mid2.1);
            let (y6, y7, y8) = self.bf3.butterfly3(mid0.2, mid1.2, mid2.2);
            let (y9, y10, y11) = self.bf3.butterfly3(mid0.3, mid1.3, mid2.3);
            let (y12, y13, y14) = self.bf3.butterfly3(mid0.4, mid1.4, mid2.4);

            chunk[0] = y0;
            chunk[1] = y4;

            chunk[2] = y8;
            chunk[3] = y9;

            chunk[4] = y13;
            chunk[5] = y2;

            chunk[6] = y3;
            chunk[7] = y7;

            chunk[8] = y11;
            chunk[9] = y12;

            chunk[10] = y1;
            chunk[11] = y5;

            chunk[12] = y6;
            chunk[13] = y10;

            chunk[14] = y14;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        15
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    use rand::Rng;

    test_butterfly!(test_butterfly15, f32, Butterfly15, 15, 1e-5);
}
