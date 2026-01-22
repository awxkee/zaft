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
use crate::butterflies::fast_bf9::FastButterfly9;
use crate::butterflies::short_butterflies::FastButterfly2;
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;

#[allow(unused)]
pub(crate) struct Butterfly18<T> {
    direction: FftDirection,
    phantom_data: PhantomData<T>,
    bf9: FastButterfly9<T>,
}

#[allow(unused)]
impl<T: FftSample> Butterfly18<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly18 {
            direction: fft_direction,
            phantom_data: PhantomData,
            bf9: FastButterfly9::new(fft_direction),
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Butterfly18<T>
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

        let bf2 = FastButterfly2::new(self.direction);

        for chunk in in_place.chunks_exact_mut(18) {
            let u0 = chunk[0]; // 0
            let u3 = chunk[1]; // 3
            let u4 = chunk[2]; // 4
            let u7 = chunk[3]; // 7

            let u8 = chunk[4]; // 8
            let u11 = chunk[5]; // 11
            let u12 = chunk[6]; // 12
            let u15 = chunk[7]; // 15

            let u16 = chunk[8]; // 16
            let u1 = chunk[9]; // 1
            let u2 = chunk[10]; // 2
            let u5 = chunk[11]; // 5

            let u6 = chunk[12]; // 6
            let u9 = chunk[13]; // 9
            let u10 = chunk[14]; // 10
            let u13 = chunk[15]; // 13

            let u14 = chunk[16]; // 14
            let u17 = chunk[17]; // 17

            let (t0, t1) = bf2.butterfly2(u0, u1);
            let (t2, t3) = bf2.butterfly2(u2, u3);
            let (t4, t5) = bf2.butterfly2(u4, u5);
            let (t6, t7) = bf2.butterfly2(u6, u7);
            let (t8, t9) = bf2.butterfly2(u8, u9);
            let (t10, t11) = bf2.butterfly2(u10, u11);
            let (t12, t13) = bf2.butterfly2(u12, u13);
            let (t14, t15) = bf2.butterfly2(u14, u15);
            let (t16, t17) = bf2.butterfly2(u16, u17);

            let (u0, u2, u4, u6, u8, u10, u12, u14, u16) =
                self.bf9.exec(t0, t2, t4, t6, t8, t10, t12, t14, t16);
            let (u9, u11, u13, u15, u17, u1, u3, u5, u7) =
                self.bf9.exec(t1, t3, t5, t7, t9, t11, t13, t15, t17);

            chunk[0] = u0;
            chunk[1] = u1;
            chunk[2] = u2;
            chunk[3] = u3;
            chunk[4] = u4;
            chunk[5] = u5;
            chunk[6] = u6;
            chunk[7] = u7;
            chunk[8] = u8;
            chunk[9] = u9;
            chunk[10] = u10;
            chunk[11] = u11;
            chunk[12] = u12;
            chunk[13] = u13;
            chunk[14] = u14;
            chunk[15] = u15;
            chunk[16] = u16;
            chunk[17] = u17;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        18
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;

    test_butterfly!(test_butterfly18, f32, Butterfly18, 18, 1e-5);
}
