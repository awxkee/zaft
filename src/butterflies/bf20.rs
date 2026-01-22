/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
use crate::butterflies::short_butterflies::{FastButterfly4, FastButterfly5};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;

#[allow(unused)]
pub(crate) struct Butterfly20<T> {
    direction: FftDirection,
    phantom_data: PhantomData<T>,
    bf5: FastButterfly5<T>,
    bf4: FastButterfly4<T>,
}

#[allow(unused)]
impl<T: FftSample> Butterfly20<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly20 {
            direction: fft_direction,
            phantom_data: PhantomData,
            bf5: FastButterfly5::new(fft_direction),
            bf4: FastButterfly4::new(fft_direction),
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Butterfly20<T>
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

        for chunk in in_place.chunks_exact_mut(20) {
            let u0 = chunk[0]; // 0
            let u5 = chunk[1]; // 5
            let u10 = chunk[2]; // 10
            let u15 = chunk[3]; // 15
            let u16 = chunk[4]; // 16

            let u1 = chunk[5]; // 1
            let u6 = chunk[6]; // 6
            let u11 = chunk[7]; // 11
            let u12 = chunk[8]; // 12
            let u17 = chunk[9]; // 17

            let u2 = chunk[10]; // 2
            let u7 = chunk[11]; // 7
            let u8 = chunk[12]; // 8
            let u13 = chunk[13]; // 13
            let u18 = chunk[14]; // 18

            let u3 = chunk[15]; // 3
            let u4 = chunk[16]; // 4
            let u9 = chunk[17]; // 9
            let u14 = chunk[18]; // 14
            let u19 = chunk[19]; // 19

            let (t0, t1, t2, t3) = self.bf4.butterfly4(u0, u1, u2, u3);
            let (t4, t5, t6, t7) = self.bf4.butterfly4(u4, u5, u6, u7);
            let (t8, t9, t10, t11) = self.bf4.butterfly4(u8, u9, u10, u11);
            let (t12, t13, t14, t15) = self.bf4.butterfly4(u12, u13, u14, u15);
            let (t16, t17, t18, t19) = self.bf4.butterfly4(u16, u17, u18, u19);

            let (u0, u4, u8, u12, u16) = self.bf5.bf5(t0, t4, t8, t12, t16);
            let (u5, u9, u13, u17, u1) = self.bf5.bf5(t1, t5, t9, t13, t17);
            let (u10, u14, u18, u2, u6) = self.bf5.bf5(t2, t6, t10, t14, t18);
            let (u15, u19, u3, u7, u11) = self.bf5.bf5(t3, t7, t11, t15, t19);

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
            chunk[18] = u18;
            chunk[19] = u19;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        20
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;

    test_butterfly!(test_butterfly20, f32, Butterfly20, 20, 1e-5);
}
