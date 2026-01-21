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
use crate::butterflies::Butterfly2;
use crate::butterflies::fast_bf7::FastButterfly7;
use crate::butterflies::short_butterflies::FastButterfly2;
use crate::{FftDirection, FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;

#[allow(unused)]
pub(crate) struct Butterfly14<T> {
    direction: FftDirection,
    phantom_data: PhantomData<T>,
    bf7: FastButterfly7<T>,
}

#[allow(unused)]
impl<T: FftSample> Butterfly14<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly14 {
            direction: fft_direction,
            phantom_data: PhantomData,
            bf7: FastButterfly7::new(fft_direction),
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Butterfly14<T>
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

        for chunk in in_place.chunks_exact_mut(14) {
            let u0 = chunk[0];
            let u1 = chunk[7];

            let u2 = chunk[8];
            let u3 = chunk[1];

            let u4 = chunk[2];
            let u5 = chunk[9];

            let u6 = chunk[10];
            let u7 = chunk[3];

            let u8 = chunk[4];
            let u9 = chunk[11];

            let u10 = chunk[12];
            let u11 = chunk[5];

            let u12 = chunk[6];
            let u13 = chunk[13];

            // Good-Thomas algorithm

            // Inner 2-point butterflies
            let (u0, u1) = bf2.butterfly2(u0, u1);
            let (u2, u3) = bf2.butterfly2(u2, u3);
            let (u4, u5) = bf2.butterfly2(u4, u5);
            let (u6, u7) = bf2.butterfly2(u6, u7);
            let (u8, u9) = bf2.butterfly2(u8, u9);
            let (u10, u11) = bf2.butterfly2(u10, u11);
            let (u12, u13) = bf2.butterfly2(u12, u13);

            // Outer 7-point butterflies
            let (v0, v2, v4, v6, v8, v10, v12) = self.bf7.exec(u0, u2, u4, u6, u8, u10, u12); // (v0, v1, v2, v3, v4, v5, v6)
            let (v7, v9, v11, v13, v1, v3, v5) = self.bf7.exec(u1, u3, u5, u7, u9, u11, u13); // (v7, v8, v9, v10, v11, v12, v13)

            // // Map back to natural order
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
            chunk[12] = v12;
            chunk[13] = v13;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        14
    }
}

impl<T: FftSample> R2CFftExecutor<T> for Butterfly14<T>
where
    f64: AsPrimitive<T>,
{
    #[inline]
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !input.len().is_multiple_of(14) {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), 14));
        }
        if !output.len().is_multiple_of(8) {
            return Err(ZaftError::InvalidSizeMultiplier(output.len(), 8));
        }
        if input.len() / 14 != output.len() / 8 {
            return Err(ZaftError::InvalidSamplesCount(
                input.len() / 14,
                output.len() / 8,
            ));
        }

        for (chunk, complex) in input.chunks_exact(14).zip(output.chunks_exact_mut(8)) {
            let u0 = Complex::new(chunk[0], T::zero());
            let u1 = Complex::new(chunk[7], T::zero());

            let u2 = Complex::new(chunk[8], T::zero());
            let u3 = Complex::new(chunk[1], T::zero());

            let u4 = Complex::new(chunk[2], T::zero());
            let u5 = Complex::new(chunk[9], T::zero());

            let u6 = Complex::new(chunk[10], T::zero());
            let u7 = Complex::new(chunk[3], T::zero());

            let u8 = Complex::new(chunk[4], T::zero());
            let u9 = Complex::new(chunk[11], T::zero());

            let u10 = Complex::new(chunk[12], T::zero());
            let u11 = Complex::new(chunk[5], T::zero());

            let u12 = Complex::new(chunk[6], T::zero());
            let u13 = Complex::new(chunk[13], T::zero());

            // Good-Thomas algorithm

            // Inner 2-point butterflies
            let [u0, u1] = Butterfly2::exec(&[u0, u1]);
            let [u2, u3] = Butterfly2::exec(&[u2, u3]);
            let [u4, u5] = Butterfly2::exec(&[u4, u5]);
            let [u6, u7] = Butterfly2::exec(&[u6, u7]);
            let [u8, u9] = Butterfly2::exec(&[u8, u9]);
            let [u10, u11] = Butterfly2::exec(&[u10, u11]);
            let [u12, u13] = Butterfly2::exec(&[u12, u13]);

            // Outer 7-point butterflies
            let (v0, v2, v4, v6, _, _, _) = self.bf7.exec(u0, u2, u4, u6, u8, u10, u12); // (v0, v1, v2, v3, v4, v5, v6)
            let (v7, _, _, _, v1, v3, v5) = self.bf7.exec(u1, u3, u5, u7, u9, u11, u13); // (v7, v8, v9, v10, v11, v12, v13)

            // // Map back to natural order
            complex[0] = v0;
            complex[1] = v1;
            complex[2] = v2;
            complex[3] = v3;
            complex[4] = v4;
            complex[5] = v5;
            complex[6] = v6;
            complex[7] = v7;
        }
        Ok(())
    }

    fn real_length(&self) -> usize {
        14
    }

    #[inline]
    fn complex_length(&self) -> usize {
        8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    use crate::r2c::test_r2c_butterfly;

    test_r2c_butterfly!(test_r2c_butterfly14, f32, Butterfly14, 14, 1e-5);
    test_butterfly!(test_butterfly14, f32, Butterfly14, 14, 1e-5);
}
