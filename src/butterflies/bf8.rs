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
#![allow(unused)]
use crate::butterflies::short_butterflies::{FastButterfly2, FastButterfly4};
use crate::butterflies::util::boring_scalar_butterfly;
use crate::butterflies::{Butterfly2, rotate_90};
use crate::store::BidirectionalStore;
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};

pub(crate) struct Butterfly8<T> {
    direction: FftDirection,
    root2: T,
    bf4: FastButterfly4<T>,
}

impl<T: FftTrigonometry + Float + 'static + Default> Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly8 {
            direction: fft_direction,
            root2: (0.5f64.sqrt()).as_(),
            bf4: FastButterfly4::new(fft_direction),
        }
    }
}

impl<T: FftSample> Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn run<S: BidirectionalStore<Complex<T>>>(&self, chunk: &mut S) {
        let bf2 = FastButterfly2::new(self.direction);
        let u0 = chunk[0];
        let u1 = chunk[1];
        let u2 = chunk[2];
        let u3 = chunk[3];
        let u4 = chunk[4];
        let u5 = chunk[5];
        let u6 = chunk[6];
        let u7 = chunk[7];

        // Radix-8 butterfly
        let (u0, u2, u4, u6) = self.bf4.butterfly4(u0, u2, u4, u6);
        let (u1, mut u3, mut u5, mut u7) = self.bf4.butterfly4(u1, u3, u5, u7);

        u3 = (rotate_90(u3, self.direction) + u3) * self.root2;
        u5 = rotate_90(u5, self.direction);
        u7 = (rotate_90(u7, self.direction) - u7) * self.root2;

        let (u0, u1) = bf2.butterfly2(u0, u1);
        let (u2, u3) = bf2.butterfly2(u2, u3);
        let (u4, u5) = bf2.butterfly2(u4, u5);
        let (u6, u7) = bf2.butterfly2(u6, u7);

        chunk[0] = u0;
        chunk[1] = u2;
        chunk[2] = u4;
        chunk[3] = u6;
        chunk[4] = u1;
        chunk[5] = u3;
        chunk[6] = u5;
        chunk[7] = u7;
    }
}

boring_scalar_butterfly!(Butterfly8, 8);

impl<T: FftSample> R2CFftExecutor<T> for Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.real_length(),
            output.len(),
            self.complex_length(),
        )?;

        for (input, complex) in input
            .as_chunks::<8>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<5>().0.iter_mut())
        {
            let u0 = input[0];
            let u1 = input[1];
            let u2 = input[2];
            let u3 = input[3];
            let u4 = input[4];
            let u5 = input[5];
            let u6 = input[6];
            let u7 = input[7];

            let e0 = u0 + u4;
            let e1 = u0 - u4;
            let e2 = u2 + u6;
            let e3 = u2 - u6;

            let o0 = u1 + u5;
            let o1 = u1 - u5;
            let o2 = u3 + u7;
            let o3 = u3 - u7;

            let re_x0 = e0 + e2;
            let re_x4 = e0 - e2;
            let re_x2r = e1;
            let re_x2i = -e3;

            let odd_f0r = o0 + o2;
            let odd_f2r = o0 - o2;
            let odd_f1r = o1;
            let odd_f1i = -o3;

            let x0 = re_x0 + odd_f0r;

            let x4 = re_x0 - odd_f0r;

            let tw2_f2i = -odd_f2r;

            let tw1_f1r = T::FRAC_1_SQRT_2 * (odd_f1r + odd_f1i);
            let tw1_f1i = T::FRAC_1_SQRT_2 * (odd_f1i - odd_f1r);

            let x1r = re_x2r + tw1_f1r;
            let x1i = re_x2i + tw1_f1i;

            let x5r = re_x2r - tw1_f1r;
            let x5i = re_x2i - tw1_f1i;

            let x2r_fixed = re_x4;
            let x2i_fixed = tw2_f2i;

            complex[0] = Complex::new(x0, T::zero());
            complex[1] = Complex::new(x1r, x1i);
            complex[2] = Complex::new(x2r_fixed, x2i_fixed);
            complex[3] = Complex::new(x5r, -x5i);
            complex[4] = Complex::new(x4, T::zero());
        }
        Ok(())
    }

    fn execute_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        _: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.real_length(),
            output.len(),
            self.complex_length(),
        )?;
        R2CFftExecutor::execute(self, input, output)
    }

    fn real_length(&self) -> usize {
        8
    }

    fn complex_length(&self) -> usize {
        5
    }

    fn complex_scratch_length(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    use crate::r2c::test_r2c_butterfly;

    test_r2c_butterfly!(test_r2c_butterfly8, f32, Butterfly8, 8, 1e-5);
    test_butterfly!(test_butterfly8, f32, Butterfly8, 8, 1e-5);
    test_oof_butterfly!(test_oof_butterfly8, f32, Butterfly8, 8, 1e-5);
}
