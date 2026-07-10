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
use crate::butterflies::Butterfly3;
use crate::butterflies::util::boring_scalar_butterfly;
use crate::complex_fma::c_mul_fast;
use crate::mla::fmla;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::AsPrimitive;

pub(crate) struct Butterfly9<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle4: Complex<T>,
    bf3: Butterfly3<T>,
}

#[allow(unused)]
impl<T: FftSample> Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly9 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 9, fft_direction),
            twiddle2: compute_twiddle(2, 9, fft_direction),
            twiddle4: compute_twiddle(4, 9, fft_direction),
            bf3: Butterfly3::new(fft_direction),
        }
    }
}

impl<T: FftSample> Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn run<S: BidirectionalStore<Complex<T>>>(&self, chunk: &mut S) {
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

        let [u0, u3, u6] = self.bf3.exec(&[u0, u3, u6]);
        let [u1, mut u4, mut u7] = self.bf3.exec(&[u1, u4, u7]);
        let [u2, mut u5, mut u8] = self.bf3.exec(&[u2, u5, u8]);

        u4 = c_mul_fast(u4, self.twiddle1);
        u7 = c_mul_fast(u7, self.twiddle2);
        u5 = c_mul_fast(u5, self.twiddle2);
        u8 = c_mul_fast(u8, self.twiddle4);

        let [zu0, zu3, zu6] = self.bf3.exec(&[u0, u1, u2]);
        let [zu1, zu4, zu7] = self.bf3.exec(&[u3, u4, u5]);
        let [zu2, zu5, zu8] = self.bf3.exec(&[u6, u7, u8]);

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
}

boring_scalar_butterfly!(Butterfly9, 9);

pub(crate) struct RfftButterfly9<T> {
    direction: FftDirection,
    twiddle1: T,
    twiddle2: T,
    twiddle3: T,
    twiddle4: T,
    d6: T,
    d7: T,
    d8: T,
}

impl<T: FftSample> RfftButterfly9<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        let one_over_3 = (1f64 / 3f64).as_();
        let twiddle1 = compute_twiddle(2, 9, fft_direction);
        let twiddle2 = compute_twiddle(4, 9, fft_direction);
        let twiddle3 = compute_twiddle(6, 9, fft_direction);
        let twiddle4 = compute_twiddle(8, 9, fft_direction);
        let h0 = twiddle4.re + twiddle1.re; // cos(φ) + cos(2φ)
        let d6 = (2.0.as_() * twiddle4.re - twiddle1.re - twiddle2.re) * one_over_3;
        let d7 = (-twiddle4.re + 2.0.as_() * twiddle1.re - twiddle2.re) * one_over_3;
        let d8 = (-h0 + 2.0.as_() * twiddle2.re) * one_over_3;

        RfftButterfly9 {
            direction: fft_direction,
            twiddle1: twiddle1.im,
            twiddle2: twiddle2.im,
            twiddle3: twiddle3.im,
            twiddle4: twiddle4.im,
            d6,
            d7,
            d8,
        }
    }
}

impl<T: FftSample> R2CFftExecutor<T> for RfftButterfly9<T>
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
            .as_chunks::<9>()
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
            let u8 = input[8];

            // Radix-9 butterfly, R2C
            let t1 = u1 + u8;
            let t2 = u2 + u7;
            let t3 = u3 + u6;
            let t4 = u4 + u5;
            let t5 = u4 - u5;
            let t6 = u3 - u6;
            let t7 = u2 - u7;
            let t8 = u1 - u8;

            // DC
            let r0 = t1 + t2 + t4;
            let z0 = t8 - t7 + t5;
            let y0 = u0 + r0 + t3;

            let y3 = fmla(-T::HALF, r0, u0 + t3);

            let r1 = t1 - t4;
            let r2 = t2 - t4;
            let r3 = -t1 + t2;

            let m2 = self.d6 * r1;
            let m3 = self.d7 * r2;
            let m4 = self.d8 * r3;

            let re_base = fmla(-T::HALF, t3, u0);
            let y1 = re_base + m2 + m3;
            let y2 = re_base - m2 + m4;
            let y4 = re_base - m3 - m4;

            let y5 = -fmla(
                self.twiddle4,
                t8,
                fmla(
                    -self.twiddle1,
                    t7,
                    fmla(self.twiddle3, t6, -self.twiddle2 * t5),
                ),
            ); // X[1].im

            let y6 = -fmla(
                -self.twiddle1,
                t8,
                fmla(
                    -self.twiddle2,
                    t7,
                    fmla(-self.twiddle3, t6, -self.twiddle4 * t5),
                ),
            ); // X[2].im

            let y7 = -self.twiddle3 * z0; // X[3].im: -sin(3φ)*(t8-t7+t5)

            let y8 = fmla(
                self.twiddle2,
                t8,
                fmla(
                    self.twiddle4,
                    t7,
                    fmla(-self.twiddle3, t6, -self.twiddle1 * t5),
                ),
            );

            complex[0] = Complex::new(y0, T::zero());
            complex[1] = Complex::new(y1, y5);
            complex[2] = Complex::new(y2, y6);
            complex[3] = Complex::new(y3, y7);
            complex[4] = Complex::new(y4, y8);
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
        9
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

    test_r2c_butterfly!(test_r2c_butterfly9, f32, RfftButterfly9, 9, 1e-5);
    test_r2c_butterfly!(test_r2c_butterfly9_f64, f64, RfftButterfly9, 9, 1e-8);
    test_butterfly!(test_butterfly9, f32, Butterfly9, 9, 1e-5);
    test_oof_butterfly!(test_oof_butterfly9, f32, Butterfly9, 9, 1e-5);
}
