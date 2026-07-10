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
use crate::butterflies::fast_bf8::FastButterfly8;
use crate::butterflies::rotate_90;
use crate::butterflies::short_butterflies::{FastButterfly2, FastButterfly4};
use crate::butterflies::util::boring_scalar_butterfly;
use crate::complex_fma::{c_mul_fast, c_mul_fast_conj};
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::AsPrimitive;

pub(crate) struct Butterfly16<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    bf8: FastButterfly8<T>,
    bf4: FastButterfly4<T>,
}

#[allow(unused)]
impl<T: FftSample> Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly16 {
            direction: fft_direction,
            bf8: FastButterfly8::new(fft_direction),
            bf4: FastButterfly4::new(fft_direction),
            twiddle1: compute_twiddle(1, 16, fft_direction),
            twiddle2: compute_twiddle(2, 16, fft_direction),
            twiddle3: compute_twiddle(3, 16, fft_direction),
        }
    }
}

impl<T: FftSample> Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn run<S: BidirectionalStore<Complex<T>>>(&self, chunk: &mut S) {
        let bf2 = FastButterfly2::new(self.direction);

        let evens = self.bf8.exec(
            chunk[0], chunk[2], chunk[4], chunk[6], chunk[8], chunk[10], chunk[12], chunk[14],
        );

        let odds_1 = self.bf4.butterfly4(chunk[1], chunk[5], chunk[9], chunk[13]);
        let odds_2 = self
            .bf4
            .butterfly4(chunk[15], chunk[3], chunk[7], chunk[11]);

        // lane 0 — no twiddle
        let (o0a, o0b) = bf2.butterfly2(odds_1.0, odds_2.0);
        let o0b = rotate_90(o0b, self.direction);
        chunk[0] = evens.0 + o0a;
        chunk[8] = evens.0 - o0a;
        chunk[4] = evens.4 + o0b;
        chunk[12] = evens.4 - o0b;

        // lane 1
        let t1a = c_mul_fast(odds_1.1, self.twiddle1);
        let t1b = c_mul_fast_conj(odds_2.1, self.twiddle1);
        let (o1a, o1b) = bf2.butterfly2(t1a, t1b);
        let o1b = rotate_90(o1b, self.direction);
        chunk[1] = evens.1 + o1a;
        chunk[9] = evens.1 - o1a;
        chunk[5] = evens.5 + o1b;
        chunk[13] = evens.5 - o1b;

        // lane 2
        let t2a = c_mul_fast(odds_1.2, self.twiddle2);
        let t2b = c_mul_fast_conj(odds_2.2, self.twiddle2);
        let (o2a, o2b) = bf2.butterfly2(t2a, t2b);
        let o2b = rotate_90(o2b, self.direction);
        chunk[2] = evens.2 + o2a;
        chunk[10] = evens.2 - o2a;
        chunk[6] = evens.6 + o2b;
        chunk[14] = evens.6 - o2b;

        // lane 3
        let t3a = c_mul_fast(odds_1.3, self.twiddle3);
        let t3b = c_mul_fast_conj(odds_2.3, self.twiddle3);
        let (o3a, o3b) = bf2.butterfly2(t3a, t3b);
        let o3b = rotate_90(o3b, self.direction);
        chunk[3] = evens.3 + o3a;
        chunk[11] = evens.3 - o3a;
        chunk[7] = evens.7 + o3b;
        chunk[15] = evens.7 - o3b;
    }
}

boring_scalar_butterfly!(Butterfly16, 16);

impl<T: FftSample> R2CFftExecutor<T> for Butterfly16<T>
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

        let bf2 = FastButterfly2::new(self.direction);

        for (dst, src) in output
            .as_chunks_mut::<9>()
            .0
            .iter_mut()
            .zip(input.as_chunks::<16>().0.iter())
        {
            let u0 = Complex::new(src[0], T::zero());
            let u1 = Complex::new(src[1], T::zero());
            let u2 = Complex::new(src[2], T::zero());
            let u3 = Complex::new(src[3], T::zero());

            let u4 = Complex::new(src[4], T::zero());
            let u5 = Complex::new(src[5], T::zero());
            let u6 = Complex::new(src[6], T::zero());
            let u7 = Complex::new(src[7], T::zero());

            let u8 = Complex::new(src[8], T::zero());
            let u9 = Complex::new(src[9], T::zero());
            let u10 = Complex::new(src[10], T::zero());
            let u11 = Complex::new(src[11], T::zero());
            let u12 = Complex::new(src[12], T::zero());

            let u13 = Complex::new(src[13], T::zero());
            let u14 = Complex::new(src[14], T::zero());
            let u15 = Complex::new(src[15], T::zero());

            let evens = self.bf8.exec(u0, u2, u4, u6, u8, u10, u12, u14);

            let mut odds_1 = self.bf4.butterfly4(u1, u5, u9, u13);
            let mut odds_2 = self.bf4.butterfly4(u15, u3, u7, u11);

            odds_1.1 = c_mul_fast(odds_1.1, self.twiddle1);
            odds_2.1 = c_mul_fast_conj(odds_2.1, self.twiddle1);

            odds_1.2 = c_mul_fast(odds_1.2, self.twiddle2);
            odds_2.2 = c_mul_fast_conj(odds_2.2, self.twiddle2);

            odds_1.3 = c_mul_fast(odds_1.3, self.twiddle3);
            odds_2.3 = c_mul_fast_conj(odds_2.3, self.twiddle3);

            // step 4: cross FFTs
            let (o01, o02) = bf2.butterfly2(odds_1.0, odds_2.0);
            odds_1.0 = o01;
            odds_2.0 = o02;

            let (o03, o04) = bf2.butterfly2(odds_1.1, odds_2.1);
            odds_1.1 = o03;
            odds_2.1 = o04;
            let (o05, o06) = bf2.butterfly2(odds_1.2, odds_2.2);
            odds_1.2 = o05;
            odds_2.2 = o06;
            let (o07, o08) = bf2.butterfly2(odds_1.3, odds_2.3);
            odds_1.3 = o07;
            odds_2.3 = o08;

            // apply the butterfly 4 twiddle factor, which is just a rotation
            odds_2.0 = rotate_90(odds_2.0, self.direction);
            odds_2.1 = rotate_90(odds_2.1, self.direction);
            odds_2.2 = rotate_90(odds_2.2, self.direction);
            odds_2.3 = rotate_90(odds_2.3, self.direction);

            dst[0] = evens.0 + odds_1.0;
            dst[1] = evens.1 + odds_1.1;
            dst[2] = evens.2 + odds_1.2;
            dst[3] = evens.3 + odds_1.3;
            dst[4] = evens.4 + odds_2.0;
            dst[5] = evens.5 + odds_2.1;
            dst[6] = evens.6 + odds_2.2;
            dst[7] = evens.7 + odds_2.3;
            dst[8] = evens.0 - odds_1.0;
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

    fn complex_length(&self) -> usize {
        9
    }

    fn real_length(&self) -> usize {
        16
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

    test_r2c_butterfly!(test_r2c_butterfly16, f32, Butterfly16, 16, 1e-5);
    test_butterfly!(test_butterfly16, f32, Butterfly16, 16, 1e-5);
    test_oof_butterfly!(test_oof_butterfly16, f32, Butterfly16, 16, 1e-5);
}
