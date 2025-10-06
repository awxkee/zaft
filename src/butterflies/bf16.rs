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
use crate::butterflies::fast_bf8::FastButterfly8;
use crate::butterflies::rotate_90;
use crate::butterflies::short_butterflies::{FastButterfly2, FastButterfly4};
use crate::complex_fma::{c_mul_fast, c_mul_fast_conj};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Butterfly16<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    bf8: FastButterfly8<T>,
    bf4: FastButterfly4<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> Butterfly16<T>
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
> FftExecutor<T> for Butterfly16<T>
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

        for chunk in in_place.chunks_exact_mut(16) {
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
            let u15 = chunk[15];

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

            chunk[0] = evens.0 + odds_1.0;
            chunk[1] = evens.1 + odds_1.1;
            chunk[2] = evens.2 + odds_1.2;
            chunk[3] = evens.3 + odds_1.3;
            chunk[4] = evens.4 + odds_2.0;
            chunk[5] = evens.5 + odds_2.1;
            chunk[6] = evens.6 + odds_2.2;
            chunk[7] = evens.7 + odds_2.3;
            chunk[8] = evens.0 - odds_1.0;
            chunk[9] = evens.1 - odds_1.1;
            chunk[10] = evens.2 - odds_1.2;
            chunk[11] = evens.3 - odds_1.3;
            chunk[12] = evens.4 - odds_2.0;
            chunk[13] = evens.5 - odds_2.1;
            chunk[14] = evens.6 - odds_2.2;
            chunk[15] = evens.7 - odds_2.3;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_butterfly16() {
        for i in 1..5 {
            let size = 16usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly16::new(FftDirection::Forward);
            let radix_inverse = Butterfly16::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 16f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }
}
