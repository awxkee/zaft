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
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_butterfly9() {
        for i in 1..5 {
            let size = 9usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly9::new(FftDirection::Forward);
            let radix_inverse = Butterfly9::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 9f32)).collect();

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
