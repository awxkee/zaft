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
use crate::complex_fma::c_mul_fast;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Butterfly27<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    twiddle7: Complex<T>,
    twiddle8: Complex<T>,
    twiddle9: Complex<T>,
    twiddle10: Complex<T>,
    twiddle11: Complex<T>,
    twiddle12: Complex<T>,
    bf9: FastButterfly9<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> Butterfly27<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly27 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 27, fft_direction),
            twiddle2: compute_twiddle(2, 27, fft_direction),
            twiddle3: compute_twiddle(3, 27, fft_direction),
            twiddle4: compute_twiddle(4, 27, fft_direction),
            twiddle5: compute_twiddle(5, 27, fft_direction),
            twiddle6: compute_twiddle(6, 27, fft_direction),
            twiddle7: compute_twiddle(7, 27, fft_direction),
            twiddle8: compute_twiddle(8, 27, fft_direction),
            twiddle9: compute_twiddle(10, 27, fft_direction),
            twiddle10: compute_twiddle(12, 27, fft_direction),
            twiddle11: compute_twiddle(14, 27, fft_direction),
            twiddle12: compute_twiddle(16, 27, fft_direction),
            bf9: FastButterfly9::new(fft_direction),
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
> FftExecutor<T> for Butterfly27<T>
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

        for chunk in in_place.chunks_exact_mut(27) {
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
            let u16 = chunk[16];

            let u17 = chunk[17];
            let u18 = chunk[18];

            let u19 = chunk[19];
            let u20 = chunk[20];

            let u21 = chunk[21];
            let u22 = chunk[22];

            let u23 = chunk[23];
            let u24 = chunk[24];

            let u25 = chunk[25];
            let u26 = chunk[26];

            let s0 = self.bf9.exec(u0, u3, u6, u9, u12, u15, u18, u21, u24);
            let mut s1 = self.bf9.exec(u1, u4, u7, u10, u13, u16, u19, u22, u25);
            let mut s2 = self.bf9.exec(u2, u5, u8, u11, u14, u17, u20, u23, u26);

            s1.1 = c_mul_fast(s1.1, self.twiddle1);
            s1.2 = c_mul_fast(s1.2, self.twiddle2);
            s1.3 = c_mul_fast(s1.3, self.twiddle3);
            s1.4 = c_mul_fast(s1.4, self.twiddle4);
            s1.5 = c_mul_fast(s1.5, self.twiddle5);
            s1.6 = c_mul_fast(s1.6, self.twiddle6);
            s1.7 = c_mul_fast(s1.7, self.twiddle7);
            s1.8 = c_mul_fast(s1.8, self.twiddle8);
            s2.1 = c_mul_fast(s2.1, self.twiddle2);
            s2.2 = c_mul_fast(s2.2, self.twiddle4);
            s2.3 = c_mul_fast(s2.3, self.twiddle6);
            s2.4 = c_mul_fast(s2.4, self.twiddle8);
            s2.5 = c_mul_fast(s2.5, self.twiddle9);
            s2.6 = c_mul_fast(s2.6, self.twiddle10);
            s2.7 = c_mul_fast(s2.7, self.twiddle11);
            s2.8 = c_mul_fast(s2.8, self.twiddle12);

            let z0 = self.bf9.bf3.butterfly3(s0.0, s1.0, s2.0);
            let z1 = self.bf9.bf3.butterfly3(s0.1, s1.1, s2.1);
            let z2 = self.bf9.bf3.butterfly3(s0.2, s1.2, s2.2);
            let z3 = self.bf9.bf3.butterfly3(s0.3, s1.3, s2.3);
            let z4 = self.bf9.bf3.butterfly3(s0.4, s1.4, s2.4);
            let z5 = self.bf9.bf3.butterfly3(s0.5, s1.5, s2.5);
            let z6 = self.bf9.bf3.butterfly3(s0.6, s1.6, s2.6);
            let z7 = self.bf9.bf3.butterfly3(s0.7, s1.7, s2.7);
            let z8 = self.bf9.bf3.butterfly3(s0.8, s1.8, s2.8);

            chunk[0] = z0.0;
            chunk[1] = z1.0;
            chunk[2] = z2.0;
            chunk[3] = z3.0;
            chunk[4] = z4.0;
            chunk[5] = z5.0;
            chunk[6] = z6.0;
            chunk[7] = z7.0;
            chunk[8] = z8.0;

            chunk[9] = z0.1;
            chunk[10] = z1.1;
            chunk[11] = z2.1;
            chunk[12] = z3.1;
            chunk[13] = z4.1;
            chunk[14] = z5.1;
            chunk[15] = z6.1;
            chunk[16] = z7.1;
            chunk[17] = z8.1;

            chunk[18] = z0.2;
            chunk[19] = z1.2;
            chunk[20] = z2.2;
            chunk[21] = z3.2;
            chunk[22] = z4.2;
            chunk[23] = z5.2;
            chunk[24] = z6.2;
            chunk[25] = z7.2;
            chunk[26] = z8.2;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        27
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::Dft;
    use rand::Rng;

    #[test]
    fn test_butterfly27() {
        for i in 1..4 {
            let size = 27usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly27::new(FftDirection::Forward);

            let dft_ref = Dft::new(27, FftDirection::Forward).unwrap();
            let mut ref_src = src.to_vec();
            dft_ref.execute(&mut ref_src).unwrap();

            let radix_inverse = Butterfly27::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();

            input
                .iter()
                .zip(ref_src.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-5,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-5,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 27f32)).collect();

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
