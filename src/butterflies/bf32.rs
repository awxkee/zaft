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

use crate::butterflies::fast_bf16::FastButterfly16;
use crate::butterflies::rotate_90;
use crate::complex_fma::{c_mul_fast, c_mul_fast_conj};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Butterfly32<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    twiddle7: Complex<T>,
    bf16: FastButterfly16<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> Butterfly32<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly32 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 32, fft_direction),
            twiddle2: compute_twiddle(2, 32, fft_direction),
            twiddle3: compute_twiddle(3, 32, fft_direction),
            twiddle4: compute_twiddle(4, 32, fft_direction),
            twiddle5: compute_twiddle(5, 32, fft_direction),
            twiddle6: compute_twiddle(6, 32, fft_direction),
            twiddle7: compute_twiddle(7, 32, fft_direction),
            bf16: FastButterfly16::new(fft_direction),
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
> FftExecutor<T> for Butterfly32<T>
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

        for chunk in in_place.chunks_exact_mut(32) {
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

            let u27 = chunk[27];
            let u28 = chunk[28];

            let u29 = chunk[29];
            let u30 = chunk[30];

            let u31 = chunk[31];

            let s_evens = self.bf16.exec(
                u0, u2, u4, u6, u8, u10, u12, u14, u16, u18, u20, u22, u24, u26, u28, u30,
            );
            let mut odds1 = self.bf16.bf8.exec(u1, u5, u9, u13, u17, u21, u25, u29);
            let mut odds2 = self.bf16.bf8.exec(u31, u3, u7, u11, u15, u19, u23, u27);

            odds1.1 = c_mul_fast(odds1.1, self.twiddle1);
            odds2.1 = c_mul_fast_conj(odds2.1, self.twiddle1);

            odds1.2 = c_mul_fast(odds1.2, self.twiddle2);
            odds2.2 = c_mul_fast_conj(odds2.2, self.twiddle2);

            odds1.3 = c_mul_fast(odds1.3, self.twiddle3);
            odds2.3 = c_mul_fast_conj(odds2.3, self.twiddle3);

            odds1.4 = c_mul_fast(odds1.4, self.twiddle4);
            odds2.4 = c_mul_fast_conj(odds2.4, self.twiddle4);

            odds1.5 = c_mul_fast(odds1.5, self.twiddle5);
            odds2.5 = c_mul_fast_conj(odds2.5, self.twiddle5);

            odds1.6 = c_mul_fast(odds1.6, self.twiddle6);
            odds2.6 = c_mul_fast_conj(odds2.6, self.twiddle6);

            odds1.7 = c_mul_fast(odds1.7, self.twiddle7);
            odds2.7 = c_mul_fast_conj(odds2.7, self.twiddle7);

            let mut q0 = self.bf16.bf2.butterfly2(odds1.0, odds2.0);
            let mut q1 = self.bf16.bf2.butterfly2(odds1.1, odds2.1);
            let mut q2 = self.bf16.bf2.butterfly2(odds1.2, odds2.2);
            let mut q3 = self.bf16.bf2.butterfly2(odds1.3, odds2.3);
            let mut q4 = self.bf16.bf2.butterfly2(odds1.4, odds2.4);
            let mut q5 = self.bf16.bf2.butterfly2(odds1.5, odds2.5);
            let mut q6 = self.bf16.bf2.butterfly2(odds1.6, odds2.6);
            let mut q7 = self.bf16.bf2.butterfly2(odds1.7, odds2.7);

            q0.1 = rotate_90(q0.1, self.direction);
            q1.1 = rotate_90(q1.1, self.direction);
            q2.1 = rotate_90(q2.1, self.direction);
            q3.1 = rotate_90(q3.1, self.direction);
            q4.1 = rotate_90(q4.1, self.direction);
            q5.1 = rotate_90(q5.1, self.direction);
            q6.1 = rotate_90(q6.1, self.direction);
            q7.1 = rotate_90(q7.1, self.direction);

            chunk[0] = s_evens.0 + q0.0;
            chunk[1] = s_evens.1 + q1.0;
            chunk[2] = s_evens.2 + q2.0;
            chunk[3] = s_evens.3 + q3.0;
            chunk[4] = s_evens.4 + q4.0;
            chunk[5] = s_evens.5 + q5.0;
            chunk[6] = s_evens.6 + q6.0;
            chunk[7] = s_evens.7 + q7.0;
            chunk[8] = s_evens.8 + q0.1;
            chunk[9] = s_evens.9 + q1.1;
            chunk[10] = s_evens.10 + q2.1;
            chunk[11] = s_evens.11 + q3.1;
            chunk[12] = s_evens.12 + q4.1;
            chunk[13] = s_evens.13 + q5.1;
            chunk[14] = s_evens.14 + q6.1;
            chunk[15] = s_evens.15 + q7.1;

            chunk[16] = s_evens.0 - q0.0;
            chunk[17] = s_evens.1 - q1.0;
            chunk[18] = s_evens.2 - q2.0;
            chunk[19] = s_evens.3 - q3.0;
            chunk[20] = s_evens.4 - q4.0;
            chunk[21] = s_evens.5 - q5.0;
            chunk[22] = s_evens.6 - q6.0;
            chunk[23] = s_evens.7 - q7.0;
            chunk[24] = s_evens.8 - q0.1;
            chunk[25] = s_evens.9 - q1.1;
            chunk[26] = s_evens.10 - q2.1;
            chunk[27] = s_evens.11 - q3.1;
            chunk[28] = s_evens.12 - q4.1;
            chunk[29] = s_evens.13 - q5.1;
            chunk[30] = s_evens.14 - q6.1;
            chunk[31] = s_evens.15 - q7.1;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::Dft;
    use rand::Rng;

    #[test]
    fn test_butterfly32() {
        for i in 1..4 {
            let size = 32usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let reference_forward = Dft::new(32, FftDirection::Forward).unwrap();

            let mut ref_src = src.to_vec();
            reference_forward.execute(&mut ref_src).unwrap();

            let radix_forward = Butterfly32::new(FftDirection::Forward);
            let radix_inverse = Butterfly32::new(FftDirection::Inverse);
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

            input = input.iter().map(|&x| x * (1.0 / 32f32)).collect();

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
