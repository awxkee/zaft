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
use crate::butterflies::fast_bf7::FastButterfly7;
use crate::butterflies::short_butterflies::FastButterfly2;
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Butterfly14<T> {
    direction: FftDirection,
    phantom_data: PhantomData<T>,
    bf7: FastButterfly7<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> Butterfly14<T>
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
        + Default,
> FftExecutor<T> for Butterfly14<T>
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_butterfly14() {
        for i in 1..4 {
            let size = 14usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly14::new(FftDirection::Forward);
            let radix_inverse = Butterfly14::new(FftDirection::Inverse);

            let mut input2 = src.to_vec();
            radix_forward.execute(&mut input).unwrap();

            input
                .iter()
                .zip(input2.iter())
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

            input = input.iter().map(|&x| x * (1.0 / 14f32)).collect();

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
