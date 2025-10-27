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
use crate::butterflies::rotate_90;
use crate::butterflies::short_butterflies::{FastButterfly2, FastButterfly4};
use crate::traits::FftTrigonometry;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Butterfly8<T> {
    direction: FftDirection,
    root2: T,
    bf4: FastButterfly4<T>,
}

#[allow(unused)]
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
> FftExecutor<T> for Butterfly8<T>
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

        for chunk in in_place.chunks_exact_mut(8) {
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
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        8
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
> FftExecutorOutOfPlace<T> for Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if src.len() % self.length() != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % self.length() != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        let bf2 = FastButterfly2::new(self.direction);

        for (dst, src) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
            let u0 = src[0];
            let u1 = src[1];
            let u2 = src[2];
            let u3 = src[3];
            let u4 = src[4];
            let u5 = src[5];
            let u6 = src[6];
            let u7 = src[7];

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

            dst[0] = u0;
            dst[1] = u2;
            dst[2] = u4;
            dst[3] = u6;
            dst[4] = u1;
            dst[5] = u3;
            dst[6] = u5;
            dst[7] = u7;
        }
        Ok(())
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
        + Sync
        + Send,
> CompositeFftExecutor<T> for Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<T> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::Dft;
    use rand::Rng;

    #[test]
    fn test_butterfly8() {
        for i in 1..5 {
            let size = 8usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly8::new(FftDirection::Forward);
            let radix_inverse = Butterfly8::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 8f32)).collect();

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

    #[test]
    fn test_butterfly8_out_of_place_f64() {
        for i in 1..4 {
            let size = 8usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut out_of_place = vec![Complex::<f64>::default(); size];
            let mut ref_input = input.to_vec();
            let radix_forward = Butterfly8::new(FftDirection::Forward);
            let radix_inverse = Butterfly8::new(FftDirection::Inverse);

            let reference_dft = Dft::new(8, FftDirection::Forward).unwrap();
            reference_dft.execute(&mut ref_input).unwrap();

            radix_forward
                .execute_out_of_place(&input, &mut out_of_place)
                .unwrap();

            out_of_place
                .iter()
                .zip(ref_input.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse
                .execute_out_of_place(&out_of_place, &mut input)
                .unwrap();

            input = input.iter().map(|&x| x * (1.0 / 8f64)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }
}
