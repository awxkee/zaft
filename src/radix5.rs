/*
 * // Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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
use crate::complex_fma::c_mul_fast;
use crate::mla::fmla;
use crate::traits::FftTrigonometry;
use crate::util::{
    compute_twiddle, digit_reverse_indices, is_power_of_five, permute_inplace,
    radixn_floating_twiddles, radixn_floating_twiddles_from_base,
};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(dead_code)]
pub(crate) struct Radix5<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    direction: FftDirection,
}

pub(crate) trait Radix5Twiddles {
    fn make_twiddles(
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized;

    #[allow(unused)]
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized;
}

impl Radix5Twiddles for f64 {
    fn make_twiddles(
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<f64>>, ZaftError> {
        radixn_floating_twiddles::<f64, 5>(size, fft_direction)
    }

    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f64, 5>(base, size, fft_direction)
    }
}

impl Radix5Twiddles for f32 {
    fn make_twiddles(
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<f32>>, ZaftError> {
        radixn_floating_twiddles::<f32, 5>(size, fft_direction)
    }

    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f32, 5>(base, size, fft_direction)
    }
}

#[allow(dead_code)]
impl<T: Default + Clone + Radix5Twiddles + 'static + Copy + FftTrigonometry + Float> Radix5<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Radix5<T>, ZaftError> {
        assert!(
            is_power_of_five(size as u64),
            "Input length must be a power of 5"
        );

        let twiddles = T::make_twiddles(size, fft_direction)?;
        let rev = digit_reverse_indices(size, 5)?;

        Ok(Radix5 {
            permutations: rev,
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 5, fft_direction),
            twiddle2: compute_twiddle(2, 5, fft_direction),
            direction: fft_direction,
        })
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
        + Default
        + FftTrigonometry,
> FftExecutor<T> for Radix5<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // Digit-reversal permutation
            permute_inplace(chunk, &self.permutations);

            let mut len = 5;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len <= self.execution_length {
                    let fifth = len / 5;

                    for data in chunk.chunks_exact_mut(len) {
                        for j in 0..fifth {
                            let u0 = *data.get_unchecked(j);
                            let u1 = c_mul_fast(
                                *data.get_unchecked(j + fifth),
                                *m_twiddles.get_unchecked(4 * j),
                            );
                            let u2 = c_mul_fast(
                                *data.get_unchecked(j + 2 * fifth),
                                *m_twiddles.get_unchecked(4 * j + 1),
                            );
                            let u3 = c_mul_fast(
                                *data.get_unchecked(j + 3 * fifth),
                                *m_twiddles.get_unchecked(4 * j + 2),
                            );
                            let u4 = c_mul_fast(
                                *data.get_unchecked(j + 4 * fifth),
                                *m_twiddles.get_unchecked(4 * j + 3),
                            );

                            // Radix-5 butterfly

                            let x14p = u1 + u4;
                            let x14n = u1 - u4;
                            let x23p = u2 + u3;
                            let x23n = u2 - u3;
                            let y0 = u0 + x14p + x23p;

                            let b14re_a = fmla(
                                self.twiddle2.re,
                                x23p.re,
                                fmla(self.twiddle1.re, x14p.re, u0.re),
                            );
                            let b14re_b =
                                fmla(self.twiddle1.im, x14n.im, self.twiddle2.im * x23n.im);
                            let b23re_a = fmla(
                                self.twiddle1.re,
                                x23p.re,
                                fmla(self.twiddle2.re, x14p.re, u0.re),
                            );
                            let b23re_b =
                                fmla(self.twiddle2.im, x14n.im, -self.twiddle1.im * x23n.im);

                            let b14im_a = fmla(
                                self.twiddle2.re,
                                x23p.im,
                                fmla(self.twiddle1.re, x14p.im, u0.im),
                            );
                            let b14im_b =
                                fmla(self.twiddle1.im, x14n.re, self.twiddle2.im * x23n.re);
                            let b23im_a = fmla(
                                self.twiddle1.re,
                                x23p.im,
                                fmla(self.twiddle2.re, x14p.im, u0.im),
                            );
                            let b23im_b =
                                fmla(self.twiddle2.im, x14n.re, -self.twiddle1.im * x23n.re);

                            let y1 = Complex {
                                re: b14re_a - b14re_b,
                                im: b14im_a + b14im_b,
                            };
                            let y2 = Complex {
                                re: b23re_a - b23re_b,
                                im: b23im_a + b23im_b,
                            };
                            let y3 = Complex {
                                re: b23re_a + b23re_b,
                                im: b23im_a - b23im_b,
                            };
                            let y4 = Complex {
                                re: b14re_a + b14re_b,
                                im: b14im_a - b14im_b,
                            };

                            *data.get_unchecked_mut(j) = y0;
                            *data.get_unchecked_mut(j + fifth) = y1;
                            *data.get_unchecked_mut(j + 2 * fifth) = y2;
                            *data.get_unchecked_mut(j + 3 * fifth) = y3;
                            *data.get_unchecked_mut(j + 4 * fifth) = y4;
                        }
                    }

                    m_twiddles = &m_twiddles[fifth * 4..];
                    len *= 5;
                }
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_radix4() {
        for i in 1..7 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Radix5::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = Radix5::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input
                .iter()
                .map(|&x| x * (1.0 / input.len() as f32))
                .collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }
}
