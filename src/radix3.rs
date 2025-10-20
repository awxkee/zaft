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
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::mla::fmla;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{
    bitreversed_transpose, compute_logarithm, compute_twiddle, is_power_of_three,
    radixn_floating_twiddles, radixn_floating_twiddles_from_base,
};
use crate::{FftDirection, FftExecutor, Zaft, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::fmt::Display;
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Radix3<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle: Complex<T>,
    direction: FftDirection,
    base_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    base_len: usize,
}

pub(crate) trait Radix3Twiddles {
    #[allow(unused)]
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

impl Radix3Twiddles for f64 {
    fn make_twiddles(
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<f64>>, ZaftError> {
        radixn_floating_twiddles::<f64, 3>(size, fft_direction)
    }

    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f64, 3>(base, size, fft_direction)
    }
}

impl Radix3Twiddles for f32 {
    fn make_twiddles(
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<f32>>, ZaftError> {
        radixn_floating_twiddles::<f32, 3>(size, fft_direction)
    }

    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f32, 3>(base, size, fft_direction)
    }
}

#[allow(unused)]
impl<
    T: Default
        + Clone
        + Radix3Twiddles
        + 'static
        + Copy
        + FftTrigonometry
        + Float
        + AlgorithmFactory<T>
        + MulAdd<T, Output = T>
        + SpectrumOpsFactory<T>
        + TransposeFactory<T>
        + Send
        + Sync
        + Display,
> Radix3<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Radix3<T>, ZaftError> {
        assert!(
            is_power_of_three(size as u64),
            "Input length must be power of 3"
        );

        let exponent = compute_logarithm::<3>(size)
            .unwrap_or_else(|| panic!("Radix3 length must be power of 3, but got {size}",));

        let base_fft = match exponent {
            0 => Zaft::strategy(1, fft_direction)?,
            1 => Zaft::strategy(3, fft_direction)?,
            2 => Zaft::strategy(9, fft_direction)?,
            _ => Zaft::strategy(27, fft_direction)?,
        };

        let base_len = base_fft.length();

        let twiddles = T::make_twiddles_with_base(base_len, size, fft_direction)?;

        Ok(Radix3 {
            execution_length: size,
            twiddles,
            twiddle: compute_twiddle::<T>(1, 3, fft_direction),
            direction: fft_direction,
            base_fft,
            base_len,
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
        + FftTrigonometry
        + std::fmt::Debug,
> FftExecutor<T> for Radix3<T>
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

        let mut scratch = try_vec![Complex::<T>::default(); self.execution_length];

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // Digit-reversal permutation
            bitreversed_transpose::<Complex<T>, 3>(self.base_len, chunk, &mut scratch);

            self.base_fft.execute(&mut scratch)?;

            let mut len = self.base_len;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 3;
                    let third = len / 3;

                    for data in scratch.chunks_exact_mut(len) {
                        for j in 0..third {
                            let u0 = *data.get_unchecked(j);
                            let u1 = c_mul_fast(
                                *data.get_unchecked(j + third),
                                *m_twiddles.get_unchecked(2 * j),
                            );
                            let u2 = c_mul_fast(
                                *data.get_unchecked(j + 2 * third),
                                *m_twiddles.get_unchecked(2 * j + 1),
                            );

                            // Radix-3 butterfly
                            let xp = u1 + u2;
                            let xn = u1 - u2;
                            let sum = u0 + xp;

                            let w_1 = Complex {
                                re: fmla(self.twiddle.re, xp.re, u0.re),
                                im: fmla(self.twiddle.re, xp.im, u0.im),
                            };
                            // let w_2 = ZComplex {
                            //     re: -self.twiddle.im * xn.im,
                            //     im: self.twiddle.im * xn.re,
                            // };

                            let y0 = sum;
                            let y1 = Complex {
                                re: fmla(-self.twiddle.im, xn.im, w_1.re),
                                im: fmla(self.twiddle.im, xn.re, w_1.im),
                            }; //w_1 + w_2;
                            let y2 = Complex {
                                re: fmla(self.twiddle.im, xn.im, w_1.re),
                                im: fmla(-self.twiddle.im, xn.re, w_1.im),
                            }; //w_1 - w_2;
                            // let y2 = w_1 - w_2;

                            *data.get_unchecked_mut(j) = y0;
                            *data.get_unchecked_mut(j + third) = y1;
                            *data.get_unchecked_mut(j + 2 * third) = y2;
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 2..];
                }
                chunk.copy_from_slice(&scratch);
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
    fn test_radix3() {
        for i in 1..9 {
            let size = 3usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Radix3::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = Radix3::new(size, FftDirection::Inverse).unwrap();
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
