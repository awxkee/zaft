/*
 * // Copyright (c) Radzivon Bartoshyk 6/2025. All rights reserved.
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
use crate::traits::FftTrigonometry;
use crate::util::{digit_reverse_indices, permute_inplace};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Radix2<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    direction: FftDirection,
}

pub(crate) trait Radix2Twiddles {
    fn make_twiddles(
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized;
}

fn radix2_floating_twiddles<
    T: Default + Float + FftTrigonometry + 'static + MulAdd<T, Output = T>,
>(
    size: usize,
    fft_direction: FftDirection,
) -> Result<Vec<Complex<T>>, ZaftError>
where
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    let mut twiddles = Vec::new();
    twiddles
        .try_reserve_exact(size - 1)
        .map_err(|_| ZaftError::OutOfMemory(size - 1))?;

    let mut len = 2;
    while len <= size {
        let half = len / 2;

        let theta = -2.0.as_() / len.as_();
        let (t_sin, t_cos) = theta.sincos_pi();

        let mut w = Complex::new(1.0.as_(), 0.0.as_());

        let twiddle = match fft_direction {
            FftDirection::Forward => Complex {
                re: t_cos,
                im: t_sin,
            },
            FftDirection::Inverse => (Complex {
                re: t_cos,
                im: t_sin,
            })
            .conj(),
        };

        for _ in 0..half {
            twiddles.push(w);
            w = w * twiddle;
        }
        len *= 2;
    }

    Ok(twiddles)
}

impl Radix2Twiddles for f64 {
    fn make_twiddles(
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<f64>>, ZaftError>
    where
        Self: Sized,
    {
        radix2_floating_twiddles(size, fft_direction)
    }
}

impl Radix2Twiddles for f32 {
    fn make_twiddles(
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<f32>>, ZaftError>
    where
        Self: Sized,
    {
        radix2_floating_twiddles(size, fft_direction)
    }
}

#[allow(unused)]
impl<T: Default + Clone + Radix2Twiddles> Radix2<T> {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Radix2<T>, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");

        let twiddles = T::make_twiddles(size, fft_direction)?;

        // Bit-reversal permutation
        let rev = digit_reverse_indices(size, 2)?;

        Ok(Radix2 {
            permutations: rev,
            execution_length: size,
            twiddles,
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
        + MulAdd<T, Output = T>,
> FftExecutor<T> for Radix2<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        permute_inplace(in_place, &self.permutations);

        let mut len = 2;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();
            while len <= self.execution_length {
                let half = len / 2;
                for data in in_place.chunks_exact_mut(len) {
                    for j in 0..half {
                        let u = *data.get_unchecked(j);
                        let tw = *m_twiddles.get_unchecked(j);
                        let t = c_mul_fast(tw, *data.get_unchecked(j + half));
                        *data.get_unchecked_mut(j) = u + t;
                        *data.get_unchecked_mut(j + half) = u - t;
                    }
                }

                len *= 2;
                m_twiddles = &m_twiddles[half..];
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
    fn test_radix2() {
        for i in 1..14 {
            let size = 2usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Radix2::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = Radix2::new(size, FftDirection::Inverse).unwrap();
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
