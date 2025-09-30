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
use crate::traits::FftTrigonometry;
use crate::util::{compute_twiddle, digit_reverse_indices, permute_inplace};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Radix4<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    direction: FftDirection,
}

pub(crate) trait Radix4Twiddles {
    fn make_twiddles(
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized;
}

fn radix4_floating_twiddles<
    T: Default + Float + FftTrigonometry + 'static + MulAdd<T, Output = T>,
>(
    size: usize,
    fft_direction: FftDirection,
) -> Result<Vec<Complex<T>>, ZaftError>
where
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    // radix-4 needs fewer stages: log4(size) instead of log2(size)
    let mut len = 4;

    let mut twiddles = Vec::new();
    twiddles
        .try_reserve_exact(size - 1)
        .map_err(|_| ZaftError::OutOfMemory(size - 1))?;

    while len <= size {
        let quarter = len / 4;
        for k in 0..quarter {
            for i in 1..4 {
                let w1 = compute_twiddle::<T>(k * i, len, fft_direction);
                twiddles.push(w1); // W_N^k
            }
        }

        len *= 4;
    }

    Ok(twiddles)
}

impl Radix4Twiddles for f64 {
    fn make_twiddles(
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<f64>>, ZaftError> {
        radix4_floating_twiddles(size, fft_direction)
    }
}

impl Radix4Twiddles for f32 {
    fn make_twiddles(
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<f32>>, ZaftError> {
        radix4_floating_twiddles(size, fft_direction)
    }
}

#[allow(unused)]
impl<T: Default + Clone + Radix4Twiddles> Radix4<T> {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Radix4<T>, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");
        assert_eq!(size.trailing_zeros() % 2, 0, "Radix-4 requires power of 4");

        let twiddles = T::make_twiddles(size, fft_direction)?;
        let rev = digit_reverse_indices(size, 4)?;

        Ok(Radix4 {
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
> FftExecutor<T> for Radix4<T>
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

        // bit reversal first
        permute_inplace(in_place, &self.permutations);

        let mut len = 4;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            let t3_twiddle = match self.direction {
                FftDirection::Forward => Complex::new(T::zero(), -T::one()),
                FftDirection::Inverse => Complex::new(T::zero(), T::one()),
            };

            while len <= self.execution_length {
                let quarter = len / 4;

                for data in in_place.chunks_exact_mut(len) {
                    for j in 0..quarter {
                        let a = *data.get_unchecked(j);
                        let b = c_mul_fast(
                            *data.get_unchecked(j + quarter),
                            *m_twiddles.get_unchecked(3 * j),
                        );
                        let c = c_mul_fast(
                            *data.get_unchecked(j + 2 * quarter),
                            *m_twiddles.get_unchecked(3 * j + 1),
                        );
                        let d = c_mul_fast(
                            *data.get_unchecked(j + 3 * quarter),
                            *m_twiddles.get_unchecked(3 * j + 2),
                        );

                        // radix-4 butterfly
                        let t0 = a + c;
                        let t1 = a - c;
                        let t2 = b + d;
                        let t3 = c_mul_fast(b - d, t3_twiddle);

                        *data.get_unchecked_mut(j) = t0 + t2;
                        *data.get_unchecked_mut(j + quarter) = t1 + t3;
                        *data.get_unchecked_mut(j + 2 * quarter) = t0 - t2;
                        *data.get_unchecked_mut(j + 3 * quarter) = t1 - t3;
                    }
                }

                m_twiddles = &m_twiddles[quarter * 3..];
                len *= 4;
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
            let size = 4usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Radix4::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = Radix4::new(size, FftDirection::Inverse).unwrap();
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
