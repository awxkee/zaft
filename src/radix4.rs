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
use crate::butterflies::rotate_90;
use crate::complex_fma::c_mul_fast;
use crate::factory::AlgorithmFactory;
use crate::util::{bitreversed_transpose, radixn_floating_twiddles_from_base};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, MulAdd, Num, Zero};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Radix4<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    direction: FftDirection,
    base_len: usize,
    base_fft: Box<dyn FftExecutor<T> + Send + Sync>,
}

pub(crate) trait Radix4Twiddles {
    fn make_twiddles(
        base_len: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized;
}

impl Radix4Twiddles for f64 {
    fn make_twiddles(
        base_len: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<f64>>, ZaftError> {
        radixn_floating_twiddles_from_base::<f64, 4>(base_len, size, fft_direction)
    }
}

impl Radix4Twiddles for f32 {
    fn make_twiddles(
        base_len: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<f32>>, ZaftError> {
        radixn_floating_twiddles_from_base::<f32, 4>(base_len, size, fft_direction)
    }
}

#[allow(unused)]
impl<T: Default + Clone + Radix4Twiddles + AlgorithmFactory<T>> Radix4<T> {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Radix4<T>, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");

        let exponent = size.trailing_zeros();
        let base_fft = match exponent {
            0 => T::butterfly1(fft_direction)?,
            1 => T::butterfly2(fft_direction)?,
            2 => T::butterfly4(fft_direction)?,
            3 => T::butterfly8(fft_direction)?,
            _ => {
                if exponent % 2 == 1 {
                    (T::butterfly32(fft_direction)?)
                } else {
                    (T::butterfly16(fft_direction)?)
                }
            }
        };

        let twiddles = T::make_twiddles(base_fft.length(), size, fft_direction)?;

        Ok(Radix4 {
            execution_length: size,
            twiddles,
            direction: fft_direction,
            base_len: base_fft.length(),
            base_fft,
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
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let mut scratch = vec![Complex::zero(); self.execution_length];

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            scratch.copy_from_slice(chunk);
            // bit reversal first
            bitreversed_transpose::<Complex<T>, 4>(self.base_len, &scratch, chunk);

            self.base_fft.execute(chunk)?;

            let mut len = self.base_len;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 4;
                    let quarter = len / 4;

                    for data in chunk.chunks_exact_mut(len) {
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
                            let t3 = rotate_90(b - d, self.direction);

                            *data.get_unchecked_mut(j) = t0 + t2;
                            *data.get_unchecked_mut(j + quarter) = t1 + t3;
                            *data.get_unchecked_mut(j + 2 * quarter) = t0 - t2;
                            *data.get_unchecked_mut(j + 3 * quarter) = t1 - t3;
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 3..];
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
