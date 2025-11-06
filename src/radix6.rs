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
use crate::traits::FftTrigonometry;
use crate::util::{
    bitreversed_transpose, compute_twiddle, is_power_of_six, radixn_floating_twiddles_from_base,
};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(dead_code)]
pub(crate) struct Radix6<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle: Complex<T>,
    direction: FftDirection,
    butterfly: Box<dyn CompositeFftExecutor<T> + Send + Sync>,
}

pub(crate) trait Radix6Twiddles {
    #[allow(unused)]
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized;
}

impl Radix6Twiddles for f64 {
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f64, 6>(base, size, fft_direction)
    }
}

impl Radix6Twiddles for f32 {
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f32, 6>(base, size, fft_direction)
    }
}

#[allow(dead_code)]
impl<
    T: Default
        + Clone
        + Radix6Twiddles
        + 'static
        + Copy
        + FftTrigonometry
        + Float
        + AlgorithmFactory<T>,
> Radix6<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Radix6<T>, ZaftError> {
        assert!(
            is_power_of_six(size as u64),
            "Input length must be a power of 6"
        );

        let twiddles = T::make_twiddles_with_base(6, size, fft_direction)?;

        Ok(Radix6 {
            execution_length: size,
            twiddles,
            twiddle: compute_twiddle(1, 3, fft_direction),
            direction: fft_direction,
            butterfly: T::butterfly6(fft_direction)?,
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
> Radix6<T>
where
    f64: AsPrimitive<T>,
{
    #[inline]
    fn butterfly3(
        &self,
        u0: Complex<T>,
        u1: Complex<T>,
        u2: Complex<T>,
    ) -> (Complex<T>, Complex<T>, Complex<T>) {
        let xp = u1 + u2;
        let xn = u1 - u2;
        let sum = u0 + xp;

        let w_1 = Complex {
            re: fmla(self.twiddle.re, xp.re, u0.re),
            im: fmla(self.twiddle.re, xp.im, u0.im),
        };

        let y0 = sum;
        let y1 = Complex {
            re: fmla(-self.twiddle.im, xn.im, w_1.re),
            im: fmla(self.twiddle.im, xn.re, w_1.im),
        };
        let y2 = Complex {
            re: fmla(self.twiddle.im, xn.im, w_1.re),
            im: fmla(-self.twiddle.im, xn.re, w_1.im),
        };
        (y0, y1, y2)
    }

    #[inline]
    fn butterfly2(&self, u0: Complex<T>, u1: Complex<T>) -> (Complex<T>, Complex<T>) {
        let t = u0 + u1;

        let y1 = u0 - u1;
        let y0 = t;
        (y0, y1)
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
> FftExecutor<T> for Radix6<T>
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
            bitreversed_transpose::<Complex<T>, 6>(6, chunk, &mut scratch);

            self.butterfly.execute_out_of_place(&scratch, chunk)?;

            let mut len = 6;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 6;
                    let sixth = len / 6;

                    for data in chunk.chunks_exact_mut(len) {
                        for j in 0..sixth {
                            let u0 = *data.get_unchecked(j);
                            let u1 = c_mul_fast(
                                *data.get_unchecked(j + sixth),
                                *m_twiddles.get_unchecked(5 * j),
                            );
                            let u2 = c_mul_fast(
                                *data.get_unchecked(j + 2 * sixth),
                                *m_twiddles.get_unchecked(5 * j + 1),
                            );
                            let u3 = c_mul_fast(
                                *data.get_unchecked(j + 3 * sixth),
                                *m_twiddles.get_unchecked(5 * j + 2),
                            );
                            let u4 = c_mul_fast(
                                *data.get_unchecked(j + 4 * sixth),
                                *m_twiddles.get_unchecked(5 * j + 3),
                            );
                            let u5 = c_mul_fast(
                                *data.get_unchecked(j + 5 * sixth),
                                *m_twiddles.get_unchecked(5 * j + 4),
                            );

                            let (t0, t2, t4) = self.butterfly3(u0, u2, u4);
                            let (t1, t3, t5) = self.butterfly3(u3, u5, u1);
                            let (y0, y3) = self.butterfly2(t0, t1);
                            let (y4, y1) = self.butterfly2(t2, t3);
                            let (y2, y5) = self.butterfly2(t4, t5);

                            // Store results
                            *data.get_unchecked_mut(j) = y0;
                            *data.get_unchecked_mut(j + sixth) = y1;
                            *data.get_unchecked_mut(j + 2 * sixth) = y2;
                            *data.get_unchecked_mut(j + 3 * sixth) = y3;
                            *data.get_unchecked_mut(j + 4 * sixth) = y4;
                            *data.get_unchecked_mut(j + 5 * sixth) = y5;
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 5..];
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
    use crate::util::test_radix;

    test_radix!(test_radix6, f32, Radix6, 5, 6, 1e-3);
}
