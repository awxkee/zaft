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
use crate::complex_fma::c_mul_fast;
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::mla::fmla;
use crate::traits::FftTrigonometry;
use crate::util::{
    bitreversed_transpose, compute_twiddle, is_power_of_thirteen,
    radixn_floating_twiddles_from_base,
};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

#[allow(dead_code)]
pub(crate) struct Radix13<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    butterfly: Arc<dyn CompositeFftExecutor<T> + Send + Sync>,
    direction: FftDirection,
}

pub(crate) trait Radix13Twiddles {
    #[allow(unused)]
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized;
}

impl Radix13Twiddles for f64 {
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f64, 13>(base, size, fft_direction)
    }
}

impl Radix13Twiddles for f32 {
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f32, 13>(base, size, fft_direction)
    }
}

#[allow(dead_code)]
impl<
    T: Default
        + Clone
        + Radix13Twiddles
        + 'static
        + Copy
        + FftTrigonometry
        + Float
        + AlgorithmFactory<T>,
> Radix13<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Radix13<T>, ZaftError> {
        assert!(
            is_power_of_thirteen(size as u64),
            "Input length must be a power of 13"
        );

        let twiddles = T::make_twiddles_with_base(13, size, fft_direction)?;

        Ok(Radix13 {
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 13, fft_direction),
            twiddle2: compute_twiddle(2, 13, fft_direction),
            twiddle3: compute_twiddle(3, 13, fft_direction),
            twiddle4: compute_twiddle(4, 13, fft_direction),
            twiddle5: compute_twiddle(5, 13, fft_direction),
            twiddle6: compute_twiddle(6, 13, fft_direction),
            butterfly: T::butterfly13(fft_direction)?,
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
> FftExecutor<T> for Radix13<T>
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
            bitreversed_transpose::<Complex<T>, 13>(13, chunk, &mut scratch);

            self.butterfly.execute_out_of_place(&scratch, chunk)?;

            let mut len = 13;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 13;
                    let thirteen = len / 13;

                    for data in chunk.chunks_exact_mut(len) {
                        for j in 0..thirteen {
                            let u0 = *data.get_unchecked(j);
                            let u1 = c_mul_fast(
                                *data.get_unchecked(j + thirteen),
                                *m_twiddles.get_unchecked(12 * j),
                            );
                            let u2 = c_mul_fast(
                                *data.get_unchecked(j + 2 * thirteen),
                                *m_twiddles.get_unchecked(12 * j + 1),
                            );
                            let u3 = c_mul_fast(
                                *data.get_unchecked(j + 3 * thirteen),
                                *m_twiddles.get_unchecked(12 * j + 2),
                            );
                            let u4 = c_mul_fast(
                                *data.get_unchecked(j + 4 * thirteen),
                                *m_twiddles.get_unchecked(12 * j + 3),
                            );
                            let u5 = c_mul_fast(
                                *data.get_unchecked(j + 5 * thirteen),
                                *m_twiddles.get_unchecked(12 * j + 4),
                            );
                            let u6 = c_mul_fast(
                                *data.get_unchecked(j + 6 * thirteen),
                                *m_twiddles.get_unchecked(12 * j + 5),
                            );
                            let u7 = c_mul_fast(
                                *data.get_unchecked(j + 7 * thirteen),
                                *m_twiddles.get_unchecked(12 * j + 6),
                            );
                            let u8 = c_mul_fast(
                                *data.get_unchecked(j + 8 * thirteen),
                                *m_twiddles.get_unchecked(12 * j + 7),
                            );
                            let u9 = c_mul_fast(
                                *data.get_unchecked(j + 9 * thirteen),
                                *m_twiddles.get_unchecked(12 * j + 8),
                            );
                            let u10 = c_mul_fast(
                                *data.get_unchecked(j + 10 * thirteen),
                                *m_twiddles.get_unchecked(12 * j + 9),
                            );
                            let u11 = c_mul_fast(
                                *data.get_unchecked(j + 11 * thirteen),
                                *m_twiddles.get_unchecked(12 * j + 10),
                            );
                            let u12 = c_mul_fast(
                                *data.get_unchecked(j + 12 * thirteen),
                                *m_twiddles.get_unchecked(12 * j + 11),
                            );

                            let x112p = u1 + u12;
                            let x112n = u1 - u12;
                            let x211p = u2 + u11;
                            let x211n = u2 - u11;
                            let x310p = u3 + u10;
                            let x310n = u3 - u10;
                            let x49p = u4 + u9;
                            let x49n = u4 - u9;
                            let x58p = u5 + u8;
                            let x58n = u5 - u8;
                            let x67p = u6 + u7;
                            let x67n = u6 - u7;
                            let y0 = u0 + x112p + x211p + x310p + x49p + x58p + x67p;
                            *data.get_unchecked_mut(j) = y0;
                            let b112re_a = fmla(self.twiddle1.re, x112p.re, u0.re)
                                + fmla(self.twiddle2.re, x211p.re, self.twiddle3.re * x310p.re)
                                + fmla(self.twiddle4.re, x49p.re, self.twiddle5.re * x58p.re)
                                + self.twiddle6.re * x67p.re;
                            let b112re_b = fmla(
                                self.twiddle1.im,
                                x112n.im,
                                fmla(
                                    self.twiddle2.im,
                                    x211n.im,
                                    fmla(self.twiddle3.im, x310n.im, self.twiddle4.im * x49n.im)
                                        + fmla(
                                            self.twiddle5.im,
                                            x58n.im,
                                            self.twiddle6.im * x67n.im,
                                        ),
                                ),
                            );
                            let b211re_a = fmla(
                                self.twiddle2.re,
                                x112p.re,
                                fmla(self.twiddle4.re, x211p.re, u0.re)
                                    + fmla(self.twiddle6.re, x310p.re, self.twiddle5.re * x49p.re)
                                    + fmla(self.twiddle3.re, x58p.re, self.twiddle1.re * x67p.re),
                            );
                            let b211re_b = fmla(
                                self.twiddle2.im,
                                x112n.im,
                                fmla(
                                    self.twiddle6.im,
                                    x310n.im,
                                    fmla(-self.twiddle5.im, x49n.im, self.twiddle4.im * x211n.im),
                                ) + fmla(-self.twiddle3.im, x58n.im, -self.twiddle1.im * x67n.im),
                            );
                            let b310re_a = fmla(self.twiddle3.re, x112p.re, u0.re)
                                + fmla(
                                    self.twiddle4.re,
                                    x310p.re,
                                    fmla(self.twiddle1.re, x49p.re, self.twiddle6.re * x211p.re),
                                )
                                + fmla(self.twiddle2.re, x58p.re, self.twiddle5.re * x67p.re);
                            let b310re_b = fmla(
                                self.twiddle3.im,
                                x112n.im,
                                fmla(
                                    self.twiddle6.im,
                                    x211n.im,
                                    fmla(-self.twiddle4.im, x310n.im, -self.twiddle1.im * x49n.im)
                                        + fmla(
                                            self.twiddle2.im,
                                            x58n.im,
                                            self.twiddle5.im * x67n.im,
                                        ),
                                ),
                            );
                            let b49re_a = fmla(
                                self.twiddle4.re,
                                x112p.re,
                                fmla(self.twiddle5.re, x211p.re, u0.re)
                                    + fmla(self.twiddle1.re, x310p.re, self.twiddle3.re * x49p.re)
                                    + fmla(self.twiddle6.re, x58p.re, self.twiddle2.re * x67p.re),
                            );
                            let b49re_b = fmla(
                                self.twiddle4.im,
                                x112n.im,
                                fmla(-self.twiddle5.im, x211n.im, -self.twiddle1.im * x310n.im)
                                    + fmla(self.twiddle3.im, x49n.im, -self.twiddle6.im * x58n.im)
                                    + -self.twiddle2.im * x67n.im,
                            );
                            let b58re_a = fmla(self.twiddle5.re, x112p.re, u0.re)
                                + fmla(self.twiddle3.re, x211p.re, self.twiddle2.re * x310p.re)
                                + fmla(self.twiddle6.re, x49p.re, self.twiddle1.re * x58p.re)
                                + self.twiddle4.re * x67p.re;
                            let b58re_b = fmla(
                                self.twiddle5.im,
                                x112n.im,
                                fmla(
                                    -self.twiddle3.im,
                                    x211n.im,
                                    self.twiddle2.im * x310n.im
                                        + fmla(
                                            -self.twiddle6.im,
                                            x49n.im,
                                            fmla(
                                                -self.twiddle1.im,
                                                x58n.im,
                                                self.twiddle4.im * x67n.im,
                                            ),
                                        ),
                                ),
                            );
                            let b67re_a = fmla(
                                self.twiddle6.re,
                                x112p.re,
                                u0.re
                                    + fmla(self.twiddle1.re, x211p.re, self.twiddle5.re * x310p.re)
                                    + self.twiddle2.re * x49p.re
                                    + fmla(self.twiddle4.re, x58p.re, self.twiddle3.re * x67p.re),
                            );
                            let b67re_b = fmla(
                                self.twiddle6.im,
                                x112n.im,
                                fmla(
                                    -self.twiddle1.im,
                                    x211n.im,
                                    self.twiddle5.im * x310n.im
                                        + fmla(
                                            -self.twiddle2.im,
                                            x49n.im,
                                            fmla(
                                                self.twiddle4.im,
                                                x58n.im,
                                                -self.twiddle3.im * x67n.im,
                                            ),
                                        ),
                                ),
                            );

                            let b112im_a = fmla(self.twiddle1.re, x112p.im, u0.im)
                                + fmla(self.twiddle2.re, x211p.im, self.twiddle3.re * x310p.im)
                                + fmla(
                                    self.twiddle4.re,
                                    x49p.im,
                                    fmla(self.twiddle5.re, x58p.im, self.twiddle6.re * x67p.im),
                                );
                            let b112im_b = fmla(
                                self.twiddle1.im,
                                x112n.re,
                                fmla(
                                    self.twiddle2.im,
                                    x211n.re,
                                    fmla(self.twiddle3.im, x310n.re, self.twiddle4.im * x49n.re)
                                        + fmla(
                                            self.twiddle5.im,
                                            x58n.re,
                                            self.twiddle6.im * x67n.re,
                                        ),
                                ),
                            );
                            let b211im_a = fmla(
                                self.twiddle2.re,
                                x112p.im,
                                fmla(self.twiddle4.re, x211p.im, u0.im)
                                    + fmla(self.twiddle6.re, x310p.im, self.twiddle5.re * x49p.im)
                                    + fmla(self.twiddle3.re, x58p.im, self.twiddle1.re * x67p.im),
                            );
                            let b211im_b = fmla(
                                self.twiddle2.im,
                                x112n.re,
                                fmla(self.twiddle4.im, x211n.re, self.twiddle6.im * x310n.re)
                                    + fmla(
                                        -self.twiddle5.im,
                                        x49n.re,
                                        fmla(
                                            -self.twiddle3.im,
                                            x58n.re,
                                            -self.twiddle1.im * x67n.re,
                                        ),
                                    ),
                            );
                            let b310im_a = fmla(
                                self.twiddle3.re,
                                x112p.im,
                                fmla(
                                    self.twiddle6.re,
                                    x211p.im,
                                    fmla(
                                        self.twiddle4.re,
                                        x310p.im,
                                        fmla(self.twiddle1.re, x49p.im, u0.im),
                                    ) + fmla(self.twiddle2.re, x58p.im, self.twiddle5.re * x67p.im),
                                ),
                            );
                            let b310im_b = fmla(
                                self.twiddle3.im,
                                x112n.re,
                                fmla(self.twiddle6.im, x211n.re, -self.twiddle4.im * x310n.re)
                                    + fmla(-self.twiddle1.im, x49n.re, self.twiddle2.im * x58n.re)
                                    + self.twiddle5.im * x67n.re,
                            );
                            let b49im_a = fmla(
                                self.twiddle4.re,
                                x112p.im,
                                fmla(self.twiddle5.re, x211p.im, self.twiddle1.re * x310p.im)
                                    + fmla(self.twiddle3.re, x49p.im, u0.im)
                                    + fmla(self.twiddle6.re, x58p.im, self.twiddle2.re * x67p.im),
                            );
                            let b49im_b = fmla(
                                self.twiddle4.im,
                                x112n.re,
                                fmla(-self.twiddle5.im, x211n.re, -self.twiddle1.im * x310n.re)
                                    + self.twiddle3.im * x49n.re
                                    + fmla(-self.twiddle6.im, x58n.re, -self.twiddle2.im * x67n.re),
                            );
                            let b58im_a = fmla(
                                self.twiddle5.re,
                                x112p.im,
                                u0.im
                                    + fmla(self.twiddle3.re, x211p.im, self.twiddle2.re * x310p.im)
                                    + self.twiddle6.re * x49p.im
                                    + fmla(self.twiddle1.re, x58p.im, self.twiddle4.re * x67p.im),
                            );
                            let b58im_b = fmla(
                                self.twiddle5.im,
                                x112n.re,
                                fmla(
                                    -self.twiddle3.im,
                                    x211n.re,
                                    fmla(self.twiddle2.im, x310n.re, -self.twiddle6.im * x49n.re)
                                        + fmla(
                                            -self.twiddle1.im,
                                            x58n.re,
                                            self.twiddle4.im * x67n.re,
                                        ),
                                ),
                            );
                            let b67im_a = fmla(
                                self.twiddle6.re,
                                x112p.im,
                                u0.im
                                    + fmla(
                                        self.twiddle2.re,
                                        x49p.im,
                                        fmla(
                                            self.twiddle1.re,
                                            x211p.im,
                                            self.twiddle5.re * x310p.im,
                                        ),
                                    )
                                    + fmla(self.twiddle4.re, x58p.im, self.twiddle3.re * x67p.im),
                            );
                            let b67im_b = fmla(
                                self.twiddle6.im,
                                x112n.re,
                                fmla(
                                    -self.twiddle1.im,
                                    x211n.re,
                                    fmla(
                                        self.twiddle5.im,
                                        x310n.re,
                                        fmla(
                                            -self.twiddle2.im,
                                            x49n.re,
                                            self.twiddle4.im * x58n.re
                                                + -self.twiddle3.im * x67n.re,
                                        ),
                                    ),
                                ),
                            );

                            *data.get_unchecked_mut(j + thirteen) = Complex {
                                re: b112re_a - b112re_b,
                                im: b112im_a + b112im_b,
                            };
                            *data.get_unchecked_mut(j + 2 * thirteen) = Complex {
                                re: b211re_a - b211re_b,
                                im: b211im_a + b211im_b,
                            };
                            *data.get_unchecked_mut(j + 3 * thirteen) = Complex {
                                re: b310re_a - b310re_b,
                                im: b310im_a + b310im_b,
                            };
                            *data.get_unchecked_mut(j + 4 * thirteen) = Complex {
                                re: b49re_a - b49re_b,
                                im: b49im_a + b49im_b,
                            };
                            *data.get_unchecked_mut(j + 5 * thirteen) = Complex {
                                re: b58re_a - b58re_b,
                                im: b58im_a + b58im_b,
                            };
                            *data.get_unchecked_mut(j + 6 * thirteen) = Complex {
                                re: b67re_a - b67re_b,
                                im: b67im_a + b67im_b,
                            };
                            *data.get_unchecked_mut(j + 7 * thirteen) = Complex {
                                re: b67re_a + b67re_b,
                                im: b67im_a - b67im_b,
                            };
                            *data.get_unchecked_mut(j + 8 * thirteen) = Complex {
                                re: b58re_a + b58re_b,
                                im: b58im_a - b58im_b,
                            };
                            *data.get_unchecked_mut(j + 9 * thirteen) = Complex {
                                re: b49re_a + b49re_b,
                                im: b49im_a - b49im_b,
                            };
                            *data.get_unchecked_mut(j + 10 * thirteen) = Complex {
                                re: b310re_a + b310re_b,
                                im: b310im_a - b310im_b,
                            };
                            *data.get_unchecked_mut(j + 11 * thirteen) = Complex {
                                re: b211re_a + b211re_b,
                                im: b211im_a - b211im_b,
                            };
                            *data.get_unchecked_mut(j + 12 * thirteen) = Complex {
                                re: b112re_a + b112re_b,
                                im: b112im_a - b112im_b,
                            };
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 12..];
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

    test_radix!(test_radix13, f32, Radix13, 4, 13, 1e-2);
}
