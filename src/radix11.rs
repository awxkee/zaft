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
    bitreversed_transpose, compute_twiddle, is_power_of_eleven, radixn_floating_twiddles_from_base,
};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(dead_code)]
pub(crate) struct Radix11<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    direction: FftDirection,
    butterfly: Box<dyn CompositeFftExecutor<T> + Send + Sync>,
}

pub(crate) trait Radix11Twiddles {
    #[allow(unused)]
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized;
}

impl Radix11Twiddles for f64 {
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f64, 11>(base, size, fft_direction)
    }
}

impl Radix11Twiddles for f32 {
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f32, 11>(base, size, fft_direction)
    }
}

#[allow(dead_code)]
impl<
    T: Default
        + Clone
        + Radix11Twiddles
        + 'static
        + Copy
        + FftTrigonometry
        + Float
        + AlgorithmFactory<T>,
> Radix11<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Radix11<T>, ZaftError> {
        assert!(
            is_power_of_eleven(size as u64),
            "Input length must be a power of 11"
        );

        let twiddles = T::make_twiddles_with_base(11, size, fft_direction)?;

        Ok(Radix11 {
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 11, fft_direction),
            twiddle2: compute_twiddle(2, 11, fft_direction),
            twiddle3: compute_twiddle(3, 11, fft_direction),
            twiddle4: compute_twiddle(4, 11, fft_direction),
            twiddle5: compute_twiddle(5, 11, fft_direction),
            direction: fft_direction,
            butterfly: T::butterfly11(fft_direction)?,
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
> FftExecutor<T> for Radix11<T>
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
            bitreversed_transpose::<Complex<T>, 11>(11, chunk, &mut scratch);

            self.butterfly.execute_out_of_place(&scratch, chunk)?;

            let mut len = 11;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 11;
                    let eleventh = len / 11;

                    for data in chunk.chunks_exact_mut(len) {
                        for j in 0..eleventh {
                            let u0 = *data.get_unchecked(j);
                            let u1 = c_mul_fast(
                                *data.get_unchecked(j + eleventh),
                                *m_twiddles.get_unchecked(10 * j),
                            );
                            let u2 = c_mul_fast(
                                *data.get_unchecked(j + 2 * eleventh),
                                *m_twiddles.get_unchecked(10 * j + 1),
                            );
                            let u3 = c_mul_fast(
                                *data.get_unchecked(j + 3 * eleventh),
                                *m_twiddles.get_unchecked(10 * j + 2),
                            );
                            let u4 = c_mul_fast(
                                *data.get_unchecked(j + 4 * eleventh),
                                *m_twiddles.get_unchecked(10 * j + 3),
                            );
                            let u5 = c_mul_fast(
                                *data.get_unchecked(j + 5 * eleventh),
                                *m_twiddles.get_unchecked(10 * j + 4),
                            );
                            let u6 = c_mul_fast(
                                *data.get_unchecked(j + 6 * eleventh),
                                *m_twiddles.get_unchecked(10 * j + 5),
                            );
                            let u7 = c_mul_fast(
                                *data.get_unchecked(j + 7 * eleventh),
                                *m_twiddles.get_unchecked(10 * j + 6),
                            );
                            let u8 = c_mul_fast(
                                *data.get_unchecked(j + 8 * eleventh),
                                *m_twiddles.get_unchecked(10 * j + 7),
                            );
                            let u9 = c_mul_fast(
                                *data.get_unchecked(j + 9 * eleventh),
                                *m_twiddles.get_unchecked(10 * j + 8),
                            );
                            let u10 = c_mul_fast(
                                *data.get_unchecked(j + 10 * eleventh),
                                *m_twiddles.get_unchecked(10 * j + 9),
                            );

                            let x110p = u1 + u10;
                            let x110n = u1 - u10;
                            let x29p = u2 + u9;
                            let x29n = u2 - u9;
                            let x38p = u3 + u8;
                            let x38n = u3 - u8;
                            let x47p = u4 + u7;
                            let x47n = u4 - u7;
                            let x56p = u5 + u6;
                            let x56n = u5 - u6;

                            let y0 = u0 + x110p + x29p + x38p + x47p + x56p;
                            *data.get_unchecked_mut(j) = y0;
                            let b110re_a = fmla(
                                self.twiddle1.re,
                                x110p.re,
                                fmla(
                                    self.twiddle2.re,
                                    x29p.re,
                                    fmla(self.twiddle3.re, x38p.re, u0.re)
                                        + fmla(
                                            self.twiddle4.re,
                                            x47p.re,
                                            self.twiddle5.re * x56p.re,
                                        ),
                                ),
                            );
                            let b110re_b = fmla(
                                self.twiddle1.im,
                                x110n.im,
                                fmla(
                                    self.twiddle2.im,
                                    x29n.im,
                                    fmla(
                                        self.twiddle3.im,
                                        x38n.im,
                                        self.twiddle4.im * x47n.im + self.twiddle5.im * x56n.im,
                                    ),
                                ),
                            );
                            let b29re_a = fmla(
                                self.twiddle2.re,
                                x110p.re,
                                fmla(
                                    self.twiddle4.re,
                                    x29p.re,
                                    fmla(self.twiddle5.re, x38p.re, u0.re)
                                        + fmla(
                                            self.twiddle3.re,
                                            x47p.re,
                                            self.twiddle1.re * x56p.re,
                                        ),
                                ),
                            );
                            let b29re_b = fmla(
                                self.twiddle2.im,
                                x110n.im,
                                fmla(
                                    self.twiddle4.im,
                                    x29n.im,
                                    fmla(
                                        -self.twiddle5.im,
                                        x38n.im,
                                        fmla(
                                            -self.twiddle3.im,
                                            x47n.im,
                                            -self.twiddle1.im * x56n.im,
                                        ),
                                    ),
                                ),
                            );
                            let b38re_a = fmla(
                                self.twiddle3.re,
                                x110p.re,
                                fmla(
                                    self.twiddle5.re,
                                    x29p.re,
                                    fmla(self.twiddle2.re, x38p.re, u0.re)
                                        + fmla(
                                            self.twiddle1.re,
                                            x47p.re,
                                            self.twiddle4.re * x56p.re,
                                        ),
                                ),
                            );
                            let b38re_b = fmla(
                                self.twiddle3.im,
                                x110n.im,
                                fmla(
                                    -self.twiddle5.im,
                                    x29n.im,
                                    fmla(
                                        -self.twiddle2.im,
                                        x38n.im,
                                        self.twiddle1.im * x47n.im + self.twiddle4.im * x56n.im,
                                    ),
                                ),
                            );
                            let b47re_a = fmla(
                                self.twiddle4.re,
                                x110p.re,
                                fmla(self.twiddle3.re, x29p.re, u0.re)
                                    + fmla(
                                        self.twiddle5.re,
                                        x47p.re,
                                        fmla(self.twiddle2.re, x56p.re, self.twiddle1.re * x38p.re),
                                    ),
                            );
                            let b47re_b = fmla(
                                self.twiddle4.im,
                                x110n.im,
                                fmla(
                                    -self.twiddle3.im,
                                    x29n.im,
                                    fmla(self.twiddle1.im, x38n.im, self.twiddle5.im * x47n.im)
                                        + -self.twiddle2.im * x56n.im,
                                ),
                            );
                            let b56re_a = fmla(
                                self.twiddle5.re,
                                x110p.re,
                                fmla(
                                    self.twiddle1.re,
                                    x29p.re,
                                    fmla(self.twiddle4.re, x38p.re, u0.re)
                                        + fmla(
                                            self.twiddle2.re,
                                            x47p.re,
                                            self.twiddle3.re * x56p.re,
                                        ),
                                ),
                            );
                            let b56re_b = fmla(
                                self.twiddle5.im,
                                x110n.im,
                                fmla(
                                    -self.twiddle1.im,
                                    x29n.im,
                                    fmla(
                                        self.twiddle4.im,
                                        x38n.im,
                                        fmla(
                                            -self.twiddle2.im,
                                            x47n.im,
                                            self.twiddle3.im * x56n.im,
                                        ),
                                    ),
                                ),
                            );

                            let b110im_a = fmla(
                                self.twiddle1.re,
                                x110p.im,
                                fmla(
                                    self.twiddle2.re,
                                    x29p.im,
                                    fmla(self.twiddle3.re, x38p.im, u0.im)
                                        + fmla(
                                            self.twiddle4.re,
                                            x47p.im,
                                            self.twiddle5.re * x56p.im,
                                        ),
                                ),
                            );
                            let b110im_b = fmla(
                                self.twiddle1.im,
                                x110n.re,
                                fmla(
                                    self.twiddle2.im,
                                    x29n.re,
                                    fmla(
                                        self.twiddle3.im,
                                        x38n.re,
                                        fmla(self.twiddle4.im, x47n.re, self.twiddle5.im * x56n.re),
                                    ),
                                ),
                            );
                            let b29im_a = fmla(
                                self.twiddle2.re,
                                x110p.im,
                                fmla(
                                    self.twiddle4.re,
                                    x29p.im,
                                    fmla(self.twiddle5.re, x38p.im, u0.im)
                                        + fmla(
                                            self.twiddle3.re,
                                            x47p.im,
                                            self.twiddle1.re * x56p.im,
                                        ),
                                ),
                            );
                            let b29im_b = fmla(
                                self.twiddle2.im,
                                x110n.re,
                                fmla(
                                    self.twiddle4.im,
                                    x29n.re,
                                    fmla(
                                        -self.twiddle5.im,
                                        x38n.re,
                                        -self.twiddle3.im * x47n.re + -self.twiddle1.im * x56n.re,
                                    ),
                                ),
                            );
                            let b38im_a = fmla(
                                self.twiddle3.re,
                                x110p.im,
                                fmla(
                                    self.twiddle5.re,
                                    x29p.im,
                                    fmla(self.twiddle2.re, x38p.im, u0.im)
                                        + fmla(
                                            self.twiddle1.re,
                                            x47p.im,
                                            self.twiddle4.re * x56p.im,
                                        ),
                                ),
                            );
                            let b38im_b = fmla(
                                self.twiddle3.im,
                                x110n.re,
                                fmla(
                                    -self.twiddle5.im,
                                    x29n.re,
                                    fmla(
                                        -self.twiddle2.im,
                                        x38n.re,
                                        self.twiddle1.im * x47n.re + self.twiddle4.im * x56n.re,
                                    ),
                                ),
                            );
                            let b47im_a = fmla(
                                self.twiddle4.re,
                                x110p.im,
                                fmla(
                                    self.twiddle3.re,
                                    x29p.im,
                                    fmla(self.twiddle1.re, x38p.im, u0.im)
                                        + fmla(
                                            self.twiddle5.re,
                                            x47p.im,
                                            self.twiddle2.re * x56p.im,
                                        ),
                                ),
                            );
                            let b47im_b = fmla(
                                self.twiddle4.im,
                                x110n.re,
                                fmla(
                                    -self.twiddle3.im,
                                    x29n.re,
                                    self.twiddle1.im * x38n.re
                                        + self.twiddle5.im * x47n.re
                                        + -self.twiddle2.im * x56n.re,
                                ),
                            );
                            let b56im_a = fmla(
                                self.twiddle5.re,
                                x110p.im,
                                fmla(
                                    self.twiddle1.re,
                                    x29p.im,
                                    fmla(self.twiddle4.re, x38p.im, u0.im)
                                        + fmla(
                                            self.twiddle2.re,
                                            x47p.im,
                                            self.twiddle3.re * x56p.im,
                                        ),
                                ),
                            );
                            let b56im_b = fmla(
                                self.twiddle5.im,
                                x110n.re,
                                fmla(
                                    -self.twiddle1.im,
                                    x29n.re,
                                    fmla(
                                        self.twiddle4.im,
                                        x38n.re,
                                        fmla(
                                            -self.twiddle2.im,
                                            x47n.re,
                                            self.twiddle3.im * x56n.re,
                                        ),
                                    ),
                                ),
                            );

                            let y1 = Complex {
                                re: b110re_a - b110re_b,
                                im: b110im_a + b110im_b,
                            };
                            let y2 = Complex {
                                re: b29re_a - b29re_b,
                                im: b29im_a + b29im_b,
                            };
                            let y3 = Complex {
                                re: b38re_a - b38re_b,
                                im: b38im_a + b38im_b,
                            };
                            let y4 = Complex {
                                re: b47re_a - b47re_b,
                                im: b47im_a + b47im_b,
                            };
                            let y5 = Complex {
                                re: b56re_a - b56re_b,
                                im: b56im_a + b56im_b,
                            };
                            let y6 = Complex {
                                re: b56re_a + b56re_b,
                                im: b56im_a - b56im_b,
                            };
                            let y7 = Complex {
                                re: b47re_a + b47re_b,
                                im: b47im_a - b47im_b,
                            };
                            let y8 = Complex {
                                re: b38re_a + b38re_b,
                                im: b38im_a - b38im_b,
                            };
                            let y9 = Complex {
                                re: b29re_a + b29re_b,
                                im: b29im_a - b29im_b,
                            };
                            let y10 = Complex {
                                re: b110re_a + b110re_b,
                                im: b110im_a - b110im_b,
                            };

                            // Store results
                            *data.get_unchecked_mut(j + eleventh) = y1;
                            *data.get_unchecked_mut(j + 2 * eleventh) = y2;
                            *data.get_unchecked_mut(j + 3 * eleventh) = y3;
                            *data.get_unchecked_mut(j + 4 * eleventh) = y4;
                            *data.get_unchecked_mut(j + 5 * eleventh) = y5;
                            *data.get_unchecked_mut(j + 6 * eleventh) = y6;
                            *data.get_unchecked_mut(j + 7 * eleventh) = y7;
                            *data.get_unchecked_mut(j + 8 * eleventh) = y8;
                            *data.get_unchecked_mut(j + 9 * eleventh) = y9;
                            *data.get_unchecked_mut(j + 10 * eleventh) = y10;
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 10..];
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
    fn test_radix11() {
        for i in 1..5 {
            let size = 11usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Radix11::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = Radix11::new(size, FftDirection::Inverse).unwrap();
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
