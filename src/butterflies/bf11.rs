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
use crate::mla::fmla;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Butterfly11<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> Butterfly11<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly11 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 11, fft_direction),
            twiddle2: compute_twiddle(2, 11, fft_direction),
            twiddle3: compute_twiddle(3, 11, fft_direction),
            twiddle4: compute_twiddle(4, 11, fft_direction),
            twiddle5: compute_twiddle(5, 11, fft_direction),
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
        + Default
        + FftTrigonometry,
> FftExecutor<T> for Butterfly11<T>
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

        for chunk in in_place.chunks_exact_mut(11) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];

            let u4 = chunk[4];
            let u5 = chunk[5];
            let u6 = chunk[6];
            let u7 = chunk[7];

            let u8 = chunk[8];
            let u9 = chunk[9];
            let u10 = chunk[10];

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
            chunk[0] = y0;
            let b110re_a = fmla(
                self.twiddle1.re,
                x110p.re,
                fmla(
                    self.twiddle2.re,
                    x29p.re,
                    fmla(self.twiddle3.re, x38p.re, u0.re)
                        + fmla(self.twiddle4.re, x47p.re, self.twiddle5.re * x56p.re),
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
                        + fmla(self.twiddle3.re, x47p.re, self.twiddle1.re * x56p.re),
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
                        fmla(-self.twiddle3.im, x47n.im, -self.twiddle1.im * x56n.im),
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
                        + fmla(self.twiddle1.re, x47p.re, self.twiddle4.re * x56p.re),
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
                        + fmla(self.twiddle2.re, x47p.re, self.twiddle3.re * x56p.re),
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
                        fmla(-self.twiddle2.im, x47n.im, self.twiddle3.im * x56n.im),
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
                        + fmla(self.twiddle4.re, x47p.im, self.twiddle5.re * x56p.im),
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
                        + fmla(self.twiddle3.re, x47p.im, self.twiddle1.re * x56p.im),
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
                        + fmla(self.twiddle1.re, x47p.im, self.twiddle4.re * x56p.im),
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
                        + fmla(self.twiddle5.re, x47p.im, self.twiddle2.re * x56p.im),
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
                        + fmla(self.twiddle2.re, x47p.im, self.twiddle3.re * x56p.im),
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
                        fmla(-self.twiddle2.im, x47n.re, self.twiddle3.im * x56n.re),
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
            chunk[1] = y1;
            chunk[2] = y2;
            chunk[3] = y3;
            chunk[4] = y4;
            chunk[5] = y5;
            chunk[6] = y6;
            chunk[7] = y7;
            chunk[8] = y8;
            chunk[9] = y9;
            chunk[10] = y10;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        11
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
        + FftTrigonometry,
> FftExecutorOutOfPlace<T> for Butterfly11<T>
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

        for (dst, src) in dst.chunks_exact_mut(11).zip(src.chunks_exact(11)) {
            let u0 = src[0];
            let u1 = src[1];
            let u2 = src[2];
            let u3 = src[3];

            let u4 = src[4];
            let u5 = src[5];
            let u6 = src[6];
            let u7 = src[7];

            let u8 = src[8];
            let u9 = src[9];
            let u10 = src[10];

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
            dst[0] = y0;
            let b110re_a = fmla(
                self.twiddle1.re,
                x110p.re,
                fmla(
                    self.twiddle2.re,
                    x29p.re,
                    fmla(self.twiddle3.re, x38p.re, u0.re)
                        + fmla(self.twiddle4.re, x47p.re, self.twiddle5.re * x56p.re),
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
                        + fmla(self.twiddle3.re, x47p.re, self.twiddle1.re * x56p.re),
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
                        fmla(-self.twiddle3.im, x47n.im, -self.twiddle1.im * x56n.im),
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
                        + fmla(self.twiddle1.re, x47p.re, self.twiddle4.re * x56p.re),
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
                        + fmla(self.twiddle2.re, x47p.re, self.twiddle3.re * x56p.re),
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
                        fmla(-self.twiddle2.im, x47n.im, self.twiddle3.im * x56n.im),
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
                        + fmla(self.twiddle4.re, x47p.im, self.twiddle5.re * x56p.im),
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
                        + fmla(self.twiddle3.re, x47p.im, self.twiddle1.re * x56p.im),
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
                        + fmla(self.twiddle1.re, x47p.im, self.twiddle4.re * x56p.im),
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
                        + fmla(self.twiddle5.re, x47p.im, self.twiddle2.re * x56p.im),
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
                        + fmla(self.twiddle2.re, x47p.im, self.twiddle3.re * x56p.im),
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
                        fmla(-self.twiddle2.im, x47n.re, self.twiddle3.im * x56n.re),
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
            dst[1] = y1;
            dst[2] = y2;
            dst[3] = y3;
            dst[4] = y4;
            dst[5] = y5;
            dst[6] = y6;
            dst[7] = y7;
            dst[8] = y8;
            dst[9] = y9;
            dst[10] = y10;
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
        + FftTrigonometry
        + Send
        + Sync,
> CompositeFftExecutor<T> for Butterfly11<T>
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
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_butterfly11, f32, Butterfly11, 11, 1e-5);
    test_oof_butterfly!(test_oof_butterfly11, f32, Butterfly11, 11, 1e-5);
}
