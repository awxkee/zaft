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
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Butterfly17<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    twiddle7: Complex<T>,
    twiddle8: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> Butterfly17<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly17 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 17, fft_direction),
            twiddle2: compute_twiddle(2, 17, fft_direction),
            twiddle3: compute_twiddle(3, 17, fft_direction),
            twiddle4: compute_twiddle(4, 17, fft_direction),
            twiddle5: compute_twiddle(5, 17, fft_direction),
            twiddle6: compute_twiddle(6, 17, fft_direction),
            twiddle7: compute_twiddle(7, 17, fft_direction),
            twiddle8: compute_twiddle(8, 17, fft_direction),
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
> FftExecutor<T> for Butterfly17<T>
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

        for chunk in in_place.chunks_exact_mut(17) {
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
            let u11 = chunk[11];
            let u12 = chunk[12];

            let u13 = chunk[13];
            let u14 = chunk[14];
            let u15 = chunk[15];
            let u16 = chunk[16];

            let x116p = u1 + u16;
            let x116n = u1 - u16;
            let x215p = u2 + u15;
            let x215n = u2 - u15;
            let x314p = u3 + u14;
            let x314n = u3 - u14;
            let x413p = u4 + u13;
            let x413n = u4 - u13;
            let x512p = u5 + u12;
            let x512n = u5 - u12;
            let x611p = u6 + u11;
            let x611n = u6 - u11;
            let x710p = u7 + u10;
            let x710n = u7 - u10;
            let x89p = u8 + u9;
            let x89n = u8 - u9;

            let sum = u0 + x116p + x215p + x314p + x413p + x512p + x611p + x710p + x89p;
            chunk[0] = sum;

            let b116re_a = fmla(self.twiddle1.re, x116p.re, u0.re)
                + fmla(self.twiddle2.re, x215p.re, self.twiddle3.re * x314p.re)
                + fmla(self.twiddle4.re, x413p.re, self.twiddle5.re * x512p.re)
                + fmla(self.twiddle6.re, x611p.re, self.twiddle7.re * x710p.re)
                + self.twiddle8.re * x89p.re;
            let b116re_b = fmla(self.twiddle1.im, x116n.im, self.twiddle2.im * x215n.im)
                + fmla(self.twiddle3.im, x314n.im, self.twiddle4.im * x413n.im)
                + fmla(self.twiddle5.im, x512n.im, self.twiddle6.im * x611n.im)
                + fmla(self.twiddle7.im, x710n.im, self.twiddle8.im * x89n.im);
            let b215re_a = fmla(self.twiddle2.re, x116p.re, u0.re)
                + fmla(self.twiddle4.re, x215p.re, self.twiddle6.re * x314p.re)
                + fmla(self.twiddle8.re, x413p.re, self.twiddle7.re * x512p.re)
                + fmla(self.twiddle5.re, x611p.re, self.twiddle3.re * x710p.re)
                + self.twiddle1.re * x89p.re;
            let b215re_b = fmla(self.twiddle2.im, x116n.im, self.twiddle4.im * x215n.im)
                + fmla(self.twiddle6.im, x314n.im, self.twiddle8.im * x413n.im)
                + fmla(-self.twiddle7.im, x512n.im, -self.twiddle5.im * x611n.im)
                + fmla(-self.twiddle3.im, x710n.im, -self.twiddle1.im * x89n.im);
            let b314re_a = fmla(self.twiddle3.re, x116p.re, u0.re)
                + fmla(self.twiddle6.re, x215p.re, self.twiddle8.re * x314p.re)
                + fmla(self.twiddle5.re, x413p.re, self.twiddle2.re * x512p.re)
                + fmla(self.twiddle1.re, x611p.re, self.twiddle4.re * x710p.re)
                + self.twiddle7.re * x89p.re;
            let b314re_b = self.twiddle3.im * x116n.im
                + fmla(self.twiddle6.im, x215n.im, -self.twiddle8.im * x314n.im)
                + fmla(-self.twiddle5.im, x413n.im, -self.twiddle2.im * x512n.im)
                + fmla(self.twiddle1.im, x611n.im, self.twiddle4.im * x710n.im)
                + self.twiddle7.im * x89n.im;
            let b413re_a = fmla(self.twiddle4.re, x116p.re, u0.re)
                + fmla(self.twiddle8.re, x215p.re, self.twiddle5.re * x314p.re)
                + fmla(self.twiddle1.re, x413p.re, self.twiddle3.re * x512p.re)
                + fmla(self.twiddle7.re, x611p.re, self.twiddle6.re * x710p.re)
                + self.twiddle2.re * x89p.re;
            let b413re_b = fmla(self.twiddle4.im, x116n.im, self.twiddle8.im * x215n.im)
                + fmla(-self.twiddle5.im, x314n.im, -self.twiddle1.im * x413n.im)
                + fmla(self.twiddle3.im, x512n.im, self.twiddle7.im * x611n.im)
                + fmla(-self.twiddle6.im, x710n.im, -self.twiddle2.im * x89n.im);
            let b512re_a = fmla(self.twiddle5.re, x116p.re, u0.re)
                + fmla(self.twiddle7.re, x215p.re, self.twiddle2.re * x314p.re)
                + fmla(self.twiddle3.re, x413p.re, self.twiddle8.re * x512p.re)
                + fmla(self.twiddle4.re, x611p.re, self.twiddle1.re * x710p.re)
                + self.twiddle6.re * x89p.re;
            let b512re_b = fmla(self.twiddle5.im, x116n.im, -self.twiddle7.im * x215n.im)
                + fmla(-self.twiddle2.im, x314n.im, self.twiddle3.im * x413n.im)
                + fmla(self.twiddle8.im, x512n.im, -self.twiddle4.im * x611n.im)
                + fmla(self.twiddle1.im, x710n.im, self.twiddle6.im * x89n.im);
            let b611re_a = fmla(self.twiddle6.re, x116p.re, u0.re)
                + fmla(self.twiddle5.re, x215p.re, self.twiddle1.re * x314p.re)
                + fmla(self.twiddle7.re, x413p.re, self.twiddle4.re * x512p.re)
                + fmla(self.twiddle2.re, x611p.re, self.twiddle8.re * x710p.re)
                + self.twiddle3.re * x89p.re;
            let b611re_b = fmla(self.twiddle6.im, x116n.im, -self.twiddle5.im * x215n.im)
                + fmla(self.twiddle1.im, x314n.im, self.twiddle7.im * x413n.im)
                + fmla(-self.twiddle4.im, x512n.im, self.twiddle2.im * x611n.im)
                + fmla(self.twiddle8.im, x710n.im, -self.twiddle3.im * x89n.im);
            let b710re_a = fmla(self.twiddle7.re, x116p.re, u0.re)
                + fmla(self.twiddle3.re, x215p.re, self.twiddle4.re * x314p.re)
                + fmla(self.twiddle6.re, x413p.re, self.twiddle1.re * x512p.re)
                + fmla(self.twiddle8.re, x611p.re, self.twiddle2.re * x710p.re)
                + self.twiddle5.re * x89p.re;
            let b710re_b = fmla(self.twiddle7.im, x116n.im, -self.twiddle3.im * x215n.im)
                + fmla(self.twiddle4.im, x314n.im, -self.twiddle6.im * x413n.im)
                + fmla(self.twiddle1.im, x512n.im, self.twiddle8.im * x611n.im)
                + fmla(-self.twiddle2.im, x710n.im, self.twiddle5.im * x89n.im);
            let b89re_a = fmla(self.twiddle8.re, x116p.re, u0.re)
                + fmla(self.twiddle1.re, x215p.re, self.twiddle7.re * x314p.re)
                + fmla(self.twiddle2.re, x413p.re, self.twiddle6.re * x512p.re)
                + fmla(self.twiddle3.re, x611p.re, self.twiddle5.re * x710p.re)
                + self.twiddle4.re * x89p.re;
            let b89re_b = fmla(self.twiddle8.im, x116n.im, -self.twiddle1.im * x215n.im)
                + fmla(self.twiddle7.im, x314n.im, -self.twiddle2.im * x413n.im)
                + fmla(self.twiddle6.im, x512n.im, -self.twiddle3.im * x611n.im)
                + fmla(self.twiddle5.im, x710n.im, -self.twiddle4.im * x89n.im);

            let b116im_a = fmla(self.twiddle1.re, x116p.im, u0.im)
                + fmla(self.twiddle2.re, x215p.im, self.twiddle3.re * x314p.im)
                + fmla(self.twiddle4.re, x413p.im, self.twiddle5.re * x512p.im)
                + fmla(self.twiddle6.re, x611p.im, self.twiddle7.re * x710p.im)
                + self.twiddle8.re * x89p.im;
            let b116im_b = fmla(self.twiddle1.im, x116n.re, self.twiddle2.im * x215n.re)
                + fmla(self.twiddle3.im, x314n.re, self.twiddle4.im * x413n.re)
                + fmla(self.twiddle5.im, x512n.re, self.twiddle6.im * x611n.re)
                + self.twiddle7.im * x710n.re
                + self.twiddle8.im * x89n.re;
            let b215im_a = fmla(self.twiddle2.re, x116p.im, u0.im)
                + fmla(self.twiddle4.re, x215p.im, self.twiddle6.re * x314p.im)
                + fmla(self.twiddle8.re, x413p.im, self.twiddle7.re * x512p.im)
                + fmla(self.twiddle5.re, x611p.im, self.twiddle3.re * x710p.im)
                + self.twiddle1.re * x89p.im;
            let b215im_b = fmla(self.twiddle2.im, x116n.re, self.twiddle4.im * x215n.re)
                + fmla(self.twiddle6.im, x314n.re, self.twiddle8.im * x413n.re)
                + fmla(-self.twiddle7.im, x512n.re, -self.twiddle5.im * x611n.re)
                + fmla(-self.twiddle3.im, x710n.re, -self.twiddle1.im * x89n.re);
            let b314im_a = fmla(self.twiddle3.re, x116p.im, u0.im)
                + fmla(self.twiddle6.re, x215p.im, self.twiddle8.re * x314p.im)
                + fmla(self.twiddle5.re, x413p.im, self.twiddle2.re * x512p.im)
                + fmla(self.twiddle1.re, x611p.im, self.twiddle4.re * x710p.im)
                + self.twiddle7.re * x89p.im;
            let b314im_b = fmla(self.twiddle3.im, x116n.re, self.twiddle6.im * x215n.re)
                + fmla(-self.twiddle8.im, x314n.re, -self.twiddle5.im * x413n.re)
                + fmla(-self.twiddle2.im, x512n.re, self.twiddle1.im * x611n.re)
                + fmla(self.twiddle4.im, x710n.re, self.twiddle7.im * x89n.re);
            let b413im_a = fmla(self.twiddle4.re, x116p.im, u0.im)
                + fmla(self.twiddle8.re, x215p.im, self.twiddle5.re * x314p.im)
                + fmla(self.twiddle1.re, x413p.im, self.twiddle3.re * x512p.im)
                + fmla(self.twiddle7.re, x611p.im, self.twiddle6.re * x710p.im)
                + self.twiddle2.re * x89p.im;
            let b413im_b = fmla(self.twiddle4.im, x116n.re, self.twiddle8.im * x215n.re)
                + fmla(-self.twiddle5.im, x314n.re, -self.twiddle1.im * x413n.re)
                + fmla(self.twiddle3.im, x512n.re, self.twiddle7.im * x611n.re)
                + fmla(-self.twiddle6.im, x710n.re, -self.twiddle2.im * x89n.re);
            let b512im_a = fmla(self.twiddle5.re, x116p.im, u0.im)
                + fmla(self.twiddle7.re, x215p.im, self.twiddle2.re * x314p.im)
                + fmla(self.twiddle3.re, x413p.im, self.twiddle8.re * x512p.im)
                + fmla(self.twiddle4.re, x611p.im, self.twiddle1.re * x710p.im)
                + self.twiddle6.re * x89p.im;
            let b512im_b = fmla(self.twiddle5.im, x116n.re, -self.twiddle7.im * x215n.re)
                + fmla(-self.twiddle2.im, x314n.re, self.twiddle3.im * x413n.re)
                + fmla(self.twiddle8.im, x512n.re, -self.twiddle4.im * x611n.re)
                + self.twiddle1.im * x710n.re
                + self.twiddle6.im * x89n.re;
            let b611im_a = fmla(self.twiddle6.re, x116p.im, u0.im)
                + fmla(self.twiddle5.re, x215p.im, self.twiddle1.re * x314p.im)
                + fmla(self.twiddle7.re, x413p.im, self.twiddle4.re * x512p.im)
                + fmla(self.twiddle2.re, x611p.im, self.twiddle8.re * x710p.im)
                + self.twiddle3.re * x89p.im;
            let b611im_b = fmla(self.twiddle6.im, x116n.re, -self.twiddle5.im * x215n.re)
                + fmla(self.twiddle1.im, x314n.re, self.twiddle7.im * x413n.re)
                + fmla(-self.twiddle4.im, x512n.re, self.twiddle2.im * x611n.re)
                + fmla(self.twiddle8.im, x710n.re, -self.twiddle3.im * x89n.re);
            let b710im_a = fmla(self.twiddle7.re, x116p.im, u0.im)
                + fmla(self.twiddle3.re, x215p.im, self.twiddle4.re * x314p.im)
                + fmla(self.twiddle6.re, x413p.im, self.twiddle1.re * x512p.im)
                + fmla(self.twiddle8.re, x611p.im, self.twiddle2.re * x710p.im)
                + self.twiddle5.re * x89p.im;
            let b710im_b = fmla(self.twiddle7.im, x116n.re, -self.twiddle3.im * x215n.re)
                + fmla(self.twiddle4.im, x314n.re, -self.twiddle6.im * x413n.re)
                + fmla(self.twiddle1.im, x512n.re, self.twiddle8.im * x611n.re)
                + fmla(-self.twiddle2.im, x710n.re, self.twiddle5.im * x89n.re);
            let b89im_a = fmla(self.twiddle8.re, x116p.im, u0.im)
                + fmla(self.twiddle1.re, x215p.im, self.twiddle7.re * x314p.im)
                + fmla(self.twiddle2.re, x413p.im, self.twiddle6.re * x512p.im)
                + fmla(self.twiddle3.re, x611p.im, self.twiddle5.re * x710p.im)
                + self.twiddle4.re * x89p.im;
            let b89im_b = fmla(self.twiddle8.im, x116n.re, -self.twiddle1.im * x215n.re)
                + fmla(self.twiddle7.im, x314n.re, -self.twiddle2.im * x413n.re)
                + fmla(self.twiddle6.im, x512n.re, -self.twiddle3.im * x611n.re)
                + fmla(self.twiddle5.im, x710n.re, -self.twiddle4.im * x89n.re);

            let out1re = b116re_a - b116re_b;
            let out1im = b116im_a + b116im_b;
            chunk[1] = Complex {
                re: out1re,
                im: out1im,
            };

            let out2re = b215re_a - b215re_b;
            let out2im = b215im_a + b215im_b;
            chunk[2] = Complex {
                re: out2re,
                im: out2im,
            };

            let out3re = b314re_a - b314re_b;
            let out3im = b314im_a + b314im_b;
            chunk[3] = Complex {
                re: out3re,
                im: out3im,
            };

            let out4re = b413re_a - b413re_b;
            let out4im = b413im_a + b413im_b;
            chunk[4] = Complex {
                re: out4re,
                im: out4im,
            };

            let out5re = b512re_a - b512re_b;
            let out5im = b512im_a + b512im_b;
            chunk[5] = Complex {
                re: out5re,
                im: out5im,
            };

            let out6re = b611re_a - b611re_b;
            let out6im = b611im_a + b611im_b;
            chunk[6] = Complex {
                re: out6re,
                im: out6im,
            };

            let out7re = b710re_a - b710re_b;
            let out7im = b710im_a + b710im_b;
            chunk[7] = Complex {
                re: out7re,
                im: out7im,
            };

            let out8re = b89re_a - b89re_b;
            let out8im = b89im_a + b89im_b;
            chunk[8] = Complex {
                re: out8re,
                im: out8im,
            };

            let out9re = b89re_a + b89re_b;
            let out9im = b89im_a - b89im_b;
            let out10re = b710re_a + b710re_b;
            let out10im = b710im_a - b710im_b;
            let out11re = b611re_a + b611re_b;
            let out11im = b611im_a - b611im_b;
            let out12re = b512re_a + b512re_b;
            let out12im = b512im_a - b512im_b;
            let out13re = b413re_a + b413re_b;
            let out13im = b413im_a - b413im_b;
            let out14re = b314re_a + b314re_b;
            let out14im = b314im_a - b314im_b;
            let out15re = b215re_a + b215re_b;
            let out15im = b215im_a - b215im_b;
            let out16re = b116re_a + b116re_b;
            let out16im = b116im_a - b116im_b;

            chunk[9] = Complex {
                re: out9re,
                im: out9im,
            };
            chunk[10] = Complex {
                re: out10re,
                im: out10im,
            };
            chunk[11] = Complex {
                re: out11re,
                im: out11im,
            };
            chunk[12] = Complex {
                re: out12re,
                im: out12im,
            };
            chunk[13] = Complex {
                re: out13re,
                im: out13im,
            };
            chunk[14] = Complex {
                re: out14re,
                im: out14im,
            };
            chunk[15] = Complex {
                re: out15re,
                im: out15im,
            };
            chunk[16] = Complex {
                re: out16re,
                im: out16im,
            };
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        17
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    use rand::Rng;

    test_butterfly!(test_butterfly17, f32, Butterfly17, 17, 1e-5);
}
