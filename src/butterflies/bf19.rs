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
pub(crate) struct Butterfly19<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    twiddle7: Complex<T>,
    twiddle8: Complex<T>,
    twiddle9: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> Butterfly19<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly19 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 19, fft_direction),
            twiddle2: compute_twiddle(2, 19, fft_direction),
            twiddle3: compute_twiddle(3, 19, fft_direction),
            twiddle4: compute_twiddle(4, 19, fft_direction),
            twiddle5: compute_twiddle(5, 19, fft_direction),
            twiddle6: compute_twiddle(6, 19, fft_direction),
            twiddle7: compute_twiddle(7, 19, fft_direction),
            twiddle8: compute_twiddle(8, 19, fft_direction),
            twiddle9: compute_twiddle(9, 19, fft_direction),
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
> FftExecutor<T> for Butterfly19<T>
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

        for chunk in in_place.chunks_exact_mut(19) {
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

            let u17 = chunk[17];
            let u18 = chunk[18];

            let x118p = u1 + u18;
            let x118n = u1 - u18;
            let x217p = u2 + u17;
            let x217n = u2 - u17;
            let x316p = u3 + u16;
            let x316n = u3 - u16;
            let x415p = u4 + u15;
            let x415n = u4 - u15;
            let x514p = u5 + u14;
            let x514n = u5 - u14;
            let x613p = u6 + u13;
            let x613n = u6 - u13;
            let x712p = u7 + u12;
            let x712n = u7 - u12;
            let x811p = u8 + u11;
            let x811n = u8 - u11;
            let x910p = u9 + u10;
            let x910n = u9 - u10;
            let y0 = u0 + x118p + x217p + x316p + x415p + x514p + x613p + x712p + x811p + x910p;
            chunk[0] = y0;
            let b118re_a = fmla(self.twiddle1.re, x118p.re, u0.re)
                + fmla(self.twiddle2.re, x217p.re, self.twiddle3.re * x316p.re)
                + fmla(self.twiddle4.re, x415p.re, self.twiddle5.re * x514p.re)
                + fmla(self.twiddle6.re, x613p.re, self.twiddle7.re * x712p.re)
                + fmla(self.twiddle8.re, x811p.re, self.twiddle9.re * x910p.re);
            let b118re_b = fmla(self.twiddle1.im, x118n.im, self.twiddle2.im * x217n.im)
                + fmla(self.twiddle3.im, x316n.im, self.twiddle4.im * x415n.im)
                + fmla(self.twiddle5.im, x514n.im, self.twiddle6.im * x613n.im)
                + fmla(self.twiddle7.im, x712n.im, self.twiddle8.im * x811n.im)
                + self.twiddle9.im * x910n.im;
            let b217re_a = fmla(self.twiddle2.re, x118p.re, u0.re)
                + fmla(self.twiddle4.re, x217p.re, self.twiddle6.re * x316p.re)
                + fmla(self.twiddle8.re, x415p.re, self.twiddle9.re * x514p.re)
                + fmla(self.twiddle7.re, x613p.re, self.twiddle5.re * x712p.re)
                + fmla(self.twiddle3.re, x811p.re, self.twiddle1.re * x910p.re);
            let b217re_b = fmla(self.twiddle2.im, x118n.im, self.twiddle4.im * x217n.im)
                + fmla(self.twiddle6.im, x316n.im, self.twiddle8.im * x415n.im)
                + fmla(-self.twiddle9.im, x514n.im, -self.twiddle7.im * x613n.im)
                + fmla(-self.twiddle5.im, x712n.im, -self.twiddle3.im * x811n.im)
                + -self.twiddle1.im * x910n.im;
            let b316re_a = fmla(self.twiddle3.re, x118p.re, u0.re)
                + fmla(self.twiddle6.re, x217p.re, self.twiddle9.re * x316p.re)
                + fmla(self.twiddle7.re, x415p.re, self.twiddle4.re * x514p.re)
                + fmla(self.twiddle1.re, x613p.re, self.twiddle2.re * x712p.re)
                + fmla(self.twiddle5.re, x811p.re, self.twiddle8.re * x910p.re);
            let b316re_b = self.twiddle3.im * x118n.im
                + fmla(self.twiddle6.im, x217n.im, self.twiddle9.im * x316n.im)
                + fmla(-self.twiddle7.im, x415n.im, -self.twiddle4.im * x514n.im)
                + fmla(-self.twiddle1.im, x613n.im, self.twiddle2.im * x712n.im)
                + fmla(self.twiddle5.im, x811n.im, self.twiddle8.im * x910n.im);
            let b415re_a = fmla(self.twiddle4.re, x118p.re, u0.re)
                + fmla(self.twiddle8.re, x217p.re, self.twiddle7.re * x316p.re)
                + fmla(self.twiddle3.re, x415p.re, self.twiddle1.re * x514p.re)
                + fmla(self.twiddle5.re, x613p.re, self.twiddle9.re * x712p.re)
                + fmla(self.twiddle6.re, x811p.re, self.twiddle2.re * x910p.re);
            let b415re_b = fmla(self.twiddle4.im, x118n.im, self.twiddle8.im * x217n.im)
                + fmla(-self.twiddle7.im, x316n.im, -self.twiddle3.im * x415n.im)
                + fmla(self.twiddle1.im, x514n.im, self.twiddle5.im * x613n.im)
                + fmla(self.twiddle9.im, x712n.im, -self.twiddle6.im * x811n.im)
                + -self.twiddle2.im * x910n.im;
            let b514re_a = fmla(self.twiddle5.re, x118p.re, u0.re)
                + fmla(self.twiddle9.re, x217p.re, self.twiddle4.re * x316p.re)
                + fmla(self.twiddle1.re, x415p.re, self.twiddle6.re * x514p.re)
                + fmla(self.twiddle8.re, x613p.re, self.twiddle3.re * x712p.re)
                + fmla(self.twiddle2.re, x811p.re, self.twiddle7.re * x910p.re);
            let b514re_b = fmla(self.twiddle5.im, x118n.im, -self.twiddle9.im * x217n.im)
                + fmla(-self.twiddle4.im, x316n.im, self.twiddle1.im * x415n.im)
                + fmla(self.twiddle6.im, x514n.im, -self.twiddle8.im * x613n.im)
                + fmla(-self.twiddle3.im, x712n.im, self.twiddle2.im * x811n.im)
                + self.twiddle7.im * x910n.im;
            let b613re_a = fmla(self.twiddle6.re, x118p.re, u0.re)
                + fmla(self.twiddle7.re, x217p.re, self.twiddle1.re * x316p.re)
                + fmla(self.twiddle5.re, x415p.re, self.twiddle8.re * x514p.re)
                + fmla(self.twiddle2.re, x613p.re, self.twiddle4.re * x712p.re)
                + fmla(self.twiddle9.re, x811p.re, self.twiddle3.re * x910p.re);
            let b613re_b = fmla(self.twiddle6.im, x118n.im, -self.twiddle7.im * x217n.im)
                + fmla(-self.twiddle1.im, x316n.im, self.twiddle5.im * x415n.im)
                + fmla(-self.twiddle8.im, x514n.im, -self.twiddle2.im * x613n.im)
                + fmla(self.twiddle4.im, x712n.im, -self.twiddle9.im * x811n.im)
                + -self.twiddle3.im * x910n.im;
            let b712re_a = fmla(self.twiddle7.re, x118p.re, u0.re)
                + fmla(self.twiddle5.re, x217p.re, self.twiddle2.re * x316p.re)
                + fmla(self.twiddle9.re, x415p.re, self.twiddle3.re * x514p.re)
                + fmla(self.twiddle4.re, x613p.re, self.twiddle8.re * x712p.re)
                + fmla(self.twiddle1.re, x811p.re, self.twiddle6.re * x910p.re);
            let b712re_b = fmla(self.twiddle7.im, x118n.im, -self.twiddle5.im * x217n.im)
                + fmla(self.twiddle2.im, x316n.im, self.twiddle9.im * x415n.im)
                + fmla(-self.twiddle3.im, x514n.im, self.twiddle4.im * x613n.im)
                + fmla(-self.twiddle8.im, x712n.im, -self.twiddle1.im * x811n.im)
                + self.twiddle6.im * x910n.im;
            let b811re_a = fmla(self.twiddle8.re, x118p.re, u0.re)
                + fmla(self.twiddle3.re, x217p.re, self.twiddle5.re * x316p.re)
                + fmla(self.twiddle6.re, x415p.re, self.twiddle2.re * x514p.re)
                + fmla(self.twiddle9.re, x613p.re, self.twiddle1.re * x712p.re)
                + fmla(self.twiddle7.re, x811p.re, self.twiddle4.re * x910p.re);
            let b811re_b = fmla(self.twiddle8.im, x118n.im, -self.twiddle3.im * x217n.im)
                + fmla(self.twiddle5.im, x316n.im, -self.twiddle6.im * x415n.im)
                + fmla(self.twiddle2.im, x514n.im, -self.twiddle9.im * x613n.im)
                + fmla(-self.twiddle1.im, x712n.im, self.twiddle7.im * x811n.im)
                + -self.twiddle4.im * x910n.im;
            let b910re_a = fmla(self.twiddle9.re, x118p.re, u0.re)
                + fmla(self.twiddle1.re, x217p.re, self.twiddle8.re * x316p.re)
                + fmla(self.twiddle2.re, x415p.re, self.twiddle7.re * x514p.re)
                + fmla(self.twiddle3.re, x613p.re, self.twiddle6.re * x712p.re)
                + fmla(self.twiddle4.re, x811p.re, self.twiddle5.re * x910p.re);
            let b910re_b = fmla(self.twiddle9.im, x118n.im, -self.twiddle1.im * x217n.im)
                + fmla(self.twiddle8.im, x316n.im, -self.twiddle2.im * x415n.im)
                + fmla(self.twiddle7.im, x514n.im, -self.twiddle3.im * x613n.im)
                + fmla(self.twiddle6.im, x712n.im, -self.twiddle4.im * x811n.im)
                + self.twiddle5.im * x910n.im;

            let b118im_a = fmla(self.twiddle1.re, x118p.im, u0.im)
                + fmla(self.twiddle2.re, x217p.im, self.twiddle3.re * x316p.im)
                + fmla(self.twiddle4.re, x415p.im, self.twiddle5.re * x514p.im)
                + fmla(self.twiddle6.re, x613p.im, self.twiddle7.re * x712p.im)
                + fmla(self.twiddle8.re, x811p.im, self.twiddle9.re * x910p.im);
            let b118im_b = fmla(self.twiddle1.im, x118n.re, self.twiddle2.im * x217n.re)
                + fmla(self.twiddle3.im, x316n.re, self.twiddle4.im * x415n.re)
                + fmla(self.twiddle5.im, x514n.re, self.twiddle6.im * x613n.re)
                + fmla(self.twiddle7.im, x712n.re, self.twiddle8.im * x811n.re)
                + self.twiddle9.im * x910n.re;
            let b217im_a = fmla(self.twiddle2.re, x118p.im, u0.im)
                + fmla(self.twiddle4.re, x217p.im, self.twiddle6.re * x316p.im)
                + fmla(self.twiddle8.re, x415p.im, self.twiddle9.re * x514p.im)
                + fmla(self.twiddle7.re, x613p.im, self.twiddle5.re * x712p.im)
                + fmla(self.twiddle3.re, x811p.im, self.twiddle1.re * x910p.im);
            let b217im_b = fmla(self.twiddle2.im, x118n.re, self.twiddle4.im * x217n.re)
                + fmla(self.twiddle6.im, x316n.re, self.twiddle8.im * x415n.re)
                + fmla(-self.twiddle9.im, x514n.re, -self.twiddle7.im * x613n.re)
                + fmla(-self.twiddle5.im, x712n.re, -self.twiddle3.im * x811n.re)
                + -self.twiddle1.im * x910n.re;
            let b316im_a = fmla(self.twiddle3.re, x118p.im, u0.im)
                + fmla(self.twiddle6.re, x217p.im, self.twiddle9.re * x316p.im)
                + fmla(self.twiddle7.re, x415p.im, self.twiddle4.re * x514p.im)
                + fmla(self.twiddle1.re, x613p.im, self.twiddle2.re * x712p.im)
                + fmla(self.twiddle5.re, x811p.im, self.twiddle8.re * x910p.im);
            let b316im_b = fmla(self.twiddle3.im, x118n.re, self.twiddle6.im * x217n.re)
                + fmla(self.twiddle9.im, x316n.re, -self.twiddle7.im * x415n.re)
                + fmla(-self.twiddle4.im, x514n.re, -self.twiddle1.im * x613n.re)
                + fmla(self.twiddle2.im, x712n.re, self.twiddle5.im * x811n.re)
                + self.twiddle8.im * x910n.re;
            let b415im_a = fmla(self.twiddle4.re, x118p.im, u0.im)
                + fmla(self.twiddle8.re, x217p.im, self.twiddle7.re * x316p.im)
                + fmla(self.twiddle3.re, x415p.im, self.twiddle1.re * x514p.im)
                + fmla(self.twiddle5.re, x613p.im, self.twiddle9.re * x712p.im)
                + fmla(self.twiddle6.re, x811p.im, self.twiddle2.re * x910p.im);
            let b415im_b = self.twiddle4.im * x118n.re
                + fmla(self.twiddle8.im, x217n.re, -self.twiddle7.im * x316n.re)
                + fmla(-self.twiddle3.im, x415n.re, self.twiddle1.im * x514n.re)
                + fmla(self.twiddle5.im, x613n.re, self.twiddle9.im * x712n.re)
                + fmla(-self.twiddle6.im, x811n.re, -self.twiddle2.im * x910n.re);
            let b514im_a = fmla(self.twiddle5.re, x118p.im, u0.im)
                + fmla(self.twiddle9.re, x217p.im, self.twiddle4.re * x316p.im)
                + fmla(self.twiddle1.re, x415p.im, self.twiddle6.re * x514p.im)
                + fmla(self.twiddle8.re, x613p.im, self.twiddle3.re * x712p.im)
                + fmla(self.twiddle2.re, x811p.im, self.twiddle7.re * x910p.im);
            let b514im_b = self.twiddle5.im * x118n.re
                + fmla(-self.twiddle9.im, x217n.re, -self.twiddle4.im * x316n.re)
                + fmla(self.twiddle1.im, x415n.re, self.twiddle6.im * x514n.re)
                + fmla(-self.twiddle8.im, x613n.re, -self.twiddle3.im * x712n.re)
                + fmla(self.twiddle2.im, x811n.re, self.twiddle7.im * x910n.re);
            let b613im_a = fmla(self.twiddle6.re, x118p.im, u0.im)
                + fmla(self.twiddle7.re, x217p.im, self.twiddle1.re * x316p.im)
                + fmla(self.twiddle5.re, x415p.im, self.twiddle8.re * x514p.im)
                + fmla(self.twiddle2.re, x613p.im, self.twiddle4.re * x712p.im)
                + fmla(self.twiddle9.re, x811p.im, self.twiddle3.re * x910p.im);
            let b613im_b = fmla(self.twiddle6.im, x118n.re, -self.twiddle7.im * x217n.re)
                + fmla(-self.twiddle1.im, x316n.re, self.twiddle5.im * x415n.re)
                + fmla(-self.twiddle8.im, x514n.re, -self.twiddle2.im * x613n.re)
                + fmla(self.twiddle4.im, x712n.re, -self.twiddle9.im * x811n.re)
                + -self.twiddle3.im * x910n.re;
            let b712im_a = fmla(self.twiddle7.re, x118p.im, u0.im)
                + fmla(self.twiddle5.re, x217p.im, self.twiddle2.re * x316p.im)
                + fmla(self.twiddle9.re, x415p.im, self.twiddle3.re * x514p.im)
                + fmla(self.twiddle4.re, x613p.im, self.twiddle8.re * x712p.im)
                + fmla(self.twiddle1.re, x811p.im, self.twiddle6.re * x910p.im);
            let b712im_b = fmla(self.twiddle7.im, x118n.re, -self.twiddle5.im * x217n.re)
                + fmla(self.twiddle2.im, x316n.re, self.twiddle9.im * x415n.re)
                + fmla(-self.twiddle3.im, x514n.re, self.twiddle4.im * x613n.re)
                + fmla(-self.twiddle8.im, x712n.re, -self.twiddle1.im * x811n.re)
                + self.twiddle6.im * x910n.re;
            let b811im_a = fmla(self.twiddle8.re, x118p.im, u0.im)
                + fmla(self.twiddle3.re, x217p.im, self.twiddle5.re * x316p.im)
                + fmla(self.twiddle6.re, x415p.im, self.twiddle2.re * x514p.im)
                + fmla(self.twiddle9.re, x613p.im, self.twiddle1.re * x712p.im)
                + fmla(self.twiddle7.re, x811p.im, self.twiddle4.re * x910p.im);
            let b811im_b = fmla(self.twiddle8.im, x118n.re, -self.twiddle3.im * x217n.re)
                + fmla(self.twiddle5.im, x316n.re, -self.twiddle6.im * x415n.re)
                + fmla(self.twiddle2.im, x514n.re, -self.twiddle9.im * x613n.re)
                + fmla(-self.twiddle1.im, x712n.re, self.twiddle7.im * x811n.re)
                + -self.twiddle4.im * x910n.re;
            let b910im_a = fmla(self.twiddle9.re, x118p.im, u0.im)
                + fmla(self.twiddle1.re, x217p.im, self.twiddle8.re * x316p.im)
                + fmla(self.twiddle2.re, x415p.im, self.twiddle7.re * x514p.im)
                + fmla(self.twiddle3.re, x613p.im, self.twiddle6.re * x712p.im)
                + fmla(self.twiddle4.re, x811p.im, self.twiddle5.re * x910p.im);
            let b910im_b = fmla(self.twiddle9.im, x118n.re, -self.twiddle1.im * x217n.re)
                + fmla(self.twiddle8.im, x316n.re, -self.twiddle2.im * x415n.re)
                + fmla(self.twiddle7.im, x514n.re, -self.twiddle3.im * x613n.re)
                + fmla(self.twiddle6.im, x712n.re, -self.twiddle4.im * x811n.re)
                + self.twiddle5.im * x910n.re;

            let out1re = b118re_a - b118re_b;
            let out1im = b118im_a + b118im_b;
            chunk[1] = Complex {
                re: out1re,
                im: out1im,
            };
            let out2re = b217re_a - b217re_b;
            let out2im = b217im_a + b217im_b;
            chunk[2] = Complex {
                re: out2re,
                im: out2im,
            };
            let out3re = b316re_a - b316re_b;
            let out3im = b316im_a + b316im_b;
            chunk[3] = Complex {
                re: out3re,
                im: out3im,
            };
            let out4re = b415re_a - b415re_b;
            let out4im = b415im_a + b415im_b;
            chunk[4] = Complex {
                re: out4re,
                im: out4im,
            };
            let out5re = b514re_a - b514re_b;
            let out5im = b514im_a + b514im_b;
            chunk[5] = Complex {
                re: out5re,
                im: out5im,
            };
            let out6re = b613re_a - b613re_b;
            let out6im = b613im_a + b613im_b;
            chunk[6] = Complex {
                re: out6re,
                im: out6im,
            };
            let out7re = b712re_a - b712re_b;
            let out7im = b712im_a + b712im_b;
            chunk[7] = Complex {
                re: out7re,
                im: out7im,
            };
            let out8re = b811re_a - b811re_b;
            let out8im = b811im_a + b811im_b;
            chunk[8] = Complex {
                re: out8re,
                im: out8im,
            };
            let out9re = b910re_a - b910re_b;
            let out9im = b910im_a + b910im_b;
            chunk[9] = Complex {
                re: out9re,
                im: out9im,
            };
            let out10re = b910re_a + b910re_b;
            let out10im = b910im_a - b910im_b;
            chunk[10] = Complex {
                re: out10re,
                im: out10im,
            };
            let out11re = b811re_a + b811re_b;
            let out11im = b811im_a - b811im_b;
            chunk[11] = Complex {
                re: out11re,
                im: out11im,
            };
            let out12re = b712re_a + b712re_b;
            let out12im = b712im_a - b712im_b;
            chunk[12] = Complex {
                re: out12re,
                im: out12im,
            };
            let out13re = b613re_a + b613re_b;
            let out13im = b613im_a - b613im_b;
            chunk[13] = Complex {
                re: out13re,
                im: out13im,
            };
            let out14re = b514re_a + b514re_b;
            let out14im = b514im_a - b514im_b;
            chunk[14] = Complex {
                re: out14re,
                im: out14im,
            };
            let out15re = b415re_a + b415re_b;
            let out15im = b415im_a - b415im_b;
            chunk[15] = Complex {
                re: out15re,
                im: out15im,
            };
            let out16re = b316re_a + b316re_b;
            let out16im = b316im_a - b316im_b;
            chunk[16] = Complex {
                re: out16re,
                im: out16im,
            };
            let out17re = b217re_a + b217re_b;
            let out17im = b217im_a - b217im_b;
            chunk[17] = Complex {
                re: out17re,
                im: out17im,
            };
            let out18re = b118re_a + b118re_b;
            let out18im = b118im_a - b118im_b;
            chunk[18] = Complex {
                re: out18re,
                im: out18im,
            };
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        19
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;

    test_butterfly!(test_butterfly18, f32, Butterfly19, 19, 1e-5);
}
