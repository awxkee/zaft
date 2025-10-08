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
use crate::FftDirection;
use crate::mla::fmla;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct FastButterfly7<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> FastButterfly7<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        FastButterfly7 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 7, fft_direction),
            twiddle2: compute_twiddle(2, 7, fft_direction),
            twiddle3: compute_twiddle(3, 7, fft_direction),
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
        + Default,
> FastButterfly7<T>
where
    f64: AsPrimitive<T>,
{
    #[inline]
    pub(crate) fn exec(
        &self,
        u0: Complex<T>,
        u1: Complex<T>,
        u2: Complex<T>,
        u3: Complex<T>,
        u4: Complex<T>,
        u5: Complex<T>,
        u6: Complex<T>,
    ) -> (
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
    ) {
        // Radix-7 butterfly

        let x16p = u1 + u6;
        let x16n = u1 - u6;
        let x25p = u2 + u5;
        let x25n = u2 - u5;
        let x34p = u3 + u4;
        let x34n = u3 - u4;
        let y0 = u0 + x16p + x25p + x34p;

        let x16re_a = fmla(
            self.twiddle1.re,
            x16p.re,
            u0.re + self.twiddle2.re * x25p.re + self.twiddle3.re * x34p.re,
        );
        let x16re_b = fmla(
            self.twiddle1.im,
            x16n.im,
            fmla(self.twiddle2.im, x25n.im, self.twiddle3.im * x34n.im),
        );
        let x25re_a = fmla(
            self.twiddle1.re,
            x34p.re,
            u0.re + self.twiddle2.re * x16p.re + self.twiddle3.re * x25p.re,
        );
        let x25re_b = fmla(
            -self.twiddle1.im,
            x34n.im,
            fmla(self.twiddle2.im, x16n.im, -self.twiddle3.im * x25n.im),
        );
        let x34re_a = fmla(
            self.twiddle1.re,
            x25p.re,
            u0.re + self.twiddle2.re * x34p.re + self.twiddle3.re * x16p.re,
        );
        let x34re_b = fmla(
            -self.twiddle1.im,
            x25n.im,
            fmla(self.twiddle2.im, x34n.im, self.twiddle3.im * x16n.im),
        );
        let x16im_a = fmla(
            self.twiddle1.re,
            x16p.im,
            u0.im + self.twiddle2.re * x25p.im + self.twiddle3.re * x34p.im,
        );
        let x16im_b = fmla(
            self.twiddle1.im,
            x16n.re,
            fmla(self.twiddle2.im, x25n.re, self.twiddle3.im * x34n.re),
        );
        let x25im_a = fmla(
            self.twiddle1.re,
            x34p.im,
            u0.im + self.twiddle2.re * x16p.im + self.twiddle3.re * x25p.im,
        );
        let x25im_b = fmla(
            -self.twiddle1.im,
            x34n.re,
            fmla(self.twiddle2.im, x16n.re, -self.twiddle3.im * x25n.re),
        );
        let x34im_a = fmla(
            self.twiddle1.re,
            x25p.im,
            u0.im + self.twiddle2.re * x34p.im + self.twiddle3.re * x16p.im,
        );
        let x34im_b = fmla(
            self.twiddle1.im,
            x25n.re,
            fmla(-self.twiddle2.im, x34n.re, -self.twiddle3.im * x16n.re),
        );

        let y1 = Complex {
            re: x16re_a - x16re_b,
            im: x16im_a + x16im_b,
        };
        let y2 = Complex {
            re: x25re_a - x25re_b,
            im: x25im_a + x25im_b,
        };
        let y3 = Complex {
            re: x34re_a - x34re_b,
            im: x34im_a - x34im_b,
        };
        let y4 = Complex {
            re: x34re_a + x34re_b,
            im: x34im_a + x34im_b,
        };
        let y5 = Complex {
            re: x25re_a + x25re_b,
            im: x25im_a - x25im_b,
        };
        let y6 = Complex {
            re: x16re_a + x16re_b,
            im: x16im_a - x16im_b,
        };

        (y0, y1, y2, y3, y4, y5, y6)
    }
}
