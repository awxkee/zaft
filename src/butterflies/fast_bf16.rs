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
use crate::butterflies::fast_bf8::FastButterfly8;
use crate::butterflies::rotate_90;
use crate::butterflies::short_butterflies::{FastButterfly2, FastButterfly4};
use crate::complex_fma::{c_mul_fast, c_mul_fast_conj};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct FastButterfly16<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    bf4: FastButterfly4<T>,
    pub(crate) bf2: FastButterfly2<T>,
    pub(crate) bf8: FastButterfly8<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> FastButterfly16<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        FastButterfly16 {
            direction: fft_direction,
            bf8: FastButterfly8::new(fft_direction),
            bf4: FastButterfly4::new(fft_direction),
            bf2: FastButterfly2::new(fft_direction),
            twiddle1: compute_twiddle(1, 16, fft_direction),
            twiddle2: compute_twiddle(2, 16, fft_direction),
            twiddle3: compute_twiddle(3, 16, fft_direction),
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
> FastButterfly16<T>
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
        u7: Complex<T>,
        u8: Complex<T>,
        u9: Complex<T>,
        u10: Complex<T>,
        u11: Complex<T>,
        u12: Complex<T>,
        u13: Complex<T>,
        u14: Complex<T>,
        u15: Complex<T>,
    ) -> (
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
    ) {
        let evens = self.bf8.exec(u0, u2, u4, u6, u8, u10, u12, u14);

        let mut odds_1 = self.bf4.butterfly4(u1, u5, u9, u13);
        let mut odds_2 = self.bf4.butterfly4(u15, u3, u7, u11);

        odds_1.1 = c_mul_fast(odds_1.1, self.twiddle1);
        odds_2.1 = c_mul_fast_conj(odds_2.1, self.twiddle1);

        odds_1.2 = c_mul_fast(odds_1.2, self.twiddle2);
        odds_2.2 = c_mul_fast_conj(odds_2.2, self.twiddle2);

        odds_1.3 = c_mul_fast(odds_1.3, self.twiddle3);
        odds_2.3 = c_mul_fast_conj(odds_2.3, self.twiddle3);

        // step 4: cross FFTs
        let (o01, o02) = self.bf2.butterfly2(odds_1.0, odds_2.0);
        odds_1.0 = o01;
        odds_2.0 = o02;

        let (o03, o04) = self.bf2.butterfly2(odds_1.1, odds_2.1);
        odds_1.1 = o03;
        odds_2.1 = o04;
        let (o05, o06) = self.bf2.butterfly2(odds_1.2, odds_2.2);
        odds_1.2 = o05;
        odds_2.2 = o06;
        let (o07, o08) = self.bf2.butterfly2(odds_1.3, odds_2.3);
        odds_1.3 = o07;
        odds_2.3 = o08;

        // apply the butterfly 4 twiddle factor, which is just a rotation
        odds_2.0 = rotate_90(odds_2.0, self.direction);
        odds_2.1 = rotate_90(odds_2.1, self.direction);
        odds_2.2 = rotate_90(odds_2.2, self.direction);
        odds_2.3 = rotate_90(odds_2.3, self.direction);

        let y0 = evens.0 + odds_1.0;
        let y1 = evens.1 + odds_1.1;
        let y2 = evens.2 + odds_1.2;
        let y3 = evens.3 + odds_1.3;
        let y4 = evens.4 + odds_2.0;
        let y5 = evens.5 + odds_2.1;
        let y6 = evens.6 + odds_2.2;
        let y7 = evens.7 + odds_2.3;
        let y8 = evens.0 - odds_1.0;
        let y9 = evens.1 - odds_1.1;
        let y10 = evens.2 - odds_1.2;
        let y11 = evens.3 - odds_1.3;
        let y12 = evens.4 - odds_2.0;
        let y13 = evens.5 - odds_2.1;
        let y14 = evens.6 - odds_2.2;
        let y15 = evens.7 - odds_2.3;
        (
            y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15,
        )
    }
}
