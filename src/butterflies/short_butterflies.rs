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
use crate::butterflies::rotate_90;
use crate::mla::fmla;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct FastButterfly2<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
}

#[allow(unused)]
impl<T: Default + Clone + 'static + Copy + Float> FastButterfly2<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            phantom_data: PhantomData,
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
        + Default,
> FastButterfly2<T>
where
    f64: AsPrimitive<T>,
{
    #[inline]
    pub(crate) fn butterfly2(&self, u0: Complex<T>, u1: Complex<T>) -> (Complex<T>, Complex<T>) {
        let t = u0 + u1;

        let y1 = u0 - u1;
        let y0 = t;
        (y0, y1)
    }
}

#[allow(unused)]
pub(crate) struct FastButterfly3<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
    twiddle: Complex<T>,
}

#[allow(unused)]
impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> FastButterfly3<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            phantom_data: PhantomData,
            twiddle: compute_twiddle(1, 3, fft_direction),
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
        + Default,
> FastButterfly3<T>
where
    f64: AsPrimitive<T>,
{
    #[inline]
    pub(crate) fn butterfly3(
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
}

#[allow(unused)]
pub(crate) struct FastButterfly4<T> {
    direction: FftDirection,
    twiddle: Complex<T>,
}

#[allow(unused)]
impl<T: Default + Clone + 'static + Copy + Float> FastButterfly4<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle: match fft_direction {
                FftDirection::Inverse => Complex::new(T::zero(), -T::one()),
                FftDirection::Forward => Complex::new(T::zero(), T::one()),
            },
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
        + MulAdd<T, Output = T>,
> FastButterfly4<T>
where
    f64: AsPrimitive<T>,
{
    #[inline]
    pub(crate) fn butterfly4(
        &self,
        a: Complex<T>,
        b: Complex<T>,
        c: Complex<T>,
        d: Complex<T>,
    ) -> (Complex<T>, Complex<T>, Complex<T>, Complex<T>) {
        let t0 = a + c;
        let t1 = a - c;
        let t2 = b + d;
        let z3 = b - d;
        let t3 = rotate_90(z3, self.direction);

        let y0 = t0 + t2;
        let y1 = t1 + t3;
        let y2 = t0 - t2;
        let y3 = t1 - t3;
        (y0, y1, y2, y3)
    }
}
