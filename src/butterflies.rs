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
use crate::mla::fmla;
use crate::radix6::Radix6Twiddles;
use crate::short_butterflies::{FastButterfly2, FastButterfly3, FastButterfly4};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};

pub(crate) struct Butterfly1<T> {
    pub(crate) phantom_data: PhantomData<T>,
    pub(crate) direction: FftDirection,
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
> FftExecutor<T> for Butterfly1<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, _: &mut [Complex<T>]) -> Result<(), ZaftError> {
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        1
    }
}

#[allow(unused)]
pub(crate) struct Butterfly2<T> {
    pub(crate) phantom_data: PhantomData<T>,
    pub(crate) direction: FftDirection,
}

#[allow(unused)]
impl<T> Butterfly2<T> {
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
        + MulAdd<T, Output = T>,
> FftExecutor<T> for Butterfly2<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if in_place.len() % 2 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(2) {
            let u0 = chunk[0];
            let u1 = chunk[1];

            let y0 = u0 + u1;
            let y1 = u0 - u1;

            chunk[0] = y0;
            chunk[1] = y1;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        2
    }
}

#[allow(unused)]
pub(crate) struct Butterfly3<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
    twiddle: Complex<T>,
}

#[allow(unused)]
impl<T: Default + Clone + Radix6Twiddles + 'static + Copy + FftTrigonometry + Float> Butterfly3<T>
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
        + MulAdd<T, Output = T>,
> FftExecutor<T> for Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if in_place.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(3) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];

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

            chunk[0] = y0;
            chunk[1] = y1;
            chunk[2] = y2;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        3
    }
}

#[allow(unused)]
pub(crate) struct Butterfly4<T> {
    direction: FftDirection,
    twiddle: Complex<T>,
}

pub(crate) fn rotate_90<T: Copy + Neg<Output = T>>(
    value: Complex<T>,
    direction: FftDirection,
) -> Complex<T> {
    match direction {
        FftDirection::Forward => Complex {
            re: value.im,
            im: -value.re,
        },
        FftDirection::Inverse => Complex {
            re: -value.im,
            im: value.re,
        },
    }
}

#[allow(unused)]
impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> Butterfly4<T>
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
> FftExecutor<T> for Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(4) {
            let a = chunk[0];
            let b = chunk[1];
            let c = chunk[2];
            let d = chunk[3];

            let t0 = a + c;
            let t1 = a - c;
            let t2 = b + d;
            let z3 = b - d;
            let t3 = rotate_90(z3, self.direction);

            chunk[0] = t0 + t2;
            chunk[1] = t1 + t3;
            chunk[2] = t0 - t2;
            chunk[3] = t1 - t3;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        4
    }
}

#[allow(unused)]
pub(crate) struct Butterfly5<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly5 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 5, fft_direction),
            twiddle2: compute_twiddle(2, 5, fft_direction),
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
> FftExecutor<T> for Butterfly5<T>
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

        for chunk in in_place.chunks_exact_mut(5) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];
            let u4 = chunk[4];

            // Radix-5 butterfly

            let x14p = u1 + u4;
            let x14n = u1 - u4;
            let x23p = u2 + u3;
            let x23n = u2 - u3;
            let y0 = u0 + x14p + x23p;

            let b14re_a = fmla(
                self.twiddle2.re,
                x23p.re,
                fmla(self.twiddle1.re, x14p.re, u0.re),
            );
            let b14re_b = fmla(self.twiddle1.im, x14n.im, self.twiddle2.im * x23n.im);
            let b23re_a = fmla(
                self.twiddle1.re,
                x23p.re,
                fmla(self.twiddle2.re, x14p.re, u0.re),
            );
            let b23re_b = fmla(self.twiddle2.im, x14n.im, -self.twiddle1.im * x23n.im);

            let b14im_a = fmla(
                self.twiddle2.re,
                x23p.im,
                fmla(self.twiddle1.re, x14p.im, u0.im),
            );
            let b14im_b = fmla(self.twiddle1.im, x14n.re, self.twiddle2.im * x23n.re);
            let b23im_a = fmla(
                self.twiddle1.re,
                x23p.im,
                fmla(self.twiddle2.re, x14p.im, u0.im),
            );
            let b23im_b = fmla(self.twiddle2.im, x14n.re, -self.twiddle1.im * x23n.re);

            let y1 = Complex {
                re: b14re_a - b14re_b,
                im: b14im_a + b14im_b,
            };
            let y2 = Complex {
                re: b23re_a - b23re_b,
                im: b23im_a + b23im_b,
            };
            let y3 = Complex {
                re: b23re_a + b23re_b,
                im: b23im_a - b23im_b,
            };
            let y4 = Complex {
                re: b14re_a + b14re_b,
                im: b14im_a - b14im_b,
            };

            chunk[0] = y0;
            chunk[1] = y1;
            chunk[2] = y2;
            chunk[3] = y3;
            chunk[4] = y4;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        5
    }
}

#[allow(unused)]
pub(crate) struct Butterfly6<T> {
    direction: FftDirection,
    twiddle: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly6 {
            direction: fft_direction,
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
        + FftTrigonometry
        + Float
        + Default,
> FftExecutor<T> for Butterfly6<T>
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

        let fast_butterfly3 = FastButterfly3::new(self.direction);
        let fast_butterfly2 = FastButterfly2::new(self.direction);

        for chunk in in_place.chunks_exact_mut(6) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];
            let u4 = chunk[4];
            let u5 = chunk[5];

            // Radix-6 butterfly

            let (t0, t2, t4) = fast_butterfly3.butterfly3(u0, u2, u4);
            let (t1, t3, t5) = fast_butterfly3.butterfly3(u3, u5, u1);
            let (y0, y3) = fast_butterfly2.butterfly2(t0, t1);
            let (y4, y1) = fast_butterfly2.butterfly2(t2, t3);
            let (y2, y5) = fast_butterfly2.butterfly2(t4, t5);

            chunk[0] = y0;
            chunk[1] = y1;
            chunk[2] = y2;
            chunk[3] = y3;
            chunk[4] = y4;
            chunk[5] = y5;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        6
    }
}

#[allow(unused)]
pub(crate) struct Butterfly7<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> Butterfly7<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly7 {
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
        + MulAdd<T, Output = T>,
> FftExecutor<T> for Butterfly7<T>
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

        for chunk in in_place.chunks_exact_mut(7) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];
            let u4 = chunk[4];
            let u5 = chunk[5];
            let u6 = chunk[6];

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

            chunk[0] = y0;
            chunk[1] = y1;
            chunk[2] = y2;
            chunk[3] = y3;
            chunk[4] = y4;
            chunk[5] = y5;
            chunk[6] = y6;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        7
    }
}

#[allow(unused)]
pub(crate) struct Butterfly8<T> {
    direction: FftDirection,
    root2: T,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly8 {
            direction: fft_direction,
            root2: (0.5f64.sqrt()).as_(),
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
> FftExecutor<T> for Butterfly8<T>
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

        let bf4 = FastButterfly4::new(self.direction);
        let bf2 = FastButterfly2::new(self.direction);

        for chunk in in_place.chunks_exact_mut(8) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];
            let u4 = chunk[4];
            let u5 = chunk[5];
            let u6 = chunk[6];
            let u7 = chunk[7];

            // Radix-8 butterfly
            let (u0, u2, u4, u6) = bf4.butterfly4(u0, u2, u4, u6);
            let (u1, mut u3, mut u5, mut u7) = bf4.butterfly4(u1, u3, u5, u7);

            u3 = (rotate_90(u3, self.direction) + u3) * self.root2;
            u5 = rotate_90(u5, self.direction);
            u7 = (rotate_90(u7, self.direction) - u7) * self.root2;

            let (u0, u1) = bf2.butterfly2(u0, u1);
            let (u2, u3) = bf2.butterfly2(u2, u3);
            let (u4, u5) = bf2.butterfly2(u4, u5);
            let (u6, u7) = bf2.butterfly2(u6, u7);

            chunk[0] = u0;
            chunk[1] = u2;
            chunk[2] = u4;
            chunk[3] = u6;
            chunk[4] = u1;
            chunk[5] = u3;
            chunk[6] = u5;
            chunk[7] = u7;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        8
    }
}

#[allow(unused)]
pub(crate) struct Butterfly9<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle4: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly9 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 9, fft_direction),
            twiddle2: compute_twiddle(2, 9, fft_direction),
            twiddle4: compute_twiddle(4, 9, fft_direction),
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
> FftExecutor<T> for Butterfly9<T>
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

        let bf3 = FastButterfly3::new(self.direction);

        for chunk in in_place.chunks_exact_mut(9) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];
            let u4 = chunk[4];
            let u5 = chunk[5];
            let u6 = chunk[6];
            let u7 = chunk[7];
            let u8 = chunk[8];

            // Radix-9 butterfly

            let (u0, u3, u6) = bf3.butterfly3(u0, u3, u6);
            let (u1, mut u4, mut u7) = bf3.butterfly3(u1, u4, u7);
            let (u2, mut u5, mut u8) = bf3.butterfly3(u2, u5, u8);

            u4 = c_mul_fast(u4, self.twiddle1);
            u7 = c_mul_fast(u7, self.twiddle2);
            u5 = c_mul_fast(u5, self.twiddle2);
            u8 = c_mul_fast(u8, self.twiddle4);

            let (zu0, zu3, zu6) = bf3.butterfly3(u0, u1, u2);
            let (zu1, zu4, zu7) = bf3.butterfly3(u3, u4, u5);
            let (zu2, zu5, zu8) = bf3.butterfly3(u6, u7, u8);

            chunk[0] = zu0;
            chunk[1] = zu1;
            chunk[2] = zu2;

            chunk[3] = zu3;
            chunk[4] = zu4;
            chunk[5] = zu5;

            chunk[6] = zu6;
            chunk[7] = zu7;
            chunk[8] = zu8;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        9
    }
}

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

#[allow(unused)]
pub(crate) struct Butterfly12<T> {
    direction: FftDirection,
    phantom_data: PhantomData<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly12 {
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
        + Float
        + Default
        + FftTrigonometry,
> FftExecutor<T> for Butterfly12<T>
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

        let bf3 = FastButterfly3::new(self.direction);
        let bf4 = FastButterfly4::new(self.direction);

        for chunk in in_place.chunks_exact_mut(12) {
            let u0 = chunk[0];
            let u1 = chunk[3];
            let u2 = chunk[6];
            let u3 = chunk[9];

            let u4 = chunk[4];
            let u5 = chunk[7];
            let u6 = chunk[10];
            let u7 = chunk[1];

            let u8 = chunk[8];
            let u9 = chunk[11];
            let u10 = chunk[2];
            let u11 = chunk[5];

            let (u0, u1, u2, u3) = bf4.butterfly4(u0, u1, u2, u3);
            let (u4, u5, u6, u7) = bf4.butterfly4(u4, u5, u6, u7);
            let (u8, u9, u10, u11) = bf4.butterfly4(u8, u9, u10, u11);

            let (v0, v4, v8) = bf3.butterfly3(u0, u4, u8);
            let (v1, v5, v9) = bf3.butterfly3(u1, u5, u9);
            let (v2, v6, v10) = bf3.butterfly3(u2, u6, u10);
            let (v3, v7, v11) = bf3.butterfly3(u3, u7, u11);

            chunk[0] = v0;
            chunk[1] = v5;
            chunk[2] = v10;
            chunk[3] = v3;
            chunk[4] = v4;
            chunk[5] = v9;
            chunk[6] = v2;
            chunk[7] = v7;
            chunk[8] = v8;
            chunk[9] = v1;
            chunk[10] = v6;
            chunk[11] = v11;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        12
    }
}

#[allow(unused)]
pub(crate) struct Butterfly13<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static> Butterfly13<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly13 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 13, fft_direction),
            twiddle2: compute_twiddle(2, 13, fft_direction),
            twiddle3: compute_twiddle(3, 13, fft_direction),
            twiddle4: compute_twiddle(4, 13, fft_direction),
            twiddle5: compute_twiddle(5, 13, fft_direction),
            twiddle6: compute_twiddle(6, 13, fft_direction),
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
> FftExecutor<T> for Butterfly13<T>
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

        for chunk in in_place.chunks_exact_mut(13) {
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
            chunk[0] = y0;
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
                        + fmla(self.twiddle5.im, x58n.im, self.twiddle6.im * x67n.im),
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
                        + fmla(self.twiddle2.im, x58n.im, self.twiddle5.im * x67n.im),
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
                            fmla(-self.twiddle1.im, x58n.im, self.twiddle4.im * x67n.im),
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
                            fmla(self.twiddle4.im, x58n.im, -self.twiddle3.im * x67n.im),
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
                        + fmla(self.twiddle5.im, x58n.re, self.twiddle6.im * x67n.re),
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
                        fmla(-self.twiddle3.im, x58n.re, -self.twiddle1.im * x67n.re),
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
                        + fmla(-self.twiddle1.im, x58n.re, self.twiddle4.im * x67n.re),
                ),
            );
            let b67im_a = fmla(
                self.twiddle6.re,
                x112p.im,
                u0.im
                    + fmla(
                        self.twiddle2.re,
                        x49p.im,
                        fmla(self.twiddle1.re, x211p.im, self.twiddle5.re * x310p.im),
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
                            self.twiddle4.im * x58n.re + -self.twiddle3.im * x67n.re,
                        ),
                    ),
                ),
            );

            chunk[1] = Complex {
                re: b112re_a - b112re_b,
                im: b112im_a + b112im_b,
            };
            chunk[2] = Complex {
                re: b211re_a - b211re_b,
                im: b211im_a + b211im_b,
            };
            chunk[3] = Complex {
                re: b310re_a - b310re_b,
                im: b310im_a + b310im_b,
            };
            chunk[4] = Complex {
                re: b49re_a - b49re_b,
                im: b49im_a + b49im_b,
            };
            chunk[5] = Complex {
                re: b58re_a - b58re_b,
                im: b58im_a + b58im_b,
            };
            chunk[6] = Complex {
                re: b67re_a - b67re_b,
                im: b67im_a + b67im_b,
            };
            chunk[7] = Complex {
                re: b67re_a + b67re_b,
                im: b67im_a - b67im_b,
            };
            chunk[8] = Complex {
                re: b58re_a + b58re_b,
                im: b58im_a - b58im_b,
            };
            chunk[9] = Complex {
                re: b49re_a + b49re_b,
                im: b49im_a - b49im_b,
            };
            chunk[10] = Complex {
                re: b310re_a + b310re_b,
                im: b310im_a - b310im_b,
            };
            chunk[11] = Complex {
                re: b211re_a + b211re_b,
                im: b211im_a - b211im_b,
            };
            chunk[12] = Complex {
                re: b112re_a + b112re_b,
                im: b112im_a - b112im_b,
            };
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        13
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_butterfly2() {
        for i in 1..6 {
            let size = 2usize.pow(i);

            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly2::new(FftDirection::Forward);
            let radix_inverse = Butterfly2::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 2f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly3() {
        for i in 1..6 {
            let size = 3usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly3::new(FftDirection::Forward);
            let radix_inverse = Butterfly3::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 3f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly4() {
        for i in 1..6 {
            let size = 4usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly4::new(FftDirection::Forward);
            let radix_inverse = Butterfly4::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 4f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly5() {
        for i in 1..6 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly5::new(FftDirection::Forward);
            let radix_inverse = Butterfly5::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 5f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly6() {
        for i in 1..6 {
            let size = 6usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly6::new(FftDirection::Forward);
            let radix_inverse = Butterfly6::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 6f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly7() {
        for i in 1..5 {
            let size = 7usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly7::new(FftDirection::Forward);
            let radix_inverse = Butterfly7::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 7f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly8() {
        for i in 1..5 {
            let size = 8usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly8::new(FftDirection::Forward);
            let radix_inverse = Butterfly8::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 8f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly9() {
        for i in 1..5 {
            let size = 9usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly9::new(FftDirection::Forward);
            let radix_inverse = Butterfly9::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 9f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly12() {
        for i in 1..4 {
            let size = 11usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly11::new(FftDirection::Forward);
            let radix_inverse = Butterfly11::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 11f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly11() {
        for i in 1..4 {
            let size = 12usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly12::new(FftDirection::Forward);
            let radix_inverse = Butterfly12::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 12f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly13() {
        for i in 1..4 {
            let size = 13usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = Butterfly13::new(FftDirection::Forward);
            let radix_inverse = Butterfly13::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 13f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }
}
