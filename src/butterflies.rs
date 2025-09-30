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
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};

pub(crate) struct Butterfly2<T> {
    pub(crate) phantom_data: PhantomData<T>,
    pub(crate) direction: FftDirection,
}

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
        if in_place.len() != 2 {
            return Err(ZaftError::InvalidInPlaceLength(2, in_place.len()));
        }

        let u0 = unsafe { *in_place.get_unchecked(0) };
        let u1 = unsafe { *in_place.get_unchecked(1) };

        let t = u0 + u1;
        let y1 = u0 - u1;
        let y0 = t;

        unsafe {
            *in_place.get_unchecked_mut(0) = y0;
        }
        unsafe {
            *in_place.get_unchecked_mut(1) = y1;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

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
        if in_place.len() != 3 {
            return Err(ZaftError::InvalidInPlaceLength(3, in_place.len()));
        }

        let u0 = unsafe { *in_place.get_unchecked(0) };
        let u1 = unsafe { *in_place.get_unchecked(1) };
        let u2 = unsafe { *in_place.get_unchecked(2) };

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

        unsafe {
            *in_place.get_unchecked_mut(0) = y0;
        }
        unsafe {
            *in_place.get_unchecked_mut(1) = y1;
        }
        unsafe {
            *in_place.get_unchecked_mut(2) = y2;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        3
    }
}

#[allow(unused)]
pub(crate) struct Butterfly4<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
    twiddle: Complex<T>,
}

#[allow(unused)]
impl<T: Default + Clone + Radix6Twiddles + 'static + Copy + FftTrigonometry + Float> Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            phantom_data: PhantomData,
            twiddle: match fft_direction {
                FftDirection::Forward => Complex::new(T::zero(), -T::one()),
                FftDirection::Inverse => Complex::new(T::zero(), T::one()),
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
        if in_place.len() != 4 {
            return Err(ZaftError::InvalidInPlaceLength(
                self.length(),
                in_place.len(),
            ));
        }

        let a = unsafe { *in_place.get_unchecked(0) };
        let b = unsafe { *in_place.get_unchecked(1) };
        let c = unsafe { *in_place.get_unchecked(2) };
        let d = unsafe { *in_place.get_unchecked(3) };

        let t0 = a + c;
        let t1 = a - c;
        let t2 = b + d;
        let t3 = c_mul_fast(b - d, self.twiddle);

        unsafe {
            *in_place.get_unchecked_mut(0) = t0 + t2;
            *in_place.get_unchecked_mut(1) = t1 + t3;
            *in_place.get_unchecked_mut(2) = t0 - t2;
            *in_place.get_unchecked_mut(3) = t1 - t3;
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_butterfly2() {
        let size = 2usize;
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

        input = input
            .iter()
            .map(|&x| x * (1.0 / input.len() as f32))
            .collect();

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

    #[test]
    fn test_butterfly3() {
        let size = 3usize;
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

        input = input
            .iter()
            .map(|&x| x * (1.0 / input.len() as f32))
            .collect();

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

    #[test]
    fn test_butterfly4() {
        let size = 4usize;
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

        input = input
            .iter()
            .map(|&x| x * (1.0 / input.len() as f32))
            .collect();

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
