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
use crate::util::compute_twiddle;
use crate::{
    CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, FftSample,
    R2CFftExecutor, ZaftError,
};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;
use std::sync::Arc;

#[allow(unused)]
pub(crate) struct Butterfly3<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
    twiddle: Complex<T>,
}

#[allow(unused)]
impl<T: FftSample> Butterfly3<T>
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

impl<T: FftSample> FftExecutor<T> for Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(3) {
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

impl<T: FftSample> FftExecutorOutOfPlace<T> for Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(3) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(3) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(3).zip(src.chunks_exact(3)) {
            let u0 = src[0];
            let u1 = src[1];
            let u2 = src[2];

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

            dst[0] = y0;
            dst[1] = y1;
            dst[2] = y2;
        }
        Ok(())
    }
}

impl<T: FftSample> CompositeFftExecutor<T> for Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<T> + Send + Sync> {
        self
    }
}

impl<T: FftSample> Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn r_exec(&self, data: &[T; 3]) -> [Complex<T>; 2] {
        let a = data[0];
        let b = data[1];
        let c = data[2];

        let w1 = b + c;
        let w2 = b - c;
        let x0 = w1 + a;
        let x1 = fmla(w1, -T::HALF, a);

        [
            Complex::new(x0, T::zero()),
            Complex::new(x1, w2 * -T::SQRT_3_OVER_2),
        ]
    }
}

impl<T: FftSample> Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn exec(&self, data: &[Complex<T>; 3]) -> [Complex<T>; 3] {
        let u0 = data[0];
        let u1 = data[1];
        let u2 = data[2];

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
        [y0, y1, y2]
    }
}

impl<T: FftSample> R2CFftExecutor<T> for Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !input.len().is_multiple_of(3) {
            return Err(ZaftError::InvalidSizeMultiplier(
                input.len(),
                self.real_length(),
            ));
        }

        if !output.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(
                input.len(),
                self.complex_length(),
            ));
        }

        for (real, complex) in input.chunks_exact(3).zip(output.chunks_exact_mut(2)) {
            let [q0, q1] = self.r_exec((&real[..3]).try_into().unwrap());

            complex[0] = q0;
            complex[1] = q1;
        }
        Ok(())
    }

    fn real_length(&self) -> usize {
        3
    }

    fn complex_length(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    use crate::r2c::test_r2c_butterfly;

    test_r2c_butterfly!(test_r2c_butterfly3, f32, Butterfly3, 3, 1e-5);
    test_butterfly!(test_butterfly3, f32, Butterfly3, 3, 1e-5);
    test_oof_butterfly!(test_oof_butterfly3, f32, Butterfly3, 3, 1e-5);
}
