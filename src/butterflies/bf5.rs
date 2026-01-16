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
use crate::{
    CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, FftSample,
    R2CFftExecutor, ZaftError,
};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::sync::Arc;

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

impl<T: FftSample> FftExecutor<T> for Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.length()) {
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

impl<T: FftSample> FftExecutorOutOfPlace<T> for Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(self.length()) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(self.length()) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(5).zip(src.chunks_exact(5)) {
            let u0 = src[0];
            let u1 = src[1];
            let u2 = src[2];
            let u3 = src[3];
            let u4 = src[4];

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

            dst[0] = y0;
            dst[1] = y1;
            dst[2] = y2;
            dst[3] = y3;
            dst[4] = y4;
        }
        Ok(())
    }
}

impl<T: FftSample> CompositeFftExecutor<T> for Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<T> + Send + Sync> {
        self
    }
}

impl<T: FftSample> R2CFftExecutor<T> for Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !input.len().is_multiple_of(5) {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), 5));
        }
        if !output.len().is_multiple_of(3) {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), 3));
        }

        for (input, complex) in input.chunks_exact(5).zip(output.chunks_exact_mut(3)) {
            let u0 = input[0];
            let u1 = input[1];
            let u2 = input[2];
            let u3 = input[3];
            let u4 = input[4];

            // Radix-5 butterfly, R2C

            let x14p = u1 + u4;
            let x14n = u1 - u4;
            let x23p = u2 + u3;
            let x23n = u2 - u3;
            let y0 = u0 + x14p + x23p;

            let b14re_a = fmla(self.twiddle2.re, x23p, fmla(self.twiddle1.re, x14p, u0));
            let b23re_a = fmla(self.twiddle1.re, x23p, fmla(self.twiddle2.re, x14p, u0));

            let b23im_b = fmla(self.twiddle2.im, x14n, -self.twiddle1.im * x23n);
            let b14im_b = fmla(self.twiddle1.im, x14n, self.twiddle2.im * x23n);

            let y1 = Complex {
                re: b14re_a,
                im: b14im_b,
            };
            let y2 = Complex {
                re: b23re_a,
                im: b23im_b,
            };

            complex[0] = Complex::new(y0, T::zero());
            complex[1] = y1;
            complex[2] = y2;
        }
        Ok(())
    }

    fn complex_length(&self) -> usize {
        3
    }

    fn real_length(&self) -> usize {
        5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_r2c_butterfly5, f32, Butterfly5, 5, 1e-5);
    test_butterfly!(test_butterfly5, f32, Butterfly5, 5, 1e-5);
    test_oof_butterfly!(test_oof_butterfly5, f32, Butterfly5, 5, 1e-5);
}
