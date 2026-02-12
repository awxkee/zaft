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
use crate::complex_fma::c_mul_fast;
use crate::err::try_vec;
use crate::mla::fmla;
use crate::util::{
    bitreversed_transpose, compute_twiddle, is_power_of_seven, radixn_floating_twiddles_from_base,
    validate_oof_sizes, validate_scratch,
};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

#[allow(dead_code)]
pub(crate) struct Radix7<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    direction: FftDirection,
    butterfly: Arc<dyn FftExecutor<T> + Send + Sync>,
}

pub(crate) trait Radix7Twiddles {
    #[allow(unused)]
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized;
}

impl Radix7Twiddles for f64 {
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f64, 7>(base, size, fft_direction)
    }
}

impl Radix7Twiddles for f32 {
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f32, 7>(base, size, fft_direction)
    }
}

#[allow(dead_code)]
impl<T: FftSample + Radix7Twiddles> Radix7<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Radix7<T>, ZaftError> {
        assert!(
            is_power_of_seven(size as u64),
            "Input length must be a power of 7"
        );

        let twiddles = T::make_twiddles_with_base(7, size, fft_direction)?;

        Ok(Radix7 {
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 7, fft_direction),
            twiddle2: compute_twiddle(2, 7, fft_direction),
            twiddle3: compute_twiddle(3, 7, fft_direction),
            direction: fft_direction,
            butterfly: T::butterfly7(fft_direction)?,
        })
    }
}

impl<T: FftSample> Radix7<T>
where
    f64: AsPrimitive<T>,
{
    fn base_run(&self, chunk: &mut [Complex<T>]) {
        let mut len = 7;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                let columns = len;
                len *= 7;
                let seventh = len / 7;

                for data in chunk.chunks_exact_mut(len) {
                    for j in 0..seventh {
                        let u0 = *data.get_unchecked(j);
                        let u1 = c_mul_fast(
                            *data.get_unchecked(j + seventh),
                            *m_twiddles.get_unchecked(6 * j),
                        );
                        let u2 = c_mul_fast(
                            *data.get_unchecked(j + 2 * seventh),
                            *m_twiddles.get_unchecked(6 * j + 1),
                        );
                        let u3 = c_mul_fast(
                            *data.get_unchecked(j + 3 * seventh),
                            *m_twiddles.get_unchecked(6 * j + 2),
                        );
                        let u4 = c_mul_fast(
                            *data.get_unchecked(j + 4 * seventh),
                            *m_twiddles.get_unchecked(6 * j + 3),
                        );
                        let u5 = c_mul_fast(
                            *data.get_unchecked(j + 5 * seventh),
                            *m_twiddles.get_unchecked(6 * j + 4),
                        );
                        let u6 = c_mul_fast(
                            *data.get_unchecked(j + 6 * seventh),
                            *m_twiddles.get_unchecked(6 * j + 5),
                        );

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

                        // Store results
                        *data.get_unchecked_mut(j) = y0;
                        *data.get_unchecked_mut(j + seventh) = y1;
                        *data.get_unchecked_mut(j + 2 * seventh) = y2;
                        *data.get_unchecked_mut(j + 3 * seventh) = y3;
                        *data.get_unchecked_mut(j + 4 * seventh) = y4;
                        *data.get_unchecked_mut(j + 5 * seventh) = y5;
                        *data.get_unchecked_mut(j + 6 * seventh) = y6;
                    }
                }

                m_twiddles = &m_twiddles[columns * 6..];
            }
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Radix7<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        let mut scratch = try_vec![Complex::zero(); self.scratch_length()];
        self.execute_with_scratch(in_place, &mut scratch)
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let scratch = validate_scratch!(scratch, self.scratch_length());

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // Digit-reversal permutation
            bitreversed_transpose::<Complex<T>, 7>(7, chunk, scratch);

            self.butterfly.execute_out_of_place(scratch, chunk)?;
            self.base_run(chunk);
        }
        Ok(())
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_with_scratch(src, dst, &mut [])
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
        _: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        validate_oof_sizes!(src, dst, self.execution_length);

        for (dst, src) in dst
            .chunks_exact_mut(self.execution_length)
            .zip(src.chunks_exact(self.execution_length))
        {
            // Digit-reversal permutation
            bitreversed_transpose::<Complex<T>, 7>(7, src, dst);
            self.butterfly.execute(dst)?;
            self.base_run(dst);
        }
        Ok(())
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_with_scratch(src, dst, scratch)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn out_of_place_scratch_length(&self) -> usize {
        0
    }

    fn destructive_scratch_length(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test_radix;

    test_radix!(test_radix7, f32, Radix7, 5, 7, 1e-2);
}
