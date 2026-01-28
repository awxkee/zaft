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
use crate::butterflies::Butterfly2;
use crate::butterflies::short_butterflies::FastButterfly5;
use crate::complex_fma::c_mul_fast;
use crate::err::try_vec;
use crate::util::{
    bitreversed_transpose, is_power_of_ten, radixn_floating_twiddles_from_base, validate_oof_sizes,
    validate_scratch,
};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

#[allow(dead_code)]
pub(crate) struct Radix10<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    bf5: FastButterfly5<T>,
    direction: FftDirection,
    butterfly: Arc<dyn FftExecutor<T> + Send + Sync>,
}

pub(crate) trait Radix10Twiddles {
    #[allow(unused)]
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized;
}

impl Radix10Twiddles for f64 {
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f64, 10>(base, size, fft_direction)
    }
}

impl Radix10Twiddles for f32 {
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f32, 10>(base, size, fft_direction)
    }
}

#[allow(dead_code)]
impl<T: FftSample + Radix10Twiddles> Radix10<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Radix10<T>, ZaftError> {
        assert!(
            is_power_of_ten(size as u64),
            "Input length must be a power of 10"
        );

        let twiddles = T::make_twiddles_with_base(10, size, fft_direction)?;

        Ok(Radix10 {
            execution_length: size,
            twiddles,
            bf5: FastButterfly5::new(fft_direction),
            direction: fft_direction,
            butterfly: T::butterfly10(fft_direction)?,
        })
    }
}

impl<T: FftSample> Radix10<T>
where
    f64: AsPrimitive<T>,
{
    fn base_run(&self, chunk: &mut [Complex<T>]) {
        let mut len = 10;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                let columns = len;
                len *= 10;
                let tenth = len / 10;

                for data in chunk.chunks_exact_mut(len) {
                    for j in 0..tenth {
                        let u0 = *data.get_unchecked(j);
                        let td = 9 * j;
                        let u1 = c_mul_fast(
                            *data.get_unchecked(j + tenth),
                            *m_twiddles.get_unchecked(td),
                        );
                        let u2 = c_mul_fast(
                            *data.get_unchecked(j + 2 * tenth),
                            *m_twiddles.get_unchecked(td + 1),
                        );
                        let u3 = c_mul_fast(
                            *data.get_unchecked(j + 3 * tenth),
                            *m_twiddles.get_unchecked(td + 2),
                        );
                        let u4 = c_mul_fast(
                            *data.get_unchecked(j + 4 * tenth),
                            *m_twiddles.get_unchecked(td + 3),
                        );
                        let u5 = c_mul_fast(
                            *data.get_unchecked(j + 5 * tenth),
                            *m_twiddles.get_unchecked(td + 4),
                        );
                        let u6 = c_mul_fast(
                            *data.get_unchecked(j + 6 * tenth),
                            *m_twiddles.get_unchecked(td + 5),
                        );
                        let u7 = c_mul_fast(
                            *data.get_unchecked(j + 7 * tenth),
                            *m_twiddles.get_unchecked(td + 6),
                        );
                        let u8 = c_mul_fast(
                            *data.get_unchecked(j + 8 * tenth),
                            *m_twiddles.get_unchecked(td + 7),
                        );
                        let u9 = c_mul_fast(
                            *data.get_unchecked(j + 9 * tenth),
                            *m_twiddles.get_unchecked(td + 8),
                        );

                        // Good-thomas butterfly-10
                        let mid0 = self.bf5.bf5(u0, u2, u4, u6, u8);
                        let mid1 = self.bf5.bf5(u5, u7, u9, u1, u3);

                        // Since this is good-thomas algorithm, we don't need twiddle factors
                        let [y0, y5] = Butterfly2::exec(&[mid0.0, mid1.0]); // (y0, y5)
                        let [y6, y1] = Butterfly2::exec(&[mid0.1, mid1.1]); // (y6, y1)
                        let [y2, y7] = Butterfly2::exec(&[mid0.2, mid1.2]); // (y2, y7)
                        let [y8, y3] = Butterfly2::exec(&[mid0.3, mid1.3]); // (y8, y3)
                        let [y4, y9] = Butterfly2::exec(&[mid0.4, mid1.4]); // (y4, y9)

                        // Store results
                        *data.get_unchecked_mut(j) = y0;
                        *data.get_unchecked_mut(j + tenth) = y1;
                        *data.get_unchecked_mut(j + 2 * tenth) = y2;

                        *data.get_unchecked_mut(j + 3 * tenth) = y3;
                        *data.get_unchecked_mut(j + 4 * tenth) = y4;
                        *data.get_unchecked_mut(j + 5 * tenth) = y5;

                        *data.get_unchecked_mut(j + 6 * tenth) = y6;
                        *data.get_unchecked_mut(j + 7 * tenth) = y7;
                        *data.get_unchecked_mut(j + 8 * tenth) = y8;

                        *data.get_unchecked_mut(j + 9 * tenth) = y9;
                    }
                }

                m_twiddles = &m_twiddles[columns * 9..];
            }
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Radix10<T>
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
            bitreversed_transpose::<Complex<T>, 10>(10, chunk, scratch);

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
            bitreversed_transpose::<Complex<T>, 10>(10, src, dst);
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

    test_radix!(test_radix10, f32, Radix10, 4, 10, 1e-3);
}
