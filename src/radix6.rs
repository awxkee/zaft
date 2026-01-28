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
use crate::butterflies::{Butterfly2, Butterfly3};
use crate::complex_fma::c_mul_fast;
use crate::err::try_vec;
use crate::util::{
    bitreversed_transpose, compute_twiddle, is_power_of_six, radixn_floating_twiddles_from_base,
    validate_oof_sizes, validate_scratch,
};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

#[allow(dead_code)]
pub(crate) struct Radix6<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle: Complex<T>,
    direction: FftDirection,
    bf3: Butterfly3<T>,
    butterfly: Arc<dyn FftExecutor<T> + Send + Sync>,
}

pub(crate) trait Radix6Twiddles {
    #[allow(unused)]
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized;
}

impl Radix6Twiddles for f64 {
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f64, 6>(base, size, fft_direction)
    }
}

impl Radix6Twiddles for f32 {
    fn make_twiddles_with_base(
        base: usize,
        size: usize,
        fft_direction: FftDirection,
    ) -> Result<Vec<Complex<Self>>, ZaftError>
    where
        Self: Sized,
    {
        radixn_floating_twiddles_from_base::<f32, 6>(base, size, fft_direction)
    }
}

#[allow(dead_code)]
impl<T: FftSample + Radix6Twiddles> Radix6<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<Radix6<T>, ZaftError> {
        assert!(
            is_power_of_six(size as u64),
            "Input length must be a power of 6"
        );

        let twiddles = T::make_twiddles_with_base(6, size, fft_direction)?;

        Ok(Radix6 {
            execution_length: size,
            twiddles,
            twiddle: compute_twiddle(1, 3, fft_direction),
            direction: fft_direction,
            butterfly: T::butterfly6(fft_direction)?,
            bf3: Butterfly3::new(fft_direction),
        })
    }
}

impl<T: FftSample> Radix6<T>
where
    f64: AsPrimitive<T>,
{
    fn base_run(&self, chunk: &mut [Complex<T>]) {
        let mut len = 6;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                let columns = len;
                len *= 6;
                let sixth = len / 6;

                for data in chunk.chunks_exact_mut(len) {
                    for j in 0..sixth {
                        let u0 = *data.get_unchecked(j);
                        let u1 = c_mul_fast(
                            *data.get_unchecked(j + sixth),
                            *m_twiddles.get_unchecked(5 * j),
                        );
                        let u2 = c_mul_fast(
                            *data.get_unchecked(j + 2 * sixth),
                            *m_twiddles.get_unchecked(5 * j + 1),
                        );
                        let u3 = c_mul_fast(
                            *data.get_unchecked(j + 3 * sixth),
                            *m_twiddles.get_unchecked(5 * j + 2),
                        );
                        let u4 = c_mul_fast(
                            *data.get_unchecked(j + 4 * sixth),
                            *m_twiddles.get_unchecked(5 * j + 3),
                        );
                        let u5 = c_mul_fast(
                            *data.get_unchecked(j + 5 * sixth),
                            *m_twiddles.get_unchecked(5 * j + 4),
                        );

                        let [t0, t2, t4] = self.bf3.exec(&[u0, u2, u4]);
                        let [t1, t3, t5] = self.bf3.exec(&[u3, u5, u1]);
                        let [y0, y3] = Butterfly2::exec(&[t0, t1]);
                        let [y4, y1] = Butterfly2::exec(&[t2, t3]);
                        let [y2, y5] = Butterfly2::exec(&[t4, t5]);

                        // Store results
                        *data.get_unchecked_mut(j) = y0;
                        *data.get_unchecked_mut(j + sixth) = y1;
                        *data.get_unchecked_mut(j + 2 * sixth) = y2;
                        *data.get_unchecked_mut(j + 3 * sixth) = y3;
                        *data.get_unchecked_mut(j + 4 * sixth) = y4;
                        *data.get_unchecked_mut(j + 5 * sixth) = y5;
                    }
                }

                m_twiddles = &m_twiddles[columns * 5..];
            }
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Radix6<T>
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
            bitreversed_transpose::<Complex<T>, 6>(6, chunk, scratch);
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
            bitreversed_transpose::<Complex<T>, 6>(6, src, dst);
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

    test_radix!(test_radix6, f32, Radix6, 5, 6, 1e-2);
}
