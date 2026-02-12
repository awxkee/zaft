/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
use crate::fast_divider::DividerU64;
use crate::spectrum_arithmetic::ComplexArith;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, FftSample, R2CFftExecutor, Zaft, ZaftError};
use num_complex::Complex;
use num_integer::Integer;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

#[allow(unused)]
pub(crate) struct Butterfly31<T> {
    convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    convolve_fft_twiddles: [Complex<T>; 30],
    execution_length: usize,
    direction: FftDirection,
    spectrum_ops: Arc<dyn ComplexArith<T> + Send + Sync>,
}

#[allow(unused)]
impl<T: FftSample> Butterfly31<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        let convolve_fft = Zaft::strategy(30, fft_direction).unwrap();
        let direction = convolve_fft.direction();
        let convolve_fft_len = convolve_fft.length();
        assert_eq!(fft_direction, direction);
        let size = 31;
        let dividing_len = DividerU64::new(size as u64);

        let primitive_root = 3;

        let gcd_data = i64::extended_gcd(&(primitive_root as i64), &(size as i64));
        let primitive_root_inverse = if gcd_data.x >= 0 {
            gcd_data.x
        } else {
            gcd_data.x + size as i64
        } as u64;

        let inner_fft_scale: T = (1f64 / convolve_fft_len as f64).as_();
        let mut inner_fft_input = [Complex::zero(); 30];
        let mut twiddle_input = 1;
        for dst in &mut inner_fft_input {
            let twiddle = compute_twiddle(twiddle_input, size, direction);
            *dst = twiddle * inner_fft_scale;

            twiddle_input =
                ((twiddle_input as u64 * primitive_root_inverse) % dividing_len) as usize;
        }

        convolve_fft.execute(&mut inner_fft_input).unwrap();

        Butterfly31 {
            execution_length: size,
            convolve_fft,
            convolve_fft_twiddles: inner_fft_input,
            direction: fft_direction,
            spectrum_ops: T::make_complex_arith(),
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Butterfly31<T>
where
    f64: AsPrimitive<T>,
{
    #[inline]
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.length()) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let mut scratch = [Complex::zero(); 30];

        for chunk in in_place.chunks_exact_mut(31) {
            let (buffer_first, buffer) = chunk.split_first_mut().unwrap();
            let buffer_first_val = *buffer_first;

            scratch[0] = buffer[2];
            scratch[1] = buffer[8];
            scratch[2] = buffer[26];
            scratch[3] = buffer[18];
            scratch[4] = buffer[25];
            scratch[5] = buffer[15];
            scratch[6] = buffer[16];
            scratch[7] = buffer[19];
            scratch[8] = buffer[28];
            scratch[9] = buffer[24];
            scratch[10] = buffer[12];
            scratch[11] = buffer[7];
            scratch[12] = buffer[23];
            scratch[13] = buffer[9];
            scratch[14] = buffer[29];
            scratch[15] = buffer[27];
            scratch[16] = buffer[21];
            scratch[17] = buffer[3];
            scratch[18] = buffer[11];
            scratch[19] = buffer[4];
            scratch[20] = buffer[14];
            scratch[21] = buffer[13];
            scratch[22] = buffer[10];
            scratch[23] = buffer[1];
            scratch[24] = buffer[5];
            scratch[25] = buffer[17];
            scratch[26] = buffer[22];
            scratch[27] = buffer[6];
            scratch[28] = buffer[20];
            scratch[29] = buffer[0];

            // perform the first of two inner FFTs
            self.convolve_fft.execute(&mut scratch)?;

            *buffer_first = *buffer_first + scratch[0];

            self.spectrum_ops
                .mul_conjugate_in_place(scratch.as_mut_slice(), &self.convolve_fft_twiddles);

            scratch[0] = scratch[0] + buffer_first_val.conj();

            // execute the second FFT
            self.convolve_fft.execute(&mut scratch)?;

            // copy the final values into the output, with reordering
            buffer[20] = scratch[0].conj();
            buffer[6] = scratch[1].conj();
            buffer[22] = scratch[2].conj();
            buffer[17] = scratch[3].conj();
            buffer[5] = scratch[4].conj();
            buffer[1] = scratch[5].conj();
            buffer[10] = scratch[6].conj();
            buffer[13] = scratch[7].conj();
            buffer[14] = scratch[8].conj();
            buffer[4] = scratch[9].conj();
            buffer[11] = scratch[10].conj();
            buffer[3] = scratch[11].conj();
            buffer[21] = scratch[12].conj();
            buffer[27] = scratch[13].conj();
            buffer[29] = scratch[14].conj();
            buffer[9] = scratch[15].conj();
            buffer[23] = scratch[16].conj();
            buffer[7] = scratch[17].conj();
            buffer[12] = scratch[18].conj();
            buffer[24] = scratch[19].conj();
            buffer[28] = scratch[20].conj();
            buffer[19] = scratch[21].conj();
            buffer[16] = scratch[22].conj();
            buffer[15] = scratch[23].conj();
            buffer[25] = scratch[24].conj();
            buffer[18] = scratch[25].conj();
            buffer[26] = scratch[26].conj();
            buffer[8] = scratch[27].conj();
            buffer[2] = scratch[28].conj();
            buffer[0] = scratch[29].conj();
        }
        Ok(())
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<T>],
        _: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        FftExecutor::execute(self, in_place)
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
        if !src.len().is_multiple_of(31) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), 31));
        }
        if !dst.len().is_multiple_of(31) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), 31));
        }

        let mut scratch = [Complex::zero(); 30];

        for (dst, src) in dst.chunks_exact_mut(31).zip(src.chunks_exact(31)) {
            let (buffer_first, buffer) = src.split_first().unwrap();
            let buffer_first_val = *buffer_first;

            scratch[0] = buffer[2];
            scratch[1] = buffer[8];
            scratch[2] = buffer[26];
            scratch[3] = buffer[18];
            scratch[4] = buffer[25];
            scratch[5] = buffer[15];
            scratch[6] = buffer[16];
            scratch[7] = buffer[19];
            scratch[8] = buffer[28];
            scratch[9] = buffer[24];
            scratch[10] = buffer[12];
            scratch[11] = buffer[7];
            scratch[12] = buffer[23];
            scratch[13] = buffer[9];
            scratch[14] = buffer[29];
            scratch[15] = buffer[27];
            scratch[16] = buffer[21];
            scratch[17] = buffer[3];
            scratch[18] = buffer[11];
            scratch[19] = buffer[4];
            scratch[20] = buffer[14];
            scratch[21] = buffer[13];
            scratch[22] = buffer[10];
            scratch[23] = buffer[1];
            scratch[24] = buffer[5];
            scratch[25] = buffer[17];
            scratch[26] = buffer[22];
            scratch[27] = buffer[6];
            scratch[28] = buffer[20];
            scratch[29] = buffer[0];

            // perform the first of two inner FFTs
            self.convolve_fft.execute(&mut scratch)?;

            dst[0] = *buffer_first + scratch[0];

            self.spectrum_ops
                .mul_conjugate_in_place(scratch.as_mut_slice(), &self.convolve_fft_twiddles);

            scratch[0] = scratch[0] + buffer_first_val.conj();

            // execute the second FFT
            self.convolve_fft.execute(&mut scratch)?;

            let (_, buffer) = dst.split_first_mut().unwrap();

            // copy the final values into the output, with reordering
            buffer[20] = scratch[0].conj();
            buffer[6] = scratch[1].conj();
            buffer[22] = scratch[2].conj();
            buffer[17] = scratch[3].conj();
            buffer[5] = scratch[4].conj();
            buffer[1] = scratch[5].conj();
            buffer[10] = scratch[6].conj();
            buffer[13] = scratch[7].conj();
            buffer[14] = scratch[8].conj();
            buffer[4] = scratch[9].conj();
            buffer[11] = scratch[10].conj();
            buffer[3] = scratch[11].conj();
            buffer[21] = scratch[12].conj();
            buffer[27] = scratch[13].conj();
            buffer[29] = scratch[14].conj();
            buffer[9] = scratch[15].conj();
            buffer[23] = scratch[16].conj();
            buffer[7] = scratch[17].conj();
            buffer[12] = scratch[18].conj();
            buffer[24] = scratch[19].conj();
            buffer[28] = scratch[20].conj();
            buffer[19] = scratch[21].conj();
            buffer[16] = scratch[22].conj();
            buffer[15] = scratch[23].conj();
            buffer[25] = scratch[24].conj();
            buffer[18] = scratch[25].conj();
            buffer[26] = scratch[26].conj();
            buffer[8] = scratch[27].conj();
            buffer[2] = scratch[28].conj();
            buffer[0] = scratch[29].conj();
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
        31
    }

    fn scratch_length(&self) -> usize {
        0
    }

    fn out_of_place_scratch_length(&self) -> usize {
        0
    }

    fn destructive_scratch_length(&self) -> usize {
        0
    }
}

impl<T: FftSample> R2CFftExecutor<T> for Butterfly31<T> {
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !input.len().is_multiple_of(self.real_length()) {
            return Err(ZaftError::InvalidSizeMultiplier(
                input.len(),
                self.real_length(),
            ));
        }
        if !output.len().is_multiple_of(self.complex_length()) {
            return Err(ZaftError::InvalidSizeMultiplier(
                input.len(),
                self.complex_length(),
            ));
        }

        let mut scratch = [Complex::zero(); 30];

        for (chunk, complex) in input.chunks_exact(31).zip(output.chunks_exact_mut(16)) {
            let (buffer_first, buffer) = chunk.split_first().unwrap();
            let buffer_first_val = Complex::new(*buffer_first, T::zero());

            scratch[0] = Complex::new(buffer[2], T::zero());
            scratch[1] = Complex::new(buffer[8], T::zero());
            scratch[2] = Complex::new(buffer[26], T::zero());
            scratch[3] = Complex::new(buffer[18], T::zero());
            scratch[4] = Complex::new(buffer[25], T::zero());
            scratch[5] = Complex::new(buffer[15], T::zero());
            scratch[6] = Complex::new(buffer[16], T::zero());
            scratch[7] = Complex::new(buffer[19], T::zero());
            scratch[8] = Complex::new(buffer[28], T::zero());
            scratch[9] = Complex::new(buffer[24], T::zero());
            scratch[10] = Complex::new(buffer[12], T::zero());
            scratch[11] = Complex::new(buffer[7], T::zero());
            scratch[12] = Complex::new(buffer[23], T::zero());
            scratch[13] = Complex::new(buffer[9], T::zero());
            scratch[14] = Complex::new(buffer[29], T::zero());
            scratch[15] = Complex::new(buffer[27], T::zero());
            scratch[16] = Complex::new(buffer[21], T::zero());
            scratch[17] = Complex::new(buffer[3], T::zero());
            scratch[18] = Complex::new(buffer[11], T::zero());
            scratch[19] = Complex::new(buffer[4], T::zero());
            scratch[20] = Complex::new(buffer[14], T::zero());
            scratch[21] = Complex::new(buffer[13], T::zero());
            scratch[22] = Complex::new(buffer[10], T::zero());
            scratch[23] = Complex::new(buffer[1], T::zero());
            scratch[24] = Complex::new(buffer[5], T::zero());
            scratch[25] = Complex::new(buffer[17], T::zero());
            scratch[26] = Complex::new(buffer[22], T::zero());
            scratch[27] = Complex::new(buffer[6], T::zero());
            scratch[28] = Complex::new(buffer[20], T::zero());
            scratch[29] = Complex::new(buffer[0], T::zero());

            // perform the first of two inner FFTs
            self.convolve_fft.execute(&mut scratch)?;

            complex[0] = buffer_first_val + scratch[0];

            self.spectrum_ops
                .mul_conjugate_in_place(&mut scratch, &self.convolve_fft_twiddles);

            scratch[0] = scratch[0] + buffer_first_val.conj();

            // execute the second FFT
            self.convolve_fft.execute(&mut scratch)?;

            complex[7] = scratch[1].conj();
            complex[6] = scratch[4].conj();
            complex[2] = scratch[5].conj();
            complex[11] = scratch[6].conj();
            complex[14] = scratch[7].conj();
            complex[15] = scratch[8].conj();
            complex[5] = scratch[9].conj();
            complex[12] = scratch[10].conj();
            complex[4] = scratch[11].conj();
            complex[10] = scratch[15].conj();
            complex[8] = scratch[17].conj();
            complex[13] = scratch[18].conj();
            complex[9] = scratch[27].conj();
            complex[3] = scratch[28].conj();
            complex[1] = scratch[29].conj();
        }
        Ok(())
    }

    fn execute_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        _: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        R2CFftExecutor::execute(self, input, output)
    }

    #[inline]
    fn real_length(&self) -> usize {
        31
    }

    #[inline]
    fn complex_length(&self) -> usize {
        16
    }

    fn complex_scratch_length(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    use crate::r2c::test_r2c_butterfly;

    test_r2c_butterfly!(test_r2c_butterfly31, f32, Butterfly31, 31, 1e-5);
    test_butterfly!(test_butterfly31, f32, Butterfly31, 31, 1e-5);
}
