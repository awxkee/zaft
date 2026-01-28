/*
 * // Copyright (c) Radzivon Bartoshyk 02/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //2028
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
use crate::{FftDirection, FftExecutor, FftSample, Zaft, ZaftError};
use num_complex::Complex;
use num_integer::Integer;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub(crate) struct Butterfly41<T> {
    convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    convolve_fft_twiddles: [Complex<T>; 40],
    direction: FftDirection,
    spectrum_ops: Arc<dyn ComplexArith<T> + Send + Sync>,
}

impl<T: FftSample> Butterfly41<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Butterfly41<T> {
        let convolve_fft = Zaft::strategy(40, fft_direction).unwrap();
        let size = 41;

        let direction = convolve_fft.direction();
        let convolve_fft_len = convolve_fft.length();
        assert_eq!(fft_direction, direction);
        let dividing_len = DividerU64::new(size as u64);

        let primitive_root = 6;

        let gcd_data = i64::extended_gcd(&(primitive_root as i64), &(size as i64));
        let primitive_root_inverse = if gcd_data.x >= 0 {
            gcd_data.x
        } else {
            gcd_data.x + size as i64
        } as u64;

        // precompute the coefficients to use inside the process method
        let inner_fft_scale: T = (1f64 / convolve_fft_len as f64).as_();
        let mut inner_fft_input = [Complex::zero(); 40];
        let mut twiddle_input = 1;
        for dst in &mut inner_fft_input {
            let twiddle = compute_twiddle(twiddle_input, size, direction);
            *dst = twiddle * inner_fft_scale;

            twiddle_input =
                ((twiddle_input as u64 * primitive_root_inverse) % dividing_len) as usize;
        }

        convolve_fft.execute(&mut inner_fft_input).unwrap();

        Butterfly41 {
            convolve_fft,
            convolve_fft_twiddles: inner_fft_input,
            direction: fft_direction,
            spectrum_ops: T::make_complex_arith(),
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Butterfly41<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(41) {
            return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), 41));
        }

        let mut scratch = [Complex::zero(); 40];

        for chunk in in_place.chunks_exact_mut(41) {
            let (buffer_first, buffer) = chunk.split_first_mut().unwrap();
            let buffer_first_val = *buffer_first;

            scratch[0] = buffer[5];
            scratch[1] = buffer[35];
            scratch[2] = buffer[10];
            scratch[3] = buffer[24];
            scratch[4] = buffer[26];
            scratch[5] = buffer[38];
            scratch[6] = buffer[28];
            scratch[7] = buffer[9];
            scratch[8] = buffer[18];
            scratch[9] = buffer[31];
            scratch[10] = buffer[27];
            scratch[11] = buffer[3];
            scratch[12] = buffer[23];
            scratch[13] = buffer[20];
            scratch[14] = buffer[2];
            scratch[15] = buffer[17];
            scratch[16] = buffer[25];
            scratch[17] = buffer[32];
            scratch[18] = buffer[33];
            scratch[19] = buffer[39];
            scratch[20] = buffer[34];
            scratch[21] = buffer[4];
            scratch[22] = buffer[29];
            scratch[23] = buffer[15];
            scratch[24] = buffer[13];
            scratch[25] = buffer[1];
            scratch[26] = buffer[11];
            scratch[27] = buffer[30];
            scratch[28] = buffer[21];
            scratch[29] = buffer[8];
            scratch[30] = buffer[12];
            scratch[31] = buffer[36];
            scratch[32] = buffer[16];
            scratch[33] = buffer[19];
            scratch[34] = buffer[37];
            scratch[35] = buffer[22];
            scratch[36] = buffer[14];
            scratch[37] = buffer[7];
            scratch[38] = buffer[6];
            scratch[39] = buffer[0];

            self.convolve_fft.execute(&mut scratch)?;

            *buffer_first = *buffer_first + scratch[0];

            self.spectrum_ops
                .mul_conjugate_in_place(&mut scratch, &self.convolve_fft_twiddles);

            scratch[0] = scratch[0] + buffer_first_val.conj();

            // execute the second FFT
            self.convolve_fft.execute(&mut scratch)?;

            buffer[6] = scratch[0].conj();
            buffer[7] = scratch[1].conj();
            buffer[14] = scratch[2].conj();
            buffer[22] = scratch[3].conj();
            buffer[37] = scratch[4].conj();
            buffer[19] = scratch[5].conj();
            buffer[16] = scratch[6].conj();
            buffer[36] = scratch[7].conj();
            buffer[12] = scratch[8].conj();
            buffer[8] = scratch[9].conj();
            buffer[21] = scratch[10].conj();
            buffer[30] = scratch[11].conj();
            buffer[11] = scratch[12].conj();
            buffer[1] = scratch[13].conj();
            buffer[13] = scratch[14].conj();
            buffer[15] = scratch[15].conj();
            buffer[29] = scratch[16].conj();
            buffer[4] = scratch[17].conj();
            buffer[34] = scratch[18].conj();
            buffer[39] = scratch[19].conj();
            buffer[33] = scratch[20].conj();
            buffer[32] = scratch[21].conj();
            buffer[25] = scratch[22].conj();
            buffer[17] = scratch[23].conj();
            buffer[2] = scratch[24].conj();
            buffer[20] = scratch[25].conj();
            buffer[23] = scratch[26].conj();
            buffer[3] = scratch[27].conj();
            buffer[27] = scratch[28].conj();
            buffer[31] = scratch[29].conj();
            buffer[18] = scratch[30].conj();
            buffer[9] = scratch[31].conj();
            buffer[28] = scratch[32].conj();
            buffer[38] = scratch[33].conj();
            buffer[26] = scratch[34].conj();
            buffer[24] = scratch[35].conj();
            buffer[10] = scratch[36].conj();
            buffer[35] = scratch[37].conj();
            buffer[5] = scratch[38].conj();
            buffer[0] = scratch[39].conj();
        }
        Ok(())
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<T>],
        _: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        self.execute(in_place)
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
        if !src.len().is_multiple_of(41) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), 41));
        }
        if !dst.len().is_multiple_of(41) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), 41));
        }

        let mut scratch = [Complex::zero(); 40];

        for (dst, src) in dst.chunks_exact_mut(41).zip(src.chunks_exact(41)) {
            let (buffer_first, buffer) = src.split_first().unwrap();
            let buffer_first_val = *buffer_first;

            scratch[0] = buffer[5];
            scratch[1] = buffer[35];
            scratch[2] = buffer[10];
            scratch[3] = buffer[24];
            scratch[4] = buffer[26];
            scratch[5] = buffer[38];
            scratch[6] = buffer[28];
            scratch[7] = buffer[9];
            scratch[8] = buffer[18];
            scratch[9] = buffer[31];
            scratch[10] = buffer[27];
            scratch[11] = buffer[3];
            scratch[12] = buffer[23];
            scratch[13] = buffer[20];
            scratch[14] = buffer[2];
            scratch[15] = buffer[17];
            scratch[16] = buffer[25];
            scratch[17] = buffer[32];
            scratch[18] = buffer[33];
            scratch[19] = buffer[39];
            scratch[20] = buffer[34];
            scratch[21] = buffer[4];
            scratch[22] = buffer[29];
            scratch[23] = buffer[15];
            scratch[24] = buffer[13];
            scratch[25] = buffer[1];
            scratch[26] = buffer[11];
            scratch[27] = buffer[30];
            scratch[28] = buffer[21];
            scratch[29] = buffer[8];
            scratch[30] = buffer[12];
            scratch[31] = buffer[36];
            scratch[32] = buffer[16];
            scratch[33] = buffer[19];
            scratch[34] = buffer[37];
            scratch[35] = buffer[22];
            scratch[36] = buffer[14];
            scratch[37] = buffer[7];
            scratch[38] = buffer[6];
            scratch[39] = buffer[0];

            self.convolve_fft.execute(&mut scratch)?;

            dst[0] = *buffer_first + scratch[0];

            self.spectrum_ops
                .mul_conjugate_in_place(&mut scratch, &self.convolve_fft_twiddles);

            scratch[0] = scratch[0] + buffer_first_val.conj();

            // execute the second FFT
            self.convolve_fft.execute(&mut scratch)?;

            let (_, buffer) = dst.split_first_mut().unwrap();

            buffer[6] = scratch[0].conj();
            buffer[7] = scratch[1].conj();
            buffer[14] = scratch[2].conj();
            buffer[22] = scratch[3].conj();
            buffer[37] = scratch[4].conj();
            buffer[19] = scratch[5].conj();
            buffer[16] = scratch[6].conj();
            buffer[36] = scratch[7].conj();
            buffer[12] = scratch[8].conj();
            buffer[8] = scratch[9].conj();
            buffer[21] = scratch[10].conj();
            buffer[30] = scratch[11].conj();
            buffer[11] = scratch[12].conj();
            buffer[1] = scratch[13].conj();
            buffer[13] = scratch[14].conj();
            buffer[15] = scratch[15].conj();
            buffer[29] = scratch[16].conj();
            buffer[4] = scratch[17].conj();
            buffer[34] = scratch[18].conj();
            buffer[39] = scratch[19].conj();
            buffer[33] = scratch[20].conj();
            buffer[32] = scratch[21].conj();
            buffer[25] = scratch[22].conj();
            buffer[17] = scratch[23].conj();
            buffer[2] = scratch[24].conj();
            buffer[20] = scratch[25].conj();
            buffer[23] = scratch[26].conj();
            buffer[3] = scratch[27].conj();
            buffer[27] = scratch[28].conj();
            buffer[31] = scratch[29].conj();
            buffer[18] = scratch[30].conj();
            buffer[9] = scratch[31].conj();
            buffer[28] = scratch[32].conj();
            buffer[38] = scratch[33].conj();
            buffer[26] = scratch[34].conj();
            buffer[24] = scratch[35].conj();
            buffer[10] = scratch[36].conj();
            buffer[35] = scratch[37].conj();
            buffer[5] = scratch[38].conj();
            buffer[0] = scratch[39].conj();
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
        41
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;

    test_butterfly!(test_butterfly41, f32, Butterfly41, 41, 1e-5);
}
