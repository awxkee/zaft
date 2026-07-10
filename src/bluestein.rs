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
use crate::err::try_vec;
use crate::fast_divider::DividerU64;
use crate::fast_divider_u128::DividerU128;
use crate::spectrum_arithmetic::ComplexArith;
use crate::traits::FftTrigonometry;
use crate::util::{compute_twiddle, validate_oof_sizes, validate_scratch};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, Zero};
use std::sync::Arc;

pub(crate) struct BluesteinFft<T> {
    convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    convolve_fft_twiddles: Vec<Complex<T>>,
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    direction: FftDirection,
    spectrum_ops: Arc<dyn ComplexArith<T> + Send + Sync>,
    convolve_scratch_length: usize,
    destructive_inner_scratch_len: usize,
}

pub(crate) fn make_bluesteins_twiddles<T: Float + FftTrigonometry + 'static>(
    destination: &mut [Complex<T>],
    direction: FftDirection,
) -> Result<(), ZaftError>
where
    f64: AsPrimitive<T>,
{
    let twice_len = destination
        .len()
        .checked_mul(2)
        .ok_or(ZaftError::Overflow)?;

    if destination.len() < u32::MAX as usize {
        let twice_len_divider = DividerU64::new(twice_len as u64);

        for (i, e) in destination.iter_mut().enumerate() {
            let i_squared = i as u64 * i as u64;
            let i_mod = i_squared % twice_len_divider;
            *e = compute_twiddle(i_mod as usize, twice_len, direction);
        }
    } else {
        let twice_len_divider = DividerU128::new(twice_len as u128);

        for (i, e) in destination.iter_mut().enumerate() {
            let i_squared = i as u128 * i as u128;
            let i_mod = i_squared % twice_len_divider;
            *e = compute_twiddle(i_mod as usize, twice_len, direction);
        }
    }

    Ok(())
}

impl<T: FftSample> BluesteinFft<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(
        size: usize,
        convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        fft_direction: FftDirection,
    ) -> Result<BluesteinFft<T>, ZaftError> {
        let convolve_fft_len = convolve_fft.length();
        let min_convolve_len = crate::util::checked_bluestein_convolution_len(size)?;
        assert!(
            min_convolve_len <= convolve_fft_len,
            "Bluestein requires convolve_fft.length() >= self.length() * 2 - 1. Expected >= {}, got {}",
            min_convolve_len,
            convolve_fft_len
        );

        let inner_fft_scale = (1f64 / convolve_fft_len as f64).as_();
        let direction = convolve_fft.direction();
        assert_eq!(
            direction, fft_direction,
            "Convolve FFT may not go with other direction"
        );

        let mut convolve_fft_twiddles = try_vec![Complex::zero(); convolve_fft_len];
        make_bluesteins_twiddles(&mut convolve_fft_twiddles[..size], direction.inverse())?;

        // Scale the computed twiddles and copy them to the end of the array
        convolve_fft_twiddles[0] = convolve_fft_twiddles[0] * inner_fft_scale;
        let (lo, hi) = convolve_fft_twiddles.split_at_mut(convolve_fft_len - size + 1);
        lo[1..size]
            .iter_mut()
            .zip(hi[..size - 1].iter_mut().rev())
            .for_each(|(t, dst)| {
                *t = *t * inner_fft_scale;
                *dst = *t;
            });

        convolve_fft.execute(&mut convolve_fft_twiddles)?;

        // also compute some more mundane twiddle factors to start and end with
        let mut twiddles = try_vec![Complex::zero(); size];
        make_bluesteins_twiddles(&mut twiddles, direction)?;

        let convolve_scratch_length = convolve_fft.scratch_length();
        let destructive_inner_scratch_len = if size >= convolve_scratch_length {
            0
        } else {
            convolve_scratch_length
        };

        Ok(BluesteinFft {
            convolve_fft,
            convolve_fft_twiddles,
            twiddles,
            execution_length: size,
            direction,
            spectrum_ops: T::make_complex_arith(),
            convolve_scratch_length,
            destructive_inner_scratch_len,
        })
    }
}

impl<T: FftSample> FftExecutor<T> for BluesteinFft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        let mut scratch = vec![Complex::zero(); self.scratch_length()];
        self.execute_with_scratch(in_place, scratch.as_mut_slice())
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
        let (inner_input, convolve_scratch) =
            scratch.split_at_mut(self.convolve_fft_twiddles.len());

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            self.spectrum_ops
                .mul(chunk, &self.twiddles, &mut inner_input[..chunk.len()]);

            inner_input[chunk.len()..].fill(Complex::zero());

            self.convolve_fft
                .execute_with_scratch(inner_input, convolve_scratch)?;

            self.spectrum_ops
                .mul_conjugate_in_place(inner_input, &self.convolve_fft_twiddles);

            self.convolve_fft
                .execute_with_scratch(inner_input, convolve_scratch)?;

            self.spectrum_ops.conjugate_mul_by_b(
                &inner_input[..chunk.len()],
                &self.twiddles,
                chunk,
            );
        }
        Ok(())
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        let mut scratch = vec![Complex::zero(); self.out_of_place_scratch_length()];
        self.execute_out_of_place_with_scratch(src, dst, scratch.as_mut_slice())
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        validate_oof_sizes!(src, dst, self.execution_length);

        let scratch = validate_scratch!(scratch, self.out_of_place_scratch_length());
        let (inner_input, convolve_scratch) =
            scratch.split_at_mut(self.convolve_fft_twiddles.len());

        for (src_chunk, dst_chunk) in src
            .chunks_exact(self.execution_length)
            .zip(dst.chunks_exact_mut(self.execution_length))
        {
            self.spectrum_ops.mul(
                src_chunk,
                &self.twiddles,
                &mut inner_input[..src_chunk.len()],
            );

            inner_input[src_chunk.len()..].fill(Complex::zero());

            self.convolve_fft
                .execute_with_scratch(inner_input, convolve_scratch)?;

            self.spectrum_ops
                .mul_conjugate_in_place(inner_input, &self.convolve_fft_twiddles);

            self.convolve_fft
                .execute_with_scratch(inner_input, convolve_scratch)?;

            self.spectrum_ops.conjugate_mul_by_b(
                &inner_input[..src_chunk.len()],
                &self.twiddles,
                dst_chunk,
            );
        }
        Ok(())
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        validate_oof_sizes!(src, dst, self.execution_length);

        let scratch = validate_scratch!(scratch, self.destructive_scratch_length());
        let (inner_input, convolve_scratch) =
            scratch.split_at_mut(self.convolve_fft_twiddles.len());
        let use_dst_as_scratch = self.execution_length >= self.convolve_scratch_length;

        for (src_chunk, dst_chunk) in src
            .chunks_exact_mut(self.execution_length)
            .zip(dst.chunks_exact_mut(self.execution_length))
        {
            self.spectrum_ops.mul(
                src_chunk,
                &self.twiddles,
                &mut inner_input[..src_chunk.len()],
            );

            inner_input[src_chunk.len()..].fill(Complex::zero());

            self.convolve_fft.execute_with_scratch(
                inner_input,
                if use_dst_as_scratch {
                    src_chunk
                } else {
                    convolve_scratch
                },
            )?;

            self.spectrum_ops
                .mul_conjugate_in_place(inner_input, &self.convolve_fft_twiddles);

            self.convolve_fft.execute_with_scratch(
                inner_input,
                if use_dst_as_scratch {
                    src_chunk
                } else {
                    convolve_scratch
                },
            )?;

            self.spectrum_ops.conjugate_mul_by_b(
                &inner_input[..src_chunk.len()],
                &self.twiddles,
                dst_chunk,
            );
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_length(&self) -> usize {
        self.convolve_scratch_length + self.convolve_fft_twiddles.len()
    }

    #[inline]
    fn out_of_place_scratch_length(&self) -> usize {
        self.convolve_scratch_length + self.convolve_fft_twiddles.len()
    }

    #[inline]
    fn destructive_scratch_length(&self) -> usize {
        self.destructive_inner_scratch_len + self.convolve_fft_twiddles.len()
    }
}
