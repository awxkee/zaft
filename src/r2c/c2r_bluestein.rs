/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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
use crate::bluestein::make_bluesteins_twiddles;
use crate::err::try_vec;
use crate::spectrum_arithmetic::ComplexArith;
use crate::util::{validate_oof_block_sizes, validate_scratch};
use crate::{C2RFftExecutor, FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

/// Odd-length inverse real transform using only the supplied half-spectrum.
pub(crate) struct BluesteinC2r<T> {
    convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    convolve_fft_twiddles: Vec<Complex<T>>,
    twiddles: Vec<Complex<T>>,
    length: usize,
    scratch_length: usize,
    spectrum_ops: Arc<dyn ComplexArith<T> + Send + Sync>,
}

impl<T: FftSample> BluesteinC2r<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        size: usize,
        convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Self, ZaftError> {
        if size == 0 {
            return Err(ZaftError::ZeroSizedFft);
        }
        assert!(
            !size.is_multiple_of(2),
            "Bluestein C2R requires an odd length"
        );
        // Chirp phase reduction needs 2N, even though the convolution is shorter.
        size.checked_mul(2).ok_or(ZaftError::Overflow)?;
        let min_len = crate::util::checked_bluestein_rfft_convolution_len(size)?;
        let m = convolve_fft.length();
        assert!(
            m >= min_len,
            "Bluestein C2R requires inner length >= N + N / 2"
        );
        assert_eq!(
            convolve_fft.direction(),
            FftDirection::Inverse,
            "Bluestein C2R requires an inverse inner FFT"
        );
        let scratch_length = m
            .checked_add(convolve_fft.scratch_length())
            .ok_or(ZaftError::Overflow)?;

        let mut twiddles = try_vec![Complex::zero(); size];
        make_bluesteins_twiddles(&mut twiddles, FftDirection::Inverse)?;
        let k = size / 2 + 1;
        let scale = (1.0 / m as f64).as_();
        let mut convolve_fft_twiddles = try_vec![Complex::zero(); m];
        // N outputs and K inputs use differences -(K-1)..=N-1. The two
        // kernel ranges stay disjoint at M >= N + K - 1.
        for (dst, &w) in convolve_fft_twiddles[..size].iter_mut().zip(&twiddles) {
            *dst = w.conj() * scale;
        }
        for (dst, &w) in convolve_fft_twiddles[m - (k - 1)..]
            .iter_mut()
            .zip(twiddles[1..k].iter().rev())
        {
            *dst = w.conj() * scale;
        }
        convolve_fft.execute(&mut convolve_fft_twiddles)?;

        Ok(Self {
            convolve_fft,
            convolve_fft_twiddles,
            twiddles,
            length: size,
            scratch_length,
            spectrum_ops: T::make_complex_arith(),
        })
    }
}

impl<T: FftSample> C2RFftExecutor<T> for BluesteinC2r<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[Complex<T>], output: &mut [T]) -> Result<(), ZaftError> {
        validate_oof_block_sizes(
            input.len(),
            self.complex_length(),
            output.len(),
            self.length,
        )?;
        let mut scratch = try_vec![Complex::zero(); self.scratch_length];
        self.execute_with_scratch(input, output, &mut scratch)
    }

    fn execute_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        let k = self.complex_length();
        validate_oof_block_sizes(input.len(), k, output.len(), self.length)?;
        let scratch = validate_scratch!(scratch, self.scratch_length);
        let (inner_input, inner_scratch) = scratch.split_at_mut(self.convolve_fft_twiddles.len());

        for (src, dst) in input
            .chunks_exact(k)
            .zip(output.chunks_exact_mut(self.length))
        {
            self.spectrum_ops
                .mul(&src[1..], &self.twiddles[1..k], &mut inner_input[1..k]);
            // x[j] = X[0].re + 2 Re(sum(k=1..(N-1)/2) X[k] exp(2 pi i j k/N)).
            // Halve DC before the convolution, ignore its imaginary component,
            // and double the real projection afterward. No Nyquist bin exists.
            inner_input[0] = Complex::new(src[0].re * T::HALF, T::zero());
            inner_input[k..].fill(Complex::zero());
            self.convolve_fft
                .execute_with_scratch(inner_input, inner_scratch)?;
            self.spectrum_ops
                .mul_conjugate_in_place(inner_input, &self.convolve_fft_twiddles);
            self.convolve_fft
                .execute_with_scratch(inner_input, inner_scratch)?;

            self.spectrum_ops.conjugate_mul_real_doubled(
                &inner_input[..self.length],
                &self.twiddles,
                dst,
            );
        }
        Ok(())
    }

    #[inline]
    fn real_length(&self) -> usize {
        self.length
    }

    #[inline]
    fn complex_length(&self) -> usize {
        self.length / 2 + 1
    }

    #[inline]
    fn complex_scratch_length(&self) -> usize {
        self.scratch_length
    }
}
