/*
 * // Copyright (c) Radzivon Bartoshyk 01/2026. All rights reserved.
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
use crate::util::validate_scratch;
use crate::{FftDirection, FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub(crate) struct BluesteinRfft<T> {
    convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    convolve_fft_twiddles: Vec<Complex<T>>,
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    spectrum_ops: Arc<dyn ComplexArith<T> + Send + Sync>,
    convolve_scratch_length: usize,
}

impl<T: FftSample> BluesteinRfft<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(
        size: usize,
        convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        fft_direction: FftDirection,
    ) -> Result<BluesteinRfft<T>, ZaftError> {
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

        let mut twiddles = try_vec![Complex::zero(); size];
        make_bluesteins_twiddles(&mut twiddles, direction)?;

        let convolve_scratch_length = convolve_fft.scratch_length();

        Ok(BluesteinRfft {
            convolve_fft,
            convolve_fft_twiddles,
            twiddles,
            execution_length: size,
            spectrum_ops: T::make_complex_arith(),
            convolve_scratch_length,
        })
    }
}

impl<T: FftSample> R2CFftExecutor<T> for BluesteinRfft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.real_length(),
            output.len(),
            self.complex_length(),
        )?;
        let mut scratch = vec![Complex::zero(); self.complex_scratch_length()];
        self.execute_with_scratch(input, output, scratch.as_mut_slice())
    }

    fn execute_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.real_length(),
            output.len(),
            self.complex_length(),
        )?;

        let scratch = validate_scratch!(scratch, self.complex_scratch_length());
        let (inner_input, convolve_scratch) =
            scratch.split_at_mut(self.convolve_fft_twiddles.len());

        let in_length = self.real_length();
        let complex_length = self.complex_length();

        for (src, complex) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(complex_length))
        {
            self.spectrum_ops.mul_expand_to_complex(
                src,
                &self.twiddles,
                &mut inner_input[..in_length],
            );

            inner_input[in_length..].fill(Complex::zero());

            self.convolve_fft
                .execute_with_scratch(inner_input, convolve_scratch)?;

            self.spectrum_ops
                .mul_conjugate_in_place(inner_input, &self.convolve_fft_twiddles);

            self.convolve_fft
                .execute_with_scratch(inner_input, convolve_scratch)?;

            self.spectrum_ops.conjugate_mul_by_b(
                &inner_input[..complex_length],
                &self.twiddles[..complex_length],
                &mut complex[..complex_length],
            );
        }
        Ok(())
    }

    #[inline]
    fn real_length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn complex_length(&self) -> usize {
        self.execution_length / 2 + 1
    }

    #[inline]
    fn complex_scratch_length(&self) -> usize {
        self.convolve_scratch_length + self.convolve_fft_twiddles.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::dft::Dft;
    use crate::r2c::rfft_bluestein::BluesteinRfft;
    use crate::{FftDirection, FftExecutor, R2CFftExecutor, Zaft};
    use num_complex::Complex;
    use num_traits::Zero;

    #[test]
    fn test_bluestein_rfft() {
        let src: [f64; 11] = [7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.4, 3.4, 5.12];
        let mx = BluesteinRfft::new(
            11,
            Zaft::strategy(24, FftDirection::Forward).unwrap(),
            FftDirection::Forward,
        )
        .unwrap();
        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(11, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        let test_value = src.to_vec();
        let mut complex_output = vec![Complex::<f64>::zero(); 11 / 2 + 1];
        mx.execute(&test_value, &mut complex_output).unwrap();
        reference_value
            .iter()
            .zip(complex_output.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_bluestein_rfft_47() {
        let src: [f64; 47] = [
            7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.4, 3.4, 5.12, 7.2, 6.2, 6.4, 7.9, 1.3, 5.6,
            2.6, 6.4, 7.4, 3.4, 5.12, 7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.4, 3.4, 5.12, 7.2,
            6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.4, 3.4, 5.12, 2.6, 6.4, 7.4,
        ];
        let mx = BluesteinRfft::new(
            47,
            Zaft::strategy(47 * 2 - 1, FftDirection::Forward).unwrap(),
            FftDirection::Forward,
        )
        .unwrap();
        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(47, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        let test_value = src.to_vec();
        let mut complex_output = vec![Complex::<f64>::zero(); 47 / 2 + 1];
        mx.execute(&test_value, &mut complex_output).unwrap();
        reference_value
            .iter()
            .zip(complex_output.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }
}
