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
        let min_convolve_len = crate::util::checked_bluestein_rfft_convolution_len(size)?;
        assert!(
            min_convolve_len <= convolve_fft_len,
            "Bluestein rfft requires convolve_fft.length() >= self.length() + self.length() / 2. Expected >= {}, got {}",
            min_convolve_len,
            convolve_fft_len
        );

        let inner_fft_scale = (1f64 / convolve_fft_len as f64).as_();
        let direction = convolve_fft.direction();
        assert_eq!(
            direction, fft_direction,
            "Convolve FFT may not go with other direction"
        );

        let mut twiddles = try_vec![Complex::zero(); size];
        make_bluesteins_twiddles(&mut twiddles, direction)?;

        // The convolution kernel is `b[m] = conj(w[m])`. Output bin `k` in `0..K` pairs with
        // input `n` in `0..N`, so only `m = k - n` in `-(N - 1)..=K - 1` is ever read:
        // non-negative `m` live at the start of the buffer, negative `m` at its end and
        // `M >= N + K - 1` keeps the two ranges apart.
        let complex_length = size / 2 + 1;
        let mut convolve_fft_twiddles = try_vec![Complex::zero(); convolve_fft_len];
        for (dst, &w) in convolve_fft_twiddles[..complex_length]
            .iter_mut()
            .zip(twiddles.iter())
        {
            *dst = w.conj() * inner_fft_scale;
        }
        let negative_start = convolve_fft_len - (size - 1);
        for (dst, &w) in convolve_fft_twiddles[negative_start..]
            .iter_mut()
            .zip(twiddles[1..].iter().rev())
        {
            *dst = w.conj() * inner_fft_scale;
        }

        convolve_fft.execute(&mut convolve_fft_twiddles)?;

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

    /// Runs `rows` real rows of length `n` through Bluestein's rfft with an inner
    /// FFT of `inner_len` and compares every bin with the plain complex DFT.
    fn check(n: usize, inner_len: usize, rows: usize) {
        let src = (0..n * rows)
            .map(|i| ((i * 7919 + 13) % 97) as f64 * 0.37 - 17.0)
            .collect::<Vec<f64>>();
        let mx = BluesteinRfft::new(
            n,
            Zaft::strategy(inner_len, FftDirection::Forward).unwrap(),
            FftDirection::Forward,
        )
        .unwrap();
        assert_eq!(mx.real_length(), n);
        assert_eq!(mx.complex_length(), n / 2 + 1);

        let mut reference = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(n, FftDirection::Forward).unwrap();
        dft.execute(&mut reference).unwrap();

        let mut output = vec![Complex::<f64>::zero(); (n / 2 + 1) * rows];
        mx.execute(&src, &mut output).unwrap();

        for row in 0..rows {
            let reference = &reference[row * n..(row + 1) * n];
            let output = &output[row * (n / 2 + 1)..(row + 1) * (n / 2 + 1)];
            for (idx, (a, b)) in reference.iter().zip(output.iter()).enumerate() {
                assert!(
                    (a.re - b.re).abs() < 1e-8,
                    "n {n} inner {inner_len} row {row} bin {idx}: re {} != {}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-8,
                    "n {n} inner {inner_len} row {row} bin {idx}: im {} != {}",
                    a.im,
                    b.im,
                );
            }
        }
    }

    #[test]
    fn test_bluestein_rfft_minimal_inner_len() {
        // `N + N / 2` is the smallest inner length the pruned kernel allows.
        for n in [3usize, 11, 47, 97, 101, 211, 1009] {
            check(n, n + n / 2, 1);
        }
    }

    #[test]
    #[should_panic(expected = "Bluestein rfft requires")]
    fn test_bluestein_rfft_below_minimal_inner_len_is_rejected() {
        let n = 47;
        let inner = Zaft::strategy(n + n / 2 - 1, FftDirection::Forward).unwrap();
        let _ = BluesteinRfft::<f64>::new(n, inner, FftDirection::Forward);
    }

    #[test]
    fn test_bluestein_rfft_larger_inner_len() {
        // Inner lengths at and above the classic `2N - 1` must still be exact.
        check(11, 24, 1);
        check(47, 47 * 2 - 1, 1);
        check(47, 128, 1);
        check(211, 512, 1);
    }

    #[test]
    fn test_bluestein_rfft_batched() {
        check(47, 72, 3);
        check(101, 160, 2);
    }

    #[test]
    fn test_bluestein_rfft_even_and_composite() {
        // Not what the planner routes here, but the kernel placement is generic.
        check(1, 1, 1);
        check(2, 3, 1);
        check(12, 18, 1);
        check(15, 22, 1);
    }
}
