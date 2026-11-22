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
use super::real_to_complex::{even_r2c_twiddles, pack_real_input};
use crate::err::try_vec;
use crate::util::validate_scratch;
use crate::{FftExecutor, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::Zero;

pub(crate) trait FusedEvenR2cPlan: FftExecutor<f64> + Send + Sync {
    /// Transform a single packed block in `output[..self.length()]`.
    /// The extra output slot holds Nyquist, and scratch has `scratch_length()` slots.
    fn execute_even_r2c(
        &self,
        output: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
        twiddles: &[Complex<f64>],
    ) -> Result<(), ZaftError>;
}

pub(super) struct FusedEvenR2c<P> {
    child: P,
    length: usize,
    twiddles: Vec<Complex<f64>>,
}

impl<P: FusedEvenR2cPlan> FusedEvenR2c<P> {
    pub(super) fn new(child: P) -> Result<Self, ZaftError> {
        let length = child.length().checked_mul(2).ok_or(ZaftError::Overflow)?;
        Ok(Self {
            child,
            length,
            twiddles: even_r2c_twiddles(length)?,
        })
    }
}

impl<P: FusedEvenR2cPlan> R2CFftExecutor<f64> for FusedEvenR2c<P> {
    fn execute(&self, input: &[f64], output: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.real_length(),
            output.len(),
            self.complex_length(),
        )?;
        let mut scratch = try_vec![Complex::zero(); self.complex_scratch_length()];
        self.execute_with_scratch(input, output, &mut scratch)
    }

    fn execute_with_scratch(
        &self,
        input: &[f64],
        output: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.real_length(),
            output.len(),
            self.complex_length(),
        )?;
        let scratch = validate_scratch!(scratch, self.complex_scratch_length());
        for (input, output) in input
            .chunks_exact(self.real_length())
            .zip(output.chunks_exact_mut(self.complex_length()))
        {
            pack_real_input(input, &mut output[..self.length / 2]);
            self.child
                .execute_even_r2c(output, scratch, &self.twiddles)?;
        }
        Ok(())
    }

    fn real_length(&self) -> usize {
        self.length
    }

    fn complex_length(&self) -> usize {
        self.length / 2 + 1
    }

    fn complex_scratch_length(&self) -> usize {
        self.child.scratch_length()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Zaft;
    use crate::r2c::R2CFftEvenInterceptor;
    use crate::r2c::r2c_factory_d::can_fuse_even_r2c_f64;

    #[test]
    fn fused_even_r2c_matches_staged_batch() {
        let length = 32_768;
        if !can_fuse_even_r2c_f64(length) {
            return;
        }
        let baseline = R2CFftEvenInterceptor::<f64>::plan(length).unwrap();
        let fused = Zaft::make_r2c_fft_f64(length).unwrap();
        let required = fused.complex_scratch_length();
        assert_eq!(required, baseline.complex_scratch_length());

        let samples: Vec<_> = (0..2 * length + 2)
            .map(|i| match i % 19 {
                0 => -0.0,
                1 => f64::from_bits(1),
                _ => (i % 97) as f64 / 97.0 - 0.5,
            })
            .collect();
        let input = &samples[1..=2 * length];
        let output_len = 2 * fused.complex_length();
        let guard = Complex::new(1234.0, -789.0);
        let mut expected = vec![guard; output_len + 2];
        let mut actual = expected.clone();
        let mut scratch = vec![guard; required + 2];

        baseline
            .execute_with_scratch(
                input,
                &mut expected[1..=output_len],
                &mut scratch[1..=required],
            )
            .unwrap();
        fused
            .execute_with_scratch(
                input,
                &mut actual[1..=output_len],
                &mut scratch[1..=required],
            )
            .unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!(
                (a.re.to_bits(), a.im.to_bits()),
                (b.re.to_bits(), b.im.to_bits())
            );
        }
        assert_eq!((scratch[0], scratch[required + 1]), (guard, guard));
        assert!(
            fused
                .execute_with_scratch(
                    input,
                    &mut actual[1..=output_len],
                    &mut scratch[..required - 1]
                )
                .is_err()
        );
    }
}
