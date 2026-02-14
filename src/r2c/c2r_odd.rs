/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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
use crate::util::validate_scratch;
use crate::{C2RFftExecutor, FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub(crate) trait C2ROddExpander<T> {
    fn expand(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        complex_length: usize,
        length: usize,
    );
}

pub(crate) trait C2ROddExpanderFactory {
    fn make_expander() -> Arc<dyn C2ROddExpander<Self> + Send + Sync>;
}

impl C2ROddExpanderFactory for f32 {
    fn make_expander() -> Arc<dyn C2ROddExpander<Self> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonC2RExpanderF;
            Arc::new(NeonC2RExpanderF::default())
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::util::has_valid_avx;
            if has_valid_avx() {
                use crate::avx::AvxC2RExpander;
                return Arc::new(AvxC2RExpander::default());
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Arc::new(DefaultC2RExpander::default())
        }
    }
}

impl C2ROddExpanderFactory for f64 {
    fn make_expander() -> Arc<dyn C2ROddExpander<Self> + Send + Sync> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::util::has_valid_avx;
            if has_valid_avx() {
                use crate::avx::AvxC2RExpander;
                return Arc::new(AvxC2RExpander::default());
            }
        }
        Arc::new(DefaultC2RExpander::default())
    }
}

#[derive(Default)]
struct DefaultC2RExpander<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FftSample> C2ROddExpander<T> for DefaultC2RExpander<T>
where
    f64: AsPrimitive<T>,
{
    fn expand(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        complex_length: usize,
        length: usize,
    ) {
        let start = &input[1..];
        let (out_left, out_right) = output.split_at_mut(complex_length);
        let q = &mut out_left[1..];
        for ((buf_left, buf_right), val) in q
            .iter_mut()
            .zip(out_right.iter_mut().rev())
            .take(length / 2)
            .zip(start)
        {
            *buf_left = *val;
            *buf_right = val.conj();
        }
        output[0].re = input[0].re;
        output[0].im = 0.0.as_();
    }
}

pub(crate) struct C2RFftOddInterceptor<T> {
    intercept: Arc<dyn FftExecutor<T> + Send + Sync>,
    expander: Arc<dyn C2ROddExpander<T> + Send + Sync>,
    length: usize,
    complex_length: usize,
    intercept_scratch_length: usize,
}

impl<T: FftSample> C2RFftOddInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn install(
        length: usize,
        intercept: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Self, ZaftError> {
        assert_ne!(length % 2, 0, "R2C must be even in even interceptor");
        assert_eq!(
            intercept.length(),
            length,
            "Underlying interceptor must have full length of real values"
        );
        assert_eq!(
            intercept.direction(),
            FftDirection::Inverse,
            "Complex to real fft must be inverse"
        );

        let intercept_scratch_length = intercept.scratch_length();

        Ok(Self {
            intercept,
            expander: T::make_expander(),
            length,
            complex_length: length / 2 + 1,
            intercept_scratch_length,
        })
    }
}

impl<T: FftSample> C2RFftExecutor<T> for C2RFftOddInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[Complex<T>], output: &mut [T]) -> Result<(), ZaftError> {
        let mut scratch = try_vec![Complex::zero(); self.complex_scratch_length()];
        self.execute_with_scratch(input, output, &mut scratch)
    }

    fn execute_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !output.len().is_multiple_of(self.length) {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), self.length));
        }
        if !input.len().is_multiple_of(self.complex_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                output.len(),
                self.complex_length,
            ));
        }

        let scratch = validate_scratch!(scratch, self.complex_scratch_length());
        let (scratch, intercept_scratch) = scratch.split_at_mut(self.length);

        for (input, output) in input
            .chunks_exact(self.complex_length)
            .zip(output.chunks_exact_mut(self.length))
        {
            self.expander
                .expand(input, scratch, self.complex_length, self.length);
            self.intercept
                .execute_with_scratch(scratch, intercept_scratch)?;
            for (dst, src) in output.iter_mut().zip(scratch.iter()) {
                *dst = src.re;
            }
        }

        Ok(())
    }

    fn real_length(&self) -> usize {
        self.length
    }

    #[inline]
    fn complex_length(&self) -> usize {
        self.complex_length
    }

    #[inline]
    fn complex_scratch_length(&self) -> usize {
        self.length + self.intercept_scratch_length
    }
}
