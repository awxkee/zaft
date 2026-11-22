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
use crate::r2c::r2c_twiddles::R2CTwiddlesHandler;
use crate::util::{compute_twiddle, validate_scratch};
use crate::{FftDirection, FftExecutor, FftSample, Zaft, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub trait R2CFftExecutor<T> {
    /// Executes the Real-to-Complex Forward FFT.
    ///
    /// The size of the `input` slice must be equal to `self.real_length()`, and the size of the
    /// `output` slice must be equal to `self.complex_length()`.
    ///
    /// # Parameters
    /// * `input`: The **real-valued** time-domain input array.
    /// * `output`: The mutable slice where the **complex-valued, Hermitian symmetric** frequency data will be written.
    ///
    /// # Errors
    /// Returns a `ZaftError` if the execution fails (e.g., due to incorrect slice lengths or internal computation errors).
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError>;
    fn execute_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError>;
    /// Returns the **length** of the **real-valued** input array (N).
    ///
    /// This is the size of the time-domain vector being transformed.
    fn real_length(&self) -> usize;
    /// Returns the **length** of the **complex-valued** output array (`N/2 + 1`).
    ///
    /// This represents the number of complex elements required to store the meaningful, non-redundant
    /// frequency components due to Hermitian symmetry.
    fn complex_length(&self) -> usize;
    fn complex_scratch_length(&self) -> usize;
}

pub(crate) struct R2CFftEvenInterceptor<T> {
    intercept: Arc<dyn FftExecutor<T> + Send + Sync>,
    twiddles: Vec<Complex<T>>,
    length: usize,
    complex_length: usize,
    twiddles_handler: Arc<dyn R2CTwiddlesHandler<T> + Send + Sync>,
    complex_scratch_len: usize,
    zero_copy: bool,
}

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) struct R2CFftOddInterceptor<T> {
    intercept: Arc<dyn FftExecutor<T> + Send + Sync>,
    length: usize,
    complex_length: usize,
    intercept_scratch_length: usize,
}

#[cfg(any(target_arch = "wasm32", test))]
impl<T: FftSample> R2CFftOddInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn install(
        length: usize,
        intercept: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Self, ZaftError> {
        assert_ne!(length % 2, 0, "R2C must be odd in odd interceptor");
        assert_eq!(
            intercept.length(),
            length,
            "Underlying interceptor must have the full real length"
        );
        assert_eq!(
            intercept.direction(),
            FftDirection::Forward,
            "Real to complex FFT must be forward"
        );

        let intercept_scratch_length = intercept.scratch_length();

        Ok(Self {
            intercept,
            length,
            complex_length: length / 2 + 1,
            intercept_scratch_length,
        })
    }
}

#[cfg(any(target_arch = "wasm32", test))]
impl<T: FftSample> R2CFftExecutor<T> for R2CFftOddInterceptor<T>
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
        let mut scratch = try_vec![Complex::zero(); self.complex_scratch_length()];
        self.execute_with_scratch(input, output, &mut scratch)
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
        let (complex_buffer, intercept_scratch) = scratch.split_at_mut(self.length);

        for (input, output) in input
            .chunks_exact(self.length)
            .zip(output.chunks_exact_mut(self.complex_length))
        {
            for (dst, src) in complex_buffer.iter_mut().zip(input) {
                *dst = Complex::new(*src, T::zero());
            }
            self.intercept
                .execute_with_scratch(complex_buffer, intercept_scratch)?;
            output.copy_from_slice(&complex_buffer[..self.complex_length]);
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

impl<T: FftSample> R2CFftEvenInterceptor<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn plan(length: usize) -> Result<Self, ZaftError> {
        let width = length / 2;

        if Self::prefer_zero_copy(width)
            && let Some(intercept) = Zaft::preferred_butterfly(width, FftDirection::Forward)
        {
            return Self::install(length, intercept?, true);
        }

        Self::install(length, Zaft::strategy(width, FftDirection::Forward)?, false)
    }

    fn prefer_zero_copy(_width: usize) -> bool {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use std::any::TypeId;

            // These child widths consistently took longer out of place in
            // single-transform or batched AVX2 measurements on AMD EPYC.
            let prefer_packing = if TypeId::of::<T>() == TypeId::of::<f32>() {
                matches!(_width, 14 | 24 | 25 | 29 | 30 | 37 | 64 | 72)
            } else {
                matches!(_width, 25 | 169)
            };
            crate::util::has_valid_avx() && !prefer_packing
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "avx")))]
        {
            cfg!(all(target_arch = "aarch64", feature = "neon"))
        }
    }

    fn install(
        length: usize,
        intercept: Arc<dyn FftExecutor<T> + Send + Sync>,
        prefer_zero_copy: bool,
    ) -> Result<Self, ZaftError> {
        assert_eq!(
            intercept.direction(),
            FftDirection::Forward,
            "Complex to real fft must be inverse"
        );
        assert_eq!(length % 2, 0, "R2C must be even in even interceptor");
        assert_eq!(
            intercept.length(),
            length / 2,
            "Underlying interceptor must have a half-length of real values"
        );

        let scratch_len = intercept.scratch_length();
        let zero_copy = prefer_zero_copy
            && intercept.out_of_place_scratch_length() <= scratch_len
            && std::mem::align_of::<T>() == std::mem::align_of::<Complex<T>>()
            && std::mem::size_of::<T>() * 2 == std::mem::size_of::<Complex<T>>();

        Ok(Self {
            intercept,
            twiddles: even_r2c_twiddles(length)?,
            length,
            complex_length: length / 2 + 1,
            twiddles_handler: T::make_r2c_twiddles_handler(),
            complex_scratch_len: scratch_len,
            zero_copy,
        })
    }
}

pub(super) fn even_r2c_twiddles<T: FftSample>(length: usize) -> Result<Vec<Complex<T>>, ZaftError>
where
    f64: AsPrimitive<T>,
{
    let mut twiddles = try_vec![Complex::<T>::zero(); length.div_ceil(4) - 1];
    for (i, twiddle) in twiddles.iter_mut().enumerate() {
        *twiddle = compute_twiddle(i + 1, length, FftDirection::Forward) * 0.5f64.as_();
    }
    Ok(twiddles)
}

#[inline]
pub(super) fn pack_real_input<T: Copy>(input: &[T], packed: &mut [Complex<T>]) {
    for (dst, &[re, im]) in packed.iter_mut().zip(input.as_chunks::<2>().0) {
        *dst = Complex::new(re, im);
    }
}

impl<T: FftSample> R2CFftExecutor<T> for R2CFftEvenInterceptor<T>
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
        let mut scratch = try_vec![Complex::zero(); self.complex_scratch_length()];
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

        for (input, output) in input
            .chunks_exact(self.length)
            .zip(output.chunks_exact_mut(self.complex_length))
        {
            let packed_output = &mut output[..self.length / 2];
            if self.zero_copy {
                // SAFETY: Complex<T> is repr(C) with two adjacent T fields and
                // the plan checked their size and alignment. Each input block
                // has an even length and the FFT only reads this view.
                let packed_input =
                    unsafe { std::slice::from_raw_parts(input.as_ptr().cast(), input.len() / 2) };
                self.intercept.execute_out_of_place_with_scratch(
                    packed_input,
                    packed_output,
                    scratch,
                )?;
            } else {
                pack_real_input(input, packed_output);
                self.intercept
                    .execute_with_scratch(packed_output, scratch)?;
            }

            let (mut output_left, mut output_right) = output.split_at_mut(output.len() / 2);

            match (output_left.first_mut(), output_right.last_mut()) {
                (Some(first_element), Some(last_element)) => {
                    let first_value = *first_element;
                    *first_element = Complex {
                        re: first_value.re + first_value.im,
                        im: T::zero(),
                    };
                    *last_element = Complex {
                        re: first_value.re - first_value.im,
                        im: T::zero(),
                    };

                    output_left = &mut output_left[1..];
                    let right_len = output_right.len();
                    output_right = &mut output_right[..right_len - 1];
                }
                _ => {
                    return Ok(());
                }
            }

            self.twiddles_handler
                .handle(&self.twiddles, output_left, output_right);

            if output.len() % 2 == 1
                && let Some(center_element) = output.get_mut(output.len() / 2)
            {
                center_element.im = -center_element.im;
            }
        }

        Ok(())
    }

    fn real_length(&self) -> usize {
        self.length
    }

    fn complex_length(&self) -> usize {
        self.complex_length
    }

    fn complex_scratch_length(&self) -> usize {
        self.complex_scratch_len
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx")
    ))]
    use super::R2CFftEvenInterceptor;
    use super::{R2CFftExecutor, R2CFftOddInterceptor};
    use crate::dft::Dft;
    use crate::{FftDirection, FftExecutor, Zaft};
    use num_complex::Complex;

    #[cfg(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx")
    ))]
    #[test]
    fn even_interceptor_uses_zero_copy_for_fixed_simd_children() {
        macro_rules! compare {
            ($ty:ty) => {
                for length in [14, 256] {
                    let plan = R2CFftEvenInterceptor::<$ty>::plan(length).unwrap();
                    let child = Zaft::strategy(length / 2, FftDirection::Forward).unwrap();
                    let baseline = R2CFftEvenInterceptor::install(length, child, false).unwrap();
                    assert_eq!(plan.zero_copy, Zaft::could_do_split_mixed_radix());
                    assert_eq!(
                        plan.complex_scratch_length(),
                        baseline.complex_scratch_length()
                    );

                    let values: [$ty; 6] = [
                        -0.0,
                        0.0,
                        0.25,
                        -0.75,
                        <$ty>::MIN_POSITIVE,
                        <$ty>::from_bits(1),
                    ];
                    let input = (0..3 * length + 2)
                        .map(|i| values[i % values.len()])
                        .collect::<Vec<_>>();
                    let original = input.clone();
                    let guard = Complex::new(42.0, -19.0);
                    let output_len = 3 * plan.complex_length();
                    let mut expected = vec![guard; output_len + 2];
                    let mut actual = expected.clone();
                    let mut scratch = vec![guard; plan.complex_scratch_length()];
                    baseline
                        .execute_with_scratch(
                            &input[1..=3 * length],
                            &mut expected[1..=output_len],
                            &mut scratch,
                        )
                        .unwrap();
                    plan.execute_with_scratch(
                        &input[1..=3 * length],
                        &mut actual[1..=output_len],
                        &mut scratch,
                    )
                    .unwrap();
                    for (old, new) in expected.iter().zip(&actual) {
                        assert_eq!(old.re.to_bits(), new.re.to_bits());
                        assert_eq!(old.im.to_bits(), new.im.to_bits());
                    }
                    for (old, new) in original.iter().zip(&input) {
                        assert_eq!(old.to_bits(), new.to_bits());
                    }
                }
                let general = R2CFftEvenInterceptor::<$ty>::plan(66).unwrap();
                assert!(!general.zero_copy);
            };
        }
        compare!(f32);
        compare!(f64);

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            assert!(!R2CFftEvenInterceptor::<f32>::plan(74).unwrap().zero_copy);
            assert!(!R2CFftEvenInterceptor::<f64>::plan(50).unwrap().zero_copy);
        }
    }

    #[test]
    fn test_odd_interceptor_prime_lengths() {
        for length in [17, 19, 23, 29] {
            let fft = Zaft::strategy(length, FftDirection::Forward).unwrap();
            let r2c = R2CFftOddInterceptor::install(length, fft).unwrap();
            let input = (0..length)
                .map(|x| ((x * 17 + 3) % 29) as f32 / 29.0)
                .collect::<Vec<_>>();
            let mut output = vec![Complex::<f32>::default(); length / 2 + 1];
            r2c.execute(&input, &mut output).unwrap();

            let dft = Dft::new(length, FftDirection::Forward).unwrap();
            let mut reference = input
                .iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect::<Vec<_>>();
            dft.execute(&mut reference).unwrap();

            for (actual, expected) in output.iter().zip(&reference) {
                assert!((actual.re - expected.re).abs() < 1e-4);
                assert!((actual.im - expected.im).abs() < 1e-4);
            }
        }
    }
}
