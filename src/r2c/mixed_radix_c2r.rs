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
//! Half-spectrum row storage for odd mixed-radix real inverses.
//!
//! For N = R*W, the inverse column transform and twiddle produce
//! Y_r(k) = exp(2*pi*i*r*k/N) * sum_j X(k+j*W)*exp(2*pi*i*r*j/R).
//! Substituting j -> R-1-j and using X(N-k) = conj(X(k)) gives
//! Y_r(W-k) = conj(Y_r(k)). Each row therefore needs only W/2+1 complex
//! values and a real inverse child, followed by a real transpose.

use crate::{C2RFftExecutor, ZaftError};
use num_complex::Complex;

#[inline]
pub(crate) fn as_real<T>(values: &[Complex<T>]) -> &[T] {
    // SAFETY: Complex<T> is repr(C), containing two adjacent T fields with
    // the same alignment as T. The view has the same lifetime and byte length.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() * 2) }
}

#[inline]
fn as_real_mut<T>(values: &mut [Complex<T>]) -> &mut [T] {
    // SAFETY: As above, with an exclusive reborrow of the complex slice.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len() * 2) }
}

/// Execute the real inverse children in half-spectrum storage. Physical rows
/// end up containing logical rows 1, 2, ..., height-1, 0; the final SIMD
/// transpose restores their order. Each physical row has 2*K scalar slots.
pub(crate) fn execute_rows<T: Copy>(
    child: &dyn C2RFftExecutor<T>,
    spectra: &mut [Complex<T>],
    output: &mut [T],
    scratch: &mut [Complex<T>],
    height: usize,
) -> Result<(), ZaftError> {
    let width = child.real_length();
    let k = child.complex_length();
    let (first_row, remaining_output) = output.split_at_mut(width);
    let child_scratch = if scratch.is_empty() {
        // SAFETY: Complex<T> has the scalar alignment and two adjacent fields.
        // The constructor checks that this remaining output block can hold the
        // child's scratch. An odd trailing scalar is excluded from the view.
        unsafe {
            std::slice::from_raw_parts_mut(
                remaining_output.as_mut_ptr().cast(),
                remaining_output.len() / 2,
            )
        }
    } else {
        scratch
    };

    child.execute_with_scratch(&spectra[..k], first_row, child_scratch)?;
    for row in 1..height {
        // The previous spectrum has already been consumed. Splitting here
        // keeps the next input disjoint from the storage used for its output.
        let (consumed, pending) = spectra.split_at_mut(row * k);
        let row_output = &mut as_real_mut(&mut consumed[(row - 1) * k..])[..width];
        child.execute_with_scratch(&pending[..k], row_output, child_scratch)?;
    }
    // Only the first row needs a copy; all other children wrote directly into
    // dead spectrum storage. K = floor(width/2)+1 gives room for width reals.
    as_real_mut(&mut spectra[(height - 1) * k..])[..width].copy_from_slice(first_row);
    Ok(())
}

/// Execute a batch of small complex FFTs, reusing output for child scratch.
#[inline]
pub(crate) fn execute_complex_rows<T>(
    child: &dyn crate::FftExecutor<T>,
    spectra: &mut [Complex<T>],
    output: &mut [T],
    scratch: &mut [Complex<T>],
) -> Result<(), ZaftError> {
    let child_scratch = if scratch.is_empty() {
        // SAFETY: The constructor checks capacity; Complex<T> has the same
        // scalar alignment. This exclusive view ends before the final transpose.
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), output.len() / 2) }
    } else {
        scratch
    };
    child.execute_with_scratch(spectra, child_scratch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::Dft;
    use crate::r2c::C2RAlgorithmFactory;
    use crate::{FftDirection, FftExecutor, Zaft};
    use std::sync::Arc;

    struct ExtraScratch<T> {
        child: Arc<dyn C2RFftExecutor<T> + Send + Sync>,
        required: usize,
    }

    impl<T: Copy + Default> C2RFftExecutor<T> for ExtraScratch<T> {
        fn execute(&self, input: &[Complex<T>], output: &mut [T]) -> Result<(), ZaftError> {
            self.execute_with_scratch(input, output, &mut vec![Complex::default(); self.required])
        }
        fn execute_with_scratch(
            &self,
            input: &[Complex<T>],
            output: &mut [T],
            scratch: &mut [Complex<T>],
        ) -> Result<(), ZaftError> {
            assert!(scratch.len() >= self.required);
            scratch[..self.required].fill(Complex::default());
            self.child.execute_with_scratch(input, output, scratch)
        }
        fn real_length(&self) -> usize {
            self.child.real_length()
        }
        fn complex_length(&self) -> usize {
            self.child.complex_length()
        }
        fn complex_scratch_length(&self) -> usize {
            self.required
        }
    }

    macro_rules! check_compact {
        ($name:ident, $ty:ty, $tolerance:expr) => {
            #[test]
            fn $name() {
                if !Zaft::could_do_split_mixed_radix() {
                    return;
                }
                for radix in [3usize, 5, 7, 9, 11] {
                    // All residues of the SIMD half-row length, including widths
                    // smaller than a vector, and both transpose tail lengths.
                    for width in (1..=33).step_by(2).chain([63, 65]) {
                        for complex_leaf in [false, true] {
                            let child = if complex_leaf {
                                crate::r2c::C2rChild::Complex(
                                    Zaft::strategy::<$ty>(width, FftDirection::Inverse).unwrap(),
                                )
                            } else {
                                crate::r2c::C2rChild::Real(
                                    crate::r2c::strategy_c2r::<$ty>(width).unwrap(),
                                )
                            };
                            let plan = match radix {
                                3 => <$ty>::c2r_mixed_radix3(child),
                                5 => <$ty>::c2r_mixed_radix5(child),
                                7 => <$ty>::c2r_mixed_radix7(child),
                                9 => <$ty>::c2r_mixed_radix9(child),
                                11 => <$ty>::c2r_mixed_radix11(child),
                                _ => unreachable!(),
                            }
                            .unwrap()
                            .unwrap();
                            let n = radix * width;
                            let k = n / 2 + 1;
                            let mut input = vec![Complex::<$ty>::default(); 3 * k];
                            for (i, value) in input[..k].iter_mut().enumerate() {
                                *value = Complex::new(
                                    (i % 17) as $ty / 17.0 - 0.5,
                                    (i % 11) as $ty / 11.0 - 0.5,
                                );
                            }
                            input[2 * k - 1] = Complex::new(0.7, -0.3);
                            input[2 * k] = Complex::new(1.25, 0.0);
                            let original_input = input.clone();
                            let mut reference = vec![Complex::<$ty>::default(); 3 * n];
                            for (half, full) in
                                input.chunks_exact(k).zip(reference.chunks_exact_mut(n))
                            {
                                full[..k].copy_from_slice(half);
                                for i in 1..k {
                                    full[n - i] = half[i].conj();
                                }
                            }
                            Dft::new(n, FftDirection::Inverse)
                                .unwrap()
                                .execute(&mut reference)
                                .unwrap();
                            let guard = Complex::new(12345.0, -6789.0);
                            let required = plan.complex_scratch_length();
                            let mut scratch = vec![guard; required + 2];
                            scratch[1..required + 1].fill(Complex::new(<$ty>::NAN, <$ty>::NAN));
                            let mut output = vec![9876.0; 3 * n + 2];
                            plan.execute_with_scratch(
                                &input,
                                &mut output[1..3 * n + 1],
                                &mut scratch[1..required + 1],
                            )
                            .unwrap();
                            for (actual, expected) in output[1..3 * n + 1].iter().zip(&reference) {
                                assert!(
                                    actual.is_finite()
                                        && (*actual - expected.re).abs()
                                            <= $tolerance * (1.0 + expected.re.abs()),
                                    "radix={radix}, width={width}, actual={actual}, expected={}",
                                    expected.re
                                );
                            }
                            assert_eq!(input, original_input);
                            assert_eq!(scratch[0], guard);
                            assert_eq!(scratch[required + 1], guard);
                            assert_eq!(output[0], 9876.0);
                            assert_eq!(output[3 * n + 1], 9876.0);
                            assert!(
                                plan.execute_with_scratch(
                                    &input,
                                    &mut output[1..3 * n + 1],
                                    &mut scratch[..required - 1]
                                )
                                .is_err()
                            );
                            assert!(
                                plan.execute_with_scratch(
                                    &input[..input.len() - 1],
                                    &mut output[1..3 * n + 1],
                                    &mut scratch
                                )
                                .is_err()
                            );
                            assert!(
                                plan.execute_with_scratch(
                                    &input[..k],
                                    &mut output[1..3 * n + 1],
                                    &mut scratch
                                )
                                .is_err()
                            );
                            assert!(
                                plan.execute_with_scratch(&[], &mut [], &mut scratch)
                                    .is_ok()
                            );
                        }
                    }
                }

                // A prime-width complex child needs more scratch than output
                // reuse can provide. It must use the separate scratch allocation.
                let child = Zaft::strategy::<$ty>(509, FftDirection::Inverse).unwrap();
                let required = child.scratch_length();
                assert!(required > 1527 / 2);
                let plan = <$ty>::c2r_mixed_radix3(crate::r2c::C2rChild::Complex(child))
                    .unwrap()
                    .unwrap();
                assert_eq!(plan.complex_scratch_length(), 1527 + required);
                let input = vec![Complex::new(1.0, 0.0); plan.complex_length()];
                let mut output = vec![<$ty>::NAN; plan.real_length()];
                let mut scratch = vec![Complex::default(); plan.complex_scratch_length()];
                plan.execute_with_scratch(&input, &mut output, &mut scratch)
                    .unwrap();
                assert!((output[0] - 1527.0).abs() < $tolerance * 1527.0);
                assert!(output[1..].iter().all(|x| x.abs() < $tolerance * 1527.0));

                // Exercise both output reuse and the separate-scratch fallback.
                for extra in [0usize, 512] {
                    let base = crate::r2c::strategy_c2r::<$ty>(31).unwrap();
                    let required = base.complex_scratch_length() + extra;
                    let child = Arc::new(ExtraScratch {
                        child: base,
                        required,
                    });
                    let plan = <$ty>::c2r_mixed_radix3(crate::r2c::C2rChild::Real(child))
                        .unwrap()
                        .unwrap();
                    let input = vec![Complex::new(1.0, 0.0); plan.complex_length()];
                    let mut output = vec![<$ty>::NAN; plan.real_length()];
                    let mut scratch = vec![Complex::default(); plan.complex_scratch_length()];
                    assert_eq!(
                        scratch.len(),
                        3 * 16 + if required > 31 { required } else { 0 }
                    );
                    plan.execute_with_scratch(&input, &mut output, &mut scratch)
                        .unwrap();
                    assert!((output[0] - 93.0).abs() < $tolerance * 93.0);
                    assert!(output[1..].iter().all(|x| x.abs() < $tolerance * 93.0));
                }
            }
        };
    }

    check_compact!(compact_c2r_f32, f32, 1e-4);
    check_compact!(compact_c2r_f64, f64, 1e-11);

    #[test]
    fn recursive_c2r_scratch_and_roundtrip() {
        if !Zaft::could_do_split_mixed_radix() {
            return;
        }
        for n in [45usize, 105, 315, 511, 513, 1527, 6561, 59049] {
            let forward = Zaft::make_r2c_fft_f64(n).unwrap();
            let inverse = Zaft::make_c2r_fft_f64(n).unwrap();
            if [45, 105, 315].contains(&n) {
                assert_eq!(inverse.complex_scratch_length(), n);
            }
            if n == 59049 {
                assert_eq!(inverse.complex_scratch_length(), 29529);
            }
            let source: Vec<_> = (0..n * 2).map(|i| (i % 97) as f64 / 97.0 - 0.5).collect();
            let mut input = vec![Complex::default(); forward.complex_length() * 2];
            forward.execute(&source, &mut input).unwrap();
            for value in &mut input {
                *value /= n as f64;
            }
            let mut output = vec![f64::NAN; source.len()];
            let mut scratch = vec![Complex::default(); inverse.complex_scratch_length()];
            inverse
                .execute_with_scratch(&input, &mut output, &mut scratch)
                .unwrap();
            assert!(
                source
                    .iter()
                    .zip(&output)
                    .all(|(a, b)| (a - b).abs() < 1e-10)
            );

            let forward = Zaft::make_r2c_fft_f32(n).unwrap();
            let inverse = Zaft::make_c2r_fft_f32(n).unwrap();
            let source: Vec<_> = source.iter().map(|x| *x as f32).collect();
            let mut input = vec![Complex::default(); forward.complex_length() * 2];
            forward.execute(&source, &mut input).unwrap();
            for value in &mut input {
                *value /= n as f32;
            }
            let mut output = vec![f32::NAN; source.len()];
            let mut scratch = vec![Complex::default(); inverse.complex_scratch_length()];
            inverse
                .execute_with_scratch(&input, &mut output, &mut scratch)
                .unwrap();
            assert!(
                source
                    .iter()
                    .zip(&output)
                    .all(|(a, b)| (a - b).abs() < 1e-5)
            );
        }
    }
}
