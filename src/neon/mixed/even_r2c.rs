/*
 * // Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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
use super::{NeonMixedRadix8, NeonStoreD};
use crate::ZaftError;
use crate::r2c::FusedEvenR2cPlan;
use num_complex::Complex;
use num_traits::MulAdd;

impl FusedEvenR2cPlan for NeonMixedRadix8 {
    fn execute_even_r2c(
        &self,
        output: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
        twiddles: &[Complex<f64>],
    ) -> Result<(), ZaftError> {
        let len = self.execution_length;
        let (rows, width_scratch) = scratch.split_at_mut(len);
        let packed = &mut output[..len];
        // This plan is installed only on the NEON backend.
        unsafe { self.process_columns_in_place(packed) };
        self.width_executor
            .execute_destructive_with_scratch(packed, rows, width_scratch)?;
        unsafe { finish_spectrum(rows, output, twiddles, self.width) };
        Ok(())
    }
}

#[inline(always)]
fn bin_pair(
    a: NeonStoreD,
    b: NeonStoreD,
    twiddle: NeonStoreD,
    conj: NeonStoreD,
    blend_mask: NeonStoreD,
    half: NeonStoreD,
) -> (NeonStoreD, NeonStoreD) {
    let sum = a + b;
    let diff = a - b;
    let selected_sum = sum.select(diff, blend_mask);
    let selected_diff = diff.select(sum, blend_mask);
    let [re, im] = twiddle.dup_even_odds();
    let rotated = selected_diff.reverse_complex_elements() * re.xor(conj);
    let correction = selected_diff.mul_add(im, rotated);
    (
        selected_sum.mul_add(half, correction),
        selected_sum.mul_add(half, -correction).xor(conj),
    )
}

#[target_feature(enable = "neon")]
fn finish_spectrum(
    rows: &[Complex<f64>],
    output: &mut [Complex<f64>],
    twiddles: &[Complex<f64>],
    width: usize,
) {
    const RADIX: usize = 8;
    let len = rows.len();
    debug_assert!(width.is_multiple_of(2));
    debug_assert_eq!(len, width * RADIX);
    let output = &mut output[..=len];
    let twiddles = &twiddles[..len / 2 - 1];

    let first = rows[0];
    output[0] = Complex::new(first.re + first.im, 0.0);
    output[len] = Complex::new(first.re - first.im, 0.0);
    output[len / 2] = rows[width / 2].conj();

    let conj = NeonStoreD::conj_flag();
    let blend_mask = NeonStoreD::set_values(f64::from_bits(u64::MAX), 0.0);
    let half = NeonStoreD::dup(0.5);
    // SAFETY: Each load is from a valid matrix row and column. Each inner
    // bin is in 1..len/2, which bounds its mirrored bin and twiddle index.
    unsafe {
        for column in 0..width / 2 {
            let base = column * RADIX;
            // For a transposed index k = column * RADIX + row, the mirror
            // len - k uses column width - column when row is zero.
            if column != 0 {
                let a = NeonStoreD::from_complex_ref(rows.get_unchecked(column..));
                let b = NeonStoreD::from_complex_ref(rows.get_unchecked(width - column..));
                let twiddle = NeonStoreD::from_complex_ref(twiddles.get_unchecked(base - 1..));
                let (lo, hi) = bin_pair(a, b, twiddle, conj, blend_mask, half);
                lo.write_single(output.get_unchecked_mut(base));
                hi.write_single(output.get_unchecked_mut(len - base));
            }
            for row in 1..RADIX {
                let k = base + row;
                let a = NeonStoreD::from_complex_ref(rows.get_unchecked(row * width + column..));
                let b = NeonStoreD::from_complex_ref(
                    rows.get_unchecked((RADIX - row) * width + width - 1 - column..),
                );
                let twiddle = NeonStoreD::from_complex_ref(twiddles.get_unchecked(k - 1..));
                let (lo, hi) = bin_pair(a, b, twiddle, conj, blend_mask, half);
                lo.write_single(output.get_unchecked_mut(k));
                hi.write_single(output.get_unchecked_mut(len - k));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FftDirection;
    use crate::neon::R2CNeonTwiddles;
    use crate::r2c::R2CTwiddlesHandler;
    use crate::util::compute_twiddle;

    #[test]
    fn fused_finish_matches_transpose_and_reconstruction() {
        let values = [0.0, -0.0, f64::from_bits(1), -0.25, 0.5, 0.75, -2.7];
        let guard = Complex::new(53.25, -3.0);
        for width in [2, 4, 6, 16, 32, 256] {
            let len = width * 8;
            let rows: Vec<_> = (0..len)
                .map(|i| Complex::new(values[i % 7], values[(i / 3 + i) % 7]))
                .collect();
            let twiddles: Vec<_> = (1..len / 2)
                .map(|k| compute_twiddle(k, 2 * len, FftDirection::Forward) * 0.5)
                .collect();
            let mut expected = vec![guard; len + 1];
            for column in 0..width {
                for row in 0..8 {
                    expected[column * 8 + row] = rows[row * width + column];
                }
            }
            let first = expected[0];
            expected[0] = Complex::new(first.re + first.im, 0.0);
            expected[len] = Complex::new(first.re - first.im, 0.0);
            let (left, right) = expected.split_at_mut(len / 2);
            R2CNeonTwiddles {}.handle(&twiddles, &mut left[1..], &mut right[1..len / 2]);
            right[0].im = -right[0].im;

            let mut actual = vec![guard; len + 3];
            unsafe { finish_spectrum(&rows, &mut actual[1..len + 2], &twiddles, width) };
            for (k, (actual, expected)) in actual[1..len + 2].iter().zip(expected).enumerate() {
                assert_eq!(
                    actual.re.to_bits(),
                    expected.re.to_bits(),
                    "width={width} k={k}"
                );
                assert_eq!(
                    actual.im.to_bits(),
                    expected.im.to_bits(),
                    "width={width} k={k}"
                );
            }
            assert_eq!((actual[0], actual[len + 2]), (guard, guard));
        }
    }
}
