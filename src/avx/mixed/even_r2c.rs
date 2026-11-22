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
use super::AvxMixedRadix8d;
use crate::avx::mixed::AvxStoreD;
use crate::avx::transpose::transpose_f64x2_2x2d;
use crate::r2c::FusedEvenR2cPlan;
use crate::{FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::MulAdd;

const RADIX: usize = 8;

impl FusedEvenR2cPlan for AvxMixedRadix8d {
    fn execute_even_r2c(
        &self,
        output: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
        twiddles: &[Complex<f64>],
    ) -> Result<(), ZaftError> {
        // SAFETY: The factory constructs this plan only with AVX2 and FMA.
        unsafe { self.execute_even_r2c_avx(output, scratch, twiddles) }
    }
}

impl AvxMixedRadix8d {
    #[target_feature(enable = "avx2,fma")]
    fn execute_even_r2c_avx(
        &self,
        output: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
        twiddles: &[Complex<f64>],
    ) -> Result<(), ZaftError> {
        let m = self.length();
        let packed = &mut output[..m];
        let (rows, width_scratch) = scratch.split_at_mut(m);
        self.process_columns_in_place(packed);
        self.width_executor
            .execute_destructive_with_scratch(packed, rows, width_scratch)?;
        finish_spectrum(rows, output, twiddles, self.width);
        Ok(())
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn bin_pair<const FMA: bool>(
    left: AvxStoreD,
    right: AvxStoreD,
    twiddle: AvxStoreD,
    conj: AvxStoreD,
    half: AvxStoreD,
) -> (AvxStoreD, AvxStoreD) {
    let sum = left + right;
    let diff = left - right;
    let even = sum.blend_real_img(diff);
    let odd = diff.blend_real_img(sum);
    let [real, imaginary] = twiddle.dup_even_odds();
    let rotated = odd.reverse_complex_elements() * real.xor(conj);
    let correction = if FMA {
        odd.mul_add(imaginary, rotated)
    } else {
        odd * imaginary + rotated
    };
    (
        even.mul_add(half, correction),
        even.mul_add(half, -correction).xor(conj),
    )
}

#[target_feature(enable = "avx2,fma")]
fn finish_spectrum(
    rows: &[Complex<f64>],
    output: &mut [Complex<f64>],
    twiddles: &[Complex<f64>],
    width: usize,
) {
    let m = rows.len();
    let mid = m / 2;
    assert!(width >= 2 && width.is_multiple_of(2));
    assert_eq!(m, width * RADIX);
    let output = &mut output[..m + 1];
    let twiddles = &twiddles[..mid - 1];

    let dc = rows[0];
    output[0] = Complex::new(dc.re + dc.im, 0.0);
    output[m] = Complex::new(dc.re - dc.im, 0.0);
    output[mid] = rows[width / 2].conj();

    let conj = AvxStoreD::conj_flag();
    let half = AvxStoreD::dup(0.5);
    let mut column = 1;

    // Skip the first and final columns so all vector pairs use the main
    // reconstruction loop's rounding and avoid the DC and midpoint bins.
    for bin in 1..RADIX {
        finish_single(rows, output, twiddles, width, bin, conj, half);
    }

    while column + 1 < width / 2 - 1 {
        let base = column * RADIX;
        for row in (0..RADIX).step_by(2) {
            let bin = base + row;
            // SAFETY: The tile contains two adjacent columns below width/2,
            // and each two-row store covers two valid bins on either side.
            unsafe {
                let lo0 = AvxStoreD::from_complex_ref(rows.get_unchecked(row * width + column..));
                let lo1 =
                    AvxStoreD::from_complex_ref(rows.get_unchecked((row + 1) * width + column..));

                // A natural-order bin is column * RADIX + row. For nonzero
                // rows its mirror is in column width - column - 1 and row
                // RADIX - row. Row zero instead mirrors to width - column.
                let mirrored_column = width - column - 2;
                let hi0_index = if row == 0 {
                    mirrored_column + 1
                } else {
                    (RADIX - row) * width + mirrored_column
                };
                let hi1_index = (RADIX - row - 1) * width + mirrored_column;
                let [hi1, hi0] = transpose_f64x2_2x2d([
                    AvxStoreD::from_complex_ref(rows.get_unchecked(hi0_index..)),
                    AvxStoreD::from_complex_ref(rows.get_unchecked(hi1_index..)),
                ]);
                let [lo0, lo1] = transpose_f64x2_2x2d([lo0, lo1]);
                let w0 = AvxStoreD::from_complex_ref(twiddles.get_unchecked(bin - 1..));
                let w1 = AvxStoreD::from_complex_ref(twiddles.get_unchecked(bin + RADIX - 1..));
                let (lo0, hi0) = bin_pair::<true>(lo0, hi0, w0, conj, half);
                let (lo1, hi1) = bin_pair::<true>(lo1, hi1, w1, conj, half);
                lo0.write(output.get_unchecked_mut(bin..));
                lo1.write(output.get_unchecked_mut(bin + RADIX..));
                hi0.reverse_complex()
                    .write(output.get_unchecked_mut(m - bin - 1..));
                hi1.reverse_complex()
                    .write(output.get_unchecked_mut(m - bin - RADIX - 1..));
            }
        }
        column += 2;
    }

    for bin in column * RADIX..mid {
        finish_single(rows, output, twiddles, width, bin, conj, half);
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn finish_single(
    rows: &[Complex<f64>],
    output: &mut [Complex<f64>],
    twiddles: &[Complex<f64>],
    width: usize,
    bin: usize,
    conj: AvxStoreD,
    half: AvxStoreD,
) {
    let m = rows.len();
    let column = bin / RADIX;
    let row = bin % RADIX;
    let mirror = if row == 0 {
        width - column
    } else {
        (RADIX - row) * width + width - column - 1
    };
    let left = AvxStoreD::from_complex(&rows[row * width + column]);
    let right = AvxStoreD::from_complex(&rows[mirror]);
    let twiddle = AvxStoreD::from_complex(&twiddles[bin - 1]);

    // For the power-of-two shape, the unfused implementation uses separate
    // multiply and add for the six bins in its two-wide tail. Its final
    // single bin and all bins in the main loop use FMA.
    let (lo, hi) = if bin > m / 2 - RADIX && bin != m / 2 - 1 {
        bin_pair::<false>(left, right, twiddle, conj, half)
    } else {
        bin_pair::<true>(left, right, twiddle, conj, half)
    };
    lo.write_single(&mut output[bin]);
    hi.write_single(&mut output[m - bin]);
}
