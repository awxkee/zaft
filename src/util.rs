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
use crate::traits::FftTrigonometry;
use crate::{FftDirection, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};

/// Digit-reversal permutation in base `radix` (radix = 3 for Radix-3)
pub(crate) fn digit_reverse_indices(n: usize, radix: usize) -> Result<Vec<usize>, ZaftError> {
    assert!(radix >= 2, "radix must be at least 2");

    let mut indices = Vec::new();
    indices
        .try_reserve_exact(n)
        .map_err(|_| ZaftError::OutOfMemory(n))?;

    for i in 0..n {
        let mut x = i;
        let mut rev = 0;

        // Count number of digits needed
        let mut tmp = n;
        let mut digits = 0;
        while tmp > 1 {
            tmp /= radix;
            digits += 1;
        }

        for _ in 0..digits {
            rev = rev * radix + (x % radix);
            x /= radix;
        }

        indices.push(rev);
    }

    Ok(indices)
}

/// Helper function to check if a number is a power of 3.
pub(crate) fn is_power_of_three(n: usize) -> bool {
    if n == 0 {
        return false;
    }
    let mut i = n;
    while i > 1 {
        if i % 3 != 0 {
            return false;
        }
        i /= 3;
    }
    true
}

pub(crate) fn compute_twiddle<T: Float + FftTrigonometry + 'static>(
    index: usize,
    fft_len: usize,
    direction: FftDirection,
) -> Complex<T>
where
    f64: AsPrimitive<T>,
{
    let angle = (-2. * index as f64 / fft_len as f64).as_();
    let (v_sin, v_cos) = angle.sincos_pi();

    let result = Complex {
        re: v_cos,
        im: v_sin,
    };

    match direction {
        FftDirection::Forward => result,
        FftDirection::Inverse => result.conj(),
    }
}

pub(crate) fn is_power_of_five(n: usize) -> bool {
    let mut n = n;
    if n == 0 {
        return false;
    }
    while n % 5 == 0 {
        n /= 5;
    }
    n == 1
}

pub(crate) fn permute_inplace<T: Copy + Clone>(table: &mut [T], lut: &[usize]) {
    for (i, &j) in lut.iter().enumerate() {
        if i < j {
            unsafe {
                let z0 = *table.get_unchecked_mut(i);
                let t0 = *table.get_unchecked_mut(j);
                *table.get_unchecked_mut(i) = t0;
                *table.get_unchecked_mut(j) = z0;
            }
        }
    }
}

pub(crate) fn is_power_of_six(n: usize) -> bool {
    let mut n = n;
    if n < 1 {
        return false;
    }
    while n % 6 == 0 {
        n /= 6;
    }
    n == 1
}
