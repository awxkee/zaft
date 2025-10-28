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
use num_traits::{AsPrimitive, Float, MulAdd};

/// Digit-reversal permutation in base `radix`
// pub(crate) fn digit_reverse_indices(n: usize, radix: usize) -> Result<Vec<usize>, ZaftError> {
//     assert!(radix >= 2, "radix must be at least 2");
//
//     let mut indices = Vec::new();
//     indices
//         .try_reserve_exact(n)
//         .map_err(|_| ZaftError::OutOfMemory(n))?;
//
//     for i in 0..n {
//         let mut x = i;
//         let mut rev = 0;
//
//         // Count number of digits needed
//         let mut tmp = n;
//         let mut digits = 0;
//         while tmp > 1 {
//             tmp /= radix;
//             digits += 1;
//         }
//
//         for _ in 0..digits {
//             rev = rev * radix + (x % radix);
//             x /= radix;
//         }
//
//         indices.push(rev);
//     }
//
//     Ok(indices)
// }

/// Helper function to check if a number is a power of 3.
pub(crate) fn is_power_of_three(n: u64) -> bool {
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

pub(crate) fn is_power_of_five(n: u64) -> bool {
    let mut n = n;
    if n == 0 {
        return false;
    }
    while n % 5 == 0 {
        n /= 5;
    }
    n == 1
}

// pub(crate) fn permute_inplace<T: Copy + Clone>(table: &mut [T], lut: &[usize]) {
//     for (i, &j) in lut.iter().enumerate() {
//         if i < j {
//             unsafe {
//                 let z0 = *table.get_unchecked_mut(i);
//                 let t0 = *table.get_unchecked_mut(j);
//                 *table.get_unchecked_mut(i) = t0;
//                 *table.get_unchecked_mut(j) = z0;
//             }
//         }
//     }
// }

pub(crate) fn is_power_of_six(n: u64) -> bool {
    let mut n = n;
    if n < 1 {
        return false;
    }
    while n % 6 == 0 {
        n /= 6;
    }
    n == 1
}

pub(crate) fn is_power_of_seven(n: u64) -> bool {
    let mut n = n;
    if n == 0 {
        return false;
    }
    while n % 7 == 0 {
        n /= 7;
    }
    n == 1
}

pub(crate) fn is_power_of_ten(n: u64) -> bool {
    let mut n = n;
    if n == 0 {
        return false;
    }
    while n % 10 == 0 {
        n /= 10;
    }
    n == 1
}

pub(crate) fn is_power_of_eleven(n: u64) -> bool {
    let mut n = n;
    if n == 0 {
        return false;
    }
    while n % 11 == 0 {
        n /= 11;
    }
    n == 1
}

pub(crate) fn is_power_of_thirteen(n: u64) -> bool {
    let mut n = n;
    if n == 0 {
        return false;
    }
    while n % 13 == 0 {
        n /= 13;
    }
    n == 1
}

// pub(crate) fn radixn_floating_twiddles<
//     T: Default + Float + FftTrigonometry + 'static + MulAdd<T, Output = T>,
//     const N: usize,
// >(
//     size: usize,
//     fft_direction: FftDirection,
// ) -> Result<Vec<Complex<T>>, ZaftError>
// where
//     usize: AsPrimitive<T>,
//     f64: AsPrimitive<T>,
// {
//     let mut len = N;
//     let mut twiddles = Vec::new(); // total twiddles = N-1 for radix-7
//     twiddles
//         .try_reserve_exact(size - 1)
//         .map_err(|_| ZaftError::OutOfMemory(size - 1))?;
//
//     while len <= size {
//         let columns = len / N;
//
//         for k in 0..columns {
//             for i in 1..N {
//                 let w = compute_twiddle::<T>(k * i, len, fft_direction);
//                 twiddles.push(w);
//             }
//         }
//
//         len *= N;
//     }
//
//     Ok(twiddles)
// }

pub(crate) fn radixn_floating_twiddles_from_base<
    T: Default + Float + FftTrigonometry + 'static + MulAdd<T, Output = T>,
    const N: usize,
>(
    base_len: usize,
    size: usize,
    fft_direction: FftDirection,
) -> Result<Vec<Complex<T>>, ZaftError>
where
    usize: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    let mut twiddles = Vec::new();
    twiddles
        .try_reserve_exact(size - 1)
        .map_err(|_| ZaftError::OutOfMemory(size - 1))?;

    let mut cross_fft_len = base_len;
    while cross_fft_len < size {
        let num_columns = cross_fft_len;
        cross_fft_len *= N;

        for i in 0..num_columns {
            for k in 1..N {
                let twiddle = compute_twiddle(i * k, cross_fft_len, fft_direction);
                twiddles.push(twiddle);
            }
        }
    }

    Ok(twiddles)
}

pub(crate) fn bitreversed_transpose<T: Copy, const D: usize>(
    height: usize,
    input: &[T],
    output: &mut [T],
) {
    let width = input.len() / height;

    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }

    // Let's make sure the arguments are ok
    assert!(D > 1 && input.len() % height == 0 && input.len() == output.len());

    let strided_width = width / D;
    let rev_digits = if D.is_power_of_two() {
        let width_bits = width.trailing_zeros();
        let d_bits = D.trailing_zeros();

        // verify that width is a power of d
        assert!(width_bits % d_bits == 0);
        width_bits / d_bits
    } else {
        compute_logarithm::<D>(width).unwrap()
    };

    if strided_width == 0 {
        output.copy_from_slice(input);
        return;
    }

    for x in 0..strided_width {
        let mut i = 0;
        let x_fwd = [(); D].map(|_| {
            let value = D * x + i;
            i += 1;
            value
        }); // If we had access to rustc 1.63, we could use std::array::from_fn instead
        let x_rev = x_fwd.map(|x| reverse_bits::<D>(x, rev_digits));

        // Assert that the the bit reversed indices will not exceed the length of the output.
        // The highest index the loop reaches is: (x_rev[n] + 1)*height - 1
        // The last element of the data is at index: width*height - 1
        // Thus it is sufficient to assert that x_rev[n]<width.
        for r in x_rev {
            assert!(r < width);
        }
        for y in 0..height {
            for (fwd, rev) in x_fwd.iter().zip(x_rev.iter()) {
                let input_index = *fwd + y * width;
                let output_index = y + *rev * height;

                unsafe {
                    let temp = *input.get_unchecked(input_index);
                    *output.get_unchecked_mut(output_index) = temp;
                }
            }
        }
    }
}

// Repeatedly divide `value` by divisor `D`, `iters` times, and apply the remainders to a new value
// When D is a power of 2, this is exactly equal (implementation and assembly)-wise to a bit reversal
// When D is not a power of 2, think of this function as a logical equivalent to a bit reversal
fn reverse_bits<const D: usize>(value: usize, rev_digits: u32) -> usize {
    assert!(D > 1);

    let mut result: usize = 0;
    let mut value = value;
    for _ in 0..rev_digits {
        result = (result * D) + (value % D);
        value = value / D;
    }
    result
}

// computes `n` such that `D ^ n == value`. Returns `None` if `value` is not a perfect power of `D`, otherwise returns `Some(n)`
pub(crate) fn compute_logarithm<const D: usize>(value: usize) -> Option<u32> {
    if value == 0 || D < 2 {
        return None;
    }

    let mut current_exponent = 0;
    let mut current_value = value;

    while current_value % D == 0 {
        current_exponent += 1;
        current_value /= D;
    }

    if current_value == 1 {
        Some(current_exponent)
    } else {
        None
    }
}
