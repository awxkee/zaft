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

pub(crate) static ALWAYS_BLUESTEIN_1000: [usize; 63] = [
    47, 53, 59, 61, 83, 103, 107, 149, 167, 173, 179, 223, 227, 233, 263, 269, 283, 317, 347, 359,
    367, 383, 389, 431, 439, 461, 467, 479, 499, 503, 509, 557, 563, 569, 587, 619, 643, 647, 653,
    659, 709, 719, 733, 739, 743, 787, 797, 821, 823, 827, 839, 853, 857, 863, 877, 887, 907, 941,
    947, 971, 977, 983, 997,
];
pub(crate) static ALWAYS_BLUESTEIN_2000: [usize; 66] = [
    1019, 1039, 1061, 1069, 1097, 1129, 1163, 1181, 1187, 1193, 1223, 1229, 1231, 1237, 1259, 1279,
    1307, 1319, 1367, 1399, 1423, 1427, 1433, 1439, 1487, 1493, 1499, 1511, 1523, 1553, 1559, 1579,
    1583, 1609, 1619, 1627, 1637, 1663, 1669, 1693, 1697, 1699, 1709, 1723, 1747, 1753, 1759, 1787,
    1789, 1811, 1823, 1831, 1847, 1867, 1877, 1879, 1889, 1907, 1913, 1949, 1973, 1979, 1987, 1993,
    1997, 1999,
];
pub(crate) static ALWAYS_BLUESTEIN_3000: [usize; 49] = [
    2011, 2027, 2039, 2063, 2069, 2083, 2087, 2099, 2153, 2207, 2297, 2339, 2351, 2371, 2423, 2447,
    2459, 2473, 2477, 2539, 2543, 2557, 2579, 2617, 2633, 2657, 2659, 2671, 2677, 2683, 2687, 2699,
    2707, 2713, 2767, 2777, 2789, 2797, 2803, 2833, 2837, 2879, 2903, 2939, 2953, 2957, 2963, 2969,
    2999,
];
pub(crate) static ALWAYS_BLUESTEIN_4000: [usize; 80] = [
    3019, 3023, 3049, 3083, 3119, 3163, 3167, 3181, 3187, 3203, 3217, 3229, 3253, 3257, 3259, 3271,
    3299, 3307, 3319, 3323, 3343, 3347, 3359, 3391, 3407, 3413, 3449, 3461, 3463, 3467, 3491, 3499,
    3517, 3527, 3533, 3539, 3541, 3547, 3557, 3559, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3643,
    3659, 3671, 3677, 3691, 3709, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3833,
    3847, 3853, 3863, 3877, 3881, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989,
];
pub(crate) static ALWAYS_BLUESTEIN_5000: [usize; 47] = [
    4003, 4007, 4013, 4019, 4021, 4027, 4073, 4079, 4091, 4099, 4127, 4133, 4139, 4153, 4157, 4231,
    4253, 4283, 4297, 4349, 4391, 4423, 4457, 4463, 4483, 4507, 4513, 4517, 4547, 4567, 4583, 4597,
    4637, 4639, 4649, 4679, 4703, 4723, 4783, 4787, 4793, 4799, 4877, 4889, 4903, 4919, 4957,
];
pub(crate) static ALWAYS_BLUESTEIN_6000: [usize; 69] = [
    5003, 5011, 5077, 5087, 5099, 5107, 5113, 5119, 5147, 5167, 5171, 5179, 5227, 5231, 5261, 5273,
    5309, 5323, 5333, 5351, 5381, 5387, 5399, 5407, 5413, 5417, 5443, 5449, 5477, 5479, 5483, 5503,
    5507, 5519, 5527, 5531, 5563, 5623, 5639, 5641, 5647, 5659, 5669, 5683, 5689, 5693, 5711, 5717,
    5737, 5741, 5749, 5779, 5783, 5807, 5821, 5827, 5839, 5843, 5849, 5857, 5861, 5867, 5879, 5897,
    5903, 5923, 5927, 5939, 5987,
];

// Digit-reversal permutation in base `radix`
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
        if !i.is_multiple_of(3) {
            return false;
        }
        i /= 3;
    }
    true
}

#[inline(always)]
#[allow(unused)]
pub(crate) fn make_twiddles<const TW: usize, T: Float + FftTrigonometry + 'static>(
    len: usize,
    direction: FftDirection,
) -> [Complex<T>; TW]
where
    f64: AsPrimitive<T>,
{
    let mut i = 1;
    [(); TW].map(|_| {
        let twiddle = compute_twiddle(i, len, direction);
        i += 1;
        twiddle
    })
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
    while n.is_multiple_of(5) {
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
    while n.is_multiple_of(6) {
        n /= 6;
    }
    n == 1
}

pub(crate) fn is_power_of_seven(n: u64) -> bool {
    let mut n = n;
    if n == 0 {
        return false;
    }
    while n.is_multiple_of(7) {
        n /= 7;
    }
    n == 1
}

pub(crate) fn is_power_of_ten(n: u64) -> bool {
    let mut n = n;
    if n == 0 {
        return false;
    }
    while n.is_multiple_of(10) {
        n /= 10;
    }
    n == 1
}

pub(crate) fn is_power_of_eleven(n: u64) -> bool {
    let mut n = n;
    if n == 0 {
        return false;
    }
    while n.is_multiple_of(11) {
        n /= 11;
    }
    n == 1
}

pub(crate) fn is_power_of_thirteen(n: u64) -> bool {
    let mut n = n;
    if n == 0 {
        return false;
    }
    while n.is_multiple_of(13) {
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
    assert!(D > 1 && input.len().is_multiple_of(height) && input.len() == output.len());

    let strided_width = width / D;
    let rev_digits = if D.is_power_of_two() {
        let width_bits = width.trailing_zeros();
        let d_bits = D.trailing_zeros();

        // verify that width is a power of d
        assert!(width_bits.is_multiple_of(d_bits));
        width_bits / d_bits
    } else {
        int_logarithm::<D>(width).unwrap()
    };

    if strided_width == 0 {
        output.copy_from_slice(input);
        return;
    }

    for x in 0..strided_width {
        let x_fwd: [usize; D] = std::array::from_fn(|i| D * x + i);
        let x_rev = x_fwd.map(|x| reverse_bits::<D>(x, rev_digits));

        let mut y = 0usize;

        while y + 6 < height {
            let y1 = y + 1;
            let y2 = y + 2;
            let y3 = y + 3;
            let y4 = y + 4;
            let y5 = y + 5;
            for (fwd, rev) in x_fwd.iter().zip(x_rev.iter()) {
                let input_index0 = *fwd + y * width;
                let output_index0 = y + *rev * height;

                let input_index1 = *fwd + y1 * width;
                let output_index1 = y1 + *rev * height;

                let input_index2 = *fwd + y2 * width;
                let output_index2 = y2 + *rev * height;

                let input_index3 = *fwd + y3 * width;
                let output_index3 = y3 + *rev * height;

                let input_index4 = *fwd + y4 * width;
                let output_index4 = y4 + *rev * height;

                let input_index5 = *fwd + y5 * width;
                let output_index5 = y5 + *rev * height;

                unsafe {
                    *output.get_unchecked_mut(output_index0) = *input.get_unchecked(input_index0);
                    *output.get_unchecked_mut(output_index1) = *input.get_unchecked(input_index1);
                    *output.get_unchecked_mut(output_index2) = *input.get_unchecked(input_index2);
                    *output.get_unchecked_mut(output_index3) = *input.get_unchecked(input_index3);
                    *output.get_unchecked_mut(output_index4) = *input.get_unchecked(input_index4);
                    *output.get_unchecked_mut(output_index5) = *input.get_unchecked(input_index5);
                }
            }

            y += 6;
        }

        while y + 4 < height {
            let y1 = y + 1;
            let y2 = y + 2;
            let y3 = y + 3;
            for (fwd, rev) in x_fwd.iter().zip(x_rev.iter()) {
                let input_index0 = *fwd + y * width;
                let output_index0 = y + *rev * height;

                let input_index1 = *fwd + y1 * width;
                let output_index1 = y1 + *rev * height;

                let input_index2 = *fwd + y2 * width;
                let output_index2 = y2 + *rev * height;

                let input_index3 = *fwd + y3 * width;
                let output_index3 = y3 + *rev * height;

                unsafe {
                    *output.get_unchecked_mut(output_index0) = *input.get_unchecked(input_index0);
                    *output.get_unchecked_mut(output_index1) = *input.get_unchecked(input_index1);
                    *output.get_unchecked_mut(output_index2) = *input.get_unchecked(input_index2);
                    *output.get_unchecked_mut(output_index3) = *input.get_unchecked(input_index3);
                }
            }

            y += 4;
        }

        for y in y..height {
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
#[inline]
pub(crate) fn reverse_bits<const D: usize>(value: usize, rev_digits: u32) -> usize {
    debug_assert!(D > 1);

    let mut result: usize = 0;
    let mut value = value;
    for _ in 0..rev_digits {
        result = (result * D) + (value % D);
        value = value / D;
    }
    result
}

// computes `n` such that `D ^ n == value`. Returns `None` if `value` is not a perfect power of `D`, otherwise returns `Some(n)`
pub(crate) fn int_logarithm<const D: usize>(value: usize) -> Option<u32> {
    if value == 0 || D < 2 {
        return None;
    }

    let mut current_exponent = 0;
    let mut current_value = value;

    while current_value.is_multiple_of(D) {
        current_exponent += 1;
        current_value /= D;
    }

    if current_value == 1 {
        Some(current_exponent)
    } else {
        None
    }
}

#[inline]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
pub(crate) fn has_valid_avx() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(test)]
macro_rules! test_radix {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $iters: expr, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            use crate::FftDirection;
            use crate::FftExecutor;
            use num_complex::Complex;
            use rand::Rng;
            for i in 1..$iters {
                let val = $scale as usize;
                let size = val.pow(i);
                let mut input = vec![Complex::<$data_type>::default(); size];
                for z in input.iter_mut() {
                    *z = Complex {
                        re: rand::rng().random(),
                        im: rand::rng().random(),
                    };
                }
                let src = input.to_vec();
                use crate::dft::Dft;
                let reference_forward = Dft::new(size, FftDirection::Forward).unwrap();

                let mut ref_src = src.to_vec();
                reference_forward.execute(&mut ref_src).unwrap();

                let radix_forward = $butterfly::new(size, FftDirection::Forward).unwrap();
                let radix_inverse = $butterfly::new(size, FftDirection::Inverse).unwrap();
                radix_forward.execute(&mut input).unwrap();

                input
                    .iter()
                    .zip(ref_src.iter())
                    .enumerate()
                    .for_each(|(idx, (a, b))| {
                        assert!(
                            (a.re - b.re).abs() < $tol,
                            "a_re {} != b_re {} for size {} at {idx}",
                            a.re,
                            b.re,
                            size
                        );
                        assert!(
                            (a.im - b.im).abs() < $tol,
                            "a_im {} != b_im {} for size {} at {idx}",
                            a.im,
                            b.im,
                            size
                        );
                    });

                radix_inverse.execute(&mut input).unwrap();

                input = input
                    .iter()
                    .map(|&x| x * (1.0 / size as $data_type))
                    .collect();

                input.iter().zip(src.iter()).for_each(|(a, b)| {
                    assert!(
                        (a.re - b.re).abs() < $tol,
                        "a_re {} != b_re {} for size {}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < $tol,
                        "a_im {} != b_im {} for size {}",
                        a.im,
                        b.im,
                        size
                    );
                });
            }
        }
    };
}

#[cfg(test)]
pub(crate) use test_radix;
