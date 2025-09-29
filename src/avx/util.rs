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
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn _m128d_fma_mul_complex(a: __m128d, b: __m128d) -> __m128d {
    let mut temp1 = _mm_unpacklo_pd(b, b);
    let mut temp2 = _mm_unpackhi_pd(b, b);
    temp1 = _mm_mul_pd(temp1, a);
    temp2 = _mm_mul_pd(temp2, a);
    temp2 = _mm_shuffle_pd(temp2, temp2, 0x01);
    _mm_addsub_pd(temp1, temp2)
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn _m256d_mul_complex(a: __m256d, b: __m256d) -> __m256d {
    // Swap real and imaginary parts of 'a' for FMA
    let a_yx = _mm256_permute_pd::<0b0101>(a); // [a_im, a_re, b_im, b_re]

    // Duplicate real and imaginary parts of 'b'
    let b_xx = _mm256_permute_pd::<0b0000>(b); // [c_re, c_re, d_re, d_re]
    let b_yy = _mm256_permute_pd::<0b1111>(b); // [c_im, c_im, d_im, d_im]

    // Compute (a_re*b_re - a_im*b_im) + i(a_re*b_im + a_im*b_re)
    _mm256_fmaddsub_pd(a, b_xx, _mm256_mul_pd(a_yx, b_yy))
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn _m128s_fma_mul_complex(a: __m128, b: __m128) -> __m128 {
    let temp1 = _mm_shuffle_ps::<0xA0>(b, b);
    let temp2 = _mm_shuffle_ps::<0xF5>(b, b);
    let mul2 = _mm_mul_ps(a, temp2);
    let mul2 = _mm_shuffle_ps::<0xB1>(mul2, mul2);
    _mm_fmaddsub_ps(a, temp1, mul2)
}

#[inline]
#[target_feature(enable = "sse4.2")]
pub(crate) unsafe fn _m128s_load_f32x2(a: *const Complex<f32>) -> __m128 {
    unsafe { _mm_castsi128_ps(_mm_loadu_si64(a.cast())) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
pub(crate) unsafe fn _m128s_store_f32x2(a: *mut Complex<f32>, b: __m128) {
    unsafe { _mm_storeu_si64(a.cast(), _mm_castps_si128(b)) }
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn _m256s_mul_complex(a: __m256, b: __m256) -> __m256 {
    // Extract real and imag parts from a
    let ar = _mm256_moveldup_ps(a); // duplicate even lanes (re parts)
    let ai = _mm256_movehdup_ps(a); // duplicate odd lanes (im parts)

    // Swap real/imag of b for cross terms
    let bswap = _mm256_permute_ps::<0b10110001>(b); // [im, re, im, re, ...]

    // re = ar*br - ai*bi
    // im = ar*bi + ai*br
    _mm256_fmaddsub_ps(ar, b, _mm256_mul_ps(ai, bswap))
}

#[inline(always)]
pub(crate) const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::ComplexFloat;

    #[test]
    fn complex_muld() {
        let values_a = [Complex::new(7.0f64, 5.0), Complex::new(5.0, -1.15)];
        let values_b = [Complex::new(-5.0f64, 3.0), Complex::new(-1.0, 1.15)];
        let r = values_a
            .iter()
            .zip(values_b.iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<Complex<_>>>();
        unsafe {
            let a0 = _mm256_loadu_pd(values_a.as_ptr().cast());
            let b0 = _mm256_loadu_pd(values_b.as_ptr().cast());
            let product = _m256d_mul_complex(a0, b0);
            let mut vec_b = vec![Complex::<f64>::default(); 2];
            _mm256_storeu_pd(vec_b.as_mut_ptr().cast(), product);
            vec_b.iter().zip(r.iter()).for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-10);
            });
        }
    }
}
