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
#[target_feature(enable = "avx")]
pub(crate) unsafe fn _mm256_set_complexd(v: Complex<f64>) -> __m256d {
    _mm256_setr_pd(v.re, v.im, v.re, v.im)
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn _mm_set_complex(v: Complex<f32>) -> __m128 {
    _mm_setr_ps(v.re, v.im, v.re, v.im)
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn _mm256_set4_complex(v0: Complex<f32>, v1: Complex<f32>, v2: Complex<f32>, v3: Complex<f32>) -> __m256 {
    _mm256_setr_ps(v0.re, v0.im, v1.re, v1.im, v2.re, v2.im, v3.re, v3.im)
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn _mm256_set2_complexd(v0: Complex<f64>, v1: Complex<f64>) -> __m256d {
    _mm256_setr_pd(v0.re, v0.im, v1.re, v1.im)
}

#[inline]
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm_fcmul_pd(a: __m128d, b: __m128d) -> __m128d {
    let temp1 = _mm_unpacklo_pd(b, b);
    let mut temp2 = _mm_unpackhi_pd(b, b);
    temp2 = _mm_mul_pd(temp2, a);
    temp2 = _mm_shuffle_pd(temp2, temp2, 0x01);
    _mm_fmaddsub_pd(temp1, a, temp2)
}

#[inline]
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm256_fcmul_pd(a: __m256d, b: __m256d) -> __m256d {
    // Swap real and imaginary parts of 'a' for FMA
    let a_yx = _mm256_permute_pd::<0b0101>(a); // [a_im, a_re, b_im, b_re]

    // Duplicate real and imaginary parts of 'b'
    let b_xx = _mm256_permute_pd::<0b0000>(b); // [c_re, c_re, d_re, d_re]
    let b_yy = _mm256_permute_pd::<0b1111>(b); // [c_im, c_im, d_im, d_im]

    // Compute (a_re*b_re - a_im*b_im) + i(a_re*b_im + a_im*b_re)
    _mm256_fmaddsub_pd(a, b_xx, _mm256_mul_pd(a_yx, b_yy))
}

#[inline]
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm_fcmul_ps(a: __m128, b: __m128) -> __m128 {
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
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm256_fcmul_ps(a: __m256, b: __m256) -> __m256 {
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

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn _mm_unpacklo_ps64(a: __m128, b: __m128) -> __m128 {
    _mm_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(a, b)
}

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn _mm_unpacklohi_ps64(a: __m128, b: __m128) -> __m128 {
    _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(a, b)
}

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn _mm_unpackhilo_ps64(a: __m128, b: __m128) -> __m128 {
    _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(a, b)
}

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn _mm_unpackhi_ps64(a: __m128, b: __m128) -> __m128 {
    _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(a, b)
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn _mm256_unpacklo_pd2(a: __m256d, b: __m256d) -> __m256d {
    _mm256_permute2f128_pd::<0x20>(a, b)
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn _mm256_unpackhi_pd2(a: __m256d, b: __m256d) -> __m256d {
    _mm256_permute2f128_pd::<0x31>(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn _mm256s_interleave4_epi64(
    a: __m256,
    b: __m256,
    c: __m256,
    d: __m256,
) -> (__m256, __m256, __m256, __m256) {
    let bg0 = _mm256_unpacklo_epi64(_mm256_castps_si256(a), _mm256_castps_si256(b));
    let bg1 = _mm256_unpackhi_epi64(_mm256_castps_si256(a), _mm256_castps_si256(b));
    let ra0 = _mm256_unpacklo_epi64(_mm256_castps_si256(c), _mm256_castps_si256(d));
    let ra1 = _mm256_unpackhi_epi64(_mm256_castps_si256(c), _mm256_castps_si256(d));

    let xy0 = _mm256_permute2x128_si256::<32>(bg0, ra0);
    let xy1 = _mm256_permute2x128_si256::<32>(bg1, ra1);
    let xy2 = _mm256_permute2x128_si256::<49>(bg0, ra0);
    let xy3 = _mm256_permute2x128_si256::<49>(bg1, ra1);
    (
        _mm256_castsi256_ps(xy0),
        _mm256_castsi256_ps(xy1),
        _mm256_castsi256_ps(xy2),
        _mm256_castsi256_ps(xy3),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn _mm256s_deinterleave4_epi64(
    a: __m256,
    b: __m256,
    c: __m256,
    d: __m256,
) -> (__m256, __m256, __m256, __m256) {
    let l02 = _mm256_permute2x128_si256::<32>(_mm256_castps_si256(a), _mm256_castps_si256(c));
    let h02 = _mm256_permute2x128_si256::<49>(_mm256_castps_si256(a), _mm256_castps_si256(c));
    let l13 = _mm256_permute2x128_si256::<32>(_mm256_castps_si256(b), _mm256_castps_si256(d));
    let h13 = _mm256_permute2x128_si256::<49>(_mm256_castps_si256(b), _mm256_castps_si256(d));

    let xy0 = _mm256_unpacklo_epi64(l02, l13);
    let xy1 = _mm256_unpackhi_epi64(l02, l13);
    let xy2 = _mm256_unpacklo_epi64(h02, h13);
    let xy3 = _mm256_unpackhi_epi64(h02, h13);
    (
        _mm256_castsi256_ps(xy0),
        _mm256_castsi256_ps(xy1),
        _mm256_castsi256_ps(xy2),
        _mm256_castsi256_ps(xy3),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn _mm256_permute4x64_ps<const IMM: i32>(a: __m256) -> __m256 {
    _mm256_castpd_ps(_mm256_permute4x64_pd::<IMM>(_mm256_castps_pd(a)))
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn _mm256s_deinterleave3_epi64(
    a: __m256,
    b: __m256,
    c: __m256,
) -> (__m256, __m256, __m256) {
    let s01 = _mm256_blend_epi32::<0xf0>(_mm256_castps_si256(a), _mm256_castps_si256(b));
    let s12 = _mm256_blend_epi32::<0xf0>(_mm256_castps_si256(b), _mm256_castps_si256(c));
    let s20r = _mm256_permute4x64_epi64::<0x1b>(_mm256_blend_epi32::<0xf0>(
        _mm256_castps_si256(c),
        _mm256_castps_si256(a),
    ));
    let xy0 = _mm256_unpacklo_epi64(s01, s20r);
    let xy1 = _mm256_alignr_epi8::<8>(s12, s01);
    let xy2 = _mm256_unpackhi_epi64(s20r, s12);

    (
        _mm256_castsi256_ps(xy0),
        _mm256_castsi256_ps(xy1),
        _mm256_castsi256_ps(xy2),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn _mm256s_interleave2_epi64(a: __m256, b: __m256) -> (__m256, __m256) {
    let xy_l = _mm256_unpacklo_pd(_mm256_castps_pd(a), _mm256_castps_pd(b));
    let xy_h = _mm256_unpackhi_pd(_mm256_castps_pd(a), _mm256_castps_pd(b));

    let xy0 = _mm256_permute2f128_pd::<32>(xy_l, xy_h);
    let xy1 = _mm256_permute2f128_pd::<49>(xy_l, xy_h);
    (_mm256_castpd_ps(xy0), _mm256_castpd_ps(xy1))
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn _mm256s_interleave3_epi64(
    a: __m256,
    b: __m256,
    c: __m256,
) -> (__m256, __m256, __m256) {
    {
        let s01 = _mm256_unpacklo_epi64(_mm256_castps_si256(a), _mm256_castps_si256(b));
        let s12 = _mm256_unpackhi_epi64(_mm256_castps_si256(b), _mm256_castps_si256(c));
        let s20 = _mm256_blend_epi32::<0xcc>(_mm256_castps_si256(c), _mm256_castps_si256(a));

        let xy0 = _mm256_permute2x128_si256::<32>(s01, s20);
        let xy1 = _mm256_blend_epi32(s01, s12, 0x0f);
        let xy2 = _mm256_permute2x128_si256::<49>(s20, s12);

        (
            _mm256_castsi256_ps(xy0),
            _mm256_castsi256_ps(xy1),
            _mm256_castsi256_ps(xy2),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn _mm128s_deinterleave3_epi64(
    a: __m128,
    b: __m128,
    c: __m128,
) -> (__m128, __m128, __m128) {
    let t1 = _mm_shuffle_epi32::<0x4e>(_mm_castps_si128(b)); // a1, c0

    let xy0 = _mm_unpacklo_epi64(_mm_castps_si128(a), t1);
    let xy1 = _mm_unpacklo_epi64(
        _mm_unpackhi_epi64(_mm_castps_si128(a), _mm_castps_si128(a)),
        _mm_castps_si128(c),
    );
    let xy2 = _mm_unpackhi_epi64(t1, _mm_castps_si128(c));

    (
        _mm_castsi128_ps(xy0),
        _mm_castsi128_ps(xy1),
        _mm_castsi128_ps(xy2),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn _mm256s_deinterleave2_epi64(a: __m256, b: __m256) -> (__m256, __m256) {
    let pl = _mm256_permute2f128_pd::<32>(_mm256_castps_pd(a), _mm256_castps_pd(b));
    let ph = _mm256_permute2f128_pd::<49>(_mm256_castps_pd(a), _mm256_castps_pd(b));
    let a0 = _mm256_unpacklo_pd(pl, ph);
    let b0 = _mm256_unpackhi_pd(pl, ph);
    (_mm256_castpd_ps(a0), _mm256_castpd_ps(b0))
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn _mm256_load4_f32x2(
    a: &[Complex<f32>],
    b: &[Complex<f32>],
    c: &[Complex<f32>],
    d: &[Complex<f32>],
) -> __m256 {
    unsafe {
        _mm256_insertf128_ps::<1>(
            _mm256_castps128_ps256(_mm_unpacklo_ps64(
                _m128s_load_f32x2(a.as_ptr().cast()),
                _m128s_load_f32x2(b.as_ptr().cast()),
            )),
            _mm_unpacklo_ps64(
                _m128s_load_f32x2(c.as_ptr().cast()),
                _m128s_load_f32x2(d.as_ptr().cast()),
            ),
        )
    }
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn _mm256_create_ps(a: __m128, b: __m128) -> __m256 {
    _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(a), b)
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn _mm256_create_pd(a: __m128d, b: __m128d) -> __m256d {
    _mm256_insertf128_pd::<1>(_mm256_castpd128_pd256(a), b)
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn _mm256_unpacklo_ps64(a: __m256, b: __m256) -> __m256 {
    _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)))
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn _mm256_blend_ps64<const IMM8: i32>(a: __m256, b: __m256) -> __m256 {
    _mm256_castpd_ps(_mm256_blend_pd::<IMM8>(
        _mm256_castps_pd(a),
        _mm256_castps_pd(b),
    ))
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn _mm256_unpackhi_ps64(a: __m256, b: __m256) -> __m256 {
    _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)))
}

// a.conj() * b
#[inline]
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm256_fcmul_ps_conj_a(a: __m256, b: __m256) -> __m256 {
    // Extract real and imag parts from a
    let ar = _mm256_moveldup_ps(a); // duplicate even lanes (re parts)
    let ai = _mm256_movehdup_ps(a); // duplicate odd lanes (im parts)

    // Swap real/imag of b for cross terms
    let bswap = _mm256_permute_ps::<0b10110001>(b); // [im, re, im, re, ...]

    // re = ar*br - -ai*bi
    // im = ar*bi - ai*br
    _mm256_fmsubadd_ps(ar, b, _mm256_mul_ps(ai, bswap))
}

// a * b.conj()
#[inline]
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm256_fcmul_ps_conj_b(a: __m256, b: __m256) -> __m256 {
    // Extract real and imag parts from a
    let ar = _mm256_moveldup_ps(b); // duplicate even lanes (re parts)
    let ai = _mm256_movehdup_ps(b); // duplicate odd lanes (im parts)

    // Swap real/imag of b for cross terms
    let bswap = _mm256_permute_ps::<0b10110001>(a); // [im, re, im, re, ...]

    // re = ar*br - -ai*bi
    // im = ar*bi - ai*br
    _mm256_fmsubadd_ps(ar, a, _mm256_mul_ps(ai, bswap))
}

// a.conj() * b
#[inline]
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm_fcmul_ps_conj_a(a: __m128, b: __m128) -> __m128 {
    let temp1 = _mm_shuffle_ps::<0xA0>(a, a);
    let temp2 = _mm_shuffle_ps::<0xF5>(a, a);
    let mul2 = _mm_mul_ps(b, temp2);
    let mul2 = _mm_shuffle_ps::<0xB1>(mul2, mul2);
    _mm_fmsubadd_ps(b, temp1, mul2)
}

// a * b.conj()
#[inline]
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm_fcmul_ps_conj_b(a: __m128, b: __m128) -> __m128 {
    let temp1 = _mm_shuffle_ps::<0xA0>(b, b);
    let temp2 = _mm_shuffle_ps::<0xF5>(b, b);
    let mul2 = _mm_mul_ps(a, temp2);
    let mul2 = _mm_shuffle_ps::<0xB1>(mul2, mul2);
    _mm_fmsubadd_ps(a, temp1, mul2)
}

// a.conj() * b
#[inline]
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm256_fcmul_pd_conj_a(a: __m256d, b: __m256d) -> __m256d {
    // Swap real and imaginary parts of 'a' for FMA
    let a_yx = _mm256_permute_pd::<0b0101>(a); // [a_im, a_re, b_im, b_re]

    // Duplicate real and imaginary parts of 'b'
    let b_xx = _mm256_permute_pd::<0b0000>(b); // [c_re, c_re, d_re, d_re]
    let b_yy = _mm256_permute_pd::<0b1111>(b); // [c_im, c_im, d_im, d_im]

    _mm256_fmsubadd_pd(a_yx, b_yy, _mm256_mul_pd(a, b_xx))
}

// a * b.conj()
#[inline]
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm256_fcmul_pd_conj_b(a: __m256d, b: __m256d) -> __m256d {
    // Swap real and imaginary parts of 'a' for FMA
    let a_yx = _mm256_permute_pd::<0b0101>(b); // [a_im, a_re, b_im, b_re]

    // Duplicate real and imaginary parts of 'b'
    let b_xx = _mm256_permute_pd::<0b0000>(a); // [c_re, c_re, d_re, d_re]
    let b_yy = _mm256_permute_pd::<0b1111>(a); // [c_im, c_im, d_im, d_im]

    _mm256_fmsubadd_pd(a_yx, b_yy, _mm256_mul_pd(b, b_xx))
}

// a.conj() * b
#[inline]
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm_fcmul_pd_conj_a(a: __m128d, b: __m128d) -> __m128d {
    let temp1 = _mm_unpacklo_pd(a, a); // [b.re, b.re]
    let mut temp2 = _mm_unpackhi_pd(a, a); // [b.im, b.im]
    temp2 = _mm_mul_pd(temp2, b); // [b.im * a.re, b.im * a.im]
    temp2 = _mm_shuffle_pd::<0x01>(temp2, temp2); // [b.im * a.im, b.im * a.im]
    _mm_fmsubadd_pd(temp1, b, temp2)
}

// a * b.conj()
#[inline]
#[target_feature(enable = "avx", enable = "fma")]
pub(crate) unsafe fn _mm_fcmul_pd_conj_b(a: __m128d, b: __m128d) -> __m128d {
    let temp1 = _mm_unpacklo_pd(b, b); // [b.re, b.re]
    let mut temp2 = _mm_unpackhi_pd(b, b); // [b.im, b.im]
    temp2 = _mm_mul_pd(temp2, a); // [b.im * a.re, b.im * a.im]
    temp2 = _mm_shuffle_pd::<0x01>(temp2, temp2); // [b.im * a.im, b.im * a.im]
    _mm_fmsubadd_pd(temp1, a, temp2)
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
            let product = _mm256_fcmul_pd(a0, b0);
            let mut vec_b = vec![Complex::<f64>::default(); 2];
            _mm256_storeu_pd(vec_b.as_mut_ptr().cast(), product);
            vec_b.iter().zip(r.iter()).for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-10);
            });
        }
    }

    #[test]
    fn complexd_a_conj_to_b_avx() {
        let values_a = [Complex::new(7.0f64, 5.0), Complex::new(5.0, -1.15)];
        let values_b = [Complex::new(-5.0f64, 3.0), Complex::new(-1.0, 1.15)];
        let r = values_a
            .iter()
            .zip(values_b.iter())
            .map(|(a, b)| a.conj() * b)
            .collect::<Vec<Complex<_>>>();
        unsafe {
            let a0 = _mm256_loadu_pd(values_a.as_ptr().cast());
            let b0 = _mm256_loadu_pd(values_b.as_ptr().cast());
            let product = _mm256_fcmul_pd_conj_a(a0, b0);
            let mut vec_b = vec![Complex::<f64>::default(); 4];
            _mm256_storeu_pd(vec_b.as_mut_ptr().cast(), product);
            vec_b.iter().zip(r.iter()).for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-10);
            });
        }
    }

    #[test]
    fn complexd_a_conj_to_b_sse() {
        let values_a = [Complex::new(7.0f64, 5.0)];
        let values_b = [Complex::new(-5.0f64, 3.0)];
        let r = values_a
            .iter()
            .zip(values_b.iter())
            .map(|(a, b)| a.conj() * b)
            .collect::<Vec<Complex<_>>>();
        unsafe {
            let a0 = _mm_loadu_pd(values_a.as_ptr().cast());
            let b0 = _mm_loadu_pd(values_b.as_ptr().cast());
            let product = _mm_fcmul_pd_conj_a(a0, b0);
            let mut vec_b = vec![Complex::<f64>::default(); 2];
            _mm_storeu_pd(vec_b.as_mut_ptr().cast(), product);
            vec_b.iter().zip(r.iter()).for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-10, "a {a}, b {b}");
            });
        }
    }

    #[test]
    fn complex_a_conj_to_b_avx() {
        let values_a = [Complex::new(7.0f32, 5.0), Complex::new(5.0, -1.15)];
        let values_b = [Complex::new(-5.0f32, 3.0), Complex::new(-1.0, 1.15)];
        let r = values_a
            .iter()
            .zip(values_b.iter())
            .map(|(a, b)| a.conj() * b)
            .collect::<Vec<Complex<_>>>();
        unsafe {
            let a0 = _mm256_loadu_ps(values_a.as_ptr().cast());
            let b0 = _mm256_loadu_ps(values_b.as_ptr().cast());
            let product = _mm256_fcmul_ps_conj_a(a0, b0);
            let mut vec_b = vec![Complex::<f32>::default(); 4];
            _mm256_storeu_ps(vec_b.as_mut_ptr().cast(), product);
            vec_b.iter().zip(r.iter()).for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-5, "complex_a_to_b_conj_avx a {a}, b {b}");
            });
        }
    }

    #[test]
    fn complex_a_conj_to_b_sse() {
        let values_a = [Complex::new(7.0f32, 5.0)];
        let values_b = [Complex::new(-5.0f32, 3.0)];
        let r = values_a
            .iter()
            .zip(values_b.iter())
            .map(|(a, b)| a.conj() * b)
            .collect::<Vec<Complex<_>>>();
        unsafe {
            let a0 = _mm_loadu_ps(values_a.as_ptr().cast());
            let b0 = _mm_loadu_ps(values_b.as_ptr().cast());
            let product = _mm_fcmul_ps_conj_a(a0, b0);
            let mut vec_b = vec![Complex::<f32>::default(); 2];
            _mm_storeu_ps(vec_b.as_mut_ptr().cast(), product);
            vec_b.iter().zip(r.iter()).for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-5, "complex_a_to_b_conj_sse a {a}, b {b}");
            });
        }
    }

    #[test]
    fn complex_a_to_b_conj_sse() {
        let values_a = [Complex::new(7.0f32, 5.0)];
        let values_b = [Complex::new(-5.0f32, 3.0)];
        let r = values_a
            .iter()
            .zip(values_b.iter())
            .map(|(a, b)| a * b.conj())
            .collect::<Vec<Complex<_>>>();
        unsafe {
            let a0 = _mm_loadu_ps(values_a.as_ptr().cast());
            let b0 = _mm_loadu_ps(values_b.as_ptr().cast());
            let product = _mm_fcmul_ps_conj_b(a0, b0);
            let mut vec_b = vec![Complex::<f32>::default(); 2];
            _mm_storeu_ps(vec_b.as_mut_ptr().cast(), product);
            vec_b.iter().zip(r.iter()).for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-5, "complex_a_to_b_conj_sse a {a}, b {b}");
            });
        }
    }

    #[test]
    fn complex_a_to_b_conj_sse_f64() {
        let values_a = [Complex::new(7.0f64, 5.0)];
        let values_b = [Complex::new(-5.0f64, 3.0)];
        let r = values_a
            .iter()
            .zip(values_b.iter())
            .map(|(a, b)| a * b.conj())
            .collect::<Vec<Complex<_>>>();
        unsafe {
            let a0 = _mm_loadu_pd(values_a.as_ptr().cast());
            let b0 = _mm_loadu_pd(values_b.as_ptr().cast());
            let product = _mm_fcmul_pd_conj_b(a0, b0);
            let mut vec_b = vec![Complex::<f64>::default(); 1];
            _mm_storeu_pd(vec_b.as_mut_ptr().cast(), product);
            vec_b.iter().zip(r.iter()).for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-5, "complex_a_to_b_conj_sse a {a}, b {b}");
            });
        }
    }

    #[test]
    fn complexd_a_to_b_conj_avx() {
        let values_a = [Complex::new(7.0f64, 5.0), Complex::new(5.0, -1.15)];
        let values_b = [Complex::new(-5.0f64, 3.0), Complex::new(-1.0, 1.15)];
        let r = values_a
            .iter()
            .zip(values_b.iter())
            .map(|(a, b)| a * b.conj())
            .collect::<Vec<Complex<_>>>();
        unsafe {
            let a0 = _mm256_loadu_pd(values_a.as_ptr().cast());
            let b0 = _mm256_loadu_pd(values_b.as_ptr().cast());
            let product = _mm256_fcmul_pd_conj_b(a0, b0);
            let mut vec_b = vec![Complex::<f64>::default(); 4];
            _mm256_storeu_pd(vec_b.as_mut_ptr().cast(), product);
            vec_b.iter().zip(r.iter()).for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-10);
            });
        }
    }

    #[test]
    fn complexf_a_to_b_conj_avx() {
        let values_a = [Complex::new(7.0f32, 5.0), Complex::new(5.0, -1.15)];
        let values_b = [Complex::new(-5.0f32, 3.0), Complex::new(-1.0, 1.15)];
        let r = values_a
            .iter()
            .zip(values_b.iter())
            .map(|(a, b)| a * b.conj())
            .collect::<Vec<Complex<_>>>();
        unsafe {
            let a0 = _mm256_loadu_ps(values_a.as_ptr().cast());
            let b0 = _mm256_loadu_ps(values_b.as_ptr().cast());
            let product = _mm256_fcmul_ps_conj_b(a0, b0);
            let mut vec_b = vec![Complex::<f32>::default(); 4];
            _mm256_storeu_ps(vec_b.as_mut_ptr().cast(), product);
            vec_b.iter().zip(r.iter()).for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-5);
            });
        }
    }
}
