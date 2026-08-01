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
use crate::avx::util::{_mm256_fcmul_pd, _mm256_fcmul_pd_conj_a, shuffle};
use num_complex::Complex;
use num_traits::MulAdd;
use std::arch::x86_64::*;
use std::mem::MaybeUninit;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct AvxStoreD {
    pub(crate) v: __m256d,
}

#[derive(Copy, Clone)]
pub(crate) struct AvxMaskD {
    v: __m256i,
    lanes: usize,
}

impl AvxMaskD {
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn real(count: usize) -> Self {
        debug_assert!(count <= 4);
        Self {
            v: _mm256_cmpgt_epi64(
                _mm256_set1_epi64x(count as i64),
                _mm256_setr_epi64x(0, 1, 2, 3),
            ),
            lanes: count,
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn complex(count: usize) -> Self {
        debug_assert!(count <= 2);
        Self::real(count * 2)
    }
}

impl AvxStoreD {
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn dup(p0: f64) -> AvxStoreD {
        AvxStoreD::raw(_mm256_set1_pd(p0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn dup_even_odds(&self) -> [Self; 2] {
        [
            AvxStoreD::raw(_mm256_movedup_pd(self.v)),
            AvxStoreD::raw(_mm256_shuffle_pd::<0x0F>(self.v, self.v)),
        ]
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse_complex(&self) -> Self {
        AvxStoreD::raw(_mm256_permute2f128_pd::<0x01>(self.v, self.v))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn dup_lo_complex(&self) -> Self {
        Self::raw(_mm256_setr_m128d(
            _mm256_castpd256_pd128(self.v),
            _mm256_castpd256_pd128(self.v),
        ))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse_complex_elements(&self) -> Self {
        AvxStoreD::raw(_mm256_shuffle_pd::<0x05>(self.v, self.v))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn xor(&self, other: Self) -> Self {
        AvxStoreD::raw(_mm256_xor_pd(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn set_values(p0: f64, p1: f64, p2: f64, p3: f64) -> Self {
        AvxStoreD::raw(_mm256_setr_pd(p0, p1, p2, p3))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn conj_flag() -> Self {
        AvxStoreD::set_values(0.0, -0.0, 0.0, -0.0)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn blend_real_img(&self, other: Self) -> Self {
        AvxStoreD::raw(_mm256_blend_pd::<0b1010>(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn neg(&self) -> AvxStoreD {
        AvxStoreD::raw(_mm256_xor_pd(self.v, _mm256_set1_pd(-0.0)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn raw(r: __m256d) -> AvxStoreD {
        AvxStoreD { v: r }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex_ref(complex: &[Complex<f64>]) -> Self {
        unsafe {
            AvxStoreD {
                v: _mm256_loadu_pd(complex.as_ptr().cast()),
            }
        }
    }

    /// Loads `count` complex values and zeroes the remaining complex lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex_partial(complex: &[Complex<f64>], mask: AvxMaskD) -> Self {
        debug_assert!(mask.lanes.is_multiple_of(2) && complex.len() >= mask.lanes / 2);
        if mask.lanes == 4 {
            return Self::from_complex_ref(complex);
        }
        unsafe {
            AvxStoreD {
                v: _mm256_maskload_pd(complex.as_ptr().cast(), mask.v),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load(complex: &[f64]) -> Self {
        unsafe {
            AvxStoreD {
                v: _mm256_loadu_pd(complex.as_ptr().cast()),
            }
        }
    }

    /// Loads `count` f64 lanes and zeroes the remaining lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load_partial(complex: &[f64], mask: AvxMaskD) -> Self {
        debug_assert!(complex.len() >= mask.lanes);
        if mask.lanes == 4 {
            return Self::load(complex);
        }
        unsafe {
            AvxStoreD {
                v: _mm256_maskload_pd(complex.as_ptr(), mask.v),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex_refu(complex: &[MaybeUninit<Complex<f64>>]) -> Self {
        unsafe {
            AvxStoreD {
                v: _mm256_loadu_pd(complex.as_ptr().cast()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex(complex: &Complex<f64>) -> Self {
        unsafe {
            AvxStoreD {
                v: _mm256_castpd128_pd256(_mm_loadu_pd(
                    complex as *const Complex<f64> as *const f64,
                )),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load2(ptr: &[f64]) -> Self {
        unsafe {
            let q0 = _mm_loadu_pd(ptr.as_ptr().cast());
            AvxStoreD::raw(_mm256_castpd128_pd256(q0))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load1(ptr: &[f64]) -> Self {
        unsafe {
            let q0 = _mm_load_sd(ptr.as_ptr().cast());
            AvxStoreD::raw(_mm256_castpd128_pd256(q0))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load1_ref(ptr: &f64) -> Self {
        unsafe {
            let q0 = _mm_shuffle_pd::<0b00>(_mm_load_sd(ptr), _mm_setzero_pd());
            AvxStoreD::raw(_mm256_castpd128_pd256(q0))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn set_complex(complex: &Complex<f64>) -> Self {
        AvxStoreD {
            v: _mm256_setr_pd(complex.re, complex.im, complex.re, complex.im),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complexu(complex: &MaybeUninit<Complex<f64>>) -> Self {
        unsafe {
            AvxStoreD {
                v: _mm256_castpd128_pd256(_mm_loadu_pd(
                    complex as *const MaybeUninit<Complex<f64>> as *const f64,
                )),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn set_complex2(p0: Complex<f64>, p1: Complex<f64>) -> AvxStoreD {
        AvxStoreD::raw(_mm256_setr_pd(p0.re, p0.im, p1.re, p1.im))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f64>]) {
        unsafe { _mm256_storeu_pd(to_ref.as_mut_ptr().cast(), self.v) }
    }

    /// Stores the first `count` complex values without touching following values.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_partial(&self, to_ref: &mut [Complex<f64>], mask: AvxMaskD) {
        debug_assert!(mask.lanes.is_multiple_of(2) && to_ref.len() >= mask.lanes / 2);
        if mask.lanes == 4 {
            return self.write(to_ref);
        }
        unsafe { _mm256_maskstore_pd(to_ref.as_mut_ptr().cast(), mask.v, self.v) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_single(&self, to_ref: &mut Complex<f64>) {
        unsafe {
            _mm_storeu_pd(
                to_ref as *mut Complex<f64> as *mut f64,
                _mm256_castpd256_pd128(self.v),
            )
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_u(&self, to_ref: &mut [MaybeUninit<Complex<f64>>]) {
        unsafe { _mm256_storeu_pd(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_lo(&self, to_ref: &mut [Complex<f64>]) {
        unsafe { _mm_storeu_pd(to_ref.as_mut_ptr().cast(), _mm256_castpd256_pd128(self.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_hi(&self, to_ref: &mut [Complex<f64>]) {
        unsafe {
            _mm_storeu_pd(
                to_ref.as_mut_ptr().cast(),
                _mm256_extractf128_pd::<1>(self.v),
            )
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_lou(&self, to_ref: &mut [MaybeUninit<Complex<f64>>]) {
        unsafe { _mm_storeu_pd(to_ref.as_mut_ptr().cast(), _mm256_castpd256_pd128(self.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_real(&self, to_ref: &mut [f64]) {
        unsafe { _mm256_storeu_pd(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_real_lo1(&self, to_ref: &mut [f64]) {
        unsafe { _mm_storel_pd(to_ref.as_mut_ptr().cast(), _mm256_castpd256_pd128(self.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_real_lo2(&self, to_ref: &mut [f64]) {
        unsafe { _mm_storeu_pd(to_ref.as_mut_ptr().cast(), _mm256_castpd256_pd128(self.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_real_lo3(&self, to_ref: &mut [f64]) {
        unsafe { _mm_storeu_pd(to_ref.as_mut_ptr().cast(), _mm256_castpd256_pd128(self.v)) }
        unsafe {
            _mm_storel_pd(
                to_ref.get_unchecked_mut(2..).as_mut_ptr().cast(),
                _mm256_extractf128_pd::<1>(self.v),
            )
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn mul_by_complex(self, other: AvxStoreD) -> Self {
        AvxStoreD {
            v: _mm256_fcmul_pd(self.v, other.v),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn mul_by_conj_a(self, other: AvxStoreD) -> Self {
        AvxStoreD {
            v: _mm256_fcmul_pd_conj_a(self.v, other.v),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn unpack_evens(&self, other: Self) -> Self {
        let q = _mm256_unpacklo_pd(self.v, other.v);
        Self::raw(_mm256_permute4x64_pd::<{ shuffle(3, 1, 2, 0) }>(q))
    }
}

impl AvxStoreD {
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn zero() -> Self {
        Self {
            v: _mm256_setzero_pd(),
        }
    }

    #[inline(always)]
    pub(crate) fn undefined() -> Self {
        Self {
            v: unsafe { _mm256_undefined_pd() },
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn to_complex(self) -> [Self; 2] {
        self.zip(AvxStoreD::zero())
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn zip(self, other: Self) -> [Self; 2] {
        let r0 = _mm256_shuffle_pd::<0b0000>(self.v, other.v);
        let r1 = _mm256_shuffle_pd::<0b1111>(self.v, other.v);
        let xy0 = _mm256_permute2f128_pd::<32>(r0, r1);
        let xy1 = _mm256_permute2f128_pd::<49>(r0, r1);
        [AvxStoreD::raw(xy0), AvxStoreD::raw(xy1)]
    }
}

impl MulAdd<AvxStoreD> for AvxStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: AvxStoreD, b: Self) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_fmadd_pd(self.v, a.v, b.v)) }
    }
}

impl Mul<f64> for AvxStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_mul_pd(self.v, _mm256_set1_pd(rhs))) }
    }
}

impl Mul<AvxStoreD> for f64 {
    type Output = AvxStoreD;

    #[inline(always)]
    fn mul(self, rhs: AvxStoreD) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_mul_pd(rhs.v, _mm256_set1_pd(self))) }
    }
}

impl Mul<AvxStoreD> for AvxStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: AvxStoreD) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_mul_pd(self.v, rhs.v)) }
    }
}

impl Add<AvxStoreD> for AvxStoreD {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: AvxStoreD) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_add_pd(self.v, rhs.v)) }
    }
}

impl Sub<AvxStoreD> for AvxStoreD {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: AvxStoreD) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_sub_pd(self.v, rhs.v)) }
    }
}

impl Neg for AvxStoreD {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_xor_pd(self.v, _mm256_set1_pd(-0.0))) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_loads_zero_and_partial_stores_preserve_masked_lanes() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let source = [1.0f64, 2.0, 3.0, 4.0];
        for count in 0..=4 {
            unsafe {
                let mask = AvxMaskD::real(count);
                let value = AvxStoreD::load_partial(&source, mask);
                let mut loaded = [-1.0; 4];
                value.write_real(&mut loaded);
                assert_eq!(&loaded[..count], &source[..count]);
                assert_eq!(&loaded[count..], &vec![0.0; 4 - count]);
            }
        }

        let source = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        for count in 0..=2 {
            unsafe {
                let mask = AvxMaskD::complex(count);
                let value = AvxStoreD::from_complex_partial(&source, mask);
                let mut loaded = [Complex::new(-1.0, -1.0); 2];
                value.write(&mut loaded);
                assert_eq!(&loaded[..count], &source[..count]);
                assert_eq!(&loaded[count..], &vec![Complex::new(0.0, 0.0); 2 - count]);

                let sentinel = Complex::new(-1.0, -1.0);
                let mut stored = [sentinel; 2];
                value.write_partial(&mut stored, mask);
                assert_eq!(&stored[..count], &source[..count]);
                assert_eq!(&stored[count..], &vec![sentinel; 2 - count]);
            }
        }
    }
}
