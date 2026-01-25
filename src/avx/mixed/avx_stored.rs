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
use crate::avx::util::_mm256_fcmul_pd;
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

impl AvxStoreD {
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn dup(p0: f64) -> AvxStoreD {
        AvxStoreD::raw(_mm256_set1_pd(p0))
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn dup_even_odds(&self) -> [Self; 2] {
        [
            AvxStoreD::raw(_mm256_movedup_pd(self.v)),
            AvxStoreD::raw(_mm256_shuffle_pd::<0x0F>(self.v, self.v)),
        ]
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn reverse_complex(&self) -> Self {
        AvxStoreD::raw(_mm256_permute2f128_pd::<0x01>(self.v, self.v))
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn reverse_complex_elements(&self) -> Self {
        AvxStoreD::raw(_mm256_shuffle_pd::<0x05>(self.v, self.v))
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn xor(&self, other: Self) -> Self {
        AvxStoreD::raw(_mm256_xor_pd(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn set_values(p0: f64, p1: f64, p2: f64, p3: f64) -> Self {
        AvxStoreD::raw(_mm256_setr_pd(p0, p1, p2, p3))
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn blend_real_img(&self, other: Self) -> Self {
        AvxStoreD::raw(_mm256_blend_pd::<0b1010>(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn neg(&self) -> AvxStoreD {
        AvxStoreD::raw(_mm256_xor_pd(self.v, _mm256_set1_pd(-0.0)))
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn raw(r: __m256d) -> AvxStoreD {
        AvxStoreD { v: r }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn from_complex_ref(complex: &[Complex<f64>]) -> Self {
        unsafe {
            AvxStoreD {
                v: _mm256_loadu_pd(complex.as_ptr().cast()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn load(complex: &[f64]) -> Self {
        unsafe {
            AvxStoreD {
                v: _mm256_loadu_pd(complex.as_ptr().cast()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn from_complex_refu(complex: &[MaybeUninit<Complex<f64>>]) -> Self {
        unsafe {
            AvxStoreD {
                v: _mm256_loadu_pd(complex.as_ptr().cast()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
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
    pub(crate) fn load3(ptr: &[f64]) -> Self {
        unsafe {
            let q0 = _mm_loadu_pd(ptr.as_ptr().cast());
            let q1 = _mm_load_sd(ptr.get_unchecked(2..).as_ptr().cast());
            AvxStoreD::raw(_mm256_setr_m128d(q0, q1))
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
    #[target_feature(enable = "avx")]
    pub(crate) fn set_complex(complex: &Complex<f64>) -> Self {
        AvxStoreD {
            v: _mm256_setr_pd(complex.re, complex.im, complex.re, complex.im),
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
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
    #[target_feature(enable = "avx")]
    pub(crate) fn set_complex2(p0: Complex<f64>, p1: Complex<f64>) -> AvxStoreD {
        AvxStoreD::raw(_mm256_setr_pd(p0.re, p0.im, p1.re, p1.im))
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f64>]) {
        unsafe { _mm256_storeu_pd(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write_single(&self, to_ref: &mut Complex<f64>) {
        unsafe {
            _mm_storeu_pd(
                to_ref as *mut Complex<f64> as *mut f64,
                _mm256_castpd256_pd128(self.v),
            )
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write_u(&self, to_ref: &mut [MaybeUninit<Complex<f64>>]) {
        unsafe { _mm256_storeu_pd(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write_lo(&self, to_ref: &mut [Complex<f64>]) {
        unsafe { _mm_storeu_pd(to_ref.as_mut_ptr().cast(), _mm256_castpd256_pd128(self.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write_hi(&self, to_ref: &mut [Complex<f64>]) {
        unsafe {
            _mm_storeu_pd(
                to_ref.as_mut_ptr().cast(),
                _mm256_extractf128_pd::<1>(self.v),
            )
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write_lou(&self, to_ref: &mut [MaybeUninit<Complex<f64>>]) {
        unsafe { _mm_storeu_pd(to_ref.as_mut_ptr().cast(), _mm256_castpd256_pd128(self.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx", enable = "fma")]
    pub(crate) fn mul_by_complex(self, other: AvxStoreD) -> Self {
        AvxStoreD {
            v: _mm256_fcmul_pd(self.v, other.v),
        }
    }
}

impl AvxStoreD {
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn zero() -> Self {
        Self {
            v: _mm256_setzero_pd(),
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
