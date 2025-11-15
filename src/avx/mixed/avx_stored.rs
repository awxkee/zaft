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
use std::arch::x86_64::*;

#[derive(Copy, Clone)]
pub(crate) struct AvxStoreD {
    pub(crate) v: __m256d,
}

impl AvxStoreD {
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
    pub(crate) fn write_lo(&self, to_ref: &mut [Complex<f64>]) {
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
}
