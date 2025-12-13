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
use crate::avx::util::{_mm256_create_ps, _mm256_fcmul_ps};
use num_complex::Complex;
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

#[derive(Copy, Clone)]
pub(crate) struct AvxStoreF {
    pub(crate) v: __m256,
}

#[derive(Copy, Clone)]
pub(crate) struct SseStoreF {
    pub(crate) v: __m128,
}

impl SseStoreF {
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn from_complex_ref(complex: &[Complex<f32>]) -> Self {
        unsafe {
            SseStoreF {
                v: _mm_loadu_ps(complex.as_ptr().cast()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn zero() -> Self {
        Self {
            v: _mm_setzero_ps(),
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn raw(r: __m128) -> Self {
        Self { v: r }
    }
}

impl AvxStoreF {
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn neg(&self) -> AvxStoreF {
        AvxStoreF {
            v: _mm256_xor_ps(self.v, _mm256_set1_ps(-0.0)),
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn raw(r: __m256) -> AvxStoreF {
        AvxStoreF { v: r }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn from_complex_ref(complex: &[Complex<f32>]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_loadu_ps(complex.as_ptr().cast()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn from_complex_refu(complex: &[MaybeUninit<Complex<f32>>]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_loadu_ps(complex.as_ptr().cast()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn from_complex(complex: &Complex<f32>) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_castps128_ps256(_mm_castsi128_ps(_mm_loadu_si64(
                    complex as *const Complex<f32> as *const u8,
                ))),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn from_complexu(complex: &MaybeUninit<Complex<f32>>) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_castps128_ps256(_mm_castsi128_ps(_mm_loadu_si64(
                    complex as *const MaybeUninit<Complex<f32>> as *const u8,
                ))),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn set_complex4(
        v0: Complex<f32>,
        v1: Complex<f32>,
        v2: Complex<f32>,
        v3: Complex<f32>,
    ) -> Self {
        AvxStoreF {
            v: _mm256_setr_ps(v0.re, v0.im, v1.re, v1.im, v2.re, v2.im, v3.re, v3.im),
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn set_complex(v0: Complex<f32>) -> Self {
        AvxStoreF {
            v: _mm256_setr_ps(v0.re, v0.im, v0.re, v0.im, v0.re, v0.im, v0.re, v0.im),
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn from_complex2(complex: &[Complex<f32>]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_castps128_ps256(_mm_loadu_ps(complex.as_ptr().cast())),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn from_complex2u(complex: &[MaybeUninit<Complex<f32>>]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_castps128_ps256(_mm_loadu_ps(complex.as_ptr().cast())),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn from_complex3(complex: &[Complex<f32>]) -> Self {
        unsafe {
            let lo = _mm256_castps128_ps256(_mm_loadu_ps(complex.as_ptr().cast()));
            let hi = _mm_castsi128_ps(_mm_loadu_si64(complex.get_unchecked(2..).as_ptr().cast()));
            AvxStoreF {
                v: _mm256_insertf128_ps::<1>(lo, hi),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn from_complex3u(complex: &[MaybeUninit<Complex<f32>>]) -> Self {
        unsafe {
            let lo = _mm256_castps128_ps256(_mm_loadu_ps(complex.as_ptr().cast()));
            let hi = _mm_castsi128_ps(_mm_loadu_si64(complex.get_unchecked(2..).as_ptr().cast()));
            AvxStoreF {
                v: _mm256_insertf128_ps::<1>(lo, hi),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { _mm256_storeu_ps(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write_u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe { _mm256_storeu_ps(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write_lo1(&self, to_ref: &mut [Complex<f32>]) {
        unsafe {
            _mm_storeu_si64(
                to_ref.as_mut_ptr().cast(),
                _mm_castps_si128(_mm256_castps256_ps128(self.v)),
            )
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write_lo1u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe {
            _mm_storeu_si64(
                to_ref.as_mut_ptr().cast(),
                _mm_castps_si128(_mm256_castps256_ps128(self.v)),
            )
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write2lo(&self, other: Self, to_ref: &mut [Complex<f32>]) {
        unsafe {
            _mm256_storeu_ps(
                to_ref.as_mut_ptr().cast(),
                _mm256_create_ps(
                    _mm256_castps256_ps128(self.v),
                    _mm256_castps256_ps128(other.v),
                ),
            )
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write_lo2(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { _mm_storeu_ps(to_ref.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write_lo2u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe { _mm_storeu_ps(to_ref.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write_lo3(&self, to_ref: &mut [Complex<f32>]) {
        unsafe {
            _mm_storeu_ps(to_ref.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v));
            _mm_storeu_si64(
                to_ref.get_unchecked_mut(2..).as_mut_ptr().cast(),
                _mm_castps_si128(_mm256_extractf128_ps::<1>(self.v)),
            );
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn write_lo3u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe {
            _mm_storeu_ps(to_ref.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v));
            _mm_storeu_si64(
                to_ref.get_unchecked_mut(2..).as_mut_ptr().cast(),
                _mm_castps_si128(_mm256_extractf128_ps::<1>(self.v)),
            );
        }
    }

    #[inline]
    #[target_feature(enable = "avx", enable = "fma")]
    pub(crate) fn mul_by_complex(self, other: AvxStoreF) -> Self {
        AvxStoreF {
            v: _mm256_fcmul_ps(self.v, other.v),
        }
    }
}

impl AvxStoreF {
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn zero() -> Self {
        Self {
            v: _mm256_setzero_ps(),
        }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn lo(&self) -> Self {
        Self { v: self.v }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn hi(&self) -> Self {
        Self {
            v: _mm256_castps128_ps256(_mm256_extractf128_ps::<1>(self.v)),
        }
    }
}
