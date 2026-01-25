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
use num_traits::MulAdd;
use std::arch::x86_64::*;
use std::mem::MaybeUninit;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct AvxStoreF {
    pub(crate) v: __m256,
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct SseStoreF {
    pub(crate) v: __m128,
}

impl SseStoreF {
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex_ref(complex: &[Complex<f32>]) -> Self {
        unsafe {
            SseStoreF {
                v: _mm_loadu_ps(complex.as_ptr().cast()),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn zero() -> Self {
        Self {
            v: _mm_setzero_ps(),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn raw(r: __m128) -> Self {
        Self { v: r }
    }
}

impl AvxStoreF {
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn neg(&self) -> AvxStoreF {
        AvxStoreF {
            v: _mm256_xor_ps(self.v, _mm256_set1_ps(-0.0)),
        }
    }

    #[inline(always)]
    pub(crate) fn raw(r: __m256) -> AvxStoreF {
        AvxStoreF { v: r }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn raw128(r: __m128) -> AvxStoreF {
        AvxStoreF {
            v: _mm256_castps128_ps256(r),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex_ref(complex: &[Complex<f32>]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_loadu_ps(complex.as_ptr().cast()),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load(complex: &[f32]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_loadu_ps(complex.as_ptr().cast()),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load2_as_complex(complex: &[f32]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_castps128_ps256(_mm_unpacklo_ps(
                    _mm_castsi128_ps(_mm_loadu_si64(complex.as_ptr().cast())),
                    _mm_setzero_ps(),
                )),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex_refu(complex: &[MaybeUninit<Complex<f32>>]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_loadu_ps(complex.as_ptr().cast()),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex(complex: &Complex<f32>) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_castps128_ps256(_mm_castsi128_ps(_mm_loadu_si64(
                    complex as *const Complex<f32> as *const u8,
                ))),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complexu(complex: &MaybeUninit<Complex<f32>>) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_castps128_ps256(_mm_castsi128_ps(_mm_loadu_si64(
                    complex as *const MaybeUninit<Complex<f32>> as *const u8,
                ))),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
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

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn set_values8(
        p0: f32,
        p1: f32,
        p2: f32,
        p3: f32,
        p4: f32,
        p5: f32,
        p6: f32,
        p7: f32,
    ) -> Self {
        AvxStoreF::raw(_mm256_setr_ps(p0, p1, p2, p3, p4, p5, p6, p7))
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn dup(p0: f32) -> Self {
        AvxStoreF::raw(_mm256_set1_ps(p0))
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn xor(&self, p0: AvxStoreF) -> Self {
        AvxStoreF::raw(_mm256_xor_ps(self.v, p0.v))
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn dup_even_odds(&self) -> [Self; 2] {
        [
            AvxStoreF::raw(_mm256_moveldup_ps(self.v)),
            AvxStoreF::raw(_mm256_movehdup_ps(self.v)),
        ]
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn blend_real_img(&self, p0: AvxStoreF) -> Self {
        AvxStoreF::raw(_mm256_blend_ps::<0xAA>(self.v, p0.v))
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse_complex(&self) -> Self {
        let permuted = _mm256_shuffle_ps::<0x4E>(self.v, self.v);
        AvxStoreF::raw(_mm256_permute2f128_ps(permuted, permuted, 0x01))
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse_complex_elements(&self) -> Self {
        AvxStoreF::raw(_mm256_shuffle_ps::<0xB1>(self.v, self.v))
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn set_complex(v0: Complex<f32>) -> Self {
        AvxStoreF {
            v: _mm256_setr_ps(v0.re, v0.im, v0.re, v0.im, v0.re, v0.im, v0.re, v0.im),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load7(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = _mm_loadu_ps(ptr.as_ptr().cast());
            let q2 = _mm_castsi128_ps(_mm_loadu_si64(ptr.get_unchecked(4..).as_ptr().cast()));
            let q3 = _mm_load_ss(ptr.get_unchecked(6..).as_ptr().cast());
            let q4 = _mm_insert_ps::<0x20>(q2, q3);
            AvxStoreF::raw(_mm256_setr_m128(q0, q4))
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load6(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = _mm_loadu_ps(ptr.as_ptr().cast());
            let q1 = _mm_castsi128_ps(_mm_loadu_si64(ptr.get_unchecked(4..).as_ptr().cast()));
            AvxStoreF::raw(_mm256_setr_m128(q0, q1))
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load5(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = _mm_loadu_ps(ptr.as_ptr().cast());
            let q1 = _mm_load_ss(ptr.get_unchecked(4..).as_ptr().cast());
            AvxStoreF::raw(_mm256_setr_m128(q0, q1))
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load4(ptr: &[f32]) -> Self {
        unsafe { AvxStoreF::raw(_mm256_castps128_ps256(_mm_loadu_ps(ptr.as_ptr().cast()))) }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load3(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = _mm_castsi128_ps(_mm_loadu_si64(ptr.as_ptr().cast()));
            let q1 = _mm_load_ss(ptr.get_unchecked(2..).as_ptr().cast());
            let q2 = _mm_insert_ps::<0x20>(q0, q1);
            AvxStoreF::raw(_mm256_castps128_ps256(q2))
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load2(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = _mm_castsi128_ps(_mm_loadu_si64(ptr.as_ptr().cast()));
            AvxStoreF::raw(_mm256_castps128_ps256(q0))
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load1(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = _mm_load_ss(ptr.as_ptr().cast());
            AvxStoreF::raw(_mm256_castps128_ps256(q0))
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex2(complex: &[Complex<f32>]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_castps128_ps256(_mm_loadu_ps(complex.as_ptr().cast())),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex2u(complex: &[MaybeUninit<Complex<f32>>]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_castps128_ps256(_mm_loadu_ps(complex.as_ptr().cast())),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex3(complex: &[Complex<f32>]) -> Self {
        unsafe {
            let lo = _mm256_castps128_ps256(_mm_loadu_ps(complex.as_ptr().cast()));
            let hi = _mm_castsi128_ps(_mm_loadu_si64(complex.get_unchecked(2..).as_ptr().cast()));
            AvxStoreF {
                v: _mm256_insertf128_ps::<1>(lo, hi),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex3u(complex: &[MaybeUninit<Complex<f32>>]) -> Self {
        unsafe {
            let lo = _mm256_castps128_ps256(_mm_loadu_ps(complex.as_ptr().cast()));
            let hi = _mm_castsi128_ps(_mm_loadu_si64(complex.get_unchecked(2..).as_ptr().cast()));
            AvxStoreF {
                v: _mm256_insertf128_ps::<1>(lo, hi),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { _mm256_storeu_ps(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_single(&self, to_ref: &mut Complex<f32>) {
        unsafe {
            _mm_storeu_si64(
                to_ref as *mut Complex<f32> as *mut f32 as *mut u8,
                _mm_castps_si128(_mm256_castps256_ps128(self.v)),
            )
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe { _mm256_storeu_ps(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_lo1(&self, to_ref: &mut [Complex<f32>]) {
        unsafe {
            _mm_storeu_si64(
                to_ref.as_mut_ptr().cast(),
                _mm_castps_si128(_mm256_castps256_ps128(self.v)),
            )
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_lo1u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe {
            _mm_storeu_si64(
                to_ref.as_mut_ptr().cast(),
                _mm_castps_si128(_mm256_castps256_ps128(self.v)),
            )
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
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

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_lo2(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { _mm_storeu_ps(to_ref.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v)) }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_lo2u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe { _mm_storeu_ps(to_ref.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v)) }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_lo3(&self, to_ref: &mut [Complex<f32>]) {
        unsafe {
            _mm_storeu_ps(to_ref.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v));
            _mm_storeu_si64(
                to_ref.get_unchecked_mut(2..).as_mut_ptr().cast(),
                _mm_castps_si128(_mm256_extractf128_ps::<1>(self.v)),
            );
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_lo3u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe {
            _mm_storeu_ps(to_ref.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v));
            _mm_storeu_si64(
                to_ref.get_unchecked_mut(2..).as_mut_ptr().cast(),
                _mm_castps_si128(_mm256_extractf128_ps::<1>(self.v)),
            );
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn mul_by_complex(self, other: AvxStoreF) -> Self {
        AvxStoreF {
            v: _mm256_fcmul_ps(self.v, other.v),
        }
    }
}

impl AvxStoreF {
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn zero() -> Self {
        Self {
            v: _mm256_setzero_ps(),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn to_complex(self) -> [Self; 2] {
        self.zip(AvxStoreF::zero())
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn zip(self, other: Self) -> [Self; 2] {
        let r0 = _mm256_unpacklo_ps(self.v, other.v);
        let r1 = _mm256_unpackhi_ps(self.v, other.v);
        let xy0 = _mm256_permute2f128_ps::<32>(r0, r1);
        let xy1 = _mm256_permute2f128_ps::<49>(r0, r1);
        [AvxStoreF::raw(xy0), AvxStoreF::raw(xy1)]
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn lo(&self) -> Self {
        Self { v: self.v }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "avx2")]
    pub(crate) fn hi(&self) -> Self {
        Self {
            v: _mm256_castps128_ps256(_mm256_extractf128_ps::<1>(self.v)),
        }
    }
}

impl MulAdd<AvxStoreF> for AvxStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: AvxStoreF, b: Self) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_fmadd_ps(self.v, a.v, b.v)) }
    }
}

impl Mul<f32> for AvxStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_mul_ps(self.v, _mm256_set1_ps(rhs))) }
    }
}

impl Mul<AvxStoreF> for f32 {
    type Output = AvxStoreF;

    #[inline(always)]
    fn mul(self, rhs: AvxStoreF) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_mul_ps(rhs.v, _mm256_set1_ps(self))) }
    }
}

impl Mul<AvxStoreF> for AvxStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: AvxStoreF) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_mul_ps(self.v, rhs.v)) }
    }
}

impl Add<AvxStoreF> for AvxStoreF {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: AvxStoreF) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_add_ps(self.v, rhs.v)) }
    }
}

impl Sub<AvxStoreF> for AvxStoreF {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: AvxStoreF) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_sub_ps(self.v, rhs.v)) }
    }
}

impl Neg for AvxStoreF {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_xor_ps(self.v, _mm256_set1_ps(-0.0))) }
    }
}
