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
use crate::avx::util::{
    _mm_unpacklo_ps64, _mm256_create_ps, _mm256_fcmul_ps, _mm256_fcmul_ps_conj_a, shuffle,
};
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
pub(crate) struct AvxMaskF {
    v: __m256i,
    lanes: usize,
}

impl AvxMaskF {
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn real(count: usize) -> Self {
        debug_assert!(count <= 8);
        Self {
            v: _mm256_cmpgt_epi32(
                _mm256_set1_epi32(count as i32),
                _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
            ),
            lanes: count,
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn complex(count: usize) -> Self {
        debug_assert!(count <= 4);
        Self::real(count * 2)
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct SseStoreF {
    pub(crate) v: __m128,
}

impl SseStoreF {
    #[inline(always)]
    pub(crate) fn from_complex_ref(complex: &[Complex<f32>]) -> Self {
        unsafe {
            SseStoreF {
                v: _mm_loadu_ps(complex.as_ptr().cast()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn zero() -> Self {
        unsafe {
            Self {
                v: _mm_setzero_ps(),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn undefined() -> Self {
        unsafe {
            Self {
                v: _mm_undefined_ps(),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn raw(r: __m128) -> Self {
        Self { v: r }
    }

    #[inline(always)]
    pub(crate) fn write_real(self, dst: &mut [f32]) {
        unsafe { _mm_storeu_ps(dst.as_mut_ptr().cast(), self.v) }
    }

    #[inline(always)]
    pub(crate) fn write_real_lo1(self, dst: &mut [f32]) {
        unsafe { _mm_store_ss(dst.as_mut_ptr().cast(), self.v) }
    }

    #[inline(always)]
    pub(crate) fn write_real_lo2(self, dst: &mut [f32]) {
        unsafe { _mm_storel_pd(dst.as_mut_ptr().cast(), _mm_castps_pd(self.v)) }
    }

    #[inline(always)]
    pub(crate) fn write_real_lo3(self, dst: &mut [f32]) {
        unsafe { _mm_storel_pd(dst.as_mut_ptr().cast(), _mm_castps_pd(self.v)) }
        unsafe {
            _mm_store_ss(
                dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(self.v, self.v),
            )
        }
    }
}

impl AvxStoreF {
    #[inline(always)]
    pub(crate) fn neg(&self) -> AvxStoreF {
        unsafe {
            AvxStoreF {
                v: _mm256_xor_ps(self.v, _mm256_set1_ps(-0.0)),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn raw(r: __m256) -> AvxStoreF {
        AvxStoreF { v: r }
    }

    // #[inline(always)]
    // pub(crate) fn raw128(r: __m128) -> AvxStoreF {
    //     unsafe {
    //         AvxStoreF {
    //             v: _mm256_castps128_ps256(r),
    //         }
    //     }
    // }

    #[inline(always)]
    pub(crate) fn from_complex_ref(complex: &[Complex<f32>]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_loadu_ps(complex.as_ptr().cast()),
            }
        }
    }

    /// Loads `count` complex values and zeroes the remaining complex lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_complex_partial(complex: &[Complex<f32>], mask: AvxMaskF) -> Self {
        debug_assert!(mask.lanes.is_multiple_of(2) && complex.len() >= mask.lanes / 2);
        if mask.lanes == 8 {
            return Self::from_complex_ref(complex);
        }
        unsafe {
            AvxStoreF {
                v: _mm256_maskload_ps(complex.as_ptr().cast(), mask.v),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn pack_evens(self) -> SseStoreF {
        unsafe {
            SseStoreF {
                v: _mm256_castps256_ps128(_mm256_permutevar8x32_ps(
                    self.v,
                    _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0),
                )),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn load(complex: &[f32]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_loadu_ps(complex.as_ptr().cast()),
            }
        }
    }

    /// Loads `count` f32 lanes and zeroes the remaining lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load_partial(complex: &[f32], mask: AvxMaskF) -> Self {
        debug_assert!(complex.len() >= mask.lanes);
        if mask.lanes == 8 {
            return Self::load(complex);
        }
        unsafe {
            AvxStoreF {
                v: _mm256_maskload_ps(complex.as_ptr(), mask.v),
            }
        }
    }

    #[inline(always)]
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

    #[inline(always)]
    pub(crate) fn from_complex_refu(complex: &[MaybeUninit<Complex<f32>>]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_loadu_ps(complex.as_ptr().cast()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn from_complex(complex: &Complex<f32>) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_castps128_ps256(_mm_castsi128_ps(_mm_loadu_si64(
                    complex as *const Complex<f32> as *const u8,
                ))),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn set_complex_lane<const LANE: i32>(self, complex: Complex<f32>) -> Self {
        unsafe {
            let re = complex.re.to_ne_bytes();
            let im = complex.im.to_ne_bytes();
            Self::raw(_mm256_castsi256_ps(_mm256_insert_epi64::<LANE>(
                _mm256_castps_si256(self.v),
                i64::from_ne_bytes([re[0], re[1], re[2], re[3], im[0], im[1], im[2], im[3]]),
            )))
        }
    }

    #[inline]
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

    #[inline]
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

    #[inline]
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

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn conj_flag() -> Self {
        AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn dup(p0: f32) -> Self {
        AvxStoreF::raw(_mm256_set1_ps(p0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn xor(&self, p0: AvxStoreF) -> Self {
        AvxStoreF::raw(_mm256_xor_ps(self.v, p0.v))
    }

    #[inline(always)]
    pub(crate) fn dup_even_odds(&self) -> [Self; 2] {
        unsafe {
            [
                AvxStoreF::raw(_mm256_moveldup_ps(self.v)),
                AvxStoreF::raw(_mm256_movehdup_ps(self.v)),
            ]
        }
    }

    #[inline(always)]
    pub(crate) fn blend_real_img(&self, p0: AvxStoreF) -> Self {
        unsafe { AvxStoreF::raw(_mm256_blend_ps::<0xAA>(self.v, p0.v)) }
    }

    #[inline(always)]
    pub(crate) fn reverse_complex(&self) -> Self {
        unsafe {
            let permuted = _mm256_shuffle_ps::<0x4E>(self.v, self.v);
            AvxStoreF::raw(_mm256_permute2f128_ps::<0x01>(permuted, permuted))
        }
    }

    #[inline(always)]
    pub(crate) fn reverse_complex_elements(&self) -> Self {
        unsafe { AvxStoreF::raw(_mm256_shuffle_ps::<0xB1>(self.v, self.v)) }
    }

    #[inline(always)]
    pub(crate) fn set_complex(v0: Complex<f32>) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_setr_ps(v0.re, v0.im, v0.re, v0.im, v0.re, v0.im, v0.re, v0.im),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn load4(ptr: &[f32]) -> Self {
        unsafe { AvxStoreF::raw(_mm256_castps128_ps256(_mm_loadu_ps(ptr.as_ptr().cast()))) }
    }

    #[inline(always)]
    pub(crate) fn load1_ref(ptr: &f32) -> Self {
        unsafe {
            let q0 = _mm_unpacklo_ps(_mm_load_ss(ptr), _mm_setzero_ps());
            AvxStoreF::raw(_mm256_castps128_ps256(q0))
        }
    }

    #[inline(always)]
    pub(crate) fn from_complex2(complex: &[Complex<f32>]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_castps128_ps256(_mm_loadu_ps(complex.as_ptr().cast())),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn from_complex2u(complex: &[MaybeUninit<Complex<f32>>]) -> Self {
        unsafe {
            AvxStoreF {
                v: _mm256_castps128_ps256(_mm_loadu_ps(complex.as_ptr().cast())),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn from_complex3(complex: &[Complex<f32>]) -> Self {
        unsafe {
            let lo = _mm256_castps128_ps256(_mm_loadu_ps(complex.as_ptr().cast()));
            let hi = _mm_castsi128_ps(_mm_loadu_si64(complex.get_unchecked(2..).as_ptr().cast()));
            AvxStoreF {
                v: _mm256_insertf128_ps::<1>(lo, hi),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn from_complex3u(complex: &[MaybeUninit<Complex<f32>>]) -> Self {
        unsafe {
            let lo = _mm256_castps128_ps256(_mm_loadu_ps(complex.as_ptr().cast()));
            let hi = _mm_castsi128_ps(_mm_loadu_si64(complex.get_unchecked(2..).as_ptr().cast()));
            AvxStoreF {
                v: _mm256_insertf128_ps::<1>(lo, hi),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { _mm256_storeu_ps(to_ref.as_mut_ptr().cast(), self.v) }
    }

    /// Stores the first `count` complex values without touching following values.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_partial(&self, to_ref: &mut [Complex<f32>], mask: AvxMaskF) {
        debug_assert!(mask.lanes.is_multiple_of(2) && to_ref.len() >= mask.lanes / 2);
        if mask.lanes == 8 {
            return self.write(to_ref);
        }
        unsafe { _mm256_maskstore_ps(to_ref.as_mut_ptr().cast(), mask.v, self.v) }
    }

    #[inline(always)]
    pub(crate) fn write_single(&self, to_ref: &mut Complex<f32>) {
        unsafe {
            _mm_storeu_si64(
                to_ref as *mut Complex<f32> as *mut f32 as *mut u8,
                _mm_castps_si128(_mm256_castps256_ps128(self.v)),
            )
        }
    }

    #[inline(always)]
    pub(crate) fn write_u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe { _mm256_storeu_ps(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline(always)]
    pub(crate) fn write_lo1(&self, to_ref: &mut [Complex<f32>]) {
        unsafe {
            _mm_storeu_si64(
                to_ref.as_mut_ptr().cast(),
                _mm_castps_si128(_mm256_castps256_ps128(self.v)),
            )
        }
    }

    #[inline(always)]
    pub(crate) fn write_lo1u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe {
            _mm_storeu_si64(
                to_ref.as_mut_ptr().cast(),
                _mm_castps_si128(_mm256_castps256_ps128(self.v)),
            )
        }
    }

    #[inline(always)]
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

    #[inline(always)]
    pub(crate) fn write_lo2(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { _mm_storeu_ps(to_ref.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v)) }
    }

    #[inline(always)]
    pub(crate) fn write_lo2u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe { _mm_storeu_ps(to_ref.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v)) }
    }

    #[inline(always)]
    pub(crate) fn write_lo3(&self, to_ref: &mut [Complex<f32>]) {
        unsafe {
            _mm_storeu_ps(to_ref.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v));
            _mm_storeu_si64(
                to_ref.get_unchecked_mut(2..).as_mut_ptr().cast(),
                _mm_castps_si128(_mm256_extractf128_ps::<1>(self.v)),
            );
        }
    }

    #[inline(always)]
    pub(crate) fn write_lo3u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe {
            _mm_storeu_ps(to_ref.as_mut_ptr().cast(), _mm256_castps256_ps128(self.v));
            _mm_storeu_si64(
                to_ref.get_unchecked_mut(2..).as_mut_ptr().cast(),
                _mm_castps_si128(_mm256_extractf128_ps::<1>(self.v)),
            );
        }
    }

    #[inline(always)]
    pub(crate) fn mul_by_complex(self, other: AvxStoreF) -> Self {
        AvxStoreF {
            v: _mm256_fcmul_ps(self.v, other.v),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn mul_by_conj_a(self, other: AvxStoreF) -> Self {
        AvxStoreF {
            v: _mm256_fcmul_ps_conj_a(self.v, other.v),
        }
    }

    #[inline(always)]
    pub(crate) fn dup_lo_complex(&self) -> Self {
        unsafe {
            Self::raw(_mm256_castps128_ps256(_mm_unpacklo_ps64(
                _mm256_castps256_ps128(self.v),
                _mm256_castps256_ps128(self.v),
            )))
        }
    }

    #[inline(always)]
    pub(crate) fn combine_lo_hi(&self, other: Self) -> Self {
        unsafe {
            Self::raw(_mm256_setr_m128(
                _mm256_castps256_ps128(self.v),
                _mm256_castps256_ps128(other.v),
            ))
        }
    }

    #[inline(always)]
    pub(crate) fn reverse_complex3(&self) -> Self {
        unsafe {
            AvxStoreF::raw(_mm256_castpd_ps(_mm256_permute4x64_pd::<
                { shuffle(0, 0, 1, 2) },
            >(_mm256_castps_pd(self.v))))
        }
    }

    #[inline(always)]
    pub(crate) fn reverse_complex2(&self) -> Self {
        unsafe {
            AvxStoreF::raw(_mm256_castpd_ps(_mm256_permute4x64_pd::<
                { shuffle(0, 0, 0, 1) },
            >(_mm256_castps_pd(self.v))))
        }
    }
}

impl AvxStoreF {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        unsafe {
            Self {
                v: _mm256_setzero_ps(),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn undefined() -> Self {
        unsafe {
            Self {
                v: _mm256_undefined_ps(),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn to_complex(self) -> [Self; 2] {
        self.zip(AvxStoreF::zero())
    }

    #[inline(always)]
    pub(crate) fn zip(self, other: Self) -> [Self; 2] {
        unsafe {
            let r0 = _mm256_unpacklo_ps(self.v, other.v);
            let r1 = _mm256_unpackhi_ps(self.v, other.v);
            let xy0 = _mm256_permute2f128_ps::<32>(r0, r1);
            let xy1 = _mm256_permute2f128_ps::<49>(r0, r1);
            [AvxStoreF::raw(xy0), AvxStoreF::raw(xy1)]
        }
    }

    #[inline(always)]
    pub(crate) fn lo(&self) -> Self {
        Self { v: self.v }
    }

    #[inline(always)]
    pub(crate) fn hi(&self) -> Self {
        unsafe {
            Self {
                v: _mm256_castps128_ps256(_mm256_extractf128_ps::<1>(self.v)),
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_loads_zero_and_partial_stores_preserve_masked_lanes() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let source = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for count in 0..=8 {
            unsafe {
                let mask = AvxMaskF::real(count);
                let value = AvxStoreF::load_partial(&source, mask);
                let mut loaded = [-1.0; 8];
                _mm256_storeu_ps(loaded.as_mut_ptr(), value.v);
                assert_eq!(&loaded[..count], &source[..count]);
                assert_eq!(&loaded[count..], &vec![0.0; 8 - count]);
            }
        }

        let source = [
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        for count in 0..=4 {
            unsafe {
                let mask = AvxMaskF::complex(count);
                let value = AvxStoreF::from_complex_partial(&source, mask);
                let mut loaded = [Complex::new(-1.0, -1.0); 4];
                value.write(&mut loaded);
                assert_eq!(&loaded[..count], &source[..count]);
                assert_eq!(&loaded[count..], &vec![Complex::new(0.0, 0.0); 4 - count]);

                let sentinel = Complex::new(-1.0, -1.0);
                let mut stored = [sentinel; 4];
                value.write_partial(&mut stored, mask);
                assert_eq!(&stored[..count], &source[..count]);
                assert_eq!(&stored[count..], &vec![sentinel; 4 - count]);
            }
        }
    }
}
