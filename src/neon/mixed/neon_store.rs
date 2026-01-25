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
use crate::neon::util::{vfcmul_f32, vfcmulq_f32, vfcmulq_f64};
use num_complex::Complex;
use num_traits::MulAdd;
use std::arch::aarch64::*;
use std::mem::MaybeUninit;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy)]
pub(crate) struct NeonStoreD {
    pub(crate) v: float64x2_t,
}

#[derive(Clone, Copy)]
pub(crate) struct NeonStoreF {
    pub(crate) v: float32x4_t,
}

#[derive(Clone, Copy)]
pub(crate) struct NeonStoreFh {
    pub(crate) v: float32x2_t,
}

impl NeonStoreD {
    #[inline]
    pub(crate) fn raw(r: float64x2_t) -> NeonStoreD {
        NeonStoreD { v: r }
    }

    #[inline(always)]
    pub(crate) fn set_values(p0: f64, p1: f64) -> NeonStoreD {
        unsafe {
            NeonStoreD {
                v: vld1q_f64([p0, p1].as_ptr()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn select(&self, other: NeonStoreD, mask: NeonStoreD) -> Self {
        unsafe {
            NeonStoreD {
                v: vbslq_f64(vreinterpretq_u64_f64(mask.v), self.v, other.v),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn dup(p0: f64) -> Self {
        unsafe { NeonStoreD::raw(vdupq_n_f64(p0)) }
    }

    #[inline(always)]
    pub(crate) fn from_complex_ref(complex: &[Complex<f64>]) -> Self {
        unsafe {
            NeonStoreD {
                v: vld1q_f64(complex.as_ptr().cast()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn dup_even_odds(&self) -> [Self; 2] {
        unsafe {
            [
                NeonStoreD {
                    v: vtrn1q_f64(self.v, self.v),
                },
                NeonStoreD {
                    v: vtrn2q_f64(self.v, self.v),
                },
            ]
        }
    }

    #[inline(always)]
    pub(crate) fn reverse_complex_elements(&self) -> NeonStoreD {
        unsafe { NeonStoreD::raw(vextq_f64::<1>(self.v, self.v)) }
    }

    #[inline(always)]
    pub(crate) fn xor(&self, other: Self) -> Self {
        unsafe {
            NeonStoreD {
                v: vreinterpretq_f64_u64(veorq_u64(
                    vreinterpretq_u64_f64(self.v),
                    vreinterpretq_u64_f64(other.v),
                )),
            }
        }
    }

    #[inline]
    pub(crate) fn load(vals: &[f64]) -> Self {
        unsafe {
            NeonStoreD {
                v: vld1q_f64(vals.as_ptr()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn load1(vals: &[f64]) -> Self {
        unsafe {
            NeonStoreD {
                v: vcombine_f64(vld1_f64(vals.as_ptr()), vdup_n_f64(0.)),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn load1_ptr(vals: *const f64) -> Self {
        unsafe {
            NeonStoreD {
                v: vcombine_f64(vld1_f64(vals), vdup_n_f64(0.)),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn from_complex_refu(complex: &[MaybeUninit<Complex<f64>>]) -> Self {
        unsafe {
            NeonStoreD {
                v: vld1q_f64(complex.as_ptr().cast()),
            }
        }
    }

    #[inline]
    pub(crate) fn to_complex(self) -> [Self; 2] {
        unsafe {
            let ql = vzip1q_f64(self.v, vdupq_n_f64(0.));
            let qh = vzip2q_f64(self.v, vdupq_n_f64(0.));
            [NeonStoreD { v: ql }, NeonStoreD { v: qh }]
        }
    }

    #[inline]
    pub(crate) fn from_complex(complex: &Complex<f64>) -> Self {
        unsafe {
            NeonStoreD {
                v: vld1q_f64(complex as *const Complex<f64> as *const f64),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f64>]) {
        unsafe { vst1q_f64(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline(always)]
    pub(crate) fn write_single(&self, to_ref: &mut Complex<f64>) {
        unsafe { vst1q_f64(to_ref as *mut Complex<f64> as *mut f64, self.v) }
    }

    #[inline(always)]
    pub(crate) fn write_uninit(&self, to_ref: &mut [MaybeUninit<Complex<f64>>]) {
        unsafe { vst1q_f64(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline(always)]
    pub(crate) fn mul_by_complex(self, other: NeonStoreD) -> Self {
        NeonStoreD {
            v: vfcmulq_f64(self.v, other.v),
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn fcmul_fcma(self, other: NeonStoreD) -> Self {
        NeonStoreD {
            v: vcmlaq_rot90_f64(
                vcmlaq_f64(vdupq_n_f64(0.), self.v, other.v),
                self.v,
                other.v,
            ),
        }
    }
}

impl NeonStoreF {
    #[inline]
    pub(crate) fn raw(r: float32x4_t) -> NeonStoreF {
        NeonStoreF { v: r }
    }

    #[inline]
    pub(crate) fn load(ptr: &[f32]) -> Self {
        unsafe {
            NeonStoreF {
                v: vld1q_f32(ptr.as_ptr().cast()),
            }
        }
    }

    #[inline]
    pub(crate) fn load1(ptr: &[f32]) -> Self {
        unsafe {
            NeonStoreF {
                v: vld1q_lane_f32::<0>(ptr.as_ptr().cast(), vdupq_n_f32(0.)),
            }
        }
    }

    #[inline]
    pub(crate) fn load2(ptr: &[f32]) -> Self {
        unsafe {
            NeonStoreF {
                v: vcombine_f32(vld1_f32(ptr.as_ptr().cast()), vdup_n_f32(0.)),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn load3(ptr: &[f32]) -> Self {
        unsafe {
            NeonStoreF {
                v: vcombine_f32(
                    vld1_f32(ptr.as_ptr().cast()),
                    vld1_lane_f32::<0>(ptr.get_unchecked(2..).as_ptr(), vdup_n_f32(0.)),
                ),
            }
        }
    }

    #[inline]
    pub(crate) fn from_complex_ref(complex: &[Complex<f32>]) -> Self {
        unsafe {
            NeonStoreF {
                v: vld1q_f32(complex.as_ptr().cast()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn to_complex(self) -> [Self; 2] {
        unsafe {
            let ql = vzip1q_f32(self.v, vdupq_n_f32(0.));
            let qh = vzip2q_f32(self.v, vdupq_n_f32(0.));
            [NeonStoreF { v: ql }, NeonStoreF { v: qh }]
        }
    }

    #[inline]
    pub(crate) fn to_lo(self) -> NeonStoreFh {
        unsafe { NeonStoreFh::raw(vget_low_f32(self.v)) }
    }

    #[inline]
    pub(crate) fn from_complex_refu(complex: &[MaybeUninit<Complex<f32>>]) -> Self {
        unsafe {
            NeonStoreF {
                v: vld1q_f32(complex.as_ptr().cast()),
            }
        }
    }

    #[inline]
    pub(crate) fn from_complex(complex: &Complex<f32>) -> Self {
        unsafe {
            NeonStoreF {
                v: vld1q_f32(
                    [complex.re, complex.im, complex.re, complex.im]
                        .as_ptr()
                        .cast(),
                ),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn dup_even_odds(&self) -> [Self; 2] {
        unsafe {
            [
                NeonStoreF {
                    v: vtrn1q_f32(self.v, self.v),
                },
                NeonStoreF {
                    v: vtrn2q_f32(self.v, self.v),
                },
            ]
        }
    }

    #[inline(always)]
    pub(crate) fn reverse_complex(&self) -> Self {
        unsafe {
            NeonStoreF {
                v: vcombine_f32(vget_high_f32(self.v), vget_low_f32(self.v)),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn xor(&self, other: Self) -> Self {
        unsafe {
            NeonStoreF {
                v: vreinterpretq_f32_u32(veorq_u32(
                    vreinterpretq_u32_f32(self.v),
                    vreinterpretq_u32_f32(other.v),
                )),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn reverse_complex_elements(&self) -> Self {
        unsafe {
            NeonStoreF {
                v: vrev64q_f32(self.v),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn select(&self, other: Self, mask: Self) -> Self {
        unsafe {
            NeonStoreF {
                v: vbslq_f32(vreinterpretq_u32_f32(mask.v), self.v, other.v),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn dup(p0: f32) -> NeonStoreF {
        unsafe { NeonStoreF::raw(vdupq_n_f32(p0)) }
    }

    #[inline(always)]
    pub(crate) fn neg(&self) -> NeonStoreF {
        unsafe { NeonStoreF::raw(vnegq_f32(self.v)) }
    }

    #[inline(always)]
    pub(crate) fn from_complexu(complex: &MaybeUninit<Complex<f32>>) -> Self {
        unsafe {
            let complex_ref: &Complex<f32> = complex.assume_init_ref();
            NeonStoreF {
                v: vld1q_f32(
                    [
                        complex_ref.re,
                        complex_ref.im,
                        complex_ref.re,
                        complex_ref.im,
                    ]
                    .as_ptr()
                    .cast(),
                ),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn lo(self) -> NeonStoreFh {
        unsafe { NeonStoreFh::raw(vget_low_f32(self.v)) }
    }

    #[inline]
    pub(crate) fn from_complex_lou(complex: &MaybeUninit<Complex<f32>>) -> Self {
        unsafe {
            NeonStoreF {
                v: vcombine_f32(vld1_f32(complex.as_ptr().cast()), vdup_n_f32(0.)),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn from_complex2(v0: Complex<f32>, v1: Complex<f32>) -> Self {
        unsafe {
            NeonStoreF {
                v: vld1q_f32([v0.re, v0.im, v1.re, v1.im].as_ptr().cast()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { vst1q_f32(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline(always)]
    pub(crate) fn write_single(&self, to_ref: &mut Complex<f32>) {
        unsafe {
            vst1_f32(
                to_ref as *mut Complex<f32> as *mut f32,
                vget_low_f32(self.v),
            )
        }
    }

    #[inline(always)]
    pub(crate) fn write_uninit(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe { vst1q_f32(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline(always)]
    pub(crate) fn write_lo(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { vst1_f32(to_ref.as_mut_ptr().cast(), vget_low_f32(self.v)) }
    }

    #[inline]
    pub(crate) fn write_hi(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { vst1q_lane_f64::<1>(to_ref.as_mut_ptr().cast(), vreinterpretq_f64_f32(self.v)) }
    }

    #[inline(always)]
    pub(crate) fn write_lo_u(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe { vst1_f32(to_ref.as_mut_ptr().cast(), vget_low_f32(self.v)) }
    }

    #[inline(always)]
    pub(crate) fn mul_by_complex(self, other: NeonStoreF) -> Self {
        NeonStoreF {
            v: vfcmulq_f32(self.v, other.v),
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn fcmul_fcma(self, other: NeonStoreF) -> Self {
        NeonStoreF {
            v: vcmlaq_rot90_f32(
                vcmlaq_f32(vdupq_n_f32(0.), self.v, other.v),
                self.v,
                other.v,
            ),
        }
    }
}

impl NeonStoreFh {
    #[inline]
    pub(crate) fn raw(r: float32x2_t) -> NeonStoreFh {
        NeonStoreFh { v: r }
    }

    #[inline]
    pub(crate) fn load(ptr: *const f32) -> Self {
        unsafe { NeonStoreFh { v: vld1_f32(ptr) } }
    }

    // #[inline]
    // pub(crate) fn from_complex_ref(complex: &[Complex<f32>]) -> Self {
    //     unsafe {
    //         NeonStoreFh {
    //             v: vld1_f32(complex.as_ptr().cast()),
    //         }
    //     }
    // }

    #[inline]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { vst1_f32(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    pub(crate) fn mul_by_complex(self, other: NeonStoreFh) -> Self {
        NeonStoreFh {
            v: vfcmul_f32(self.v, other.v),
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn fcmul_fcma(self, other: NeonStoreFh) -> Self {
        NeonStoreFh {
            v: vcmla_rot90_f32(vcmla_f32(vdup_n_f32(0.), self.v, other.v), self.v, other.v),
        }
    }
}

impl Default for NeonStoreD {
    #[inline]
    fn default() -> Self {
        unsafe { NeonStoreD { v: vdupq_n_f64(0.) } }
    }
}

impl Default for NeonStoreF {
    #[inline]
    fn default() -> Self {
        unsafe { NeonStoreF { v: vdupq_n_f32(0.) } }
    }
}

impl Default for NeonStoreFh {
    #[inline]
    fn default() -> Self {
        unsafe { NeonStoreFh { v: vdup_n_f32(0.) } }
    }
}

impl MulAdd<NeonStoreF> for NeonStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: NeonStoreF, b: Self) -> Self::Output {
        unsafe { NeonStoreF::raw(vfmaq_f32(b.v, self.v, a.v)) }
    }
}

impl Mul<f32> for NeonStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        unsafe { NeonStoreF::raw(vmulq_n_f32(self.v, rhs)) }
    }
}

impl Mul<NeonStoreF> for f32 {
    type Output = NeonStoreF;

    #[inline(always)]
    fn mul(self, rhs: NeonStoreF) -> Self::Output {
        unsafe { NeonStoreF::raw(vmulq_n_f32(rhs.v, self)) }
    }
}

impl Mul<NeonStoreF> for NeonStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: NeonStoreF) -> Self::Output {
        unsafe { NeonStoreF::raw(vmulq_f32(self.v, rhs.v)) }
    }
}

impl Add<NeonStoreF> for NeonStoreF {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: NeonStoreF) -> Self::Output {
        unsafe { NeonStoreF::raw(vaddq_f32(self.v, rhs.v)) }
    }
}

impl Sub<NeonStoreF> for NeonStoreF {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: NeonStoreF) -> Self::Output {
        unsafe { NeonStoreF::raw(vsubq_f32(self.v, rhs.v)) }
    }
}

impl Neg for NeonStoreF {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { NeonStoreF::raw(vnegq_f32(self.v)) }
    }
}

impl MulAdd<NeonStoreD> for NeonStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: NeonStoreD, b: Self) -> Self::Output {
        unsafe { NeonStoreD::raw(vfmaq_f64(b.v, self.v, a.v)) }
    }
}

impl Mul<f64> for NeonStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        unsafe { NeonStoreD::raw(vmulq_n_f64(self.v, rhs)) }
    }
}

impl Mul<NeonStoreD> for f64 {
    type Output = NeonStoreD;

    #[inline(always)]
    fn mul(self, rhs: NeonStoreD) -> Self::Output {
        unsafe { NeonStoreD::raw(vmulq_n_f64(rhs.v, self)) }
    }
}

impl Mul<NeonStoreD> for NeonStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: NeonStoreD) -> Self::Output {
        unsafe { NeonStoreD::raw(vmulq_f64(self.v, rhs.v)) }
    }
}

impl Add<NeonStoreD> for NeonStoreD {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: NeonStoreD) -> Self::Output {
        unsafe { NeonStoreD::raw(vaddq_f64(self.v, rhs.v)) }
    }
}

impl Sub<NeonStoreD> for NeonStoreD {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: NeonStoreD) -> Self::Output {
        unsafe { NeonStoreD::raw(vsubq_f64(self.v, rhs.v)) }
    }
}

impl Neg for NeonStoreD {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { NeonStoreD::raw(vnegq_f64(self.v)) }
    }
}
