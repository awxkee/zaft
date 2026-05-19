/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

use num_complex::Complex;
use num_traits::MulAdd;
use std::arch::wasm32::*;
use std::mem::MaybeUninit;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy)]
pub(crate) struct WasmStoreD {
    pub(crate) v: v128,
}

impl WasmStoreD {
    #[inline(always)]
    pub(crate) fn raw(r: v128) -> Self {
        Self { v: r }
    }

    #[inline(always)]
    pub(crate) fn set_values(p0: f64, p1: f64) -> Self {
        Self { v: f64x2(p0, p1) }
    }

    /// Blend: select `self` where `mask` bits are set, else `other`.
    /// The mask is interpreted lane-wise (any non-zero bit in a lane → take self).
    #[inline(always)]
    pub(crate) fn select(&self, other: Self, mask: Self) -> Self {
        Self {
            v: v128_bitselect(self.v, other.v, mask.v),
        }
    }

    /// Returns a mask value suitable for conjugating a complex number:
    /// XOR-ing with this flips the sign of the imaginary (high) lane.
    #[inline(always)]
    pub(crate) fn conj_flag() -> Self {
        // [+0.0, -0.0] — XOR with this negates only the imaginary lane.
        Self {
            v: f64x2(0.0_f64, -0.0_f64),
        }
    }

    #[inline(always)]
    pub(crate) fn dup(p0: f64) -> Self {
        Self { v: f64x2_splat(p0) }
    }

    #[inline(always)]
    pub(crate) fn from_complex_ref(complex: &[Complex<f64>]) -> Self {
        unsafe {
            Self {
                v: v128_load(complex.as_ptr() as *const v128),
            }
        }
    }

    /// Returns `[even-lanes-duped, odd-lanes-duped]`, i.e.
    /// `[[re,re], [im,im]]`.
    #[inline(always)]
    pub(crate) fn dup_even_odds(&self) -> [Self; 2] {
        [
            Self {
                v: i64x2_shuffle::<0, 0>(self.v, self.v),
            },
            Self {
                v: i64x2_shuffle::<1, 1>(self.v, self.v),
            },
        ]
    }

    /// Swap the two lanes (reverses a packed complex number).
    #[inline(always)]
    pub(crate) fn reverse_complex_elements(&self) -> Self {
        Self {
            v: i64x2_shuffle::<1, 0>(self.v, self.v),
        }
    }

    #[inline(always)]
    pub(crate) fn xor(&self, other: Self) -> Self {
        Self {
            v: v128_xor(self.v, other.v),
        }
    }

    #[inline(always)]
    pub(crate) fn load(vals: &[f64]) -> Self {
        unsafe {
            Self {
                v: v128_load(vals.as_ptr() as *const v128),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn from_complex_refu(complex: &[MaybeUninit<Complex<f64>>]) -> Self {
        unsafe {
            Self {
                v: v128_load(complex.as_ptr() as *const v128),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn from_complex(complex: &Complex<f64>) -> Self {
        unsafe {
            Self {
                v: v128_load(complex as *const Complex<f64> as *const v128),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f64>]) {
        unsafe { v128_store(to_ref.as_mut_ptr() as *mut v128, self.v) }
    }

    #[inline(always)]
    pub(crate) fn write_single(&self, to_ref: &mut Complex<f64>) {
        unsafe { v128_store(to_ref as *mut Complex<f64> as *mut v128, self.v) }
    }

    #[inline(always)]
    pub(crate) fn write_uninit(&self, to_ref: &mut [MaybeUninit<Complex<f64>>]) {
        unsafe { v128_store(to_ref.as_mut_ptr() as *mut v128, self.v) }
    }

    /// Complex multiplication: treats `self` and `other` each as one packed
    /// `Complex<f64>` and returns their product.
    #[inline(always)]
    pub(crate) fn mul_by_complex(self, other: Self) -> Self {
        // [-ai, ar]
        let neg_im = f64x2_neg(self.v); // [-ar, -ai]
        let temp = i64x2_shuffle::<1, 2>(neg_im, self.v); // [-ai, ar]

        let br = i64x2_shuffle::<0, 0>(other.v, other.v);
        let bi = i64x2_shuffle::<1, 1>(other.v, other.v);
        let sum = f64x2_mul(self.v, br);
        Self::raw(f64x2_add(sum, f64x2_mul(temp, bi)))
    }

    #[inline(always)]
    pub(crate) fn mul_nadd(self, a: WasmStoreD, b: Self) -> Self {
        Self::raw(f64x2_sub(b.v, f64x2_mul(self.v, a.v)))
    }
}

// ---------------------------------------------------------------------------
// WasmStoreF  (4 × f32, equivalent to float32x4_t)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct WasmStoreF {
    pub(crate) v: v128,
}

impl WasmStoreF {
    #[inline(always)]
    pub(crate) fn raw(r: v128) -> Self {
        Self { v: r }
    }

    /// Conjugate-flag mask: XOR-ing with this flips the sign of the imaginary
    /// lane of each packed complex pair ([re, im, re, im] → flip im lanes).
    #[inline(always)]
    pub(crate) fn conj_flag() -> Self {
        Self {
            v: f32x4(0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32),
        }
    }

    #[inline(always)]
    pub(crate) fn load(ptr: &[f32]) -> Self {
        unsafe {
            Self {
                v: v128_load(ptr.as_ptr() as *const v128),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn from_complex_ref(complex: &[Complex<f32>]) -> Self {
        unsafe {
            Self {
                v: v128_load(complex.as_ptr() as *const v128),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn from_complex_refu(complex: &[MaybeUninit<Complex<f32>>]) -> Self {
        unsafe {
            Self {
                v: v128_load(complex.as_ptr() as *const v128),
            }
        }
    }

    /// Broadcast a single complex value to both pairs: `[re, im, re, im]`.
    #[inline(always)]
    pub(crate) fn from_complex(complex: &Complex<f32>) -> Self {
        Self {
            v: f32x4(complex.re, complex.im, complex.re, complex.im),
        }
    }

    /// Load a single complex value into the low pair; upper pair zeroed.
    #[inline(always)]
    pub(crate) fn load_complex(complex: &Complex<f32>) -> Self {
        unsafe {
            Self {
                v: v128_load64_zero(complex as *const Complex<f32> as *const u64),
            }
        }
    }

    /// Returns `[even-lane-duped, odd-lane-duped]` across all four lanes.
    #[inline(always)]
    pub(crate) fn dup_even_odds(&self) -> [Self; 2] {
        [
            Self {
                v: i32x4_shuffle::<0, 0, 2, 2>(self.v, self.v),
            },
            Self {
                v: i32x4_shuffle::<1, 1, 3, 3>(self.v, self.v),
            },
        ]
    }

    /// Swap the two complex pairs: `[a,b,c,d]` → `[c,d,a,b]`.
    #[inline(always)]
    pub(crate) fn reverse_complex(&self) -> Self {
        Self {
            v: i64x2_shuffle::<1, 0>(self.v, self.v),
        }
    }

    #[inline(always)]
    pub(crate) fn xor(&self, other: Self) -> Self {
        Self {
            v: v128_xor(self.v, other.v),
        }
    }

    /// Swap re/im within each complex pair: `[re,im,re,im]` → `[im,re,im,re]`.
    #[inline(always)]
    pub(crate) fn reverse_complex_elements(&self) -> Self {
        Self {
            v: i32x4_shuffle::<1, 0, 3, 2>(self.v, self.v),
        }
    }

    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self {
            v: f32x4_splat(0.0),
        }
    }

    #[inline(always)]
    pub(crate) fn select(&self, other: Self, mask: Self) -> Self {
        Self {
            v: v128_bitselect(self.v, other.v, mask.v),
        }
    }

    #[inline(always)]
    pub(crate) fn dup(p0: f32) -> Self {
        Self { v: f32x4_splat(p0) }
    }

    #[inline(always)]
    pub(crate) fn from_complex2(v0: Complex<f32>, v1: Complex<f32>) -> Self {
        Self {
            v: f32x4(v0.re, v0.im, v1.re, v1.im),
        }
    }

    #[inline(always)]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { v128_store(to_ref.as_mut_ptr() as *mut v128, self.v) }
    }

    /// Write only the low complex pair (lanes 0-1).
    #[inline(always)]
    pub(crate) fn write_single(&self, to_ref: &mut Complex<f32>) {
        unsafe { v128_store64_lane::<0>(self.v, to_ref as *mut Complex<f32> as *mut u64) }
    }

    #[inline(always)]
    pub(crate) fn write_uninit(&self, to_ref: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe { v128_store(to_ref.as_mut_ptr() as *mut v128, self.v) }
    }

    #[inline(always)]
    pub(crate) fn write_lo(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { v128_store64_lane::<0>(self.v, to_ref.as_mut_ptr() as *mut u64) }
    }

    #[inline(always)]
    pub(crate) fn mul_by_complex(self, other: Self) -> Self {
        let temp1 = i32x4_shuffle::<0, 0, 2, 2>(other.v, other.v);
        let neg_other = f32x4_neg(other.v);
        let temp2 = i32x4_shuffle::<1, 5, 3, 7>(other.v, neg_other);
        let temp3 = f32x4_mul(temp2, self.v);
        let temp4 = i32x4_shuffle::<1, 0, 3, 2>(temp3, temp3);
        Self::raw(f32x4_add(f32x4_mul(temp1, self.v), temp4))
    }

    #[inline(always)]
    pub(crate) fn mul_nadd(self, a: WasmStoreF, b: Self) -> Self {
        Self::raw(f32x4_sub(b.v, f32x4_mul(self.v, a.v)))
    }
}

impl Default for WasmStoreD {
    #[inline(always)]
    fn default() -> Self {
        Self {
            v: f64x2_splat(0.0),
        }
    }
}

impl Default for WasmStoreF {
    #[inline(always)]
    fn default() -> Self {
        Self {
            v: f32x4_splat(0.0),
        }
    }
}

impl MulAdd<WasmStoreF> for WasmStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: WasmStoreF, b: Self) -> Self::Output {
        Self::raw(f32x4_add(f32x4_mul(self.v, a.v), b.v))
    }
}

impl Mul<f32> for WasmStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::raw(f32x4_mul(self.v, f32x4_splat(rhs)))
    }
}

impl Mul<WasmStoreF> for f32 {
    type Output = WasmStoreF;

    #[inline(always)]
    fn mul(self, rhs: WasmStoreF) -> Self::Output {
        WasmStoreF::raw(f32x4_mul(rhs.v, f32x4_splat(self)))
    }
}

impl Mul<WasmStoreF> for WasmStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: WasmStoreF) -> Self::Output {
        Self::raw(f32x4_mul(self.v, rhs.v))
    }
}

impl Add<WasmStoreF> for WasmStoreF {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: WasmStoreF) -> Self::Output {
        Self::raw(f32x4_add(self.v, rhs.v))
    }
}

impl Sub<WasmStoreF> for WasmStoreF {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: WasmStoreF) -> Self::Output {
        Self::raw(f32x4_sub(self.v, rhs.v))
    }
}

impl Neg for WasmStoreF {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::raw(f32x4_neg(self.v))
    }
}

impl MulAdd<WasmStoreD> for WasmStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: WasmStoreD, b: Self) -> Self::Output {
        Self::raw(f64x2_add(f64x2_mul(self.v, a.v), b.v))
    }
}

impl Mul<f64> for WasmStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        Self::raw(f64x2_mul(self.v, f64x2_splat(rhs)))
    }
}

impl Mul<WasmStoreD> for f64 {
    type Output = WasmStoreD;

    #[inline(always)]
    fn mul(self, rhs: WasmStoreD) -> Self::Output {
        WasmStoreD::raw(f64x2_mul(rhs.v, f64x2_splat(self)))
    }
}

impl Mul<WasmStoreD> for WasmStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: WasmStoreD) -> Self::Output {
        Self::raw(f64x2_mul(self.v, rhs.v))
    }
}

impl Add<WasmStoreD> for WasmStoreD {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: WasmStoreD) -> Self::Output {
        Self::raw(f64x2_add(self.v, rhs.v))
    }
}

impl Sub<WasmStoreD> for WasmStoreD {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: WasmStoreD) -> Self::Output {
        Self::raw(f64x2_sub(self.v, rhs.v))
    }
}

impl Neg for WasmStoreD {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::raw(f64x2_neg(self.v))
    }
}
