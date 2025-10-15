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
use crate::neon::util::{mul_complex_f32, mul_complex_f64, mulh_complex_f32};
use num_complex::Complex;
use std::arch::aarch64::*;

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

    // #[inline]
    // pub(crate) fn load(ptr: *const f64) -> Self {
    //     unsafe { NeonStoreD { v: vld1q_f64(ptr) } }
    // }

    #[inline]
    pub(crate) fn from_complex_ref(complex: &[Complex<f64>]) -> Self {
        unsafe {
            NeonStoreD {
                v: vld1q_f64(complex.as_ptr().cast()),
            }
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

    #[inline]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f64>]) {
        unsafe { vst1q_f64(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    pub(crate) fn mul_by_complex(self, other: NeonStoreD) -> Self {
        unsafe {
            NeonStoreD {
                v: mul_complex_f64(self.v, other.v),
            }
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) unsafe fn fcmul_fcma(self, other: NeonStoreD) -> Self {
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
    pub(crate) fn load(ptr: *const f32) -> Self {
        unsafe { NeonStoreF { v: vld1q_f32(ptr) } }
    }

    #[inline]
    pub(crate) fn from_complex_ref(complex: &[Complex<f32>]) -> Self {
        unsafe {
            NeonStoreF {
                v: vld1q_f32(complex.as_ptr().cast()),
            }
        }
    }

    // #[inline]
    // pub(crate) fn from_complex(complex: &Complex<f32>) -> Self {
    //     unsafe {
    //         NeonStoreF {
    //             v: vld1q_f32(complex as *const Complex<f32> as *const f32),
    //         }
    //     }
    // }

    #[inline]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { vst1q_f32(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    pub(crate) fn mul_by_complex(self, other: NeonStoreF) -> Self {
        unsafe {
            NeonStoreF {
                v: mul_complex_f32(self.v, other.v),
            }
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) unsafe fn fcmul_fcma(self, other: NeonStoreF) -> Self {
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
    pub(crate) fn from_complex(complex: &Complex<f32>) -> Self {
        unsafe {
            NeonStoreFh {
                v: vld1_f32(complex as *const Complex<f32> as *const f32),
            }
        }
    }

    #[inline]
    pub(crate) fn write(&self, to_ref: &mut [Complex<f32>]) {
        unsafe { vst1_f32(to_ref.as_mut_ptr().cast(), self.v) }
    }

    #[inline]
    pub(crate) fn mul_by_complex(self, other: NeonStoreFh) -> Self {
        unsafe {
            NeonStoreFh {
                v: mulh_complex_f32(self.v, other.v),
            }
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) unsafe fn fcmul_fcma(self, other: NeonStoreFh) -> Self {
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
