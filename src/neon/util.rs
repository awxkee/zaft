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
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::any::TypeId;
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) fn vfcmulq_f32(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
    unsafe {
        let temp1 = vtrn1q_f32(rhs, rhs);
        let temp2 = vtrn2q_f32(rhs, vnegq_f32(rhs));
        let temp3 = vmulq_f32(temp2, lhs);
        let temp4 = vrev64q_f32(temp3);
        vfmaq_f32(temp4, temp1, lhs)
    }
}

#[inline(always)]
pub(crate) fn vfcmulq_conj_b_f64(lhs: float64x2_t, rhs: float64x2_t) -> float64x2_t {
    // Multiply lhs * conj(rhs)
    // rhs = [re0, im0, re1, im1]
    // conj(rhs) = [re0, -im0, re1, -im1]

    unsafe {
        let temp = vcombine_f64(vneg_f64(vget_high_f64(lhs)), vget_low_f64(lhs));
        let sum = vmulq_laneq_f64::<0>(lhs, rhs);
        vfmaq_laneq_f64::<1>(sum, temp, vnegq_f64(rhs))
    }
}

#[inline]
#[cfg(feature = "fcma")]
#[target_feature(enable = "fcma")]
pub(crate) fn vfcmulq_conj_b_fcma_f64(lhs: float64x2_t, rhs: float64x2_t) -> float64x2_t {
    // Multiply lhs * conj(rhs)
    vcmlaq_rot270_f64(vcmlaq_f64(vdupq_n_f64(0.), rhs, lhs), rhs, lhs)
}

#[inline]
#[cfg(feature = "fcma")]
#[target_feature(enable = "fcma")]
pub(crate) fn vfcmulq_b_conj_fcma_f32(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
    // Multiply lhs * conj(rhs)
    vcmlaq_rot270_f32(vcmlaq_f32(vdupq_n_f32(0.), rhs, lhs), rhs, lhs)
}

#[inline]
#[cfg(feature = "fcma")]
#[target_feature(enable = "fcma")]
pub(crate) fn vfcmul_b_conj_fcma_f32(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t {
    // Multiply lhs * conj(rhs)
    vcmla_rot270_f32(vcmla_f32(vdup_n_f32(0.), rhs, lhs), rhs, lhs)
}

#[inline(always)]
pub(crate) fn vfcmul_f32(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t {
    unsafe {
        let temp1 = vtrn1_f32(rhs, rhs);
        let temp2 = vtrn2_f32(rhs, vneg_f32(rhs));
        let temp3 = vmul_f32(temp2, lhs);
        let temp4 = vrev64_f32(temp3);
        vfma_f32(temp4, temp1, lhs)
    }
}

#[cfg(feature = "fcma")]
#[inline]
#[target_feature(enable = "fcma")]
pub(crate) fn vfcmul_fcma_f32(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t {
    vcmla_rot90_f32(vcmla_f32(vdup_n_f32(0.), lhs, rhs), lhs, rhs)
}

#[cfg(feature = "fcma")]
#[inline]
#[target_feature(enable = "fcma")]
pub(crate) fn vfcmulq_fcma_f32(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
    vcmlaq_rot90_f32(vcmlaq_f32(vdupq_n_f32(0.), lhs, rhs), lhs, rhs)
}

#[cfg(feature = "fcma")]
#[inline]
#[target_feature(enable = "fcma")]
pub(crate) fn vfcmulq_fcma_f64(lhs: float64x2_t, rhs: float64x2_t) -> float64x2_t {
    vcmlaq_rot90_f64(vcmlaq_f64(vdupq_n_f64(0.), lhs, rhs), lhs, rhs)
}

#[inline(always)]
pub(crate) fn vfcmulq_f64(lhs: float64x2_t, rhs: float64x2_t) -> float64x2_t {
    unsafe {
        let temp = vcombine_f64(vneg_f64(vget_high_f64(lhs)), vget_low_f64(lhs));
        let sum = vmulq_laneq_f64::<0>(lhs, rhs);
        vfmaq_laneq_f64::<1>(sum, temp, rhs)
    }
}

#[inline(always)]
pub(crate) fn v_rotate90_f64(values: float64x2_t, sign: float64x2_t) -> float64x2_t {
    unsafe {
        let temp = vextq_f64::<1>(values, values);
        vreinterpretq_f64_u64(veorq_u64(
            vreinterpretq_u64_f64(temp),
            vreinterpretq_u64_f64(sign),
        ))
    }
}

#[inline(always)]
pub(crate) fn vh_rotate90_f32(values: float32x2_t, sign: float32x2_t) -> float32x2_t {
    unsafe {
        let temp = vext_f32::<1>(values, values);
        vreinterpret_f32_u32(veor_u32(
            vreinterpret_u32_f32(temp),
            vreinterpret_u32_f32(sign),
        ))
    }
}

#[inline(always)]
pub(crate) fn v_rotate90_f32(values: float32x4_t, sign: float32x4_t) -> float32x4_t {
    unsafe {
        let temp = vrev64q_f32(values);
        vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(temp),
            vreinterpretq_u32_f32(sign),
        ))
    }
}

#[inline(always)]
pub(crate) unsafe fn v_transpose_complex_f32(
    a: float32x4_t,
    b: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    unsafe {
        let u0_0 = vreinterpretq_f32_f64(vtrn1q_f64(
            vreinterpretq_f64_f32(a),
            vreinterpretq_f64_f32(b),
        ));
        let u1_0 = vreinterpretq_f32_f64(vtrn2q_f64(
            vreinterpretq_f64_f32(a),
            vreinterpretq_f64_f32(b),
        ));
        (u0_0, u1_0)
    }
}

#[inline(always)]
pub(crate) fn conjq_f32(v: float32x4_t, a: float32x4_t) -> float32x4_t {
    unsafe {
        vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(v),
            vreinterpretq_u32_f32(a),
        ))
    }
}

#[inline(always)]
pub(crate) fn conj_f64(v: float64x2_t, a: float64x2_t) -> float64x2_t {
    unsafe {
        vreinterpretq_f64_u64(veorq_u64(
            vreinterpretq_u64_f64(v),
            vreinterpretq_u64_f64(a),
        ))
    }
}

#[cfg(feature = "fcma")]
#[inline(always)]
pub(crate) fn conj_f32(v: float32x2_t, a: float32x2_t) -> float32x2_t {
    unsafe { vreinterpret_f32_u32(veor_u32(vreinterpret_u32_f32(v), vreinterpret_u32_f32(a))) }
}

#[inline(always)]
pub(crate) fn vfcmulq_conj_b_f32(lhs: float32x4_t, rhs: float32x4_t) -> float32x4_t {
    unsafe {
        let temp1 = vtrn1q_f32(rhs, rhs);
        let v_rhs = vnegq_f32(rhs);
        let temp2 = vtrn2q_f32(v_rhs, vnegq_f32(v_rhs));
        let temp3 = vmulq_f32(temp2, lhs);
        let temp4 = vrev64q_f32(temp3);
        vfmaq_f32(temp4, temp1, lhs)
    }
}

#[inline(always)]
pub(crate) fn vfcmul_conj_b_f32(lhs: float32x2_t, rhs: float32x2_t) -> float32x2_t {
    unsafe {
        let temp1 = vtrn1_f32(rhs, rhs);
        let v_rhs = vneg_f32(rhs);
        let temp2 = vtrn2_f32(v_rhs, vneg_f32(v_rhs));
        let temp3 = vmul_f32(temp2, lhs);
        let temp4 = vrev64_f32(temp3);
        vfma_f32(temp4, temp1, lhs)
    }
}

pub(crate) fn create_neon_twiddles<T: FftTrigonometry + 'static + Float + Sized, const N: usize>(
    base: usize,
    size: usize,
    fft_direction: FftDirection,
) -> Result<Vec<Complex<T>>, ZaftError>
where
    f64: AsPrimitive<T>,
{
    let mut twiddles = Vec::new();
    twiddles
        .try_reserve_exact(size - 1)
        .map_err(|_| ZaftError::OutOfMemory(size - 1))?;

    let mut cross_fft_len = base;
    while cross_fft_len < size {
        let num_columns = cross_fft_len;
        cross_fft_len *= N;

        let mut i = 0usize;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            while i + 2 <= num_columns {
                for k in 1..N {
                    let twiddle0 = compute_twiddle(i * k, cross_fft_len, fft_direction);
                    let twiddle1 = compute_twiddle((i + 1) * k, cross_fft_len, fft_direction);
                    twiddles.push(twiddle0);
                    twiddles.push(twiddle1);
                }
                i += 2;
            }
        }

        for i in i..num_columns {
            for k in 1..N {
                let twiddle = compute_twiddle(i * k, cross_fft_len, fft_direction);
                twiddles.push(twiddle);
            }
        }
    }
    Ok(twiddles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex, ComplexFloat};

    #[test]
    fn complex_a_to_b_conj_neon() {
        let values_a = [Complex::new(7.0f32, 5.0)];
        let values_b = [Complex::new(-5.0f32, 3.0)];
        let r = values_a
            .iter()
            .zip(values_b.iter())
            .map(|(a, b)| a * b.conj())
            .collect::<Vec<Complex<_>>>();
        unsafe {
            let a0 = vld1q_f32(values_a.as_ptr().cast());
            let b0 = vld1q_f32(values_b.as_ptr().cast());
            let product = vfcmulq_conj_b_f32(a0, b0);
            let mut vec_b = vec![Complex::<f32>::default(); 2];
            vst1q_f32(vec_b.as_mut_ptr().cast(), product);
            vec_b.iter().zip(r.iter()).for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-5, "complex_a_to_b_conj_sse a {a}, b {b}");
            });
        }
    }
}
