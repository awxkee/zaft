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
    _m128d_fma_mul_complex, _m128s_fma_mul_complex, _m128s_load_f32x2, _m128s_store_f32x2,
    _m256_fcmul_ps, _mm256_fcmul_pd,
};
use crate::spectrum_arithmetic::SpectrumOps;
use num_complex::Complex;
use std::arch::x86_64::*;
use std::marker::PhantomData;

pub(crate) struct AvxSpectrumArithmetic<T> {
    pub(crate) phantom_data: PhantomData<T>,
}

impl AvxSpectrumArithmetic<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn mul_f32_avx(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe {
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(16)
                .zip(a.chunks_exact(16))
                .zip(b.chunks_exact(16))
            {
                let s0 = _mm256_loadu_ps(src.as_ptr().cast());
                let s1 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let s2 = _mm256_loadu_ps(src.get_unchecked(8..).as_ptr().cast());
                let s3 = _mm256_loadu_ps(src.get_unchecked(12..).as_ptr().cast());

                let q0 = _mm256_loadu_ps(twiddle.as_ptr().cast());
                let q1 = _mm256_loadu_ps(twiddle.get_unchecked(4..).as_ptr().cast());
                let q2 = _mm256_loadu_ps(twiddle.get_unchecked(8..).as_ptr().cast());
                let q3 = _mm256_loadu_ps(twiddle.get_unchecked(12..).as_ptr().cast());

                let p0 = _m256_fcmul_ps(s0, q0);
                let p1 = _m256_fcmul_ps(s1, q1);
                let p2 = _m256_fcmul_ps(s2, q2);
                let p3 = _m256_fcmul_ps(s3, q3);

                _mm256_storeu_ps(dst.as_mut_ptr().cast(), p0);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), p1);
                _mm256_storeu_ps(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), p2);
                _mm256_storeu_ps(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(16).into_remainder();
            let a = a.chunks_exact(16).remainder();
            let b = b.chunks_exact(16).remainder();

            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(2)
                .zip(a.chunks_exact(2))
                .zip(b.chunks_exact(2))
            {
                let s0 = _mm_loadu_ps(src.as_ptr().cast());
                let q0 = _mm_loadu_ps(twiddle.as_ptr().cast());

                let p0 = _m128s_fma_mul_complex(s0, q0);

                _mm_storeu_ps(dst.as_mut_ptr().cast(), p0);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let a = a.chunks_exact(2).remainder();
            let b = b.chunks_exact(2).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = _m128s_load_f32x2(src as *const Complex<f32>);
                let q0 = _m128s_load_f32x2(twiddle as *const Complex<f32>);

                let p0 = _m128s_fma_mul_complex(s0, q0);

                _m128s_store_f32x2(dst as *mut Complex<f32>, p0);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn mul_conjugate_in_place_f32(&self, dst: &mut [Complex<f32>], b: &[Complex<f32>]) {
        unsafe {
            let factors = _mm256_loadu_ps([0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0].as_ptr());
            for (dst, twiddle) in dst.chunks_exact_mut(16).zip(b.chunks_exact(16)) {
                let s0 = _mm256_loadu_ps(dst.as_ptr().cast());
                let s1 = _mm256_loadu_ps(dst.get_unchecked(4..).as_ptr().cast());
                let s2 = _mm256_loadu_ps(dst.get_unchecked(8..).as_ptr().cast());
                let s3 = _mm256_loadu_ps(dst.get_unchecked(12..).as_ptr().cast());

                let q0 = _mm256_loadu_ps(twiddle.as_ptr().cast());
                let q1 = _mm256_loadu_ps(twiddle.get_unchecked(4..).as_ptr().cast());
                let q2 = _mm256_loadu_ps(twiddle.get_unchecked(8..).as_ptr().cast());
                let q3 = _mm256_loadu_ps(twiddle.get_unchecked(12..).as_ptr().cast());

                let mut p0 = _m256_fcmul_ps(s0, q0);
                let mut p1 = _m256_fcmul_ps(s1, q1);
                let mut p2 = _m256_fcmul_ps(s2, q2);
                let mut p3 = _m256_fcmul_ps(s3, q3);

                p0 = _mm256_xor_ps(p0, factors);
                p1 = _mm256_xor_ps(p1, factors);
                p2 = _mm256_xor_ps(p2, factors);
                p3 = _mm256_xor_ps(p3, factors);

                _mm256_storeu_ps(dst.as_mut_ptr().cast(), p0);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), p1);
                _mm256_storeu_ps(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), p2);
                _mm256_storeu_ps(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(16).into_remainder();
            let b = b.chunks_exact(16).remainder();

            for (dst, twiddle) in dst.chunks_exact_mut(2).zip(b.chunks_exact(2)) {
                let s0 = _mm_loadu_ps(dst.as_ptr().cast());
                let q0 = _mm_loadu_ps(twiddle.as_ptr().cast());

                let mut p0 = _m128s_fma_mul_complex(s0, q0);

                p0 = _mm_xor_ps(p0, _mm256_castps256_ps128(factors));

                _mm_storeu_ps(dst.as_mut_ptr().cast(), p0);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let b = b.chunks_exact(2).remainder();

            for (dst, twiddle) in dst.iter_mut().zip(b.iter()) {
                let s0 = _m128s_load_f32x2(dst as *const Complex<f32>);
                let q0 = _m128s_load_f32x2(twiddle as *const Complex<f32>);

                let mut p0 = _m128s_fma_mul_complex(s0, q0);

                p0 = _mm_xor_ps(p0, _mm256_castps256_ps128(factors));

                _m128s_store_f32x2(dst as *mut Complex<f32>, p0);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn conjugate_mul_by_b_f32(
        &self,
        a: &[Complex<f32>],
        b: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) {
        unsafe {
            let factors = _mm256_loadu_ps([0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0].as_ptr());
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(16)
                .zip(a.chunks_exact(16))
                .zip(b.chunks_exact(16))
            {
                let mut s0 = _mm256_loadu_ps(src.as_ptr().cast());
                let mut s1 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let mut s2 = _mm256_loadu_ps(src.get_unchecked(8..).as_ptr().cast());
                let mut s3 = _mm256_loadu_ps(src.get_unchecked(12..).as_ptr().cast());

                s0 = _mm256_xor_ps(s0, factors);
                s1 = _mm256_xor_ps(s1, factors);
                s2 = _mm256_xor_ps(s2, factors);
                s3 = _mm256_xor_ps(s3, factors);

                let q0 = _mm256_loadu_ps(twiddle.as_ptr().cast());
                let q1 = _mm256_loadu_ps(twiddle.get_unchecked(4..).as_ptr().cast());
                let q2 = _mm256_loadu_ps(twiddle.get_unchecked(8..).as_ptr().cast());
                let q3 = _mm256_loadu_ps(twiddle.get_unchecked(12..).as_ptr().cast());

                let p0 = _m256_fcmul_ps(s0, q0);
                let p1 = _m256_fcmul_ps(s1, q1);
                let p2 = _m256_fcmul_ps(s2, q2);
                let p3 = _m256_fcmul_ps(s3, q3);

                _mm256_storeu_ps(dst.as_mut_ptr().cast(), p0);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), p1);
                _mm256_storeu_ps(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), p2);
                _mm256_storeu_ps(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(16).into_remainder();
            let a = a.chunks_exact(16).remainder();
            let b = b.chunks_exact(16).remainder();

            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(2)
                .zip(a.chunks_exact(2))
                .zip(b.chunks_exact(2))
            {
                let mut s0 = _mm_loadu_ps(src.as_ptr().cast());
                let q0 = _mm_loadu_ps(twiddle.as_ptr().cast());

                s0 = _mm_xor_ps(s0, _mm256_castps256_ps128(factors));

                let p0 = _m128s_fma_mul_complex(s0, q0);

                _mm_storeu_ps(dst.as_mut_ptr().cast(), p0);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let a = a.chunks_exact(2).remainder();
            let b = b.chunks_exact(2).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let mut s0 = _m128s_load_f32x2(src as *const Complex<f32>);
                let q0 = _m128s_load_f32x2(twiddle as *const Complex<f32>);

                s0 = _mm_xor_ps(s0, _mm256_castps256_ps128(factors));

                let p0 = _m128s_fma_mul_complex(s0, q0);

                _m128s_store_f32x2(dst as *mut Complex<f32>, p0);
            }
        }
    }
}
impl SpectrumOps<f32> for AvxSpectrumArithmetic<f32> {
    fn mul(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe { self.mul_f32_avx(a, b, dst) }
    }

    fn mul_conjugate_in_place(&self, dst: &mut [Complex<f32>], b: &[Complex<f32>]) {
        unsafe {
            self.mul_conjugate_in_place_f32(dst, b);
        }
    }

    fn conjugate_mul_by_b(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe {
            self.conjugate_mul_by_b_f32(a, b, dst);
        }
    }
}

impl AvxSpectrumArithmetic<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn mul_f64_avx(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe {
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let s0 = _mm256_loadu_pd(src.as_ptr().cast());
                let s1 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let s2 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let s3 = _mm256_loadu_pd(src.get_unchecked(6..).as_ptr().cast());

                let q0 = _mm256_loadu_pd(twiddle.as_ptr().cast());
                let q1 = _mm256_loadu_pd(twiddle.get_unchecked(2..).as_ptr().cast());
                let q2 = _mm256_loadu_pd(twiddle.get_unchecked(4..).as_ptr().cast());
                let q3 = _mm256_loadu_pd(twiddle.get_unchecked(6..).as_ptr().cast());

                let p0 = _mm256_fcmul_pd(s0, q0);
                let p1 = _mm256_fcmul_pd(s1, q1);
                let p2 = _mm256_fcmul_pd(s2, q2);
                let p3 = _mm256_fcmul_pd(s3, q3);

                _mm256_storeu_pd(dst.as_mut_ptr().cast(), p0);
                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p1);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), p2);
                _mm256_storeu_pd(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(4).into_remainder();
            let a = a.chunks_exact(4).remainder();
            let b = b.chunks_exact(4).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = _mm_loadu_pd(src as *const Complex<f64> as *const f64);
                let q0 = _mm_loadu_pd(twiddle as *const Complex<f64> as *const f64);

                let p0 = _m128d_fma_mul_complex(s0, q0);

                _mm_storeu_pd(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn mul_conjugate_in_place_f64(&self, dst: &mut [Complex<f64>], b: &[Complex<f64>]) {
        unsafe {
            let factors = _mm256_loadu_pd([0.0f64, -0.0, 0.0, -0.0].as_ptr());
            for (dst, twiddle) in dst.chunks_exact_mut(8).zip(b.chunks_exact(8)) {
                let s0 = _mm256_loadu_pd(dst.as_ptr().cast());
                let s1 = _mm256_loadu_pd(dst.get_unchecked(2..).as_ptr().cast());
                let s2 = _mm256_loadu_pd(dst.get_unchecked(4..).as_ptr().cast());
                let s3 = _mm256_loadu_pd(dst.get_unchecked(6..).as_ptr().cast());

                let q0 = _mm256_loadu_pd(twiddle.as_ptr().cast());
                let q1 = _mm256_loadu_pd(twiddle.get_unchecked(2..).as_ptr().cast());
                let q2 = _mm256_loadu_pd(twiddle.get_unchecked(4..).as_ptr().cast());
                let q3 = _mm256_loadu_pd(twiddle.get_unchecked(6..).as_ptr().cast());

                let mut p0 = _mm256_fcmul_pd(s0, q0);
                let mut p1 = _mm256_fcmul_pd(s1, q1);
                let mut p2 = _mm256_fcmul_pd(s2, q2);
                let mut p3 = _mm256_fcmul_pd(s3, q3);

                p0 = _mm256_xor_pd(p0, factors);
                p1 = _mm256_xor_pd(p1, factors);
                p2 = _mm256_xor_pd(p2, factors);
                p3 = _mm256_xor_pd(p3, factors);

                _mm256_storeu_pd(dst.as_mut_ptr().cast(), p0);
                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p1);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), p2);
                _mm256_storeu_pd(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(4).into_remainder();
            let b = b.chunks_exact(4).remainder();

            for (dst, twiddle) in dst.iter_mut().zip(b.iter()) {
                let s0 = _mm_loadu_pd(dst as *const Complex<f64> as *const f64);
                let q0 = _mm_loadu_pd(twiddle as *const Complex<f64> as *const f64);

                let mut p0 = _m128d_fma_mul_complex(s0, q0);

                p0 = _mm_xor_pd(p0, _mm256_castpd256_pd128(factors));

                _mm_storeu_pd(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn conjugate_mul_by_b_f64(
        &self,
        a: &[Complex<f64>],
        b: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) {
        unsafe {
            let factors = _mm256_loadu_pd([0.0f64, -0.0, 0.0, -0.0].as_ptr());
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let mut s0 = _mm256_loadu_pd(src.as_ptr().cast());
                let mut s1 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let mut s2 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let mut s3 = _mm256_loadu_pd(src.get_unchecked(6..).as_ptr().cast());

                s0 = _mm256_xor_pd(s0, factors);
                s1 = _mm256_xor_pd(s1, factors);
                s2 = _mm256_xor_pd(s2, factors);
                s3 = _mm256_xor_pd(s3, factors);

                let q0 = _mm256_loadu_pd(twiddle.as_ptr().cast());
                let q1 = _mm256_loadu_pd(twiddle.get_unchecked(2..).as_ptr().cast());
                let q2 = _mm256_loadu_pd(twiddle.get_unchecked(4..).as_ptr().cast());
                let q3 = _mm256_loadu_pd(twiddle.get_unchecked(6..).as_ptr().cast());

                let p0 = _mm256_fcmul_pd(s0, q0);
                let p1 = _mm256_fcmul_pd(s1, q1);
                let p2 = _mm256_fcmul_pd(s2, q2);
                let p3 = _mm256_fcmul_pd(s3, q3);

                _mm256_storeu_pd(dst.as_mut_ptr().cast(), p0);
                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p1);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), p2);
                _mm256_storeu_pd(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(4).into_remainder();
            let a = a.chunks_exact(4).remainder();
            let b = b.chunks_exact(4).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let mut s0 = _mm_loadu_pd(src as *const Complex<f64> as *const f64);
                let q0 = _mm_loadu_pd(twiddle as *const Complex<f64> as *const f64);

                s0 = _mm_xor_pd(s0, _mm256_castpd256_pd128(factors));

                let p0 = _m128d_fma_mul_complex(s0, q0);

                _mm_storeu_pd(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }
}

impl SpectrumOps<f64> for AvxSpectrumArithmetic<f64> {
    fn mul(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe { self.mul_f64_avx(a, b, dst) }
    }

    fn mul_conjugate_in_place(&self, dst: &mut [Complex<f64>], b: &[Complex<f64>]) {
        unsafe { self.mul_conjugate_in_place_f64(dst, b) }
    }

    fn conjugate_mul_by_b(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe { self.conjugate_mul_by_b_f64(a, b, dst) }
    }
}
