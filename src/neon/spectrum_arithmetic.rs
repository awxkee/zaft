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
use crate::complex_fma::c_mul_fast;
use crate::neon::util::{conj_f64, conjq_f32, vfcmulq_f32, vfcmulq_f64};
use crate::spectrum_arithmetic::SpectrumOps;
use num_complex::Complex;
use std::arch::aarch64::*;
use std::marker::PhantomData;

pub(crate) struct NeonSpectrumArithmetic<T> {
    pub(crate) phantom_data: PhantomData<T>,
}

impl SpectrumOps<f32> for NeonSpectrumArithmetic<f32> {
    fn mul(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe {
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let s0 = vld1q_f32(src.as_ptr().cast());
                let s1 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let s2 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let s3 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());

                let q0 = vld1q_f32(twiddle.as_ptr().cast());
                let q1 = vld1q_f32(twiddle.get_unchecked(2..).as_ptr().cast());
                let q2 = vld1q_f32(twiddle.get_unchecked(4..).as_ptr().cast());
                let q3 = vld1q_f32(twiddle.get_unchecked(6..).as_ptr().cast());

                let p0 = vfcmulq_f32(s0, q0);
                let p1 = vfcmulq_f32(s1, q1);
                let p2 = vfcmulq_f32(s2, q2);
                let p3 = vfcmulq_f32(s3, q3);

                vst1q_f32(dst.as_mut_ptr().cast(), p0);
                vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p1);
                vst1q_f32(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), p2);
                vst1q_f32(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(8).into_remainder();
            let a = a.chunks_exact(8).remainder();
            let b = b.chunks_exact(8).remainder();

            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(2)
                .zip(a.chunks_exact(2))
                .zip(b.chunks_exact(2))
            {
                let s0 = vld1q_f32(src.as_ptr().cast());
                let q0 = vld1q_f32(twiddle.as_ptr().cast());

                let p0 = vfcmulq_f32(s0, q0);

                vst1q_f32(dst.as_mut_ptr().cast(), p0);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let a = a.chunks_exact(2).remainder();
            let b = b.chunks_exact(2).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                *dst = c_mul_fast(*src, *twiddle);
            }
        }
    }

    fn mul_conjugate_in_place(&self, dst: &mut [Complex<f32>], b: &[Complex<f32>]) {
        unsafe {
            let conjugate_factors = vld1q_f32([0.0f32, -0.0f32, 0.0f32, -0.0f32].as_ptr());
            for (dst, twiddle) in dst.chunks_exact_mut(8).zip(b.chunks_exact(8)) {
                let s0 = vld1q_f32(dst.as_ptr().cast());
                let s1 = vld1q_f32(dst.get_unchecked(2..).as_ptr().cast());
                let s2 = vld1q_f32(dst.get_unchecked(4..).as_ptr().cast());
                let s3 = vld1q_f32(dst.get_unchecked(6..).as_ptr().cast());

                let q0 = vld1q_f32(twiddle.as_ptr().cast());
                let q1 = vld1q_f32(twiddle.get_unchecked(2..).as_ptr().cast());
                let q2 = vld1q_f32(twiddle.get_unchecked(4..).as_ptr().cast());
                let q3 = vld1q_f32(twiddle.get_unchecked(6..).as_ptr().cast());

                let mut p0 = vfcmulq_f32(s0, q0);
                let mut p1 = vfcmulq_f32(s1, q1);
                let mut p2 = vfcmulq_f32(s2, q2);
                let mut p3 = vfcmulq_f32(s3, q3);

                p0 = conjq_f32(p0, conjugate_factors);
                p1 = conjq_f32(p1, conjugate_factors);
                p2 = conjq_f32(p2, conjugate_factors);
                p3 = conjq_f32(p3, conjugate_factors);

                vst1q_f32(dst.as_mut_ptr().cast(), p0);
                vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p1);
                vst1q_f32(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), p2);
                vst1q_f32(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(8).into_remainder();
            let b = b.chunks_exact(8).remainder();

            for (dst, twiddle) in dst.chunks_exact_mut(2).zip(b.chunks_exact(2)) {
                let s0 = vld1q_f32(dst.as_ptr().cast());
                let q0 = vld1q_f32(twiddle.as_ptr().cast());

                let p0 = conjq_f32(vfcmulq_f32(s0, q0), conjugate_factors);

                vst1q_f32(dst.as_mut_ptr().cast(), p0);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let b = b.chunks_exact(2).remainder();

            for (dst, twiddle) in dst.iter_mut().zip(b.iter()) {
                *dst = c_mul_fast(*dst, *twiddle).conj();
            }
        }
    }

    fn conjugate_mul_by_b(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe {
            let conjugate_factors = vld1q_f32([0.0f32, -0.0f32, 0.0f32, -0.0f32].as_ptr());
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let mut s0 = vld1q_f32(src.as_ptr().cast());
                let mut s1 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let mut s2 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let mut s3 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());

                s0 = conjq_f32(s0, conjugate_factors);
                s1 = conjq_f32(s1, conjugate_factors);
                s2 = conjq_f32(s2, conjugate_factors);
                s3 = conjq_f32(s3, conjugate_factors);

                let q0 = vld1q_f32(twiddle.as_ptr().cast());
                let q1 = vld1q_f32(twiddle.get_unchecked(2..).as_ptr().cast());
                let q2 = vld1q_f32(twiddle.get_unchecked(4..).as_ptr().cast());
                let q3 = vld1q_f32(twiddle.get_unchecked(6..).as_ptr().cast());

                let p0 = vfcmulq_f32(s0, q0);
                let p1 = vfcmulq_f32(s1, q1);
                let p2 = vfcmulq_f32(s2, q2);
                let p3 = vfcmulq_f32(s3, q3);

                vst1q_f32(dst.as_mut_ptr().cast(), p0);
                vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p1);
                vst1q_f32(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), p2);
                vst1q_f32(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(8).into_remainder();
            let a = a.chunks_exact(8).remainder();
            let b = b.chunks_exact(8).remainder();

            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(2)
                .zip(a.chunks_exact(2))
                .zip(b.chunks_exact(2))
            {
                let mut s0 = vld1q_f32(src.as_ptr().cast());
                s0 = conjq_f32(s0, conjugate_factors);
                let q0 = vld1q_f32(twiddle.as_ptr().cast());

                let p0 = vfcmulq_f32(s0, q0);

                vst1q_f32(dst.as_mut_ptr().cast(), p0);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let a = a.chunks_exact(2).remainder();
            let b = b.chunks_exact(2).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                *dst = c_mul_fast(src.conj(), *twiddle);
            }
        }
    }
}

impl SpectrumOps<f64> for NeonSpectrumArithmetic<f64> {
    fn mul(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe {
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(4)
                .zip(a.chunks_exact(4))
                .zip(b.chunks_exact(4))
            {
                let s0 = vld1q_f64(src.as_ptr().cast());
                let s1 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());
                let s2 = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());
                let s3 = vld1q_f64(src.get_unchecked(3..).as_ptr().cast());

                let q0 = vld1q_f64(twiddle.as_ptr().cast());
                let q1 = vld1q_f64(twiddle.get_unchecked(1..).as_ptr().cast());
                let q2 = vld1q_f64(twiddle.get_unchecked(2..).as_ptr().cast());
                let q3 = vld1q_f64(twiddle.get_unchecked(3..).as_ptr().cast());

                let p0 = vfcmulq_f64(s0, q0);
                let p1 = vfcmulq_f64(s1, q1);
                let p2 = vfcmulq_f64(s2, q2);
                let p3 = vfcmulq_f64(s3, q3);

                vst1q_f64(dst.as_mut_ptr().cast(), p0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), p1);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p2);
                vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(4).into_remainder();
            let a = a.chunks_exact(4).remainder();
            let b = b.chunks_exact(4).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = vld1q_f64(src as *const Complex<f64> as *const f64);
                let q0 = vld1q_f64(twiddle as *const Complex<f64> as *const f64);

                let p0 = vfcmulq_f64(s0, q0);

                vst1q_f64(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }

    fn mul_conjugate_in_place(&self, dst: &mut [Complex<f64>], b: &[Complex<f64>]) {
        unsafe {
            let conjugate_factors = vld1q_f64([0.0f64, -0.0f64].as_ptr());
            for (dst, twiddle) in dst.chunks_exact_mut(4).zip(b.chunks_exact(4)) {
                let s0 = vld1q_f64(dst.as_ptr().cast());
                let s1 = vld1q_f64(dst.get_unchecked(1..).as_ptr().cast());
                let s2 = vld1q_f64(dst.get_unchecked(2..).as_ptr().cast());
                let s3 = vld1q_f64(dst.get_unchecked(3..).as_ptr().cast());

                let q0 = vld1q_f64(twiddle.as_ptr().cast());
                let q1 = vld1q_f64(twiddle.get_unchecked(1..).as_ptr().cast());
                let q2 = vld1q_f64(twiddle.get_unchecked(2..).as_ptr().cast());
                let q3 = vld1q_f64(twiddle.get_unchecked(3..).as_ptr().cast());

                let mut p0 = vfcmulq_f64(s0, q0);
                let mut p1 = vfcmulq_f64(s1, q1);
                let mut p2 = vfcmulq_f64(s2, q2);
                let mut p3 = vfcmulq_f64(s3, q3);

                p0 = conj_f64(p0, conjugate_factors);
                p1 = conj_f64(p1, conjugate_factors);
                p2 = conj_f64(p2, conjugate_factors);
                p3 = conj_f64(p3, conjugate_factors);

                vst1q_f64(dst.as_mut_ptr().cast(), p0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), p1);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p2);
                vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(4).into_remainder();
            let b = b.chunks_exact(4).remainder();

            for (dst, twiddle) in dst.iter_mut().zip(b.iter()) {
                let s0 = vld1q_f64(dst as *const Complex<f64> as *const f64);
                let q0 = vld1q_f64(twiddle as *const Complex<f64> as *const f64);

                let p0 = conj_f64(vfcmulq_f64(s0, q0), conjugate_factors);

                vst1q_f64(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }

    fn conjugate_mul_by_b(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe {
            let conjugate_factors = vld1q_f64([0.0f64, -0.0f64].as_ptr());
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(4)
                .zip(a.chunks_exact(4))
                .zip(b.chunks_exact(4))
            {
                let mut s0 = vld1q_f64(src.as_ptr().cast());
                let mut s1 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());
                let mut s2 = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());
                let mut s3 = vld1q_f64(src.get_unchecked(3..).as_ptr().cast());

                s0 = conj_f64(s0, conjugate_factors);
                s1 = conj_f64(s1, conjugate_factors);
                s2 = conj_f64(s2, conjugate_factors);
                s3 = conj_f64(s3, conjugate_factors);

                let q0 = vld1q_f64(twiddle.as_ptr().cast());
                let q1 = vld1q_f64(twiddle.get_unchecked(1..).as_ptr().cast());
                let q2 = vld1q_f64(twiddle.get_unchecked(2..).as_ptr().cast());
                let q3 = vld1q_f64(twiddle.get_unchecked(3..).as_ptr().cast());

                let p0 = vfcmulq_f64(s0, q0);
                let p1 = vfcmulq_f64(s1, q1);
                let p2 = vfcmulq_f64(s2, q2);
                let p3 = vfcmulq_f64(s3, q3);

                vst1q_f64(dst.as_mut_ptr().cast(), p0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), p1);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p2);
                vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(4).into_remainder();
            let a = a.chunks_exact(4).remainder();
            let b = b.chunks_exact(4).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let mut s0 = vld1q_f64(src as *const Complex<f64> as *const f64);
                let q0 = vld1q_f64(twiddle as *const Complex<f64> as *const f64);

                s0 = conj_f64(s0, conjugate_factors);

                let p0 = vfcmulq_f64(s0, q0);

                vst1q_f64(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }
}
