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
use crate::neon::util::{mul_complex_f32, mul_complex_f64};
use crate::spectrum_arithmetic::SpectrumArithmetic;
use num_complex::Complex;
use std::arch::aarch64::*;
use std::marker::PhantomData;

pub(crate) struct NeonSpectrumArithmetic<T> {
    pub(crate) phantom_data: PhantomData<T>,
}

impl SpectrumArithmetic<f32> for NeonSpectrumArithmetic<f32> {
    fn mul(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe {
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let s0 = vld1q_f32(src.as_ptr().cast());
                let s1 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let s2 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());
                let s3 = vld1q_f32(src.get_unchecked(8..).as_ptr().cast());

                let q0 = vld1q_f32(twiddle.as_ptr().cast());
                let q1 = vld1q_f32(twiddle.get_unchecked(2..).as_ptr().cast());
                let q2 = vld1q_f32(twiddle.get_unchecked(6..).as_ptr().cast());
                let q3 = vld1q_f32(twiddle.get_unchecked(8..).as_ptr().cast());

                let p0 = mul_complex_f32(s0, q0);
                let p1 = mul_complex_f32(s1, q1);
                let p2 = mul_complex_f32(s2, q2);
                let p3 = mul_complex_f32(s3, q3);

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

                let p0 = mul_complex_f32(s0, q0);

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
}

impl SpectrumArithmetic<f64> for NeonSpectrumArithmetic<f64> {
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

                let p0 = mul_complex_f64(s0, q0);
                let p1 = mul_complex_f64(s1, q1);
                let p2 = mul_complex_f64(s2, q2);
                let p3 = mul_complex_f64(s3, q3);

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

                let p0 = mul_complex_f64(s0, q0);

                vst1q_f64(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }
}
