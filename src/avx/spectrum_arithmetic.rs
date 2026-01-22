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
use crate::avx::mixed::{AvxStoreD, AvxStoreF};
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_fcmul_pd, _mm_fcmul_pd_conj_a, _mm_fcmul_ps,
    _mm_fcmul_ps_conj_a, _mm256_fcmul_pd, _mm256_fcmul_pd_conj_a, _mm256_fcmul_ps,
    _mm256_fcmul_ps_conj_a,
};
use crate::spectrum_arithmetic::ComplexArith;
use num_complex::Complex;
use std::arch::x86_64::*;
use std::marker::PhantomData;

pub(crate) struct AvxSpectrumArithmetic<T> {
    pub(crate) phantom_data: PhantomData<T>,
}

impl AvxSpectrumArithmetic<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_f32_avx(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe {
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(16)
                .zip(a.chunks_exact(16))
                .zip(b.chunks_exact(16))
            {
                let s0 = AvxStoreF::from_complex_ref(src);
                let s1 = AvxStoreF::from_complex_ref(src.get_unchecked(4..));
                let s2 = AvxStoreF::from_complex_ref(src.get_unchecked(8..));
                let s3 = AvxStoreF::from_complex_ref(src.get_unchecked(12..));

                let q0 = AvxStoreF::from_complex_ref(twiddle);
                let q1 = AvxStoreF::from_complex_ref(twiddle.get_unchecked(4..));
                let q2 = AvxStoreF::from_complex_ref(twiddle.get_unchecked(8..));
                let q3 = AvxStoreF::from_complex_ref(twiddle.get_unchecked(12..));

                let p0 = AvxStoreF::mul_by_complex(s0, q0);
                let p1 = AvxStoreF::mul_by_complex(s1, q1);
                let p2 = AvxStoreF::mul_by_complex(s2, q2);
                let p3 = AvxStoreF::mul_by_complex(s3, q3);

                p0.write(dst);
                p1.write(dst.get_unchecked_mut(4..));
                p2.write(dst.get_unchecked_mut(8..));
                p3.write(dst.get_unchecked_mut(12..));
            }

            let dst = dst.chunks_exact_mut(16).into_remainder();
            let a = a.chunks_exact(16).remainder();
            let b = b.chunks_exact(16).remainder();

            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(2)
                .zip(a.chunks_exact(2))
                .zip(b.chunks_exact(2))
            {
                let s0 = AvxStoreF::from_complex2(src);
                let q0 = AvxStoreF::from_complex2(twiddle);

                let p0 = AvxStoreF::mul_by_complex(s0, q0);

                p0.write_lo2(dst);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let a = a.chunks_exact(2).remainder();
            let b = b.chunks_exact(2).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = _m128s_load_f32x2(src as *const Complex<f32>);
                let q0 = _m128s_load_f32x2(twiddle as *const Complex<f32>);

                let p0 = _mm_fcmul_ps(s0, q0);

                _m128s_store_f32x2(dst as *mut Complex<f32>, p0);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_expand_f32_avx(&self, a: &[f32], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe {
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(16)
                .zip(a.chunks_exact(16))
                .zip(b.chunks_exact(16))
            {
                let q0 = AvxStoreF::load(src);
                let q1 = AvxStoreF::load(src.get_unchecked(8..));

                let [s0, s1] = q0.to_complex();
                let [s2, s3] = q1.to_complex();

                let q0 = AvxStoreF::from_complex_ref(twiddle);
                let q1 = AvxStoreF::from_complex_ref(twiddle.get_unchecked(4..));
                let q2 = AvxStoreF::from_complex_ref(twiddle.get_unchecked(8..));
                let q3 = AvxStoreF::from_complex_ref(twiddle.get_unchecked(12..));

                let p0 = AvxStoreF::mul_by_complex(s0, q0);
                let p1 = AvxStoreF::mul_by_complex(s1, q1);
                let p2 = AvxStoreF::mul_by_complex(s2, q2);
                let p3 = AvxStoreF::mul_by_complex(s3, q3);

                p0.write(dst);
                p1.write(dst.get_unchecked_mut(4..));
                p2.write(dst.get_unchecked_mut(8..));
                p3.write(dst.get_unchecked_mut(12..));
            }

            let dst = dst.chunks_exact_mut(16).into_remainder();
            let a = a.chunks_exact(16).remainder();
            let b = b.chunks_exact(16).remainder();

            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(2)
                .zip(a.chunks_exact(2))
                .zip(b.chunks_exact(2))
            {
                let s0 = AvxStoreF::load2_as_complex(src);
                let q0 = AvxStoreF::from_complex2(twiddle);

                let p0 = AvxStoreF::mul_by_complex(s0, q0);

                p0.write_lo2(dst);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let a = a.chunks_exact(2).remainder();
            let b = b.chunks_exact(2).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = _mm_unpacklo_ps(_mm_load_ss(src), _mm_setzero_ps());
                let q0 = _m128s_load_f32x2(twiddle as *const Complex<f32>);

                let p0 = _mm_fcmul_ps(s0, q0);

                _m128s_store_f32x2(dst as *mut Complex<f32>, p0);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_conjugate_in_place_f32(&self, dst: &mut [Complex<f32>], b: &[Complex<f32>]) {
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

                let mut p0 = _mm256_fcmul_ps(s0, q0);
                let mut p1 = _mm256_fcmul_ps(s1, q1);
                let mut p2 = _mm256_fcmul_ps(s2, q2);
                let mut p3 = _mm256_fcmul_ps(s3, q3);

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

                let mut p0 = _mm_fcmul_ps(s0, q0);

                p0 = _mm_xor_ps(p0, _mm256_castps256_ps128(factors));

                _mm_storeu_ps(dst.as_mut_ptr().cast(), p0);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let b = b.chunks_exact(2).remainder();

            for (dst, twiddle) in dst.iter_mut().zip(b.iter()) {
                let s0 = _m128s_load_f32x2(dst as *const Complex<f32>);
                let q0 = _m128s_load_f32x2(twiddle as *const Complex<f32>);

                let mut p0 = _mm_fcmul_ps(s0, q0);

                p0 = _mm_xor_ps(p0, _mm256_castps256_ps128(factors));

                _m128s_store_f32x2(dst as *mut Complex<f32>, p0);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn conjugate_mul_by_b_f32(
        &self,
        a: &[Complex<f32>],
        b: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) {
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

                let p0 = _mm256_fcmul_ps_conj_a(s0, q0);
                let p1 = _mm256_fcmul_ps_conj_a(s1, q1);
                let p2 = _mm256_fcmul_ps_conj_a(s2, q2);
                let p3 = _mm256_fcmul_ps_conj_a(s3, q3);

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

                let p0 = _mm_fcmul_ps_conj_a(s0, q0);

                _mm_storeu_ps(dst.as_mut_ptr().cast(), p0);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let a = a.chunks_exact(2).remainder();
            let b = b.chunks_exact(2).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = _m128s_load_f32x2(src as *const Complex<f32>);
                let q0 = _m128s_load_f32x2(twiddle as *const Complex<f32>);

                let p0 = _mm_fcmul_ps_conj_a(s0, q0);

                _m128s_store_f32x2(dst as *mut Complex<f32>, p0);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_and_cut_f32(
        &self,
        a: &[Complex<f32>],
        original_width: usize,
        b: &[Complex<f32>],
        cut_width: usize,
        dst: &mut [Complex<f32>],
    ) {
        assert_eq!(b.len(), dst.len());
        assert_eq!(a.len() / original_width, dst.len() / cut_width);

        let remainder = cut_width - (cut_width / 4) * 4;

        for ((source, twiddle), dst) in b
            .chunks_exact(cut_width)
            .zip(a.chunks_exact(original_width))
            .zip(dst.chunks_exact_mut(cut_width))
        {
            let mut src_x = 0usize;
            while src_x + 8 < cut_width {
                let s0 = AvxStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x..) });
                let s1 = AvxStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x + 4..) });

                let tw0 = AvxStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x..) });
                let tw1 =
                    AvxStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x + 4..) });

                let p0 = AvxStoreF::mul_by_complex(s0, tw0);
                let p1 = AvxStoreF::mul_by_complex(s1, tw1);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                p1.write(unsafe { dst.get_unchecked_mut(src_x + 4..) });

                src_x += 8;
            }

            while src_x + 4 <= cut_width {
                let s0 = AvxStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x..) });
                let tw0 = AvxStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x..) });
                let p0 = AvxStoreF::mul_by_complex(s0, tw0);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                src_x += 4;
            }

            if remainder == 3 {
                let s0 = AvxStoreF::from_complex3(unsafe { source.get_unchecked(src_x..) });
                let tw0 = AvxStoreF::from_complex3(unsafe { twiddle.get_unchecked(src_x..) });
                let p0 = AvxStoreF::mul_by_complex(s0, tw0);

                p0.write_lo3(unsafe { dst.get_unchecked_mut(src_x..) });
            } else if remainder == 2 {
                let s0 = AvxStoreF::from_complex2(unsafe { source.get_unchecked(src_x..) });
                let tw0 = AvxStoreF::from_complex2(unsafe { twiddle.get_unchecked(src_x..) });
                let p0 = AvxStoreF::mul_by_complex(s0, tw0);

                p0.write_lo2(unsafe { dst.get_unchecked_mut(src_x..) });
            } else if remainder == 1 {
                let s0 = AvxStoreF::from_complex(unsafe { source.get_unchecked(src_x) });
                let tw0 = AvxStoreF::from_complex(unsafe { twiddle.get_unchecked(src_x) });
                let p0 = AvxStoreF::mul_by_complex(s0, tw0);

                p0.write_lo1(unsafe { dst.get_unchecked_mut(src_x..) });
            }
        }
    }
}
impl ComplexArith<f32> for AvxSpectrumArithmetic<f32> {
    fn mul(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe { self.mul_f32_avx(a, b, dst) }
    }

    fn mul_and_cut(
        &self,
        a: &[Complex<f32>],
        original_width: usize,
        b: &[Complex<f32>],
        cut_width: usize,
        dst: &mut [Complex<f32>],
    ) {
        unsafe { self.mul_and_cut_f32(a, original_width, b, cut_width, dst) }
    }

    fn mul_expand_to_complex(&self, a: &[f32], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe { self.mul_expand_f32_avx(a, b, dst) }
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
    fn mul_f64_avx(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe {
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let s0 = AvxStoreD::from_complex_ref(src);
                let s1 = AvxStoreD::from_complex_ref(src.get_unchecked(2..));
                let s2 = AvxStoreD::from_complex_ref(src.get_unchecked(4..));
                let s3 = AvxStoreD::from_complex_ref(src.get_unchecked(6..));

                let q0 = AvxStoreD::from_complex_ref(twiddle);
                let q1 = AvxStoreD::from_complex_ref(twiddle.get_unchecked(2..));
                let q2 = AvxStoreD::from_complex_ref(twiddle.get_unchecked(4..));
                let q3 = AvxStoreD::from_complex_ref(twiddle.get_unchecked(6..));

                let p0 = AvxStoreD::mul_by_complex(s0, q0);
                let p1 = AvxStoreD::mul_by_complex(s1, q1);
                let p2 = AvxStoreD::mul_by_complex(s2, q2);
                let p3 = AvxStoreD::mul_by_complex(s3, q3);

                p0.write(dst);
                p1.write(dst.get_unchecked_mut(2..));
                p2.write(dst.get_unchecked_mut(4..));
                p3.write(dst.get_unchecked_mut(6..));
            }

            let dst = dst.chunks_exact_mut(8).into_remainder();
            let a = a.chunks_exact(8).remainder();
            let b = b.chunks_exact(8).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = _mm_loadu_pd(src as *const Complex<f64> as *const f64);
                let q0 = _mm_loadu_pd(twiddle as *const Complex<f64> as *const f64);

                let p0 = _mm_fcmul_pd(s0, q0);

                _mm_storeu_pd(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_expand_f64_avx(&self, a: &[f64], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe {
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let q0 = AvxStoreD::load(src);
                let q1 = AvxStoreD::load(src.get_unchecked(4..));

                let [s0, s1] = q0.to_complex();
                let [s2, s3] = q1.to_complex();

                let q0 = AvxStoreD::from_complex_ref(twiddle);
                let q1 = AvxStoreD::from_complex_ref(twiddle.get_unchecked(2..));
                let q2 = AvxStoreD::from_complex_ref(twiddle.get_unchecked(4..));
                let q3 = AvxStoreD::from_complex_ref(twiddle.get_unchecked(6..));

                let p0 = AvxStoreD::mul_by_complex(s0, q0);
                let p1 = AvxStoreD::mul_by_complex(s1, q1);
                let p2 = AvxStoreD::mul_by_complex(s2, q2);
                let p3 = AvxStoreD::mul_by_complex(s3, q3);

                p0.write(dst);
                p1.write(dst.get_unchecked_mut(2..));
                p2.write(dst.get_unchecked_mut(4..));
                p3.write(dst.get_unchecked_mut(6..));
            }

            let dst = dst.chunks_exact_mut(8).into_remainder();
            let a = a.chunks_exact(8).remainder();
            let b = b.chunks_exact(8).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = _mm_shuffle_pd::<0b00>(_mm_load_sd(src), _mm_setzero_pd());
                let q0 = _mm_loadu_pd(twiddle as *const Complex<f64> as *const f64);

                let p0 = _mm_fcmul_pd(s0, q0);

                _mm_storeu_pd(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_conjugate_in_place_f64(&self, dst: &mut [Complex<f64>], b: &[Complex<f64>]) {
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

            let dst = dst.chunks_exact_mut(8).into_remainder();
            let b = b.chunks_exact(8).remainder();

            for (dst, twiddle) in dst.iter_mut().zip(b.iter()) {
                let s0 = _mm_loadu_pd(dst as *const Complex<f64> as *const f64);
                let q0 = _mm_loadu_pd(twiddle as *const Complex<f64> as *const f64);

                let mut p0 = _mm_fcmul_pd(s0, q0);

                p0 = _mm_xor_pd(p0, _mm256_castpd256_pd128(factors));

                _mm_storeu_pd(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn conjugate_mul_by_b_f64(
        &self,
        a: &[Complex<f64>],
        b: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) {
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

                let p0 = _mm256_fcmul_pd_conj_a(s0, q0);
                let p1 = _mm256_fcmul_pd_conj_a(s1, q1);
                let p2 = _mm256_fcmul_pd_conj_a(s2, q2);
                let p3 = _mm256_fcmul_pd_conj_a(s3, q3);

                _mm256_storeu_pd(dst.as_mut_ptr().cast(), p0);
                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p1);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), p2);
                _mm256_storeu_pd(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), p3);
            }

            let dst = dst.chunks_exact_mut(8).into_remainder();
            let a = a.chunks_exact(8).remainder();
            let b = b.chunks_exact(8).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = _mm_loadu_pd(src as *const Complex<f64> as *const f64);
                let q0 = _mm_loadu_pd(twiddle as *const Complex<f64> as *const f64);

                let p0 = _mm_fcmul_pd_conj_a(s0, q0);

                _mm_storeu_pd(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_and_cut_f64(
        &self,
        a: &[Complex<f64>],
        original_width: usize,
        b: &[Complex<f64>],
        cut_width: usize,
        dst: &mut [Complex<f64>],
    ) {
        assert_eq!(b.len(), dst.len());
        assert_eq!(a.len() / original_width, dst.len() / cut_width);

        for ((source, twiddle), dst) in b
            .chunks_exact(cut_width)
            .zip(a.chunks_exact(original_width))
            .zip(dst.chunks_exact_mut(cut_width))
        {
            let mut src_x = 0usize;
            while src_x + 4 < cut_width {
                let s0 = AvxStoreD::from_complex_ref(unsafe { source.get_unchecked(src_x..) });
                let s1 = AvxStoreD::from_complex_ref(unsafe { source.get_unchecked(src_x + 2..) });

                let tw0 = AvxStoreD::from_complex_ref(unsafe { twiddle.get_unchecked(src_x..) });
                let tw1 =
                    AvxStoreD::from_complex_ref(unsafe { twiddle.get_unchecked(src_x + 2..) });

                let p0 = AvxStoreD::mul_by_complex(s0, tw0);
                let p1 = AvxStoreD::mul_by_complex(s1, tw1);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                p1.write(unsafe { dst.get_unchecked_mut(src_x + 2..) });

                src_x += 4;
            }

            while src_x + 2 < cut_width {
                let s0 = AvxStoreD::from_complex_ref(unsafe { source.get_unchecked(src_x..) });
                let tw0 = AvxStoreD::from_complex_ref(unsafe { twiddle.get_unchecked(src_x..) });
                let p0 = AvxStoreD::mul_by_complex(s0, tw0);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                src_x += 2;
            }

            while src_x < cut_width {
                let s0 = AvxStoreD::from_complex(unsafe { source.get_unchecked(src_x) });
                let tw0 = AvxStoreD::from_complex(unsafe { twiddle.get_unchecked(src_x) });
                let p0 = AvxStoreD::mul_by_complex(s0, tw0);

                p0.write_lo(unsafe { dst.get_unchecked_mut(src_x..) });
                src_x += 1;
            }
        }
    }
}

impl ComplexArith<f64> for AvxSpectrumArithmetic<f64> {
    fn mul(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe { self.mul_f64_avx(a, b, dst) }
    }

    fn mul_and_cut(
        &self,
        a: &[Complex<f64>],
        original_width: usize,
        b: &[Complex<f64>],
        cut_width: usize,
        dst: &mut [Complex<f64>],
    ) {
        unsafe { self.mul_and_cut_f64(a, original_width, b, cut_width, dst) }
    }

    fn mul_expand_to_complex(&self, a: &[f64], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe { self.mul_expand_f64_avx(a, b, dst) }
    }

    fn mul_conjugate_in_place(&self, dst: &mut [Complex<f64>], b: &[Complex<f64>]) {
        unsafe { self.mul_conjugate_in_place_f64(dst, b) }
    }

    fn conjugate_mul_by_b(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe { self.conjugate_mul_by_b_f64(a, b, dst) }
    }
}
