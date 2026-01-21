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
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::neon::util::{conj_f32, conj_f64, conjq_f32};
use crate::spectrum_arithmetic::ComplexArith;
use num_complex::Complex;
use std::arch::aarch64::*;
use std::marker::PhantomData;

pub(crate) struct NeonFcmaSpectrumArithmetic<T> {
    pub(crate) phantom_data: PhantomData<T>,
}

impl NeonFcmaSpectrumArithmetic<f32> {
    #[target_feature(enable = "fcma")]
    fn mul_f32(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe {
            let zero = vdupq_n_f32(0.);
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let s0 = NeonStoreF::from_complex_ref(src);
                let s1 = NeonStoreF::from_complex_ref(src.get_unchecked(2..));
                let s2 = NeonStoreF::from_complex_ref(src.get_unchecked(4..));
                let s3 = NeonStoreF::from_complex_ref(src.get_unchecked(6..));

                let q0 = NeonStoreF::from_complex_ref(twiddle);
                let q1 = NeonStoreF::from_complex_ref(twiddle.get_unchecked(2..));
                let q2 = NeonStoreF::from_complex_ref(twiddle.get_unchecked(4..));
                let q3 = NeonStoreF::from_complex_ref(twiddle.get_unchecked(6..));

                let p0 = NeonStoreF::fcmul_fcma(s0, q0);
                let p1 = NeonStoreF::fcmul_fcma(s1, q1);
                let p2 = NeonStoreF::fcmul_fcma(s2, q2);
                let p3 = NeonStoreF::fcmul_fcma(s3, q3);

                p0.write(dst);
                p1.write(dst.get_unchecked_mut(2..));
                p2.write(dst.get_unchecked_mut(4..));
                p3.write(dst.get_unchecked_mut(6..));
            }

            let dst = dst.chunks_exact_mut(8).into_remainder();
            let a = a.chunks_exact(8).remainder();
            let b = b.chunks_exact(8).remainder();

            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(2)
                .zip(a.chunks_exact(2))
                .zip(b.chunks_exact(2))
            {
                let s0 = NeonStoreF::from_complex_ref(src);
                let q0 = NeonStoreF::from_complex_ref(twiddle);

                let p0 = NeonStoreF::fcmul_fcma(s0, q0);

                p0.write(dst);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let a = a.chunks_exact(2).remainder();
            let b = b.chunks_exact(2).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = vld1_f32(src as *const Complex<f32> as *const f32);
                let q0 = vld1_f32(twiddle as *const Complex<f32> as *const f32);

                let p0 = vcmla_rot90_f32(vcmla_f32(vget_low_f32(zero), s0, q0), s0, q0);

                vst1_f32(dst as *mut Complex<f32> as *mut f32, p0);
            }
        }
    }

    #[target_feature(enable = "fcma")]
    fn mul_expand_to_complex_impl(&self, a: &[f32], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe {
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let q0 = NeonStoreF::load(src);
                let q1 = NeonStoreF::load(src.get_unchecked(4..));

                let [s0, s1] = q0.to_complex();
                let [s2, s3] = q1.to_complex();

                let q0 = NeonStoreF::from_complex_ref(twiddle);
                let q1 = NeonStoreF::from_complex_ref(twiddle.get_unchecked(2..));
                let q2 = NeonStoreF::from_complex_ref(twiddle.get_unchecked(4..));
                let q3 = NeonStoreF::from_complex_ref(twiddle.get_unchecked(6..));

                let p0 = NeonStoreF::mul_by_complex(s0, q0);
                let p1 = NeonStoreF::mul_by_complex(s1, q1);
                let p2 = NeonStoreF::mul_by_complex(s2, q2);
                let p3 = NeonStoreF::mul_by_complex(s3, q3);

                p0.write(dst);
                p1.write(dst.get_unchecked_mut(2..));
                p2.write(dst.get_unchecked_mut(4..));
                p3.write(dst.get_unchecked_mut(6..));
            }

            let dst = dst.chunks_exact_mut(8).into_remainder();
            let a = a.chunks_exact(8).remainder();
            let b = b.chunks_exact(8).remainder();

            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(2)
                .zip(a.chunks_exact(2))
                .zip(b.chunks_exact(2))
            {
                let s0 = NeonStoreF::load2(src).to_complex()[0];
                let q0 = NeonStoreF::from_complex_ref(twiddle);

                let p0 = NeonStoreF::mul_by_complex(s0, q0);

                p0.write(dst);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let a = a.chunks_exact(2).remainder();
            let b = b.chunks_exact(2).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = vld1_lane_f32::<0>(src as *const f32, vdup_n_f32(0.));
                let q0 = vld1_f32(twiddle as *const Complex<f32> as *const f32);

                let p0 = vcmla_rot90_f32(vcmla_f32(vdup_n_f32(0.), s0, q0), s0, q0);

                vst1_f32(dst as *mut Complex<f32> as *mut f32, p0);
            }
        }
    }

    #[target_feature(enable = "fcma")]
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

        let remainder = cut_width - (cut_width / 2) * 2;

        for ((source, twiddle), dst) in b
            .chunks_exact(cut_width)
            .zip(a.chunks_exact(original_width))
            .zip(dst.chunks_exact_mut(cut_width))
        {
            let mut src_x = 0usize;

            while src_x + 8 < cut_width {
                let s0 = NeonStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x..) });
                let s1 = NeonStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x + 2..) });
                let s2 = NeonStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x + 4..) });
                let s3 = NeonStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x + 6..) });

                let tw0 = NeonStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x..) });
                let tw1 =
                    NeonStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x + 2..) });
                let tw2 =
                    NeonStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x + 4..) });
                let tw3 =
                    NeonStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x + 6..) });

                let p0 = NeonStoreF::fcmul_fcma(s0, tw0);
                let p1 = NeonStoreF::fcmul_fcma(s1, tw1);
                let p2 = NeonStoreF::fcmul_fcma(s2, tw2);
                let p3 = NeonStoreF::fcmul_fcma(s3, tw3);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                p1.write(unsafe { dst.get_unchecked_mut(src_x + 2..) });
                p2.write(unsafe { dst.get_unchecked_mut(src_x + 4..) });
                p3.write(unsafe { dst.get_unchecked_mut(src_x + 6..) });

                src_x += 8;
            }

            while src_x + 4 < cut_width {
                let s0 = NeonStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x..) });
                let s1 = NeonStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x + 2..) });

                let tw0 = NeonStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x..) });
                let tw1 =
                    NeonStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x + 2..) });

                let p0 = NeonStoreF::fcmul_fcma(s0, tw0);
                let p1 = NeonStoreF::fcmul_fcma(s1, tw1);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                p1.write(unsafe { dst.get_unchecked_mut(src_x + 2..) });

                src_x += 4;
            }

            while src_x + 2 <= cut_width {
                let s0 = NeonStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x..) });
                let tw0 = NeonStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x..) });
                let p0 = NeonStoreF::fcmul_fcma(s0, tw0);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                src_x += 2;
            }

            if remainder == 1 {
                let s0 = NeonStoreF::from_complex(unsafe { source.get_unchecked(src_x) });
                let tw0 = NeonStoreF::from_complex(unsafe { twiddle.get_unchecked(src_x) });
                let p0 = NeonStoreF::fcmul_fcma(s0, tw0);

                unsafe {
                    p0.write_lo(dst.get_unchecked_mut(src_x..));
                }
            }
        }
    }

    #[target_feature(enable = "fcma")]
    fn mul_conjugate_in_place_f32(&self, dst: &mut [Complex<f32>], b: &[Complex<f32>]) {
        unsafe {
            let conjugate_factors = vld1q_f32([0.0f32, -0.0f32, 0.0f32, -0.0f32].as_ptr());
            let zero = vdupq_n_f32(0.);
            for (dst, twiddle) in dst.chunks_exact_mut(8).zip(b.chunks_exact(8)) {
                let s0 = vld1q_f32(dst.as_ptr().cast());
                let s1 = vld1q_f32(dst.get_unchecked(2..).as_ptr().cast());
                let s2 = vld1q_f32(dst.get_unchecked(4..).as_ptr().cast());
                let s3 = vld1q_f32(dst.get_unchecked(6..).as_ptr().cast());

                let q0 = vld1q_f32(twiddle.as_ptr().cast());
                let q1 = vld1q_f32(twiddle.get_unchecked(2..).as_ptr().cast());
                let q2 = vld1q_f32(twiddle.get_unchecked(4..).as_ptr().cast());
                let q3 = vld1q_f32(twiddle.get_unchecked(6..).as_ptr().cast());

                let mut p0 = vcmlaq_rot90_f32(vcmlaq_f32(zero, s0, q0), s0, q0);
                let mut p1 = vcmlaq_rot90_f32(vcmlaq_f32(zero, s1, q1), s1, q1);
                let mut p2 = vcmlaq_rot90_f32(vcmlaq_f32(zero, s2, q2), s2, q2);
                let mut p3 = vcmlaq_rot90_f32(vcmlaq_f32(zero, s3, q3), s3, q3);

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

                let p0 = conjq_f32(
                    vcmlaq_rot90_f32(vcmlaq_f32(zero, s0, q0), s0, q0),
                    conjugate_factors,
                );

                vst1q_f32(dst.as_mut_ptr().cast(), p0);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let b = b.chunks_exact(2).remainder();

            for (dst, twiddle) in dst.iter_mut().zip(b.iter()) {
                let s0 = vld1_f32(dst as *const Complex<f32> as *const f32);
                let q0 = vld1_f32(twiddle as *const Complex<f32> as *const f32);

                let p0 = conj_f32(
                    vcmla_rot90_f32(vcmla_f32(vget_low_f32(zero), s0, q0), s0, q0),
                    vget_low_f32(conjugate_factors),
                );

                vst1_f32(dst as *mut Complex<f32> as *mut f32, p0);
            }
        }
    }

    #[target_feature(enable = "fcma")]
    fn conjugate_mul_by_b_f32(
        &self,
        a: &[Complex<f32>],
        b: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) {
        unsafe {
            let zero = vdupq_n_f32(0.);
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

                let p0 = vcmlaq_rot270_f32(vcmlaq_f32(zero, s0, q0), s0, q0);
                let p1 = vcmlaq_rot270_f32(vcmlaq_f32(zero, s1, q1), s1, q1);
                let p2 = vcmlaq_rot270_f32(vcmlaq_f32(zero, s2, q2), s2, q2);
                let p3 = vcmlaq_rot270_f32(vcmlaq_f32(zero, s3, q3), s3, q3);

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

                let p0 = vcmlaq_rot270_f32(vcmlaq_f32(zero, s0, q0), s0, q0);

                vst1q_f32(dst.as_mut_ptr().cast(), p0);
            }

            let dst = dst.chunks_exact_mut(2).into_remainder();
            let a = a.chunks_exact(2).remainder();
            let b = b.chunks_exact(2).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = vld1_f32(src as *const Complex<f32> as *const f32);
                let q0 = vld1_f32(twiddle as *const Complex<f32> as *const f32);

                let p0 = vcmla_rot270_f32(vcmla_f32(vget_low_f32(zero), s0, q0), s0, q0);

                vst1_f32(dst as *mut Complex<f32> as *mut f32, p0);
            }
        }
    }
}

impl ComplexArith<f32> for NeonFcmaSpectrumArithmetic<f32> {
    fn mul(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        unsafe { self.mul_f32(a, b, dst) }
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
        unsafe { self.mul_expand_to_complex_impl(a, b, dst) }
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

impl NeonFcmaSpectrumArithmetic<f64> {
    #[target_feature(enable = "fcma")]
    fn mul_f64(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe {
            let zero = vdupq_n_f64(0.);
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

                let p0 = vcmlaq_rot90_f64(vcmlaq_f64(zero, s0, q0), s0, q0);
                let p1 = vcmlaq_rot90_f64(vcmlaq_f64(zero, s1, q1), s1, q1);
                let p2 = vcmlaq_rot90_f64(vcmlaq_f64(zero, s2, q2), s2, q2);
                let p3 = vcmlaq_rot90_f64(vcmlaq_f64(zero, s3, q3), s3, q3);

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

                let p0 = vcmlaq_rot90_f64(vcmlaq_f64(zero, s0, q0), s0, q0);

                vst1q_f64(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }

    #[target_feature(enable = "fcma")]
    fn mul_expand_to_complex_impl(&self, a: &[f64], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe {
            for ((dst, src), twiddle) in dst
                .chunks_exact_mut(4)
                .zip(a.chunks_exact(4))
                .zip(b.chunks_exact(4))
            {
                let q0 = NeonStoreD::load(src);
                let q2 = NeonStoreD::load(src.get_unchecked(2..));

                let [s0, s1] = q0.to_complex();
                let [s2, s3] = q2.to_complex();

                let q0 = NeonStoreD::from_complex_ref(twiddle);
                let q1 = NeonStoreD::from_complex_ref(twiddle.get_unchecked(1..));
                let q2 = NeonStoreD::from_complex_ref(twiddle.get_unchecked(2..));
                let q3 = NeonStoreD::from_complex_ref(twiddle.get_unchecked(3..));

                let p0 = NeonStoreD::fcmul_fcma(s0, q0);
                let p1 = NeonStoreD::fcmul_fcma(s1, q1);
                let p2 = NeonStoreD::fcmul_fcma(s2, q2);
                let p3 = NeonStoreD::fcmul_fcma(s3, q3);

                p0.write(dst);
                p1.write(dst.get_unchecked_mut(1..));
                p2.write(dst.get_unchecked_mut(2..));
                p3.write(dst.get_unchecked_mut(3..));
            }

            let dst = dst.chunks_exact_mut(4).into_remainder();
            let a = a.chunks_exact(4).remainder();
            let b = b.chunks_exact(4).remainder();

            for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
                let s0 = NeonStoreD::load1_ptr(src);
                let q0 = NeonStoreD::from_complex(twiddle);

                let p0 = NeonStoreD::fcmul_fcma(s0, q0);
                p0.write_single(dst);
            }
        }
    }

    #[target_feature(enable = "fcma")]
    fn mul_conjugate_in_place_f64(&self, dst: &mut [Complex<f64>], b: &[Complex<f64>]) {
        unsafe {
            let conjugate_factors = vld1q_f64([0.0f64, -0.0f64].as_ptr());
            let zero = vdupq_n_f64(0.);
            for (dst, twiddle) in dst.chunks_exact_mut(4).zip(b.chunks_exact(4)) {
                let s0 = vld1q_f64(dst.as_ptr().cast());
                let s1 = vld1q_f64(dst.get_unchecked(1..).as_ptr().cast());
                let s2 = vld1q_f64(dst.get_unchecked(2..).as_ptr().cast());
                let s3 = vld1q_f64(dst.get_unchecked(3..).as_ptr().cast());

                let q0 = vld1q_f64(twiddle.as_ptr().cast());
                let q1 = vld1q_f64(twiddle.get_unchecked(1..).as_ptr().cast());
                let q2 = vld1q_f64(twiddle.get_unchecked(2..).as_ptr().cast());
                let q3 = vld1q_f64(twiddle.get_unchecked(3..).as_ptr().cast());

                let mut p0 = vcmlaq_rot90_f64(vcmlaq_f64(zero, s0, q0), s0, q0);
                let mut p1 = vcmlaq_rot90_f64(vcmlaq_f64(zero, s1, q1), s1, q1);
                let mut p2 = vcmlaq_rot90_f64(vcmlaq_f64(zero, s2, q2), s2, q2);
                let mut p3 = vcmlaq_rot90_f64(vcmlaq_f64(zero, s3, q3), s3, q3);

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

                let p0 = conj_f64(
                    vcmlaq_rot90_f64(vcmlaq_f64(zero, s0, q0), s0, q0),
                    conjugate_factors,
                );

                vst1q_f64(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }

    #[target_feature(enable = "fcma")]
    fn conjugate_mul_by_b_f64(
        &self,
        a: &[Complex<f64>],
        b: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) {
        unsafe {
            let zero = vdupq_n_f64(0.);
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

                let p0 = vcmlaq_rot270_f64(vcmlaq_f64(zero, s0, q0), s0, q0);
                let p1 = vcmlaq_rot270_f64(vcmlaq_f64(zero, s1, q1), s1, q1);
                let p2 = vcmlaq_rot270_f64(vcmlaq_f64(zero, s2, q2), s2, q2);
                let p3 = vcmlaq_rot270_f64(vcmlaq_f64(zero, s3, q3), s3, q3);

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

                let p0 = vcmlaq_rot270_f64(vcmlaq_f64(zero, s0, q0), s0, q0);

                vst1q_f64(dst as *mut Complex<f64> as *mut f64, p0);
            }
        }
    }

    #[target_feature(enable = "fcma")]
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
            while src_x + 2 < cut_width {
                let s0 = NeonStoreD::from_complex_ref(unsafe { source.get_unchecked(src_x..) });
                let s1 = NeonStoreD::from_complex_ref(unsafe { source.get_unchecked(src_x + 1..) });

                let tw0 = NeonStoreD::from_complex_ref(unsafe { twiddle.get_unchecked(src_x..) });
                let tw1 =
                    NeonStoreD::from_complex_ref(unsafe { twiddle.get_unchecked(src_x + 1..) });

                let p0 = NeonStoreD::fcmul_fcma(s0, tw0);
                let p1 = NeonStoreD::fcmul_fcma(s1, tw1);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                p1.write(unsafe { dst.get_unchecked_mut(src_x + 1..) });

                src_x += 2;
            }

            while src_x < cut_width {
                let s0 = NeonStoreD::from_complex(unsafe { source.get_unchecked(src_x) });
                let tw0 = NeonStoreD::from_complex(unsafe { twiddle.get_unchecked(src_x) });
                let p0 = NeonStoreD::fcmul_fcma(s0, tw0);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                src_x += 1;
            }
        }
    }
}

impl ComplexArith<f64> for NeonFcmaSpectrumArithmetic<f64> {
    fn mul(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe { self.mul_f64(a, b, dst) }
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
        unsafe { self.mul_expand_to_complex_impl(a, b, dst) }
    }

    fn mul_conjugate_in_place(&self, dst: &mut [Complex<f64>], b: &[Complex<f64>]) {
        unsafe {
            self.mul_conjugate_in_place_f64(dst, b);
        }
    }

    fn conjugate_mul_by_b(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe {
            self.conjugate_mul_by_b_f64(a, b, dst);
        }
    }
}
