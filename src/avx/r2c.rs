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
use crate::FftDirection;
use crate::avx::rotate::AvxRotate;
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_unpackhi_ps64, _mm_unpackhilo_ps64,
    _mm_unpacklo_ps64, shuffle,
};
use crate::r2c::R2CTwiddlesHandler;
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct R2CAvxTwiddles {}

impl R2CAvxTwiddles {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn handle_f64(
        &self,
        twiddles: &[Complex<f64>],
        left: &mut [Complex<f64>],
        right: &mut [Complex<f64>],
    ) {
        unsafe {
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            let rot_90 = _mm_loadu_pd(ROT_90.as_ptr().cast());

            let rotate = AvxRotate::<f64>::new(FftDirection::Forward);

            for ((twiddle, s_out), s_out_rev) in twiddles
                .iter()
                .zip(left.iter_mut())
                .zip(right.iter_mut().rev())
            {
                let twiddle = _mm_loadu_pd(twiddle as *const Complex<f64> as *const f64);
                let out = _mm_loadu_pd(s_out as *const Complex<f64> as *const f64);
                let out_rev = _mm_loadu_pd(s_out_rev as *const Complex<f64> as *const f64);

                let sum = _mm_add_pd(out, out_rev);
                let diff = _mm_sub_pd(out, out_rev);

                let twiddled_diff = _mm_mul_pd(_mm_unpacklo_pd(diff, diff), twiddle);

                let sum_diff = _mm_shuffle_pd::<0b10>(sum, diff);

                let rot_270_half_sum =
                    _mm_xor_pd(sum_diff, _mm256_castpd256_pd128(rotate.rot_flag));

                let rot_diff = rotate.rotate_m128d(twiddled_diff);

                let output_twiddled = _mm_fmadd_pd(_mm_unpackhi_pd(sum, sum), twiddle, rot_diff);
                let output_rot90 = _mm_xor_pd(output_twiddled, rot_90);

                // We finally have all the data we need to write the transformed data back out where we found it.
                let v_out = _mm_fmadd_pd(sum_diff, _mm_set1_pd(0.5), output_twiddled);
                let v_out_rev = _mm_fmadd_pd(rot_270_half_sum, _mm_set1_pd(0.5), output_rot90);

                _mm_storeu_pd(s_out as *mut Complex<f64> as *mut f64, v_out);
                _mm_storeu_pd(s_out_rev as *mut Complex<f64> as *mut f64, v_out_rev);
            }
        }
    }
}

impl R2CTwiddlesHandler<f64> for R2CAvxTwiddles {
    fn handle(
        &self,
        twiddles: &[Complex<f64>],
        left: &mut [Complex<f64>],
        right: &mut [Complex<f64>],
    ) {
        unsafe {
            self.handle_f64(twiddles, left, right);
        }
    }
}

impl R2CAvxTwiddles {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn handle_f32(
        &self,
        twiddles: &[Complex<f32>],
        left: &mut [Complex<f32>],
        right: &mut [Complex<f32>],
    ) {
        unsafe {
            static ROT_90: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            let rot_90 = _mm_loadu_ps(ROT_90.as_ptr().cast());

            let rotate = AvxRotate::<f32>::new(FftDirection::Forward);

            let right_len = right.len();
            let rls2 = &mut right[if !right_len.is_multiple_of(2) { 1 } else { 0 }..];

            for ((twiddle, s_out), s_out_rev) in twiddles
                .chunks_exact(2)
                .zip(left.chunks_exact_mut(2))
                .zip(rls2.chunks_exact_mut(2).rev())
            {
                let twiddle = _mm_loadu_ps(twiddle.as_ptr().cast());
                let out = _mm_loadu_ps(s_out.as_ptr().cast());
                let mut out_rev = _mm_loadu_ps(s_out_rev.as_ptr().cast());
                out_rev = _mm_unpackhilo_ps64(out_rev, out_rev);

                let sum = _mm_add_ps(out, out_rev);
                let diff = _mm_sub_ps(out, out_rev);

                let diff_re_re = _mm_shuffle_ps::<{ shuffle(2, 2, 0, 0) }>(diff, diff);
                let twiddled_diff = _mm_mul_ps(diff_re_re, twiddle);

                let perm_lo = _mm_unpacklo_ps64(diff, sum);
                let perm_hi = _mm_unpackhi_ps64(diff, sum);
                let diff_sw = _mm_unpacklo_ps64(
                    _mm_shuffle_ps::<{ shuffle(2, 0, 2, 1) }>(perm_lo, perm_lo),
                    _mm_shuffle_ps::<{ shuffle(2, 0, 2, 1) }>(perm_hi, perm_hi),
                );
                let sum_diff = _mm_shuffle_ps::<{ shuffle(2, 3, 0, 1) }>(diff_sw, diff_sw);

                let rot_270_half_sum = _mm_xor_ps(
                    sum_diff,
                    _mm256_castps256_ps128(_mm256_castpd_ps(rotate.rot_flag)),
                );

                let rot_diff = rotate.rotate_m128(twiddled_diff);

                let sum_im = _mm_shuffle_ps::<{ shuffle(3, 3, 1, 1) }>(sum, sum);

                let output_twiddled = _mm_fmadd_ps(
                    sum_im, // [im, im]
                    twiddle, rot_diff,
                );
                let output_rot90 = _mm_xor_ps(output_twiddled, rot_90);

                // We finally have all the data we need to write the transformed data back out where we found it.
                let v_out = _mm_fmadd_ps(sum_diff, _mm_set1_ps(0.5), output_twiddled);
                let v_out_rev = _mm_fmadd_ps(rot_270_half_sum, _mm_set1_ps(0.5), output_rot90);

                _mm_storeu_ps(s_out.as_mut_ptr().cast(), v_out);
                _mm_storeu_ps(
                    s_out_rev.as_mut_ptr().cast(),
                    _mm_unpackhilo_ps64(v_out_rev, v_out_rev),
                );
            }

            if !twiddles.len().is_multiple_of(2) {
                let rem_twiddles = twiddles.chunks_exact(2).remainder();
                let min_length = left.len().min(right.len());
                let rem_left = left.chunks_exact_mut(2).into_remainder();
                let full_right_chunks = right.len() - (min_length / 2) * 2;
                let rem_right = &mut right[..full_right_chunks];

                for ((twiddle, s_out), s_out_rev) in rem_twiddles
                    .iter()
                    .zip(rem_left.iter_mut())
                    .zip(rem_right.iter_mut().rev())
                {
                    let twiddle = _m128s_load_f32x2(twiddle as *const Complex<f32> as *const _);
                    let out = _m128s_load_f32x2(s_out as *const Complex<f32> as *const _);
                    let out_rev = _m128s_load_f32x2(s_out_rev as *const Complex<f32> as *const _);

                    let sum = _mm_add_ps(out, out_rev);
                    let diff = _mm_sub_ps(out, out_rev);

                    let diff_re_re = _mm_shuffle_ps::<{ shuffle(2, 2, 0, 0) }>(diff, diff);
                    let twiddled_diff = _mm_mul_ps(diff_re_re, twiddle);

                    let diff_sum = _mm_unpacklo_ps64(diff, sum);
                    let sum_diff = _mm_shuffle_ps::<{ shuffle(2, 0, 1, 2) }>(diff_sum, diff_sum);

                    let rot_270_half_sum = _mm_xor_ps(
                        sum_diff,
                        _mm256_castps256_ps128(_mm256_castpd_ps(rotate.rot_flag)),
                    );

                    let rot_diff = rotate.rotate_m128(twiddled_diff);

                    let sum_im = _mm_shuffle_ps::<{ shuffle(3, 3, 1, 1) }>(sum, sum);

                    let output_twiddled = _mm_fmadd_ps(
                        sum_im, // [im, im]
                        twiddle, rot_diff,
                    );
                    let output_rot90 = _mm_xor_ps(output_twiddled, rot_90);

                    // We finally have all the data we need to write the transformed data back out where we found it.
                    let v_out = _mm_fmadd_ps(sum_diff, _mm_set1_ps(0.5), output_twiddled);
                    let v_out_rev = _mm_fmadd_ps(rot_270_half_sum, _mm_set1_ps(0.5), output_rot90);

                    _m128s_store_f32x2(s_out as *mut Complex<f32> as *mut _, v_out);
                    _m128s_store_f32x2(s_out_rev as *mut Complex<f32> as *mut _, v_out_rev);
                }
            }
        }
    }
}

impl R2CTwiddlesHandler<f32> for R2CAvxTwiddles {
    fn handle(
        &self,
        twiddles: &[Complex<f32>],
        left: &mut [Complex<f32>],
        right: &mut [Complex<f32>],
    ) {
        unsafe {
            self.handle_f32(twiddles, left, right);
        }
    }
}
