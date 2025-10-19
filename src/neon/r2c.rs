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
use crate::neon::util::{v_rotate90_f32, v_rotate90_f64, vh_rotate90_f32};
use crate::r2c::R2CTwiddlesHandler;
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) struct R2CNeonTwiddles {}

impl R2CTwiddlesHandler<f64> for R2CNeonTwiddles {
    fn handle(
        &self,
        twiddles: &[Complex<f64>],
        left: &mut [Complex<f64>],
        right: &mut [Complex<f64>],
    ) {
        unsafe {
            static ROT_270: [f64; 2] = [0.0, -0.0];
            let rot_270 = vld1q_f64(ROT_270.as_ptr().cast());
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            let rot_90 = vld1q_f64(ROT_90.as_ptr().cast());

            for ((twiddle, s_out), s_out_rev) in twiddles
                .iter()
                .zip(left.iter_mut())
                .zip(right.iter_mut().rev())
            {
                let twiddle = vld1q_f64(twiddle as *const Complex<f64> as *const f64);
                let out = vld1q_f64(s_out as *const Complex<f64> as *const f64);
                let out_rev = vld1q_f64(s_out_rev as *const Complex<f64> as *const f64);

                let sum = vaddq_f64(out, out_rev);
                let diff = vsubq_f64(out, out_rev);

                let twiddled_diff = vmulq_f64(
                    vcombine_f64(vget_low_f64(diff), vget_low_f64(diff)),
                    twiddle,
                );

                let sum_diff = vcombine_f64(vget_low_f64(sum), vget_high_f64(diff));

                let rot_270_half_sum = vreinterpretq_f64_u64(veorq_u64(
                    vreinterpretq_u64_f64(sum_diff),
                    vreinterpretq_u64_f64(rot_270),
                ));

                let rot_diff = v_rotate90_f64(twiddled_diff, rot_270);

                let output_twiddled = vfmaq_f64(
                    rot_diff,
                    vcombine_f64(vget_high_f64(sum), vget_high_f64(sum)),
                    twiddle,
                );
                let output_rot90 = vreinterpretq_f64_u64(veorq_u64(
                    vreinterpretq_u64_f64(output_twiddled),
                    vreinterpretq_u64_f64(rot_90),
                ));

                // We finally have all the data we need to write the transformed data back out where we found it.
                let v_out = vfmaq_n_f64(output_twiddled, sum_diff, 0.5);
                let v_out_rev = vfmaq_n_f64(output_rot90, rot_270_half_sum, 0.5);

                vst1q_f64(s_out as *mut Complex<f64> as *mut f64, v_out);
                vst1q_f64(s_out_rev as *mut Complex<f64> as *mut f64, v_out_rev);
            }
        }
    }
}

impl R2CTwiddlesHandler<f32> for R2CNeonTwiddles {
    fn handle(
        &self,
        twiddles: &[Complex<f32>],
        left: &mut [Complex<f32>],
        right: &mut [Complex<f32>],
    ) {
        unsafe {
            static ROT_270: [f32; 4] = [0.0, -0.0, 0.0, -0.0];
            let rot_270 = vld1q_f32(ROT_270.as_ptr().cast());
            static ROT_90: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            let rot_90 = vld1q_f32(ROT_90.as_ptr().cast());

            static DUP_FIRST_F32: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 8, 9, 10, 11, 8, 9, 10, 11];
            let dup_first_f32 = vld1q_u8(DUP_FIRST_F32.as_ptr().cast());

            for ((twiddle, s_out), s_out_rev) in twiddles
                .chunks_exact(2)
                .zip(left.chunks_exact_mut(2))
                .zip(right.chunks_exact_mut(2).rev())
            {
                let twiddle = vld1q_f32(twiddle.as_ptr().cast());
                let out = vld1q_f32(s_out.as_ptr().cast());
                let mut out_rev = vld1q_f32(s_out_rev.as_ptr().cast());
                out_rev = vcombine_f32(vget_high_f32(out_rev), vget_low_f32(out_rev));

                let sum = vaddq_f32(out, out_rev);
                let diff = vsubq_f32(out, out_rev);

                let diff_re_re =
                    vreinterpretq_f32_u8(vqtbl1q_u8(vreinterpretq_u8_f32(diff), dup_first_f32));
                let twiddled_diff = vmulq_f32(diff_re_re, twiddle);

                let diff_sum = vcombine_f32(
                    vext_f32::<1>(vget_low_f32(diff), vget_low_f32(sum)),
                    vext_f32::<1>(vget_high_f32(diff), vget_high_f32(sum)),
                );
                let sum_diff = vrev64q_f32(diff_sum);

                let rot_270_half_sum = vreinterpretq_f32_u32(veorq_u32(
                    vreinterpretq_u32_f32(sum_diff),
                    vreinterpretq_u32_f32(rot_270),
                ));

                let rot_diff = v_rotate90_f32(twiddled_diff, rot_270);

                let sum_im = vtrn2q_f32(sum, sum);

                let output_twiddled = vfmaq_f32(
                    rot_diff, sum_im, // [im, im]
                    twiddle,
                );
                let output_rot90 = vreinterpretq_f32_u32(veorq_u32(
                    vreinterpretq_u32_f32(output_twiddled),
                    vreinterpretq_u32_f32(rot_90),
                ));

                // We finally have all the data we need to write the transformed data back out where we found it.
                let v_out = vfmaq_n_f32(output_twiddled, sum_diff, 0.5);
                let v_out_rev = vfmaq_n_f32(output_rot90, rot_270_half_sum, 0.5);

                vst1q_f32(s_out.as_mut_ptr().cast(), v_out);
                vst1q_f32(
                    s_out_rev.as_mut_ptr().cast(),
                    vcombine_f32(vget_high_f32(v_out_rev), vget_low_f32(v_out_rev)),
                );
            }

            if twiddles.len() % 2 != 0 {
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
                    let twiddle = vld1_f32(twiddle as *const Complex<f32> as *const f32);
                    let out = vld1_f32(s_out as *const Complex<f32> as *const f32);
                    let out_rev = vld1_f32(s_out_rev as *const Complex<f32> as *const f32);

                    let sum = vadd_f32(out, out_rev);
                    let diff = vsub_f32(out, out_rev);

                    let diff_re_re = vreinterpret_f32_u8(vqtbl1_u8(
                        vreinterpretq_u8_f32(vcombine_f32(diff, diff)),
                        vget_low_u8(dup_first_f32),
                    ));
                    let twiddled_diff = vmul_f32(diff_re_re, twiddle);

                    let diff_sum = vext_f32::<1>(diff, sum);
                    let sum_diff = vrev64_f32(diff_sum);

                    let rot_270_half_sum = vreinterpret_f32_u32(veor_u32(
                        vreinterpret_u32_f32(sum_diff),
                        vreinterpret_u32_f32(vget_low_f32(rot_270)),
                    ));

                    let rot_diff = vh_rotate90_f32(twiddled_diff, vget_low_f32(rot_270));

                    let output_twiddled = vfma_f32(
                        rot_diff,
                        vtrn2_f32(sum, sum), // [im, im]
                        twiddle,
                    );
                    let output_rot90 = vreinterpret_f32_u32(veor_u32(
                        vreinterpret_u32_f32(output_twiddled),
                        vreinterpret_u32_f32(vget_low_f32(rot_90)),
                    ));

                    // We finally have all the data we need to write the transformed data back out where we found it.
                    let v_out = vfma_n_f32(output_twiddled, sum_diff, 0.5);
                    let v_out_rev = vfma_n_f32(output_rot90, rot_270_half_sum, 0.5);

                    vst1_f32(s_out as *mut Complex<f32> as *mut f32, v_out);
                    vst1_f32(s_out_rev as *mut Complex<f32> as *mut f32, v_out_rev);
                }
            }
        }
    }
}
