/*
 * // Copyright (c) Radzivon Bartoshyk 04/2026. All rights reserved.
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
use crate::r2c::R2CTwiddlesHandler;
use num_complex::Complex;
use num_traits::MulAdd;
use std::arch::aarch64::*;

pub(crate) struct R2CNeonTwiddlesFcma {}

impl R2CTwiddlesHandler<f64> for R2CNeonTwiddlesFcma {
    fn handle(
        &self,
        twiddles: &[Complex<f64>],
        left: &mut [Complex<f64>],
        right: &mut [Complex<f64>],
    ) {
        unsafe { self.handle_impl_f64(twiddles, left, right) }
    }
}

impl R2CNeonTwiddlesFcma {
    #[target_feature(enable = "fcma")]
    fn handle_impl_f64(
        &self,
        twiddles: &[Complex<f64>],
        left: &mut [Complex<f64>],
        right: &mut [Complex<f64>],
    ) {
        let conj = NeonStoreD::set_values(0.0, -0.0);

        let blend_mask = NeonStoreD::set_values(f64::from_bits(0xFFFF_FFFF_FFFF_FFFFu64), 0.0);

        for ((twiddle, s_out), s_out_rev) in twiddles
            .chunks_exact(4)
            .zip(left.chunks_exact_mut(4))
            .zip(right.rchunks_exact_mut(4))
        {
            let twiddle0 = NeonStoreD::from_complex(&twiddle[0]);
            let twiddle1 = NeonStoreD::from_complex(&twiddle[1]);
            let twiddle2 = NeonStoreD::from_complex(&twiddle[2]);
            let twiddle3 = NeonStoreD::from_complex(&twiddle[3]);

            let out0 = NeonStoreD::from_complex(&s_out[0]);
            let out1 = NeonStoreD::from_complex(&s_out[1]);
            let out2 = NeonStoreD::from_complex(&s_out[2]);
            let out3 = NeonStoreD::from_complex(&s_out[3]);

            let out_rev0 = NeonStoreD::from_complex(&s_out_rev[3]);
            let out_rev1 = NeonStoreD::from_complex(&s_out_rev[2]);
            let out_rev2 = NeonStoreD::from_complex(&s_out_rev[1]);
            let out_rev3 = NeonStoreD::from_complex(&s_out_rev[0]);

            let sum0 = out0 + out_rev0;
            let sum1 = out1 + out_rev1;
            let sum2 = out2 + out_rev2;
            let sum3 = out3 + out_rev3;

            let diff0 = out0 - out_rev0;
            let diff1 = out1 - out_rev1;
            let diff2 = out2 - out_rev2;
            let diff3 = out3 - out_rev3;

            let sumdiff_blended0 = sum0.select(diff0, blend_mask);
            let sumdiff_blended1 = sum1.select(diff1, blend_mask);
            let sumdiff_blended2 = sum2.select(diff2, blend_mask);
            let sumdiff_blended3 = sum3.select(diff3, blend_mask);

            let diffsum_blended0 = diff0.select(sum0, blend_mask);
            let diffsum_blended1 = diff1.select(sum1, blend_mask);
            let diffsum_blended2 = diff2.select(sum2, blend_mask);
            let diffsum_blended3 = diff3.select(sum3, blend_mask);

            let dt0 = NeonStoreD::fcmul_fcma(diffsum_blended0, twiddle0);
            let dt1 = NeonStoreD::fcmul_fcma(diffsum_blended1, twiddle1);
            let dt2 = NeonStoreD::fcmul_fcma(diffsum_blended2, twiddle2);
            let dt3 = NeonStoreD::fcmul_fcma(diffsum_blended3, twiddle3);

            let twiddled_output0 = dt0.reverse_complex_elements().xor(conj);
            let twiddled_output1 = dt1.reverse_complex_elements().xor(conj);
            let twiddled_output2 = dt2.reverse_complex_elements().xor(conj);
            let twiddled_output3 = dt3.reverse_complex_elements().xor(conj);

            let half = NeonStoreD::dup(0.5);

            let out_fwd0 = sumdiff_blended0.mul_add(half, twiddled_output0);
            let out_fwd1 = sumdiff_blended1.mul_add(half, twiddled_output1);
            let out_fwd2 = sumdiff_blended2.mul_add(half, twiddled_output2);
            let out_fwd3 = sumdiff_blended3.mul_add(half, twiddled_output3);

            let out_rev0 = sumdiff_blended0.mul_add(half, -twiddled_output0).xor(conj);
            let out_rev1 = sumdiff_blended1.mul_add(half, -twiddled_output1).xor(conj);
            let out_rev2 = sumdiff_blended2.mul_add(half, -twiddled_output2).xor(conj);
            let out_rev3 = sumdiff_blended3.mul_add(half, -twiddled_output3).xor(conj);

            out_fwd0.write_single(&mut s_out[0]);
            out_fwd1.write_single(&mut s_out[1]);
            out_fwd2.write_single(&mut s_out[2]);
            out_fwd3.write_single(&mut s_out[3]);

            out_rev0.write_single(&mut s_out_rev[3]);
            out_rev1.write_single(&mut s_out_rev[2]);
            out_rev2.write_single(&mut s_out_rev[1]);
            out_rev3.write_single(&mut s_out_rev[0]);
        }

        let main_count = left.len() / 4;
        let l_remainder_start = main_count * 4;
        let r_remainder_end = right.len() - main_count * 4;

        let tw0 = twiddles.chunks_exact(4).remainder();
        let l0 = &mut left[l_remainder_start..];
        let r0 = &mut right[..r_remainder_end];

        for ((twiddle, s_out), s_out_rev) in tw0.iter().zip(l0.iter_mut()).zip(r0.iter_mut().rev())
        {
            let twiddle = NeonStoreD::from_complex(twiddle);
            let out = NeonStoreD::from_complex(s_out);
            let out_rev = NeonStoreD::from_complex(s_out_rev);

            let sum = out + out_rev;
            let diff = out - out_rev;

            let sumdiff_blended = sum.select(diff, blend_mask);
            let diffsum_blended = diff.select(sum, blend_mask);

            let dt = NeonStoreD::fcmul_fcma(diffsum_blended, twiddle);
            let twiddled_output = dt.reverse_complex_elements().xor(conj);

            let out_fwd = sumdiff_blended.mul_add(NeonStoreD::dup(0.5), twiddled_output);
            let out_rev = sumdiff_blended
                .mul_add(NeonStoreD::dup(0.5), -twiddled_output)
                .xor(conj);

            out_fwd.write_single(s_out);
            out_rev.write_single(s_out_rev);
        }
    }
}

impl R2CTwiddlesHandler<f32> for R2CNeonTwiddlesFcma {
    fn handle(
        &self,
        twiddles: &[Complex<f32>],
        left: &mut [Complex<f32>],
        right: &mut [Complex<f32>],
    ) {
        unsafe {
            self.handle_impl(twiddles, left, right);
        }
    }
}

impl R2CNeonTwiddlesFcma {
    #[target_feature(enable = "fcma")]
    fn handle_impl(
        &self,
        twiddles: &[Complex<f32>],
        left: &mut [Complex<f32>],
        right: &mut [Complex<f32>],
    ) {
        unsafe {
            static ROT_270: [f32; 4] = [0.0, -0.0, 0.0, -0.0];
            let conj = NeonStoreF::raw(vld1q_f32(ROT_270.as_ptr().cast()));

            let blend_mask = NeonStoreF::raw(vreinterpretq_f32_u32(vld1q_u32(
                [0xFFFFFFFFu32, 0, 0xFFFFFFFF, 0].as_ptr(),
            )));
            let right_len = right.len();
            let rls2 = &mut right[if !right_len.is_multiple_of(2) { 1 } else { 0 }..];

            for ((twiddle, s_out), s_out_rev) in twiddles
                .chunks_exact(8)
                .zip(left.chunks_exact_mut(8))
                .zip(rls2.rchunks_exact_mut(8))
            {
                let twiddle0 = NeonStoreF::from_complex_ref(twiddle);
                NeonStoreF::from_complex_ref(twiddle);
                let twiddle1 = NeonStoreF::from_complex_ref(&twiddle[2..]);
                let twiddle2 = NeonStoreF::from_complex_ref(&twiddle[4..]);
                let twiddle3 = NeonStoreF::from_complex_ref(&twiddle[6..]);

                let out0 = NeonStoreF::from_complex_ref(s_out);
                let out1 = NeonStoreF::from_complex_ref(&s_out[2..]);
                let out2 = NeonStoreF::from_complex_ref(&s_out[4..]);
                let out3 = NeonStoreF::from_complex_ref(&s_out[6..]);

                let out_rev0 = NeonStoreF::from_complex_ref(&s_out_rev[6..]).reverse_complex();
                let out_rev1 = NeonStoreF::from_complex_ref(&s_out_rev[4..]).reverse_complex();
                let out_rev2 = NeonStoreF::from_complex_ref(&s_out_rev[2..]).reverse_complex();
                let out_rev3 = NeonStoreF::from_complex_ref(s_out_rev).reverse_complex();

                let sum0 = out0 + out_rev0;
                let sum1 = out1 + out_rev1;
                let sum2 = out2 + out_rev2;
                let sum3 = out3 + out_rev3;

                let diff0 = out0 - out_rev0;
                let diff1 = out1 - out_rev1;
                let diff2 = out2 - out_rev2;
                let diff3 = out3 - out_rev3;

                let sumdiff_blended0 = sum0.select(diff0, blend_mask);
                let sumdiff_blended1 = sum1.select(diff1, blend_mask);
                let sumdiff_blended2 = sum2.select(diff2, blend_mask);
                let sumdiff_blended3 = sum3.select(diff3, blend_mask);

                let diffsum_blended0 = diff0.select(sum0, blend_mask);
                let diffsum_blended1 = diff1.select(sum1, blend_mask);
                let diffsum_blended2 = diff2.select(sum2, blend_mask);
                let diffsum_blended3 = diff3.select(sum3, blend_mask);

                let dt0 = NeonStoreF::fcmul_fcma(diffsum_blended0, twiddle0);
                let dt1 = NeonStoreF::fcmul_fcma(diffsum_blended1, twiddle1);
                let dt2 = NeonStoreF::fcmul_fcma(diffsum_blended2, twiddle2);
                let dt3 = NeonStoreF::fcmul_fcma(diffsum_blended3, twiddle3);

                let twiddled_output0 = dt0.reverse_complex_elements().xor(conj);
                let twiddled_output1 = dt1.reverse_complex_elements().xor(conj);
                let twiddled_output2 = dt2.reverse_complex_elements().xor(conj);
                let twiddled_output3 = dt3.reverse_complex_elements().xor(conj);

                let half = NeonStoreF::dup(0.5);

                let out_fwd0 = sumdiff_blended0.mul_add(half, twiddled_output0);
                let out_fwd1 = sumdiff_blended1.mul_add(half, twiddled_output1);
                let out_fwd2 = sumdiff_blended2.mul_add(half, twiddled_output2);
                let out_fwd3 = sumdiff_blended3.mul_add(half, twiddled_output3);

                let out_rev_final0 = sumdiff_blended0
                    .mul_add(half, -twiddled_output0)
                    .xor(conj)
                    .reverse_complex();

                let out_rev_final1 = sumdiff_blended1
                    .mul_add(half, -twiddled_output1)
                    .xor(conj)
                    .reverse_complex();

                let out_rev_final2 = sumdiff_blended2
                    .mul_add(half, -twiddled_output2)
                    .xor(conj)
                    .reverse_complex();

                let out_rev_final3 = sumdiff_blended3
                    .mul_add(half, -twiddled_output3)
                    .xor(conj)
                    .reverse_complex();

                out_fwd0.write(s_out);
                out_rev_final0.write(&mut s_out_rev[6..]);

                out_fwd1.write(&mut s_out[2..]);
                out_rev_final1.write(&mut s_out_rev[4..]);

                out_fwd2.write(&mut s_out[4..]);
                out_rev_final2.write(&mut s_out_rev[2..]);

                out_fwd3.write(&mut s_out[6..]);
                out_rev_final3.write(s_out_rev);
            }

            let main_count = left.len() / 8;
            let l_remainder_start = main_count * 8;
            let r_remainder_end = rls2.len() - main_count * 8;

            let tw0 = twiddles.chunks_exact(8).remainder();
            let l0 = &mut left[l_remainder_start..];
            let r0 = &mut rls2[..r_remainder_end];

            for ((twiddle, s_out), s_out_rev) in tw0
                .chunks_exact(2)
                .zip(l0.chunks_exact_mut(2))
                .zip(r0.rchunks_exact_mut(2))
            {
                let out = NeonStoreF::from_complex_ref(s_out);
                let out_rev = NeonStoreF::from_complex_ref(s_out_rev).reverse_complex();

                let sum = out + out_rev;
                let diff = out - out_rev;

                let sumdiff_blended = sum.select(diff, blend_mask);
                let diffsum_blended = diff.select(sum, blend_mask);

                let twiddle = NeonStoreF::from_complex_ref(twiddle);

                let dt = NeonStoreF::fcmul_fcma(diffsum_blended, twiddle);
                let twiddled_output = dt.reverse_complex_elements().xor(conj);

                let out_fwd = sumdiff_blended.mul_add(NeonStoreF::dup(0.5), twiddled_output);
                let out_rev = sumdiff_blended
                    .mul_add(NeonStoreF::dup(0.5), -twiddled_output)
                    .xor(conj);

                let out_rev = out_rev.reverse_complex();

                out_fwd.write(s_out);
                out_rev.write(s_out_rev);
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
                    let out = NeonStoreF::from_complex(s_out);
                    let out_rev = NeonStoreF::from_complex(s_out_rev);

                    let sum = out + out_rev;
                    let diff = out - out_rev;

                    let sumdiff_blended = sum.select(diff, blend_mask);
                    let diffsum_blended = diff.select(sum, blend_mask);

                    let twiddle = NeonStoreF::from_complex(twiddle);
                    let dt = NeonStoreF::fcmul_fcma(diffsum_blended, twiddle);
                    let twiddled_output = dt.reverse_complex_elements().xor(conj);

                    let out_fwd = sumdiff_blended.mul_add(NeonStoreF::dup(0.5), twiddled_output);
                    let out_rev = sumdiff_blended
                        .mul_add(NeonStoreF::dup(0.5), -twiddled_output)
                        .xor(conj);

                    out_fwd.write_single(s_out);
                    out_rev.write_single(s_out_rev);
                }
            }
        }
    }
}
