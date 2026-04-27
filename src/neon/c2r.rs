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
use crate::r2c::C2RTwiddlesHandler;
use num_complex::Complex;
use num_traits::MulAdd;
use std::arch::aarch64::*;

pub(crate) struct C2RNeonTwiddles {}

impl C2RTwiddlesHandler<f64> for C2RNeonTwiddles {
    fn handle(
        &self,
        twiddles: &[Complex<f64>],
        left_input: &[Complex<f64>],
        right_input: &[Complex<f64>],
        left: &mut [Complex<f64>],
        right: &mut [Complex<f64>],
    ) {
        let conj = NeonStoreD::set_values(0.0, -0.0);
        let blend_mask = NeonStoreD::set_values(f64::from_bits(0xFFFF_FFFF_FFFF_FFFFu64), 0.0);

        for ((((twiddle, s_out), s_out_rev), left_input), right_input) in twiddles
            .iter()
            .zip(left.iter_mut())
            .zip(right.iter_mut().rev())
            .zip(left_input.iter())
            .zip(right_input.iter().rev())
        {
            let [twiddle_re, twiddle_im] = NeonStoreD::from_complex(twiddle).dup_even_odds();
            let twiddle_re = twiddle_re.xor(conj);
            let out = NeonStoreD::from_complex(left_input);
            let out_rev = NeonStoreD::from_complex(right_input);

            let sum = out + out_rev;
            let diff = out - out_rev;

            let sumdiff_blended = sum.select(diff, blend_mask);
            let diffsum_blended = diff.select(sum, blend_mask);
            let diffsum_swapped = diffsum_blended.reverse_complex_elements();

            let twiddled_output = diffsum_blended.mul_add(twiddle_im, diffsum_swapped * twiddle_re);

            let out_fwd = sumdiff_blended - twiddled_output;
            let out_rev = sumdiff_blended.xor(conj) + twiddled_output.xor(conj);

            out_fwd.write_single(s_out);
            out_rev.write_single(s_out_rev);
        }
    }
}

impl C2RTwiddlesHandler<f32> for C2RNeonTwiddles {
    fn handle(
        &self,
        twiddles: &[Complex<f32>],
        left_input: &[Complex<f32>],
        right_input: &[Complex<f32>],
        left: &mut [Complex<f32>],
        right: &mut [Complex<f32>],
    ) {
        unsafe {
            let conj = NeonStoreF::conj_flag();

            let blend_mask = NeonStoreF::raw(vreinterpretq_f32_u32(vld1q_u32(
                [0xFFFFFFFFu32, 0, 0xFFFFFFFF, 0].as_ptr(),
            )));

            let _right_len = right.len();

            for ((((twiddle, s_out), s_out_rev), left_input), right_input) in twiddles
                .chunks_exact(8)
                .zip(left.chunks_exact_mut(8))
                .zip(right.rchunks_exact_mut(8))
                .zip(left_input.chunks_exact(8))
                .zip(right_input.rchunks_exact(8))
            {
                let [twiddle_re0, twiddle_im0] =
                    NeonStoreF::from_complex_ref(twiddle).dup_even_odds();
                let [twiddle_re1, twiddle_im1] =
                    NeonStoreF::from_complex_ref(&twiddle[2..]).dup_even_odds();
                let [twiddle_re2, twiddle_im2] =
                    NeonStoreF::from_complex_ref(&twiddle[4..]).dup_even_odds();
                let [twiddle_re3, twiddle_im3] =
                    NeonStoreF::from_complex_ref(&twiddle[6..]).dup_even_odds();

                let twiddle_re0 = twiddle_re0.xor(conj);
                let twiddle_re1 = twiddle_re1.xor(conj);
                let twiddle_re2 = twiddle_re2.xor(conj);
                let twiddle_re3 = twiddle_re3.xor(conj);

                let out0 = NeonStoreF::from_complex_ref(left_input);
                let out1 = NeonStoreF::from_complex_ref(&left_input[2..]);
                let out2 = NeonStoreF::from_complex_ref(&left_input[4..]);
                let out3 = NeonStoreF::from_complex_ref(&left_input[6..]);

                let out_rev0 = NeonStoreF::from_complex_ref(&right_input[6..]).reverse_complex();
                let out_rev1 = NeonStoreF::from_complex_ref(&right_input[4..]).reverse_complex();
                let out_rev2 = NeonStoreF::from_complex_ref(&right_input[2..]).reverse_complex();
                let out_rev3 = NeonStoreF::from_complex_ref(right_input).reverse_complex();

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

                let diffsum_swapped0 = diffsum_blended0.reverse_complex_elements();
                let diffsum_swapped1 = diffsum_blended1.reverse_complex_elements();
                let diffsum_swapped2 = diffsum_blended2.reverse_complex_elements();
                let diffsum_swapped3 = diffsum_blended3.reverse_complex_elements();

                let twiddled_output0 =
                    diffsum_blended0.mul_add(twiddle_im0, diffsum_swapped0 * twiddle_re0);
                let twiddled_output1 =
                    diffsum_blended1.mul_add(twiddle_im1, diffsum_swapped1 * twiddle_re1);
                let twiddled_output2 =
                    diffsum_blended2.mul_add(twiddle_im2, diffsum_swapped2 * twiddle_re2);
                let twiddled_output3 =
                    diffsum_blended3.mul_add(twiddle_im3, diffsum_swapped3 * twiddle_re3);

                let out_fwd0 = sumdiff_blended0 - twiddled_output0;
                let out_fwd1 = sumdiff_blended1 - twiddled_output1;
                let out_fwd2 = sumdiff_blended2 - twiddled_output2;
                let out_fwd3 = sumdiff_blended3 - twiddled_output3;

                let out_rev0 =
                    (sumdiff_blended0.xor(conj) + twiddled_output0.xor(conj)).reverse_complex();
                let out_rev1 =
                    (sumdiff_blended1.xor(conj) + twiddled_output1.xor(conj)).reverse_complex();
                let out_rev2 =
                    (sumdiff_blended2.xor(conj) + twiddled_output2.xor(conj)).reverse_complex();
                let out_rev3 =
                    (sumdiff_blended3.xor(conj) + twiddled_output3.xor(conj)).reverse_complex();

                out_fwd0.write(s_out);
                out_fwd1.write(&mut s_out[2..]);
                out_fwd2.write(&mut s_out[4..]);
                out_fwd3.write(&mut s_out[6..]);

                out_rev0.write(&mut s_out_rev[6..]);
                out_rev1.write(&mut s_out_rev[4..]);
                out_rev2.write(&mut s_out_rev[2..]);
                out_rev3.write(s_out_rev);
            }

            let main_count = left_input.len() / 8;
            let li_remainder_start = main_count * 8;
            let ri_remainder_end = right_input.len() - main_count * 8;

            let tw0 = twiddles.chunks_exact(8).remainder();
            let l0 = &mut left[li_remainder_start..];
            let r0 = &mut right[..ri_remainder_end];
            let li0 = &left_input[li_remainder_start..];
            let ri0 = &right_input[..ri_remainder_end];

            for ((((twiddle, s_out), s_out_rev), left_input), right_input) in tw0
                .chunks_exact(2)
                .zip(l0.chunks_exact_mut(2))
                .zip(r0.rchunks_exact_mut(2))
                .zip(li0.chunks_exact(2))
                .zip(ri0.rchunks_exact(2))
            {
                let [twiddle_re, twiddle_im] =
                    NeonStoreF::from_complex_ref(twiddle).dup_even_odds();
                let twiddle_re = twiddle_re.xor(conj);
                let out = NeonStoreF::from_complex_ref(left_input);
                let out_rev = NeonStoreF::from_complex_ref(right_input).reverse_complex();

                let sum = out + out_rev;
                let diff = out - out_rev;

                let sumdiff_blended = sum.select(diff, blend_mask);
                let diffsum_blended = diff.select(sum, blend_mask);
                let diffsum_swapped = diffsum_blended.reverse_complex_elements();

                let twiddled_output =
                    diffsum_blended.mul_add(twiddle_im, diffsum_swapped * twiddle_re);

                let out_fwd = sumdiff_blended - twiddled_output;
                let out_rev = sumdiff_blended.xor(conj) + twiddled_output.xor(conj);

                let out_rev = out_rev.reverse_complex();

                out_fwd.write(s_out);
                out_rev.write(s_out_rev);
            }

            if !twiddles.len().is_multiple_of(2) {
                let rem_twiddles = twiddles.chunks_exact(2).remainder();
                let min_length = left.len().min(right.len());
                let rem_left = left.chunks_exact_mut(2).into_remainder();
                let rem_left_input = left_input.chunks_exact(2).remainder();
                let full_right_chunks = right.len() - (min_length / 2) * 2;
                let rem_right = &mut right[..full_right_chunks];
                let rem_right_input = &right_input[..full_right_chunks];

                for ((((twiddle, s_out), s_out_rev), left_input), right_input) in rem_twiddles
                    .iter()
                    .zip(rem_left.iter_mut())
                    .zip(rem_right.iter_mut().rev())
                    .zip(rem_left_input.iter())
                    .zip(rem_right_input.iter().rev())
                {
                    let [twiddle_re, twiddle_im] =
                        NeonStoreF::from_complex(twiddle).dup_even_odds();
                    let twiddle_re = twiddle_re.xor(conj);
                    let out = NeonStoreF::from_complex(left_input);
                    let out_rev = NeonStoreF::from_complex(right_input);

                    let sum = out + out_rev;
                    let diff = out - out_rev;

                    let sumdiff_blended = sum.select(diff, blend_mask);
                    let diffsum_blended = diff.select(sum, blend_mask);
                    let diffsum_swapped = diffsum_blended.reverse_complex_elements();

                    let twiddled_output =
                        diffsum_blended.mul_add(twiddle_im, diffsum_swapped * twiddle_re);

                    let out_fwd = sumdiff_blended - twiddled_output;
                    let out_rev = sumdiff_blended.xor(conj) + twiddled_output.xor(conj);

                    out_fwd.write_single(s_out);
                    out_rev.write_single(s_out_rev);
                }
            }
        }
    }
}
