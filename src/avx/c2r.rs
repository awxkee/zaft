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
use crate::r2c::C2RTwiddlesHandler;
use num_complex::Complex;
use num_traits::MulAdd;

pub(crate) struct C2RAvxTwiddles {}

impl C2RTwiddlesHandler<f64> for C2RAvxTwiddles {
    fn handle(
        &self,
        twiddles: &[Complex<f64>],
        left_input: &[Complex<f64>],
        right_input: &[Complex<f64>],
        left: &mut [Complex<f64>],
        right: &mut [Complex<f64>],
    ) {
        unsafe {
            self.handle_impl(twiddles, left_input, right_input, left, right);
        }
    }
}

impl C2RAvxTwiddles {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn handle_impl(
        &self,
        twiddles: &[Complex<f64>],
        left_input: &[Complex<f64>],
        right_input: &[Complex<f64>],
        left: &mut [Complex<f64>],
        right: &mut [Complex<f64>],
    ) {
        let conj = AvxStoreD::conj_flag();

        for ((((twiddle, s_out), s_out_rev), left_input), right_input) in twiddles
            .chunks_exact(8)
            .zip(left.chunks_exact_mut(8))
            .zip(right.rchunks_exact_mut(8))
            .zip(left_input.chunks_exact(8))
            .zip(right_input.rchunks_exact(8))
        {
            let [twiddle_re0, twiddle_im0] = AvxStoreD::from_complex_ref(twiddle).dup_even_odds();
            let [twiddle_re1, twiddle_im1] =
                AvxStoreD::from_complex_ref(&twiddle[2..]).dup_even_odds();
            let [twiddle_re2, twiddle_im2] =
                AvxStoreD::from_complex_ref(&twiddle[4..]).dup_even_odds();
            let [twiddle_re3, twiddle_im3] =
                AvxStoreD::from_complex_ref(&twiddle[6..]).dup_even_odds();

            let twiddle_re0 = twiddle_re0.xor(conj);
            let twiddle_re1 = twiddle_re1.xor(conj);
            let twiddle_re2 = twiddle_re2.xor(conj);
            let twiddle_re3 = twiddle_re3.xor(conj);

            let out0 = AvxStoreD::from_complex_ref(left_input);
            let out1 = AvxStoreD::from_complex_ref(&left_input[2..]);
            let out2 = AvxStoreD::from_complex_ref(&left_input[4..]);
            let out3 = AvxStoreD::from_complex_ref(&left_input[6..]);

            let out_rev0 = AvxStoreD::from_complex_ref(&right_input[6..]).reverse_complex();
            let out_rev1 = AvxStoreD::from_complex_ref(&right_input[4..]).reverse_complex();
            let out_rev2 = AvxStoreD::from_complex_ref(&right_input[2..]).reverse_complex();
            let out_rev3 = AvxStoreD::from_complex_ref(right_input).reverse_complex();

            let sum0 = out0 + out_rev0;
            let sum1 = out1 + out_rev1;
            let sum2 = out2 + out_rev2;
            let sum3 = out3 + out_rev3;

            let diff0 = out0 - out_rev0;
            let diff1 = out1 - out_rev1;
            let diff2 = out2 - out_rev2;
            let diff3 = out3 - out_rev3;

            let sumdiff_blended0 = sum0.blend_real_img(diff0);
            let sumdiff_blended1 = sum1.blend_real_img(diff1);
            let sumdiff_blended2 = sum2.blend_real_img(diff2);
            let sumdiff_blended3 = sum3.blend_real_img(diff3);

            let diffsum_blended0 = diff0.blend_real_img(sum0);
            let diffsum_blended1 = diff1.blend_real_img(sum1);
            let diffsum_blended2 = diff2.blend_real_img(sum2);
            let diffsum_blended3 = diff3.blend_real_img(sum3);

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
            let [twiddle_re, twiddle_im] = AvxStoreD::from_complex_ref(twiddle).dup_even_odds();
            let twiddle_re = twiddle_re.xor(conj);
            let out = AvxStoreD::from_complex_ref(left_input);
            let out_rev = AvxStoreD::from_complex_ref(right_input).reverse_complex();

            let sum = out + out_rev;
            let diff = out - out_rev;

            let sumdiff_blended = sum.blend_real_img(diff);
            let diffsum_blended = diff.blend_real_img(sum);
            let diffsum_swapped = diffsum_blended.reverse_complex_elements();

            let twiddled_output = diffsum_blended.mul_add(twiddle_im, diffsum_swapped * twiddle_re);

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
                let [twiddle_re, twiddle_im] = AvxStoreD::from_complex(twiddle).dup_even_odds();
                let twiddle_re = twiddle_re.xor(conj);
                let out = AvxStoreD::from_complex(left_input);
                let out_rev = AvxStoreD::from_complex(right_input);

                let sum = out + out_rev;
                let diff = out - out_rev;

                let sumdiff_blended = sum.blend_real_img(diff);
                let diffsum_blended = diff.blend_real_img(sum);
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

impl C2RTwiddlesHandler<f32> for C2RAvxTwiddles {
    fn handle(
        &self,
        twiddles: &[Complex<f32>],
        left_input: &[Complex<f32>],
        right_input: &[Complex<f32>],
        left: &mut [Complex<f32>],
        right: &mut [Complex<f32>],
    ) {
        unsafe {
            self.handle_impl_f32(twiddles, left_input, right_input, left, right);
        }
    }
}

impl C2RAvxTwiddles {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn handle_impl_f32(
        &self,
        twiddles: &[Complex<f32>],
        left_input: &[Complex<f32>],
        right_input: &[Complex<f32>],
        left: &mut [Complex<f32>],
        right: &mut [Complex<f32>],
    ) {
        let conj = AvxStoreF::conj_flag();

        for ((((twiddle, s_out), s_out_rev), left_input), right_input) in twiddles
            .chunks_exact(16)
            .zip(left.chunks_exact_mut(16))
            .zip(right.rchunks_exact_mut(16))
            .zip(left_input.chunks_exact(16))
            .zip(right_input.rchunks_exact(16))
        {
            let [twiddle_re0, twiddle_im0] = AvxStoreF::from_complex_ref(twiddle).dup_even_odds();
            let [twiddle_re1, twiddle_im1] =
                AvxStoreF::from_complex_ref(&twiddle[4..]).dup_even_odds();
            let [twiddle_re2, twiddle_im2] =
                AvxStoreF::from_complex_ref(&twiddle[8..]).dup_even_odds();
            let [twiddle_re3, twiddle_im3] =
                AvxStoreF::from_complex_ref(&twiddle[12..]).dup_even_odds();

            let twiddle_re0 = twiddle_re0.xor(conj);
            let twiddle_re1 = twiddle_re1.xor(conj);
            let twiddle_re2 = twiddle_re2.xor(conj);
            let twiddle_re3 = twiddle_re3.xor(conj);

            let out0 = AvxStoreF::from_complex_ref(left_input);
            let out1 = AvxStoreF::from_complex_ref(&left_input[4..]);
            let out2 = AvxStoreF::from_complex_ref(&left_input[8..]);
            let out3 = AvxStoreF::from_complex_ref(&left_input[12..]);

            let out_rev0 = AvxStoreF::from_complex_ref(&right_input[12..]).reverse_complex();
            let out_rev1 = AvxStoreF::from_complex_ref(&right_input[8..]).reverse_complex();
            let out_rev2 = AvxStoreF::from_complex_ref(&right_input[4..]).reverse_complex();
            let out_rev3 = AvxStoreF::from_complex_ref(right_input).reverse_complex();

            let sum0 = out0 + out_rev0;
            let sum1 = out1 + out_rev1;
            let sum2 = out2 + out_rev2;
            let sum3 = out3 + out_rev3;

            let diff0 = out0 - out_rev0;
            let diff1 = out1 - out_rev1;
            let diff2 = out2 - out_rev2;
            let diff3 = out3 - out_rev3;

            let sumdiff_blended0 = sum0.blend_real_img(diff0);
            let sumdiff_blended1 = sum1.blend_real_img(diff1);
            let sumdiff_blended2 = sum2.blend_real_img(diff2);
            let sumdiff_blended3 = sum3.blend_real_img(diff3);

            let diffsum_blended0 = diff0.blend_real_img(sum0);
            let diffsum_blended1 = diff1.blend_real_img(sum1);
            let diffsum_blended2 = diff2.blend_real_img(sum2);
            let diffsum_blended3 = diff3.blend_real_img(sum3);

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
            out_fwd1.write(&mut s_out[4..]);
            out_fwd2.write(&mut s_out[8..]);
            out_fwd3.write(&mut s_out[12..]);

            out_rev0.write(&mut s_out_rev[12..]);
            out_rev1.write(&mut s_out_rev[8..]);
            out_rev2.write(&mut s_out_rev[4..]);
            out_rev3.write(s_out_rev);
        }

        let main_count = left_input.len() / 16;
        let li_remainder_start = main_count * 16;
        let ri_remainder_end = right_input.len() - main_count * 16;

        let tw0 = twiddles.chunks_exact(16).remainder();
        let l0 = &mut left[li_remainder_start..];
        let r0 = &mut right[..ri_remainder_end];
        let li0 = &left_input[li_remainder_start..];
        let ri0 = &right_input[..ri_remainder_end];

        for ((((twiddle, s_out), s_out_rev), left_input), right_input) in tw0
            .chunks_exact(4)
            .zip(l0.chunks_exact_mut(4))
            .zip(r0.rchunks_exact_mut(4))
            .zip(li0.chunks_exact(4))
            .zip(ri0.rchunks_exact(4))
        {
            let [twiddle_re, twiddle_im] = AvxStoreF::from_complex_ref(twiddle).dup_even_odds();
            let twiddle_re = twiddle_re.xor(conj);
            let out = AvxStoreF::from_complex_ref(left_input);
            let out_rev = AvxStoreF::from_complex_ref(right_input).reverse_complex();

            let sum = out + out_rev;
            let diff = out - out_rev;

            let sumdiff_blended = sum.blend_real_img(diff);
            let diffsum_blended = diff.blend_real_img(sum);
            let diffsum_swapped = diffsum_blended.reverse_complex_elements();

            let twiddled_output = diffsum_blended.mul_add(twiddle_im, diffsum_swapped * twiddle_re);

            let out_fwd = sumdiff_blended - twiddled_output;
            let out_rev = sumdiff_blended.xor(conj) + twiddled_output.xor(conj);

            let out_rev = out_rev.reverse_complex();

            out_fwd.write(s_out);
            out_rev.write(s_out_rev);
        }

        if !twiddles.len().is_multiple_of(4) {
            let rem_twiddles = twiddles.chunks_exact(4).remainder();
            let min_length = left.len().min(right.len());
            let rem_left = left.chunks_exact_mut(4).into_remainder();
            let rem_left_input = left_input.chunks_exact(4).remainder();
            let full_right_chunks = right.len() - (min_length / 4) * 4;
            let rem_right = &mut right[..full_right_chunks];
            let rem_right_input = &right_input[..full_right_chunks];

            for ((((twiddle, s_out), s_out_rev), left_input), right_input) in rem_twiddles
                .iter()
                .zip(rem_left.iter_mut())
                .zip(rem_right.iter_mut().rev())
                .zip(rem_left_input.iter())
                .zip(rem_right_input.iter().rev())
            {
                let [twiddle_re, twiddle_im] = AvxStoreF::from_complex(twiddle).dup_even_odds();
                let twiddle_re = twiddle_re.xor(conj);
                let out = AvxStoreF::from_complex(left_input);
                let out_rev = AvxStoreF::from_complex(right_input);

                let sum = out + out_rev;
                let diff = out - out_rev;

                let sumdiff_blended = sum.blend_real_img(diff);
                let diffsum_blended = diff.blend_real_img(sum);
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
