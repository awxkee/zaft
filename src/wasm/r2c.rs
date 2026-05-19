/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use crate::r2c::R2CTwiddlesHandler;
use crate::wasm::store::{WasmStoreD, WasmStoreF};
use num_complex::Complex;
use num_traits::MulAdd;
use std::arch::wasm32::u32x4;

pub(crate) struct R2CWasmTwiddles {}

impl R2CTwiddlesHandler<f64> for R2CWasmTwiddles {
    fn handle(
        &self,
        twiddles: &[Complex<f64>],
        left: &mut [Complex<f64>],
        right: &mut [Complex<f64>],
    ) {
        let conj = WasmStoreD::set_values(0.0, -0.0);

        let blend_mask = WasmStoreD::set_values(f64::from_bits(0xFFFF_FFFF_FFFF_FFFFu64), 0.0);

        for ((twiddle, s_out), s_out_rev) in twiddles
            .as_chunks::<4>()
            .0
            .iter()
            .zip(left.as_chunks_mut::<4>().0.iter_mut())
            .zip(right.as_rchunks_mut::<4>().1.iter_mut().rev())
        {
            let [twiddle_re0, twiddle_im0] = WasmStoreD::from_complex(&twiddle[0]).dup_even_odds();
            let [twiddle_re1, twiddle_im1] = WasmStoreD::from_complex(&twiddle[1]).dup_even_odds();
            let [twiddle_re2, twiddle_im2] = WasmStoreD::from_complex(&twiddle[2]).dup_even_odds();
            let [twiddle_re3, twiddle_im3] = WasmStoreD::from_complex(&twiddle[3]).dup_even_odds();

            let twiddle_re0 = twiddle_re0.xor(conj);
            let twiddle_re1 = twiddle_re1.xor(conj);
            let twiddle_re2 = twiddle_re2.xor(conj);
            let twiddle_re3 = twiddle_re3.xor(conj);

            let out0 = WasmStoreD::from_complex(&s_out[0]);
            let out1 = WasmStoreD::from_complex(&s_out[1]);
            let out2 = WasmStoreD::from_complex(&s_out[2]);
            let out3 = WasmStoreD::from_complex(&s_out[3]);

            let out_rev0 = WasmStoreD::from_complex(&s_out_rev[3]);
            let out_rev1 = WasmStoreD::from_complex(&s_out_rev[2]);
            let out_rev2 = WasmStoreD::from_complex(&s_out_rev[1]);
            let out_rev3 = WasmStoreD::from_complex(&s_out_rev[0]);

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

            let half = WasmStoreD::dup(0.5);

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

        let tw0 = twiddles.as_chunks::<4>().1;
        let l0 = &mut left[l_remainder_start..];
        let r0 = &mut right[..r_remainder_end];

        for ((twiddle, s_out), s_out_rev) in tw0.iter().zip(l0.iter_mut()).zip(r0.iter_mut().rev())
        {
            let [twiddle_re, twiddle_im] = WasmStoreD::from_complex(twiddle).dup_even_odds();
            let twiddle_re = twiddle_re.xor(conj);
            let out = WasmStoreD::from_complex(s_out);
            let out_rev = WasmStoreD::from_complex(s_out_rev);

            let sum = out + out_rev;
            let diff = out - out_rev;

            let sumdiff_blended = sum.select(diff, blend_mask);
            let diffsum_blended = diff.select(sum, blend_mask);
            let diffsum_swapped = diffsum_blended.reverse_complex_elements();

            let twiddled_output = diffsum_blended.mul_add(twiddle_im, diffsum_swapped * twiddle_re);

            let out_fwd = sumdiff_blended.mul_add(WasmStoreD::dup(0.5), twiddled_output);
            let out_rev = sumdiff_blended
                .mul_add(WasmStoreD::dup(0.5), -twiddled_output)
                .xor(conj);

            out_fwd.write_single(s_out);
            out_rev.write_single(s_out_rev);
        }
    }
}

impl R2CTwiddlesHandler<f32> for R2CWasmTwiddles {
    fn handle(
        &self,
        twiddles: &[Complex<f32>],
        left: &mut [Complex<f32>],
        right: &mut [Complex<f32>],
    ) {
        let conj = WasmStoreF::conj_flag();

        let blend_mask = WasmStoreF::raw(u32x4(0xFFFFFFFFu32, 0, 0xFFFFFFFFu32, 0));
        let right_len = right.len();
        let rls2 = &mut right[if !right_len.is_multiple_of(2) { 1 } else { 0 }..];

        for ((twiddle, s_out), s_out_rev) in twiddles
            .as_chunks::<8>()
            .0
            .iter()
            .zip(left.as_chunks_mut::<8>().0.iter_mut())
            .zip(rls2.as_rchunks_mut::<8>().1.iter_mut().rev())
        {
            let [twiddle_re0, twiddle_im0] = WasmStoreF::from_complex_ref(twiddle).dup_even_odds();
            let [twiddle_re1, twiddle_im1] =
                WasmStoreF::from_complex_ref(&twiddle[2..]).dup_even_odds();
            let [twiddle_re2, twiddle_im2] =
                WasmStoreF::from_complex_ref(&twiddle[4..]).dup_even_odds();
            let [twiddle_re3, twiddle_im3] =
                WasmStoreF::from_complex_ref(&twiddle[6..]).dup_even_odds();

            let twiddle_re0 = twiddle_re0.xor(conj);
            let twiddle_re1 = twiddle_re1.xor(conj);
            let twiddle_re2 = twiddle_re2.xor(conj);
            let twiddle_re3 = twiddle_re3.xor(conj);

            let out0 = WasmStoreF::from_complex_ref(s_out);
            let out1 = WasmStoreF::from_complex_ref(&s_out[2..]);
            let out2 = WasmStoreF::from_complex_ref(&s_out[4..]);
            let out3 = WasmStoreF::from_complex_ref(&s_out[6..]);

            let out_rev0 = WasmStoreF::from_complex_ref(&s_out_rev[6..]).reverse_complex();
            let out_rev1 = WasmStoreF::from_complex_ref(&s_out_rev[4..]).reverse_complex();
            let out_rev2 = WasmStoreF::from_complex_ref(&s_out_rev[2..]).reverse_complex();
            let out_rev3 = WasmStoreF::from_complex_ref(s_out_rev).reverse_complex();

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

            let half = WasmStoreF::dup(0.5);

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

        let tw0 = twiddles.as_chunks::<8>().1;
        let l0 = &mut left[l_remainder_start..];
        let r0 = &mut rls2[..r_remainder_end];

        for ((twiddle, s_out), s_out_rev) in tw0
            .as_chunks::<2>()
            .0
            .iter()
            .zip(l0.as_chunks_mut::<2>().0.iter_mut())
            .zip(r0.as_rchunks_mut::<2>().1.iter_mut().rev())
        {
            let [twiddle_re, twiddle_im] = WasmStoreF::from_complex_ref(twiddle).dup_even_odds();
            let twiddle_re = twiddle_re.xor(conj);
            let out = WasmStoreF::from_complex_ref(s_out);
            let out_rev = WasmStoreF::from_complex_ref(s_out_rev).reverse_complex();

            let sum = out + out_rev;
            let diff = out - out_rev;

            let sumdiff_blended = sum.select(diff, blend_mask);
            let diffsum_blended = diff.select(sum, blend_mask);
            let diffsum_swapped = diffsum_blended.reverse_complex_elements();

            let twiddled_output = diffsum_blended.mul_add(twiddle_im, diffsum_swapped * twiddle_re);

            let out_fwd = sumdiff_blended.mul_add(WasmStoreF::dup(0.5), twiddled_output);
            let out_rev = sumdiff_blended
                .mul_add(WasmStoreF::dup(0.5), -twiddled_output)
                .xor(conj);

            let out_rev = out_rev.reverse_complex();

            out_fwd.write(s_out);
            out_rev.write(s_out_rev);
        }

        if !twiddles.len().is_multiple_of(2) {
            let rem_twiddles = twiddles.as_chunks::<2>().1;
            let min_length = left.len().min(right.len());
            let rem_left = left.as_chunks_mut::<2>().1;
            let full_right_chunks = right.len() - (min_length / 2) * 2;
            let rem_right = &mut right[..full_right_chunks];

            for ((twiddle, s_out), s_out_rev) in rem_twiddles
                .iter()
                .zip(rem_left.iter_mut())
                .zip(rem_right.iter_mut().rev())
            {
                let [twiddle_re, twiddle_im] = WasmStoreF::load_complex(twiddle).dup_even_odds();
                let twiddle_re = twiddle_re.xor(conj);
                let out = WasmStoreF::load_complex(s_out);
                let out_rev = WasmStoreF::load_complex(s_out_rev);

                let sum = out + out_rev;
                let diff = out - out_rev;

                let sumdiff_blended = sum.select(diff, blend_mask);
                let diffsum_blended = diff.select(sum, blend_mask);
                let diffsum_swapped = diffsum_blended.reverse_complex_elements();

                let twiddled_output =
                    diffsum_blended.mul_add(twiddle_im, diffsum_swapped * twiddle_re);

                let out_fwd = sumdiff_blended.mul_add(WasmStoreF::dup(0.5), twiddled_output);
                let out_rev = sumdiff_blended
                    .mul_add(WasmStoreF::dup(0.5), -twiddled_output)
                    .xor(conj);

                out_fwd.write_single(s_out);
                out_rev.write_single(s_out_rev);
            }
        }
    }
}
