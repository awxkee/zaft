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
                .chunks_exact(2)
                .zip(left.chunks_exact_mut(2))
                .zip(right.rchunks_exact_mut(2))
                .zip(left_input.chunks_exact(2))
                .zip(right_input.rchunks_exact(2))
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
