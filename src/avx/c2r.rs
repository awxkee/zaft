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
use crate::r2c::R2CTwiddlesHandler;
use num_complex::Complex;
use num_traits::MulAdd;

pub(crate) struct C2RAvxTwiddles {}

impl R2CTwiddlesHandler<f64> for C2RAvxTwiddles {
    fn handle(
        &self,
        twiddles: &[Complex<f64>],
        left: &mut [Complex<f64>],
        right: &mut [Complex<f64>],
    ) {
        unsafe {
            self.handle_impl(twiddles, left, right);
        }
    }
}

impl C2RAvxTwiddles {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn handle_impl(
        &self,
        twiddles: &[Complex<f64>],
        left: &mut [Complex<f64>],
        right: &mut [Complex<f64>],
    ) {
        let conj = AvxStoreD::set_values(0.0, -0.0, 0.0, -0.0);

        for ((twiddle, s_out), s_out_rev) in twiddles
            .chunks_exact(2)
            .zip(left.chunks_exact_mut(2))
            .zip(right.rchunks_exact_mut(2))
        {
            let [twiddle_re, twiddle_im] = AvxStoreD::from_complex_ref(twiddle).dup_even_odds();
            let twiddle_re = twiddle_re.xor(conj);
            let out = AvxStoreD::from_complex_ref(s_out);
            let out_rev = AvxStoreD::from_complex_ref(s_out_rev).reverse_complex();

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
            let full_right_chunks = right.len() - (min_length / 2) * 2;
            let rem_right = &mut right[..full_right_chunks];

            for ((twiddle, s_out), s_out_rev) in rem_twiddles
                .iter()
                .zip(rem_left.iter_mut())
                .zip(rem_right.iter_mut().rev())
            {
                let [twiddle_re, twiddle_im] = AvxStoreD::from_complex(twiddle).dup_even_odds();
                let twiddle_re = twiddle_re.xor(conj);
                let out = AvxStoreD::from_complex(s_out);
                let out_rev = AvxStoreD::from_complex(s_out_rev);

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

impl R2CTwiddlesHandler<f32> for C2RAvxTwiddles {
    fn handle(
        &self,
        twiddles: &[Complex<f32>],
        left: &mut [Complex<f32>],
        right: &mut [Complex<f32>],
    ) {
        unsafe {
            self.handle_impl_f32(twiddles, left, right);
        }
    }
}

impl C2RAvxTwiddles {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn handle_impl_f32(
        &self,
        twiddles: &[Complex<f32>],
        left: &mut [Complex<f32>],
        right: &mut [Complex<f32>],
    ) {
        let conj = AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

        for ((twiddle, s_out), s_out_rev) in twiddles
            .chunks_exact(4)
            .zip(left.chunks_exact_mut(4))
            .zip(right.rchunks_exact_mut(4))
        {
            let [twiddle_re, twiddle_im] = AvxStoreF::from_complex_ref(twiddle).dup_even_odds();
            let twiddle_re = twiddle_re.xor(conj);
            let out = AvxStoreF::from_complex_ref(s_out);
            let out_rev = AvxStoreF::from_complex_ref(s_out_rev).reverse_complex();

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
            let full_right_chunks = right.len() - (min_length / 4) * 4;
            let rem_right = &mut right[..full_right_chunks];

            for ((twiddle, s_out), s_out_rev) in rem_twiddles
                .iter()
                .zip(rem_left.iter_mut())
                .zip(rem_right.iter_mut().rev())
            {
                let [twiddle_re, twiddle_im] = AvxStoreF::from_complex(twiddle).dup_even_odds();
                let twiddle_re = twiddle_re.xor(conj);
                let out = AvxStoreF::from_complex(s_out);
                let out_rev = AvxStoreF::from_complex(s_out_rev);

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
