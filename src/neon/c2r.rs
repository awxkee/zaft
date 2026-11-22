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
        let conj = NeonStoreD::conj_flag();

        for ((((twiddle, s_out), s_out_rev), left_input), right_input) in twiddles
            .iter()
            .zip(left.iter_mut())
            .zip(right.iter_mut().rev())
            .zip(left_input.iter())
            .zip(right_input.iter().rev())
        {
            let out = NeonStoreD::from_complex(left_input);
            let out_rev = NeonStoreD::from_complex(right_input);

            // Conjugate the mirrored bin before forming the sum and difference.
            let mirrored = out_rev.xor(conj);
            let sum = out + mirrored;
            let diff = out - mirrored;

            let twiddled_output = twiddle_lanes(diff, NeonStoreD::from_complex(twiddle));

            let out_fwd = sum - twiddled_output;
            let out_rev = sum.xor(conj) + twiddled_output.xor(conj);

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
        let conj = NeonStoreF::conj_flag();

        for ((((twiddle, s_out), s_out_rev), left_input), right_input) in twiddles
            .as_chunks::<8>()
            .0
            .iter()
            .zip(left.as_chunks_mut::<8>().0.iter_mut())
            .zip(right.as_rchunks_mut::<8>().1.iter_mut().rev())
            .zip(left_input.as_chunks::<8>().0.iter())
            .zip(right_input.as_rchunks::<8>().1.iter().rev())
        {
            let [twiddle_re0, twiddle_im0] = NeonStoreF::from_complex_ref(twiddle).dup_even_odds();
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

            let mirrored0 = out_rev0.xor(conj);
            let sum0 = out0 + mirrored0;
            let diff0 = out0 - mirrored0;
            let mirrored1 = out_rev1.xor(conj);
            let sum1 = out1 + mirrored1;
            let diff1 = out1 - mirrored1;
            let mirrored2 = out_rev2.xor(conj);
            let sum2 = out2 + mirrored2;
            let diff2 = out2 - mirrored2;
            let mirrored3 = out_rev3.xor(conj);
            let sum3 = out3 + mirrored3;
            let diff3 = out3 - mirrored3;

            let rotated0 = diff0.reverse_complex_elements();
            let rotated1 = diff1.reverse_complex_elements();
            let rotated2 = diff2.reverse_complex_elements();
            let rotated3 = diff3.reverse_complex_elements();

            let twiddled_output0 = diff0.mul_add(twiddle_im0, rotated0 * twiddle_re0);
            let twiddled_output1 = diff1.mul_add(twiddle_im1, rotated1 * twiddle_re1);
            let twiddled_output2 = diff2.mul_add(twiddle_im2, rotated2 * twiddle_re2);
            let twiddled_output3 = diff3.mul_add(twiddle_im3, rotated3 * twiddle_re3);

            let out_fwd0 = sum0 - twiddled_output0;
            let out_fwd1 = sum1 - twiddled_output1;
            let out_fwd2 = sum2 - twiddled_output2;
            let out_fwd3 = sum3 - twiddled_output3;

            let out_rev0 = (sum0.xor(conj) + twiddled_output0.xor(conj)).reverse_complex();
            let out_rev1 = (sum1.xor(conj) + twiddled_output1.xor(conj)).reverse_complex();
            let out_rev2 = (sum2.xor(conj) + twiddled_output2.xor(conj)).reverse_complex();
            let out_rev3 = (sum3.xor(conj) + twiddled_output3.xor(conj)).reverse_complex();

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

        let tw0 = twiddles.as_chunks::<8>().1;
        let l0 = &mut left[li_remainder_start..];
        let r0 = &mut right[..ri_remainder_end];
        let li0 = &left_input[li_remainder_start..];
        let ri0 = &right_input[..ri_remainder_end];

        for ((((twiddle, s_out), s_out_rev), left_input), right_input) in tw0
            .as_chunks::<2>()
            .0
            .iter()
            .zip(l0.as_chunks_mut::<2>().0.iter_mut())
            .zip(r0.as_rchunks_mut::<2>().1.iter_mut().rev())
            .zip(li0.as_chunks::<2>().0.iter())
            .zip(ri0.as_rchunks::<2>().1.iter().rev())
        {
            let [twiddle_re, twiddle_im] = NeonStoreF::from_complex_ref(twiddle).dup_even_odds();
            let twiddle_re = twiddle_re.xor(conj);
            let out = NeonStoreF::from_complex_ref(left_input);
            let out_rev = NeonStoreF::from_complex_ref(right_input).reverse_complex();

            let mirrored = out_rev.xor(conj);
            let sum = out + mirrored;
            let diff = out - mirrored;
            let rotated = diff.reverse_complex_elements();

            let twiddled_output = diff.mul_add(twiddle_im, rotated * twiddle_re);

            let out_fwd = sum - twiddled_output;
            let out_rev = sum.xor(conj) + twiddled_output.xor(conj);

            let out_rev = out_rev.reverse_complex();

            out_fwd.write(s_out);
            out_rev.write(s_out_rev);
        }

        if !twiddles.len().is_multiple_of(2) {
            let rem_twiddles = twiddles.as_chunks::<2>().1;
            let min_length = left.len().min(right.len());
            let rem_left = left.as_chunks_mut::<2>().1;
            let rem_left_input = left_input.as_chunks::<2>().1;
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
                let [twiddle_re, twiddle_im] = NeonStoreF::from_complex(twiddle).dup_even_odds();
                let twiddle_re = twiddle_re.xor(conj);
                let out = NeonStoreF::from_complex(left_input);
                let out_rev = NeonStoreF::from_complex(right_input);

                let mirrored = out_rev.xor(conj);
                let sum = out + mirrored;
                let diff = out - mirrored;
                let rotated = diff.reverse_complex_elements();

                let twiddled_output = diff.mul_add(twiddle_im, rotated * twiddle_re);

                let out_fwd = sum - twiddled_output;
                let out_rev = sum.xor(conj) + twiddled_output.xor(conj);

                out_fwd.write_single(s_out);
                out_rev.write_single(s_out_rev);
            }
        }
    }
}

#[inline(always)]
fn twiddle_lanes(diff: NeonStoreD, twiddle: NeonStoreD) -> NeonStoreD {
    unsafe {
        // The rotated pair is [diff.im, -diff.re]. Multiply by the real
        // twiddle first so that only the imaginary product is fused.
        let rotated = vextq_f64::<1>(diff.v, vnegq_f64(diff.v));
        let rounded = vmulq_laneq_f64::<0>(rotated, twiddle.v);
        NeonStoreD::raw(vfmaq_laneq_f64::<1>(rounded, diff.v, twiddle.v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FftDirection, util::compute_twiddle};

    #[test]
    fn inverse_bin_pairs_preserve_rounding_and_signed_zero() {
        macro_rules! check {
            ($ty:ty) => {{
                // Run the eight-bin block, the two-bin remainder, and the
                // single-bin tail with and without the separate center bin.
                let pairs = 11;
                let values: [$ty; 8] = [0.0, -0.0, 1.25, -0.625, 3.125, -0.375, 1.1, -2.3];
                let bins = |offset: usize, count: usize| {
                    (offset..offset + count)
                        .map(|i| Complex::new(values[i % 8], values[(3 * i + i / 8) % 8]))
                        .collect::<Vec<_>>()
                };
                for center in [0, 1] {
                    let mut left_input = bins(0, pairs);
                    let mut right_input = bins(29, pairs + center);
                    for i in [0, 8, 10] {
                        let high = pairs + center - 1 - i;
                        left_input[i] = Complex::new(0.0, -0.0);
                        right_input[high] = left_input[i];
                    }
                    let guard = Complex::new(10.0, -11.0);
                    let mut left = vec![guard; pairs];
                    let mut right = vec![guard; pairs + center];
                    let length = 4 * pairs + 2 + 2 * center;
                    let twiddles: Vec<_> = (1..=pairs)
                        .map(|i| compute_twiddle::<$ty>(i, length, FftDirection::Inverse))
                        .collect();
                    C2RNeonTwiddles {}.handle(
                        &twiddles,
                        &left_input,
                        &right_input,
                        &mut left,
                        &mut right,
                    );

                    for (i, (a, twiddle)) in left_input.iter().zip(twiddles).enumerate() {
                        let high = right_input.len() - 1 - i;
                        let b = right_input[high];
                        let sum = Complex::new(a.re + b.re, a.im - b.im);
                        let diff = Complex::new(a.re - b.re, a.im + b.im);
                        let term = Complex::new(
                            diff.re.mul_add(twiddle.im, diff.im * twiddle.re),
                            diff.im.mul_add(twiddle.im, diff.re * -twiddle.re),
                        );
                        let expected = [
                            (left[i], sum - term),
                            (right[high], sum.conj() + term.conj()),
                        ];
                        for (actual, expected) in expected {
                            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
                            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
                        }
                    }
                    if center != 0 {
                        assert_eq!(right[0], guard);
                    }
                }
            }};
        }
        check!(f32);
        check!(f64);
    }
}
