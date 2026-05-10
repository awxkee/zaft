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
use crate::complex_fma::c_mul_fast;
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::spectrum_arithmetic::ComplexArith;
use num_complex::Complex;
use std::marker::PhantomData;

pub(crate) struct NeonFcmaSpectrumArithmetic<T> {
    pub(crate) phantom_data: PhantomData<T>,
}

impl NeonFcmaSpectrumArithmetic<f32> {
    #[target_feature(enable = "fcma")]
    fn mul_f32(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        for ((dst, src), twiddle) in dst
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(a.as_chunks::<8>().0.iter())
            .zip(b.as_chunks::<8>().0.iter())
        {
            let s0 = NeonStoreF::from_complex_ref(src);
            let s1 = NeonStoreF::from_complex_ref(&src[2..]);
            let s2 = NeonStoreF::from_complex_ref(&src[4..]);
            let s3 = NeonStoreF::from_complex_ref(&src[6..]);

            let q0 = NeonStoreF::from_complex_ref(twiddle);
            let q1 = NeonStoreF::from_complex_ref(&twiddle[2..]);
            let q2 = NeonStoreF::from_complex_ref(&twiddle[4..]);
            let q3 = NeonStoreF::from_complex_ref(&twiddle[6..]);

            let p0 = NeonStoreF::fcmul_fcma(s0, q0);
            let p1 = NeonStoreF::fcmul_fcma(s1, q1);
            let p2 = NeonStoreF::fcmul_fcma(s2, q2);
            let p3 = NeonStoreF::fcmul_fcma(s3, q3);

            p0.write(dst);
            p1.write(&mut dst[2..]);
            p2.write(&mut dst[4..]);
            p3.write(&mut dst[6..]);
        }

        let dst = dst.as_chunks_mut::<8>().1;
        let a = a.as_chunks::<8>().1;
        let b = b.as_chunks::<8>().1;

        for ((dst, src), twiddle) in dst
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(a.as_chunks::<2>().0.iter())
            .zip(b.as_chunks::<2>().0.iter())
        {
            let s0 = NeonStoreF::from_complex_ref(src);
            let q0 = NeonStoreF::from_complex_ref(twiddle);

            let p0 = NeonStoreF::fcmul_fcma(s0, q0);

            p0.write(dst);
        }

        let dst = dst.as_chunks_mut::<2>().1;
        let a = a.as_chunks::<2>().1;
        let b = b.as_chunks::<2>().1;

        for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            let s0 = NeonStoreF::from_complex(src);
            let q0 = NeonStoreF::from_complex(twiddle);

            let p0 = NeonStoreF::fcmul_fcma(s0, q0);

            p0.write_single(dst);
        }
    }

    #[target_feature(enable = "fcma")]
    fn mul_expand_to_complex_impl(&self, a: &[f32], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        for ((dst, src), twiddle) in dst
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(a.as_chunks::<8>().0.iter())
            .zip(b.as_chunks::<8>().0.iter())
        {
            let q0 = NeonStoreF::load(src);
            let q1 = NeonStoreF::load(&src[4..]);

            let [s0, s1] = q0.to_complex();
            let [s2, s3] = q1.to_complex();

            let q0 = NeonStoreF::from_complex_ref(twiddle);
            let q1 = NeonStoreF::from_complex_ref(&twiddle[2..]);
            let q2 = NeonStoreF::from_complex_ref(&twiddle[4..]);
            let q3 = NeonStoreF::from_complex_ref(&twiddle[6..]);

            let p0 = NeonStoreF::fcmul_fcma(s0, q0);
            let p1 = NeonStoreF::fcmul_fcma(s1, q1);
            let p2 = NeonStoreF::fcmul_fcma(s2, q2);
            let p3 = NeonStoreF::fcmul_fcma(s3, q3);

            p0.write(dst);
            p1.write(&mut dst[2..]);
            p2.write(&mut dst[4..]);
            p3.write(&mut dst[6..]);
        }

        let dst = dst.as_chunks_mut::<8>().1;
        let a = a.as_chunks::<8>().1;
        let b = b.as_chunks::<8>().1;

        for ((dst, src), twiddle) in dst
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(a.as_chunks::<2>().0.iter())
            .zip(b.as_chunks::<2>().0.iter())
        {
            let s0 = NeonStoreF::load2(src).to_complex()[0];
            let q0 = NeonStoreF::from_complex_ref(twiddle);

            let p0 = NeonStoreF::fcmul_fcma(s0, q0);

            p0.write(dst);
        }

        let dst = dst.as_chunks_mut::<2>().1;
        let a = a.as_chunks::<2>().1;
        let b = b.as_chunks::<2>().1;

        for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            let s0 = NeonStoreF::load1_ref(src);
            let q0 = NeonStoreF::from_complex(twiddle);

            let p0 = NeonStoreF::fcmul_fcma(s0, q0);

            p0.write_single(dst);
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

            while src_x + 8 <= cut_width {
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

            while src_x + 4 <= cut_width {
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
        let conj_flag = NeonStoreF::conj_flag();
        for (dst, twiddle) in dst
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(b.as_chunks::<8>().0.iter())
        {
            let s0 = NeonStoreF::from_complex_ref(dst);
            let s1 = NeonStoreF::from_complex_ref(&dst[2..]);
            let s2 = NeonStoreF::from_complex_ref(&dst[4..]);
            let s3 = NeonStoreF::from_complex_ref(&dst[6..]);

            let q0 = NeonStoreF::from_complex_ref(twiddle);
            let q1 = NeonStoreF::from_complex_ref(&twiddle[2..]);
            let q2 = NeonStoreF::from_complex_ref(&twiddle[4..]);
            let q3 = NeonStoreF::from_complex_ref(&twiddle[6..]);

            let mut p0 = NeonStoreF::fcmul_fcma(s0, q0);
            let mut p1 = NeonStoreF::fcmul_fcma(s1, q1);
            let mut p2 = NeonStoreF::fcmul_fcma(s2, q2);
            let mut p3 = NeonStoreF::fcmul_fcma(s3, q3);

            p0 = p0.xor(conj_flag);
            p1 = p1.xor(conj_flag);
            p2 = p2.xor(conj_flag);
            p3 = p3.xor(conj_flag);

            p0.write(dst);
            p1.write(&mut dst[2..]);
            p2.write(&mut dst[4..]);
            p3.write(&mut dst[6..]);
        }

        let dst = dst.as_chunks_mut::<8>().1;
        let b = b.as_chunks::<8>().1;

        for (dst, twiddle) in dst
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(b.as_chunks::<2>().0.iter())
        {
            let s0 = NeonStoreF::from_complex_ref(dst);
            let q0 = NeonStoreF::from_complex_ref(twiddle);

            let mut p0 = NeonStoreF::fcmul_fcma(s0, q0);
            p0 = p0.xor(conj_flag);

            p0.write(dst);
        }

        let dst = dst.as_chunks_mut::<2>().1;
        let b = b.as_chunks::<2>().1;

        for (dst, twiddle) in dst.iter_mut().zip(b.iter()) {
            let s0 = NeonStoreF::from_complex(dst);
            let q0 = NeonStoreF::from_complex(twiddle);

            let mut p0 = NeonStoreF::fcmul_fcma(s0, q0);
            p0 = p0.xor(conj_flag);

            p0.write_single(dst);
        }
    }

    #[target_feature(enable = "fcma")]
    fn conjugate_mul_by_b_f32(
        &self,
        a: &[Complex<f32>],
        b: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) {
        for ((dst, src), twiddle) in dst
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(a.as_chunks::<8>().0.iter())
            .zip(b.as_chunks::<8>().0.iter())
        {
            let s0 = NeonStoreF::from_complex_ref(src);
            let s1 = NeonStoreF::from_complex_ref(&src[2..]);
            let s2 = NeonStoreF::from_complex_ref(&src[4..]);
            let s3 = NeonStoreF::from_complex_ref(&src[6..]);

            let q0 = NeonStoreF::from_complex_ref(twiddle);
            let q1 = NeonStoreF::from_complex_ref(&twiddle[2..]);
            let q2 = NeonStoreF::from_complex_ref(&twiddle[4..]);
            let q3 = NeonStoreF::from_complex_ref(&twiddle[6..]);

            let p0 = NeonStoreF::fcmul_conj_a(s0, q0);
            let p1 = NeonStoreF::fcmul_conj_a(s1, q1);
            let p2 = NeonStoreF::fcmul_conj_a(s2, q2);
            let p3 = NeonStoreF::fcmul_conj_a(s3, q3);

            p0.write(dst);
            p1.write(&mut dst[2..]);
            p2.write(&mut dst[4..]);
            p3.write(&mut dst[6..]);
        }

        let dst = dst.as_chunks_mut::<8>().1;
        let a = a.as_chunks::<8>().1;
        let b = b.as_chunks::<8>().1;

        for ((dst, src), twiddle) in dst
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(a.as_chunks::<2>().0.iter())
            .zip(b.as_chunks::<2>().0.iter())
        {
            let s0 = NeonStoreF::from_complex_ref(src);
            let q0 = NeonStoreF::from_complex_ref(twiddle);

            let p0 = NeonStoreF::fcmul_conj_a(s0, q0);

            p0.write(dst);
        }

        let dst = dst.as_chunks_mut::<2>().1;
        let a = a.as_chunks::<2>().1;
        let b = b.as_chunks::<2>().1;

        for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            let s0 = NeonStoreF::from_complex(src);
            let q0 = NeonStoreF::from_complex(twiddle);

            let p0 = NeonStoreF::fcmul_conj_a(s0, q0);

            p0.write_single(dst);
        }
    }

    #[target_feature(enable = "fcma")]
    fn mul_conjugate_expand_h2c_impl(&self, dst: &mut [Complex<f32>], b: &[Complex<f32>]) {
        assert_eq!(dst.len(), b.len());
        if dst.is_empty() {
            return;
        }
        dst[0] = c_mul_fast(dst[0], b[0]).conj();

        let (_, rem_dst) = dst.split_at_mut(1);
        let (mut left, mut right) = rem_dst.split_at_mut(b.len() / 2);

        let conjugate_factors = NeonStoreF::conj_flag();

        let mut forward_twiddles = &b[1..b.len() / 2];
        let mut backward_twiddles = &b[b.len() / 2..];

        for (((scratch_cell, scratch_cell_rev), twiddle), twiddle_rev) in left
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(right.as_rchunks_mut::<8>().1.iter_mut().rev())
            .zip(forward_twiddles.as_chunks::<8>().0.iter())
            .zip(backward_twiddles.as_rchunks::<8>().1.iter().rev())
        {
            let cell0 = NeonStoreF::from_complex_ref(scratch_cell);
            let cell1 = NeonStoreF::from_complex_ref(&scratch_cell[2..]);
            let cell2 = NeonStoreF::from_complex_ref(&scratch_cell[4..]);
            let cell3 = NeonStoreF::from_complex_ref(&scratch_cell[6..]);

            let fw0 = cell0
                .fcmul_fcma(NeonStoreF::from_complex_ref(twiddle))
                .xor(conjugate_factors);
            let bw0 = cell0
                .reverse_complex()
                .fcmul_conj_a(NeonStoreF::from_complex_ref(&twiddle_rev[6..]))
                .xor(conjugate_factors);

            let fw1 = cell1
                .fcmul_fcma(NeonStoreF::from_complex_ref(&twiddle[2..]))
                .xor(conjugate_factors);
            let bw1 = cell1
                .reverse_complex()
                .fcmul_conj_a(NeonStoreF::from_complex_ref(&twiddle_rev[4..]))
                .xor(conjugate_factors);

            let fw2 = cell2
                .fcmul_fcma(NeonStoreF::from_complex_ref(&twiddle[4..]))
                .xor(conjugate_factors);
            let bw2 = cell2
                .reverse_complex()
                .fcmul_conj_a(NeonStoreF::from_complex_ref(&twiddle_rev[2..]))
                .xor(conjugate_factors);

            let fw3 = cell3
                .fcmul_fcma(NeonStoreF::from_complex_ref(&twiddle[6..]))
                .xor(conjugate_factors);
            let bw3 = cell3
                .reverse_complex()
                .fcmul_conj_a(NeonStoreF::from_complex_ref(twiddle_rev))
                .xor(conjugate_factors);

            fw0.write(scratch_cell);
            fw1.write(&mut scratch_cell[2..]);
            fw2.write(&mut scratch_cell[4..]);
            fw3.write(&mut scratch_cell[6..]);

            bw0.write(&mut scratch_cell_rev[6..]);
            bw1.write(&mut scratch_cell_rev[4..]);
            bw2.write(&mut scratch_cell_rev[2..]);
            bw3.write(scratch_cell_rev);
        }

        let consumed =
            (left.as_chunks_mut::<8>().0.len() * 8).min(right.as_chunks_mut::<8>().0.len() * 8);

        let r_len = right.len();

        left = &mut left[consumed..];
        right = &mut right[..r_len - consumed];

        forward_twiddles = &forward_twiddles[consumed..];
        backward_twiddles = &backward_twiddles[..backward_twiddles.len() - consumed];

        for (((scratch_cell, scratch_cell_rev), twiddle), twiddle_rev) in left
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(right.as_rchunks_mut::<2>().1.iter_mut().rev())
            .zip(forward_twiddles.as_chunks::<2>().0.iter())
            .zip(backward_twiddles.as_rchunks::<2>().1.iter().rev())
        {
            let cell0 = NeonStoreF::from_complex_ref(scratch_cell);

            let fw0 = cell0
                .fcmul_fcma(NeonStoreF::from_complex_ref(twiddle))
                .xor(conjugate_factors);
            let bw0 = cell0
                .reverse_complex()
                .fcmul_conj_a(NeonStoreF::from_complex_ref(twiddle_rev))
                .xor(conjugate_factors);

            fw0.write(scratch_cell);
            bw0.write(scratch_cell_rev);
        }

        let consumed =
            (left.as_chunks_mut::<2>().0.len() * 2).min(right.as_chunks_mut::<2>().0.len() * 2);

        let r_len = right.len();

        left = &mut left[consumed..];
        right = &mut right[..r_len - consumed];

        forward_twiddles = &forward_twiddles[consumed..];
        backward_twiddles = &backward_twiddles[..backward_twiddles.len() - consumed];

        for (((scratch_cell, scratch_cell_rev), twiddle), twiddle_rev) in left
            .iter_mut()
            .zip(right.iter_mut().rev())
            .zip(forward_twiddles.iter())
            .zip(backward_twiddles.iter().rev())
        {
            let cell = NeonStoreF::from_complex(scratch_cell);
            let fw = cell
                .fcmul_fcma(NeonStoreF::from_complex(twiddle))
                .xor(conjugate_factors);
            let bw = cell
                .fcmul_conj_a(NeonStoreF::from_complex(twiddle_rev))
                .xor(conjugate_factors);
            fw.write_single(scratch_cell);
            bw.write_single(scratch_cell_rev);
        }

        if b.len().is_multiple_of(2) {
            let mid = b.len() / 2;
            dst[mid] = c_mul_fast(dst[mid], b[mid]).conj();
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

    fn mul_conjugate_expand_h2c(&self, dst: &mut [Complex<f32>], b: &[Complex<f32>]) {
        unsafe { self.mul_conjugate_expand_h2c_impl(dst, b) }
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
        for ((dst, src), twiddle) in dst
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(a.as_chunks::<4>().0.iter())
            .zip(b.as_chunks::<4>().0.iter())
        {
            let s0 = NeonStoreD::from_complex(&src[0]);
            let s1 = NeonStoreD::from_complex(&src[1]);
            let s2 = NeonStoreD::from_complex(&src[2]);
            let s3 = NeonStoreD::from_complex(&src[3]);

            let q0 = NeonStoreD::from_complex(&twiddle[0]);
            let q1 = NeonStoreD::from_complex(&twiddle[1]);
            let q2 = NeonStoreD::from_complex(&twiddle[2]);
            let q3 = NeonStoreD::from_complex(&twiddle[3]);

            let p0 = NeonStoreD::fcmul_fcma(s0, q0);
            let p1 = NeonStoreD::fcmul_fcma(s1, q1);
            let p2 = NeonStoreD::fcmul_fcma(s2, q2);
            let p3 = NeonStoreD::fcmul_fcma(s3, q3);

            p0.write_single(&mut dst[0]);
            p1.write_single(&mut dst[1]);
            p2.write_single(&mut dst[2]);
            p3.write_single(&mut dst[3]);
        }

        let dst = dst.as_chunks_mut::<4>().1;
        let a = a.as_chunks::<4>().1;
        let b = b.as_chunks::<4>().1;

        for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            let s0 = NeonStoreD::from_complex(src);
            let q0 = NeonStoreD::from_complex(twiddle);

            let p0 = NeonStoreD::fcmul_fcma(s0, q0);

            p0.write_single(dst);
        }
    }

    #[target_feature(enable = "fcma")]
    fn mul_expand_to_complex_impl(&self, a: &[f64], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        for ((dst, src), twiddle) in dst
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(a.as_chunks::<4>().0.iter())
            .zip(b.as_chunks::<4>().0.iter())
        {
            let q0 = NeonStoreD::load(src);
            let q2 = NeonStoreD::load(&src[2..]);

            let [s0, s1] = q0.to_complex();
            let [s2, s3] = q2.to_complex();

            let q0 = NeonStoreD::from_complex(&twiddle[0]);
            let q1 = NeonStoreD::from_complex(&twiddle[1]);
            let q2 = NeonStoreD::from_complex(&twiddle[2]);
            let q3 = NeonStoreD::from_complex(&twiddle[3]);

            let p0 = NeonStoreD::fcmul_fcma(s0, q0);
            let p1 = NeonStoreD::fcmul_fcma(s1, q1);
            let p2 = NeonStoreD::fcmul_fcma(s2, q2);
            let p3 = NeonStoreD::fcmul_fcma(s3, q3);

            p0.write_single(&mut dst[0]);
            p1.write_single(&mut dst[1]);
            p2.write_single(&mut dst[2]);
            p3.write_single(&mut dst[3]);
        }

        let dst = dst.as_chunks_mut::<4>().1;
        let a = a.as_chunks::<4>().1;
        let b = b.as_chunks::<4>().1;

        for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            let s0 = NeonStoreD::load1_ptr(src);
            let q0 = NeonStoreD::from_complex(twiddle);

            let p0 = NeonStoreD::fcmul_fcma(s0, q0);
            p0.write_single(dst);
        }
    }

    #[target_feature(enable = "fcma")]
    fn mul_conjugate_in_place_f64(&self, dst: &mut [Complex<f64>], b: &[Complex<f64>]) {
        let conj_flag = NeonStoreD::conj_flag();
        for (dst, twiddle) in dst
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(b.as_chunks::<4>().0.iter())
        {
            let s0 = NeonStoreD::from_complex(&dst[0]);
            let s1 = NeonStoreD::from_complex(&dst[1]);
            let s2 = NeonStoreD::from_complex(&dst[2]);
            let s3 = NeonStoreD::from_complex(&dst[3]);

            let q0 = NeonStoreD::from_complex(&twiddle[0]);
            let q1 = NeonStoreD::from_complex(&twiddle[1]);
            let q2 = NeonStoreD::from_complex(&twiddle[2]);
            let q3 = NeonStoreD::from_complex(&twiddle[3]);

            let mut p0 = NeonStoreD::fcmul_fcma(s0, q0);
            let mut p1 = NeonStoreD::fcmul_fcma(s1, q1);
            let mut p2 = NeonStoreD::fcmul_fcma(s2, q2);
            let mut p3 = NeonStoreD::fcmul_fcma(s3, q3);

            p0 = p0.xor(conj_flag);
            p1 = p1.xor(conj_flag);
            p2 = p2.xor(conj_flag);
            p3 = p3.xor(conj_flag);

            p0.write_single(&mut dst[0]);
            p1.write_single(&mut dst[1]);
            p2.write_single(&mut dst[2]);
            p3.write_single(&mut dst[3]);
        }

        let dst = dst.as_chunks_mut::<4>().1;
        let b = b.as_chunks::<4>().1;

        for (dst, twiddle) in dst.iter_mut().zip(b.iter()) {
            let s0 = NeonStoreD::from_complex(dst);
            let q0 = NeonStoreD::from_complex(twiddle);

            let mut p0 = NeonStoreD::fcmul_fcma(s0, q0);

            p0 = p0.xor(conj_flag);

            p0.write_single(dst);
        }
    }

    #[target_feature(enable = "fcma")]
    fn mul_conjugate_expand_h2c_impl_f64(&self, dst: &mut [Complex<f64>], b: &[Complex<f64>]) {
        assert_eq!(dst.len(), b.len());
        if dst.is_empty() {
            return;
        }
        dst[0] = c_mul_fast(dst[0], b[0]).conj();

        let (_, rem_dst) = dst.split_at_mut(1);
        let (mut left, mut right) = rem_dst.split_at_mut(b.len() / 2);

        let conjugate_factors = NeonStoreD::conj_flag();

        let mut forward_twiddles = &b[1..b.len() / 2];
        let mut backward_twiddles = &b[b.len() / 2..];

        for (((scratch_cell, scratch_cell_rev), twiddle), twiddle_rev) in left
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(right.as_rchunks_mut::<4>().1.iter_mut().rev())
            .zip(forward_twiddles.as_chunks::<4>().0.iter())
            .zip(backward_twiddles.as_rchunks::<4>().1.iter().rev())
        {
            let cell0 = NeonStoreD::from_complex(&scratch_cell[0]);
            let cell1 = NeonStoreD::from_complex(&scratch_cell[1]);
            let cell2 = NeonStoreD::from_complex(&scratch_cell[2]);
            let cell3 = NeonStoreD::from_complex(&scratch_cell[3]);

            let fw0 = cell0
                .fcmul_fcma(NeonStoreD::from_complex(&twiddle[0]))
                .xor(conjugate_factors);
            let bw0 = cell0
                .fcmul_conj_a(NeonStoreD::from_complex(&twiddle_rev[3]))
                .xor(conjugate_factors);

            let fw1 = cell1
                .fcmul_fcma(NeonStoreD::from_complex(&twiddle[1]))
                .xor(conjugate_factors);
            let bw1 = cell1
                .fcmul_conj_a(NeonStoreD::from_complex(&twiddle_rev[2]))
                .xor(conjugate_factors);

            let fw2 = cell2
                .fcmul_fcma(NeonStoreD::from_complex(&twiddle[2]))
                .xor(conjugate_factors);
            let bw2 = cell2
                .fcmul_conj_a(NeonStoreD::from_complex(&twiddle_rev[1]))
                .xor(conjugate_factors);

            let fw3 = cell3
                .fcmul_fcma(NeonStoreD::from_complex(&twiddle[3]))
                .xor(conjugate_factors);
            let bw3 = cell3
                .fcmul_conj_a(NeonStoreD::from_complex(&twiddle_rev[0]))
                .xor(conjugate_factors);

            fw0.write_ref(&mut scratch_cell[0]);
            fw1.write_ref(&mut scratch_cell[1]);
            fw2.write_ref(&mut scratch_cell[2]);
            fw3.write_ref(&mut scratch_cell[3]);

            bw0.write_ref(&mut scratch_cell_rev[3]);
            bw1.write_ref(&mut scratch_cell_rev[2]);
            bw2.write_ref(&mut scratch_cell_rev[1]);
            bw3.write_ref(&mut scratch_cell_rev[0]);
        }

        let consumed =
            (left.as_chunks_mut::<4>().0.len() * 4).min(right.as_chunks_mut::<4>().0.len() * 4);

        let r_len = right.len();

        left = &mut left[consumed..];
        right = &mut right[..r_len - consumed];

        forward_twiddles = &forward_twiddles[consumed..];
        backward_twiddles = &backward_twiddles[..backward_twiddles.len() - consumed];

        for (((scratch_cell, scratch_cell_rev), twiddle), twiddle_rev) in left
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(right.as_rchunks_mut::<2>().1.iter_mut().rev())
            .zip(forward_twiddles.as_chunks::<2>().0.iter())
            .zip(backward_twiddles.as_rchunks::<2>().1.iter().rev())
        {
            let cell0 = NeonStoreD::from_complex(&scratch_cell[0]);
            let cell1 = NeonStoreD::from_complex(&scratch_cell[1]);

            let fw0 = cell0
                .fcmul_fcma(NeonStoreD::from_complex(&twiddle[0]))
                .xor(conjugate_factors);
            let fw1 = cell1
                .fcmul_fcma(NeonStoreD::from_complex(&twiddle[1]))
                .xor(conjugate_factors);
            let bw0 = cell0
                .fcmul_conj_a(NeonStoreD::from_complex(&twiddle_rev[1]))
                .xor(conjugate_factors);
            let bw1 = cell1
                .fcmul_conj_a(NeonStoreD::from_complex(&twiddle_rev[0]))
                .xor(conjugate_factors);

            fw0.write_ref(&mut scratch_cell[0]);
            fw1.write_ref(&mut scratch_cell[1]);
            bw0.write_ref(&mut scratch_cell_rev[1]);
            bw1.write_ref(&mut scratch_cell_rev[0]);
        }

        let consumed =
            (left.as_chunks_mut::<2>().0.len() * 2).min(right.as_chunks_mut::<2>().0.len() * 2);

        let r_len = right.len();

        left = &mut left[consumed..];
        right = &mut right[..r_len - consumed];

        forward_twiddles = &forward_twiddles[consumed..];
        backward_twiddles = &backward_twiddles[..backward_twiddles.len() - consumed];

        for (((scratch_cell, scratch_cell_rev), twiddle), twiddle_rev) in left
            .iter_mut()
            .zip(right.iter_mut().rev())
            .zip(forward_twiddles.iter())
            .zip(backward_twiddles.iter().rev())
        {
            let cell = NeonStoreD::from_complex(scratch_cell);
            let fw = cell
                .fcmul_fcma(NeonStoreD::from_complex(twiddle))
                .xor(conjugate_factors);
            let bw = cell
                .fcmul_conj_a(NeonStoreD::from_complex(twiddle_rev))
                .xor(conjugate_factors);
            fw.write_ref(scratch_cell);
            bw.write_ref(scratch_cell_rev);
        }

        if b.len().is_multiple_of(2) {
            let mid = b.len() / 2;
            dst[mid] = c_mul_fast(dst[mid], b[mid]).conj();
        }
    }

    #[target_feature(enable = "fcma")]
    fn conjugate_mul_by_b_f64(
        &self,
        a: &[Complex<f64>],
        b: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) {
        for ((dst, src), twiddle) in dst
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(a.as_chunks::<4>().0.iter())
            .zip(b.as_chunks::<4>().0.iter())
        {
            let s0 = NeonStoreD::from_complex(&src[0]);
            let s1 = NeonStoreD::from_complex(&src[1]);
            let s2 = NeonStoreD::from_complex(&src[2]);
            let s3 = NeonStoreD::from_complex(&src[3]);

            let q0 = NeonStoreD::from_complex(&twiddle[0]);
            let q1 = NeonStoreD::from_complex(&twiddle[1]);
            let q2 = NeonStoreD::from_complex(&twiddle[2]);
            let q3 = NeonStoreD::from_complex(&twiddle[3]);

            let p0 = NeonStoreD::fcmul_conj_a(s0, q0);
            let p1 = NeonStoreD::fcmul_conj_a(s1, q1);
            let p2 = NeonStoreD::fcmul_conj_a(s2, q2);
            let p3 = NeonStoreD::fcmul_conj_a(s3, q3);

            p0.write_single(&mut dst[0]);
            p1.write_single(&mut dst[1]);
            p2.write_single(&mut dst[2]);
            p3.write_single(&mut dst[3]);
        }

        let dst = dst.as_chunks_mut::<4>().1;
        let a = a.as_chunks::<4>().1;
        let b = b.as_chunks::<4>().1;

        for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            let s0 = NeonStoreD::from_complex(src);
            let q0 = NeonStoreD::from_complex(twiddle);

            let p0 = NeonStoreD::fcmul_conj_a(s0, q0);

            p0.write_single(dst);
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
            while src_x + 2 <= cut_width {
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

    fn mul_conjugate_expand_h2c(&self, dst: &mut [Complex<f64>], b: &[Complex<f64>]) {
        unsafe {
            self.mul_conjugate_expand_h2c_impl_f64(dst, b);
        }
    }

    fn conjugate_mul_by_b(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        unsafe {
            self.conjugate_mul_by_b_f64(a, b, dst);
        }
    }
}
