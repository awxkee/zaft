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
use crate::complex_fma::{c_conj_mul_fast, c_mul_fast};
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::spectrum_arithmetic::ComplexArith;
use num_complex::Complex;
use num_traits::Zero;
use std::marker::PhantomData;

pub(crate) struct NeonSpectrumArithmetic<T> {
    pub(crate) phantom_data: PhantomData<T>,
}

impl ComplexArith<f32> for NeonSpectrumArithmetic<f32> {
    fn mul(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
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

            let p0 = NeonStoreF::mul_by_complex(s0, q0);
            let p1 = NeonStoreF::mul_by_complex(s1, q1);
            let p2 = NeonStoreF::mul_by_complex(s2, q2);
            let p3 = NeonStoreF::mul_by_complex(s3, q3);

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

            let p0 = NeonStoreF::mul_by_complex(s0, q0);

            p0.write(dst);
        }

        let dst = dst.as_chunks_mut::<2>().1;
        let a = a.as_chunks::<2>().1;
        let b = b.as_chunks::<2>().1;

        for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            *dst = c_mul_fast(*src, *twiddle);
        }
    }

    fn mul_and_cut(
        &self,
        a: &[Complex<f32>],
        original_width: usize,
        b: &[Complex<f32>],
        cut_width: usize,
        dst: &mut [Complex<f32>],
    ) {
        assert_eq!(b.len(), dst.len());
        assert_eq!(a.len() / original_width, dst.len() / cut_width);

        for ((source, twiddle), dst) in b
            .chunks_exact(cut_width)
            .zip(a.chunks_exact(original_width))
            .zip(dst.chunks_exact_mut(cut_width))
        {
            let mut src_x = 0usize;
            while src_x + 4 <= cut_width {
                let s0 = NeonStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x..) });
                let s1 = NeonStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x + 2..) });

                let tw0 = NeonStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x..) });
                let tw1 =
                    NeonStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x + 2..) });

                let p0 = NeonStoreF::mul_by_complex(s0, tw0);
                let p1 = NeonStoreF::mul_by_complex(s1, tw1);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                p1.write(unsafe { dst.get_unchecked_mut(src_x + 2..) });

                src_x += 4;
            }

            while src_x + 2 <= cut_width {
                let s0 = NeonStoreF::from_complex_ref(unsafe { source.get_unchecked(src_x..) });
                let tw0 = NeonStoreF::from_complex_ref(unsafe { twiddle.get_unchecked(src_x..) });
                let p0 = NeonStoreF::mul_by_complex(s0, tw0);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                src_x += 2;
            }

            while src_x < cut_width {
                let s0 = NeonStoreF::from_complex(unsafe { source.get_unchecked(src_x) });
                let tw0 = NeonStoreF::from_complex(unsafe { twiddle.get_unchecked(src_x) });
                let p0 = NeonStoreF::mul_by_complex(s0, tw0);

                p0.write_lo(unsafe { dst.get_unchecked_mut(src_x..) });
                src_x += 1;
            }
        }
    }

    fn mul_expand_to_complex(&self, a: &[f32], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
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

            let p0 = NeonStoreF::mul_by_complex(s0, q0);
            let p1 = NeonStoreF::mul_by_complex(s1, q1);
            let p2 = NeonStoreF::mul_by_complex(s2, q2);
            let p3 = NeonStoreF::mul_by_complex(s3, q3);

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

            let p0 = NeonStoreF::mul_by_complex(s0, q0);

            p0.write(dst);
        }

        let dst = dst.as_chunks_mut::<2>().1;
        let a = a.as_chunks::<2>().1;
        let b = b.as_chunks::<2>().1;

        for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            *dst = c_mul_fast(Complex::new(*src, f32::zero()), *twiddle);
        }
    }

    fn mul_conjugate_in_place(&self, dst: &mut [Complex<f32>], b: &[Complex<f32>]) {
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

            let mut p0 = NeonStoreF::mul_by_complex(s0, q0);
            let mut p1 = NeonStoreF::mul_by_complex(s1, q1);
            let mut p2 = NeonStoreF::mul_by_complex(s2, q2);
            let mut p3 = NeonStoreF::mul_by_complex(s3, q3);

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

            let mut p0 = NeonStoreF::mul_by_complex(s0, q0);
            p0 = p0.xor(conj_flag);

            p0.write(dst);
        }

        let dst = dst.as_chunks_mut::<2>().1;
        let b = b.as_chunks::<2>().1;

        for (dst, twiddle) in dst.iter_mut().zip(b.iter()) {
            let s0 = NeonStoreF::from_complex(dst);
            let q0 = NeonStoreF::from_complex(twiddle);

            let mut p0 = NeonStoreF::mul_by_complex(s0, q0);
            p0 = p0.xor(conj_flag);

            p0.write_single(dst);
        }
    }

    fn conjugate_mul_by_b(&self, a: &[Complex<f32>], b: &[Complex<f32>], dst: &mut [Complex<f32>]) {
        let conj_flag = NeonStoreF::conj_flag();
        for ((dst, src), twiddle) in dst
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(a.as_chunks::<8>().0.iter())
            .zip(b.as_chunks::<8>().0.iter())
        {
            let mut s0 = NeonStoreF::from_complex_ref(src);
            let mut s1 = NeonStoreF::from_complex_ref(&src[2..]);
            let mut s2 = NeonStoreF::from_complex_ref(&src[4..]);
            let mut s3 = NeonStoreF::from_complex_ref(&src[6..]);

            let q0 = NeonStoreF::from_complex_ref(twiddle);
            let q1 = NeonStoreF::from_complex_ref(&twiddle[2..]);
            let q2 = NeonStoreF::from_complex_ref(&twiddle[4..]);
            let q3 = NeonStoreF::from_complex_ref(&twiddle[6..]);

            s0 = s0.xor(conj_flag);
            s1 = s1.xor(conj_flag);
            s2 = s2.xor(conj_flag);
            s3 = s3.xor(conj_flag);

            let p0 = NeonStoreF::mul_by_complex(s0, q0);
            let p1 = NeonStoreF::mul_by_complex(s1, q1);
            let p2 = NeonStoreF::mul_by_complex(s2, q2);
            let p3 = NeonStoreF::mul_by_complex(s3, q3);

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
            let mut s0 = NeonStoreF::from_complex_ref(src);
            let q0 = NeonStoreF::from_complex_ref(twiddle);

            s0 = s0.xor(conj_flag);

            let p0 = NeonStoreF::mul_by_complex(s0, q0);

            p0.write(dst);
        }

        let dst = dst.as_chunks_mut::<2>().1;
        let a = a.as_chunks::<2>().1;
        let b = b.as_chunks::<2>().1;

        for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            *dst = c_conj_mul_fast(*src, *twiddle);
        }
    }
}

impl ComplexArith<f64> for NeonSpectrumArithmetic<f64> {
    fn mul(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
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

            let p0 = NeonStoreD::mul_by_complex(s0, q0);
            let p1 = NeonStoreD::mul_by_complex(s1, q1);
            let p2 = NeonStoreD::mul_by_complex(s2, q2);
            let p3 = NeonStoreD::mul_by_complex(s3, q3);

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

            let p0 = NeonStoreD::mul_by_complex(s0, q0);

            p0.write_single(dst);
        }
    }

    fn mul_and_cut(
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

                let p0 = NeonStoreD::mul_by_complex(s0, tw0);
                let p1 = NeonStoreD::mul_by_complex(s1, tw1);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                p1.write(unsafe { dst.get_unchecked_mut(src_x + 1..) });

                src_x += 2;
            }

            while src_x < cut_width {
                let s0 = NeonStoreD::from_complex(unsafe { source.get_unchecked(src_x) });
                let tw0 = NeonStoreD::from_complex(unsafe { twiddle.get_unchecked(src_x) });
                let p0 = NeonStoreD::mul_by_complex(s0, tw0);

                p0.write(unsafe { dst.get_unchecked_mut(src_x..) });
                src_x += 1;
            }
        }
    }

    fn mul_expand_to_complex(&self, a: &[f64], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
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

            let p0 = NeonStoreD::mul_by_complex(s0, q0);
            let p1 = NeonStoreD::mul_by_complex(s1, q1);
            let p2 = NeonStoreD::mul_by_complex(s2, q2);
            let p3 = NeonStoreD::mul_by_complex(s3, q3);

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

            let p0 = NeonStoreD::mul_by_complex(s0, q0);
            p0.write_single(dst);
        }
    }

    fn mul_conjugate_in_place(&self, dst: &mut [Complex<f64>], b: &[Complex<f64>]) {
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

            let mut p0 = NeonStoreD::mul_by_complex(s0, q0);
            let mut p1 = NeonStoreD::mul_by_complex(s1, q1);
            let mut p2 = NeonStoreD::mul_by_complex(s2, q2);
            let mut p3 = NeonStoreD::mul_by_complex(s3, q3);

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

            let mut p0 = NeonStoreD::mul_by_complex(s0, q0);

            p0 = p0.xor(conj_flag);

            p0.write_single(dst);
        }
    }

    fn conjugate_mul_by_b(&self, a: &[Complex<f64>], b: &[Complex<f64>], dst: &mut [Complex<f64>]) {
        let conj_flag = NeonStoreD::conj_flag();
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

            let p0 = NeonStoreD::mul_by_complex(s0.xor(conj_flag), q0);
            let p1 = NeonStoreD::mul_by_complex(s1.xor(conj_flag), q1);
            let p2 = NeonStoreD::mul_by_complex(s2.xor(conj_flag), q2);
            let p3 = NeonStoreD::mul_by_complex(s3.xor(conj_flag), q3);

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

            let p0 = NeonStoreD::mul_by_complex(s0.xor(conj_flag), q0);

            p0.write_single(dst);
        }
    }
}
