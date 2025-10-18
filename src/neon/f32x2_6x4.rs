/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
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
use crate::neon::f32x2_2x2::neon_transpose_f32x2_2x2_impl;
use num_complex::Complex;
use std::arch::aarch64::*;

#[inline]
pub(crate) fn neon_transpose_f32x2_6x4(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let a0 = vld1q_f32(src.as_ptr().cast());
        let a1 = vld1q_f32(src.get_unchecked(src_stride..).as_ptr().cast());

        let b0 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
        let b1 = vld1q_f32(src.get_unchecked(2 + src_stride..).as_ptr().cast());

        let c0 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
        let c1 = vld1q_f32(src.get_unchecked(4 + src_stride..).as_ptr().cast());

        let d0 = vld1q_f32(src.get_unchecked(src_stride * 2..).as_ptr().cast());
        let d1 = vld1q_f32(src.get_unchecked(src_stride * 3..).as_ptr().cast());

        let e0 = vld1q_f32(src.get_unchecked(2 + src_stride * 2..).as_ptr().cast());
        let e1 = vld1q_f32(src.get_unchecked(2 + src_stride * 3..).as_ptr().cast());

        let f0 = vld1q_f32(src.get_unchecked(4 + src_stride * 2..).as_ptr().cast());
        let f1 = vld1q_f32(src.get_unchecked(4 + src_stride * 3..).as_ptr().cast());

        // Perform an 6 x 4 matrix transpose by building on top of the existing 2 x 2
        // matrix transpose implementation:
        // [ A B C ]^T => [ A^T D^T ]
        // [ D E F ]      [ B^T E^T ]
        //                [ C^T F^T ]

        let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(a0, a1));
        let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(b0, b1));
        let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(c0, c1));
        let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(d0, d1));
        let e0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(e0, e1));
        let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(f0, f1));

        vst1q_f32(dst.as_mut_ptr().cast(), a0.0);
        vst1q_f32(
            dst.get_unchecked_mut(dst_stride..).as_mut_ptr().cast(),
            a0.1,
        );

        vst1q_f32(
            dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr().cast(),
            b0.0,
        );
        vst1q_f32(
            dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr().cast(),
            b0.1,
        );

        vst1q_f32(
            dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr().cast(),
            c0.0,
        );
        vst1q_f32(
            dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr().cast(),
            c0.1,
        );

        vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), d0.0);
        vst1q_f32(
            dst.get_unchecked_mut(2 + dst_stride..).as_mut_ptr().cast(),
            d0.1,
        );

        vst1q_f32(
            dst.get_unchecked_mut(2 + dst_stride * 2..)
                .as_mut_ptr()
                .cast(),
            e0.0,
        );
        vst1q_f32(
            dst.get_unchecked_mut(2 + dst_stride * 3..)
                .as_mut_ptr()
                .cast(),
            e0.1,
        );

        vst1q_f32(
            dst.get_unchecked_mut(2 + dst_stride * 4..)
                .as_mut_ptr()
                .cast(),
            f0.0,
        );
        vst1q_f32(
            dst.get_unchecked_mut(2 + dst_stride * 5..)
                .as_mut_ptr()
                .cast(),
            f0.1,
        );
    }
}
