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
pub(crate) fn neon_transpose_f32x2_4x4(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let r0 = vld1q_f32(src.get_unchecked(0..).as_ptr().cast());
        let r1 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());

        let r2 = vld1q_f32(src.get_unchecked(src_stride..).as_ptr().cast());
        let r3 = vld1q_f32(src.get_unchecked(2 + src_stride..).as_ptr().cast());

        let r4 = vld1q_f32(src.get_unchecked(2 * src_stride..).as_ptr().cast());
        let r5 = vld1q_f32(src.get_unchecked(2 + 2 * src_stride..).as_ptr().cast());

        let r6 = vld1q_f32(src.get_unchecked(3 * src_stride..).as_ptr().cast());
        let r7 = vld1q_f32(src.get_unchecked(2 + 3 * src_stride..).as_ptr().cast());

        // Perform an 4 x 4 matrix transpose by building on top of the existing 2 x 2
        // matrix transpose implementation:
        // [ A B ]^T => [ A^T C^T ]
        // [ C D ]      [ B^T D^T ]

        let q0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r0, r2));
        let q1 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r1, r3));
        let q2 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r4, r6));
        let q3 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(r5, r7));

        vst1q_f32(dst.as_mut_ptr().cast(), q0.0);
        vst1q_f32(
            dst.get_unchecked_mut(dst_stride..).as_mut_ptr().cast(),
            q0.1,
        );

        vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), q2.0);
        vst1q_f32(
            dst.get_unchecked_mut(2 + dst_stride..).as_mut_ptr().cast(),
            q2.1,
        );

        vst1q_f32(
            dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr().cast(),
            q1.0,
        );
        vst1q_f32(
            dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr().cast(),
            q1.1,
        );

        vst1q_f32(
            dst.get_unchecked_mut(2 + 2 * dst_stride..)
                .as_mut_ptr()
                .cast(),
            q3.0,
        );
        vst1q_f32(
            dst.get_unchecked_mut(2 + 3 * dst_stride..)
                .as_mut_ptr()
                .cast(),
            q3.1,
        );
    }
}
