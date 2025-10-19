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

use num_complex::Complex;
use std::arch::aarch64::*;

#[inline]
pub(crate) fn neon_transpose_f64x2_2x2(
    src: &[Complex<f64>],
    src_stride: usize,
    dst: &mut [Complex<f64>],
    dst_stride: usize,
) {
    unsafe {
        let a0 = vld1q_f64(src.as_ptr().cast());
        let a1 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());

        let a2 = vld1q_f64(src.get_unchecked(src_stride..).as_ptr().cast());
        let a3 = vld1q_f64(src.get_unchecked(1 + src_stride..).as_ptr().cast());

        // [a0 a1]^T = [a0^T a2^T]
        // [a2 a3]     [a1^T a3^T]

        vst1q_f64(dst.as_mut_ptr().cast(), a0);
        vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), a2);
        vst1q_f64(dst.get_unchecked_mut(dst_stride..).as_mut_ptr().cast(), a1);
        vst1q_f64(
            dst.get_unchecked_mut(1 + dst_stride..).as_mut_ptr().cast(),
            a3,
        );
    }
}
