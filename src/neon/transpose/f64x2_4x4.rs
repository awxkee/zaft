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

use crate::neon::transpose::neon_transpose_f64x2_4x4_impl;
use num_complex::Complex;
use std::arch::aarch64::*;

#[inline]
pub(crate) fn block_transpose_f64x2_4x4(
    src: &[Complex<f64>],
    src_stride: usize,
    dst: &mut [Complex<f64>],
    dst_stride: usize,
) {
    unsafe {
        let a0 = float64x2x4_t(
            vld1q_f64(src.as_ptr().cast()),
            vld1q_f64(src.get_unchecked(1..).as_ptr().cast()),
            vld1q_f64(src.get_unchecked(2..).as_ptr().cast()),
            vld1q_f64(src.get_unchecked(3..).as_ptr().cast()),
        );
        let row1 = src.get_unchecked(src_stride..);
        let a1 = float64x2x4_t(
            vld1q_f64(row1.as_ptr().cast()),
            vld1q_f64(row1.get_unchecked(1..).as_ptr().cast()),
            vld1q_f64(row1.get_unchecked(2..).as_ptr().cast()),
            vld1q_f64(row1.get_unchecked(3..).as_ptr().cast()),
        );
        let row2 = src.get_unchecked(src_stride * 2..);
        let a2 = float64x2x4_t(
            vld1q_f64(row2.as_ptr().cast()),
            vld1q_f64(row2.get_unchecked(1..).as_ptr().cast()),
            vld1q_f64(row2.get_unchecked(2..).as_ptr().cast()),
            vld1q_f64(row2.get_unchecked(3..).as_ptr().cast()),
        );
        let row3 = src.get_unchecked(src_stride * 3..);
        let a3 = float64x2x4_t(
            vld1q_f64(row3.as_ptr().cast()),
            vld1q_f64(row3.get_unchecked(1..).as_ptr().cast()),
            vld1q_f64(row3.get_unchecked(2..).as_ptr().cast()),
            vld1q_f64(row3.get_unchecked(3..).as_ptr().cast()),
        );

        let (v0, v1, v2, v3) = neon_transpose_f64x2_4x4_impl(a0, a1, a2, a3);

        // [a0 a1]^T = [a0^T a2^T]
        // [a2 a3]     [a1^T a3^T]

        vst1q_f64(dst.as_mut_ptr().cast(), v0.0);
        vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), v0.1);
        vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), v0.2);
        vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), v0.3);

        let row1 = dst.get_unchecked_mut(dst_stride..);
        vst1q_f64(row1.as_mut_ptr().cast(), v1.0);
        vst1q_f64(row1.get_unchecked_mut(1..).as_mut_ptr().cast(), v1.1);
        vst1q_f64(row1.get_unchecked_mut(2..).as_mut_ptr().cast(), v1.2);
        vst1q_f64(row1.get_unchecked_mut(3..).as_mut_ptr().cast(), v1.3);

        let row2 = dst.get_unchecked_mut(dst_stride * 2..);
        vst1q_f64(row2.as_mut_ptr().cast(), v2.0);
        vst1q_f64(row2.get_unchecked_mut(1..).as_mut_ptr().cast(), v2.1);
        vst1q_f64(row2.get_unchecked_mut(2..).as_mut_ptr().cast(), v2.2);
        vst1q_f64(row2.get_unchecked_mut(3..).as_mut_ptr().cast(), v2.3);

        let row3 = dst.get_unchecked_mut(dst_stride * 3..);
        vst1q_f64(row3.as_mut_ptr().cast(), v3.0);
        vst1q_f64(row3.get_unchecked_mut(1..).as_mut_ptr().cast(), v3.1);
        vst1q_f64(row3.get_unchecked_mut(2..).as_mut_ptr().cast(), v3.2);
        vst1q_f64(row3.get_unchecked_mut(3..).as_mut_ptr().cast(), v3.3);
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use num_complex::Complex;
//
//     #[test]
//     fn test_block_transpose_f64x2_4x4() {
//         let mut src = vec![Complex::<f64>::new(0.0, 0.0); 4 * 4];
//         for (i, q) in src.iter_mut().enumerate() {
//             *q = Complex::<f64>::new(i as f64, 0.);
//         }
//         // Output buffer
//         let mut dst = vec![Complex::<f64>::new(-1.0, -1.0); 4 * 4];
//
//         // Call the transpose
//         block_transpose_f64x2_4x4(&src, 4, &mut dst, 4);
//
//         for chunk in src.chunks_exact(4) {
//             println!("{:?}", chunk.iter().map(|x| x.re).collect::<Vec<_>>());
//         }
//         println!("-----");
//
//         for chunk in dst.chunks_exact(4) {
//             println!("{:?}", chunk.iter().map(|x| x.re).collect::<Vec<_>>());
//         }
//     }
// }
