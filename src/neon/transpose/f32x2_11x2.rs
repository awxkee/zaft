/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use num_complex::Complex;
use std::arch::aarch64::float32x4x2_t;

#[inline(always)]
pub(crate) fn transpose_11x2(rows0: [NeonStoreF; 6], rows1: [NeonStoreF; 6]) -> [NeonStoreF; 12] {
    let a = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows1[0].v));
    let b = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[1].v, rows1[1].v));
    let c = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[2].v, rows1[2].v));
    let d = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[3].v, rows1[3].v));
    let e = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[4].v, rows1[4].v));
    let f = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[5].v, rows1[5].v));

    [
        NeonStoreF::raw(a.0),
        NeonStoreF::raw(a.1),
        NeonStoreF::raw(b.0),
        NeonStoreF::raw(b.1),
        NeonStoreF::raw(c.0),
        NeonStoreF::raw(c.1),
        NeonStoreF::raw(d.0),
        NeonStoreF::raw(d.1),
        NeonStoreF::raw(e.0),
        NeonStoreF::raw(e.1),
        NeonStoreF::raw(f.0),
        NeonStoreF::raw(f.1),
    ]
}

#[inline]
pub(crate) fn block_transpose_f32x2_11x2(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let rows0: [NeonStoreF; 6] = std::array::from_fn(|x| {
            if x == 5 {
                NeonStoreF::from_complex(src.get_unchecked(x * 2))
            } else {
                NeonStoreF::from_complex_ref(src.get_unchecked(x * 2..))
            }
        });
        let rows1: [NeonStoreF; 6] = std::array::from_fn(|x| {
            if x == 5 {
                NeonStoreF::from_complex(src.get_unchecked(src_stride + x * 2))
            } else {
                NeonStoreF::from_complex_ref(src.get_unchecked(src_stride + x * 2..))
            }
        });

        let t = transpose_11x2(rows0, rows1);

        for i in 0..10 {
            t[i].write(dst.get_unchecked_mut(i * dst_stride..));
        }

        // Last partial (11th) pair
        t[11]
            .write(dst.get_unchecked_mut(10 * dst_stride..));
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use num_complex::Complex;
//
//     #[test]
//     fn test_block_transpose_f32x2_11x2() {
//         let mut src = vec![Complex::<f32>::new(0.0, 0.0); 11 * 2];
//         for (i, q) in src.iter_mut().enumerate() {
//             *q = Complex::<f32>::new(i as f32, 0.);
//         }
//         // Output buffer
//         let mut dst = vec![Complex::<f32>::new(-1.0, -1.0); 11 * 2];
//
//         // Call the transpose
//         block_transpose_f32x2_11x2(&src, 11, &mut dst, 2);
//
//         for chunk in src.chunks_exact(11) {
//             println!("{:?}", chunk.iter().map(|x| x.re).collect::<Vec<_>>());
//         }
//         println!("-----");
//
//         for chunk in dst.chunks_exact(2) {
//             println!("{:?}", chunk.iter().map(|x| x.re).collect::<Vec<_>>());
//         }
//     }
// }
