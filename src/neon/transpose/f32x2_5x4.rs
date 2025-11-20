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
use crate::neon::transpose::transpose_6x5;
use num_complex::Complex;

#[inline]
pub(crate) fn block_transpose_f32x2_5x4(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let rows0: [NeonStoreF; 4] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride..))
        });
        let rows1: [NeonStoreF; 4] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 2..))
        });
        let rows2: [NeonStoreF; 4] = std::array::from_fn(|x| {
            NeonStoreF::from_complex(src.get_unchecked(x * src_stride + 4))
        });

        let q = transpose_6x5(
            [
                rows0[0],
                rows0[1],
                rows0[2],
                rows0[3],
                NeonStoreF::default(),
            ],
            [
                rows1[0],
                rows1[1],
                rows1[2],
                rows1[3],
                NeonStoreF::default(),
            ],
            [
                rows2[0],
                rows2[1],
                rows2[2],
                rows2[3],
                NeonStoreF::default(),
            ],
        );

        for i in 0..5 {
            q[i].write(dst.get_unchecked_mut(i * dst_stride..));
            q[i + 6].write(dst.get_unchecked_mut(i * dst_stride + 2..));
        }
    }
}

#[inline]
pub(crate) fn block_transpose_f32x2_5x3(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let rows0: [NeonStoreF; 3] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride..))
        });
        let rows1: [NeonStoreF; 3] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 2..))
        });
        let rows2: [NeonStoreF; 3] = std::array::from_fn(|x| {
            NeonStoreF::from_complex(src.get_unchecked(x * src_stride + 4))
        });

        let q = transpose_6x5(
            [
                rows0[0],
                rows0[1],
                rows0[2],
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
            [
                rows1[0],
                rows1[1],
                rows1[2],
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
            [
                rows2[0],
                rows2[1],
                rows2[2],
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
        );

        for i in 0..5 {
            q[i].write(dst.get_unchecked_mut(i * dst_stride..));
            q[i + 6].write_lo(dst.get_unchecked_mut(i * dst_stride + 2..));
        }
    }
}

#[inline]
pub(crate) fn block_transpose_f32x2_5x2(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let rows0: [NeonStoreF; 2] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride..))
        });
        let rows1: [NeonStoreF; 2] = std::array::from_fn(|x| {
            NeonStoreF::from_complex_ref(src.get_unchecked(x * src_stride + 2..))
        });
        let rows2: [NeonStoreF; 2] = std::array::from_fn(|x| {
            NeonStoreF::from_complex(src.get_unchecked(x * src_stride + 4))
        });

        let q = transpose_6x5(
            [
                rows0[0],
                rows0[1],
                NeonStoreF::default(),
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
            [
                rows1[0],
                rows1[1],
                NeonStoreF::default(),
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
            [
                rows2[0],
                rows2[1],
                NeonStoreF::default(),
                NeonStoreF::default(),
                NeonStoreF::default(),
            ],
        );

        for i in 0..5 {
            q[i].write(dst.get_unchecked_mut(i * dst_stride..));
        }
    }
}
