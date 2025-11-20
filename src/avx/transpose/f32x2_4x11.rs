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

use crate::avx::mixed::AvxStoreF;
use crate::avx::transpose::f32x2_2x2::transpose_f32_2x2_impl;
use crate::avx::transpose::f32x2_4x4::avx_transpose_f32x2_4x4_impl;
use num_complex::Complex;
use std::arch::x86_64::_mm256_setzero_ps;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_4x11(rows: [AvxStoreF; 11]) -> [AvxStoreF; 12] {
    let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, rows[3].v);
    let b0 = avx_transpose_f32x2_4x4_impl(rows[4].v, rows[5].v, rows[6].v, rows[7].v);
    let c0 = avx_transpose_f32x2_4x4_impl(rows[8].v, rows[9].v, rows[10].v, _mm256_setzero_ps());
    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
        AvxStoreF::raw(b0.0),
        AvxStoreF::raw(b0.1),
        AvxStoreF::raw(b0.2),
        AvxStoreF::raw(b0.3),
        AvxStoreF::raw(c0.0),
        AvxStoreF::raw(c0.1),
        AvxStoreF::raw(c0.2),
        AvxStoreF::raw(c0.3),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_2x11(rows: [AvxStoreF; 11]) -> [AvxStoreF; 12] {
    let a0 = transpose_f32_2x2_impl((rows[0].v, rows[1].v));
    let b0 = transpose_f32_2x2_impl((rows[2].v, rows[3].v));
    let c0 = transpose_f32_2x2_impl((rows[4].v, rows[5].v));
    let d0 = transpose_f32_2x2_impl((rows[6].v, rows[7].v));
    let f0 = transpose_f32_2x2_impl((rows[8].v, rows[9].v));
    let g0 = transpose_f32_2x2_impl((rows[10].v, _mm256_setzero_ps()));
    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(b0.0),
        AvxStoreF::raw(b0.1),
        AvxStoreF::raw(c0.0),
        AvxStoreF::raw(c0.1),
        AvxStoreF::raw(d0.0),
        AvxStoreF::raw(d0.1),
        AvxStoreF::raw(f0.0),
        AvxStoreF::raw(f0.1),
        AvxStoreF::raw(g0.0),
        AvxStoreF::raw(g0.1),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn block_transpose_f32x2_4x11(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let rows0: [AvxStoreF; 11] = std::array::from_fn(|x| {
            AvxStoreF::from_complex_ref(src.get_unchecked(x * src_stride..))
        });

        let t = transpose_4x11(rows0);

        for i in 0..2 {
            t[i * 4].write(dst.get_unchecked_mut(i * 4..));
            t[i * 4 + 1].write(dst.get_unchecked_mut(dst_stride + i * 4..));
            t[i * 4 + 2].write(dst.get_unchecked_mut(dst_stride * 2 + i * 4..));
            t[i * 4 + 3].write(dst.get_unchecked_mut(dst_stride * 3 + i * 4..));
        }

        t[8].write_lo3(dst.get_unchecked_mut(8..));
        t[9].write_lo3(dst.get_unchecked_mut(dst_stride + 8..));
        t[10].write_lo3(dst.get_unchecked_mut(dst_stride * 2 + 8..));
        t[11].write_lo3(dst.get_unchecked_mut(dst_stride * 3 + 8..));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn block_transpose_f32x2_2x11(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let rows0: [AvxStoreF; 11] =
            std::array::from_fn(|x| AvxStoreF::from_complex2(src.get_unchecked(x * src_stride..)));

        let t = transpose_2x11(rows0);

        for i in 0..2 {
            t[i * 4]
                .unpack_lo(t[i * 4 + 2])
                .write(dst.get_unchecked_mut(i * 4..));
            t[i * 4 + 1]
                .unpack_lo(t[i * 4 + 3])
                .write(dst.get_unchecked_mut(dst_stride + i * 4..));
        }

        t[8].write_lo2(dst.get_unchecked_mut(8..));
        t[9].write_lo2(dst.get_unchecked_mut(dst_stride + 8..));
        t[10].write_lo1(dst.get_unchecked_mut(10..));
        t[11].write_lo1(dst.get_unchecked_mut(dst_stride + 10..));
    }
}
