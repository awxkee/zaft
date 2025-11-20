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

use crate::avx::mixed::AvxStoreF;
use crate::avx::transpose::f32x2_4x4::transpose_f32x2_4x4_aos;
use num_complex::Complex;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn block_transpose_f32x2_8x3(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let a0 = AvxStoreF::from_complex_ref(src);
        let a1 = AvxStoreF::from_complex_ref(src.get_unchecked(src_stride..));
        let a2 = AvxStoreF::from_complex_ref(src.get_unchecked(2 * src_stride..));

        let b0 = AvxStoreF::from_complex_ref(src.get_unchecked(4..));
        let b1 = AvxStoreF::from_complex_ref(src.get_unchecked(4 + src_stride..));
        let b2 = AvxStoreF::from_complex_ref(src.get_unchecked(4 + 2 * src_stride..));

        let v0 = transpose_f32x2_4x4_aos([a0, a1, a2, AvxStoreF::zero()]);
        let v1 = transpose_f32x2_4x4_aos([b0, b1, b2, AvxStoreF::zero()]);

        v0[0].write_lo3(dst);
        v0[1].write_lo3(dst.get_unchecked_mut(dst_stride..));
        v0[2].write_lo3(dst.get_unchecked_mut(dst_stride * 2..));
        v0[3].write_lo3(dst.get_unchecked_mut(dst_stride * 3..));
        v1[0].write_lo3(dst.get_unchecked_mut(dst_stride * 4..));
        v1[1].write_lo3(dst.get_unchecked_mut(dst_stride * 5..));
        v1[2].write_lo3(dst.get_unchecked_mut(dst_stride * 6..));
        v1[3].write_lo3(dst.get_unchecked_mut(dst_stride * 7..));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn block_transpose_f32x2_3x8(
    src: &[Complex<f32>],
    src_stride: usize,
    dst: &mut [Complex<f32>],
    dst_stride: usize,
) {
    unsafe {
        let a0 = AvxStoreF::from_complex3(src);
        let a1 = AvxStoreF::from_complex3(src.get_unchecked(src_stride..));
        let a2 = AvxStoreF::from_complex3(src.get_unchecked(2 * src_stride..));
        let a3 = AvxStoreF::from_complex3(src.get_unchecked(3 * src_stride..));

        let b0 = AvxStoreF::from_complex3(src.get_unchecked(4 * src_stride..));
        let b1 = AvxStoreF::from_complex3(src.get_unchecked(5 * src_stride..));
        let b2 = AvxStoreF::from_complex3(src.get_unchecked(6 * src_stride..));
        let b3 = AvxStoreF::from_complex3(src.get_unchecked(7 * src_stride..));

        let v0 = transpose_f32x2_4x4_aos([a0, a1, a2, a3]);
        let v1 = transpose_f32x2_4x4_aos([b0, b1, b2, b3]);

        v0[0].write(dst);
        v0[1].write(dst.get_unchecked_mut(dst_stride..));
        v0[2].write(dst.get_unchecked_mut(dst_stride * 2..));

        v1[0].write(dst.get_unchecked_mut(4..));
        v1[1].write(dst.get_unchecked_mut(4 + dst_stride..));
        v1[2].write(dst.get_unchecked_mut(4 + dst_stride * 2..));
    }
}
