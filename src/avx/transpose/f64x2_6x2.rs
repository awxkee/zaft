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
use crate::avx::mixed::AvxStoreD;
use crate::avx::transpose::transpose_f64x2_2x2;
use num_complex::Complex;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x2_2x6(rows: [AvxStoreD; 6]) -> [AvxStoreD; 6] {
    let a0 = transpose_f64x2_2x2(rows[0].v, rows[1].v);
    let b0 = transpose_f64x2_2x2(rows[2].v, rows[3].v);
    let c0 = transpose_f64x2_2x2(rows[4].v, rows[5].v);
    [
        AvxStoreD::raw(a0.0),
        AvxStoreD::raw(a0.1),
        AvxStoreD::raw(b0.0),
        AvxStoreD::raw(b0.1),
        AvxStoreD::raw(c0.0),
        AvxStoreD::raw(c0.1),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn block_transpose_f64x2_2x6(
    src: &[Complex<f64>],
    src_stride: usize,
    dst: &mut [Complex<f64>],
    dst_stride: usize,
) {
    unsafe {
        let rows0: [AvxStoreD; 6] = std::array::from_fn(|x| {
            AvxStoreD::from_complex_ref(src.get_unchecked(x * src_stride..))
        });
        let v0 = transpose_f64x2_2x6(rows0);

        for i in 0..3 {
            v0[i * 2].write(dst.get_unchecked_mut(i * 2..));
            v0[i * 2 + 1].write(dst.get_unchecked_mut(dst_stride + i * 2..));
        }
    }
}
