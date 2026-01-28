/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
#![allow(clippy::needless_range_loop)]

use crate::avx::butterflies::shared::{boring_avx_butterfly, gen_butterfly_twiddles_f32};
use crate::avx::mixed::{AvxStoreF, ColumnButterfly7f, ColumnButterfly9f};
use crate::avx::transpose::{transpose_4x3, transpose_f32x2_4x4_aos};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f32x2_9x7(
    rows1: [AvxStoreF; 7],
    rows2: [AvxStoreF; 7],
    rows3: [AvxStoreF; 7],
) -> ([AvxStoreF; 9], [AvxStoreF; 9]) {
    let tl = transpose_f32x2_4x4_aos([rows1[0], rows1[1], rows1[2], rows1[3]]);

    let bl = transpose_4x3([rows1[4], rows1[5], rows1[6]]);

    let tr = transpose_f32x2_4x4_aos([rows2[0], rows2[1], rows2[2], rows2[3]]);

    let br = transpose_4x3([rows2[4], rows2[5], rows2[6]]);

    let far_top = transpose_f32x2_4x4_aos([rows3[0], rows3[1], rows3[2], rows3[3]]);

    let far_bottom = transpose_4x3([rows3[4], rows3[5], rows3[6]]);

    (
        [
            tl[0], tl[1], tl[2], tl[3], tr[0], tr[1], tr[2], tr[3], far_top[0],
        ],
        [
            bl[0],
            bl[1],
            bl[2],
            bl[3],
            br[0],
            br[1],
            br[2],
            br[3],
            far_bottom[0],
        ],
    )
}

pub(crate) struct AvxButterfly63f {
    direction: FftDirection,
    bf7: ColumnButterfly7f,
    bf9: ColumnButterfly9f,
    twiddles: [AvxStoreF; 18],
}

impl AvxButterfly63f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(9, 7, fft_direction, 63),
            bf9: ColumnButterfly9f::new(fft_direction),
            bf7: ColumnButterfly7f::new(fft_direction),
        }
    }
}

boring_avx_butterfly!(AvxButterfly63f, f32, 63);

impl AvxButterfly63f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows0: [AvxStoreF; 7] = [AvxStoreF::zero(); 7];
        let mut rows1: [AvxStoreF; 7] = [AvxStoreF::zero(); 7];
        let mut rows2: [AvxStoreF; 7] = [AvxStoreF::zero(); 7];
        // columns
        // 0-4
        for i in 0..7 {
            rows0[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 9..));
        }

        rows0 = self.bf7.exec(rows0);

        for i in 1..7 {
            rows0[i] = AvxStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
        }

        // 4-8

        for i in 0..7 {
            rows1[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 9 + 4..));
        }

        rows1 = self.bf7.exec(rows1);

        for i in 1..7 {
            rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1 + 6]);
        }

        // 8-9

        for i in 0..7 {
            rows2[i] = AvxStoreF::from_complex(chunk.index(i * 9 + 8));
        }

        rows2 = self.bf7.exec(rows2);

        for i in 1..7 {
            rows2[i] = AvxStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 12]);
        }

        let (mut t0, mut t1) = transpose_f32x2_9x7(rows0, rows1, rows2);

        // rows

        // 0-4
        t0 = self.bf9.exec(t0);

        for i in 0..9 {
            t0[i].write(chunk.slice_from_mut(i * 7..));
        }

        // 4-7

        t1 = self.bf9.exec(t1);

        for i in 0..9 {
            t1[i].write_lo3(chunk.slice_from_mut(i * 7 + 4..));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly63, f32, AvxButterfly63f, 63, 1e-3);
}
