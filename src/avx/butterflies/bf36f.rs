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
#![allow(clippy::needless_range_loop)]

use crate::avx::butterflies::shared::boring_avx_butterfly;
use crate::avx::mixed::{AvxStoreF, ColumnButterfly4f, ColumnButterfly9f, SseStoreF};
use crate::avx::transpose::avx_transpose_f32x2_4x4_impl;
use crate::avx::util::_mm_unpacklo_ps64;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_9x4_to_4x9_emptycolumn1_f32(
    rows0: [SseStoreF; 4],
    rows1: [AvxStoreF; 4],
    rows2: [AvxStoreF; 4],
) -> [AvxStoreF; 9] {
    // the first row of the output will be the first column of the input
    let unpacked0 = _mm_unpacklo_ps64(rows0[0].v, rows0[1].v);
    let unpacked1 = _mm_unpacklo_ps64(rows0[2].v, rows0[3].v);
    let output0 = _mm256_setr_m128(unpacked0, unpacked1);

    let transposed0 = avx_transpose_f32x2_4x4_impl(rows1[0].v, rows1[1].v, rows1[2].v, rows1[3].v);
    let transposed1 = avx_transpose_f32x2_4x4_impl(rows2[0].v, rows2[1].v, rows2[2].v, rows2[3].v);

    [
        AvxStoreF::raw(output0),
        AvxStoreF::raw(transposed0.0),
        AvxStoreF::raw(transposed0.1),
        AvxStoreF::raw(transposed0.2),
        AvxStoreF::raw(transposed0.3),
        AvxStoreF::raw(transposed1.0),
        AvxStoreF::raw(transposed1.1),
        AvxStoreF::raw(transposed1.2),
        AvxStoreF::raw(transposed1.3),
    ]
}

pub(crate) struct AvxButterfly36f {
    direction: FftDirection,
    twiddles: [AvxStoreF; 6],
    bf4: ColumnButterfly4f,
    bf9: ColumnButterfly9f,
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn make_mixedradix_twiddle_chunk(
    x: usize,
    y: usize,
    len: usize,
    direction: FftDirection,
) -> AvxStoreF {
    let mut twiddle_chunk = [Complex::<f32>::default(); 4];
    for i in 0..4 {
        twiddle_chunk[i] = compute_twiddle(y * (x + i), len, direction);
    }

    AvxStoreF::from_complex_ref(twiddle_chunk.as_slice())
}

macro_rules! gen_butterfly_twiddles_interleaved_columns {
    ($num_rows:expr, $num_cols:expr, $skip_cols:expr, $direction: expr) => {{
        const FFT_LEN: usize = $num_rows * $num_cols;
        const TWIDDLE_ROWS: usize = $num_rows - 1;
        const TWIDDLE_COLS: usize = $num_cols - $skip_cols;
        const TWIDDLE_VECTOR_COLS: usize = TWIDDLE_COLS / 4;
        const TWIDDLE_VECTOR_COUNT: usize = TWIDDLE_VECTOR_COLS * TWIDDLE_ROWS;
        let mut twiddles = [AvxStoreF::zero(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index / TWIDDLE_VECTOR_COLS) + 1;
            let x = (index % TWIDDLE_VECTOR_COLS) * 4 + $skip_cols;

            twiddles[index] = make_mixedradix_twiddle_chunk(x, y, FFT_LEN, $direction);
        }
        twiddles
    }};
}

impl AvxButterfly36f {
    pub(crate) fn new(direction: FftDirection) -> Self {
        unsafe { Self::new_init(direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(direction: FftDirection) -> Self {
        AvxButterfly36f {
            direction,
            twiddles: unsafe { gen_butterfly_twiddles_interleaved_columns!(4, 9, 1, direction) },
            bf4: ColumnButterfly4f::new(direction),
            bf9: ColumnButterfly9f::new(direction),
        }
    }
}

boring_avx_butterfly!(AvxButterfly36f, f32, 36);

impl AvxButterfly36f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows0 = [SseStoreF::zero(); 4];
        let mut rows1 = [AvxStoreF::zero(); 4];
        let mut rows2 = [AvxStoreF::zero(); 4];
        // Mixed Radix 9x4
        for r in 0..4 {
            rows0[r] = SseStoreF::from_complex_ref(chunk.slice_from(r * 9..));
            rows1[r] = AvxStoreF::from_complex_ref(chunk.slice_from(r * 9 + 1..));
            rows2[r] = AvxStoreF::from_complex_ref(chunk.slice_from(r * 9 + 5..));
        }

        let mid0 = self.bf4.exech(rows0);
        let mut mid1 = self.bf4.exec(rows1);
        let mut mid2 = self.bf4.exec(rows2);

        for r in 1..4 {
            mid1[r] = AvxStoreF::mul_by_complex(mid1[r], self.twiddles[2 * r - 2]);
            mid2[r] = AvxStoreF::mul_by_complex(mid2[r], self.twiddles[2 * r - 1]);
        }

        let transposed = transpose_9x4_to_4x9_emptycolumn1_f32(mid0, mid1, mid2);

        let output_rows = self.bf9.exec(transposed);

        output_rows[0].write(chunk.slice_from_mut(0..));
        output_rows[3].write(chunk.slice_from_mut(12..));
        output_rows[6].write(chunk.slice_from_mut(24..));

        output_rows[1].write(chunk.slice_from_mut(4..));
        output_rows[4].write(chunk.slice_from_mut(16..));
        output_rows[7].write(chunk.slice_from_mut(28..));

        output_rows[2].write(chunk.slice_from_mut(8..));
        output_rows[5].write(chunk.slice_from_mut(20..));
        output_rows[8].write(chunk.slice_from_mut(32..));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly36, f32, AvxButterfly36f, 36, 1e-4);
    test_oof_avx_butterfly!(test_avx_neon_butterfly36, f32, AvxButterfly36f, 36, 1e-4);
}
