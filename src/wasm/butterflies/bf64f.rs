/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

use crate::store::BidirectionalStore;
use crate::wasm::column::ColumnButterfly8f;
use crate::wasm::store::WasmStoreF;
use crate::wasm::transpose::transpose_f32x2_2x2;
use crate::wasm::{boring_wasm_butterfly, gen_butterfly_twiddles_f32};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

#[inline]
pub(crate) fn transpose_f32x2_8x2(rows: [WasmStoreF; 8]) -> [WasmStoreF; 8] {
    // matrix transpose implementation (8x2 -> 2x8):
    // [ A B ]^T => [ A^T C^T E^T G^T I^T K^T M^T O^T ]
    // [ C D ]      [ B^T D^T F^T H^T J^T L^T N^T P^T ]
    // [ E F ]
    // [ G H ]
    // [ I J ]
    // [ K L ]
    // [ M N ]
    // [ O P ]
    let a0 = transpose_f32x2_2x2([rows[0], rows[1]]);
    let b0 = transpose_f32x2_2x2([rows[2], rows[3]]);
    let c0 = transpose_f32x2_2x2([rows[4], rows[5]]);
    let d0 = transpose_f32x2_2x2([rows[6], rows[7]]);
    [a0[0], a0[1], b0[0], b0[1], c0[0], c0[1], d0[0], d0[1]]
}

#[inline]
pub(crate) fn transpose_8x8_f32(
    rows0: [WasmStoreF; 8],
    rows1: [WasmStoreF; 8],
    rows2: [WasmStoreF; 8],
    rows3: [WasmStoreF; 8],
) -> (
    [WasmStoreF; 8],
    [WasmStoreF; 8],
    [WasmStoreF; 8],
    [WasmStoreF; 8],
) {
    // matrix transpose implementation (8x2 -> 2x8):
    // [ A B ]^T => [ A^T C^T E^T G^T I^T K^T M^T O^T ]
    // [ C D ]      [ B^T D^T F^T H^T J^T L^T N^T P^T ]
    // [ E F ]
    // [ G H ]
    // [ I J ]
    // [ K L ]
    // [ M N ]
    // [ O P ]

    let transposed00 = transpose_f32x2_8x2(rows0);
    let transposed01 = transpose_f32x2_8x2(rows1);
    let transposed10 = transpose_f32x2_8x2(rows2);
    let transposed11 = transpose_f32x2_8x2(rows3);

    (
        [
            transposed00[0],
            transposed00[1],
            transposed01[0],
            transposed01[1],
            transposed10[0],
            transposed10[1],
            transposed11[0],
            transposed11[1],
        ],
        [
            transposed00[2],
            transposed00[3],
            transposed01[2],
            transposed01[3],
            transposed10[2],
            transposed10[3],
            transposed11[2],
            transposed11[3],
        ],
        [
            transposed00[4],
            transposed00[5],
            transposed01[4],
            transposed01[5],
            transposed10[4],
            transposed10[5],
            transposed11[4],
            transposed11[5],
        ],
        [
            transposed00[6],
            transposed00[7],
            transposed01[6],
            transposed01[7],
            transposed10[6],
            transposed10[7],
            transposed11[6],
            transposed11[7],
        ],
    )
}

pub(crate) struct WasmButterfly64f {
    direction: FftDirection,
    twiddles: Box<[WasmStoreF; 28]>,
    bf8: ColumnButterfly8f,
}

impl WasmButterfly64f {
    pub(crate) fn new(direction: FftDirection) -> Self {
        Self {
            direction,
            twiddles: Box::new(gen_butterfly_twiddles_f32(8, 8, direction, 64)),
            bf8: ColumnButterfly8f::new(direction),
        }
    }
}

impl WasmButterfly64f {
    #[inline]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows0 = [WasmStoreF::default(); 8];
        let mut rows1 = [WasmStoreF::default(); 8];
        let mut rows2 = [WasmStoreF::default(); 8];
        let mut rows3 = [WasmStoreF::default(); 8];
        for r in 0..8 {
            rows0[r] = WasmStoreF::from_complex_ref(chunk.slice_from(8 * r..));
        }
        let mut mid0 = self.bf8.exec(rows0);

        for r in 1..8 {
            mid0[r] = WasmStoreF::mul_by_complex(mid0[r], self.twiddles[r - 1]);
        }

        for r in 0..8 {
            rows1[r] = WasmStoreF::from_complex_ref(chunk.slice_from(8 * r + 2..));
        }

        let mut mid1 = self.bf8.exec(rows1);

        for r in 1..8 {
            mid1[r] = WasmStoreF::mul_by_complex(mid1[r], self.twiddles[r - 1 + 7]);
        }

        for r in 0..8 {
            rows2[r] = WasmStoreF::from_complex_ref(chunk.slice_from(8 * r + 4..));
        }

        let mut mid2 = self.bf8.exec(rows2);

        for r in 1..8 {
            mid2[r] = WasmStoreF::mul_by_complex(mid2[r], self.twiddles[r - 1 + 7 * 2]);
        }

        for r in 0..8 {
            rows3[r] = WasmStoreF::from_complex_ref(chunk.slice_from(8 * r + 6..));
        }

        let mut mid3 = self.bf8.exec(rows3);

        for r in 1..8 {
            mid3[r] = WasmStoreF::mul_by_complex(mid3[r], self.twiddles[r - 1 + 7 * 3]);
        }

        let (transposed0, transposed1, transposed2, transposed3) =
            transpose_8x8_f32(mid0, mid1, mid2, mid3);

        let output0 = self.bf8.exec(transposed0);
        for r in 0..8 {
            output0[r].write(chunk.slice_from_mut(8 * r..));
        }

        let output1 = self.bf8.exec(transposed1);
        for r in 0..8 {
            output1[r].write(chunk.slice_from_mut(8 * r + 2..));
        }

        let output2 = self.bf8.exec(transposed2);
        for r in 0..8 {
            output2[r].write(chunk.slice_from_mut(8 * r + 4..));
        }

        let output3 = self.bf8.exec(transposed3);
        for r in 0..8 {
            output3[r].write(chunk.slice_from_mut(8 * r + 6..));
        }
    }
}

boring_wasm_butterfly!(WasmButterfly64f, f32, 64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::{test_wasm_butterfly, test_wasm_oof_butterfly};

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

    test_wasm_butterfly!(test_wasm_butterfly64, f32, WasmButterfly64f, 64, 1e-4);
    test_wasm_oof_butterfly!(test_oof_wasm_butterfly64, f32, WasmButterfly64f, 64, 1e-4);
}
