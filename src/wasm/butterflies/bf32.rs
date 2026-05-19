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
use crate::store::BidirectionalStore;
use crate::wasm::column::{ColumnButterfly4f, ColumnButterfly8f, ColumnButterfly32d};
use crate::wasm::store::WasmStoreF;
use crate::wasm::transpose::transpose_f32x2_2x2;
use crate::wasm::{boring_wasm_butterfly, gen_butterfly_twiddles_f32};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

pub(crate) struct WasmButterfly32d {
    direction: FftDirection,
    bf32: ColumnButterfly32d,
}

impl WasmButterfly32d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf32: ColumnButterfly32d::new(fft_direction),
        }
    }
}

impl WasmButterfly32d {
    #[inline]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        self.bf32.exec_store(chunk);
    }
}

boring_wasm_butterfly!(WasmButterfly32d, f64, 32);

#[inline(always)]
pub(crate) fn transpose_8x4_to_4x8_f32(
    rows0: [WasmStoreF; 4],
    rows1: [WasmStoreF; 4],
    rows2: [WasmStoreF; 4],
    rows3: [WasmStoreF; 4],
) -> ([WasmStoreF; 8], [WasmStoreF; 8]) {
    let output00 = transpose_f32x2_2x2([rows0[0], rows0[1]]);
    let output01 = transpose_f32x2_2x2([rows1[0], rows1[1]]);
    let output02 = transpose_f32x2_2x2([rows2[0], rows2[1]]);
    let output03 = transpose_f32x2_2x2([rows3[0], rows3[1]]);
    let output10 = transpose_f32x2_2x2([rows0[2], rows0[3]]);
    let output11 = transpose_f32x2_2x2([rows1[2], rows1[3]]);
    let output12 = transpose_f32x2_2x2([rows2[2], rows2[3]]);
    let output13 = transpose_f32x2_2x2([rows3[2], rows3[3]]);

    (
        [
            output00[0],
            output00[1],
            output01[0],
            output01[1],
            output02[0],
            output02[1],
            output03[0],
            output03[1],
        ],
        [
            output10[0],
            output10[1],
            output11[0],
            output11[1],
            output12[0],
            output12[1],
            output13[0],
            output13[1],
        ],
    )
}

pub(crate) struct WasmButterfly32f {
    direction: FftDirection,
    bf8: ColumnButterfly8f,
    bf4: ColumnButterfly4f,
    twiddles: [WasmStoreF; 12],
}

impl WasmButterfly32f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf8: ColumnButterfly8f::new(fft_direction),
            bf4: ColumnButterfly4f::new(fft_direction),
            twiddles: gen_butterfly_twiddles_f32(8, 4, fft_direction, 32),
        }
    }
}

impl WasmButterfly32f {
    #[inline]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows0: [WasmStoreF; 4] = [WasmStoreF::default(); 4];
        let mut rows1: [WasmStoreF; 4] = [WasmStoreF::default(); 4];
        let mut rows2: [WasmStoreF; 4] = [WasmStoreF::default(); 4];
        let mut rows3: [WasmStoreF; 4] = [WasmStoreF::default(); 4];
        for i in 0..4 {
            rows0[i] = WasmStoreF::from_complex_ref(chunk.slice_from(i * 8..));
            rows1[i] = WasmStoreF::from_complex_ref(chunk.slice_from(i * 8 + 2..));
            rows2[i] = WasmStoreF::from_complex_ref(chunk.slice_from(i * 8 + 4..));
            rows3[i] = WasmStoreF::from_complex_ref(chunk.slice_from(i * 8 + 6..));
        }

        rows0 = self.bf4.exec(rows0);
        rows1 = self.bf4.exec(rows1);
        rows2 = self.bf4.exec(rows2);
        rows3 = self.bf4.exec(rows3);

        for i in 1..4 {
            rows0[i] = WasmStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
            rows1[i] = WasmStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1 + 3]);
            rows2[i] = WasmStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 6]);
            rows3[i] = WasmStoreF::mul_by_complex(rows3[i], self.twiddles[i - 1 + 9]);
        }

        let (mut q0, mut q1) = transpose_8x4_to_4x8_f32(rows0, rows1, rows2, rows3);

        q0 = self.bf8.exec(q0);
        q1 = self.bf8.exec(q1);

        for i in 0..8 {
            q0[i].write(chunk.slice_from_mut(i * 4..));
            q1[i].write(chunk.slice_from_mut(i * 4 + 2..));
        }
    }
}

boring_wasm_butterfly!(WasmButterfly32f, f32, 32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::{test_wasm_butterfly, test_wasm_oof_butterfly};

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

    test_wasm_butterfly!(test_wasm_butterfly32, f32, WasmButterfly32f, 32, 1e-5);
    test_wasm_butterfly!(test_wasm_butterfly32_f64, f64, WasmButterfly32d, 32, 1e-7);
    test_wasm_oof_butterfly!(test_oof_wasm_butterfly32, f32, WasmButterfly32f, 32, 1e-5);
    test_wasm_oof_butterfly!(
        test_oof_wasm_butterfly32_f64,
        f64,
        WasmButterfly32d,
        32,
        1e-9
    );
}
