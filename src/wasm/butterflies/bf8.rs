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
use crate::wasm::column::{ColumnButterfly2f, ColumnButterfly4f, ColumnButterfly8d};
use crate::wasm::store::{WasmStoreD, WasmStoreF};
use crate::wasm::transpose::transpose_f32x2_4x2;
use crate::wasm::{boring_wasm_butterfly, gen_butterfly_twiddles_f32};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

pub(crate) struct WasmButterfly8d {
    direction: FftDirection,
    bf: ColumnButterfly8d,
}

impl WasmButterfly8d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf: ColumnButterfly8d::new(fft_direction),
        }
    }
}

impl WasmButterfly8d {
    #[inline]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let u0 = WasmStoreD::from_complex_ref(chunk.slice_from(0..));
        let u1 = WasmStoreD::from_complex_ref(chunk.slice_from(1..));
        let u2 = WasmStoreD::from_complex_ref(chunk.slice_from(2..));
        let u3 = WasmStoreD::from_complex_ref(chunk.slice_from(3..));
        let u4 = WasmStoreD::from_complex_ref(chunk.slice_from(4..));
        let u5 = WasmStoreD::from_complex_ref(chunk.slice_from(5..));
        let u6 = WasmStoreD::from_complex_ref(chunk.slice_from(6..));
        let u7 = WasmStoreD::from_complex_ref(chunk.slice_from(7..));

        let [y0, y1, y2, y3, y4, y5, y6, y7] = self.bf.exec([u0, u1, u2, u3, u4, u5, u6, u7]);

        y0.write(chunk.slice_from_mut(0..));
        y1.write(chunk.slice_from_mut(1..));
        y2.write(chunk.slice_from_mut(2..));
        y3.write(chunk.slice_from_mut(3..));
        y4.write(chunk.slice_from_mut(4..));
        y5.write(chunk.slice_from_mut(5..));
        y6.write(chunk.slice_from_mut(6..));
        y7.write(chunk.slice_from_mut(7..));
    }
}

boring_wasm_butterfly!(WasmButterfly8d, f64, 8);

pub(crate) struct WasmButterfly8f {
    direction: FftDirection,
    bf4: ColumnButterfly4f,
    bf2: ColumnButterfly2f,
    twiddles: [WasmStoreF; 2],
}

impl WasmButterfly8f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf4: ColumnButterfly4f::new(fft_direction),
            bf2: ColumnButterfly2f::new(fft_direction),
            twiddles: gen_butterfly_twiddles_f32(4, 2, fft_direction, 8),
        }
    }
}

impl WasmButterfly8f {
    #[inline]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows0: [WasmStoreF; 2] = [WasmStoreF::zero(); 2];
        let mut rows1: [WasmStoreF; 2] = [WasmStoreF::zero(); 2];
        // columns
        for i in 0..2 {
            rows0[i] = WasmStoreF::from_complex_ref(chunk.slice_from(i * 4..));
            rows1[i] = WasmStoreF::from_complex_ref(chunk.slice_from(i * 4 + 2..));
        }

        rows0 = self.bf2.exec(rows0);
        rows1 = self.bf2.exec(rows1);

        rows0[1] = WasmStoreF::mul_by_complex(rows0[1], self.twiddles[0]);
        rows1[1] = WasmStoreF::mul_by_complex(rows1[1], self.twiddles[1]);

        let transposed = transpose_f32x2_4x2(rows0, rows1);

        let q0 = self.bf4.exec(transposed);

        for i in 0..4 {
            q0[i].write(chunk.slice_from_mut(i * 2..));
        }
    }
}

boring_wasm_butterfly!(WasmButterfly8f, f32, 8);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::test_wasm_butterfly;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

    test_wasm_butterfly!(test_wasm_butterfly8, f32, WasmButterfly8f, 8, 1e-5);
    test_wasm_butterfly!(test_wasm_butterfly8_f64, f64, WasmButterfly8d, 8, 1e-7);
    test_wasm_butterfly!(test_oof_wasm_butterfly8, f32, WasmButterfly8f, 8, 1e-5);
}
