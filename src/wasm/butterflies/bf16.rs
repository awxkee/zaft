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
use crate::FftDirection;
use crate::store::BidirectionalStore;
use crate::wasm::column::{ColumnButterfly4f, ColumnButterfly16d};
use crate::wasm::store::{WasmStoreD, WasmStoreF};
use crate::wasm::transpose::transpose_f32x2_4x4;
use crate::wasm::{boring_wasm_butterfly, gen_butterfly_twiddles_f32};
use crate::{FftExecutor, ZaftError};
use num_complex::Complex;

pub(crate) struct WasmButterfly16d {
    direction: FftDirection,
    bf16: ColumnButterfly16d,
}

impl WasmButterfly16d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf16: ColumnButterfly16d::new(fft_direction),
        }
    }
}

impl WasmButterfly16d {
    #[inline]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows = [WasmStoreD::default(); 16];
        for i in 0..16 {
            rows[i] = WasmStoreD::from_complex_ref(chunk.slice_from(i..));
        }
        self.bf16
            .exec_streaming(|i| rows[i], |i, v| v.write(chunk.slice_from_mut(i..)));
    }
}

boring_wasm_butterfly!(WasmButterfly16d, f64, 16);

pub(crate) struct WasmButterfly16f {
    direction: FftDirection,
    bf4: ColumnButterfly4f,
    twiddles: [WasmStoreF; 6],
}

impl WasmButterfly16f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(4, 4, fft_direction, 16),
            bf4: ColumnButterfly4f::new(fft_direction),
        }
    }
}

impl WasmButterfly16f {
    #[inline]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows0: [WasmStoreF; 4] = [WasmStoreF::zero(); 4];
        let mut rows1: [WasmStoreF; 4] = [WasmStoreF::zero(); 4];
        // columns
        for i in 0..4 {
            rows0[i] = WasmStoreF::from_complex_ref(chunk.slice_from(i * 4..));
            rows1[i] = WasmStoreF::from_complex_ref(chunk.slice_from(i * 4 + 2..));
        }

        rows0 = self.bf4.exec(rows0);
        rows1 = self.bf4.exec(rows1);

        for i in 1..4 {
            rows0[i] = WasmStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
            rows1[i] = WasmStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1 + 3]);
        }

        let transposed = transpose_f32x2_4x4(rows0, rows1);

        let q0 = self.bf4.exec(transposed.0);
        let q1 = self.bf4.exec(transposed.1);

        for i in 0..4 {
            q0[i].write(chunk.slice_from_mut(i * 4..));
            q1[i].write(chunk.slice_from_mut(i * 4 + 2..));
        }
    }
}

boring_wasm_butterfly!(WasmButterfly16f, f32, 16);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    use crate::wasm::{test_wasm_butterfly, test_wasm_oof_butterfly};

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

    test_butterfly!(test_wasm_butterfly16, f32, WasmButterfly16f, 16, 1e-5);
    test_wasm_butterfly!(test_wasm_butterfly16_f64, f64, WasmButterfly16d, 16, 1e-7);

    test_oof_butterfly!(test_oof_wasm_butterfly16, f32, WasmButterfly16f, 16, 1e-5);
    test_wasm_oof_butterfly!(
        test_oof_wasm_butterfly16_f64,
        f64,
        WasmButterfly16d,
        16,
        1e-9
    );
}
