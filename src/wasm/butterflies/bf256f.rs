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
use crate::wasm::column::ColumnButterfly16f;
use crate::wasm::store::WasmStoreF;
use crate::wasm::transpose::transpose_2x16;
use crate::wasm::{boring_wasm_butterfly, gen_butterfly_twiddles_f32};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct WasmButterfly256f {
    direction: FftDirection,
    bf16: ColumnButterfly16f,
    twiddles: [WasmStoreF; 120],
}

impl WasmButterfly256f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(16, 16, fft_direction, 256),
            bf16: ColumnButterfly16f::new(fft_direction),
        }
    }
}

impl WasmButterfly256f {
    #[inline]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows: [WasmStoreF; 16] = [WasmStoreF::default(); 16];
        let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 256];
        unsafe {
            // columns
            for k in 0..8 {
                for i in 0..16 {
                    rows[i] = WasmStoreF::from_complex_ref(chunk.slice_from(i * 16 + k * 2..));
                }

                rows = self.bf16.exec(rows);

                for i in 1..16 {
                    rows[i] = WasmStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 15 * k]);
                }

                let transposed = transpose_2x16(rows);

                for i in 0..8 {
                    transposed[i * 2].write_uninit(scratch.get_unchecked_mut(k * 2 * 16 + i * 2..));
                    transposed[i * 2 + 1]
                        .write_uninit(scratch.get_unchecked_mut((k * 2 + 1) * 16 + i * 2..));
                }
            }

            // rows

            for k in 0..8 {
                self.bf16.exec_streaming(
                    |i| WasmStoreF::from_complex_refu(scratch.get_unchecked(i * 16 + k * 2..)),
                    |i, v| {
                        v.write(chunk.slice_from_mut(i * 16 + k * 2..));
                    },
                )
            }
        }
    }
}

boring_wasm_butterfly!(WasmButterfly256f, f32, 256);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

    test_butterfly!(test_wasm_butterfly256, f32, WasmButterfly256f, 256, 1e-3);
    test_oof_butterfly!(
        test_oof_wasm_butterfly256,
        f32,
        WasmButterfly256f,
        256,
        1e-3
    );
}
