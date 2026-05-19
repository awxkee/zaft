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
use crate::wasm::column::ColumnButterfly16d;
use crate::wasm::store::WasmStoreD;
use crate::wasm::{boring_wasm_butterfly, gen_butterfly_twiddles_f64};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct WasmButterfly256d {
    direction: FftDirection,
    bf16: ColumnButterfly16d,
    twiddles: [WasmStoreD; 240],
}

impl WasmButterfly256d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(16, 16, fft_direction, 256),
            bf16: ColumnButterfly16d::new(fft_direction),
        }
    }
}

impl WasmButterfly256d {
    #[inline]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows: [WasmStoreD; 16] = [WasmStoreD::default(); 16];
        let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 256];
        unsafe {
            // columns
            for k in 0..16 {
                for i in 0..16 {
                    rows[i] = WasmStoreD::from_complex_ref(chunk.slice_from(i * 16 + k..));
                }

                rows = self.bf16.exec(rows);

                if k > 0 {
                    for i in 1..16 {
                        rows[i] =
                            WasmStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 15 * k]);
                    }
                }

                for i in 0..16 {
                    rows[i].write_uninit(scratch.get_unchecked_mut(k * 16 + i..));
                }
            }

            // rows

            for k in 0..16 {
                self.bf16.exec_streaming(
                    |i| WasmStoreD::from_complex_refu(scratch.get_unchecked(i * 16 + k..)),
                    |i, v| v.write(chunk.slice_from_mut(i * 16 + k..)),
                );
            }
        }
    }
}

boring_wasm_butterfly!(WasmButterfly256d, f64, 256);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::{test_wasm_butterfly, test_wasm_oof_butterfly};

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

    test_wasm_butterfly!(
        test_wasm_butterfly256_f64,
        f64,
        WasmButterfly256d,
        256,
        1e-7
    );
    test_wasm_oof_butterfly!(
        test_oof_wasm_butterfly256_f64,
        f64,
        WasmButterfly256d,
        256,
        1e-7
    );
}
