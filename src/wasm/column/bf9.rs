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
use crate::util::compute_twiddle;

macro_rules! generate_bf9 {
    ($bf_name: ident, $store: ident, $bf3_name: ident) => {
        use crate::wasm::column::$bf3_name;
        use crate::wasm::store::$store;

        pub(crate) struct $bf_name {
            tw1: $store,
            tw2: $store,
            tw4: $store,
            pub(crate) bf3: $bf3_name,
        }

        impl $bf_name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                let tw1 = compute_twiddle(1, 9, fft_direction);
                let tw2 = compute_twiddle(2, 9, fft_direction);
                let tw4 = compute_twiddle(4, 9, fft_direction);
                Self {
                    tw1: $store::from_complex(&tw1),
                    tw2: $store::from_complex(&tw2),
                    tw4: $store::from_complex(&tw4),
                    bf3: $bf3_name::new(fft_direction),
                }
            }

            #[inline(always)]
            pub(crate) fn exec(&self, store: [$store; 9]) -> [$store; 9] {
                let [u0, u3, u6] = self.bf3.exec([store[0], store[3], store[6]]);
                let [u1, mut u4, mut u7] = self.bf3.exec([store[1], store[4], store[7]]);
                let [u2, mut u5, mut u8] = self.bf3.exec([store[2], store[5], store[8]]);

                u4 = $store::mul_by_complex(u4, self.tw1);
                u7 = $store::mul_by_complex(u7, self.tw2);
                u5 = $store::mul_by_complex(u5, self.tw2);
                u8 = $store::mul_by_complex(u8, self.tw4);

                let [y0, y3, y6] = self.bf3.exec([u0, u1, u2]);
                let [y1, y4, y7] = self.bf3.exec([u3, u4, u5]);
                let [y2, y5, y8] = self.bf3.exec([u6, u7, u8]);
                [y0, y1, y2, y3, y4, y5, y6, y7, y8]
            }
        }
    };
}

generate_bf9!(ColumnButterfly9d, WasmStoreD, ColumnButterfly3d);
generate_bf9!(ColumnButterfly9f, WasmStoreF, ColumnButterfly3f);
