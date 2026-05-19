/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use std::ops::Neg;

macro_rules! define_butterfly_8 {
    ($bf_name: ident, $store: ident, $bf4_name: ident, $bf2_name: ident, $val: expr) => {
        use crate::wasm::column::bf2::$bf2_name;
        use crate::wasm::column::$bf4_name;
        use crate::wasm::store::$store;
        pub(crate) struct $bf_name {
            pub(crate) bf4: $bf4_name,
            pub(crate) bf2: $bf2_name,
            root2: $store,
        }

        impl $bf_name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    bf4: $bf4_name::new(fft_direction),
                    bf2: $bf2_name::new(fft_direction),
                    root2: $store::dup($val.sqrt()),
                }
            }

            #[inline(always)]
            pub(crate) fn rotate(&self, v: $store) -> $store {
                self.bf4.rotate.rotate(v)
            }

            #[inline(always)]
            pub(crate) fn rotate45(&self, v: $store) -> $store {
                (self.bf4.rotate.rotate(v) + v) * self.root2
            }

            #[inline(always)]
            pub(crate) fn rotate135(&self, v: $store) -> $store {
                (self.bf4.rotate.rotate(v) - v) * self.root2
            }

            #[allow(unused)]
            #[inline(always)]
            pub(crate) fn rotate225(&self, v: $store) -> $store {
                (self.bf4.rotate.rotate(v) + v).neg() * self.root2
            }

            #[allow(unused)]
            #[inline(always)]
            pub(crate) fn rotate270(&self, v: $store) -> $store {
                self.bf4.rotate.rotate(v).neg()
            }

            #[inline(always)]
            pub(crate) fn exec(&self, store: [$store; 8]) -> [$store; 8] {
                let [u0, u2, u4, u6] = self.bf4.exec([store[0], store[2], store[4], store[6]]);
                let [u1, u3, u5, u7] = self.bf4.exec([store[1], store[3], store[5], store[7]]);

                let u3 = self.rotate45(u3);
                let u5 = self.rotate(u5);
                let u7 = self.rotate135(u7);

                let [y0, y1] = self.bf2.exec([u0, u1]);
                let [y2, y3] = self.bf2.exec([u2, u3]);
                let [y4, y5] = self.bf2.exec([u4, u5]);
                let [y6, y7] = self.bf2.exec([u6, u7]);
                [y0, y2, y4, y6, y1, y3, y5, y7]
            }
        }
    };
}

define_butterfly_8!(
    ColumnButterfly8d,
    WasmStoreD,
    ColumnButterfly4d,
    ColumnButterfly2d,
    0.5f64
);
define_butterfly_8!(
    ColumnButterfly8f,
    WasmStoreF,
    ColumnButterfly4f,
    ColumnButterfly2f,
    0.5f32
);
