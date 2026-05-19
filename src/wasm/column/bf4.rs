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

macro_rules! define_butterfly4 {
    ($bf_name: ident, $store:ident, $rotate: ident) => {
        use crate::wasm::rotate::$rotate;
        use crate::wasm::store::$store;
        pub(crate) struct $bf_name {
            pub(crate) rotate: $rotate,
        }

        impl $bf_name {
            pub(crate) fn new(direction: FftDirection) -> Self {
                Self {
                    rotate: $rotate::new(direction),
                }
            }

            #[inline(always)]
            pub(crate) fn exec(&self, store: [$store; 4]) -> [$store; 4] {
                let t0 = store[0] + store[2];
                let t1 = store[0] - store[2];
                let t2 = store[1] + store[3];
                let mut t3 = store[1] - store[3];
                t3 = self.rotate.rotate(t3);
                [t0 + t2, t1 + t3, t0 - t2, t1 - t3]
            }
        }
    };
}

define_butterfly4!(ColumnButterfly4d, WasmStoreD, WasmRotate90D);
define_butterfly4!(ColumnButterfly4f, WasmStoreF, WasmRotate90F);
