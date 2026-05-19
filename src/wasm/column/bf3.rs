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

macro_rules! define_column_bf3 {
    ($bf_name: ident, $store: ident) => {
        use crate::wasm::store::$store;

        pub(crate) struct $bf_name {
            tw_re: $store,
            tw_im: $store,
        }

        impl $bf_name {
            pub(crate) fn new(direction: FftDirection) -> Self {
                let twiddle = compute_twiddle(1, 3, direction);
                Self {
                    tw_re: $store::dup(twiddle.re),
                    tw_im: $store::load(&[-twiddle.im, twiddle.im, -twiddle.im, twiddle.im]),
                }
            }

            #[inline(always)]
            pub(crate) fn exec(&self, store: [$store; 3]) -> [$store; 3] {
                let xp = store[1] + store[2];
                let xn = store[1] - store[2];
                let sum = store[0] + xp;

                let w_1 = store[0] + self.tw_re * xp;

                let xn_rot = xn.reverse_complex_elements();

                let tw_xn = self.tw_im * xn_rot;
                let y1 = w_1 + tw_xn;
                let y2 = w_1 - tw_xn;

                [sum, y1, y2]
            }
        }
    };
}

define_column_bf3!(ColumnButterfly3d, WasmStoreD);
define_column_bf3!(ColumnButterfly3f, WasmStoreF);
