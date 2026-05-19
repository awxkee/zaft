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

macro_rules! define_bf5 {
    ($bf_name: ident, $bf_rotate: ident, $store: ident) => {
        use crate::wasm::rotate::$bf_rotate;
        use crate::wasm::store::$store;
        pub(crate) struct $bf_name {
            rotate: $bf_rotate,
            tw1_re: $store,
            tw1_im: $store,
            tw2_re: $store,
            tw2_im: $store,
        }

        impl $bf_name {
            pub(crate) fn new(direction: FftDirection) -> Self {
                let tw1 = compute_twiddle(1, 5, direction);
                let tw2 = compute_twiddle(2, 5, direction);
                Self {
                    rotate: $bf_rotate::new(FftDirection::Inverse),
                    tw1_re: $store::dup(tw1.re),
                    tw1_im: $store::dup(tw1.im),
                    tw2_re: $store::dup(tw2.re),
                    tw2_im: $store::dup(tw2.im),
                }
            }

            #[inline(always)]
            pub(crate) fn exec(&self, v: [$store; 5]) -> [$store; 5] {
                let x14p = v[1] + v[4];
                let x14n = v[1] - v[4];
                let x23p = v[2] + v[3];
                let x23n = v[2] - v[3];
                let y0 = v[0] + x14p + x23p;

                let temp_b1_1 = self.tw1_im * x14n;
                let temp_b2_1 = self.tw2_im * x14n;

                let temp_a1 = v[0] + (self.tw1_re * x14p + self.tw2_re * x23p);
                let temp_a2 = v[0] + (self.tw2_re * x14p + self.tw1_re * x23p);

                let temp_b1 = temp_b1_1 + self.tw2_im * x23n;
                let temp_b2 = temp_b2_1 - self.tw1_im * x23n;

                let temp_b1_rot = self.rotate.rotate(temp_b1);
                let temp_b2_rot = self.rotate.rotate(temp_b2);

                let y1 = temp_a1 + temp_b1_rot;
                let y2 = temp_a2 + temp_b2_rot;
                let y3 = temp_a2 - temp_b2_rot;
                let y4 = temp_a1 - temp_b1_rot;

                [y0, y1, y2, y3, y4]
            }
        }
    };
}

define_bf5!(ColumnButterfly5d, WasmRotate90D, WasmStoreD);
define_bf5!(ColumnButterfly5f, WasmRotate90F, WasmStoreF);
