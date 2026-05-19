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
use num_traits::MulAdd;

macro_rules! column_butterfly_7 {
    ($bf_name: ident, $store: ident, $bf2_name: ident, $rotate: ident) => {
        use crate::wasm::column::bf2::$bf2_name;
        use crate::wasm::rotate::$rotate;
        use crate::wasm::store::$store;
        pub(crate) struct $bf_name {
            rotate: $rotate,
            bf2: $bf2_name,
            twiddle1_re: $store,
            twiddle2_re: $store,
            twiddle3_re: $store,
            twiddle1_im: $store,
            twiddle2_im: $store,
            twiddle3_im: $store,
        }

        impl $bf_name {
            pub(crate) fn new(direction: FftDirection) -> Self {
                let twiddle1 = compute_twiddle(1, 7, direction);
                let twiddle2 = compute_twiddle(2, 7, direction);
                let twiddle3 = compute_twiddle(3, 7, direction);
                Self {
                    rotate: $rotate::new(FftDirection::Inverse),
                    bf2: $bf2_name::new(direction),
                    twiddle1_re: $store::dup(twiddle1.re),
                    twiddle1_im: $store::dup(twiddle1.im),
                    twiddle2_re: $store::dup(twiddle2.re),
                    twiddle2_im: $store::dup(twiddle2.im),
                    twiddle3_re: $store::dup(twiddle3.re),
                    twiddle3_im: $store::dup(twiddle3.im),
                }
            }
        }

        impl $bf_name {
            #[inline(always)]
            pub(crate) fn exec(&self, v: [$store; 7]) -> [$store; 7] {
                let [x1p6, x1m6] = self.bf2.exec([v[1], v[6]]);
                let x1m6 = self.rotate.rotate(x1m6);
                let y00 = v[0] + x1p6;
                let [x2p5, x2m5] = self.bf2.exec([v[2], v[5]]);
                let x2m5 = self.rotate.rotate(x2m5);
                let y00 = y00 + x2p5;
                let [x3p4, x3m4] = self.bf2.exec([v[3], v[4]]);
                let x3m4 = self.rotate.rotate(x3m4);
                let y00 = y00 + x3p4;

                let m0106a = x1p6.mul_add(self.twiddle1_re, v[0]);
                let m0106a = x2p5.mul_add(self.twiddle2_re, m0106a);
                let m0106a = x3p4.mul_add(self.twiddle3_re, m0106a);
                let m0106b = x1m6 * self.twiddle1_im;
                let m0106b = x2m5.mul_add(self.twiddle2_im, m0106b);
                let m0106b = x3m4.mul_add(self.twiddle3_im, m0106b);
                let [y01, y06] = self.bf2.exec([m0106a, m0106b]);

                let m0205a = x1p6.mul_add(self.twiddle2_re, v[0]);
                let m0205a = x2p5.mul_add(self.twiddle3_re, m0205a);
                let m0205a = x3p4.mul_add(self.twiddle1_re, m0205a);
                let m0205b = x1m6 * self.twiddle2_im;
                let m0205b = x2m5.mul_nadd(self.twiddle3_im, m0205b);
                let m0205b = x3m4.mul_nadd(self.twiddle1_im, m0205b);
                let [y02, y05] = self.bf2.exec([m0205a, m0205b]);

                let m0304a = x1p6.mul_add(self.twiddle3_re, v[0]);
                let m0304a = x2p5.mul_add(self.twiddle1_re, m0304a);
                let m0304a = x3p4.mul_add(self.twiddle2_re, m0304a);
                let m0304b = x1m6 * self.twiddle3_im;
                let m0304b = x2m5.mul_nadd(self.twiddle1_im, m0304b);
                let m0304b = x3m4.mul_add(self.twiddle2_im, m0304b);
                let [y03, y04] = self.bf2.exec([m0304a, m0304b]);

                [y00, y01, y02, y03, y04, y05, y06]
            }
        }
    };
}

column_butterfly_7!(
    ColumnButterfly7d,
    WasmStoreD,
    ColumnButterfly2d,
    WasmRotate90D
);
column_butterfly_7!(
    ColumnButterfly7f,
    WasmStoreF,
    ColumnButterfly2f,
    WasmRotate90F
);
