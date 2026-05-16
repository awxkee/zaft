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
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::transpose_2x2;

macro_rules! define_bf40 {
    ($bf_name: ident, $features: literal, $inner_bf8: ident, $inner_bf5: ident, $mul: ident) => {
        use crate::neon::mixed::{$inner_bf5, $inner_bf8};
        pub(crate) struct $bf_name {
            pub(crate) bf8: $inner_bf8,
            pub(crate) bf5: $inner_bf5,
        }

        impl $bf_name {
            pub(crate) fn new(direction: FftDirection) -> Self {
                Self {
                    bf8: $inner_bf8::new(direction),
                    bf5: $inner_bf5::new(direction),
                }
            }

            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn exec_transpose_streaming<
                A: Fn(usize) -> NeonStoreF,
                T: Fn(usize) -> NeonStoreF,
                J: FnMut(usize, NeonStoreF),
            >(
                &self,
                v: A,
                twiddle: T,
                mut store: J,
            ) {
                let input0 = [v(0), v(16), v(32), v(8), v(24)];
                let mid0 = self.bf5.exec(input0);
                let input1 = [v(25), v(1), v(17), v(33), v(9)];
                let mid1 = self.bf5.exec(input1);
                let input2 = [v(10), v(26), v(2), v(18), v(34)];
                let mid2 = self.bf5.exec(input2);
                let input3 = [v(35), v(11), v(27), v(3), v(19)];
                let mid3 = self.bf5.exec(input3);
                let input4 = [v(20), v(36), v(12), v(28), v(4)];
                let mid4 = self.bf5.exec(input4);
                let input5 = [v(5), v(21), v(37), v(13), v(29)];
                let mid5 = self.bf5.exec(input5);
                let input6 = [v(30), v(6), v(22), v(38), v(14)];
                let mid6 = self.bf5.exec(input6);
                let input7 = [v(15), v(31), v(7), v(23), v(39)];
                let mid7 = self.bf5.exec(input7);

                let cols1 = self.bf8.exec([
                    mid0[1], mid1[1], mid2[1], mid3[1], mid4[1], mid5[1], mid6[1], mid7[1],
                ]);
                let cols2 = self.bf8.exec([
                    mid0[2], mid1[2], mid2[2], mid3[2], mid4[2], mid5[2], mid6[2], mid7[2],
                ]);
                let cols3 = self.bf8.exec([
                    mid0[3], mid1[3], mid2[3], mid3[3], mid4[3], mid5[3], mid6[3], mid7[3],
                ]);
                let cols4 = self.bf8.exec([
                    mid0[4], mid1[4], mid2[4], mid3[4], mid4[4], mid5[4], mid6[4], mid7[4],
                ]);

                let cols0 = self.bf8.exec([
                    mid0[0], mid1[0], mid2[0], mid3[0], mid4[0], mid5[0], mid6[0], mid7[0],
                ]);

                {
                    let t = transpose_2x2([cols0[0], NeonStoreF::$mul(cols2[5], twiddle(0))]);
                    store(0, t[0]);
                    store(1, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols4[2], twiddle(1)),
                        NeonStoreF::$mul(cols1[7], twiddle(2)),
                    ]);
                    store(2, t[0]);
                    store(3, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols3[4], twiddle(3)),
                        NeonStoreF::$mul(cols0[1], twiddle(4)),
                    ]);
                    store(4, t[0]);
                    store(5, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols2[6], twiddle(5)),
                        NeonStoreF::$mul(cols4[3], twiddle(6)),
                    ]);
                    store(6, t[0]);
                    store(7, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols1[0], twiddle(7)),
                        NeonStoreF::$mul(cols3[5], twiddle(8)),
                    ]);
                    store(8, t[0]);
                    store(9, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols0[2], twiddle(9)),
                        NeonStoreF::$mul(cols2[7], twiddle(10)),
                    ]);
                    store(10, t[0]);
                    store(11, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols4[4], twiddle(11)),
                        NeonStoreF::$mul(cols1[1], twiddle(12)),
                    ]);
                    store(12, t[0]);
                    store(13, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols3[6], twiddle(13)),
                        NeonStoreF::$mul(cols0[3], twiddle(14)),
                    ]);
                    store(14, t[0]);
                    store(15, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols2[0], twiddle(15)),
                        NeonStoreF::$mul(cols4[5], twiddle(16)),
                    ]);
                    store(16, t[0]);
                    store(17, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols1[2], twiddle(17)),
                        NeonStoreF::$mul(cols3[7], twiddle(18)),
                    ]);
                    store(18, t[0]);
                    store(19, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols0[4], twiddle(19)),
                        NeonStoreF::$mul(cols2[1], twiddle(20)),
                    ]);
                    store(20, t[0]);
                    store(21, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols4[6], twiddle(21)),
                        NeonStoreF::$mul(cols1[3], twiddle(22)),
                    ]);
                    store(22, t[0]);
                    store(23, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols3[0], twiddle(23)),
                        NeonStoreF::$mul(cols0[5], twiddle(24)),
                    ]);
                    store(24, t[0]);
                    store(25, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols2[2], twiddle(25)),
                        NeonStoreF::$mul(cols4[7], twiddle(26)),
                    ]);
                    store(26, t[0]);
                    store(27, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols1[4], twiddle(27)),
                        NeonStoreF::$mul(cols3[1], twiddle(28)),
                    ]);
                    store(28, t[0]);
                    store(29, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols0[6], twiddle(29)),
                        NeonStoreF::$mul(cols2[3], twiddle(30)),
                    ]);
                    store(30, t[0]);
                    store(31, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols4[0], twiddle(31)),
                        NeonStoreF::$mul(cols1[5], twiddle(32)),
                    ]);
                    store(32, t[0]);
                    store(33, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols3[2], twiddle(33)),
                        NeonStoreF::$mul(cols0[7], twiddle(34)),
                    ]);
                    store(34, t[0]);
                    store(35, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols2[4], twiddle(35)),
                        NeonStoreF::$mul(cols4[1], twiddle(36)),
                    ]);
                    store(36, t[0]);
                    store(37, t[1]);
                }
                {
                    let t = transpose_2x2([
                        NeonStoreF::$mul(cols1[6], twiddle(37)),
                        NeonStoreF::$mul(cols3[3], twiddle(38)),
                    ]);
                    store(38, t[0]);
                    store(39, t[1]);
                }
            }
        }
    };
}

define_bf40!(
    ColumnButterfly40f,
    "neon",
    ColumnButterfly8f,
    ColumnButterfly5f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_bf40!(
    ColumnFcmaButterfly40f,
    "fcma",
    ColumnFcmaButterfly8f,
    ColumnFcmaButterfly5f,
    fcmul_fcma
);
