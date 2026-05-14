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
use crate::util::compute_twiddle;

macro_rules! define_column_bf32 {
    ($name: ident, $features: literal, $bf_name: ident, $mul: ident) => {
        use crate::neon::mixed::$bf_name;
        pub(crate) struct $name {
            pub(crate) bf16: $bf_name,
            pub(crate) twiddles32: [NeonStoreF; 6],
        }

        impl $name {
            pub(crate) fn new(direction: FftDirection) -> Self {
                Self {
                    bf16: $bf_name::new(direction),
                    twiddles32: [
                        NeonStoreF::from_complex(&compute_twiddle(1, 32, direction)),
                        NeonStoreF::from_complex(&compute_twiddle(2, 32, direction)),
                        NeonStoreF::from_complex(&compute_twiddle(3, 32, direction)),
                        NeonStoreF::from_complex(&compute_twiddle(5, 32, direction)),
                        NeonStoreF::from_complex(&compute_twiddle(6, 32, direction)),
                        NeonStoreF::from_complex(&compute_twiddle(7, 32, direction)),
                    ],
                }
            }

            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn exec_streaming<
                A: Fn(usize) -> NeonStoreF,
                J: FnMut(usize, NeonStoreF),
            >(
                &self,
                v: A,
                mut store: J,
            ) {
                let input1 = [v(1), v(9), v(17), v(25)];
                let mut mid1 = self.bf16.bf8.bf4.exec(input1);

                mid1[1] = NeonStoreF::$mul(mid1[1], self.twiddles32[0]);
                mid1[2] = NeonStoreF::$mul(mid1[2], self.twiddles32[1]);
                mid1[3] = NeonStoreF::$mul(mid1[3], self.twiddles32[2]);

                let input2 = [v(2), v(10), v(18), v(26)];
                let mut mid2 = self.bf16.bf8.bf4.exec(input2);

                mid2[1] = NeonStoreF::$mul(mid2[1], self.twiddles32[1]);
                mid2[2] = self.bf16.bf8.rotate45(mid2[2]);
                mid2[3] = NeonStoreF::$mul(mid2[3], self.twiddles32[4]);

                let input3 = [v(3), v(11), v(19), v(27)];
                let mut mid3 = self.bf16.bf8.bf4.exec(input3);

                mid3[1] = NeonStoreF::$mul(mid3[1], self.twiddles32[2]);
                mid3[2] = NeonStoreF::$mul(mid3[2], self.twiddles32[4]);
                mid3[3] = NeonStoreF::$mul(mid3[3], self.bf16.bf8.rotate(self.twiddles32[0]));

                let input4 = [v(4), v(12), v(20), v(28)];
                let mut mid4 = self.bf16.bf8.bf4.exec(input4);

                mid4[1] = self.bf16.bf8.rotate45(mid4[1]);
                mid4[2] = self.bf16.bf8.rotate(mid4[2]);
                mid4[3] = self.bf16.bf8.rotate135(mid4[3]);

                let input5 = [v(5), v(13), v(21), v(29)];
                let mut mid5 = self.bf16.bf8.bf4.exec(input5);

                mid5[1] = NeonStoreF::$mul(mid5[1], self.twiddles32[3]);
                mid5[2] = NeonStoreF::$mul(mid5[2], self.bf16.bf8.rotate(self.twiddles32[1]));
                mid5[3] = NeonStoreF::$mul(mid5[3], self.bf16.bf8.rotate(self.twiddles32[5]));

                let input6 = [v(6), v(14), v(22), v(30)];
                let mut mid6 = self.bf16.bf8.bf4.exec(input6);

                mid6[1] = NeonStoreF::$mul(mid6[1], self.twiddles32[4]);
                mid6[2] = self.bf16.bf8.rotate135(mid6[2]);
                mid6[3] = NeonStoreF::$mul(mid6[3], self.twiddles32[1].neg());

                let input7 = [v(7), v(15), v(23), v(31)];
                let mut mid7 = self.bf16.bf8.bf4.exec(input7);

                mid7[1] = NeonStoreF::$mul(mid7[1], self.twiddles32[5]);
                mid7[2] = NeonStoreF::$mul(mid7[2], self.bf16.bf8.rotate(self.twiddles32[4]));
                mid7[3] = NeonStoreF::$mul(mid7[3], self.twiddles32[3].neg());

                let input0 = [v(0), v(8), v(16), v(24)];
                let mid0 = self.bf16.bf8.bf4.exec(input0);

                for i in 0..4 {
                    let output = self.bf16.bf8.exec([
                        mid0[i], mid1[i], mid2[i], mid3[i], mid4[i], mid5[i], mid6[i], mid7[i],
                    ]);
                    store(i, output[0]);
                    store(i + 4, output[1]);
                    store(i + 8, output[2]);
                    store(i + 12, output[3]);
                    store(i + 16, output[4]);
                    store(i + 20, output[5]);
                    store(i + 24, output[6]);
                    store(i + 28, output[7]);
                }
            }

            #[allow(unused)]
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
                let input1 = [v(1), v(9), v(17), v(25)];
                let mut mid1 = self.bf16.bf8.bf4.exec(input1);

                mid1[1] = NeonStoreF::$mul(mid1[1], self.twiddles32[0]);
                mid1[2] = NeonStoreF::$mul(mid1[2], self.twiddles32[1]);
                mid1[3] = NeonStoreF::$mul(mid1[3], self.twiddles32[2]);

                let input2 = [v(2), v(10), v(18), v(26)];
                let mut mid2 = self.bf16.bf8.bf4.exec(input2);

                mid2[1] = NeonStoreF::$mul(mid2[1], self.twiddles32[1]);
                mid2[2] = self.bf16.bf8.rotate45(mid2[2]);
                mid2[3] = NeonStoreF::$mul(mid2[3], self.twiddles32[4]);

                let input3 = [v(3), v(11), v(19), v(27)];
                let mut mid3 = self.bf16.bf8.bf4.exec(input3);

                mid3[1] = NeonStoreF::$mul(mid3[1], self.twiddles32[2]);
                mid3[2] = NeonStoreF::$mul(mid3[2], self.twiddles32[4]);
                mid3[3] = NeonStoreF::$mul(mid3[3], self.bf16.bf8.rotate(self.twiddles32[0]));

                let input4 = [v(4), v(12), v(20), v(28)];
                let mut mid4 = self.bf16.bf8.bf4.exec(input4);

                mid4[1] = self.bf16.bf8.rotate45(mid4[1]);
                mid4[2] = self.bf16.bf8.rotate(mid4[2]);
                mid4[3] = self.bf16.bf8.rotate135(mid4[3]);

                let input5 = [v(5), v(13), v(21), v(29)];
                let mut mid5 = self.bf16.bf8.bf4.exec(input5);

                mid5[1] = NeonStoreF::$mul(mid5[1], self.twiddles32[3]);
                mid5[2] = NeonStoreF::$mul(mid5[2], self.bf16.bf8.rotate(self.twiddles32[1]));
                mid5[3] = NeonStoreF::$mul(mid5[3], self.bf16.bf8.rotate(self.twiddles32[5]));

                let input6 = [v(6), v(14), v(22), v(30)];
                let mut mid6 = self.bf16.bf8.bf4.exec(input6);

                mid6[1] = NeonStoreF::$mul(mid6[1], self.twiddles32[4]);
                mid6[2] = self.bf16.bf8.rotate135(mid6[2]);
                mid6[3] = NeonStoreF::$mul(mid6[3], self.twiddles32[1].neg());

                let input7 = [v(7), v(15), v(23), v(31)];
                let mut mid7 = self.bf16.bf8.bf4.exec(input7);

                mid7[1] = NeonStoreF::$mul(mid7[1], self.twiddles32[5]);
                mid7[2] = NeonStoreF::$mul(mid7[2], self.bf16.bf8.rotate(self.twiddles32[4]));
                mid7[3] = NeonStoreF::$mul(mid7[3], self.twiddles32[3].neg());

                let input0 = [v(0), v(8), v(16), v(24)];
                let mid0 = self.bf16.bf8.bf4.exec(input0);

                let output0 = self.bf16.bf8.exec([
                    mid0[0], mid1[0], mid2[0], mid3[0], mid4[0], mid5[0], mid6[0], mid7[0],
                ]);
                let output1 = self.bf16.bf8.exec([
                    mid0[1], mid1[1], mid2[1], mid3[1], mid4[1], mid5[1], mid6[1], mid7[1],
                ]);
                let output2 = self.bf16.bf8.exec([
                    mid0[2], mid1[2], mid2[2], mid3[2], mid4[2], mid5[2], mid6[2], mid7[2],
                ]);
                let output3 = self.bf16.bf8.exec([
                    mid0[3], mid1[3], mid2[3], mid3[3], mid4[3], mid5[3], mid6[3], mid7[3],
                ]);

                {
                    let q1 = NeonStoreF::$mul(output1[0], twiddle(0));
                    let q2 = NeonStoreF::$mul(output2[0], twiddle(1));
                    let q3 = NeonStoreF::$mul(output3[0], twiddle(2));
                    let t = transpose_2x2([output0[0], q1]);
                    let t1 = transpose_2x2([q2, q3]);
                    store(0, t[0]);
                    store(1, t[1]);
                    store(2, t1[0]);
                    store(3, t1[1]);
                }

                for q in 1..8 {
                    let q0 = NeonStoreF::$mul(output0[q], twiddle(q * 4 - 1));
                    let q1 = NeonStoreF::$mul(output1[q], twiddle(q * 4));
                    let q2 = NeonStoreF::$mul(output2[q], twiddle(q * 4 + 1));
                    let q3 = NeonStoreF::$mul(output3[q], twiddle(q * 4 + 2));
                    let t = transpose_2x2([q0, q1]);
                    let t1 = transpose_2x2([q2, q3]);
                    store(q * 4, t[0]);
                    store(q * 4 + 1, t[1]);
                    store(q * 4 + 2, t1[0]);
                    store(q * 4 + 3, t1[1]);
                }
            }
        }
    };
}

define_column_bf32!(
    ColumnButterfly32f,
    "neon",
    ColumnButterfly16f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
define_column_bf32!(
    ColumnFcmaButterfly32f,
    "fcma",
    ColumnFcmaButterfly16f,
    fcmul_fcma
);
#[cfg(feature = "fcma")]
define_column_bf32!(
    ColumnFcmaForwardButterfly32f,
    "fcma",
    ColumnFcmaForwardButterfly16f,
    fcmul_fcma
);
#[cfg(feature = "fcma")]
define_column_bf32!(
    ColumnFcmaInverseButterfly32f,
    "fcma",
    ColumnFcmaInverseButterfly16f,
    fcmul_fcma
);
