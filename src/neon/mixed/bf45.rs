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

macro_rules! define_butterfly_45f {
    ($bf_name: ident, $features: literal, $inner_bf9: ident, $inner_bf5: ident) => {
        use crate::neon::mixed::{$inner_bf5, $inner_bf9};
        pub(crate) struct $bf_name {
            pub(crate) bf9: $inner_bf9,
            pub(crate) bf5: $inner_bf5,
        }

        impl $bf_name {
            pub(crate) fn new(direction: FftDirection) -> Self {
                Self {
                    bf9: $inner_bf9::new(direction),
                    bf5: $inner_bf5::new(direction),
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
                let input0 = [v(0), v(36), v(27), v(18), v(9)];
                let mid0 = self.bf5.exec(input0);

                let input1 = [v(10), v(1), v(37), v(28), v(19)];
                let mid1 = self.bf5.exec(input1);

                let input2 = [v(20), v(11), v(2), v(38), v(29)];
                let mid2 = self.bf5.exec(input2);

                let input3 = [v(30), v(21), v(12), v(3), v(39)];
                let mid3 = self.bf5.exec(input3);

                let input4 = [v(40), v(31), v(22), v(13), v(4)];
                let mid4 = self.bf5.exec(input4);

                let input5 = [v(5), v(41), v(32), v(23), v(14)];
                let mid5 = self.bf5.exec(input5);

                let input6 = [v(15), v(6), v(42), v(33), v(24)];
                let mid6 = self.bf5.exec(input6);

                let input7 = [v(25), v(16), v(7), v(43), v(34)];
                let mid7 = self.bf5.exec(input7);

                let input8 = [v(35), v(26), v(17), v(8), v(44)];
                let mid8 = self.bf5.exec(input8);

                let output0 = self.bf9.exec([
                    mid0[0], mid1[0], mid2[0], mid3[0], mid4[0], mid5[0], mid6[0], mid7[0], mid8[0],
                ]);
                store(0, output0[0]);
                store(5, output0[1]);
                store(10, output0[2]);
                store(15, output0[3]);
                store(20, output0[4]);
                store(25, output0[5]);
                store(30, output0[6]);
                store(35, output0[7]);
                store(40, output0[8]);

                let output1 = self.bf9.exec([
                    mid0[1], mid1[1], mid2[1], mid3[1], mid4[1], mid5[1], mid6[1], mid7[1], mid8[1],
                ]);
                store(9, output1[0]);
                store(14, output1[1]);
                store(19, output1[2]);
                store(24, output1[3]);
                store(29, output1[4]);
                store(34, output1[5]);
                store(39, output1[6]);
                store(44, output1[7]);
                store(4, output1[8]);

                let output2 = self.bf9.exec([
                    mid0[2], mid1[2], mid2[2], mid3[2], mid4[2], mid5[2], mid6[2], mid7[2], mid8[2],
                ]);
                store(18, output2[0]);
                store(23, output2[1]);
                store(28, output2[2]);
                store(33, output2[3]);
                store(38, output2[4]);
                store(43, output2[5]);
                store(3, output2[6]);
                store(8, output2[7]);
                store(13, output2[8]);

                let output3 = self.bf9.exec([
                    mid0[3], mid1[3], mid2[3], mid3[3], mid4[3], mid5[3], mid6[3], mid7[3], mid8[3],
                ]);
                store(27, output3[0]);
                store(32, output3[1]);
                store(37, output3[2]);
                store(42, output3[3]);
                store(2, output3[4]);
                store(7, output3[5]);
                store(12, output3[6]);
                store(17, output3[7]);
                store(22, output3[8]);

                let output4 = self.bf9.exec([
                    mid0[4], mid1[4], mid2[4], mid3[4], mid4[4], mid5[4], mid6[4], mid7[4], mid8[4],
                ]);
                store(36, output4[0]);
                store(41, output4[1]);
                store(1, output4[2]);
                store(6, output4[3]);
                store(11, output4[4]);
                store(16, output4[5]);
                store(21, output4[6]);
                store(26, output4[7]);
                store(31, output4[8]);
            }
        }
    };
}

define_butterfly_45f!(
    ColumnButterfly45f,
    "neon",
    ColumnButterfly9f,
    ColumnButterfly5f
);
#[cfg(feature = "fcma")]
define_butterfly_45f!(
    ColumnFcmaButterfly45f,
    "fcma",
    ColumnFcmaButterfly9f,
    ColumnFcmaButterfly5f
);
