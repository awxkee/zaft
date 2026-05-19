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
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::wasm::column::{ColumnButterfly8d, ColumnButterfly16f};
use crate::wasm::store::{WasmStoreD, WasmStoreF};
use num_complex::Complex;
use std::ops::Neg;

pub(crate) struct ColumnButterfly32d {
    pub(crate) bf8: ColumnButterfly8d,
    pub(crate) twiddles32: [WasmStoreD; 6],
}

impl ColumnButterfly32d {
    pub(crate) fn new(direction: FftDirection) -> Self {
        Self {
            bf8: ColumnButterfly8d::new(direction),
            twiddles32: [
                WasmStoreD::from_complex(&compute_twiddle(1, 32, direction)),
                WasmStoreD::from_complex(&compute_twiddle(2, 32, direction)),
                WasmStoreD::from_complex(&compute_twiddle(3, 32, direction)),
                WasmStoreD::from_complex(&compute_twiddle(5, 32, direction)),
                WasmStoreD::from_complex(&compute_twiddle(6, 32, direction)),
                WasmStoreD::from_complex(&compute_twiddle(7, 32, direction)),
            ],
        }
    }

    #[inline]
    pub(crate) fn exec_store<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let input1 = [
            WasmStoreD::from_complex(chunk.index(1)),
            WasmStoreD::from_complex(chunk.index(9)),
            WasmStoreD::from_complex(chunk.index(17)),
            WasmStoreD::from_complex(chunk.index(25)),
        ];
        let mut mid1 = self.bf8.bf4.exec(input1);

        mid1[1] = mid1[1].mul_by_complex(self.twiddles32[0]);
        mid1[2] = mid1[2].mul_by_complex(self.twiddles32[1]);
        mid1[3] = mid1[3].mul_by_complex(self.twiddles32[2]);

        let input2 = [
            WasmStoreD::from_complex(chunk.index(2)),
            WasmStoreD::from_complex(chunk.index(10)),
            WasmStoreD::from_complex(chunk.index(18)),
            WasmStoreD::from_complex(chunk.index(26)),
        ];
        let mut mid2 = self.bf8.bf4.exec(input2);

        mid2[1] = mid2[1].mul_by_complex(self.twiddles32[1]);
        mid2[2] = self.bf8.rotate45(mid2[2]);
        mid2[3] = mid2[3].mul_by_complex(self.twiddles32[4]);

        let input3 = [
            WasmStoreD::from_complex(chunk.index(3)),
            WasmStoreD::from_complex(chunk.index(11)),
            WasmStoreD::from_complex(chunk.index(19)),
            WasmStoreD::from_complex(chunk.index(27)),
        ];
        let mut mid3 = self.bf8.bf4.exec(input3);

        mid3[1] = mid3[1].mul_by_complex(self.twiddles32[2]);
        mid3[2] = mid3[2].mul_by_complex(self.twiddles32[4]);
        mid3[3] = mid3[3].mul_by_complex(self.bf8.rotate(self.twiddles32[0]));

        let input4 = [
            WasmStoreD::from_complex(chunk.index(4)),
            WasmStoreD::from_complex(chunk.index(12)),
            WasmStoreD::from_complex(chunk.index(20)),
            WasmStoreD::from_complex(chunk.index(28)),
        ];
        let mut mid4 = self.bf8.bf4.exec(input4);

        mid4[1] = self.bf8.rotate45(mid4[1]);
        mid4[2] = self.bf8.rotate(mid4[2]);
        mid4[3] = self.bf8.rotate135(mid4[3]);

        let input5 = [
            WasmStoreD::from_complex(chunk.index(5)),
            WasmStoreD::from_complex(chunk.index(13)),
            WasmStoreD::from_complex(chunk.index(21)),
            WasmStoreD::from_complex(chunk.index(29)),
        ];
        let mut mid5 = self.bf8.bf4.exec(input5);

        mid5[1] = mid5[1].mul_by_complex(self.twiddles32[3]);
        mid5[2] = mid5[2].mul_by_complex(self.bf8.rotate(self.twiddles32[1]));
        mid5[3] = mid5[3].mul_by_complex(self.bf8.rotate(self.twiddles32[5]));

        let input6 = [
            WasmStoreD::from_complex(chunk.index(6)),
            WasmStoreD::from_complex(chunk.index(14)),
            WasmStoreD::from_complex(chunk.index(22)),
            WasmStoreD::from_complex(chunk.index(30)),
        ];
        let mut mid6 = self.bf8.bf4.exec(input6);

        mid6[1] = mid6[1].mul_by_complex(self.twiddles32[4]);
        mid6[2] = self.bf8.rotate135(mid6[2]);
        mid6[3] = mid6[3].mul_by_complex(self.twiddles32[1].neg());

        let input7 = [
            WasmStoreD::from_complex(chunk.index(7)),
            WasmStoreD::from_complex(chunk.index(15)),
            WasmStoreD::from_complex(chunk.index(23)),
            WasmStoreD::from_complex(chunk.index(31)),
        ];
        let mut mid7 = self.bf8.bf4.exec(input7);

        mid7[1] = mid7[1].mul_by_complex(self.twiddles32[5]);
        mid7[2] = mid7[2].mul_by_complex(self.bf8.rotate(self.twiddles32[4]));
        mid7[3] = mid7[3].mul_by_complex(self.twiddles32[3].neg());

        let input0 = [
            WasmStoreD::from_complex(chunk.index(0)),
            WasmStoreD::from_complex(chunk.index(8)),
            WasmStoreD::from_complex(chunk.index(16)),
            WasmStoreD::from_complex(chunk.index(24)),
        ];
        let mid0 = self.bf8.bf4.exec(input0);

        for i in 0..4 {
            let output = self.bf8.exec([
                mid0[i], mid1[i], mid2[i], mid3[i], mid4[i], mid5[i], mid6[i], mid7[i],
            ]);
            output[0].write_single(chunk.index_mut(i));
            output[1].write_single(chunk.index_mut(i + 4));
            output[2].write_single(chunk.index_mut(i + 8));
            output[3].write_single(chunk.index_mut(i + 12));
            output[4].write_single(chunk.index_mut(i + 16));
            output[5].write_single(chunk.index_mut(i + 20));
            output[6].write_single(chunk.index_mut(i + 24));
            output[7].write_single(chunk.index_mut(i + 28));
        }
    }
}

pub(crate) struct ColumnButterfly32f {
    pub(crate) bf16: ColumnButterfly16f,
    pub(crate) twiddles32: [WasmStoreF; 6],
}

impl ColumnButterfly32f {
    pub(crate) fn new(direction: FftDirection) -> Self {
        Self {
            bf16: ColumnButterfly16f::new(direction),
            twiddles32: [
                WasmStoreF::from_complex(&compute_twiddle(1, 32, direction)),
                WasmStoreF::from_complex(&compute_twiddle(2, 32, direction)),
                WasmStoreF::from_complex(&compute_twiddle(3, 32, direction)),
                WasmStoreF::from_complex(&compute_twiddle(5, 32, direction)),
                WasmStoreF::from_complex(&compute_twiddle(6, 32, direction)),
                WasmStoreF::from_complex(&compute_twiddle(7, 32, direction)),
            ],
        }
    }

    #[inline]
    pub(crate) fn exec_streaming<A: Fn(usize) -> WasmStoreF, J: FnMut(usize, WasmStoreF)>(
        &self,
        v: A,
        mut store: J,
    ) {
        let input1 = [v(1), v(9), v(17), v(25)];
        let mut mid1 = self.bf16.bf8.bf4.exec(input1);

        mid1[1] = WasmStoreF::mul_by_complex(mid1[1], self.twiddles32[0]);
        mid1[2] = WasmStoreF::mul_by_complex(mid1[2], self.twiddles32[1]);
        mid1[3] = WasmStoreF::mul_by_complex(mid1[3], self.twiddles32[2]);

        let input2 = [v(2), v(10), v(18), v(26)];
        let mut mid2 = self.bf16.bf8.bf4.exec(input2);

        mid2[1] = WasmStoreF::mul_by_complex(mid2[1], self.twiddles32[1]);
        mid2[2] = self.bf16.bf8.rotate45(mid2[2]);
        mid2[3] = WasmStoreF::mul_by_complex(mid2[3], self.twiddles32[4]);

        let input3 = [v(3), v(11), v(19), v(27)];
        let mut mid3 = self.bf16.bf8.bf4.exec(input3);

        mid3[1] = WasmStoreF::mul_by_complex(mid3[1], self.twiddles32[2]);
        mid3[2] = WasmStoreF::mul_by_complex(mid3[2], self.twiddles32[4]);
        mid3[3] = WasmStoreF::mul_by_complex(mid3[3], self.bf16.bf8.rotate(self.twiddles32[0]));

        let input4 = [v(4), v(12), v(20), v(28)];
        let mut mid4 = self.bf16.bf8.bf4.exec(input4);

        mid4[1] = self.bf16.bf8.rotate45(mid4[1]);
        mid4[2] = self.bf16.bf8.rotate(mid4[2]);
        mid4[3] = self.bf16.bf8.rotate135(mid4[3]);

        let input5 = [v(5), v(13), v(21), v(29)];
        let mut mid5 = self.bf16.bf8.bf4.exec(input5);

        mid5[1] = WasmStoreF::mul_by_complex(mid5[1], self.twiddles32[3]);
        mid5[2] = WasmStoreF::mul_by_complex(mid5[2], self.bf16.bf8.rotate(self.twiddles32[1]));
        mid5[3] = WasmStoreF::mul_by_complex(mid5[3], self.bf16.bf8.rotate(self.twiddles32[5]));

        let input6 = [v(6), v(14), v(22), v(30)];
        let mut mid6 = self.bf16.bf8.bf4.exec(input6);

        mid6[1] = WasmStoreF::mul_by_complex(mid6[1], self.twiddles32[4]);
        mid6[2] = self.bf16.bf8.rotate135(mid6[2]);
        mid6[3] = WasmStoreF::mul_by_complex(mid6[3], self.twiddles32[1].neg());

        let input7 = [v(7), v(15), v(23), v(31)];
        let mut mid7 = self.bf16.bf8.bf4.exec(input7);

        mid7[1] = WasmStoreF::mul_by_complex(mid7[1], self.twiddles32[5]);
        mid7[2] = WasmStoreF::mul_by_complex(mid7[2], self.bf16.bf8.rotate(self.twiddles32[4]));
        mid7[3] = WasmStoreF::mul_by_complex(mid7[3], self.twiddles32[3].neg());

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
}
