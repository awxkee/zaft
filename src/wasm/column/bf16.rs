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
use crate::wasm::column::{ColumnButterfly8d, ColumnButterfly8f};
use crate::wasm::store::{WasmStoreD, WasmStoreF};
use std::ops::Neg;

pub(crate) struct ColumnButterfly16d {
    pub(crate) bf8: ColumnButterfly8d,
    twiddles16: [WasmStoreD; 2],
}

impl ColumnButterfly16d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 16, fft_direction);
        let tw3 = compute_twiddle(3, 16, fft_direction);
        Self {
            bf8: ColumnButterfly8d::new(fft_direction),
            twiddles16: [
                WasmStoreD::from_complex(&tw1),
                WasmStoreD::from_complex(&tw3),
            ],
        }
    }

    #[inline]
    pub(crate) fn exec(&self, v: [WasmStoreD; 16]) -> [WasmStoreD; 16] {
        let mut col1 = self.bf8.bf4.exec([v[1], v[5], v[9], v[13]]);

        col1[1] = WasmStoreD::mul_by_complex(col1[1], self.twiddles16[0]);
        col1[2] = self.bf8.rotate45(col1[2]);
        col1[3] = WasmStoreD::mul_by_complex(col1[3], self.twiddles16[1]);

        let mut col2 = self.bf8.bf4.exec([v[2], v[6], v[10], v[14]]);

        col2[1] = self.bf8.rotate45(col2[1]);
        col2[2] = self.bf8.rotate(col2[2]);
        col2[3] = self.bf8.rotate135(col2[3]);

        let mut col3 = self.bf8.bf4.exec([v[3], v[7], v[11], v[15]]);

        col3[1] = WasmStoreD::mul_by_complex(col3[1], self.twiddles16[1]);
        col3[2] = self.bf8.rotate135(col3[2]);
        col3[3] = WasmStoreD::mul_by_complex(col3[3], self.twiddles16[0].neg());

        let col0 = self.bf8.bf4.exec([v[0], v[4], v[8], v[12]]);

        let r0 = self.bf8.bf4.exec([col0[0], col1[0], col2[0], col3[0]]);
        let r1 = self.bf8.bf4.exec([col0[1], col1[1], col2[1], col3[1]]);
        let r2 = self.bf8.bf4.exec([col0[2], col1[2], col2[2], col3[2]]);
        let r3 = self.bf8.bf4.exec([col0[3], col1[3], col2[3], col3[3]]);

        [
            r0[0], r1[0], r2[0], r3[0], r0[1], r1[1], r2[1], r3[1], r0[2], r1[2], r2[2], r3[2],
            r0[3], r1[3], r2[3], r3[3],
        ]
    }

    #[inline(always)]
    pub(crate) fn exec_streaming<A: Fn(usize) -> WasmStoreD, J: FnMut(usize, WasmStoreD)>(
        &self,
        v: A,
        mut store: J,
    ) {
        let mut col1 = self.bf8.bf4.exec([v(1), v(5), v(9), v(13)]);

        col1[1] = WasmStoreD::mul_by_complex(col1[1], self.twiddles16[0]);
        col1[2] = self.bf8.rotate45(col1[2]);
        col1[3] = WasmStoreD::mul_by_complex(col1[3], self.twiddles16[1]);

        let mut col2 = self.bf8.bf4.exec([v(2), v(6), v(10), v(14)]);

        col2[1] = self.bf8.rotate45(col2[1]);
        col2[2] = self.bf8.rotate(col2[2]);
        col2[3] = self.bf8.rotate135(col2[3]);

        let mut col3 = self.bf8.bf4.exec([v(3), v(7), v(11), v(15)]);

        col3[1] = WasmStoreD::mul_by_complex(col3[1], self.twiddles16[1]);
        col3[2] = self.bf8.rotate135(col3[2]);
        col3[3] = WasmStoreD::mul_by_complex(col3[3], self.twiddles16[0].neg());

        let col0 = self.bf8.bf4.exec([v(0), v(4), v(8), v(12)]);

        let r0 = self.bf8.bf4.exec([col0[0], col1[0], col2[0], col3[0]]);
        store(0, r0[0]);
        store(4, r0[1]);
        store(8, r0[2]);
        store(12, r0[3]);

        let r1 = self.bf8.bf4.exec([col0[1], col1[1], col2[1], col3[1]]);
        store(1, r1[0]);
        store(5, r1[1]);
        store(9, r1[2]);
        store(13, r1[3]);

        let r2 = self.bf8.bf4.exec([col0[2], col1[2], col2[2], col3[2]]);
        store(2, r2[0]);
        store(6, r2[1]);
        store(10, r2[2]);
        store(14, r2[3]);

        let r3 = self.bf8.bf4.exec([col0[3], col1[3], col2[3], col3[3]]);
        store(3, r3[0]);
        store(7, r3[1]);
        store(11, r3[2]);
        store(15, r3[3]);
    }
}

pub(crate) struct ColumnButterfly16f {
    pub(crate) bf8: ColumnButterfly8f,
    twiddles16: [WasmStoreF; 2],
}

impl ColumnButterfly16f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 16, fft_direction);
        let tw3 = compute_twiddle(3, 16, fft_direction);
        Self {
            bf8: ColumnButterfly8f::new(fft_direction),
            twiddles16: [
                WasmStoreF::from_complex(&tw1),
                WasmStoreF::from_complex(&tw3),
            ],
        }
    }

    #[inline]
    pub(crate) fn exec(&self, v: [WasmStoreF; 16]) -> [WasmStoreF; 16] {
        let mut col1 = self.bf8.bf4.exec([v[1], v[5], v[9], v[13]]);

        col1[1] = WasmStoreF::mul_by_complex(col1[1], self.twiddles16[0]);
        col1[2] = self.bf8.rotate45(col1[2]);
        col1[3] = WasmStoreF::mul_by_complex(col1[3], self.twiddles16[1]);

        let mut col2 = self.bf8.bf4.exec([v[2], v[6], v[10], v[14]]);

        col2[1] = self.bf8.rotate45(col2[1]);
        col2[2] = self.bf8.rotate(col2[2]);
        col2[3] = self.bf8.rotate135(col2[3]);

        let mut col3 = self.bf8.bf4.exec([v[3], v[7], v[11], v[15]]);

        col3[1] = WasmStoreF::mul_by_complex(col3[1], self.twiddles16[1]);
        col3[2] = self.bf8.rotate135(col3[2]);
        col3[3] = WasmStoreF::mul_by_complex(col3[3], self.twiddles16[0].neg());

        let col0 = self.bf8.bf4.exec([v[0], v[4], v[8], v[12]]);

        let r0 = self.bf8.bf4.exec([col0[0], col1[0], col2[0], col3[0]]);
        let r1 = self.bf8.bf4.exec([col0[1], col1[1], col2[1], col3[1]]);
        let r2 = self.bf8.bf4.exec([col0[2], col1[2], col2[2], col3[2]]);
        let r3 = self.bf8.bf4.exec([col0[3], col1[3], col2[3], col3[3]]);

        [
            r0[0], r1[0], r2[0], r3[0], r0[1], r1[1], r2[1], r3[1], r0[2], r1[2], r2[2], r3[2],
            r0[3], r1[3], r2[3], r3[3],
        ]
    }

    #[inline(always)]
    pub(crate) fn exec_streaming<A: Fn(usize) -> WasmStoreF, J: FnMut(usize, WasmStoreF)>(
        &self,
        v: A,
        mut store: J,
    ) {
        let mut col1 = self.bf8.bf4.exec([v(1), v(5), v(9), v(13)]);

        col1[1] = WasmStoreF::mul_by_complex(col1[1], self.twiddles16[0]);
        col1[2] = self.bf8.rotate45(col1[2]);
        col1[3] = WasmStoreF::mul_by_complex(col1[3], self.twiddles16[1]);

        let mut col2 = self.bf8.bf4.exec([v(2), v(6), v(10), v(14)]);

        col2[1] = self.bf8.rotate45(col2[1]);
        col2[2] = self.bf8.rotate(col2[2]);
        col2[3] = self.bf8.rotate135(col2[3]);

        let mut col3 = self.bf8.bf4.exec([v(3), v(7), v(11), v(15)]);

        col3[1] = WasmStoreF::mul_by_complex(col3[1], self.twiddles16[1]);
        col3[2] = self.bf8.rotate135(col3[2]);
        col3[3] = WasmStoreF::mul_by_complex(col3[3], self.twiddles16[0].neg());

        let col0 = self.bf8.bf4.exec([v(0), v(4), v(8), v(12)]);

        let r0 = self.bf8.bf4.exec([col0[0], col1[0], col2[0], col3[0]]);
        store(0, r0[0]);
        store(4, r0[1]);
        store(8, r0[2]);
        store(12, r0[3]);

        let r1 = self.bf8.bf4.exec([col0[1], col1[1], col2[1], col3[1]]);
        store(1, r1[0]);
        store(5, r1[1]);
        store(9, r1[2]);
        store(13, r1[3]);

        let r2 = self.bf8.bf4.exec([col0[2], col1[2], col2[2], col3[2]]);
        store(2, r2[0]);
        store(6, r2[1]);
        store(10, r2[2]);
        store(14, r2[3]);

        let r3 = self.bf8.bf4.exec([col0[3], col1[3], col2[3], col3[3]]);
        store(3, r3[0]);
        store(7, r3[1]);
        store(11, r3[2]);
        store(15, r3[3]);
    }
}
