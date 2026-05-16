/*
 * // Copyright (c) Radzivon Bartoshyk 1/2026. All rights reserved.
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
use crate::avx::mixed::{
    AvxStoreD, AvxStoreF, ColumnButterfly2d, ColumnButterfly2f, ColumnButterfly9d,
    ColumnButterfly9f,
};

pub(crate) struct ColumnButterfly18d {
    bf9: ColumnButterfly9d,
    bf2: ColumnButterfly2d,
}

impl ColumnButterfly18d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            bf9: ColumnButterfly9d::new(fft_direction),
            bf2: ColumnButterfly2d::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec_streaming<A: Fn(usize) -> AvxStoreD, J: FnMut(usize, AvxStoreD)>(
        &self,
        v: A,
        mut store: J,
    ) {
        let [t0, t1] = self.bf2.exec([v(0), v(9)]);
        let [t2, t3] = self.bf2.exec([v(10), v(1)]);
        let [t4, t5] = self.bf2.exec([v(2), v(11)]);
        let [t6, t7] = self.bf2.exec([v(12), v(3)]);
        let [t8, t9] = self.bf2.exec([v(4), v(13)]);
        let [t10, t11] = self.bf2.exec([v(14), v(5)]);
        let [t12, t13] = self.bf2.exec([v(6), v(15)]);
        let [t14, t15] = self.bf2.exec([v(16), v(7)]);
        let [t16, t17] = self.bf2.exec([v(8), v(17)]);

        let [u0, u2, u4, u6, u8, u10, u12, u14, u16] =
            self.bf9.exec([t0, t2, t4, t6, t8, t10, t12, t14, t16]);
        store(0, u0);
        store(2, u2);
        store(4, u4);
        store(6, u6);
        store(8, u8);
        store(10, u10);
        store(12, u12);
        store(14, u14);
        store(16, u16);

        let [u9, u11, u13, u15, u17, u1, u3, u5, u7] =
            self.bf9.exec([t1, t3, t5, t7, t9, t11, t13, t15, t17]);

        store(9, u9);
        store(11, u11);
        store(13, u13);
        store(15, u15);
        store(17, u17);
        store(1, u1);
        store(3, u3);
        store(5, u5);
        store(7, u7);
    }
}

pub(crate) struct ColumnButterfly18f {
    bf9: ColumnButterfly9f,
    bf2: ColumnButterfly2f,
}

impl ColumnButterfly18f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            bf9: ColumnButterfly9f::new(fft_direction),
            bf2: ColumnButterfly2f::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec_streaming<A: Fn(usize) -> AvxStoreF, J: FnMut(usize, AvxStoreF)>(
        &self,
        v: A,
        mut store: J,
    ) {
        let [t0, t1] = self.bf2.exec([v(0), v(9)]);
        let [t2, t3] = self.bf2.exec([v(10), v(1)]);
        let [t4, t5] = self.bf2.exec([v(2), v(11)]);
        let [t6, t7] = self.bf2.exec([v(12), v(3)]);
        let [t8, t9] = self.bf2.exec([v(4), v(13)]);
        let [t10, t11] = self.bf2.exec([v(14), v(5)]);
        let [t12, t13] = self.bf2.exec([v(6), v(15)]);
        let [t14, t15] = self.bf2.exec([v(16), v(7)]);
        let [t16, t17] = self.bf2.exec([v(8), v(17)]);

        let [u0, u2, u4, u6, u8, u10, u12, u14, u16] =
            self.bf9.exec([t0, t2, t4, t6, t8, t10, t12, t14, t16]);
        store(0, u0);
        store(2, u2);
        store(4, u4);
        store(6, u6);
        store(8, u8);
        store(10, u10);
        store(12, u12);
        store(14, u14);
        store(16, u16);

        let [u9, u11, u13, u15, u17, u1, u3, u5, u7] =
            self.bf9.exec([t1, t3, t5, t7, t9, t11, t13, t15, t17]);

        store(9, u9);
        store(11, u11);
        store(13, u13);
        store(15, u15);
        store(17, u17);
        store(1, u1);
        store(3, u3);
        store(5, u5);
        store(7, u7);
    }
}
