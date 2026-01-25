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

    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreD; 18]) -> [AvxStoreD; 18] {
        unsafe {
            let u0 = v[0]; // 0
            let u3 = v[1]; // 3
            let u4 = v[2]; // 4
            let u7 = v[3]; // 7

            let u8 = v[4]; // 8
            let u11 = v[5]; // 11
            let u12 = v[6]; // 12
            let u15 = v[7]; // 15

            let u16 = v[8]; // 16
            let u1 = v[9]; // 1
            let u2 = v[10]; // 2
            let u5 = v[11]; // 5

            let u6 = v[12]; // 6
            let u9 = v[13]; // 9
            let u10 = v[14]; // 10
            let u13 = v[15]; // 13

            let u14 = v[16]; // 14
            let u17 = v[17]; // 17

            let [t0, t1] = self.bf2.exec([u0, u1]);
            let [t2, t3] = self.bf2.exec([u2, u3]);
            let [t4, t5] = self.bf2.exec([u4, u5]);
            let [t6, t7] = self.bf2.exec([u6, u7]);
            let [t8, t9] = self.bf2.exec([u8, u9]);
            let [t10, t11] = self.bf2.exec([u10, u11]);
            let [t12, t13] = self.bf2.exec([u12, u13]);
            let [t14, t15] = self.bf2.exec([u14, u15]);
            let [t16, t17] = self.bf2.exec([u16, u17]);

            let [u0, u2, u4, u6, u8, u10, u12, u14, u16] =
                self.bf9.exec([t0, t2, t4, t6, t8, t10, t12, t14, t16]);
            let [u9, u11, u13, u15, u17, u1, u3, u5, u7] =
                self.bf9.exec([t1, t3, t5, t7, t9, t11, t13, t15, t17]);

            [
                u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17,
            ]
        }
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

    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 18]) -> [AvxStoreF; 18] {
        let u0 = v[0]; // 0
        let u3 = v[1]; // 3
        let u4 = v[2]; // 4
        let u7 = v[3]; // 7

        let u8 = v[4]; // 8
        let u11 = v[5]; // 11
        let u12 = v[6]; // 12
        let u15 = v[7]; // 15

        let u16 = v[8]; // 16
        let u1 = v[9]; // 1
        let u2 = v[10]; // 2
        let u5 = v[11]; // 5

        let u6 = v[12]; // 6
        let u9 = v[13]; // 9
        let u10 = v[14]; // 10
        let u13 = v[15]; // 13

        let u14 = v[16]; // 14
        let u17 = v[17]; // 17

        let [t0, t1] = self.bf2.exec([u0, u1]);
        let [t2, t3] = self.bf2.exec([u2, u3]);
        let [t4, t5] = self.bf2.exec([u4, u5]);
        let [t6, t7] = self.bf2.exec([u6, u7]);
        let [t8, t9] = self.bf2.exec([u8, u9]);
        let [t10, t11] = self.bf2.exec([u10, u11]);
        let [t12, t13] = self.bf2.exec([u12, u13]);
        let [t14, t15] = self.bf2.exec([u14, u15]);
        let [t16, t17] = self.bf2.exec([u16, u17]);

        let [u0, u2, u4, u6, u8, u10, u12, u14, u16] =
            self.bf9.exec([t0, t2, t4, t6, t8, t10, t12, t14, t16]);
        let [u9, u11, u13, u15, u17, u1, u3, u5, u7] =
            self.bf9.exec([t1, t3, t5, t7, t9, t11, t13, t15, t17]);

        [
            u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17,
        ]
    }
}
