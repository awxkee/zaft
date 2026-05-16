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
use crate::avx::mixed::{
    AvxStoreD, AvxStoreF, ColumnButterfly6d, ColumnButterfly6f, ColumnButterfly8d,
    ColumnButterfly8f,
};
use crate::util::compute_twiddle;

pub(crate) struct ColumnButterfly48d {
    pub(crate) bf8: ColumnButterfly8d,
    pub(crate) bf6: ColumnButterfly6d,
    twiddles48: [AvxStoreD; 5],
}

impl ColumnButterfly48d {
    pub(crate) fn new(direction: FftDirection) -> Self {
        unsafe { Self::new_init(direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(direction: FftDirection) -> Self {
        Self {
            bf8: ColumnButterfly8d::new(direction),
            bf6: ColumnButterfly6d::new(direction),
            twiddles48: [
                AvxStoreD::set_complex(&compute_twiddle(1, 48, direction)),
                AvxStoreD::set_complex(&compute_twiddle(2, 48, direction)),
                AvxStoreD::set_complex(&compute_twiddle(3, 48, direction)),
                AvxStoreD::set_complex(&compute_twiddle(4, 48, direction)),
                AvxStoreD::set_complex(&compute_twiddle(5, 48, direction)),
            ],
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec_streaming<A: Fn(usize) -> AvxStoreD, J: FnMut(usize, AvxStoreD)>(
        &self,
        v: A,
        mut store: J,
    ) {
        let mut mid1 = self.bf6.exec([v(1), v(9), v(17), v(25), v(33), v(41)]);
        mid1[1] = AvxStoreD::mul_by_complex(mid1[1], self.twiddles48[0]);
        mid1[2] = AvxStoreD::mul_by_complex(mid1[2], self.twiddles48[1]);
        mid1[3] = AvxStoreD::mul_by_complex(mid1[3], self.twiddles48[2]);
        mid1[4] = AvxStoreD::mul_by_complex(mid1[4], self.twiddles48[3]);
        mid1[5] = AvxStoreD::mul_by_complex(mid1[5], self.twiddles48[4]);

        let mut mid2 = self.bf6.exec([v(2), v(10), v(18), v(26), v(34), v(42)]);
        mid2[1] = AvxStoreD::mul_by_complex(mid2[1], self.twiddles48[1]);
        mid2[2] = AvxStoreD::mul_by_complex(mid2[2], self.twiddles48[3]);
        mid2[3] = self.bf8.rotate45(mid2[3]);
        mid2[4] = AvxStoreD::mul_by_complex(mid2[4], self.bf8.rotate45(self.twiddles48[1]));
        mid2[5] = AvxStoreD::mul_by_complex(mid2[5], self.bf8.rotate45(self.twiddles48[3]));

        let mut mid3 = self.bf6.exec([v(3), v(11), v(19), v(27), v(35), v(43)]);
        let tw_mid3_5 = self.bf8.rotate(self.twiddles48[2]);
        mid3[1] = AvxStoreD::mul_by_complex(mid3[1], self.twiddles48[2]);
        mid3[2] = self.bf8.rotate45(mid3[2]);
        mid3[3] = AvxStoreD::mul_by_complex(mid3[3], self.bf8.rotate45(self.twiddles48[2]));
        mid3[4] = self.bf8.rotate(mid3[4]);
        mid3[5] = AvxStoreD::mul_by_complex(mid3[5], tw_mid3_5);

        let mut mid4 = self.bf6.exec([v(4), v(12), v(20), v(28), v(36), v(44)]);
        let tw_mid_4_5 = self.bf8.rotate135(self.twiddles48[1]);
        mid4[1] = AvxStoreD::mul_by_complex(mid4[1], self.twiddles48[3]);
        mid4[2] = AvxStoreD::mul_by_complex(mid4[2], self.bf8.rotate45(self.twiddles48[1]));
        mid4[3] = self.bf8.rotate(mid4[3]);
        mid4[4] = AvxStoreD::mul_by_complex(mid4[4], self.bf8.rotate(self.twiddles48[3]));
        mid4[5] = AvxStoreD::mul_by_complex(mid4[5], tw_mid_4_5);

        let mut mid5 = self.bf6.exec([v(5), v(13), v(21), v(29), v(37), v(45)]);
        mid5[1] = AvxStoreD::mul_by_complex(mid5[1], self.twiddles48[4]);
        mid5[2] = AvxStoreD::mul_by_complex(mid5[2], self.bf8.rotate45(self.twiddles48[3]));
        mid5[3] = AvxStoreD::mul_by_complex(mid5[3], tw_mid3_5);
        mid5[4] = AvxStoreD::mul_by_complex(mid5[4], tw_mid_4_5);
        mid5[5] = AvxStoreD::mul_by_complex(mid5[5], self.twiddles48[0].neg());

        let mut mid6 = self.bf6.exec([v(6), v(14), v(22), v(30), v(38), v(46)]);
        mid6[1] = self.bf8.rotate45(mid6[1]);
        mid6[2] = self.bf8.rotate(mid6[2]);
        mid6[3] = self.bf8.rotate135(mid6[3]);
        mid6[4] = mid6[4].neg();
        mid6[5] = self.bf8.rotate45(mid6[5]).neg();

        let mut mid7 = self.bf6.exec([v(7), v(15), v(23), v(31), v(39), v(47)]);
        mid7[1] = AvxStoreD::mul_by_complex(mid7[1], self.bf8.rotate45(self.twiddles48[0]));
        mid7[2] = AvxStoreD::mul_by_complex(mid7[2], self.bf8.rotate(self.twiddles48[1]));
        mid7[3] = AvxStoreD::mul_by_complex(mid7[3], self.bf8.rotate135(self.twiddles48[2]));
        mid7[4] = AvxStoreD::mul_by_complex(mid7[4], self.twiddles48[3].neg());
        mid7[5] = AvxStoreD::mul_by_complex(mid7[5], self.bf8.rotate45(self.twiddles48[4])).neg();

        let mid0 = self.bf6.exec([v(0), v(8), v(16), v(24), v(32), v(40)]);

        for i in 0..6 {
            let output = self.bf8.exec([
                mid0[i], mid1[i], mid2[i], mid3[i], mid4[i], mid5[i], mid6[i], mid7[i],
            ]);
            store(i, output[0]);
            store(i + 6, output[1]);
            store(i + 12, output[2]);
            store(i + 18, output[3]);
            store(i + 24, output[4]);
            store(i + 30, output[5]);
            store(i + 36, output[6]);
            store(i + 42, output[7]);
        }
    }
}

pub(crate) struct ColumnButterfly48f {
    pub(crate) bf8: ColumnButterfly8f,
    pub(crate) bf6: ColumnButterfly6f,
    twiddles48: [AvxStoreF; 5],
}

impl ColumnButterfly48f {
    pub(crate) fn new(direction: FftDirection) -> Self {
        unsafe { Self::new_init(direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(direction: FftDirection) -> Self {
        Self {
            bf8: ColumnButterfly8f::new(direction),
            bf6: ColumnButterfly6f::new(direction),
            twiddles48: [
                AvxStoreF::set_complex(compute_twiddle(1, 48, direction)),
                AvxStoreF::set_complex(compute_twiddle(2, 48, direction)),
                AvxStoreF::set_complex(compute_twiddle(3, 48, direction)),
                AvxStoreF::set_complex(compute_twiddle(4, 48, direction)),
                AvxStoreF::set_complex(compute_twiddle(5, 48, direction)),
            ],
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec_streaming<A: Fn(usize) -> AvxStoreF, J: FnMut(usize, AvxStoreF)>(
        &self,
        v: A,
        mut store: J,
    ) {
        let mut mid1 = self.bf6.exec([v(1), v(9), v(17), v(25), v(33), v(41)]);
        mid1[1] = AvxStoreF::mul_by_complex(mid1[1], self.twiddles48[0]);
        mid1[2] = AvxStoreF::mul_by_complex(mid1[2], self.twiddles48[1]);
        mid1[3] = AvxStoreF::mul_by_complex(mid1[3], self.twiddles48[2]);
        mid1[4] = AvxStoreF::mul_by_complex(mid1[4], self.twiddles48[3]);
        mid1[5] = AvxStoreF::mul_by_complex(mid1[5], self.twiddles48[4]);

        let mut mid2 = self.bf6.exec([v(2), v(10), v(18), v(26), v(34), v(42)]);
        mid2[1] = AvxStoreF::mul_by_complex(mid2[1], self.twiddles48[1]);
        mid2[2] = AvxStoreF::mul_by_complex(mid2[2], self.twiddles48[3]);
        mid2[3] = self.bf8.rotate45(mid2[3]);
        mid2[4] = AvxStoreF::mul_by_complex(mid2[4], self.bf8.rotate45(self.twiddles48[1]));
        mid2[5] = AvxStoreF::mul_by_complex(mid2[5], self.bf8.rotate45(self.twiddles48[3]));

        let mut mid3 = self.bf6.exec([v(3), v(11), v(19), v(27), v(35), v(43)]);
        let tw_mid3_5 = self.bf8.rotate(self.twiddles48[2]);
        mid3[1] = AvxStoreF::mul_by_complex(mid3[1], self.twiddles48[2]);
        mid3[2] = self.bf8.rotate45(mid3[2]);
        mid3[3] = AvxStoreF::mul_by_complex(mid3[3], self.bf8.rotate45(self.twiddles48[2]));
        mid3[4] = self.bf8.rotate(mid3[4]);
        mid3[5] = AvxStoreF::mul_by_complex(mid3[5], tw_mid3_5);

        let mut mid4 = self.bf6.exec([v(4), v(12), v(20), v(28), v(36), v(44)]);
        let tw_mid_4_5 = self.bf8.rotate135(self.twiddles48[1]);
        mid4[1] = AvxStoreF::mul_by_complex(mid4[1], self.twiddles48[3]);
        mid4[2] = AvxStoreF::mul_by_complex(mid4[2], self.bf8.rotate45(self.twiddles48[1]));
        mid4[3] = self.bf8.rotate(mid4[3]);
        mid4[4] = AvxStoreF::mul_by_complex(mid4[4], self.bf8.rotate(self.twiddles48[3]));
        mid4[5] = AvxStoreF::mul_by_complex(mid4[5], tw_mid_4_5);

        let mut mid5 = self.bf6.exec([v(5), v(13), v(21), v(29), v(37), v(45)]);
        mid5[1] = AvxStoreF::mul_by_complex(mid5[1], self.twiddles48[4]);
        mid5[2] = AvxStoreF::mul_by_complex(mid5[2], self.bf8.rotate45(self.twiddles48[3]));
        mid5[3] = AvxStoreF::mul_by_complex(mid5[3], tw_mid3_5);
        mid5[4] = AvxStoreF::mul_by_complex(mid5[4], tw_mid_4_5);
        mid5[5] = AvxStoreF::mul_by_complex(mid5[5], self.twiddles48[0].neg());

        let mut mid6 = self.bf6.exec([v(6), v(14), v(22), v(30), v(38), v(46)]);
        mid6[1] = self.bf8.rotate45(mid6[1]);
        mid6[2] = self.bf8.rotate(mid6[2]);
        mid6[3] = self.bf8.rotate135(mid6[3]);
        mid6[4] = mid6[4].neg();
        mid6[5] = self.bf8.rotate45(mid6[5]).neg();

        let mut mid7 = self.bf6.exec([v(7), v(15), v(23), v(31), v(39), v(47)]);
        mid7[1] = AvxStoreF::mul_by_complex(mid7[1], self.bf8.rotate45(self.twiddles48[0]));
        mid7[2] = AvxStoreF::mul_by_complex(mid7[2], self.bf8.rotate(self.twiddles48[1]));
        mid7[3] = AvxStoreF::mul_by_complex(mid7[3], self.bf8.rotate135(self.twiddles48[2]));
        mid7[4] = AvxStoreF::mul_by_complex(mid7[4], self.twiddles48[3].neg());
        mid7[5] = AvxStoreF::mul_by_complex(mid7[5], self.bf8.rotate45(self.twiddles48[4])).neg();

        let mid0 = self.bf6.exec([v(0), v(8), v(16), v(24), v(32), v(40)]);

        for i in 0..6 {
            let output = self.bf8.exec([
                mid0[i], mid1[i], mid2[i], mid3[i], mid4[i], mid5[i], mid6[i], mid7[i],
            ]);
            store(i, output[0]);
            store(i + 6, output[1]);
            store(i + 12, output[2]);
            store(i + 18, output[3]);
            store(i + 24, output[4]);
            store(i + 30, output[5]);
            store(i + 36, output[6]);
            store(i + 42, output[7]);
        }
    }
}
