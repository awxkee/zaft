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
use crate::avx::mixed::{AvxStoreD, AvxStoreF, ColumnButterfly6d, ColumnButterfly6f};
use crate::avx::transpose::{transpose_f32x2_4x4_aos, transpose_f64x2_2x2d};
use crate::util::compute_twiddle;

pub(crate) struct ColumnButterfly36f {
    pub(crate) bf6: ColumnButterfly6f,
    pub(crate) twiddles36: [AvxStoreF; 13],
}

impl ColumnButterfly36f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly36f {
        Self {
            bf6: ColumnButterfly6f::new(direction),
            twiddles36: [
                AvxStoreF::set_complex(compute_twiddle(1, 36, direction)),
                AvxStoreF::set_complex(compute_twiddle(2, 36, direction)),
                AvxStoreF::set_complex(compute_twiddle(3, 36, direction)),
                AvxStoreF::set_complex(compute_twiddle(4, 36, direction)),
                AvxStoreF::set_complex(compute_twiddle(5, 36, direction)),
                AvxStoreF::set_complex(compute_twiddle(6, 36, direction)),
                AvxStoreF::set_complex(compute_twiddle(8, 36, direction)),
                AvxStoreF::set_complex(compute_twiddle(9, 36, direction)),
                AvxStoreF::set_complex(compute_twiddle(10, 36, direction)),
                AvxStoreF::set_complex(compute_twiddle(12, 36, direction)),
                AvxStoreF::set_complex(compute_twiddle(15, 36, direction)),
                AvxStoreF::set_complex(compute_twiddle(16, 36, direction)),
                AvxStoreF::set_complex(compute_twiddle(25, 36, direction)),
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
        let input1 = [v(1), v(7), v(13), v(19), v(25), v(31)];
        let mut mid1 = self.bf6.exec(input1);
        mid1[1] = AvxStoreF::mul_by_complex(mid1[1], self.twiddles36[0]);
        mid1[2] = AvxStoreF::mul_by_complex(mid1[2], self.twiddles36[1]);
        mid1[3] = AvxStoreF::mul_by_complex(mid1[3], self.twiddles36[2]);
        mid1[4] = AvxStoreF::mul_by_complex(mid1[4], self.twiddles36[3]);
        mid1[5] = AvxStoreF::mul_by_complex(mid1[5], self.twiddles36[4]);

        let input2 = [v(2), v(8), v(14), v(20), v(26), v(32)];
        let mut mid2 = self.bf6.exec(input2);
        mid2[1] = AvxStoreF::mul_by_complex(mid2[1], self.twiddles36[1]);
        mid2[2] = AvxStoreF::mul_by_complex(mid2[2], self.twiddles36[3]);
        mid2[3] = AvxStoreF::mul_by_complex(mid2[3], self.twiddles36[5]);
        mid2[4] = AvxStoreF::mul_by_complex(mid2[4], self.twiddles36[6]);
        mid2[5] = AvxStoreF::mul_by_complex(mid2[5], self.twiddles36[8]);

        let input3 = [v(3), v(9), v(15), v(21), v(27), v(33)];
        let mut mid3 = self.bf6.exec(input3);
        mid3[1] = AvxStoreF::mul_by_complex(mid3[1], self.twiddles36[2]);
        mid3[2] = AvxStoreF::mul_by_complex(mid3[2], self.twiddles36[5]);
        mid3[3] = AvxStoreF::mul_by_complex(mid3[3], self.twiddles36[7]);
        mid3[4] = AvxStoreF::mul_by_complex(mid3[4], self.twiddles36[9]);
        mid3[5] = AvxStoreF::mul_by_complex(mid3[5], self.twiddles36[10]);

        let input4 = [v(4), v(10), v(16), v(22), v(28), v(34)];
        let mut mid4 = self.bf6.exec(input4);
        mid4[1] = AvxStoreF::mul_by_complex(mid4[1], self.twiddles36[3]);
        mid4[2] = AvxStoreF::mul_by_complex(mid4[2], self.twiddles36[6]);
        mid4[3] = AvxStoreF::mul_by_complex(mid4[3], self.twiddles36[9]);
        mid4[4] = AvxStoreF::mul_by_complex(mid4[4], self.twiddles36[11]);
        mid4[5] = AvxStoreF::mul_by_complex(mid4[5], self.twiddles36[1].neg());

        let input5 = [v(5), v(11), v(17), v(23), v(29), v(35)];
        let mut mid5 = self.bf6.exec(input5);
        mid5[1] = AvxStoreF::mul_by_complex(mid5[1], self.twiddles36[4]);
        mid5[2] = AvxStoreF::mul_by_complex(mid5[2], self.twiddles36[8]);
        mid5[3] = AvxStoreF::mul_by_complex(mid5[3], self.twiddles36[10]);
        mid5[4] = AvxStoreF::mul_by_complex(mid5[4], self.twiddles36[1].neg());
        mid5[5] = AvxStoreF::mul_by_complex(mid5[5], self.twiddles36[12]);

        let input0 = [v(0), v(6), v(12), v(18), v(24), v(30)];
        let mid0 = self.bf6.exec(input0);

        for i in 0..6 {
            let output = self
                .bf6
                .exec([mid0[i], mid1[i], mid2[i], mid3[i], mid4[i], mid5[i]]);
            store(i, output[0]);
            store(i + 6, output[1]);
            store(i + 12, output[2]);
            store(i + 18, output[3]);
            store(i + 24, output[4]);
            store(i + 30, output[5]);
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec_transpose_streaming<
        A: Fn(usize) -> AvxStoreF,
        T: Fn(usize) -> AvxStoreF,
        J: FnMut(usize, bool, AvxStoreF),
    >(
        &self,
        v: A,
        twiddle: T,
        mut store: J,
    ) {
        let input1 = [v(1), v(7), v(13), v(19), v(25), v(31)];
        let mut mid1 = self.bf6.exec(input1);
        mid1[1] = AvxStoreF::mul_by_complex(mid1[1], self.twiddles36[0]);
        mid1[2] = AvxStoreF::mul_by_complex(mid1[2], self.twiddles36[1]);
        mid1[3] = AvxStoreF::mul_by_complex(mid1[3], self.twiddles36[2]);
        mid1[4] = AvxStoreF::mul_by_complex(mid1[4], self.twiddles36[3]);
        mid1[5] = AvxStoreF::mul_by_complex(mid1[5], self.twiddles36[4]);

        let input2 = [v(2), v(8), v(14), v(20), v(26), v(32)];
        let mut mid2 = self.bf6.exec(input2);
        mid2[1] = AvxStoreF::mul_by_complex(mid2[1], self.twiddles36[1]);
        mid2[2] = AvxStoreF::mul_by_complex(mid2[2], self.twiddles36[3]);
        mid2[3] = AvxStoreF::mul_by_complex(mid2[3], self.twiddles36[5]);
        mid2[4] = AvxStoreF::mul_by_complex(mid2[4], self.twiddles36[6]);
        mid2[5] = AvxStoreF::mul_by_complex(mid2[5], self.twiddles36[8]);

        let input3 = [v(3), v(9), v(15), v(21), v(27), v(33)];
        let mut mid3 = self.bf6.exec(input3);
        mid3[1] = AvxStoreF::mul_by_complex(mid3[1], self.twiddles36[2]);
        mid3[2] = AvxStoreF::mul_by_complex(mid3[2], self.twiddles36[5]);
        mid3[3] = AvxStoreF::mul_by_complex(mid3[3], self.twiddles36[7]);
        mid3[4] = AvxStoreF::mul_by_complex(mid3[4], self.twiddles36[9]);
        mid3[5] = AvxStoreF::mul_by_complex(mid3[5], self.twiddles36[10]);

        let input4 = [v(4), v(10), v(16), v(22), v(28), v(34)];
        let mut mid4 = self.bf6.exec(input4);
        mid4[1] = AvxStoreF::mul_by_complex(mid4[1], self.twiddles36[3]);
        mid4[2] = AvxStoreF::mul_by_complex(mid4[2], self.twiddles36[6]);
        mid4[3] = AvxStoreF::mul_by_complex(mid4[3], self.twiddles36[9]);
        mid4[4] = AvxStoreF::mul_by_complex(mid4[4], self.twiddles36[11]);
        mid4[5] = AvxStoreF::mul_by_complex(mid4[5], self.twiddles36[1].neg());

        let input5 = [v(5), v(11), v(17), v(23), v(29), v(35)];
        let mut mid5 = self.bf6.exec(input5);
        mid5[1] = AvxStoreF::mul_by_complex(mid5[1], self.twiddles36[4]);
        mid5[2] = AvxStoreF::mul_by_complex(mid5[2], self.twiddles36[8]);
        mid5[3] = AvxStoreF::mul_by_complex(mid5[3], self.twiddles36[10]);
        mid5[4] = AvxStoreF::mul_by_complex(mid5[4], self.twiddles36[1].neg());
        mid5[5] = AvxStoreF::mul_by_complex(mid5[5], self.twiddles36[12]);

        let mid0 = self.bf6.exec([v(0), v(6), v(12), v(18), v(24), v(30)]);

        let cols0 = self
            .bf6
            .exec([mid0[0], mid1[0], mid2[0], mid3[0], mid4[0], mid5[0]]);
        let cols1 = self
            .bf6
            .exec([mid0[1], mid1[1], mid2[1], mid3[1], mid4[1], mid5[1]]);
        let cols2 = self
            .bf6
            .exec([mid0[2], mid1[2], mid2[2], mid3[2], mid4[2], mid5[2]]);
        let cols3 = self
            .bf6
            .exec([mid0[3], mid1[3], mid2[3], mid3[3], mid4[3], mid5[3]]);
        let cols4 = self
            .bf6
            .exec([mid0[4], mid1[4], mid2[4], mid3[4], mid4[4], mid5[4]]);
        let cols5 = self
            .bf6
            .exec([mid0[5], mid1[5], mid2[5], mid3[5], mid4[5], mid5[5]]);

        // i=0
        {
            let q1 = AvxStoreF::mul_by_complex(cols1[0], twiddle(0));
            let q2 = AvxStoreF::mul_by_complex(cols2[0], twiddle(1));
            let q3 = AvxStoreF::mul_by_complex(cols3[0], twiddle(2));
            let q4 = AvxStoreF::mul_by_complex(cols4[0], twiddle(3));
            let q5 = AvxStoreF::mul_by_complex(cols5[0], twiddle(4));
            let t = transpose_f32x2_4x4_aos([cols0[0], q1, q2, q3]);
            let t1 =
                transpose_f32x2_4x4_aos([q4, q5, AvxStoreF::undefined(), AvxStoreF::undefined()]);
            store(0, false, t[0]);
            store(1, false, t[1]);
            store(2, false, t[2]);
            store(3, false, t[3]);
            store(4, true, t1[0]);
            store(5, true, t1[1]);
            store(6, true, t1[2]);
            store(7, true, t1[3]);
        }

        for i in 1..6 {
            let base = i * 6 - 1;
            let q0 = AvxStoreF::mul_by_complex(cols0[i], twiddle(base));
            let q1 = AvxStoreF::mul_by_complex(cols1[i], twiddle(base + 1));
            let q2 = AvxStoreF::mul_by_complex(cols2[i], twiddle(base + 2));
            let q3 = AvxStoreF::mul_by_complex(cols3[i], twiddle(base + 3));
            let q4 = AvxStoreF::mul_by_complex(cols4[i], twiddle(base + 4));
            let q5 = AvxStoreF::mul_by_complex(cols5[i], twiddle(base + 5));
            let t = transpose_f32x2_4x4_aos([q0, q1, q2, q3]);
            let t1 =
                transpose_f32x2_4x4_aos([q4, q5, AvxStoreF::undefined(), AvxStoreF::undefined()]);
            store(i * 8, false, t[0]);
            store(i * 8 + 1, false, t[1]);
            store(i * 8 + 2, false, t[2]);
            store(i * 8 + 3, false, t[3]);
            store(i * 8 + 4, true, t1[0]);
            store(i * 8 + 5, true, t1[1]);
            store(i * 8 + 6, true, t1[2]);
            store(i * 8 + 7, true, t1[3]);
        }
    }
}

pub(crate) struct ColumnButterfly36d {
    pub(crate) bf6: ColumnButterfly6d,
    pub(crate) twiddles36: [AvxStoreD; 13],
}

impl ColumnButterfly36d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly36d {
        Self {
            bf6: ColumnButterfly6d::new(direction),
            twiddles36: [
                AvxStoreD::set_complex(&compute_twiddle(1, 36, direction)),
                AvxStoreD::set_complex(&compute_twiddle(2, 36, direction)),
                AvxStoreD::set_complex(&compute_twiddle(3, 36, direction)),
                AvxStoreD::set_complex(&compute_twiddle(4, 36, direction)),
                AvxStoreD::set_complex(&compute_twiddle(5, 36, direction)),
                AvxStoreD::set_complex(&compute_twiddle(6, 36, direction)),
                AvxStoreD::set_complex(&compute_twiddle(8, 36, direction)),
                AvxStoreD::set_complex(&compute_twiddle(9, 36, direction)),
                AvxStoreD::set_complex(&compute_twiddle(10, 36, direction)),
                AvxStoreD::set_complex(&compute_twiddle(12, 36, direction)),
                AvxStoreD::set_complex(&compute_twiddle(15, 36, direction)),
                AvxStoreD::set_complex(&compute_twiddle(16, 36, direction)),
                AvxStoreD::set_complex(&compute_twiddle(25, 36, direction)),
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
        let input1 = [v(1), v(7), v(13), v(19), v(25), v(31)];
        let mut mid1 = self.bf6.exec(input1);
        mid1[1] = AvxStoreD::mul_by_complex(mid1[1], self.twiddles36[0]);
        mid1[2] = AvxStoreD::mul_by_complex(mid1[2], self.twiddles36[1]);
        mid1[3] = AvxStoreD::mul_by_complex(mid1[3], self.twiddles36[2]);
        mid1[4] = AvxStoreD::mul_by_complex(mid1[4], self.twiddles36[3]);
        mid1[5] = AvxStoreD::mul_by_complex(mid1[5], self.twiddles36[4]);

        let input2 = [v(2), v(8), v(14), v(20), v(26), v(32)];
        let mut mid2 = self.bf6.exec(input2);
        mid2[1] = AvxStoreD::mul_by_complex(mid2[1], self.twiddles36[1]);
        mid2[2] = AvxStoreD::mul_by_complex(mid2[2], self.twiddles36[3]);
        mid2[3] = AvxStoreD::mul_by_complex(mid2[3], self.twiddles36[5]);
        mid2[4] = AvxStoreD::mul_by_complex(mid2[4], self.twiddles36[6]);
        mid2[5] = AvxStoreD::mul_by_complex(mid2[5], self.twiddles36[8]);

        let input3 = [v(3), v(9), v(15), v(21), v(27), v(33)];
        let mut mid3 = self.bf6.exec(input3);
        mid3[1] = AvxStoreD::mul_by_complex(mid3[1], self.twiddles36[2]);
        mid3[2] = AvxStoreD::mul_by_complex(mid3[2], self.twiddles36[5]);
        mid3[3] = AvxStoreD::mul_by_complex(mid3[3], self.twiddles36[7]);
        mid3[4] = AvxStoreD::mul_by_complex(mid3[4], self.twiddles36[9]);
        mid3[5] = AvxStoreD::mul_by_complex(mid3[5], self.twiddles36[10]);

        let input4 = [v(4), v(10), v(16), v(22), v(28), v(34)];
        let mut mid4 = self.bf6.exec(input4);
        mid4[1] = AvxStoreD::mul_by_complex(mid4[1], self.twiddles36[3]);
        mid4[2] = AvxStoreD::mul_by_complex(mid4[2], self.twiddles36[6]);
        mid4[3] = AvxStoreD::mul_by_complex(mid4[3], self.twiddles36[9]);
        mid4[4] = AvxStoreD::mul_by_complex(mid4[4], self.twiddles36[11]);
        mid4[5] = AvxStoreD::mul_by_complex(mid4[5], self.twiddles36[1].neg());

        let input5 = [v(5), v(11), v(17), v(23), v(29), v(35)];
        let mut mid5 = self.bf6.exec(input5);
        mid5[1] = AvxStoreD::mul_by_complex(mid5[1], self.twiddles36[4]);
        mid5[2] = AvxStoreD::mul_by_complex(mid5[2], self.twiddles36[8]);
        mid5[3] = AvxStoreD::mul_by_complex(mid5[3], self.twiddles36[10]);
        mid5[4] = AvxStoreD::mul_by_complex(mid5[4], self.twiddles36[1].neg());
        mid5[5] = AvxStoreD::mul_by_complex(mid5[5], self.twiddles36[12]);

        let mid0 = self.bf6.exec([v(0), v(6), v(12), v(18), v(24), v(30)]);

        for i in 0..6 {
            let output = self
                .bf6
                .exec([mid0[i], mid1[i], mid2[i], mid3[i], mid4[i], mid5[i]]);
            store(i, output[0]);
            store(i + 6, output[1]);
            store(i + 12, output[2]);
            store(i + 18, output[3]);
            store(i + 24, output[4]);
            store(i + 30, output[5]);
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec_transpose_streaming<
        A: Fn(usize) -> AvxStoreD,
        T: Fn(usize) -> AvxStoreD,
        J: FnMut(usize, AvxStoreD),
    >(
        &self,
        v: A,
        twiddle: T,
        mut store: J,
    ) {
        let input1 = [v(1), v(7), v(13), v(19), v(25), v(31)];
        let mut mid1 = self.bf6.exec(input1);
        mid1[1] = AvxStoreD::mul_by_complex(mid1[1], self.twiddles36[0]);
        mid1[2] = AvxStoreD::mul_by_complex(mid1[2], self.twiddles36[1]);
        mid1[3] = AvxStoreD::mul_by_complex(mid1[3], self.twiddles36[2]);
        mid1[4] = AvxStoreD::mul_by_complex(mid1[4], self.twiddles36[3]);
        mid1[5] = AvxStoreD::mul_by_complex(mid1[5], self.twiddles36[4]);

        let input2 = [v(2), v(8), v(14), v(20), v(26), v(32)];
        let mut mid2 = self.bf6.exec(input2);
        mid2[1] = AvxStoreD::mul_by_complex(mid2[1], self.twiddles36[1]);
        mid2[2] = AvxStoreD::mul_by_complex(mid2[2], self.twiddles36[3]);
        mid2[3] = AvxStoreD::mul_by_complex(mid2[3], self.twiddles36[5]);
        mid2[4] = AvxStoreD::mul_by_complex(mid2[4], self.twiddles36[6]);
        mid2[5] = AvxStoreD::mul_by_complex(mid2[5], self.twiddles36[8]);

        let input3 = [v(3), v(9), v(15), v(21), v(27), v(33)];
        let mut mid3 = self.bf6.exec(input3);
        mid3[1] = AvxStoreD::mul_by_complex(mid3[1], self.twiddles36[2]);
        mid3[2] = AvxStoreD::mul_by_complex(mid3[2], self.twiddles36[5]);
        mid3[3] = AvxStoreD::mul_by_complex(mid3[3], self.twiddles36[7]);
        mid3[4] = AvxStoreD::mul_by_complex(mid3[4], self.twiddles36[9]);
        mid3[5] = AvxStoreD::mul_by_complex(mid3[5], self.twiddles36[10]);

        let input4 = [v(4), v(10), v(16), v(22), v(28), v(34)];
        let mut mid4 = self.bf6.exec(input4);
        mid4[1] = AvxStoreD::mul_by_complex(mid4[1], self.twiddles36[3]);
        mid4[2] = AvxStoreD::mul_by_complex(mid4[2], self.twiddles36[6]);
        mid4[3] = AvxStoreD::mul_by_complex(mid4[3], self.twiddles36[9]);
        mid4[4] = AvxStoreD::mul_by_complex(mid4[4], self.twiddles36[11]);
        mid4[5] = AvxStoreD::mul_by_complex(mid4[5], self.twiddles36[1].neg());

        let input5 = [v(5), v(11), v(17), v(23), v(29), v(35)];
        let mut mid5 = self.bf6.exec(input5);
        mid5[1] = AvxStoreD::mul_by_complex(mid5[1], self.twiddles36[4]);
        mid5[2] = AvxStoreD::mul_by_complex(mid5[2], self.twiddles36[8]);
        mid5[3] = AvxStoreD::mul_by_complex(mid5[3], self.twiddles36[10]);
        mid5[4] = AvxStoreD::mul_by_complex(mid5[4], self.twiddles36[1].neg());
        mid5[5] = AvxStoreD::mul_by_complex(mid5[5], self.twiddles36[12]);

        let input0 = [v(0), v(6), v(12), v(18), v(24), v(30)];
        let mid0 = self.bf6.exec(input0);

        let cols0 = self
            .bf6
            .exec([mid0[0], mid1[0], mid2[0], mid3[0], mid4[0], mid5[0]]);
        let cols1 = self
            .bf6
            .exec([mid0[1], mid1[1], mid2[1], mid3[1], mid4[1], mid5[1]]);
        let cols2 = self
            .bf6
            .exec([mid0[2], mid1[2], mid2[2], mid3[2], mid4[2], mid5[2]]);
        let cols3 = self
            .bf6
            .exec([mid0[3], mid1[3], mid2[3], mid3[3], mid4[3], mid5[3]]);
        let cols4 = self
            .bf6
            .exec([mid0[4], mid1[4], mid2[4], mid3[4], mid4[4], mid5[4]]);
        let cols5 = self
            .bf6
            .exec([mid0[5], mid1[5], mid2[5], mid3[5], mid4[5], mid5[5]]);

        {
            let q1 = AvxStoreD::mul_by_complex(cols1[0], twiddle(0));
            let q2 = AvxStoreD::mul_by_complex(cols2[0], twiddle(1));
            let q3 = AvxStoreD::mul_by_complex(cols3[0], twiddle(2));
            let q4 = AvxStoreD::mul_by_complex(cols4[0], twiddle(3));
            let q5 = AvxStoreD::mul_by_complex(cols5[0], twiddle(4));
            let t = transpose_f64x2_2x2d([cols0[0], q1]);
            let t1 = transpose_f64x2_2x2d([q2, q3]);
            let t2 = transpose_f64x2_2x2d([q4, q5]);
            store(0, t[0]);
            store(1, t[1]);
            store(2, t1[0]);
            store(3, t1[1]);
            store(4, t2[0]);
            store(5, t2[1]);
        }

        for i in 1..6 {
            let base = i * 6 - 1;
            let q0 = AvxStoreD::mul_by_complex(cols0[i], twiddle(base));
            let q1 = AvxStoreD::mul_by_complex(cols1[i], twiddle(base + 1));
            let q2 = AvxStoreD::mul_by_complex(cols2[i], twiddle(base + 2));
            let q3 = AvxStoreD::mul_by_complex(cols3[i], twiddle(base + 3));
            let q4 = AvxStoreD::mul_by_complex(cols4[i], twiddle(base + 4));
            let q5 = AvxStoreD::mul_by_complex(cols5[i], twiddle(base + 5));
            let t = transpose_f64x2_2x2d([q0, q1]);
            let t1 = transpose_f64x2_2x2d([q2, q3]);
            let t2 = transpose_f64x2_2x2d([q4, q5]);
            store(i * 6, t[0]);
            store(i * 6 + 1, t[1]);
            store(i * 6 + 2, t1[0]);
            store(i * 6 + 3, t1[1]);
            store(i * 6 + 4, t2[0]);
            store(i * 6 + 5, t2[1]);
        }
    }
}
