/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF};
use crate::neon::mixed::{
    ColumnButterfly2d, ColumnButterfly2f, ColumnButterfly8d, ColumnButterfly8f,
};
#[cfg(feature = "fcma")]
use crate::neon::mixed::{ColumnFcmaButterfly8d, ColumnFcmaButterfly8f};
use crate::util::compute_twiddle;

pub(crate) struct ColumnButterfly16d {
    bf8: ColumnButterfly8d,
    bf2: ColumnButterfly2d,
    twiddle1: NeonStoreD,
    twiddle2: NeonStoreD,
    twiddle3: NeonStoreD,
}

impl ColumnButterfly16d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 16, fft_direction);
        let tw2 = compute_twiddle(2, 16, fft_direction);
        let tw3 = compute_twiddle(3, 16, fft_direction);
        Self {
            bf8: ColumnButterfly8d::new(fft_direction),
            bf2: ColumnButterfly2d::new(fft_direction),
            twiddle1: NeonStoreD::from_complex(&tw1),
            twiddle2: NeonStoreD::from_complex(&tw2),
            twiddle3: NeonStoreD::from_complex(&tw3),
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, v: [NeonStoreD; 16]) -> [NeonStoreD; 16] {
        let evens = self
            .bf8
            .exec([v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]]);

        let odds_1 = self.bf8.bf4.exec([v[1], v[5], v[9], v[13]]);
        let odds_2 = self.bf8.bf4.exec([v[15], v[3], v[7], v[11]]);

        // Twiddle + butterfly2 + rotate + final add/sub, one lane at a time.
        // Each group keeps only 2 odds registers live alongside evens[i]/evens[i+4],
        // freeing them before moving to the next group.

        // lane 0 — no twiddle
        let [o0a, o0b] = self.bf2.exec([odds_1[0], odds_2[0]]);
        let o0b = self.bf8.rotate(o0b);
        let (y00, y08) = (evens[0] + o0a, evens[0] - o0a);
        let (y04, y12) = (evens[4] + o0b, evens[4] - o0b);

        // lane 1
        let o1a = NeonStoreD::mul_by_complex(odds_1[1], self.twiddle1);
        let o1b = NeonStoreD::mul_by_complex_conj_b(odds_2[1], self.twiddle1);
        let [o1a, o1b] = self.bf2.exec([o1a, o1b]);
        let o1b = self.bf8.rotate(o1b);
        let (y01, y09) = (evens[1] + o1a, evens[1] - o1a);
        let (y05, y13) = (evens[5] + o1b, evens[5] - o1b);

        // lane 2
        let o2a = NeonStoreD::mul_by_complex(odds_1[2], self.twiddle2);
        let o2b = NeonStoreD::mul_by_complex_conj_b(odds_2[2], self.twiddle2);
        let [o2a, o2b] = self.bf2.exec([o2a, o2b]);
        let o2b = self.bf8.rotate(o2b);
        let (y02, y10) = (evens[2] + o2a, evens[2] - o2a);
        let (y06, y14) = (evens[6] + o2b, evens[6] - o2b);

        // lane 3
        let o3a = NeonStoreD::mul_by_complex(odds_1[3], self.twiddle3);
        let o3b = NeonStoreD::mul_by_complex_conj_b(odds_2[3], self.twiddle3);
        let [o3a, o3b] = self.bf2.exec([o3a, o3b]);
        let o3b = self.bf8.rotate(o3b);
        let (y03, y11) = (evens[3] + o3a, evens[3] - o3a);
        let (y07, y15) = (evens[7] + o3b, evens[7] - o3b);

        [
            y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14, y15,
        ]
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly16d {
    bf8: ColumnFcmaButterfly8d,
    bf2: ColumnButterfly2d,
    twiddle1: NeonStoreD,
    twiddle2: NeonStoreD,
    twiddle3: NeonStoreD,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly16d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 16, fft_direction);
        let tw2 = compute_twiddle(2, 16, fft_direction);
        let tw3 = compute_twiddle(3, 16, fft_direction);
        Self {
            bf8: ColumnFcmaButterfly8d::new(fft_direction),
            bf2: ColumnButterfly2d::new(fft_direction),
            twiddle1: NeonStoreD::from_complex(&tw1),
            twiddle2: NeonStoreD::from_complex(&tw2),
            twiddle3: NeonStoreD::from_complex(&tw3),
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, v: [NeonStoreD; 16]) -> [NeonStoreD; 16] {
        let evens = self
            .bf8
            .exec([v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]]);

        let odds_1 = self.bf8.bf4.exec([v[1], v[5], v[9], v[13]]);
        let odds_2 = self.bf8.bf4.exec([v[15], v[3], v[7], v[11]]);

        // Twiddle + butterfly2 + rotate + final add/sub, one lane at a time.
        // Each group keeps only 2 odds registers live alongside evens[i]/evens[i+4],
        // freeing them before moving to the next group.

        // lane 0 — no twiddle
        let [o0a, o0b] = self.bf2.exec([odds_1[0], odds_2[0]]);
        let o0b = self.bf8.rotate(o0b);
        let (y00, y08) = (evens[0] + o0a, evens[0] - o0a);
        let (y04, y12) = (evens[4] + o0b, evens[4] - o0b);

        // lane 1
        let o1a = NeonStoreD::fcmul_fcma(odds_1[1], self.twiddle1);
        let o1b = NeonStoreD::fcmul_conj_b(odds_2[1], self.twiddle1);
        let [o1a, o1b] = self.bf2.exec([o1a, o1b]);
        let o1b = self.bf8.rotate(o1b);
        let (y01, y09) = (evens[1] + o1a, evens[1] - o1a);
        let (y05, y13) = (evens[5] + o1b, evens[5] - o1b);

        // lane 2
        let o2a = NeonStoreD::fcmul_fcma(odds_1[2], self.twiddle2);
        let o2b = NeonStoreD::fcmul_conj_b(odds_2[2], self.twiddle2);
        let [o2a, o2b] = self.bf2.exec([o2a, o2b]);
        let o2b = self.bf8.rotate(o2b);
        let (y02, y10) = (evens[2] + o2a, evens[2] - o2a);
        let (y06, y14) = (evens[6] + o2b, evens[6] - o2b);

        // lane 3
        let o3a = NeonStoreD::fcmul_fcma(odds_1[3], self.twiddle3);
        let o3b = NeonStoreD::fcmul_conj_b(odds_2[3], self.twiddle3);
        let [o3a, o3b] = self.bf2.exec([o3a, o3b]);
        let o3b = self.bf8.rotate(o3b);
        let (y03, y11) = (evens[3] + o3a, evens[3] - o3a);
        let (y07, y15) = (evens[7] + o3b, evens[7] - o3b);

        [
            y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14, y15,
        ]
    }
}

pub(crate) struct ColumnButterfly16f {
    pub(crate) bf8: ColumnButterfly8f,
    bf2: ColumnButterfly2f,
    twiddle1: NeonStoreF,
    twiddle2: NeonStoreF,
    twiddle3: NeonStoreF,
}

impl ColumnButterfly16f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 16, fft_direction);
        let tw2 = compute_twiddle(2, 16, fft_direction);
        let tw3 = compute_twiddle(3, 16, fft_direction);
        Self {
            bf8: ColumnButterfly8f::new(fft_direction),
            bf2: ColumnButterfly2f::new(fft_direction),
            twiddle1: NeonStoreF::from_complex(&tw1),
            twiddle2: NeonStoreF::from_complex(&tw2),
            twiddle3: NeonStoreF::from_complex(&tw3),
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, v: [NeonStoreF; 16]) -> [NeonStoreF; 16] {
        let evens = self
            .bf8
            .exec([v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]]);

        let odds_1 = self.bf8.bf4.exec([v[1], v[5], v[9], v[13]]);
        let odds_2 = self.bf8.bf4.exec([v[15], v[3], v[7], v[11]]);

        // Twiddle + butterfly2 + rotate + final add/sub, one lane at a time.
        // Each group keeps only 2 odds registers live alongside evens[i]/evens[i+4],
        // freeing them before moving to the next group.

        // lane 0 — no twiddle
        let [o0a, o0b] = self.bf2.exec([odds_1[0], odds_2[0]]);
        let o0b = self.bf8.rotate(o0b);
        let (y00, y08) = (evens[0] + o0a, evens[0] - o0a);
        let (y04, y12) = (evens[4] + o0b, evens[4] - o0b);

        // lane 1
        let o1a = NeonStoreF::mul_by_complex(odds_1[1], self.twiddle1);
        let o1b = NeonStoreF::mul_by_complex_conj_b(odds_2[1], self.twiddle1);
        let [o1a, o1b] = self.bf2.exec([o1a, o1b]);
        let o1b = self.bf8.rotate(o1b);
        let (y01, y09) = (evens[1] + o1a, evens[1] - o1a);
        let (y05, y13) = (evens[5] + o1b, evens[5] - o1b);

        // lane 2
        let o2a = NeonStoreF::mul_by_complex(odds_1[2], self.twiddle2);
        let o2b = NeonStoreF::mul_by_complex_conj_b(odds_2[2], self.twiddle2);
        let [o2a, o2b] = self.bf2.exec([o2a, o2b]);
        let o2b = self.bf8.rotate(o2b);
        let (y02, y10) = (evens[2] + o2a, evens[2] - o2a);
        let (y06, y14) = (evens[6] + o2b, evens[6] - o2b);

        // lane 3
        let o3a = NeonStoreF::mul_by_complex(odds_1[3], self.twiddle3);
        let o3b = NeonStoreF::mul_by_complex_conj_b(odds_2[3], self.twiddle3);
        let [o3a, o3b] = self.bf2.exec([o3a, o3b]);
        let o3b = self.bf8.rotate(o3b);
        let (y03, y11) = (evens[3] + o3a, evens[3] - o3a);
        let (y07, y15) = (evens[7] + o3b, evens[7] - o3b);

        [
            y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14, y15,
        ]
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly16f {
    pub(crate) bf8: ColumnFcmaButterfly8f,
    bf2: ColumnButterfly2f,
    twiddle1: NeonStoreF,
    twiddle2: NeonStoreF,
    twiddle3: NeonStoreF,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly16f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 16, fft_direction);
        let tw2 = compute_twiddle(2, 16, fft_direction);
        let tw3 = compute_twiddle(3, 16, fft_direction);
        Self {
            bf8: ColumnFcmaButterfly8f::new(fft_direction),
            bf2: ColumnButterfly2f::new(fft_direction),
            twiddle1: NeonStoreF::from_complex(&tw1),
            twiddle2: NeonStoreF::from_complex(&tw2),
            twiddle3: NeonStoreF::from_complex(&tw3),
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, v: [NeonStoreF; 16]) -> [NeonStoreF; 16] {
        let evens = self
            .bf8
            .exec([v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]]);

        let odds_1 = self.bf8.bf4.exec([v[1], v[5], v[9], v[13]]);
        let odds_2 = self.bf8.bf4.exec([v[15], v[3], v[7], v[11]]);

        // Twiddle + butterfly2 + rotate + final add/sub, one lane at a time.
        // Each group keeps only 2 odds registers live alongside evens[i]/evens[i+4],
        // freeing them before moving to the next group.

        // lane 0 — no twiddle
        let [o0a, o0b] = self.bf2.exec([odds_1[0], odds_2[0]]);
        let o0b = self.bf8.rotate(o0b);
        let (y00, y08) = (evens[0] + o0a, evens[0] - o0a);
        let (y04, y12) = (evens[4] + o0b, evens[4] - o0b);

        // lane 1
        let o1a = NeonStoreF::fcmul_fcma(odds_1[1], self.twiddle1);
        let o1b = NeonStoreF::fcmul_conj_b(odds_2[1], self.twiddle1);
        let [o1a, o1b] = self.bf2.exec([o1a, o1b]);
        let o1b = self.bf8.rotate(o1b);
        let (y01, y09) = (evens[1] + o1a, evens[1] - o1a);
        let (y05, y13) = (evens[5] + o1b, evens[5] - o1b);

        // lane 2
        let o2a = NeonStoreF::fcmul_fcma(odds_1[2], self.twiddle2);
        let o2b = NeonStoreF::fcmul_conj_b(odds_2[2], self.twiddle2);
        let [o2a, o2b] = self.bf2.exec([o2a, o2b]);
        let o2b = self.bf8.rotate(o2b);
        let (y02, y10) = (evens[2] + o2a, evens[2] - o2a);
        let (y06, y14) = (evens[6] + o2b, evens[6] - o2b);

        // lane 3
        let o3a = NeonStoreF::fcmul_fcma(odds_1[3], self.twiddle3);
        let o3b = NeonStoreF::fcmul_conj_b(odds_2[3], self.twiddle3);
        let [o3a, o3b] = self.bf2.exec([o3a, o3b]);
        let o3b = self.bf8.rotate(o3b);
        let (y03, y11) = (evens[3] + o3a, evens[3] - o3a);
        let (y07, y15) = (evens[7] + o3b, evens[7] - o3b);

        [
            y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14, y15,
        ]
    }
}
