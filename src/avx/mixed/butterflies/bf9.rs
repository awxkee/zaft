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
use crate::avx::mixed::avx_stored::AvxStoreD;
use crate::avx::mixed::avx_storef::AvxStoreF;
use crate::avx::mixed::{ColumnButterfly3d, ColumnButterfly3f};
use crate::util::compute_twiddle;
use crate::{FftDirection, FftSample};
use num_traits::MulAdd;

pub(crate) struct ColumnButterfly9d {
    pub(crate) bf3: ColumnButterfly3d,
    twiddle1: AvxStoreD,
    twiddle2: AvxStoreD,
    twiddle4: AvxStoreD,
}

impl ColumnButterfly9d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly9d {
        let tw1 = compute_twiddle::<f64>(1, 9, direction);
        let tw2 = compute_twiddle::<f64>(2, 9, direction);
        let tw4 = compute_twiddle::<f64>(4, 9, direction);
        Self {
            twiddle1: AvxStoreD::set_complex(&tw1),
            twiddle2: AvxStoreD::set_complex(&tw2),
            twiddle4: AvxStoreD::set_complex(&tw4),
            bf3: ColumnButterfly3d::new(direction),
        }
    }
}

impl ColumnButterfly9d {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    pub(crate) fn exec(&self, v: [AvxStoreD; 9]) -> [AvxStoreD; 9] {
        let [u0, u3, u6] = self.bf3.exec([v[0], v[3], v[6]]);
        let [u1, mut u4, mut u7] = self.bf3.exec([v[1], v[4], v[7]]);
        let [u2, mut u5, mut u8] = self.bf3.exec([v[2], v[5], v[8]]);

        u4 = AvxStoreD::mul_by_complex(u4, self.twiddle1);
        u7 = AvxStoreD::mul_by_complex(u7, self.twiddle2);
        u5 = AvxStoreD::mul_by_complex(u5, self.twiddle2);
        u8 = AvxStoreD::mul_by_complex(u8, self.twiddle4);

        let [y0, y3, y6] = self.bf3.exec([u0, u1, u2]);
        let [y1, y4, y7] = self.bf3.exec([u3, u4, u5]);
        let [y2, y5, y8] = self.bf3.exec([u6, u7, u8]);
        [y0, y1, y2, y3, y4, y5, y6, y7, y8]
    }
}

pub(crate) struct ColumnButterfly9f {
    pub(crate) bf3: ColumnButterfly3f,
    twiddle1: AvxStoreF,
    twiddle2: AvxStoreF,
    twiddle4: AvxStoreF,
}

impl ColumnButterfly9f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly9f {
        let tw1 = compute_twiddle::<f32>(1, 9, direction);
        let tw2 = compute_twiddle::<f32>(2, 9, direction);
        let tw4 = compute_twiddle::<f32>(4, 9, direction);
        Self {
            twiddle1: AvxStoreF::set_complex(tw1),
            twiddle2: AvxStoreF::set_complex(tw2),
            twiddle4: AvxStoreF::set_complex(tw4),
            bf3: ColumnButterfly3f::new(direction),
        }
    }
}

impl ColumnButterfly9f {
    #[inline(always)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 9]) -> [AvxStoreF; 9] {
        let [u0, u3, u6] = self.bf3.exec([v[0], v[3], v[6]]);
        let [u1, mut u4, mut u7] = self.bf3.exec([v[1], v[4], v[7]]);
        let [u2, mut u5, mut u8] = self.bf3.exec([v[2], v[5], v[8]]);

        u4 = AvxStoreF::mul_by_complex(u4, self.twiddle1);
        u7 = AvxStoreF::mul_by_complex(u7, self.twiddle2);
        u5 = AvxStoreF::mul_by_complex(u5, self.twiddle2);
        u8 = AvxStoreF::mul_by_complex(u8, self.twiddle4);

        let [y0, y3, y6] = self.bf3.exec([u0, u1, u2]);
        let [y1, y4, y7] = self.bf3.exec([u3, u4, u5]);
        let [y2, y5, y8] = self.bf3.exec([u6, u7, u8]);
        [y0, y1, y2, y3, y4, y5, y6, y7, y8]
    }
}

pub(crate) struct ColumnRdftButterfly9f {
    twiddle1: AvxStoreF,
    twiddle2: AvxStoreF,
    twiddle3: AvxStoreF,
    twiddle4: AvxStoreF,
    d6: AvxStoreF,
    d7: AvxStoreF,
    d8: AvxStoreF,
    m_half: AvxStoreF,
}

impl ColumnRdftButterfly9f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new() -> Self {
        let one_over_3 = (1f64 / 3f64) as f32;
        let twiddle1 = compute_twiddle::<f32>(2, 9, FftDirection::Forward);
        let twiddle2 = compute_twiddle::<f32>(4, 9, FftDirection::Forward);
        let twiddle3 = compute_twiddle::<f32>(6, 9, FftDirection::Forward);
        let twiddle4 = compute_twiddle::<f32>(8, 9, FftDirection::Forward);
        let h0 = twiddle4.re + twiddle1.re; // cos(φ) + cos(2φ)
        let d6 = (2.0 * twiddle4.re - twiddle1.re - twiddle2.re) * one_over_3;
        let d7 = (-twiddle4.re + 2.0 * twiddle1.re - twiddle2.re) * one_over_3;
        let d8 = (-h0 + 2.0 * twiddle2.re) * one_over_3;

        Self {
            twiddle1: AvxStoreF::dup(twiddle1.im),
            twiddle2: AvxStoreF::dup(twiddle2.im),
            twiddle3: AvxStoreF::dup(twiddle3.im),
            twiddle4: AvxStoreF::dup(twiddle4.im),
            d6: AvxStoreF::dup(d6),
            d7: AvxStoreF::dup(d7),
            d8: AvxStoreF::dup(d8),
            m_half: AvxStoreF::dup(-f32::HALF),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec(&self, store: [AvxStoreF; 9]) -> [[AvxStoreF; 5]; 2] {
        let t1 = store[1] + store[8];
        let t2 = store[2] + store[7];
        let t3 = store[3] + store[6];
        let t4 = store[4] + store[5];
        let t5 = store[4] - store[5];
        let t6 = store[3] - store[6];
        let t7 = store[2] - store[7];
        let t8 = store[1] - store[8];

        // DC
        let r0 = t1 + t2 + t4;
        let z0 = t8 - t7 + t5;
        let y0 = store[0] + r0 + t3;

        let y3 = r0.mul_add(self.m_half, store[0] + t3);

        let r1 = t1 - t4;
        let r2 = t2 - t4;
        let r3 = -t1 + t2;

        let m2 = self.d6 * r1;
        let m3 = self.d7 * r2;
        let m4 = self.d8 * r3;

        let re_base = t3.mul_add(self.m_half, store[0]);
        let y1 = re_base + m2 + m3;
        let y2 = re_base - m2 + m4;
        let y4 = re_base - m3 - m4;

        let y5 = -t8.mul_add(
            self.twiddle4,
            t7.mul_add(
                -self.twiddle1,
                t6.mul_add(self.twiddle3, t5 * -self.twiddle2),
            ),
        ); // X[1].im

        let y6 = -t8.mul_add(
            -self.twiddle1,
            t7.mul_add(
                -self.twiddle2,
                t6.mul_add(-self.twiddle3, t5 * -self.twiddle4),
            ),
        ); // X[2].im

        let y7 = z0 * -self.twiddle3; // X[3].im: -sin(3φ)*(t8-t7+t5)

        let y8 = t8.mul_add(
            self.twiddle2,
            t7.mul_add(
                self.twiddle4,
                t6.mul_add(-self.twiddle3, t5 * -self.twiddle1),
            ),
        );

        let v0 = y0.zip(AvxStoreF::zero());
        let v1 = y1.zip(y5);
        let v2 = y2.zip(y6);
        let v3 = y3.zip(y7);
        let v4 = y4.zip(y8);
        [
            [v0[0], v1[0], v2[0], v3[0], v4[0]],
            [v0[1], v1[1], v2[1], v3[1], v4[1]],
        ]
    }
}

pub(crate) struct ColumnRdftButterfly9d {
    twiddle1: AvxStoreD,
    twiddle2: AvxStoreD,
    twiddle3: AvxStoreD,
    twiddle4: AvxStoreD,
    d6: AvxStoreD,
    d7: AvxStoreD,
    d8: AvxStoreD,
    m_half: AvxStoreD,
}

impl ColumnRdftButterfly9d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new() -> Self {
        let one_over_3 = 1f64 / 3f64;
        let twiddle1 = compute_twiddle::<f64>(2, 9, FftDirection::Forward);
        let twiddle2 = compute_twiddle::<f64>(4, 9, FftDirection::Forward);
        let twiddle3 = compute_twiddle::<f64>(6, 9, FftDirection::Forward);
        let twiddle4 = compute_twiddle::<f64>(8, 9, FftDirection::Forward);
        let h0 = twiddle4.re + twiddle1.re; // cos(φ) + cos(2φ)
        let d6 = (2.0 * twiddle4.re - twiddle1.re - twiddle2.re) * one_over_3;
        let d7 = (-twiddle4.re + 2.0 * twiddle1.re - twiddle2.re) * one_over_3;
        let d8 = (-h0 + 2.0 * twiddle2.re) * one_over_3;

        Self {
            twiddle1: AvxStoreD::dup(twiddle1.im),
            twiddle2: AvxStoreD::dup(twiddle2.im),
            twiddle3: AvxStoreD::dup(twiddle3.im),
            twiddle4: AvxStoreD::dup(twiddle4.im),
            d6: AvxStoreD::dup(d6),
            d7: AvxStoreD::dup(d7),
            d8: AvxStoreD::dup(d8),
            m_half: AvxStoreD::dup(-f64::HALF),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn exec(&self, store: [AvxStoreD; 9]) -> [[AvxStoreD; 5]; 2] {
        let t1 = store[1] + store[8];
        let t2 = store[2] + store[7];
        let t3 = store[3] + store[6];
        let t4 = store[4] + store[5];
        let t5 = store[4] - store[5];
        let t6 = store[3] - store[6];
        let t7 = store[2] - store[7];
        let t8 = store[1] - store[8];

        // DC
        let r0 = t1 + t2 + t4;
        let z0 = t8 - t7 + t5;
        let y0 = store[0] + r0 + t3;

        let y3 = r0.mul_add(self.m_half, store[0] + t3);

        let r1 = t1 - t4;
        let r2 = t2 - t4;
        let r3 = -t1 + t2;

        let m2 = self.d6 * r1;
        let m3 = self.d7 * r2;
        let m4 = self.d8 * r3;

        let re_base = t3.mul_add(self.m_half, store[0]);
        let y1 = re_base + m2 + m3;
        let y2 = re_base - m2 + m4;
        let y4 = re_base - m3 - m4;

        let y5 = -t8.mul_add(
            self.twiddle4,
            t7.mul_add(
                -self.twiddle1,
                t6.mul_add(self.twiddle3, t5 * -self.twiddle2),
            ),
        ); // X[1].im

        let y6 = -t8.mul_add(
            -self.twiddle1,
            t7.mul_add(
                -self.twiddle2,
                t6.mul_add(-self.twiddle3, t5 * -self.twiddle4),
            ),
        ); // X[2].im

        let y7 = z0 * -self.twiddle3; // X[3].im: -sin(3φ)*(t8-t7+t5)

        let y8 = t8.mul_add(
            self.twiddle2,
            t7.mul_add(
                self.twiddle4,
                t6.mul_add(-self.twiddle3, t5 * -self.twiddle1),
            ),
        );

        let v0 = y0.zip(AvxStoreD::zero());
        let v1 = y1.zip(y5);
        let v2 = y2.zip(y6);
        let v3 = y3.zip(y7);
        let v4 = y4.zip(y8);
        [
            [v0[0], v1[0], v2[0], v3[0], v4[0]],
            [v0[1], v1[1], v2[1], v3[1], v4[1]],
        ]
    }
}
