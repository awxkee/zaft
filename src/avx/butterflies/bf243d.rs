/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
#![allow(clippy::needless_range_loop)]

use crate::avx::butterflies::shared::{boring_avx_butterfly, gen_butterfly_twiddles_f64};
use crate::avx::mixed::{AvxStoreD, ColumnButterfly9d};
use crate::avx::transpose::transpose_f64x2_2x2d;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct ColumnButterfly27d {
    bf9: ColumnButterfly9d,
    twiddle1: AvxStoreD,
    twiddle2: AvxStoreD,
    twiddle3: AvxStoreD,
    twiddle4: AvxStoreD,
    twiddle5: AvxStoreD,
    twiddle6: AvxStoreD,
    twiddle7: AvxStoreD,
    twiddle8: AvxStoreD,
    twiddle9: AvxStoreD,
    twiddle10: AvxStoreD,
    twiddle11: AvxStoreD,
    twiddle12: AvxStoreD,
}

impl ColumnButterfly27d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            twiddle1: AvxStoreD::set_complex(&compute_twiddle(1, 27, fft_direction)),
            twiddle2: AvxStoreD::set_complex(&compute_twiddle(2, 27, fft_direction)),
            twiddle3: AvxStoreD::set_complex(&compute_twiddle(3, 27, fft_direction)),
            twiddle4: AvxStoreD::set_complex(&compute_twiddle(4, 27, fft_direction)),
            twiddle5: AvxStoreD::set_complex(&compute_twiddle(5, 27, fft_direction)),
            twiddle6: AvxStoreD::set_complex(&compute_twiddle(6, 27, fft_direction)),
            twiddle7: AvxStoreD::set_complex(&compute_twiddle(7, 27, fft_direction)),
            twiddle8: AvxStoreD::set_complex(&compute_twiddle(8, 27, fft_direction)),
            twiddle9: AvxStoreD::set_complex(&compute_twiddle(10, 27, fft_direction)),
            twiddle10: AvxStoreD::set_complex(&compute_twiddle(12, 27, fft_direction)),
            twiddle11: AvxStoreD::set_complex(&compute_twiddle(14, 27, fft_direction)),
            twiddle12: AvxStoreD::set_complex(&compute_twiddle(16, 27, fft_direction)),
            bf9: ColumnButterfly9d::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn exec(&self, src: &[MaybeUninit<Complex<f64>>], dst: &mut [Complex<f64>]) {
        macro_rules! load {
            ($src: expr, $idx: expr) => {{ unsafe { AvxStoreD::from_complex_refu($src.get_unchecked($idx * 9..)) } }};
        }

        let mut s1 = self.bf9.exec([
            load!(src, 1),
            load!(src, 4),
            load!(src, 7),
            load!(src, 10),
            load!(src, 13),
            load!(src, 16),
            load!(src, 19),
            load!(src, 22),
            load!(src, 25),
        ]);

        let mut s2 = self.bf9.exec([
            load!(src, 2),
            load!(src, 5),
            load!(src, 8),
            load!(src, 11),
            load!(src, 14),
            load!(src, 17),
            load!(src, 20),
            load!(src, 23),
            load!(src, 26),
        ]);

        macro_rules! store {
            ($v: expr, $idx: expr, $dst: expr) => {{ unsafe { $v.write($dst.get_unchecked_mut($idx * 9..)) } }};
        }

        let s0 = self.bf9.exec([
            load!(src, 0),
            load!(src, 3),
            load!(src, 6),
            load!(src, 9),
            load!(src, 12),
            load!(src, 15),
            load!(src, 18),
            load!(src, 21),
            load!(src, 24),
        ]);

        let z0 = self.bf9.bf3.exec([s0[0], s1[0], s2[0]]);
        store!(z0[0], 0, dst);
        store!(z0[1], 9, dst);
        store!(z0[2], 18, dst);

        s1[1] = AvxStoreD::mul_by_complex(s1[1], self.twiddle1);
        s2[1] = AvxStoreD::mul_by_complex(s2[1], self.twiddle2);
        let z1 = self.bf9.bf3.exec([s0[1], s1[1], s2[1]]);
        store!(z1[0], 1, dst);
        store!(z1[1], 10, dst);
        store!(z1[2], 19, dst);

        s1[2] = AvxStoreD::mul_by_complex(s1[2], self.twiddle2);
        s2[2] = AvxStoreD::mul_by_complex(s2[2], self.twiddle4);
        let z2 = self.bf9.bf3.exec([s0[2], s1[2], s2[2]]);
        store!(z2[0], 2, dst);
        store!(z2[1], 11, dst);
        store!(z2[2], 20, dst);

        s1[3] = AvxStoreD::mul_by_complex(s1[3], self.twiddle3);
        s2[3] = AvxStoreD::mul_by_complex(s2[3], self.twiddle6);
        let z3 = self.bf9.bf3.exec([s0[3], s1[3], s2[3]]);
        store!(z3[0], 3, dst);
        store!(z3[1], 12, dst);
        store!(z3[2], 21, dst);

        s1[4] = AvxStoreD::mul_by_complex(s1[4], self.twiddle4);
        s2[4] = AvxStoreD::mul_by_complex(s2[4], self.twiddle8);
        let z4 = self.bf9.bf3.exec([s0[4], s1[4], s2[4]]);
        store!(z4[0], 4, dst);
        store!(z4[1], 13, dst);
        store!(z4[2], 22, dst);

        s1[5] = AvxStoreD::mul_by_complex(s1[5], self.twiddle5);
        s2[5] = AvxStoreD::mul_by_complex(s2[5], self.twiddle9);
        let z5 = self.bf9.bf3.exec([s0[5], s1[5], s2[5]]);
        store!(z5[0], 5, dst);
        store!(z5[1], 14, dst);
        store!(z5[2], 23, dst);

        s1[6] = AvxStoreD::mul_by_complex(s1[6], self.twiddle6);
        s2[6] = AvxStoreD::mul_by_complex(s2[6], self.twiddle10);
        let z6 = self.bf9.bf3.exec([s0[6], s1[6], s2[6]]);
        store!(z6[0], 6, dst);
        store!(z6[1], 15, dst);
        store!(z6[2], 24, dst);

        s1[7] = AvxStoreD::mul_by_complex(s1[7], self.twiddle7);
        s2[7] = AvxStoreD::mul_by_complex(s2[7], self.twiddle11);
        let z7 = self.bf9.bf3.exec([s0[7], s1[7], s2[7]]);
        store!(z7[0], 7, dst);
        store!(z7[1], 16, dst);
        store!(z7[2], 25, dst);

        s1[8] = AvxStoreD::mul_by_complex(s1[8], self.twiddle8);
        s2[8] = AvxStoreD::mul_by_complex(s2[8], self.twiddle12);
        let z8 = self.bf9.bf3.exec([s0[8], s1[8], s2[8]]);
        store!(z8[0], 8, dst);
        store!(z8[1], 17, dst);
        store!(z8[2], 26, dst);
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn exech(&self, src: &[MaybeUninit<Complex<f64>>], dst: &mut [Complex<f64>]) {
        macro_rules! load {
            ($src: expr, $idx: expr) => {{ unsafe { AvxStoreD::from_complexu($src.get_unchecked($idx * 9)) } }};
        }

        let mut s1 = self.bf9.exec([
            load!(src, 1),
            load!(src, 4),
            load!(src, 7),
            load!(src, 10),
            load!(src, 13),
            load!(src, 16),
            load!(src, 19),
            load!(src, 22),
            load!(src, 25),
        ]);

        let mut s2 = self.bf9.exec([
            load!(src, 2),
            load!(src, 5),
            load!(src, 8),
            load!(src, 11),
            load!(src, 14),
            load!(src, 17),
            load!(src, 20),
            load!(src, 23),
            load!(src, 26),
        ]);

        macro_rules! store {
            ($v: expr, $idx: expr, $dst: expr) => {{ unsafe { $v.write_lo($dst.get_unchecked_mut($idx * 9..)) } }};
        }

        let s0 = self.bf9.exec([
            load!(src, 0),
            load!(src, 3),
            load!(src, 6),
            load!(src, 9),
            load!(src, 12),
            load!(src, 15),
            load!(src, 18),
            load!(src, 21),
            load!(src, 24),
        ]);

        let z0 = self.bf9.bf3.exec([s0[0], s1[0], s2[0]]);
        store!(z0[0], 0, dst);
        store!(z0[1], 9, dst);
        store!(z0[2], 18, dst);

        s1[1] = AvxStoreD::mul_by_complex(s1[1], self.twiddle1);
        s2[1] = AvxStoreD::mul_by_complex(s2[1], self.twiddle2);
        let z1 = self.bf9.bf3.exec([s0[1], s1[1], s2[1]]);
        store!(z1[0], 1, dst);
        store!(z1[1], 10, dst);
        store!(z1[2], 19, dst);

        s1[2] = AvxStoreD::mul_by_complex(s1[2], self.twiddle2);
        s2[2] = AvxStoreD::mul_by_complex(s2[2], self.twiddle4);
        let z2 = self.bf9.bf3.exec([s0[2], s1[2], s2[2]]);
        store!(z2[0], 2, dst);
        store!(z2[1], 11, dst);
        store!(z2[2], 20, dst);

        s1[3] = AvxStoreD::mul_by_complex(s1[3], self.twiddle3);
        s2[3] = AvxStoreD::mul_by_complex(s2[3], self.twiddle6);
        let z3 = self.bf9.bf3.exec([s0[3], s1[3], s2[3]]);
        store!(z3[0], 3, dst);
        store!(z3[1], 12, dst);
        store!(z3[2], 21, dst);

        s1[4] = AvxStoreD::mul_by_complex(s1[4], self.twiddle4);
        s2[4] = AvxStoreD::mul_by_complex(s2[4], self.twiddle8);
        let z4 = self.bf9.bf3.exec([s0[4], s1[4], s2[4]]);
        store!(z4[0], 4, dst);
        store!(z4[1], 13, dst);
        store!(z4[2], 22, dst);

        s1[5] = AvxStoreD::mul_by_complex(s1[5], self.twiddle5);
        s2[5] = AvxStoreD::mul_by_complex(s2[5], self.twiddle9);
        let z5 = self.bf9.bf3.exec([s0[5], s1[5], s2[5]]);
        store!(z5[0], 5, dst);
        store!(z5[1], 14, dst);
        store!(z5[2], 23, dst);

        s1[6] = AvxStoreD::mul_by_complex(s1[6], self.twiddle6);
        s2[6] = AvxStoreD::mul_by_complex(s2[6], self.twiddle10);
        let z6 = self.bf9.bf3.exec([s0[6], s1[6], s2[6]]);
        store!(z6[0], 6, dst);
        store!(z6[1], 15, dst);
        store!(z6[2], 24, dst);

        s1[7] = AvxStoreD::mul_by_complex(s1[7], self.twiddle7);
        s2[7] = AvxStoreD::mul_by_complex(s2[7], self.twiddle11);
        let z7 = self.bf9.bf3.exec([s0[7], s1[7], s2[7]]);
        store!(z7[0], 7, dst);
        store!(z7[1], 16, dst);
        store!(z7[2], 25, dst);

        s1[8] = AvxStoreD::mul_by_complex(s1[8], self.twiddle8);
        s2[8] = AvxStoreD::mul_by_complex(s2[8], self.twiddle12);
        let z8 = self.bf9.bf3.exec([s0[8], s1[8], s2[8]]);
        store!(z8[0], 8, dst);
        store!(z8[1], 17, dst);
        store!(z8[2], 26, dst);
    }
}

pub(crate) struct AvxButterfly243d {
    direction: FftDirection,
    bf27: ColumnButterfly27d,
    twiddles: [AvxStoreD; 112],
}

impl AvxButterfly243d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2,fma")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(27, 9, fft_direction, 243),
            bf27: ColumnButterfly27d::new(fft_direction),
        }
    }
}

impl AvxButterfly243d {
    #[target_feature(enable = "avx2,fma")]
    fn exec_bf9(&self, src: &[Complex<f64>], dst: &mut [MaybeUninit<Complex<f64>>; 243]) {
        let mut rows: [AvxStoreD; 9] = [AvxStoreD::zero(); 9];
        // columns
        for k in 0..13 {
            for i in 0..9 {
                unsafe {
                    rows[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 27 + k * 2..));
                }
            }

            rows = self.bf27.bf9.exec(rows);

            let q1 = AvxStoreD::mul_by_complex(rows[1], self.twiddles[8 * k]);
            let t = transpose_f64x2_2x2d([rows[0], q1]);
            unsafe {
                t[0].write_u(dst.get_unchecked_mut(k * 2 * 9..));
                t[1].write_u(dst.get_unchecked_mut((k * 2 + 1) * 9..));
            }

            for i in 1..4 {
                let q0 =
                    AvxStoreD::mul_by_complex(rows[i * 2], self.twiddles[(i - 1) * 2 + 1 + 8 * k]);
                let q1 = AvxStoreD::mul_by_complex(
                    rows[i * 2 + 1],
                    self.twiddles[(i - 1) * 2 + 2 + 8 * k],
                );
                let t = transpose_f64x2_2x2d([q0, q1]);
                unsafe {
                    t[0].write_u(dst.get_unchecked_mut(k * 2 * 9 + i * 2..));
                    t[1].write_u(dst.get_unchecked_mut((k * 2 + 1) * 9 + i * 2..));
                }
            }

            {
                let i = 4;
                let q0 =
                    AvxStoreD::mul_by_complex(rows[i * 2], self.twiddles[(i - 1) * 2 + 1 + 8 * k]);
                let t = transpose_f64x2_2x2d([q0, AvxStoreD::undefined()]);
                unsafe {
                    t[0].write_lou(dst.get_unchecked_mut(k * 2 * 9 + i * 2..));
                    t[1].write_lou(dst.get_unchecked_mut((k * 2 + 1) * 9 + i * 2..));
                }
            }
        }

        {
            let k = 13;
            for i in 0..9 {
                unsafe {
                    rows[i] = AvxStoreD::from_complex(src.get_unchecked(i * 27 + k * 2));
                }
            }

            rows = self.bf27.bf9.exec(rows);

            let q1 = AvxStoreD::mul_by_complex(rows[1], self.twiddles[8 * k]);
            let t = transpose_f64x2_2x2d([rows[0], q1]);
            unsafe {
                t[0].write_u(dst.get_unchecked_mut(k * 2 * 9..));
            }

            for i in 1..4 {
                let q0 =
                    AvxStoreD::mul_by_complex(rows[i * 2], self.twiddles[(i - 1) * 2 + 1 + 8 * k]);
                let q1 = AvxStoreD::mul_by_complex(
                    rows[i * 2 + 1],
                    self.twiddles[(i - 1) * 2 + 2 + 8 * k],
                );
                let t = transpose_f64x2_2x2d([q0, q1]);
                unsafe {
                    t[0].write_u(dst.get_unchecked_mut(k * 2 * 9 + i * 2..));
                }
            }

            {
                let i = 4;
                let q0 =
                    AvxStoreD::mul_by_complex(rows[i * 2], self.twiddles[(i - 1) * 2 + 1 + 8 * k]);
                let t = transpose_f64x2_2x2d([q0, AvxStoreD::undefined()]);
                unsafe {
                    t[0].write_lou(dst.get_unchecked_mut(k * 2 * 9 + i * 2..));
                }
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    fn exec_bf27(&self, src: &[MaybeUninit<Complex<f64>>; 243], dst: &mut [Complex<f64>]) {
        unsafe {
            for k in 0..4 {
                self.bf27
                    .exec(src.get_unchecked(k * 2..), dst.get_unchecked_mut(k * 2..));
            }

            {
                let k = 4;
                self.bf27
                    .exech(src.get_unchecked(k * 2..), dst.get_unchecked_mut(k * 2..));
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 243];

        self.exec_bf9(chunk.slice_from(0..), &mut scratch);
        // rows
        self.exec_bf27(&scratch, chunk.slice_from_mut(0..));
    }
}

boring_avx_butterfly!(AvxButterfly243d, f64, 243);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly243_f64, f64, AvxButterfly243d, 243, 1e-7);
    test_oof_avx_butterfly!(
        test_oof_avx_butterfly243_f64,
        f64,
        AvxButterfly243d,
        243,
        1e-7
    );
}
