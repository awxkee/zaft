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

use crate::avx::butterflies::shared::{boring_avx_butterfly, gen_butterfly_twiddles_f32};
use crate::avx::mixed::{AvxStoreF, ColumnButterfly9f};
use crate::avx::transpose::avx_transpose_f32x2_4x4_impl;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::_mm256_setzero_ps;
use std::mem::MaybeUninit;

pub(crate) struct ColumnButterfly27f {
    bf9: ColumnButterfly9f,
    twiddle1: AvxStoreF,
    twiddle2: AvxStoreF,
    twiddle3: AvxStoreF,
    twiddle4: AvxStoreF,
    twiddle5: AvxStoreF,
    twiddle6: AvxStoreF,
    twiddle7: AvxStoreF,
    twiddle8: AvxStoreF,
    twiddle9: AvxStoreF,
    twiddle10: AvxStoreF,
    twiddle11: AvxStoreF,
    twiddle12: AvxStoreF,
}

impl ColumnButterfly27f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            twiddle1: AvxStoreF::set_complex(compute_twiddle(1, 27, fft_direction)),
            twiddle2: AvxStoreF::set_complex(compute_twiddle(2, 27, fft_direction)),
            twiddle3: AvxStoreF::set_complex(compute_twiddle(3, 27, fft_direction)),
            twiddle4: AvxStoreF::set_complex(compute_twiddle(4, 27, fft_direction)),
            twiddle5: AvxStoreF::set_complex(compute_twiddle(5, 27, fft_direction)),
            twiddle6: AvxStoreF::set_complex(compute_twiddle(6, 27, fft_direction)),
            twiddle7: AvxStoreF::set_complex(compute_twiddle(7, 27, fft_direction)),
            twiddle8: AvxStoreF::set_complex(compute_twiddle(8, 27, fft_direction)),
            twiddle9: AvxStoreF::set_complex(compute_twiddle(10, 27, fft_direction)),
            twiddle10: AvxStoreF::set_complex(compute_twiddle(12, 27, fft_direction)),
            twiddle11: AvxStoreF::set_complex(compute_twiddle(14, 27, fft_direction)),
            twiddle12: AvxStoreF::set_complex(compute_twiddle(16, 27, fft_direction)),
            bf9: ColumnButterfly9f::new(fft_direction),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn exec(&self, src: &[MaybeUninit<Complex<f32>>], dst: &mut [Complex<f32>]) {
        macro_rules! load {
            ($src: expr, $idx: expr) => {{ unsafe { AvxStoreF::from_complex_refu($src.get_unchecked($idx * 9..)) } }};
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

        let z = self.bf9.bf3.exec([s0[0], s1[0], s2[0]]);
        store!(z[0], 0, dst);
        store!(z[1], 9, dst);
        store!(z[2], 18, dst);

        s1[1] = AvxStoreF::mul_by_complex(s1[1], self.twiddle1);
        s1[2] = AvxStoreF::mul_by_complex(s1[2], self.twiddle2);
        s1[3] = AvxStoreF::mul_by_complex(s1[3], self.twiddle3);
        s1[4] = AvxStoreF::mul_by_complex(s1[4], self.twiddle4);
        s1[5] = AvxStoreF::mul_by_complex(s1[5], self.twiddle5);
        s1[6] = AvxStoreF::mul_by_complex(s1[6], self.twiddle6);
        s1[7] = AvxStoreF::mul_by_complex(s1[7], self.twiddle7);
        s1[8] = AvxStoreF::mul_by_complex(s1[8], self.twiddle8);
        s2[1] = AvxStoreF::mul_by_complex(s2[1], self.twiddle2);
        s2[2] = AvxStoreF::mul_by_complex(s2[2], self.twiddle4);
        s2[3] = AvxStoreF::mul_by_complex(s2[3], self.twiddle6);
        s2[4] = AvxStoreF::mul_by_complex(s2[4], self.twiddle8);
        s2[5] = AvxStoreF::mul_by_complex(s2[5], self.twiddle9);
        s2[6] = AvxStoreF::mul_by_complex(s2[6], self.twiddle10);
        s2[7] = AvxStoreF::mul_by_complex(s2[7], self.twiddle11);
        s2[8] = AvxStoreF::mul_by_complex(s2[8], self.twiddle12);

        for i in 1..9 {
            let z = self.bf9.bf3.exec([s0[i], s1[i], s2[i]]);
            store!(z[0], i, dst);
            store!(z[1], i + 9, dst);
            store!(z[2], i + 18, dst);
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn exech(&self, src: &[MaybeUninit<Complex<f32>>], dst: &mut [Complex<f32>]) {
        macro_rules! load {
            ($src: expr, $idx: expr) => {{ unsafe { AvxStoreF::from_complexu($src.get_unchecked($idx * 9)) } }};
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
            ($v: expr, $idx: expr, $dst: expr) => {{ unsafe { $v.write_lo1($dst.get_unchecked_mut($idx * 9..)) } }};
        }

        let z = self.bf9.bf3.exec([s0[0], s1[0], s2[0]]);
        store!(z[0], 0, dst);
        store!(z[1], 9, dst);
        store!(z[2], 18, dst);

        s1[1] = AvxStoreF::mul_by_complex(s1[1], self.twiddle1);
        s1[2] = AvxStoreF::mul_by_complex(s1[2], self.twiddle2);
        s1[3] = AvxStoreF::mul_by_complex(s1[3], self.twiddle3);
        s1[4] = AvxStoreF::mul_by_complex(s1[4], self.twiddle4);
        s1[5] = AvxStoreF::mul_by_complex(s1[5], self.twiddle5);
        s1[6] = AvxStoreF::mul_by_complex(s1[6], self.twiddle6);
        s1[7] = AvxStoreF::mul_by_complex(s1[7], self.twiddle7);
        s1[8] = AvxStoreF::mul_by_complex(s1[8], self.twiddle8);
        s2[1] = AvxStoreF::mul_by_complex(s2[1], self.twiddle2);
        s2[2] = AvxStoreF::mul_by_complex(s2[2], self.twiddle4);
        s2[3] = AvxStoreF::mul_by_complex(s2[3], self.twiddle6);
        s2[4] = AvxStoreF::mul_by_complex(s2[4], self.twiddle8);
        s2[5] = AvxStoreF::mul_by_complex(s2[5], self.twiddle9);
        s2[6] = AvxStoreF::mul_by_complex(s2[6], self.twiddle10);
        s2[7] = AvxStoreF::mul_by_complex(s2[7], self.twiddle11);
        s2[8] = AvxStoreF::mul_by_complex(s2[8], self.twiddle12);

        for i in 1..9 {
            let z = self.bf9.bf3.exec([s0[i], s1[i], s2[i]]);
            store!(z[0], i, dst);
            store!(z[1], i + 9, dst);
            store!(z[2], i + 18, dst);
        }
    }
}

pub(crate) struct AvxButterfly243f {
    direction: FftDirection,
    bf27: ColumnButterfly27f,
    twiddles: [AvxStoreF; 56],
}

impl AvxButterfly243f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(27, 9, fft_direction, 243),
            bf27: ColumnButterfly27f::new(fft_direction),
        }
    }
}

boring_avx_butterfly!(AvxButterfly243f, f32, 243);

impl AvxButterfly243f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows: [AvxStoreF; 9] = [AvxStoreF::zero(); 9];
        let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 243];
        unsafe {
            // columns
            for k in 0..6 {
                for i in 0..9 {
                    rows[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 27 + k * 4..));
                }

                rows = self.bf27.bf9.exec(rows);

                let q1 = AvxStoreF::mul_by_complex(rows[1], self.twiddles[8 * k]);
                let q2 = AvxStoreF::mul_by_complex(rows[2], self.twiddles[8 * k + 1]);
                let q3 = AvxStoreF::mul_by_complex(rows[3], self.twiddles[8 * k + 2]);
                let t = avx_transpose_f32x2_4x4_impl(rows[0].v, q1.v, q2.v, q3.v);
                AvxStoreF::raw(t.0).write_u(scratch.get_unchecked_mut(k * 4 * 9..));
                AvxStoreF::raw(t.1).write_u(scratch.get_unchecked_mut((k * 4 + 1) * 9..));
                AvxStoreF::raw(t.2).write_u(scratch.get_unchecked_mut((k * 4 + 2) * 9..));
                AvxStoreF::raw(t.3).write_u(scratch.get_unchecked_mut((k * 4 + 3) * 9..));

                {
                    let i = 1;
                    let q0 = AvxStoreF::mul_by_complex(
                        rows[i * 4],
                        self.twiddles[(i - 1) * 4 + 3 + 8 * k],
                    );
                    let q1 = AvxStoreF::mul_by_complex(
                        rows[i * 4 + 1],
                        self.twiddles[(i - 1) * 4 + 4 + 8 * k],
                    );
                    let q2 = AvxStoreF::mul_by_complex(
                        rows[i * 4 + 2],
                        self.twiddles[(i - 1) * 4 + 5 + 8 * k],
                    );
                    let q3 = AvxStoreF::mul_by_complex(
                        rows[i * 4 + 3],
                        self.twiddles[(i - 1) * 4 + 6 + 8 * k],
                    );
                    let t = avx_transpose_f32x2_4x4_impl(q0.v, q1.v, q2.v, q3.v);
                    AvxStoreF::raw(t.0).write_u(scratch.get_unchecked_mut(k * 4 * 9 + i * 4..));
                    AvxStoreF::raw(t.1)
                        .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 9 + i * 4..));
                    AvxStoreF::raw(t.2)
                        .write_u(scratch.get_unchecked_mut((k * 4 + 2) * 9 + i * 4..));
                    AvxStoreF::raw(t.3)
                        .write_u(scratch.get_unchecked_mut((k * 4 + 3) * 9 + i * 4..));
                }

                {
                    let i = 2;
                    let q0 = AvxStoreF::mul_by_complex(
                        rows[i * 4],
                        self.twiddles[(i - 1) * 4 + 3 + 8 * k],
                    );
                    let t = avx_transpose_f32x2_4x4_impl(
                        q0.v,
                        _mm256_setzero_ps(),
                        _mm256_setzero_ps(),
                        _mm256_setzero_ps(),
                    );
                    AvxStoreF::raw(t.0).write_lo1u(scratch.get_unchecked_mut(k * 4 * 9 + i * 4..));
                    AvxStoreF::raw(t.1)
                        .write_lo1u(scratch.get_unchecked_mut((k * 4 + 1) * 9 + i * 4..));
                    AvxStoreF::raw(t.2)
                        .write_lo1u(scratch.get_unchecked_mut((k * 4 + 2) * 9 + i * 4..));
                    AvxStoreF::raw(t.3)
                        .write_lo1u(scratch.get_unchecked_mut((k * 4 + 3) * 9 + i * 4..));
                }
            }

            {
                let k = 6;
                for i in 0..9 {
                    rows[i] = AvxStoreF::from_complex3(chunk.slice_from(i * 27 + k * 4..));
                }

                rows = self.bf27.bf9.exec(rows);

                let q1 = AvxStoreF::mul_by_complex(rows[1], self.twiddles[8 * k]);
                let q2 = AvxStoreF::mul_by_complex(rows[2], self.twiddles[8 * k + 1]);
                let q3 = AvxStoreF::mul_by_complex(rows[3], self.twiddles[8 * k + 2]);
                let t = avx_transpose_f32x2_4x4_impl(rows[0].v, q1.v, q2.v, q3.v);
                AvxStoreF::raw(t.0).write_u(scratch.get_unchecked_mut(k * 4 * 9..));
                AvxStoreF::raw(t.1).write_u(scratch.get_unchecked_mut((k * 4 + 1) * 9..));
                AvxStoreF::raw(t.2).write_u(scratch.get_unchecked_mut((k * 4 + 2) * 9..));

                {
                    let i = 1;
                    let q0 = AvxStoreF::mul_by_complex(
                        rows[i * 4],
                        self.twiddles[(i - 1) * 4 + 3 + 8 * k],
                    );
                    let q1 = AvxStoreF::mul_by_complex(
                        rows[i * 4 + 1],
                        self.twiddles[(i - 1) * 4 + 4 + 8 * k],
                    );
                    let q2 = AvxStoreF::mul_by_complex(
                        rows[i * 4 + 2],
                        self.twiddles[(i - 1) * 4 + 5 + 8 * k],
                    );
                    let q3 = AvxStoreF::mul_by_complex(
                        rows[i * 4 + 3],
                        self.twiddles[(i - 1) * 4 + 6 + 8 * k],
                    );
                    let t = avx_transpose_f32x2_4x4_impl(q0.v, q1.v, q2.v, q3.v);
                    AvxStoreF::raw(t.0).write_u(scratch.get_unchecked_mut(k * 4 * 9 + i * 4..));
                    AvxStoreF::raw(t.1)
                        .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 9 + i * 4..));
                    AvxStoreF::raw(t.2)
                        .write_u(scratch.get_unchecked_mut((k * 4 + 2) * 9 + i * 4..));
                }

                {
                    let i = 2;
                    let q0 = AvxStoreF::mul_by_complex(
                        rows[i * 4],
                        self.twiddles[(i - 1) * 4 + 3 + 8 * k],
                    );
                    let t = avx_transpose_f32x2_4x4_impl(
                        q0.v,
                        _mm256_setzero_ps(),
                        _mm256_setzero_ps(),
                        _mm256_setzero_ps(),
                    );
                    AvxStoreF::raw(t.0).write_lo1u(scratch.get_unchecked_mut(k * 4 * 9 + i * 4..));
                    AvxStoreF::raw(t.1)
                        .write_lo1u(scratch.get_unchecked_mut((k * 4 + 1) * 9 + i * 4..));
                    AvxStoreF::raw(t.2)
                        .write_lo1u(scratch.get_unchecked_mut((k * 4 + 2) * 9 + i * 4..));
                }
            }

            // rows

            for k in 0..2 {
                self.bf27.exec(
                    scratch.get_unchecked(k * 4..),
                    chunk.slice_from_mut(k * 4..),
                );
            }

            {
                let k = 2;
                self.bf27.exech(
                    scratch.get_unchecked(k * 4..),
                    chunk.slice_from_mut(k * 4..),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly243, f32, AvxButterfly243f, 243, 1e-3);
    test_oof_avx_butterfly!(test_oof_avx_butterfly243, f32, AvxButterfly243f, 243, 1e-3);
}
