/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
use crate::avx::mixed::{AvxStoreF, ColumnButterfly13f};
use crate::avx::transpose::avx_transpose_f32x2_4x4_impl;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::_mm256_setzero_ps;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly169f {
    direction: FftDirection,
    bf13: ColumnButterfly13f,
    twiddles: [AvxStoreF; 48],
}

impl AvxButterfly169f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(13, 13, fft_direction, 169),
            bf13: ColumnButterfly13f::new(fft_direction),
        }
    }
}

boring_avx_butterfly!(AvxButterfly169f, f32, 169);

impl AvxButterfly169f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows: [AvxStoreF; 13] = [AvxStoreF::zero(); 13];
        let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 169];
        unsafe {
            // columns
            for k in 0..3 {
                for i in 0..13 {
                    rows[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 13 + k * 4..));
                }

                rows = self.bf13.exec(rows);

                let q1 = AvxStoreF::mul_by_complex(rows[1], self.twiddles[12 * k]);
                let q2 = AvxStoreF::mul_by_complex(rows[2], self.twiddles[12 * k + 1]);
                let q3 = AvxStoreF::mul_by_complex(rows[3], self.twiddles[12 * k + 2]);
                let t = avx_transpose_f32x2_4x4_impl(rows[0].v, q1.v, q2.v, q3.v);
                AvxStoreF::raw(t.0).write_u(scratch.get_unchecked_mut(k * 4 * 13..));
                AvxStoreF::raw(t.1).write_u(scratch.get_unchecked_mut((k * 4 + 1) * 13..));
                AvxStoreF::raw(t.2).write_u(scratch.get_unchecked_mut((k * 4 + 2) * 13..));
                AvxStoreF::raw(t.3).write_u(scratch.get_unchecked_mut((k * 4 + 3) * 13..));

                for i in 1..3 {
                    let q0 = AvxStoreF::mul_by_complex(
                        rows[i * 4],
                        self.twiddles[(i - 1) * 4 + 3 + 12 * k],
                    );
                    let q1 = AvxStoreF::mul_by_complex(
                        rows[i * 4 + 1],
                        self.twiddles[(i - 1) * 4 + 4 + 12 * k],
                    );
                    let q2 = AvxStoreF::mul_by_complex(
                        rows[i * 4 + 2],
                        self.twiddles[(i - 1) * 4 + 5 + 12 * k],
                    );
                    let q3 = AvxStoreF::mul_by_complex(
                        rows[i * 4 + 3],
                        self.twiddles[(i - 1) * 4 + 6 + 12 * k],
                    );
                    let t = avx_transpose_f32x2_4x4_impl(q0.v, q1.v, q2.v, q3.v);
                    AvxStoreF::raw(t.0).write_u(scratch.get_unchecked_mut(k * 4 * 13 + i * 4..));
                    AvxStoreF::raw(t.1)
                        .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 13 + i * 4..));
                    AvxStoreF::raw(t.2)
                        .write_u(scratch.get_unchecked_mut((k * 4 + 2) * 13 + i * 4..));
                    AvxStoreF::raw(t.3)
                        .write_u(scratch.get_unchecked_mut((k * 4 + 3) * 13 + i * 4..));
                }

                {
                    let i = 3;
                    let q0 = AvxStoreF::mul_by_complex(
                        rows[i * 4],
                        self.twiddles[(i - 1) * 4 + 3 + 12 * k],
                    );
                    let t = avx_transpose_f32x2_4x4_impl(
                        q0.v,
                        _mm256_setzero_ps(),
                        _mm256_setzero_ps(),
                        _mm256_setzero_ps(),
                    );
                    AvxStoreF::raw(t.0).write_lo1u(scratch.get_unchecked_mut(k * 4 * 13 + i * 4..));
                    AvxStoreF::raw(t.1)
                        .write_lo1u(scratch.get_unchecked_mut((k * 4 + 1) * 13 + i * 4..));
                    AvxStoreF::raw(t.2)
                        .write_lo1u(scratch.get_unchecked_mut((k * 4 + 2) * 13 + i * 4..));
                    AvxStoreF::raw(t.3)
                        .write_lo1u(scratch.get_unchecked_mut((k * 4 + 3) * 13 + i * 4..));
                }
            }

            {
                let k = 3;
                for i in 0..13 {
                    rows[i] = AvxStoreF::from_complex(chunk.index(i * 13 + 12));
                }

                rows = self.bf13.exec(rows);

                let q1 = AvxStoreF::mul_by_complex(rows[1], self.twiddles[12 * k]);
                let q2 = AvxStoreF::mul_by_complex(rows[2], self.twiddles[12 * k + 1]);
                let q3 = AvxStoreF::mul_by_complex(rows[3], self.twiddles[12 * k + 2]);
                let t = avx_transpose_f32x2_4x4_impl(rows[0].v, q1.v, q2.v, q3.v);
                AvxStoreF::raw(t.0).write_u(scratch.get_unchecked_mut(k * 4 * 13..));

                for i in 1..3 {
                    let q0 = AvxStoreF::mul_by_complex(
                        rows[i * 4],
                        self.twiddles[(i - 1) * 4 + 3 + 12 * k],
                    );
                    let q1 = AvxStoreF::mul_by_complex(
                        rows[i * 4 + 1],
                        self.twiddles[(i - 1) * 4 + 4 + 12 * k],
                    );
                    let q2 = AvxStoreF::mul_by_complex(
                        rows[i * 4 + 2],
                        self.twiddles[(i - 1) * 4 + 5 + 12 * k],
                    );
                    let q3 = AvxStoreF::mul_by_complex(
                        rows[i * 4 + 3],
                        self.twiddles[(i - 1) * 4 + 6 + 12 * k],
                    );
                    let t = avx_transpose_f32x2_4x4_impl(q0.v, q1.v, q2.v, q3.v);
                    AvxStoreF::raw(t.0).write_u(scratch.get_unchecked_mut(k * 4 * 13 + i * 4..));
                }

                {
                    let i = 3;
                    let q0 = AvxStoreF::mul_by_complex(
                        rows[i * 4],
                        self.twiddles[(i - 1) * 4 + 3 + 12 * k],
                    );
                    let t = avx_transpose_f32x2_4x4_impl(
                        q0.v,
                        _mm256_setzero_ps(),
                        _mm256_setzero_ps(),
                        _mm256_setzero_ps(),
                    );
                    AvxStoreF::raw(t.0).write_lo1u(scratch.get_unchecked_mut(k * 4 * 13 + i * 4..));
                }
            }

            // rows

            for k in 0..3 {
                for i in 0..13 {
                    rows[i] = AvxStoreF::from_complex_refu(scratch.get_unchecked(i * 13 + k * 4..));
                }
                rows = self.bf13.exec(rows);
                for i in 0..13 {
                    rows[i].write(chunk.slice_from_mut(i * 13 + k * 4..));
                }
            }

            {
                for i in 0..13 {
                    rows[i] = AvxStoreF::from_complexu(scratch.get_unchecked(i * 13 + 12));
                }
                rows = self.bf13.exec(rows);
                for i in 0..13 {
                    rows[i].write_lo1(chunk.slice_from_mut(i * 13 + 12..));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly169, f32, AvxButterfly169f, 169, 1e-3);
    test_oof_avx_butterfly!(test_oof_avx_butterfly169, f32, AvxButterfly169f, 169, 1e-3);
}
