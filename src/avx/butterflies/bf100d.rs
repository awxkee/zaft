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

use crate::avx::butterflies::shared::{boring_avx_butterfly, gen_butterfly_twiddles_f64};
use crate::avx::mixed::{AvxStoreD, ColumnButterfly10d};
use crate::avx::transpose::transpose_f64x2_2x2;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly100d {
    direction: FftDirection,
    bf10: ColumnButterfly10d,
    twiddles: [AvxStoreD; 45],
}

impl AvxButterfly100d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(10, 10, fft_direction, 100),
            bf10: ColumnButterfly10d::new(fft_direction),
        }
    }
}

boring_avx_butterfly!(AvxButterfly100d, f64, 100);

impl AvxButterfly100d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows: [AvxStoreD; 10] = [AvxStoreD::zero(); 10];
        let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 100];
        unsafe {
            // columns
            for k in 0..5 {
                for i in 0..10 {
                    rows[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 10 + k * 2..));
                }

                rows = self.bf10.exec(rows);

                let q1 = AvxStoreD::mul_by_complex(rows[1], self.twiddles[9 * k]);
                let t = transpose_f64x2_2x2(rows[0].v, q1.v);
                AvxStoreD::raw(t.0).write_u(scratch.get_unchecked_mut(k * 2 * 10..));
                AvxStoreD::raw(t.1).write_u(scratch.get_unchecked_mut((k * 2 + 1) * 10..));

                for i in 1..5 {
                    let q0 = AvxStoreD::mul_by_complex(
                        rows[i * 2],
                        self.twiddles[(i - 1) * 2 + 1 + 9 * k],
                    );
                    let q1 = AvxStoreD::mul_by_complex(
                        rows[i * 2 + 1],
                        self.twiddles[(i - 1) * 2 + 2 + 9 * k],
                    );
                    let t = transpose_f64x2_2x2(q0.v, q1.v);
                    AvxStoreD::raw(t.0).write_u(scratch.get_unchecked_mut(k * 2 * 10 + i * 2..));
                    AvxStoreD::raw(t.1)
                        .write_u(scratch.get_unchecked_mut((k * 2 + 1) * 10 + i * 2..));
                }
            }

            // rows

            for k in 0..5 {
                for i in 0..10 {
                    rows[i] = AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 10 + k * 2..));
                }
                rows = self.bf10.exec(rows);
                for i in 0..10 {
                    rows[i].write(chunk.slice_from_mut(i * 10 + k * 2..));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly100_f64, f64, AvxButterfly100d, 100, 1e-7);
    test_oof_avx_butterfly!(
        test_oof_avx_butterfly100_f64,
        f64,
        AvxButterfly100d,
        100,
        1e-7
    );
}
