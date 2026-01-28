/*
 * // Copyright (c) Radzivon Bartoshyk 01/2026. All rights reserved.
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
use crate::avx::mixed::{AvxStoreD, ColumnButterfly12d, ColumnButterfly18d};
use crate::avx::transpose::transpose_f64x2_2x12;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly216d {
    direction: FftDirection,
    bf18: ColumnButterfly18d,
    bf12: ColumnButterfly12d,
    twiddles: [AvxStoreD; 99],
}

impl AvxButterfly216d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: unsafe { gen_butterfly_twiddles_f64(18, 12, fft_direction, 216) },
            bf18: unsafe { ColumnButterfly18d::new(fft_direction) },
            bf12: unsafe { ColumnButterfly12d::new(fft_direction) },
        }
    }
}

boring_avx_butterfly!(AvxButterfly216d, f64, 216);

impl AvxButterfly216d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows12: [AvxStoreD; 12] = [AvxStoreD::zero(); 12];
        let mut rows18: [AvxStoreD; 18] = [AvxStoreD::zero(); 18];
        let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 216];
        unsafe {
            // columns
            for k in 0..9 {
                for i in 0..12 {
                    rows12[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 18 + k * 2..));
                }

                rows12 = self.bf12.exec(rows12);

                for i in 1..12 {
                    rows12[i] = AvxStoreD::mul_by_complex(rows12[i], self.twiddles[i - 1 + 11 * k]);
                }

                let transposed = transpose_f64x2_2x12(rows12);

                for i in 0..6 {
                    transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 12 + i * 2..));
                    transposed[i * 2 + 1]
                        .write_u(scratch.get_unchecked_mut((k * 2 + 1) * 12 + i * 2..));
                }
            }

            // rows

            for k in 0..6 {
                for i in 0..18 {
                    rows18[i] =
                        AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 12 + k * 2..));
                }
                rows18 = self.bf18.exec(rows18);
                for i in 0..18 {
                    rows18[i].write(chunk.slice_from_mut(i * 12 + k * 2..));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly216, f64, AvxButterfly216d, 216, 1e-3);
    test_oof_avx_butterfly!(test_avx_butterfly216_oof, f64, AvxButterfly216d, 216, 1e-3);
}
