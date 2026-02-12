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

use crate::avx::butterflies::shared::{
    boring_avx_butterfly, gen_butterfly_twiddles_f32, gen_butterfly_twiddles_f64,
};
use crate::avx::mixed::{AvxStoreD, AvxStoreF, ColumnButterfly5d, ColumnButterfly5f};
use crate::avx::transpose::{transpose_5x5_f32, transpose_5x5_f64};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

pub(crate) struct AvxButterfly25d {
    direction: FftDirection,
    bf5: ColumnButterfly5d,
    twiddles: [AvxStoreD; 12],
}

impl AvxButterfly25d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                direction: fft_direction,
                twiddles: gen_butterfly_twiddles_f64(5, 5, fft_direction, 25),
                bf5: ColumnButterfly5d::new(fft_direction),
            }
        }
    }
}

boring_avx_butterfly!(AvxButterfly25d, f64, 25);

impl AvxButterfly25d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows1: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];
        let mut rows2: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];
        let mut rows3: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];
        for i in 0..5 {
            rows1[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 5..));
        }
        for i in 0..5 {
            rows2[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 5 + 2..));
        }

        rows1 = self.bf5.exec(rows1);
        rows2 = self.bf5.exec(rows2);

        for i in 1..5 {
            rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1]);
            rows2[i] = AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 4]);
        }

        for i in 0..5 {
            rows3[i] = AvxStoreD::from_complex(chunk.index(i * 5 + 4));
        }

        rows3 = self.bf5.exec(rows3);

        for i in 1..5 {
            rows3[i] = AvxStoreD::mul_by_complex(rows3[i], self.twiddles[i - 1 + 8]);
        }

        let (mut transposed0, mut transposed1, mut transposed2) =
            transpose_5x5_f64(rows1, rows2, rows3);

        transposed0 = self.bf5.exec(transposed0);
        transposed1 = self.bf5.exec(transposed1);

        transposed0[0].write(chunk.slice_from_mut(0..));
        transposed0[1].write(chunk.slice_from_mut(5..));
        transposed0[2].write(chunk.slice_from_mut(10..));
        transposed0[3].write(chunk.slice_from_mut(15..));
        transposed0[4].write(chunk.slice_from_mut(20..));

        transposed2 = self.bf5.exec(transposed2);

        transposed1[0].write(chunk.slice_from_mut(2..));
        transposed1[1].write(chunk.slice_from_mut(7..));
        transposed1[2].write(chunk.slice_from_mut(12..));
        transposed1[3].write(chunk.slice_from_mut(17..));
        transposed1[4].write(chunk.slice_from_mut(22..));

        transposed2[0].write_lo(chunk.slice_from_mut(4..));
        transposed2[1].write_lo(chunk.slice_from_mut(9..));
        transposed2[2].write_lo(chunk.slice_from_mut(14..));
        transposed2[3].write_lo(chunk.slice_from_mut(19..));
        transposed2[4].write_lo(chunk.slice_from_mut(24..));
    }
}

pub(crate) struct AvxButterfly25f {
    direction: FftDirection,
    bf5: ColumnButterfly5f,
    twiddles: [AvxStoreF; 8],
}

impl AvxButterfly25f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                direction: fft_direction,
                twiddles: gen_butterfly_twiddles_f32(5, 5, fft_direction, 25),
                bf5: ColumnButterfly5f::new(fft_direction),
            }
        }
    }
}

boring_avx_butterfly!(AvxButterfly25f, f32, 25);

impl AvxButterfly25f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows1: [AvxStoreF; 5] = [AvxStoreF::zero(); 5];
        let mut rows2: [AvxStoreF; 5] = [AvxStoreF::zero(); 5];
        for i in 0..5 {
            rows1[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 5..));
        }
        for i in 0..5 {
            rows2[i] = AvxStoreF::from_complex(chunk.index(i * 5 + 4));
        }

        rows1 = self.bf5.exec(rows1);
        rows2 = self.bf5.exec(rows2);

        for i in 1..5 {
            rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1]);
            rows2[i] = AvxStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 4]);
        }

        let (mut transposed0, mut transposed1) = transpose_5x5_f32(rows1, rows2);

        transposed0 = self.bf5.exec(transposed0);
        transposed1 = self.bf5.exec(transposed1);

        transposed0[0].write(chunk.slice_from_mut(0..));
        transposed0[1].write(chunk.slice_from_mut(5..));
        transposed0[2].write(chunk.slice_from_mut(10..));
        transposed0[3].write(chunk.slice_from_mut(15..));
        transposed0[4].write(chunk.slice_from_mut(20..));

        transposed1[0].write_lo1(chunk.slice_from_mut(4..));
        transposed1[1].write_lo1(chunk.slice_from_mut(9..));
        transposed1[2].write_lo1(chunk.slice_from_mut(14..));
        transposed1[3].write_lo1(chunk.slice_from_mut(19..));
        transposed1[4].write_lo1(chunk.slice_from_mut(24..));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly25, f32, AvxButterfly25f, 25, 1e-5);
    test_avx_butterfly!(test_avx_butterfly25_f64, f64, AvxButterfly25d, 25, 1e-7);
    test_oof_avx_butterfly!(test_oof_avx_butterfly25, f32, AvxButterfly25f, 25, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly25_f64, f64, AvxButterfly25d, 25, 1e-9);
}
