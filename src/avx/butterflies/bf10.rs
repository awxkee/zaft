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

use crate::avx::butterflies::shared::{
    boring_avx_butterfly, gen_butterfly_twiddles_f32, gen_butterfly_twiddles_f64,
};
use crate::avx::mixed::{
    AvxStoreD, AvxStoreF, ColumnButterfly2d, ColumnButterfly2f, ColumnButterfly5d,
    ColumnButterfly5f,
};
use crate::avx::transpose::{avx_transpose_f32x2_4x4_impl, transpose_f64x2_2x2};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x2_5x2(
    rows0: [AvxStoreD; 2],
    rows1: [AvxStoreD; 2],
    rows2: [AvxStoreD; 2],
) -> [AvxStoreD; 5] {
    let a0 = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);

    let b0 = transpose_f64x2_2x2(rows1[0].v, rows1[1].v);

    let c0 = transpose_f64x2_2x2(rows2[0].v, rows2[1].v);
    [
        AvxStoreD::raw(a0.0),
        AvxStoreD::raw(a0.1),
        AvxStoreD::raw(b0.0),
        AvxStoreD::raw(b0.1),
        AvxStoreD::raw(c0.0),
    ]
}

pub(crate) struct AvxButterfly10d {
    direction: FftDirection,
    bf2: ColumnButterfly2d,
    bf5: ColumnButterfly5d,
    twiddles: [AvxStoreD; 4],
}

impl AvxButterfly10d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(5, 2, fft_direction, 10),
            bf2: ColumnButterfly2d::new(fft_direction),
            bf5: ColumnButterfly5d::new(fft_direction),
        }
    }
}

boring_avx_butterfly!(AvxButterfly10d, f64, 10);

impl AvxButterfly10d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows0: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
        let mut rows1: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
        let mut rows2: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
        // columns
        for i in 0..2 {
            rows0[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 5..));
            rows1[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 5 + 2..));
            rows2[i] = AvxStoreD::from_complex(chunk.index(i * 5 + 4));
        }

        rows0 = self.bf2.exec(rows0);
        rows1 = self.bf2.exec(rows1);
        rows2 = self.bf2.exec(rows2);

        rows0[1] = AvxStoreD::mul_by_complex(rows0[1], self.twiddles[0]);
        rows1[1] = AvxStoreD::mul_by_complex(rows1[1], self.twiddles[1]);
        rows2[1] = AvxStoreD::mul_by_complex(rows2[1], self.twiddles[2]);

        let transposed = transpose_f64x2_5x2(rows0, rows1, rows2);

        let q0 = self.bf5.exec(transposed);
        // rows
        for i in 0..5 {
            q0[i].write(chunk.slice_from_mut(i * 2..));
        }
    }
}

pub(crate) struct AvxButterfly10f {
    direction: FftDirection,
    bf2: ColumnButterfly2f,
    bf5: ColumnButterfly5f,
    twiddles: [AvxStoreF; 2],
}

impl AvxButterfly10f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(5, 2, fft_direction, 10),
            bf2: ColumnButterfly2f::new(fft_direction),
            bf5: ColumnButterfly5f::new(fft_direction),
        }
    }
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) fn transpose_f32x2_5x2(rows0: [AvxStoreF; 2], rows1: [AvxStoreF; 2]) -> [AvxStoreF; 5] {
    let a0 = avx_transpose_f32x2_4x4_impl(
        rows0[0].v,
        rows0[1].v,
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let b0 = avx_transpose_f32x2_4x4_impl(
        rows1[0].v,
        rows1[1].v,
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );

    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
        AvxStoreF::raw(b0.0),
    ]
}

boring_avx_butterfly!(AvxButterfly10f, f32, 10);

impl AvxButterfly10f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows0: [AvxStoreF; 2] = [AvxStoreF::zero(); 2];
        let mut rows1: [AvxStoreF; 2] = [AvxStoreF::zero(); 2];
        // columns
        for i in 0..2 {
            rows0[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 5..));
            rows1[i] = AvxStoreF::from_complex(chunk.index(i * 5 + 4));
        }

        rows0 = self.bf2.exec(rows0);
        rows1 = self.bf2.exec(rows1);

        rows0[1] = AvxStoreF::mul_by_complex(rows0[1], self.twiddles[0]);
        rows1[1] = AvxStoreF::mul_by_complex(rows1[1], self.twiddles[1]);

        let transposed = transpose_f32x2_5x2(rows0, rows1);

        let q0 = self.bf5.exec(transposed);

        q0[0].write2lo(q0[1], chunk.slice_from_mut(0..));
        q0[2].write2lo(q0[3], chunk.slice_from_mut(4..));
        q0[4].write_lo2(chunk.slice_from_mut(8..));
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly10, f32, AvxButterfly10f, 10, 1e-5);
    test_avx_butterfly!(test_avx_butterfly10_f64, f64, AvxButterfly10d, 10, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly10, f32, AvxButterfly10f, 10, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly10_f64, f64, AvxButterfly10d, 10, 1e-7);
}
