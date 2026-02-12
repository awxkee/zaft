// Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::avx::butterflies::shared::{
    boring_avx_butterfly, gen_butterfly_twiddles_f32, gen_butterfly_twiddles_f64,
};
use crate::avx::mixed::{
    AvxStoreD, AvxStoreF, ColumnButterfly3d, ColumnButterfly3f, ColumnButterfly4d,
    ColumnButterfly4f,
};
use crate::avx::transpose::{avx_transpose_f32x2_4x4_impl, transpose_f64x2_2x2};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x2_4x3(
    rows0: [AvxStoreD; 3],
    rows1: [AvxStoreD; 3],
) -> ([AvxStoreD; 4], [AvxStoreD; 4]) {
    let a0 = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);
    let d0 = transpose_f64x2_2x2(rows0[2].v, _mm256_setzero_pd());

    let b0 = transpose_f64x2_2x2(rows1[0].v, rows1[1].v);
    let e0 = transpose_f64x2_2x2(rows1[2].v, _mm256_setzero_pd());
    (
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
        ],
        [
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
            AvxStoreD::raw(e0.1),
        ],
    )
}

pub(crate) struct AvxButterfly12d {
    direction: FftDirection,
    bf3: ColumnButterfly3d,
    bf4: ColumnButterfly4d,
    twiddles: [AvxStoreD; 4],
}

impl AvxButterfly12d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(4, 3, fft_direction, 12),
            bf3: ColumnButterfly3d::new(fft_direction),
            bf4: ColumnButterfly4d::new(fft_direction),
        }
    }
}

boring_avx_butterfly!(AvxButterfly12d, f64, 12);

impl AvxButterfly12d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows0: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
        let mut rows1: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
        // columns
        for i in 0..3 {
            rows0[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 4..));
            rows1[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 4 + 2..));
        }

        rows0 = self.bf3.exec(rows0);
        rows1 = self.bf3.exec(rows1);

        for i in 1..3 {
            rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1]);
            rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 2]);
        }

        let transposed = transpose_f64x2_4x3(rows0, rows1);

        let q0 = self.bf4.exec(transposed.0);
        let q1 = self.bf4.exec(transposed.1);

        for i in 0..4 {
            q0[i].write(chunk.slice_from_mut(i * 3..));
            q1[i].write_lo(chunk.slice_from_mut(i * 3 + 2..));
        }
    }
}

pub(crate) struct AvxButterfly12f {
    direction: FftDirection,
    bf3: ColumnButterfly3f,
    bf4: ColumnButterfly4f,
    twiddles: [AvxStoreF; 2],
}

impl AvxButterfly12f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(4, 3, fft_direction, 12),
            bf3: ColumnButterfly3f::new(fft_direction),
            bf4: ColumnButterfly4f::new(fft_direction),
        }
    }
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) fn transpose_f32x2_4x3(rows0: [AvxStoreF; 3]) -> [AvxStoreF; 4] {
    let a0 = avx_transpose_f32x2_4x4_impl(rows0[0].v, rows0[1].v, rows0[2].v, _mm256_setzero_ps());

    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
    ]
}

boring_avx_butterfly!(AvxButterfly12f, f32, 12);

impl AvxButterfly12f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows0: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];
        // columns
        for i in 0..3 {
            rows0[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 4..));
        }

        rows0 = self.bf3.exec(rows0);

        rows0[1] = AvxStoreF::mul_by_complex(rows0[1], self.twiddles[0]);
        rows0[2] = AvxStoreF::mul_by_complex(rows0[2], self.twiddles[1]);

        let transposed = transpose_f32x2_4x3(rows0);

        let q0 = self.bf4.exec(transposed);

        for i in 0..4 {
            q0[i].write_lo3(chunk.slice_from_mut(i * 3..));
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly12, f32, AvxButterfly12f, 12, 1e-5);
    test_avx_butterfly!(test_avx_butterfly12_f64, f64, AvxButterfly12d, 12, 1e-7);

    test_oof_avx_butterfly!(test_avx_oof_butterfly12, f32, AvxButterfly12f, 12, 1e-5);
    test_oof_avx_butterfly!(test_avx_oof_butterfly12_f64, f64, AvxButterfly12d, 12, 1e-7);
}
