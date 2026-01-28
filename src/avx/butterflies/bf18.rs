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
use crate::avx::butterflies::shared::{
    boring_avx_butterfly, gen_butterfly_twiddles_f32, gen_butterfly_twiddles_f64,
};
use crate::avx::mixed::{
    AvxStoreD, AvxStoreF, ColumnButterfly2d, ColumnButterfly2f, ColumnButterfly9d,
    ColumnButterfly9f,
};
use crate::avx::transpose::{transpose_f32x2_4x2, transpose_f64x2_2x2};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

pub(crate) struct AvxButterfly18d {
    direction: FftDirection,
    bf2: ColumnButterfly2d,
    bf9: ColumnButterfly9d,
    twiddles: [AvxStoreD; 5],
}

impl AvxButterfly18d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(9, 2, fft_direction, 18),
            bf2: ColumnButterfly2d::new(fft_direction),
            bf9: ColumnButterfly9d::new(fft_direction),
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x2_9x2(
    rows0: [AvxStoreD; 2],
    rows1: [AvxStoreD; 2],
    rows2: [AvxStoreD; 2],
    rows3: [AvxStoreD; 2],
    rows4: [AvxStoreD; 2],
) -> [AvxStoreD; 9] {
    let a = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);
    let b = transpose_f64x2_2x2(rows1[0].v, rows1[1].v);
    let c = transpose_f64x2_2x2(rows2[0].v, rows2[1].v);
    let d = transpose_f64x2_2x2(rows3[0].v, rows3[1].v);
    let e = transpose_f64x2_2x2(rows4[0].v, rows4[1].v);
    [
        AvxStoreD::raw(a.0),
        AvxStoreD::raw(a.1),
        AvxStoreD::raw(b.0),
        AvxStoreD::raw(b.1),
        AvxStoreD::raw(c.0),
        AvxStoreD::raw(c.1),
        AvxStoreD::raw(d.0),
        AvxStoreD::raw(d.1),
        AvxStoreD::raw(e.0),
    ]
}

boring_avx_butterfly!(AvxButterfly18d, f64, 18);

impl AvxButterfly18d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows0: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
        let mut rows1: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
        let mut rows2: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
        let mut rows3: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
        let mut rows4: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
        // columns
        for i in 0..2 {
            rows0[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 9..));
            rows1[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 9 + 2..));
            rows2[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 9 + 4..));
            rows3[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 9 + 6..));
            rows4[i] = AvxStoreD::from_complex(chunk.index(i * 9 + 8));
        }

        rows0 = self.bf2.exec(rows0);
        rows1 = self.bf2.exec(rows1);
        rows2 = self.bf2.exec(rows2);
        rows3 = self.bf2.exec(rows3);
        rows4 = self.bf2.exec(rows4);

        rows0[1] = AvxStoreD::mul_by_complex(rows0[1], self.twiddles[0]);
        rows1[1] = AvxStoreD::mul_by_complex(rows1[1], self.twiddles[1]);
        rows2[1] = AvxStoreD::mul_by_complex(rows2[1], self.twiddles[2]);
        rows3[1] = AvxStoreD::mul_by_complex(rows3[1], self.twiddles[3]);
        rows4[1] = AvxStoreD::mul_by_complex(rows4[1], self.twiddles[4]);

        let t = transpose_f64x2_9x2(rows0, rows1, rows2, rows3, rows4);

        let q0 = self.bf9.exec(t);

        for i in 0..9 {
            q0[i].write(chunk.slice_from_mut(i * 2..));
        }
    }
}

pub(crate) struct AvxButterfly18f {
    direction: FftDirection,
    bf2: ColumnButterfly2f,
    bf9: ColumnButterfly9f,
    twiddles: [AvxStoreF; 3],
}

impl AvxButterfly18f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf9: ColumnButterfly9f::new(fft_direction),
            bf2: ColumnButterfly2f::new(fft_direction),
            twiddles: gen_butterfly_twiddles_f32(9, 2, fft_direction, 18),
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f32x2_9x2(
    rows0: [AvxStoreF; 2],
    rows1: [AvxStoreF; 2],
    rows2: [AvxStoreF; 2],
) -> [AvxStoreF; 9] {
    let a = transpose_f32x2_4x2(rows0[0].v, rows0[1].v);
    let b = transpose_f32x2_4x2(rows1[0].v, rows1[1].v);
    let c = transpose_f32x2_4x2(rows2[0].v, rows2[1].v);
    [
        AvxStoreF::raw(a.0).lo(),
        AvxStoreF::raw(a.0).hi(),
        AvxStoreF::raw(a.1).lo(),
        AvxStoreF::raw(a.1).hi(),
        AvxStoreF::raw(b.0).lo(),
        AvxStoreF::raw(b.0).hi(),
        AvxStoreF::raw(b.1).lo(),
        AvxStoreF::raw(b.1).hi(),
        AvxStoreF::raw(c.0),
    ]
}

boring_avx_butterfly!(AvxButterfly18f, f32, 18);

impl AvxButterfly18f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows0: [AvxStoreF; 2] = [AvxStoreF::zero(); 2];
        let mut rows1: [AvxStoreF; 2] = [AvxStoreF::zero(); 2];
        let mut rows2: [AvxStoreF; 2] = [AvxStoreF::zero(); 2];
        // columns
        for i in 0..2 {
            rows0[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 9..));
            rows1[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 9 + 4..));
            rows2[i] = AvxStoreF::from_complex(chunk.index(i * 9 + 8));
        }

        rows0 = self.bf2.exec(rows0);
        rows1 = self.bf2.exec(rows1);
        rows2 = self.bf2.exec(rows2);

        rows0[1] = AvxStoreF::mul_by_complex(rows0[1], self.twiddles[0]);
        rows1[1] = AvxStoreF::mul_by_complex(rows1[1], self.twiddles[1]);
        rows2[1] = AvxStoreF::mul_by_complex(rows2[1], self.twiddles[2]);

        let t = transpose_f32x2_9x2(rows0, rows1, rows2);

        let q0 = self.bf9.exec(t);

        for i in 0..9 {
            q0[i].write_lo2(chunk.slice_from_mut(i * 2..));
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly18, f32, AvxButterfly18f, 18, 1e-5);
    test_avx_butterfly!(test_avx_butterfly18_f64, f64, AvxButterfly18d, 18, 1e-7);
}
