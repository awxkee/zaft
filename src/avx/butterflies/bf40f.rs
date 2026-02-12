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
use crate::avx::mixed::{AvxStoreF, ColumnButterfly5f, ColumnButterfly8f};
use crate::avx::transpose::transpose_f32x2_8x5;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

pub(crate) struct AvxButterfly40f {
    direction: FftDirection,
    bf5: ColumnButterfly5f,
    bf8: ColumnButterfly8f,
    twiddles: [AvxStoreF; 8],
}

impl AvxButterfly40f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                direction: fft_direction,
                twiddles: gen_butterfly_twiddles_f32(8, 5, fft_direction, 40),
                bf8: ColumnButterfly8f::new(fft_direction),
                bf5: ColumnButterfly5f::new(fft_direction),
            }
        }
    }
}

boring_avx_butterfly!(AvxButterfly40f, f32, 40);

impl AvxButterfly40f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows1: [AvxStoreF; 5] = [AvxStoreF::zero(); 5];
        let mut rows2: [AvxStoreF; 5] = [AvxStoreF::zero(); 5];
        for i in 0..5 {
            rows1[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 8..));
        }
        for i in 0..5 {
            rows2[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 8 + 4..));
        }

        rows1 = self.bf5.exec(rows1);
        rows2 = self.bf5.exec(rows2);

        for i in 1..5 {
            rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1]);
            rows2[i] = AvxStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 4]);
        }

        let (mut transposed0, mut transposed1) = transpose_f32x2_8x5(
            [rows1[0], rows1[1], rows1[2], rows1[3], rows1[4]],
            [rows2[0], rows2[1], rows2[2], rows2[3], rows2[4]],
        );

        transposed0 = self.bf8.exec(transposed0);

        transposed0[0].write(chunk.slice_from_mut(0..));
        transposed0[1].write(chunk.slice_from_mut(5..));
        transposed0[2].write(chunk.slice_from_mut(10..));
        transposed0[3].write(chunk.slice_from_mut(15..));
        transposed0[4].write(chunk.slice_from_mut(20..));
        transposed0[5].write(chunk.slice_from_mut(25..));
        transposed0[6].write(chunk.slice_from_mut(30..));
        transposed0[7].write(chunk.slice_from_mut(35..));

        transposed1 = self.bf8.exec(transposed1);

        transposed1[0].write_lo1(chunk.slice_from_mut(4..));
        transposed1[1].write_lo1(chunk.slice_from_mut(9..));
        transposed1[2].write_lo1(chunk.slice_from_mut(14..));
        transposed1[3].write_lo1(chunk.slice_from_mut(19..));
        transposed1[4].write_lo1(chunk.slice_from_mut(24..));
        transposed1[5].write_lo1(chunk.slice_from_mut(29..));
        transposed1[6].write_lo1(chunk.slice_from_mut(34..));
        transposed1[7].write_lo1(chunk.slice_from_mut(39..));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly40, f32, AvxButterfly40f, 40, 1e-3);
}
