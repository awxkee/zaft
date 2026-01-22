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

use crate::avx::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::avx::mixed::{AvxStoreF, ColumnButterfly6f, ColumnButterfly12f};
use crate::avx::transpose::transpose_f32x2_4x4_aos;
use crate::avx::util::shuffle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn unpack_2(v0: AvxStoreF, v1: AvxStoreF) -> [AvxStoreF; 2] {
    [
        AvxStoreF::raw(_mm256_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(v0.v, v1.v)),
        AvxStoreF::raw(_mm256_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(v0.v, v1.v)),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_12x6_to_6x12_f32(
    rows0: [AvxStoreF; 6],
    rows1: [AvxStoreF; 6],
    rows2: [AvxStoreF; 6],
) -> ([AvxStoreF; 12], [AvxStoreF; 12]) {
    let [unpacked0, unpacked1] = unpack_2(rows0[0], rows0[1]);
    let [unpacked2, unpacked3] = unpack_2(rows1[0], rows1[1]);
    let [unpacked4, unpacked5] = unpack_2(rows2[0], rows2[1]);

    let output0 = [
        unpacked0.lo(),
        unpacked1.lo(),
        unpacked0.hi(),
        unpacked1.hi(),
        unpacked2.lo(),
        unpacked3.lo(),
        unpacked2.hi(),
        unpacked3.hi(),
        unpacked4.lo(),
        unpacked5.lo(),
        unpacked4.hi(),
        unpacked5.hi(),
    ];
    let transposed0 = transpose_f32x2_4x4_aos([rows0[2], rows0[3], rows0[4], rows0[5]]);
    let transposed1 = transpose_f32x2_4x4_aos([rows1[2], rows1[3], rows1[4], rows1[5]]);
    let transposed2 = transpose_f32x2_4x4_aos([rows2[2], rows2[3], rows2[4], rows2[5]]);

    let output1 = [
        transposed0[0],
        transposed0[1],
        transposed0[2],
        transposed0[3],
        transposed1[0],
        transposed1[1],
        transposed1[2],
        transposed1[3],
        transposed2[0],
        transposed2[1],
        transposed2[2],
        transposed2[3],
    ];

    (output0, output1)
}

pub(crate) struct AvxButterfly72f {
    direction: FftDirection,
    bf6: ColumnButterfly6f,
    bf12: ColumnButterfly12f,
    twiddles: [AvxStoreF; 21],
}

impl AvxButterfly72f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(12, 6, fft_direction, 72),
            bf6: ColumnButterfly6f::new(fft_direction),
            bf12: ColumnButterfly12f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly72f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        72
    }
}

impl AvxButterfly72f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(72) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(72) {
                let mut rows0 = [AvxStoreF::zero(); 6];
                for r in 0..6 {
                    rows0[r] = AvxStoreF::from_complex_ref(chunk.get_unchecked(12 * r..));
                }
                let mut mid0 = self.bf6.exec(rows0);
                for r in 1..6 {
                    mid0[r] = AvxStoreF::mul_by_complex(mid0[r], self.twiddles[r - 1]);
                }

                let mut rows1 = [AvxStoreF::zero(); 6];
                for r in 0..6 {
                    rows1[r] = AvxStoreF::from_complex_ref(chunk.get_unchecked(12 * r + 4..));
                }
                let mut mid1 = self.bf6.exec(rows1);
                for r in 1..6 {
                    mid1[r] = AvxStoreF::mul_by_complex(mid1[r], self.twiddles[r - 1 + 5]);
                }

                let mut rows2 = [AvxStoreF::zero(); 6];
                for r in 0..6 {
                    rows2[r] = AvxStoreF::from_complex_ref(chunk.get_unchecked(12 * r + 8..));
                }
                let mut mid2 = self.bf6.exec(rows2);
                for r in 1..6 {
                    mid2[r] = AvxStoreF::mul_by_complex(mid2[r], self.twiddles[r - 1 + 10]);
                }

                let (transposed0, transposed1) = transpose_12x6_to_6x12_f32(mid0, mid1, mid2);

                let output0 = self.bf12.exec(transposed0);
                for r in 0..12 {
                    output0[r].write_lo2(chunk.get_unchecked_mut(6 * r..));
                }

                let output1 = self.bf12.exec(transposed1);
                for r in 0..12 {
                    output1[r].write(chunk.get_unchecked_mut(6 * r + 2..));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly72, f32, AvxButterfly72f, 72, 1e-3);
}
