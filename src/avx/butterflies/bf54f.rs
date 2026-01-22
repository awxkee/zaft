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
use crate::avx::mixed::{AvxStoreF, ColumnButterfly6f, ColumnButterfly9f};
use crate::avx::transpose::{transpose_4x2, transpose_f32x2_4x4_aos};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f32x2_9x6(
    rows1: [AvxStoreF; 6],
    rows2: [AvxStoreF; 6],
    rows3: [AvxStoreF; 6],
) -> ([AvxStoreF; 9], [AvxStoreF; 9]) {
    let tl = transpose_f32x2_4x4_aos([rows1[0], rows1[1], rows1[2], rows1[3]]);

    let bl = transpose_4x2([rows1[4], rows1[5]]);

    let tr = transpose_f32x2_4x4_aos([rows2[0], rows2[1], rows2[2], rows2[3]]);

    let br = transpose_4x2([rows2[4], rows2[5]]);

    let far_top = transpose_f32x2_4x4_aos([rows3[0], rows3[1], rows3[2], rows3[3]]);

    let far_bottom = transpose_4x2([rows3[4], rows3[5]]);

    (
        [
            tl[0], tl[1], tl[2], tl[3], tr[0], tr[1], tr[2], tr[3], far_top[0],
        ],
        [
            bl[0],
            bl[1],
            bl[2],
            bl[3],
            br[0],
            br[1],
            br[2],
            br[3],
            far_bottom[0],
        ],
    )
}

pub(crate) struct AvxButterfly54f {
    direction: FftDirection,
    bf6: ColumnButterfly6f,
    bf9: ColumnButterfly9f,
    twiddles: [AvxStoreF; 15],
}

impl AvxButterfly54f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(9, 6, fft_direction, 54),
            bf9: ColumnButterfly9f::new(fft_direction),
            bf6: ColumnButterfly6f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly54f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        54
    }
}

impl AvxButterfly54f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(54) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 6] = [AvxStoreF::zero(); 6];
            let mut rows1: [AvxStoreF; 6] = [AvxStoreF::zero(); 6];
            let mut rows2: [AvxStoreF; 6] = [AvxStoreF::zero(); 6];

            for chunk in in_place.chunks_exact_mut(54) {
                // columns
                // 0-4
                for i in 0..6 {
                    rows0[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 9..));
                }

                rows0 = self.bf6.exec(rows0);

                for i in 1..6 {
                    rows0[i] = AvxStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                }

                // 4-8

                for i in 0..6 {
                    rows1[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 9 + 4..));
                }

                rows1 = self.bf6.exec(rows1);

                for i in 1..6 {
                    rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1 + 5]);
                }

                // 8-9

                for i in 0..6 {
                    rows2[i] = AvxStoreF::from_complex(chunk.get_unchecked(i * 9 + 8));
                }

                rows2 = self.bf6.exec(rows2);

                for i in 1..6 {
                    rows2[i] = AvxStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 10]);
                }

                let (mut t0, mut t1) = transpose_f32x2_9x6(rows0, rows1, rows2);

                // rows

                // 0-4
                t0 = self.bf9.exec(t0);

                for i in 0..9 {
                    t0[i].write(chunk.get_unchecked_mut(i * 6..));
                }

                // 4-6

                t1 = self.bf9.exec(t1);

                for i in 0..9 {
                    t1[i].write_lo2(chunk.get_unchecked_mut(i * 6 + 4..));
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

    test_avx_butterfly!(test_avx_butterfly54, f32, AvxButterfly54f, 54, 1e-3);
}
