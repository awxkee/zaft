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
use crate::avx::mixed::{AvxStoreF, ColumnButterfly6f, ColumnButterfly13f};
use crate::avx::transpose::transpose_4x6;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly78f {
    direction: FftDirection,
    bf6: ColumnButterfly6f,
    bf13: ColumnButterfly13f,
    twiddles: [AvxStoreF; 21],
}

impl AvxButterfly78f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(13, 6, fft_direction, 78),
            bf6: ColumnButterfly6f::new(fft_direction),
            bf13: ColumnButterfly13f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly78f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        78
    }
}

impl AvxButterfly78f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(78) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [AvxStoreF; 6] = [AvxStoreF::zero(); 6];
            let mut rows13: [AvxStoreF; 13] = [AvxStoreF::zero(); 13];
            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 78];

            for chunk in in_place.chunks_exact_mut(78) {
                // columns
                for k in 0..3 {
                    for i in 0..6 {
                        rows[i] =
                            AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 13 + k * 4..));
                    }

                    rows = self.bf6.exec(rows);

                    for i in 1..6 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 5 * k]);
                    }

                    let transposed = transpose_4x6(rows);

                    {
                        let i = 0;
                        transposed[i * 4].write_u(scratch.get_unchecked_mut(k * 4 * 6 + i * 4..));
                        transposed[i * 4 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 6 + i * 4..));
                        transposed[i * 4 + 2]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 2) * 6 + i * 4..));
                        transposed[i * 4 + 3]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 3) * 6 + i * 4..));
                    }
                    {
                        let i = 1;
                        transposed[i * 4]
                            .write_lo2u(scratch.get_unchecked_mut(k * 4 * 6 + i * 4..));
                        transposed[i * 4 + 1]
                            .write_lo2u(scratch.get_unchecked_mut((k * 4 + 1) * 6 + i * 4..));
                        transposed[i * 4 + 2]
                            .write_lo2u(scratch.get_unchecked_mut((k * 4 + 2) * 6 + i * 4..));
                        transposed[i * 4 + 3]
                            .write_lo2u(scratch.get_unchecked_mut((k * 4 + 3) * 6 + i * 4..));
                    }
                }

                {
                    let k = 3;
                    for i in 0..6 {
                        rows[i] = AvxStoreF::from_complex(chunk.get_unchecked(i * 13 + k * 4));
                    }

                    rows = self.bf6.exec(rows);

                    for i in 1..6 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 5 * k]);
                    }

                    let transposed = transpose_4x6(rows);

                    transposed[0].write_u(scratch.get_unchecked_mut(k * 4 * 6..));
                    transposed[4].write_lo2u(scratch.get_unchecked_mut(k * 4 * 6 + 4..));
                }

                // rows

                {
                    let k = 0;
                    for i in 0..13 {
                        rows13[i] =
                            AvxStoreF::from_complex_refu(scratch.get_unchecked(i * 6 + k * 4..));
                    }
                    rows13 = self.bf13.exec(rows13);
                    for i in 0..13 {
                        rows13[i].write(chunk.get_unchecked_mut(i * 6 + k * 4..));
                    }
                }

                {
                    let k = 1;
                    for i in 0..13 {
                        rows13[i] =
                            AvxStoreF::from_complex2u(scratch.get_unchecked(i * 6 + k * 4..));
                    }
                    rows13 = self.bf13.exec(rows13);
                    for i in 0..13 {
                        rows13[i].write_lo2(chunk.get_unchecked_mut(i * 6 + k * 4..));
                    }
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

    test_avx_butterfly!(test_avx_butterfly78, f32, AvxButterfly78f, 78, 1e-3);
}
