/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
use crate::avx::mixed::{AvxStoreF, ColumnButterfly7f, ColumnButterfly9f};
use crate::avx::transpose::transpose_4x7;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly63f {
    direction: FftDirection,
    bf7: ColumnButterfly7f,
    bf9: ColumnButterfly9f,
    twiddles: [AvxStoreF; 18],
}

impl AvxButterfly63f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(9, 7, fft_direction, 63),
            bf9: ColumnButterfly9f::new(fft_direction),
            bf7: ColumnButterfly7f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly63f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        63
    }
}

impl AvxButterfly63f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 63 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [AvxStoreF; 7] = [AvxStoreF::zero(); 7];
            let mut rows9: [AvxStoreF; 9] = [AvxStoreF::zero(); 9];
            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 63];

            for chunk in in_place.chunks_exact_mut(63) {
                // columns
                for k in 0..2 {
                    for i in 0..7 {
                        rows[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 9 + k * 4..));
                    }

                    rows = self.bf7.exec(rows);

                    for i in 1..7 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 6 * k]);
                    }

                    let transposed = transpose_4x7(rows);

                    {
                        let i = 0;
                        transposed[i * 4].write_u(scratch.get_unchecked_mut(k * 4 * 7 + i * 4..));
                        transposed[i * 4 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 7 + i * 4..));
                        transposed[i * 4 + 2]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 2) * 7 + i * 4..));
                        transposed[i * 4 + 3]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 3) * 7 + i * 4..));
                    }
                    {
                        let i = 1;
                        transposed[i * 4]
                            .write_lo3u(scratch.get_unchecked_mut(k * 4 * 7 + i * 4..));
                        transposed[i * 4 + 1]
                            .write_lo3u(scratch.get_unchecked_mut((k * 4 + 1) * 7 + i * 4..));
                        transposed[i * 4 + 2]
                            .write_lo3u(scratch.get_unchecked_mut((k * 4 + 2) * 7 + i * 4..));
                        transposed[i * 4 + 3]
                            .write_lo3u(scratch.get_unchecked_mut((k * 4 + 3) * 7 + i * 4..));
                    }
                }

                {
                    let k = 2;
                    for i in 0..7 {
                        rows[i] = AvxStoreF::from_complex(chunk.get_unchecked(i * 9 + k * 4));
                    }

                    rows = self.bf7.exec(rows);

                    for i in 1..7 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 6 * k]);
                    }

                    let transposed = transpose_4x7(rows);

                    transposed[0].write_u(scratch.get_unchecked_mut(k * 4 * 7..));
                    transposed[4].write_lo3u(scratch.get_unchecked_mut(k * 4 * 7 + 4..));
                }

                // rows

                {
                    let k = 0;
                    for i in 0..9 {
                        rows9[i] =
                            AvxStoreF::from_complex_refu(scratch.get_unchecked(i * 7 + k * 4..));
                    }
                    rows9 = self.bf9.exec(rows9);
                    for i in 0..9 {
                        rows9[i].write(chunk.get_unchecked_mut(i * 7 + k * 4..));
                    }
                }
                {
                    let k = 1;
                    for i in 0..9 {
                        rows9[i] =
                            AvxStoreF::from_complex3u(scratch.get_unchecked(i * 7 + k * 4..));
                    }
                    rows9 = self.bf9.exec(rows9);
                    for i in 0..9 {
                        rows9[i].write_lo3(chunk.get_unchecked_mut(i * 7 + k * 4..));
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

    test_avx_butterfly!(test_avx_butterfly63, f32, AvxButterfly63f, 63, 1e-3);
}
