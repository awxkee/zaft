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
use crate::avx::mixed::{AvxStoreF, ColumnButterfly9f, ColumnButterfly16f};
use crate::avx::transpose::transpose_4x9;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly144f {
    direction: FftDirection,
    bf16: ColumnButterfly16f,
    bf9: ColumnButterfly9f,
    twiddles: [AvxStoreF; 32],
}

impl AvxButterfly144f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(16, 9, fft_direction, 144),
            bf16: ColumnButterfly16f::new(fft_direction),
            bf9: ColumnButterfly9f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly144f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        144
    }
}

impl AvxButterfly144f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(144) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows16: [AvxStoreF; 16] = [AvxStoreF::zero(); 16];
            let mut rows: [AvxStoreF; 9] = [AvxStoreF::zero(); 9];
            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 144];

            for chunk in in_place.chunks_exact_mut(144) {
                // columns
                for k in 0..4 {
                    for i in 0..9 {
                        rows[i] =
                            AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 16 + k * 4..));
                    }

                    rows = self.bf9.exec(rows);

                    for i in 1..9 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 8 * k]);
                    }

                    let transposed = transpose_4x9(rows);

                    for i in 0..2 {
                        transposed[i * 4].write_u(scratch.get_unchecked_mut(k * 4 * 9 + i * 4..));
                        transposed[i * 4 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 9 + i * 4..));
                        transposed[i * 4 + 2]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 2) * 9 + i * 4..));
                        transposed[i * 4 + 3]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 3) * 9 + i * 4..));
                    }

                    transposed[8].write_lo1u(scratch.get_unchecked_mut(k * 4 * 9 + 8..));
                    transposed[9].write_lo1u(scratch.get_unchecked_mut((k * 4 + 1) * 9 + 8..));
                    transposed[10].write_lo1u(scratch.get_unchecked_mut((k * 4 + 2) * 9 + 8..));
                    transposed[11].write_lo1u(scratch.get_unchecked_mut((k * 4 + 3) * 9 + 8..));
                }

                // rows

                for k in 0..2 {
                    for i in 0..16 {
                        rows16[i] =
                            AvxStoreF::from_complex_refu(scratch.get_unchecked(i * 9 + k * 4..));
                    }
                    rows16 = self.bf16.exec(rows16);
                    for i in 0..16 {
                        rows16[i].write(chunk.get_unchecked_mut(i * 9 + k * 4..));
                    }
                }

                {
                    let k = 2;
                    for i in 0..16 {
                        rows16[i] = AvxStoreF::from_complexu(scratch.get_unchecked(i * 9 + k * 4));
                    }
                    rows16 = self.bf16.exec(rows16);
                    for i in 0..16 {
                        rows16[i].write_lo1(chunk.get_unchecked_mut(i * 9 + k * 4..));
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

    test_avx_butterfly!(test_avx_butterfly128, f32, AvxButterfly144f, 144, 1e-3);
}
