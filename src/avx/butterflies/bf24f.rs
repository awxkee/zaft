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
use crate::avx::mixed::{AvxStoreF, ColumnButterfly3f, ColumnButterfly8f};
use crate::avx::transpose::avx_transpose_f32x2_4x4_impl;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::_mm256_setzero_ps;

pub(crate) struct AvxButterfly24f {
    direction: FftDirection,
    bf3: ColumnButterfly3f,
    bf8: ColumnButterfly8f,
    twiddles: [AvxStoreF; 4],
}

impl AvxButterfly24f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(8, 3, fft_direction, 24),
            bf3: ColumnButterfly3f::new(fft_direction),
            bf8: ColumnButterfly8f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly24f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        24
    }
}

#[inline]
#[target_feature(enable = "avx")]
fn transpose_8x3(rows0: [AvxStoreF; 3], rows1: [AvxStoreF; 3]) -> [AvxStoreF; 8] {
    let a0 = avx_transpose_f32x2_4x4_impl(rows0[0].v, rows0[1].v, rows0[2].v, _mm256_setzero_ps());
    let b0 = avx_transpose_f32x2_4x4_impl(rows1[0].v, rows1[1].v, rows1[2].v, _mm256_setzero_ps());
    [
        // row 0
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
        AvxStoreF::raw(b0.0),
        AvxStoreF::raw(b0.1),
        AvxStoreF::raw(b0.2),
        AvxStoreF::raw(b0.3),
    ]
}

impl AvxButterfly24f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 24 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];
            let mut rows1: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];

            for chunk in in_place.chunks_exact_mut(24) {
                // columns
                {
                    for i in 0..3 {
                        rows0[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 8..));
                        rows1[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 8 + 4..));
                    }

                    rows0 = self.bf3.exec(rows0);
                    rows1 = self.bf3.exec(rows1);

                    for i in 1..3 {
                        rows0[i] = AvxStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                        rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1 + 2]);
                    }

                    let transposed = transpose_8x3(rows0, rows1);

                    let q0 = self.bf8.exec(transposed);

                    for i in 0..8 {
                        q0[i].write_lo3(chunk.get_unchecked_mut(i * 3..));
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

    test_avx_butterfly!(test_avx_butterfly24, f32, AvxButterfly24f, 24, 1e-3);
}
