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
use crate::avx::mixed::{AvxStoreF, ColumnButterfly6f, ColumnButterfly7f};
use crate::avx::transpose::store_transpose_7x7_f32;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

pub(crate) struct AvxButterfly42f {
    direction: FftDirection,
    bf6: ColumnButterfly6f,
    bf7: ColumnButterfly7f,
    twiddles: [AvxStoreF; 12],
}

impl AvxButterfly42f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                direction: fft_direction,
                twiddles: gen_butterfly_twiddles_f32(7, 6, fft_direction, 42),
                bf7: ColumnButterfly7f::new(fft_direction),
                bf6: ColumnButterfly6f::new(fft_direction),
            }
        }
    }
}

impl FftExecutor<f32> for AvxButterfly42f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        42
    }
}

impl AvxButterfly42f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 42 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows1: [AvxStoreF; 6] = [AvxStoreF::zero(); 6];
            let mut rows2: [AvxStoreF; 6] = [AvxStoreF::zero(); 6];

            for chunk in in_place.chunks_exact_mut(42) {
                for i in 0..6 {
                    rows1[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 7..));
                }
                for i in 0..6 {
                    rows2[i] = AvxStoreF::from_complex3(chunk.get_unchecked(i * 7 + 4..));
                }

                rows1 = self.bf6.exec(rows1);
                rows2 = self.bf6.exec(rows2);

                for i in 1..6 {
                    rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1]);
                    rows2[i] = AvxStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 5]);
                }

                let (mut transposed0, mut transposed1) = store_transpose_7x7_f32(
                    [
                        rows1[0],
                        rows1[1],
                        rows1[2],
                        rows1[3],
                        rows1[4],
                        rows1[5],
                        AvxStoreF::zero(),
                    ],
                    [
                        rows2[0],
                        rows2[1],
                        rows2[2],
                        rows2[3],
                        rows2[4],
                        rows2[5],
                        AvxStoreF::zero(),
                    ],
                );

                transposed0 = self.bf7.exec(transposed0);
                transposed1 = self.bf7.exec(transposed1);

                transposed0[0].write(chunk);
                transposed0[1].write(chunk.get_unchecked_mut(6..));
                transposed0[2].write(chunk.get_unchecked_mut(12..));
                transposed0[3].write(chunk.get_unchecked_mut(18..));
                transposed0[4].write(chunk.get_unchecked_mut(24..));
                transposed0[5].write(chunk.get_unchecked_mut(30..));
                transposed0[6].write(chunk.get_unchecked_mut(36..));

                transposed1[0].write_lo2(chunk.get_unchecked_mut(4..));
                transposed1[1].write_lo2(chunk.get_unchecked_mut(10..));
                transposed1[2].write_lo2(chunk.get_unchecked_mut(16..));
                transposed1[3].write_lo2(chunk.get_unchecked_mut(22..));
                transposed1[4].write_lo2(chunk.get_unchecked_mut(28..));
                transposed1[5].write_lo2(chunk.get_unchecked_mut(34..));
                transposed1[6].write_lo2(chunk.get_unchecked_mut(40..));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly42, f32, AvxButterfly42f, 42, 1e-3);
}
