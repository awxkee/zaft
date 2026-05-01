/*
 * // Copyright (c) Radzivon Bartoshyk 01/2026. All rights reserved.
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
use crate::avx::mixed::{AvxStoreF, ColumnButterfly12f, ColumnButterfly18f};
use crate::avx::transpose::transpose_4x12;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly216f {
    direction: FftDirection,
    bf18: ColumnButterfly18f,
    bf12: ColumnButterfly12f,
    twiddles: [AvxStoreF; 55],
}

impl AvxButterfly216f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2,fma")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(18, 12, fft_direction, 216),
            bf18: ColumnButterfly18f::new(fft_direction),
            bf12: ColumnButterfly12f::new(fft_direction),
        }
    }
}

impl AvxButterfly216f {
    #[target_feature(enable = "avx2,fma")]
    fn exec_bf12(&self, src: &[Complex<f32>], dst: &mut [MaybeUninit<Complex<f32>>; 216]) {
        let mut rows12: [AvxStoreF; 12] = [AvxStoreF::zero(); 12];
        unsafe {
            // columns
            for k in 0..4 {
                for i in 0..12 {
                    rows12[i] = AvxStoreF::from_complex_ref(src.get_unchecked(i * 18 + k * 4..));
                }

                rows12 = self.bf12.exec(rows12);

                for i in 1..12 {
                    rows12[i] = AvxStoreF::mul_by_complex(rows12[i], self.twiddles[i - 1 + 11 * k]);
                }

                let transposed = transpose_4x12(rows12);

                for i in 0..3 {
                    transposed[i * 4].write_u(dst.get_unchecked_mut(k * 4 * 12 + i * 4..));
                    transposed[i * 4 + 1]
                        .write_u(dst.get_unchecked_mut((k * 4 + 1) * 12 + i * 4..));
                    transposed[i * 4 + 2]
                        .write_u(dst.get_unchecked_mut((k * 4 + 2) * 12 + i * 4..));
                    transposed[i * 4 + 3]
                        .write_u(dst.get_unchecked_mut((k * 4 + 3) * 12 + i * 4..));
                }
            }

            {
                let k = 4;
                for i in 0..12 {
                    rows12[i] = AvxStoreF::from_complex2(src.get_unchecked(i * 18 + k * 4..));
                }

                rows12 = self.bf12.exec(rows12);

                for i in 1..12 {
                    rows12[i] = AvxStoreF::mul_by_complex(rows12[i], self.twiddles[i - 1 + 11 * k]);
                }

                let transposed = transpose_4x12(rows12);

                for i in 0..3 {
                    transposed[i * 4].write_u(dst.get_unchecked_mut(k * 4 * 12 + i * 4..));
                    transposed[i * 4 + 1]
                        .write_u(dst.get_unchecked_mut((k * 4 + 1) * 12 + i * 4..));
                }
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn exec_bf18(&self, src: &[MaybeUninit<Complex<f32>>; 216], dst: &mut [Complex<f32>]) {
        let mut rows18: [AvxStoreF; 18] = [AvxStoreF::zero(); 18];
        for k in 0..3 {
            for i in 0..18 {
                unsafe {
                    rows18[i] = AvxStoreF::from_complex_refu(src.get_unchecked(i * 12 + k * 4..));
                }
            }
            rows18 = self.bf18.exec(rows18);
            for i in 0..18 {
                unsafe {
                    rows18[i].write(dst.get_unchecked_mut(i * 12 + k * 4..));
                }
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 216];
        self.exec_bf12(chunk.slice_from(0..), &mut scratch);
        // rows
        self.exec_bf18(&scratch, chunk.slice_from_mut(0..));
    }
}

boring_avx_butterfly!(AvxButterfly216f, f32, 216);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly216, f32, AvxButterfly216f, 216, 1e-3);
    test_oof_avx_butterfly!(test_avx_butterfly216_oof, f32, AvxButterfly216f, 216, 1e-3);
}
