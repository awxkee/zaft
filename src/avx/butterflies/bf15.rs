// Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::avx::butterflies::shared::{gen_butterfly_twiddles_f32, gen_butterfly_twiddles_f64};
use crate::avx::mixed::{
    AvxStoreD, AvxStoreF, ColumnButterfly3d, ColumnButterfly3f, ColumnButterfly5d,
    ColumnButterfly5f,
};
use crate::avx::transpose::{avx_transpose_f32x2_4x4_impl, transpose_f64x2_2x2};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly15d {
    direction: FftDirection,
    bf3: ColumnButterfly3d,
    bf5: ColumnButterfly5d,
    twiddles: [AvxStoreD; 6],
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x2_5x3(
    rows0: [AvxStoreD; 3],
    rows1: [AvxStoreD; 3],
    rows2: [AvxStoreD; 3],
) -> ([AvxStoreD; 5], [AvxStoreD; 5]) {
    let a0 = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);
    let d0 = transpose_f64x2_2x2(rows0[2].v, _mm256_setzero_pd());

    let b0 = transpose_f64x2_2x2(rows1[0].v, rows1[1].v);
    let e0 = transpose_f64x2_2x2(rows1[2].v, _mm256_setzero_pd());

    let c0 = transpose_f64x2_2x2(rows2[0].v, rows2[1].v);
    let f0 = transpose_f64x2_2x2(rows2[2].v, _mm256_setzero_pd());
    (
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
        ],
        [
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
            AvxStoreD::raw(e0.1),
            AvxStoreD::raw(f0.0),
        ],
    )
}

impl AvxButterfly15d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(5, 3, fft_direction, 15),
            bf3: ColumnButterfly3d::new(fft_direction),
            bf5: ColumnButterfly5d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly15d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        15
    }
}

impl AvxButterfly15d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(15) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
            let mut rows1: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
            let mut rows2: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];

            for chunk in in_place.chunks_exact_mut(15) {
                // columns
                for i in 0..3 {
                    rows0[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 5..));
                    rows1[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 5 + 2..));
                    rows2[i] = AvxStoreD::from_complex(chunk.get_unchecked(i * 5 + 4));
                }

                rows0 = self.bf3.exec(rows0);
                rows1 = self.bf3.exec(rows1);
                rows2 = self.bf3.exec(rows2);

                for i in 1..3 {
                    rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 2]);
                    rows2[i] = AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 4]);
                }

                let transposed = transpose_f64x2_5x3(rows0, rows1, rows2);

                let q0 = self.bf5.exec(transposed.0);
                let q1 = self.bf5.exec(transposed.1);

                for i in 0..5 {
                    q0[i].write(chunk.get_unchecked_mut(i * 3..));
                    q1[i].write_lo(chunk.get_unchecked_mut(i * 3 + 2..));
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct AvxButterfly15f {
    direction: FftDirection,
    bf3: ColumnButterfly3f,
    bf5: ColumnButterfly5f,
    twiddles: [AvxStoreF; 6],
}

impl AvxButterfly15f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(5, 3, fft_direction, 15),
            bf3: ColumnButterfly3f::new(fft_direction),
            bf5: ColumnButterfly5f::new(fft_direction),
        }
    }
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) fn transpose_f32x2_5x3(rows0: [AvxStoreF; 3], rows1: [AvxStoreF; 3]) -> [AvxStoreF; 5] {
    let a0 = avx_transpose_f32x2_4x4_impl(rows0[0].v, rows0[1].v, rows0[2].v, _mm256_setzero_ps());
    let b0 = avx_transpose_f32x2_4x4_impl(rows1[0].v, rows1[1].v, rows1[2].v, _mm256_setzero_ps());

    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
        AvxStoreF::raw(b0.0),
    ]
}

impl AvxButterfly15f {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(15) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];
            let mut rows1: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];

            for chunk in in_place.chunks_exact_mut(15) {
                // columns
                for i in 0..3 {
                    rows0[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 5..));
                    rows1[i] = AvxStoreF::from_complex(chunk.get_unchecked(i * 5 + 4));
                }

                rows0 = self.bf3.exec(rows0);
                rows1 = self.bf3.exec(rows1);

                for i in 1..3 {
                    rows0[i] = AvxStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1 + 2]);
                }

                let transposed = transpose_f32x2_5x3(rows0, rows1);

                let q0 = self.bf5.exec(transposed);

                for i in 0..5 {
                    q0[i].write_lo3(chunk.get_unchecked_mut(i * 3..));
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly15f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        15
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly15, f32, AvxButterfly15f, 15, 1e-5);
    test_avx_butterfly!(test_avx_butterfly15_f64, f64, AvxButterfly15d, 15, 1e-7);
}
