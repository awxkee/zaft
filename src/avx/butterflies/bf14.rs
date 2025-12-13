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
    AvxStoreD, AvxStoreF, ColumnButterfly2d, ColumnButterfly2f, ColumnButterfly7d,
    ColumnButterfly7f,
};
use crate::avx::transpose::{avx_transpose_f32x2_4x4_impl, transpose_f64x2_2x2};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x2_7x2(
    rows0: [AvxStoreD; 2],
    rows1: [AvxStoreD; 2],
    rows2: [AvxStoreD; 2],
    rows3: [AvxStoreD; 2],
) -> [AvxStoreD; 7] {
    let a0 = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);
    let b0 = transpose_f64x2_2x2(rows1[0].v, rows1[1].v);
    let c0 = transpose_f64x2_2x2(rows2[0].v, rows2[1].v);
    let d0 = transpose_f64x2_2x2(rows3[0].v, rows3[1].v);
    [
        AvxStoreD::raw(a0.0),
        AvxStoreD::raw(a0.1),
        AvxStoreD::raw(b0.0),
        AvxStoreD::raw(b0.1),
        AvxStoreD::raw(c0.0),
        AvxStoreD::raw(c0.1),
        AvxStoreD::raw(d0.0),
    ]
}

pub(crate) struct AvxButterfly14d {
    direction: FftDirection,
    bf2: ColumnButterfly2d,
    bf7: ColumnButterfly7d,
    twiddles: [AvxStoreD; 4],
}

impl AvxButterfly14d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(7, 2, fft_direction, 14),
            bf2: ColumnButterfly2d::new(fft_direction),
            bf7: ColumnButterfly7d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly14d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        14
    }
}

impl AvxButterfly14d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(14) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
            let mut rows1: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
            let mut rows2: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
            let mut rows3: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];

            for chunk in in_place.chunks_exact_mut(14) {
                // columns
                for i in 0..2 {
                    rows0[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7..));
                    rows1[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7 + 2..));
                    rows2[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7 + 4..));
                    rows3[i] = AvxStoreD::from_complex(chunk.get_unchecked(i * 7 + 6));
                }

                rows0 = self.bf2.exec(rows0);
                rows1 = self.bf2.exec(rows1);
                rows2 = self.bf2.exec(rows2);
                rows3 = self.bf2.exec(rows3);

                rows0[1] = AvxStoreD::mul_by_complex(rows0[1], self.twiddles[0]);
                rows1[1] = AvxStoreD::mul_by_complex(rows1[1], self.twiddles[1]);
                rows2[1] = AvxStoreD::mul_by_complex(rows2[1], self.twiddles[2]);
                rows3[1] = AvxStoreD::mul_by_complex(rows3[1], self.twiddles[3]);

                let transposed = transpose_f64x2_7x2(rows0, rows1, rows2, rows3);

                let q0 = self.bf7.exec(transposed);

                for i in 0..7 {
                    q0[i].write(chunk.get_unchecked_mut(i * 2..));
                }
            }
        }
        Ok(())
    }
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) fn transpose_f32x2_7x2(rows0: [AvxStoreF; 2], rows1: [AvxStoreF; 2]) -> [AvxStoreF; 7] {
    let a0 = avx_transpose_f32x2_4x4_impl(
        rows0[0].v,
        rows0[1].v,
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let b0 = avx_transpose_f32x2_4x4_impl(
        rows1[0].v,
        rows1[1].v,
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );

    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
        AvxStoreF::raw(b0.0),
        AvxStoreF::raw(b0.1),
        AvxStoreF::raw(b0.2),
    ]
}

pub(crate) struct AvxButterfly14f {
    direction: FftDirection,
    bf2: ColumnButterfly2f,
    bf7: ColumnButterfly7f,
    twiddles: [AvxStoreF; 2],
}

impl AvxButterfly14f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(7, 2, fft_direction, 14),
            bf2: ColumnButterfly2f::new(fft_direction),
            bf7: ColumnButterfly7f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly14f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        14
    }
}

impl AvxButterfly14f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(14) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 2] = [AvxStoreF::zero(); 2];
            let mut rows1: [AvxStoreF; 2] = [AvxStoreF::zero(); 2];

            for chunk in in_place.chunks_exact_mut(14) {
                // columns
                for i in 0..2 {
                    rows0[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 7..));
                    rows1[i] = AvxStoreF::from_complex3(chunk.get_unchecked(i * 7 + 4..));
                }

                rows0 = self.bf2.exec(rows0);
                rows1 = self.bf2.exec(rows1);

                rows0[1] = AvxStoreF::mul_by_complex(rows0[1], self.twiddles[0]);
                rows1[1] = AvxStoreF::mul_by_complex(rows1[1], self.twiddles[1]);

                let transposed = transpose_f32x2_7x2(rows0, rows1);

                let q0 = self.bf7.exec(transposed);

                for i in 0..7 {
                    q0[i].write_lo2(chunk.get_unchecked_mut(i * 2..));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly14, f32, AvxButterfly14f, 14, 1e-4);
    test_avx_butterfly!(test_avx_butterfly14_f64, f64, AvxButterfly14d, 14, 1e-7);
}
