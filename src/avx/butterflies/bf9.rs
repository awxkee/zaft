// Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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
use crate::avx::mixed::{AvxStoreD, AvxStoreF, ColumnButterfly3d, ColumnButterfly3f};
use crate::avx::transpose::{avx_transpose_f32x2_4x4_impl, transpose_f64x2_2x2};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;
use std::sync::Arc;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x2_3x3(
    rows0: [AvxStoreD; 3],
    rows1: [AvxStoreD; 3],
) -> ([AvxStoreD; 3], [AvxStoreD; 3]) {
    let a0 = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);
    let d0 = transpose_f64x2_2x2(rows0[2].v, _mm256_setzero_pd());

    let b0 = transpose_f64x2_2x2(rows1[0].v, rows1[1].v);
    let e0 = transpose_f64x2_2x2(rows1[2].v, _mm256_setzero_pd());
    (
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
        ],
        [
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
        ],
    )
}

pub(crate) struct AvxButterfly9d {
    direction: FftDirection,
    twiddles: [AvxStoreD; 4],
    bf3: ColumnButterfly3d,
}

impl AvxButterfly9d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(3, 3, fft_direction, 9),
            bf3: ColumnButterfly3d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly9d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        9
    }
}

impl AvxButterfly9d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(9) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
            let mut rows1: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];

            for chunk in in_place.chunks_exact_mut(9) {
                // columns
                for i in 0..3 {
                    rows0[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 3..));
                    rows1[i] = AvxStoreD::from_complex(chunk.get_unchecked(i * 3 + 2));
                }

                rows0 = self.bf3.exec(rows0);
                rows1 = self.bf3.exec(rows1);

                for i in 1..3 {
                    rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 2]);
                }

                (rows0, rows1) = transpose_f64x2_3x3(rows0, rows1);

                // rows

                rows0 = self.bf3.exec(rows0);
                rows1 = self.bf3.exec(rows1);

                for i in 0..3 {
                    rows0[i].write(chunk.get_unchecked_mut(i * 3..));
                    rows1[i].write_lo(chunk.get_unchecked_mut(i * 3 + 2..));
                }
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(9) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(9) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
            let mut rows1: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];

            for (dst, src) in dst.chunks_exact_mut(9).zip(src.chunks_exact(9)) {
                // columns
                for i in 0..3 {
                    rows0[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 3..));
                    rows1[i] = AvxStoreD::from_complex(src.get_unchecked(i * 3 + 2));
                }

                rows0 = self.bf3.exec(rows0);
                rows1 = self.bf3.exec(rows1);

                for i in 1..3 {
                    rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 2]);
                }

                (rows0, rows1) = transpose_f64x2_3x3(rows0, rows1);

                // rows

                rows0 = self.bf3.exec(rows0);
                rows1 = self.bf3.exec(rows1);

                for i in 0..3 {
                    rows0[i].write(dst.get_unchecked_mut(i * 3..));
                    rows1[i].write_lo(dst.get_unchecked_mut(i * 3 + 2..));
                }
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly9d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly9d {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_f32x2_3x3(rows: [AvxStoreF; 3]) -> [AvxStoreF; 3] {
    let a0 = avx_transpose_f32x2_4x4_impl(rows[0].v, rows[1].v, rows[2].v, _mm256_setzero_ps());
    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
    ]
}

pub(crate) struct AvxButterfly9f {
    direction: FftDirection,
    twiddles: [AvxStoreF; 2],
    bf3: ColumnButterfly3f,
}

impl AvxButterfly9f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(3, 3, fft_direction, 9),
            bf3: ColumnButterfly3f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly9f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        9
    }
}

impl AvxButterfly9f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(9) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];

            for chunk in in_place.chunks_exact_mut(9) {
                // columns
                for i in 0..3 {
                    rows0[i] = AvxStoreF::from_complex3(chunk.get_unchecked(i * 3..));
                }

                rows0 = self.bf3.exec(rows0);

                rows0[1] = AvxStoreF::mul_by_complex(rows0[1], self.twiddles[0]);
                rows0[2] = AvxStoreF::mul_by_complex(rows0[2], self.twiddles[1]);

                rows0 = transpose_f32x2_3x3(rows0);

                // rows

                rows0 = self.bf3.exec(rows0);

                for i in 0..3 {
                    rows0[i].write_lo3(chunk.get_unchecked_mut(i * 3..));
                }
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(9) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(9) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];

            for (dst, src) in dst.chunks_exact_mut(9).zip(src.chunks_exact(9)) {
                // columns
                for i in 0..3 {
                    rows0[i] = AvxStoreF::from_complex3(src.get_unchecked(i * 3..));
                }

                rows0 = self.bf3.exec(rows0);

                rows0[1] = AvxStoreF::mul_by_complex(rows0[1], self.twiddles[0]);
                rows0[2] = AvxStoreF::mul_by_complex(rows0[2], self.twiddles[1]);

                rows0 = transpose_f32x2_3x3(rows0);

                // rows

                rows0 = self.bf3.exec(rows0);

                for i in 0..3 {
                    rows0[i].write_lo3(dst.get_unchecked_mut(i * 3..));
                }
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly9f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly9f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly9, f32, AvxButterfly9f, 9, 1e-5);
    test_avx_butterfly!(test_avx_butterfly9_f64, f64, AvxButterfly9d, 9, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly9, f32, AvxButterfly9f, 9, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly9_f64, f64, AvxButterfly9d, 9, 1e-7);
}
