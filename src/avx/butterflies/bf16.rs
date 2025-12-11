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
use crate::avx::mixed::{AvxStoreD, AvxStoreF, ColumnButterfly4d, ColumnButterfly4f};
use crate::avx::transpose::{transpose_f32x2_4x4_aos, transpose_f64x2_2x2};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::sync::Arc;

pub(crate) struct AvxButterfly16d {
    direction: FftDirection,
    bf4: ColumnButterfly4d,
    twiddles: [AvxStoreD; 6],
}

impl AvxButterfly16d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(4, 4, fft_direction, 16),
            bf4: ColumnButterfly4d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly16d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        16
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_4x4(
    rows0: [AvxStoreD; 4],
    rows1: [AvxStoreD; 4],
) -> ([AvxStoreD; 4], [AvxStoreD; 4]) {
    let a0 = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);
    let d0 = transpose_f64x2_2x2(rows0[2].v, rows0[3].v);

    let b0 = transpose_f64x2_2x2(rows1[0].v, rows1[1].v);
    let e0 = transpose_f64x2_2x2(rows1[2].v, rows1[3].v);
    (
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
        ],
        [
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
            AvxStoreD::raw(e0.1),
        ],
    )
}

impl AvxButterfly16d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(16) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
            let mut rows1: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];

            for chunk in in_place.chunks_exact_mut(16) {
                // columns
                for i in 0..4 {
                    rows0[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 4..));
                    rows1[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 4 + 2..));
                }

                rows0 = self.bf4.exec(rows0);
                rows1 = self.bf4.exec(rows1);

                for i in 1..4 {
                    rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 3]);
                }

                let transposed = transpose_4x4(rows0, rows1);

                let q0 = self.bf4.exec(transposed.0);
                let q1 = self.bf4.exec(transposed.1);

                for i in 0..4 {
                    q0[i].write(chunk.get_unchecked_mut(i * 4..));
                    q1[i].write(chunk.get_unchecked_mut(i * 4 + 2..));
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
        if !src.len().is_multiple_of(16) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(16) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
            let mut rows1: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];

            for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
                // columns
                for i in 0..4 {
                    rows0[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 4..));
                    rows1[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 4 + 2..));
                }

                rows0 = self.bf4.exec(rows0);
                rows1 = self.bf4.exec(rows1);

                for i in 1..4 {
                    rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 3]);
                }

                let transposed = transpose_4x4(rows0, rows1);

                let q0 = self.bf4.exec(transposed.0);
                let q1 = self.bf4.exec(transposed.1);

                for i in 0..4 {
                    q0[i].write(dst.get_unchecked_mut(i * 4..));
                    q1[i].write(dst.get_unchecked_mut(i * 4 + 2..));
                }
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly16d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly16d {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

// float
pub(crate) struct AvxButterfly16f {
    direction: FftDirection,
    bf4: ColumnButterfly4f,
    twiddles: [AvxStoreF; 3],
}

impl AvxButterfly16f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(4, 4, fft_direction, 16),
            bf4: ColumnButterfly4f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly16f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        16
    }
}

impl AvxButterfly16f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(16) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 4] = [AvxStoreF::zero(); 4];

            for chunk in in_place.chunks_exact_mut(16) {
                // columns
                for i in 0..4 {
                    rows0[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 4..));
                }

                rows0 = self.bf4.exec(rows0);

                for i in 1..4 {
                    rows0[i] = AvxStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                }

                let transposed = transpose_f32x2_4x4_aos(rows0);

                let q0 = self.bf4.exec(transposed);

                for i in 0..4 {
                    q0[i].write(chunk.get_unchecked_mut(i * 4..));
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
        if !src.len().is_multiple_of(16) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(16) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 4] = [AvxStoreF::zero(); 4];

            for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
                // columns
                for i in 0..4 {
                    rows0[i] = AvxStoreF::from_complex_ref(src.get_unchecked(i * 4..));
                }

                rows0 = self.bf4.exec(rows0);

                for i in 1..4 {
                    rows0[i] = AvxStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                }

                let transposed = transpose_f32x2_4x4_aos(rows0);

                let q0 = self.bf4.exec(transposed);

                for i in 0..4 {
                    q0[i].write(dst.get_unchecked_mut(i * 4..));
                }
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly16f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly16f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly16, f32, AvxButterfly16f, 16, 1e-5);
    test_avx_butterfly!(test_avx_butterfly16_f64, f64, AvxButterfly16d, 16, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly16, f32, AvxButterfly16f, 16, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly16_f64, f64, AvxButterfly16d, 16, 1e-7);
}
