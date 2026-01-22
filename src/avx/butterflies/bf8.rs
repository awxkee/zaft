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
use crate::avx::mixed::{
    AvxStoreD, AvxStoreF, ColumnButterfly2d, ColumnButterfly2f, ColumnButterfly4d,
    ColumnButterfly4f,
};
use crate::avx::transpose::{avx_transpose_f32x2_4x4_impl, transpose_f64x2_2x2};
use crate::{
    CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, R2CFftExecutor,
    ZaftError,
};
use num_complex::Complex;
use std::arch::x86_64::*;
use std::sync::Arc;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_f64x2_4x2(rows0: [AvxStoreD; 2], rows1: [AvxStoreD; 2]) -> [AvxStoreD; 4] {
    let a0 = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);
    let b0 = transpose_f64x2_2x2(rows1[0].v, rows1[1].v);
    [
        AvxStoreD::raw(a0.0),
        AvxStoreD::raw(a0.1),
        AvxStoreD::raw(b0.0),
        AvxStoreD::raw(b0.1),
    ]
}

pub(crate) struct AvxButterfly8d {
    direction: FftDirection,
    bf4: ColumnButterfly4d,
    bf2: ColumnButterfly2d,
    twiddles: [AvxStoreD; 2],
}

impl AvxButterfly8d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }
    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf4: ColumnButterfly4d::new(fft_direction),
            bf2: ColumnButterfly2d::new(fft_direction),
            twiddles: gen_butterfly_twiddles_f64(4, 2, fft_direction, 8),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly8d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        8
    }
}

impl AvxButterfly8d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(8) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
            let mut rows1: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];

            for chunk in in_place.chunks_exact_mut(8) {
                // columns
                for i in 0..2 {
                    rows0[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 4..));
                    rows1[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 4 + 2..));
                }

                rows0 = self.bf2.exec(rows0);
                rows1 = self.bf2.exec(rows1);

                rows0[1] = AvxStoreD::mul_by_complex(rows0[1], self.twiddles[0]);
                rows1[1] = AvxStoreD::mul_by_complex(rows1[1], self.twiddles[1]);

                let transposed = transpose_f64x2_4x2(rows0, rows1);

                let q0 = self.bf4.exec(transposed);

                for i in 0..4 {
                    q0[i].write(chunk.get_unchecked_mut(i * 2..));
                }
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_r2c(&self, src: &[f64], dst: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(8) {
            return Err(ZaftError::InvalidSizeMultiplier(
                src.len(),
                self.real_length(),
            ));
        }
        if !dst.len().is_multiple_of(5) {
            return Err(ZaftError::InvalidSizeMultiplier(
                src.len(),
                self.complex_length(),
            ));
        }
        if src.len() / 8 != dst.len() / 5 {
            return Err(ZaftError::InvalidSamplesCount(src.len() / 8, dst.len() / 5));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
            let mut rows1: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];

            for (chunk, complex) in src.chunks_exact(8).zip(dst.chunks_exact_mut(5)) {
                // columns
                for i in 0..2 {
                    let q = AvxStoreD::load(chunk.get_unchecked(i * 4..));
                    let [v0, v1] = q.to_complex();
                    rows0[i] = v0;
                    rows1[i] = v1;
                }

                rows0 = self.bf2.exec(rows0);
                rows1 = self.bf2.exec(rows1);

                rows0[1] = AvxStoreD::mul_by_complex(rows0[1], self.twiddles[0]);
                rows1[1] = AvxStoreD::mul_by_complex(rows1[1], self.twiddles[1]);

                let transposed = transpose_f64x2_4x2(rows0, rows1);

                let q0 = self.bf4.exec(transposed);

                for i in 0..2 {
                    q0[i].write(complex.get_unchecked_mut(i * 2..));
                }
                q0[2].write_lo(complex.get_unchecked_mut(4..));
            }
        }
        Ok(())
    }
}

impl R2CFftExecutor<f64> for AvxButterfly8d {
    fn execute(&self, input: &[f64], output: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_r2c(input, output) }
    }

    fn real_length(&self) -> usize {
        8
    }

    fn complex_length(&self) -> usize {
        5
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly8d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly8d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(8) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(8) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];
            let mut rows1: [AvxStoreD; 2] = [AvxStoreD::zero(); 2];

            for (dst, src) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
                // columns
                for i in 0..2 {
                    rows0[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 4..));
                    rows1[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 4 + 2..));
                }

                rows0 = self.bf2.exec(rows0);
                rows1 = self.bf2.exec(rows1);

                rows0[1] = AvxStoreD::mul_by_complex(rows0[1], self.twiddles[0]);
                rows1[1] = AvxStoreD::mul_by_complex(rows1[1], self.twiddles[1]);

                let transposed = transpose_f64x2_4x2(rows0, rows1);

                let q0 = self.bf4.exec(transposed);

                for i in 0..4 {
                    q0[i].write(dst.get_unchecked_mut(i * 2..));
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly8d {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) fn transpose_f32x2_4x2(rows0: [AvxStoreF; 2]) -> [AvxStoreF; 4] {
    let a0 = avx_transpose_f32x2_4x4_impl(
        rows0[0].v,
        rows0[1].v,
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );

    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
    ]
}

pub(crate) struct AvxButterfly8f {
    direction: FftDirection,
    bf4: ColumnButterfly4f,
    bf2: ColumnButterfly2f,
    twiddles: [AvxStoreF; 1],
}

impl AvxButterfly8f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }
    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf4: ColumnButterfly4f::new(fft_direction),
            bf2: ColumnButterfly2f::new(fft_direction),
            twiddles: gen_butterfly_twiddles_f32(4, 2, fft_direction, 8),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly8f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        8
    }
}

impl AvxButterfly8f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(8) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 2] = [AvxStoreF::zero(); 2];

            for chunk in in_place.chunks_exact_mut(8) {
                // columns
                for i in 0..2 {
                    rows0[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 4..));
                }

                rows0 = self.bf2.exec(rows0);

                rows0[1] = AvxStoreF::mul_by_complex(rows0[1], self.twiddles[0]);

                let transposed = transpose_f32x2_4x2(rows0);

                // rows
                let q0 = self.bf4.exec(transposed);

                for i in 0..4 {
                    q0[i].write_lo2(chunk.get_unchecked_mut(i * 2..));
                }
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_r2c(&self, src: &[f32], dst: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(8) {
            return Err(ZaftError::InvalidSizeMultiplier(
                src.len(),
                self.real_length(),
            ));
        }
        if !dst.len().is_multiple_of(5) {
            return Err(ZaftError::InvalidSizeMultiplier(
                src.len(),
                self.complex_length(),
            ));
        }
        if src.len() / 8 != dst.len() / 5 {
            return Err(ZaftError::InvalidSamplesCount(src.len() / 8, dst.len() / 5));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 2] = [AvxStoreF::zero(); 2];

            for (chunk, complex) in src.chunks_exact(8).zip(dst.chunks_exact_mut(5)) {
                // columns
                let q0 = AvxStoreF::load(chunk);
                let [v0, v1] = q0.to_complex();
                rows0[0] = v0;
                rows0[1] = v1;

                rows0 = self.bf2.exec(rows0);

                rows0[1] = AvxStoreF::mul_by_complex(rows0[1], self.twiddles[0]);

                let transposed = transpose_f32x2_4x2(rows0);

                // rows
                let q0 = self.bf4.exec(transposed);

                q0[0].write_lo2(complex);
                q0[1].write_lo2(complex.get_unchecked_mut(2..));
                q0[2].write_lo1(complex.get_unchecked_mut(4..));
            }
        }
        Ok(())
    }
}

impl R2CFftExecutor<f32> for AvxButterfly8f {
    fn execute(&self, input: &[f32], output: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_r2c(input, output) }
    }

    #[inline]
    fn real_length(&self) -> usize {
        8
    }

    #[inline]
    fn complex_length(&self) -> usize {
        5
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly8f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly8f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(8) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(8) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 2] = [AvxStoreF::zero(); 2];

            for (dst, src) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
                // columns
                for i in 0..2 {
                    rows0[i] = AvxStoreF::from_complex_ref(src.get_unchecked(i * 4..));
                }

                rows0 = self.bf2.exec(rows0);

                rows0[1] = AvxStoreF::mul_by_complex(rows0[1], self.twiddles[0]);

                let transposed = transpose_f32x2_4x2(rows0);

                // rows
                let q0 = self.bf4.exec(transposed);

                for i in 0..4 {
                    q0[i].write_lo2(dst.get_unchecked_mut(i * 2..));
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly8f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{
        test_avx_butterfly, test_oof_avx_butterfly, test_r2c_avx_butterfly,
    };

    test_r2c_avx_butterfly!(test_avx_r2c_butterfly8, f32, AvxButterfly8f, 8, 1e-5);
    test_r2c_avx_butterfly!(test_avx_r2c_butterfly8d, f64, AvxButterfly8d, 8, 1e-5);

    test_avx_butterfly!(test_avx_butterfly8, f32, AvxButterfly8f, 8, 1e-5);
    test_avx_butterfly!(test_avx_butterfly8_f64, f64, AvxButterfly8d, 8, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly8, f32, AvxButterfly8f, 8, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly8_f64, f64, AvxButterfly8d, 8, 1e-7);
}
