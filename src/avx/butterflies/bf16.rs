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

use crate::avx::butterflies::shared::{
    boring_avx_butterfly, gen_butterfly_twiddles_f32, gen_butterfly_twiddles_f64,
};
use crate::avx::mixed::{AvxStoreD, AvxStoreF, ColumnButterfly4d, ColumnButterfly4f};
use crate::avx::transpose::{transpose_f32x2_4x4_aos, transpose_f64x2_2x2};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, R2CFftExecutor, ZaftError};
use num_complex::Complex;

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

boring_avx_butterfly!(AvxButterfly16d, f64, 16);

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
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows0: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
        let mut rows1: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
        // columns
        for i in 0..4 {
            rows0[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 4..));
            rows1[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 4 + 2..));
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
            q0[i].write(chunk.slice_from_mut(i * 4..));
            q1[i].write(chunk.slice_from_mut(i * 4 + 2..));
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_c2r(&self, src: &[f64], dst: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(16) {
            return Err(ZaftError::InvalidSizeMultiplier(
                src.len(),
                self.real_length(),
            ));
        }
        if !dst.len().is_multiple_of(9) {
            return Err(ZaftError::InvalidSizeMultiplier(
                dst.len(),
                self.complex_length(),
            ));
        }
        if src.len() / 16 != dst.len() / 9 {
            return Err(ZaftError::InvalidSamplesCount(
                src.len() / 16,
                dst.len() / 9,
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
            let mut rows1: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];

            for (dst, src) in dst.chunks_exact_mut(9).zip(src.chunks_exact(16)) {
                // columns
                for i in 0..4 {
                    let q = AvxStoreD::load(src.get_unchecked(i * 4..));
                    let [v0, v1] = q.to_complex();
                    rows0[i] = v0;
                    rows1[i] = v1;
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

                for i in 0..2 {
                    q0[i].write(dst.get_unchecked_mut(i * 4..));
                    q1[i].write(dst.get_unchecked_mut(i * 4 + 2..));
                }
                q0[2].write_lo(dst.get_unchecked_mut(8..));
            }
        }
        Ok(())
    }
}

impl R2CFftExecutor<f64> for AvxButterfly16d {
    fn execute(&self, input: &[f64], output: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_c2r(input, output) }
    }

    fn execute_with_scratch(
        &self,
        input: &[f64],
        output: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_c2r(input, output) }
    }

    #[inline]
    fn real_length(&self) -> usize {
        16
    }

    #[inline]
    fn complex_length(&self) -> usize {
        9
    }

    fn complex_scratch_length(&self) -> usize {
        0
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

boring_avx_butterfly!(AvxButterfly16f, f32, 16);

impl AvxButterfly16f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows0: [AvxStoreF; 4] = [AvxStoreF::zero(); 4];
        // columns
        for i in 0..4 {
            rows0[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 4..));
        }

        rows0 = self.bf4.exec(rows0);

        for i in 1..4 {
            rows0[i] = AvxStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
        }

        let transposed = transpose_f32x2_4x4_aos(rows0);

        let q0 = self.bf4.exec(transposed);

        for i in 0..4 {
            q0[i].write(chunk.slice_from_mut(i * 4..));
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_r2c(&self, src: &[f32], dst: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(16) {
            return Err(ZaftError::InvalidSizeMultiplier(
                src.len(),
                self.real_length(),
            ));
        }
        if !dst.len().is_multiple_of(9) {
            return Err(ZaftError::InvalidSizeMultiplier(
                src.len(),
                self.complex_length(),
            ));
        }
        if src.len() / 16 != dst.len() / 9 {
            return Err(ZaftError::InvalidSamplesCount(
                src.len() / 16,
                dst.len() / 9,
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 4] = [AvxStoreF::zero(); 4];

            for (dst, src) in dst.chunks_exact_mut(9).zip(src.chunks_exact(16)) {
                // columns
                let [u0, u1] = AvxStoreF::load(src).to_complex();
                let [u2, u3] = AvxStoreF::load(src.get_unchecked(8..)).to_complex();
                rows0[0] = u0;
                rows0[1] = u1;
                rows0[2] = u2;
                rows0[3] = u3;

                rows0 = self.bf4.exec(rows0);

                for i in 1..4 {
                    rows0[i] = AvxStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                }

                let transposed = transpose_f32x2_4x4_aos(rows0);

                let q0 = self.bf4.exec(transposed);

                for i in 0..2 {
                    q0[i].write(dst.get_unchecked_mut(i * 4..));
                }
                q0[2].write_lo1(dst.get_unchecked_mut(8..));
            }
        }
        Ok(())
    }
}

impl R2CFftExecutor<f32> for AvxButterfly16f {
    fn execute(&self, input: &[f32], output: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_r2c(input, output) }
    }

    fn execute_with_scratch(
        &self,
        input: &[f32],
        output: &mut [Complex<f32>],
        _: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_r2c(input, output) }
    }

    #[inline]
    fn real_length(&self) -> usize {
        16
    }

    #[inline]
    fn complex_length(&self) -> usize {
        9
    }

    fn complex_scratch_length(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{
        test_avx_butterfly, test_oof_avx_butterfly, test_r2c_avx_butterfly,
    };

    test_r2c_avx_butterfly!(test_avx_r2c_butterfly16d, f64, AvxButterfly16d, 16, 1e-5);
    test_r2c_avx_butterfly!(test_avx_r2c_butterfly16, f32, AvxButterfly16f, 16, 1e-5);

    test_avx_butterfly!(test_avx_butterfly16, f32, AvxButterfly16f, 16, 1e-5);
    test_avx_butterfly!(test_avx_butterfly16_f64, f64, AvxButterfly16d, 16, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly16, f32, AvxButterfly16f, 16, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly16_f64, f64, AvxButterfly16d, 16, 1e-7);
}
