/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use crate::avx::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::avx::mixed::{AvxStoreF, ColumnButterfly9f};
use crate::avx::transpose::avx_transpose_f32x2_4x4_impl;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;
use std::sync::Arc;

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_f32x2_9x3(
    rows0: [AvxStoreF; 3],
    rows1: [AvxStoreF; 3],
    rows2: [AvxStoreF; 3],
) -> [AvxStoreF; 9] {
    let a0 = avx_transpose_f32x2_4x4_impl(rows0[0].v, rows0[1].v, rows0[2].v, _mm256_setzero_ps());
    let b0 = avx_transpose_f32x2_4x4_impl(rows1[0].v, rows1[1].v, rows1[2].v, _mm256_setzero_ps());
    let c0 = avx_transpose_f32x2_4x4_impl(rows2[0].v, rows2[1].v, rows2[2].v, _mm256_setzero_ps());
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
        AvxStoreF::raw(c0.0),
    ]
}

pub(crate) struct AvxButterfly27f {
    direction: FftDirection,
    bf9: ColumnButterfly9f,
    twiddles: [AvxStoreF; 6],
}

impl AvxButterfly27f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_f32(9, 3, fft_direction, 27),
            bf9: ColumnButterfly9f::new(fft_direction),
            direction: fft_direction,
        }
    }
}

impl AvxButterfly27f {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(27) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];
            let mut rows1: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];
            let mut rows2: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];

            for chunk in in_place.chunks_exact_mut(27) {
                // columns
                for i in 0..3 {
                    rows0[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 9..));
                    rows1[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 9 + 4..));
                    rows2[i] = AvxStoreF::from_complex(chunk.get_unchecked(i * 9 + 8));
                }

                rows0 = self.bf9.bf3.exec(rows0);
                rows1 = self.bf9.bf3.exec(rows1);
                rows2 = self.bf9.bf3.exec(rows2);

                for i in 1..3 {
                    rows0[i] = AvxStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1 + 2]);
                    rows2[i] = AvxStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 4]);
                }

                let t = transpose_f32x2_9x3(rows0, rows1, rows2);

                // rows
                let left = self.bf9.exec(t);

                for i in 0..9 {
                    left[i].write_lo3(chunk.get_unchecked_mut(i * 3..));
                }
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_out_of_place_f32(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(27) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(27) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows0: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];
            let mut rows1: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];
            let mut rows2: [AvxStoreF; 3] = [AvxStoreF::zero(); 3];

            for (dst, src) in dst.chunks_exact_mut(27).zip(src.chunks_exact(27)) {
                // columns
                for i in 0..3 {
                    rows0[i] = AvxStoreF::from_complex_ref(src.get_unchecked(i * 9..));
                    rows1[i] = AvxStoreF::from_complex_ref(src.get_unchecked(i * 9 + 4..));
                    rows2[i] = AvxStoreF::from_complex(src.get_unchecked(i * 9 + 8));
                }

                rows0 = self.bf9.bf3.exec(rows0);
                rows1 = self.bf9.bf3.exec(rows1);
                rows2 = self.bf9.bf3.exec(rows2);

                for i in 1..3 {
                    rows0[i] = AvxStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1 + 2]);
                    rows2[i] = AvxStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 4]);
                }

                let t = transpose_f32x2_9x3(rows0, rows1, rows2);

                // rows
                let left = self.bf9.exec(t);

                for i in 0..9 {
                    left[i].write_lo3(dst.get_unchecked_mut(i * 3..));
                }
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly27f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly27f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for AvxButterfly27f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        27
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly27, f32, AvxButterfly27f, 27, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly27, f32, AvxButterfly27f, 27, 1e-5);
}
