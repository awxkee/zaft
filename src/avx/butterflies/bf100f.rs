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
use crate::avx::mixed::{AvxStoreF, ColumnButterfly10f};
use crate::avx::transpose::avx_transpose_f32x2_4x4_impl;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::_mm256_setzero_ps;
use std::mem::MaybeUninit;
use std::sync::Arc;

pub(crate) struct AvxButterfly100f {
    direction: FftDirection,
    bf10: ColumnButterfly10f,
    twiddles: [AvxStoreF; 27],
}

impl AvxButterfly100f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(10, 10, fft_direction, 100),
            bf10: ColumnButterfly10f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly100f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        100
    }
}

impl AvxButterfly100f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(100) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [AvxStoreF; 10] = [AvxStoreF::zero(); 10];
            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 100];

            for chunk in in_place.chunks_exact_mut(100) {
                // columns
                for k in 0..2 {
                    for i in 0..10 {
                        rows[i] =
                            AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 10 + k * 4..));
                    }

                    rows = self.bf10.exec(rows);

                    let q1 = AvxStoreF::mul_by_complex(rows[1], self.twiddles[9 * k]);
                    let q2 = AvxStoreF::mul_by_complex(rows[2], self.twiddles[9 * k + 1]);
                    let q3 = AvxStoreF::mul_by_complex(rows[3], self.twiddles[9 * k + 2]);
                    let t = avx_transpose_f32x2_4x4_impl(rows[0].v, q1.v, q2.v, q3.v);
                    AvxStoreF::raw(t.0).write_u(scratch.get_unchecked_mut(k * 4 * 10..));
                    AvxStoreF::raw(t.1).write_u(scratch.get_unchecked_mut((k * 4 + 1) * 10..));
                    AvxStoreF::raw(t.2).write_u(scratch.get_unchecked_mut((k * 4 + 2) * 10..));
                    AvxStoreF::raw(t.3).write_u(scratch.get_unchecked_mut((k * 4 + 3) * 10..));

                    for i in 1..2 {
                        let q0 = AvxStoreF::mul_by_complex(
                            rows[i * 4],
                            self.twiddles[(i - 1) * 4 + 3 + 9 * k],
                        );
                        let q1 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 1],
                            self.twiddles[(i - 1) * 4 + 4 + 9 * k],
                        );
                        let q2 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 2],
                            self.twiddles[(i - 1) * 4 + 5 + 9 * k],
                        );
                        let q3 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 3],
                            self.twiddles[(i - 1) * 4 + 6 + 9 * k],
                        );
                        let t = avx_transpose_f32x2_4x4_impl(q0.v, q1.v, q2.v, q3.v);
                        AvxStoreF::raw(t.0)
                            .write_u(scratch.get_unchecked_mut(k * 4 * 10 + i * 4..));
                        AvxStoreF::raw(t.1)
                            .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 10 + i * 4..));
                        AvxStoreF::raw(t.2)
                            .write_u(scratch.get_unchecked_mut((k * 4 + 2) * 10 + i * 4..));
                        AvxStoreF::raw(t.3)
                            .write_u(scratch.get_unchecked_mut((k * 4 + 3) * 10 + i * 4..));
                    }

                    {
                        let i = 2;
                        let q0 = AvxStoreF::mul_by_complex(
                            rows[i * 4],
                            self.twiddles[(i - 1) * 4 + 3 + 9 * k],
                        );
                        let q1 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 1],
                            self.twiddles[(i - 1) * 4 + 4 + 9 * k],
                        );
                        let t = avx_transpose_f32x2_4x4_impl(
                            q0.v,
                            q1.v,
                            _mm256_setzero_ps(),
                            _mm256_setzero_ps(),
                        );
                        AvxStoreF::raw(t.0)
                            .write_lo2u(scratch.get_unchecked_mut(k * 4 * 10 + i * 4..));
                        AvxStoreF::raw(t.1)
                            .write_lo2u(scratch.get_unchecked_mut((k * 4 + 1) * 10 + i * 4..));
                        AvxStoreF::raw(t.2)
                            .write_lo2u(scratch.get_unchecked_mut((k * 4 + 2) * 10 + i * 4..));
                        AvxStoreF::raw(t.3)
                            .write_lo2u(scratch.get_unchecked_mut((k * 4 + 3) * 10 + i * 4..));
                    }
                }

                {
                    let k = 2;
                    for i in 0..10 {
                        rows[i] = AvxStoreF::from_complex2(chunk.get_unchecked(i * 10 + 8..));
                    }

                    rows = self.bf10.exec(rows);

                    let q1 = AvxStoreF::mul_by_complex(rows[1], self.twiddles[9 * k]);
                    let q2 = AvxStoreF::mul_by_complex(rows[2], self.twiddles[9 * k + 1]);
                    let q3 = AvxStoreF::mul_by_complex(rows[3], self.twiddles[9 * k + 2]);
                    let t = avx_transpose_f32x2_4x4_impl(rows[0].v, q1.v, q2.v, q3.v);
                    AvxStoreF::raw(t.0).write_u(scratch.get_unchecked_mut(k * 4 * 10..));
                    AvxStoreF::raw(t.1).write_u(scratch.get_unchecked_mut((k * 4 + 1) * 10..));

                    for i in 1..2 {
                        let q0 = AvxStoreF::mul_by_complex(
                            rows[i * 4],
                            self.twiddles[(i - 1) * 4 + 3 + 9 * k],
                        );
                        let q1 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 1],
                            self.twiddles[(i - 1) * 4 + 4 + 9 * k],
                        );
                        let q2 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 2],
                            self.twiddles[(i - 1) * 4 + 5 + 9 * k],
                        );
                        let q3 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 3],
                            self.twiddles[(i - 1) * 4 + 6 + 9 * k],
                        );
                        let t = avx_transpose_f32x2_4x4_impl(q0.v, q1.v, q2.v, q3.v);
                        AvxStoreF::raw(t.0)
                            .write_u(scratch.get_unchecked_mut(k * 4 * 10 + i * 4..));
                        AvxStoreF::raw(t.1)
                            .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 10 + i * 4..));
                    }

                    {
                        let i = 2;
                        let q0 = AvxStoreF::mul_by_complex(
                            rows[i * 4],
                            self.twiddles[(i - 1) * 4 + 3 + 9 * k],
                        );
                        let q1 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 1],
                            self.twiddles[(i - 1) * 4 + 4 + 9 * k],
                        );
                        let t = avx_transpose_f32x2_4x4_impl(
                            q0.v,
                            q1.v,
                            _mm256_setzero_ps(),
                            _mm256_setzero_ps(),
                        );
                        AvxStoreF::raw(t.0)
                            .write_lo2u(scratch.get_unchecked_mut(k * 4 * 10 + i * 4..));
                        AvxStoreF::raw(t.1)
                            .write_lo2u(scratch.get_unchecked_mut((k * 4 + 1) * 10 + i * 4..));
                    }
                }

                // rows

                for k in 0..2 {
                    for i in 0..10 {
                        rows[i] =
                            AvxStoreF::from_complex_refu(scratch.get_unchecked(i * 10 + k * 4..));
                    }
                    rows = self.bf10.exec(rows);
                    for i in 0..10 {
                        rows[i].write(chunk.get_unchecked_mut(i * 10 + k * 4..));
                    }
                }

                {
                    for i in 0..10 {
                        rows[i] = AvxStoreF::from_complex2u(scratch.get_unchecked(i * 10 + 8..));
                    }
                    rows = self.bf10.exec(rows);
                    for i in 0..10 {
                        rows[i].write_lo2(chunk.get_unchecked_mut(i * 10 + 8..));
                    }
                }
            }
        }
        Ok(())
    }
}
impl FftExecutorOutOfPlace<f32> for AvxButterfly100f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly100f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(100) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(100) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows: [AvxStoreF; 10] = [AvxStoreF::zero(); 10];
            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 100];

            for (dst, src) in dst.chunks_exact_mut(100).zip(src.chunks_exact(100)) {
                // columns
                for k in 0..2 {
                    for i in 0..10 {
                        rows[i] = AvxStoreF::from_complex_ref(src.get_unchecked(i * 10 + k * 4..));
                    }

                    rows = self.bf10.exec(rows);

                    let q1 = AvxStoreF::mul_by_complex(rows[1], self.twiddles[9 * k]);
                    let q2 = AvxStoreF::mul_by_complex(rows[2], self.twiddles[9 * k + 1]);
                    let q3 = AvxStoreF::mul_by_complex(rows[3], self.twiddles[9 * k + 2]);
                    let t = avx_transpose_f32x2_4x4_impl(rows[0].v, q1.v, q2.v, q3.v);
                    AvxStoreF::raw(t.0).write_u(scratch.get_unchecked_mut(k * 4 * 10..));
                    AvxStoreF::raw(t.1).write_u(scratch.get_unchecked_mut((k * 4 + 1) * 10..));
                    AvxStoreF::raw(t.2).write_u(scratch.get_unchecked_mut((k * 4 + 2) * 10..));
                    AvxStoreF::raw(t.3).write_u(scratch.get_unchecked_mut((k * 4 + 3) * 10..));

                    for i in 1..2 {
                        let q0 = AvxStoreF::mul_by_complex(
                            rows[i * 4],
                            self.twiddles[(i - 1) * 4 + 3 + 9 * k],
                        );
                        let q1 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 1],
                            self.twiddles[(i - 1) * 4 + 4 + 9 * k],
                        );
                        let q2 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 2],
                            self.twiddles[(i - 1) * 4 + 5 + 9 * k],
                        );
                        let q3 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 3],
                            self.twiddles[(i - 1) * 4 + 6 + 9 * k],
                        );
                        let t = avx_transpose_f32x2_4x4_impl(q0.v, q1.v, q2.v, q3.v);
                        AvxStoreF::raw(t.0)
                            .write_u(scratch.get_unchecked_mut(k * 4 * 10 + i * 4..));
                        AvxStoreF::raw(t.1)
                            .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 10 + i * 4..));
                        AvxStoreF::raw(t.2)
                            .write_u(scratch.get_unchecked_mut((k * 4 + 2) * 10 + i * 4..));
                        AvxStoreF::raw(t.3)
                            .write_u(scratch.get_unchecked_mut((k * 4 + 3) * 10 + i * 4..));
                    }

                    {
                        let i = 2;
                        let q0 = AvxStoreF::mul_by_complex(
                            rows[i * 4],
                            self.twiddles[(i - 1) * 4 + 3 + 9 * k],
                        );
                        let q1 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 1],
                            self.twiddles[(i - 1) * 4 + 4 + 9 * k],
                        );
                        let t = avx_transpose_f32x2_4x4_impl(
                            q0.v,
                            q1.v,
                            _mm256_setzero_ps(),
                            _mm256_setzero_ps(),
                        );
                        AvxStoreF::raw(t.0)
                            .write_lo2u(scratch.get_unchecked_mut(k * 4 * 10 + i * 4..));
                        AvxStoreF::raw(t.1)
                            .write_lo2u(scratch.get_unchecked_mut((k * 4 + 1) * 10 + i * 4..));
                        AvxStoreF::raw(t.2)
                            .write_lo2u(scratch.get_unchecked_mut((k * 4 + 2) * 10 + i * 4..));
                        AvxStoreF::raw(t.3)
                            .write_lo2u(scratch.get_unchecked_mut((k * 4 + 3) * 10 + i * 4..));
                    }
                }

                {
                    let k = 2;
                    for i in 0..10 {
                        rows[i] = AvxStoreF::from_complex2(src.get_unchecked(i * 10 + 8..));
                    }

                    rows = self.bf10.exec(rows);

                    let q1 = AvxStoreF::mul_by_complex(rows[1], self.twiddles[9 * k]);
                    let q2 = AvxStoreF::mul_by_complex(rows[2], self.twiddles[9 * k + 1]);
                    let q3 = AvxStoreF::mul_by_complex(rows[3], self.twiddles[9 * k + 2]);
                    let t = avx_transpose_f32x2_4x4_impl(rows[0].v, q1.v, q2.v, q3.v);
                    AvxStoreF::raw(t.0).write_u(scratch.get_unchecked_mut(k * 4 * 10..));
                    AvxStoreF::raw(t.1).write_u(scratch.get_unchecked_mut((k * 4 + 1) * 10..));

                    for i in 1..2 {
                        let q0 = AvxStoreF::mul_by_complex(
                            rows[i * 4],
                            self.twiddles[(i - 1) * 4 + 3 + 9 * k],
                        );
                        let q1 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 1],
                            self.twiddles[(i - 1) * 4 + 4 + 9 * k],
                        );
                        let q2 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 2],
                            self.twiddles[(i - 1) * 4 + 5 + 9 * k],
                        );
                        let q3 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 3],
                            self.twiddles[(i - 1) * 4 + 6 + 9 * k],
                        );
                        let t = avx_transpose_f32x2_4x4_impl(q0.v, q1.v, q2.v, q3.v);
                        AvxStoreF::raw(t.0)
                            .write_u(scratch.get_unchecked_mut(k * 4 * 10 + i * 4..));
                        AvxStoreF::raw(t.1)
                            .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 10 + i * 4..));
                    }

                    {
                        let i = 2;
                        let q0 = AvxStoreF::mul_by_complex(
                            rows[i * 4],
                            self.twiddles[(i - 1) * 4 + 3 + 9 * k],
                        );
                        let q1 = AvxStoreF::mul_by_complex(
                            rows[i * 4 + 1],
                            self.twiddles[(i - 1) * 4 + 4 + 9 * k],
                        );
                        let t = avx_transpose_f32x2_4x4_impl(
                            q0.v,
                            q1.v,
                            _mm256_setzero_ps(),
                            _mm256_setzero_ps(),
                        );
                        AvxStoreF::raw(t.0)
                            .write_lo2u(scratch.get_unchecked_mut(k * 4 * 10 + i * 4..));
                        AvxStoreF::raw(t.1)
                            .write_lo2u(scratch.get_unchecked_mut((k * 4 + 1) * 10 + i * 4..));
                    }
                }

                // rows

                for k in 0..2 {
                    for i in 0..10 {
                        rows[i] =
                            AvxStoreF::from_complex_refu(scratch.get_unchecked(i * 10 + k * 4..));
                    }
                    rows = self.bf10.exec(rows);
                    for i in 0..10 {
                        rows[i].write(dst.get_unchecked_mut(i * 10 + k * 4..));
                    }
                }

                {
                    for i in 0..10 {
                        rows[i] = AvxStoreF::from_complex2u(scratch.get_unchecked(i * 10 + 8..));
                    }
                    rows = self.bf10.exec(rows);
                    for i in 0..10 {
                        rows[i].write_lo2(dst.get_unchecked_mut(i * 10 + 8..));
                    }
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly100f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly100, f32, AvxButterfly100f, 100, 1e-3);
    test_oof_avx_butterfly!(test_oof_avx_butterfly100, f32, AvxButterfly100f, 100, 1e-3);
}
