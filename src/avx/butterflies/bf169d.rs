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

use crate::avx::butterflies::shared::gen_butterfly_twiddles_f64;
use crate::avx::mixed::{AvxStoreD, ColumnButterfly13d};
use crate::avx::transpose::{transpose_f64x2_2x2, transpose_f64x2_2x13};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::_mm256_setzero_pd;
use std::mem::MaybeUninit;
use std::sync::Arc;

pub(crate) struct AvxButterfly169d {
    direction: FftDirection,
    bf13: ColumnButterfly13d,
    twiddles: [AvxStoreD; 84],
}

impl AvxButterfly169d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f64(13, 13, fft_direction, 169),
            bf13: ColumnButterfly13d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly169d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        169
    }
}

impl AvxButterfly169d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 169 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [AvxStoreD; 13] = [AvxStoreD::zero(); 13];
            let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 169];

            for chunk in in_place.chunks_exact_mut(169) {
                // columns
                for k in 0..6 {
                    for i in 0..13 {
                        rows[i] =
                            AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 13 + k * 2..));
                    }

                    rows = self.bf13.exec(rows);

                    let q1 = AvxStoreD::mul_by_complex(rows[1], self.twiddles[12 * k]);
                    let t = transpose_f64x2_2x2(rows[0].v, q1.v);
                    AvxStoreD::raw(t.0).write_u(scratch.get_unchecked_mut(k * 2 * 13..));
                    AvxStoreD::raw(t.1).write_u(scratch.get_unchecked_mut((k * 2 + 1) * 13..));

                    for i in 1..6 {
                        let q0 = AvxStoreD::mul_by_complex(
                            rows[i * 2],
                            self.twiddles[(i - 1) * 2 + 1 + 12 * k],
                        );
                        let q1 = AvxStoreD::mul_by_complex(
                            rows[i * 2 + 1],
                            self.twiddles[(i - 1) * 2 + 2 + 12 * k],
                        );
                        let t = transpose_f64x2_2x2(q0.v, q1.v);
                        AvxStoreD::raw(t.0)
                            .write_u(scratch.get_unchecked_mut(k * 2 * 13 + i * 2..));
                        AvxStoreD::raw(t.1)
                            .write_u(scratch.get_unchecked_mut((k * 2 + 1) * 13 + i * 2..));
                    }

                    {
                        let i = 6;
                        let q0 = AvxStoreD::mul_by_complex(
                            rows[i * 2],
                            self.twiddles[(i - 1) * 2 + 1 + 12 * k],
                        );
                        let t = transpose_f64x2_2x2(q0.v, _mm256_setzero_pd());
                        AvxStoreD::raw(t.0)
                            .write_lou(scratch.get_unchecked_mut(k * 2 * 13 + i * 2..));
                        AvxStoreD::raw(t.1)
                            .write_lou(scratch.get_unchecked_mut((k * 2 + 1) * 13 + i * 2..));
                    }
                }

                {
                    let k = 6;
                    for i in 0..13 {
                        rows[i] = AvxStoreD::from_complex(chunk.get_unchecked(i * 13 + k * 2));
                    }

                    rows = self.bf13.exec(rows);

                    let q1 = AvxStoreD::mul_by_complex(rows[1], self.twiddles[12 * k]);
                    let t = transpose_f64x2_2x2(rows[0].v, q1.v);
                    AvxStoreD::raw(t.0).write_u(scratch.get_unchecked_mut(k * 2 * 13..));

                    for i in 1..6 {
                        let q0 = AvxStoreD::mul_by_complex(
                            rows[i * 2],
                            self.twiddles[(i - 1) * 2 + 1 + 12 * k],
                        );
                        let q1 = AvxStoreD::mul_by_complex(
                            rows[i * 2 + 1],
                            self.twiddles[(i - 1) * 2 + 2 + 12 * k],
                        );
                        let t = transpose_f64x2_2x2(q0.v, q1.v);
                        AvxStoreD::raw(t.0)
                            .write_u(scratch.get_unchecked_mut(k * 2 * 13 + i * 2..));
                    }

                    {
                        let i = 6;
                        let q0 = AvxStoreD::mul_by_complex(
                            rows[i * 2],
                            self.twiddles[(i - 1) * 2 + 1 + 12 * k],
                        );
                        let t = transpose_f64x2_2x2(q0.v, _mm256_setzero_pd());
                        AvxStoreD::raw(t.0)
                            .write_lou(scratch.get_unchecked_mut(k * 2 * 13 + i * 2..));
                    }
                }

                // rows

                for k in 0..6 {
                    for i in 0..13 {
                        rows[i] =
                            AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 13 + k * 2..));
                    }
                    rows = self.bf13.exec(rows);
                    for i in 0..13 {
                        rows[i].write(chunk.get_unchecked_mut(i * 13 + k * 2..));
                    }
                }
                {
                    let k = 6;
                    for i in 0..13 {
                        rows[i] = AvxStoreD::from_complexu(scratch.get_unchecked(i * 13 + k * 2));
                    }
                    rows = self.bf13.exec(rows);
                    for i in 0..13 {
                        rows[i].write_lo(chunk.get_unchecked_mut(i * 13 + k * 2..));
                    }
                }
            }
        }
        Ok(())
    }
}
impl FftExecutorOutOfPlace<f64> for AvxButterfly169d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly169d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 169 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 169 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows: [AvxStoreD; 13] = [AvxStoreD::zero(); 13];
            let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 169];

            for (dst, src) in dst.chunks_exact_mut(169).zip(src.chunks_exact(169)) {
                // columns
                for k in 0..6 {
                    for i in 0..13 {
                        rows[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 13 + k * 2..));
                    }

                    rows = self.bf13.exec(rows);

                    for i in 1..13 {
                        rows[i] = AvxStoreD::mul_by_complex(rows[i], self.twiddles[i - 1 + 12 * k]);
                    }

                    let transposed = transpose_f64x2_2x13(rows);

                    for i in 0..6 {
                        transposed[i * 2].write_u(scratch.get_unchecked_mut(k * 2 * 13 + i * 2..));
                        transposed[i * 2 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 2 + 1) * 13 + i * 2..));
                    }
                    {
                        let i = 6;
                        transposed[i * 2]
                            .write_lou(scratch.get_unchecked_mut(k * 2 * 13 + i * 2..));
                        transposed[i * 2 + 1]
                            .write_lou(scratch.get_unchecked_mut((k * 2 + 1) * 13 + i * 2..));
                    }
                }

                {
                    let k = 6;
                    for i in 0..13 {
                        rows[i] = AvxStoreD::from_complex(src.get_unchecked(i * 13 + k * 2));
                    }

                    rows = self.bf13.exec(rows);

                    let q1 = AvxStoreD::mul_by_complex(rows[1], self.twiddles[12 * k]);
                    let t = transpose_f64x2_2x2(rows[0].v, q1.v);
                    AvxStoreD::raw(t.0).write_u(scratch.get_unchecked_mut(k * 2 * 13..));

                    for i in 1..6 {
                        let q0 = AvxStoreD::mul_by_complex(
                            rows[i * 2],
                            self.twiddles[(i - 1) * 2 + 1 + 12 * k],
                        );
                        let q1 = AvxStoreD::mul_by_complex(
                            rows[i * 2 + 1],
                            self.twiddles[(i - 1) * 2 + 2 + 12 * k],
                        );
                        let t = transpose_f64x2_2x2(q0.v, q1.v);
                        AvxStoreD::raw(t.0)
                            .write_u(scratch.get_unchecked_mut(k * 2 * 13 + i * 2..));
                    }

                    {
                        let i = 6;
                        let q0 = AvxStoreD::mul_by_complex(
                            rows[i * 2],
                            self.twiddles[(i - 1) * 2 + 1 + 12 * k],
                        );
                        let t = transpose_f64x2_2x2(q0.v, _mm256_setzero_pd());
                        AvxStoreD::raw(t.0)
                            .write_lou(scratch.get_unchecked_mut(k * 2 * 13 + i * 2..));
                    }
                }

                // rows

                for k in 0..6 {
                    for i in 0..13 {
                        rows[i] =
                            AvxStoreD::from_complex_refu(scratch.get_unchecked(i * 13 + k * 2..));
                    }
                    rows = self.bf13.exec(rows);
                    for i in 0..13 {
                        rows[i].write(dst.get_unchecked_mut(i * 13 + k * 2..));
                    }
                }
                {
                    let k = 6;
                    for i in 0..13 {
                        rows[i] = AvxStoreD::from_complexu(scratch.get_unchecked(i * 13 + k * 2));
                    }
                    rows = self.bf13.exec(rows);
                    for i in 0..13 {
                        rows[i].write_lo(dst.get_unchecked_mut(i * 13 + k * 2..));
                    }
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly169d {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly169, f64, AvxButterfly169d, 169, 1e-3);
    test_oof_avx_butterfly!(test_oof_avx_butterfly169, f64, AvxButterfly169d, 169, 1e-3);
}
