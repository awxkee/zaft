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

use crate::avx::mixed::{AvxStoreD, AvxStoreF, ColumnButterfly5d, ColumnButterfly5f};
use crate::avx::transpose::{transpose_5x5_f32, transpose_5x5_f64};
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::sync::Arc;

pub(crate) struct AvxButterfly25d {
    direction: FftDirection,
    bf5: ColumnButterfly5d,
    twiddles: [AvxStoreD; 12],
}

impl AvxButterfly25d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let mut twiddles = [AvxStoreD::zero(); 12];
            let mut q = 0usize;
            let len_per_row = 5;
            const COMPLEX_PER_VECTOR: usize = 2;
            let quotient = len_per_row / COMPLEX_PER_VECTOR;
            let remainder = len_per_row % COMPLEX_PER_VECTOR;

            let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
            for x in 0..num_twiddle_columns {
                for y in 1..5 {
                    twiddles[q] = AvxStoreD::set_complex2(
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 25, fft_direction),
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 25, fft_direction),
                    );
                    q += 1;
                }
            }
            Self {
                direction: fft_direction,
                twiddles,
                bf5: ColumnButterfly5d::new(fft_direction),
            }
        }
    }
}

impl FftExecutor<f64> for AvxButterfly25d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        25
    }
}

impl AvxButterfly25d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 25 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let mut rows1: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];
        let mut rows2: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];
        let mut rows3: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];

        unsafe {
            for chunk in in_place.chunks_exact_mut(25) {
                for i in 0..5 {
                    rows1[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 5..));
                }
                for i in 0..5 {
                    rows2[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 5 + 2..));
                }

                rows1 = self.bf5.exec(rows1);
                rows2 = self.bf5.exec(rows2);

                for i in 1..5 {
                    rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1]);
                    rows2[i] = AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 4]);
                }

                for i in 0..5 {
                    rows3[i] = AvxStoreD::from_complex(chunk.get_unchecked(i * 5 + 4));
                }

                rows3 = self.bf5.exec(rows3);

                for i in 1..5 {
                    rows3[i] = AvxStoreD::mul_by_complex(rows3[i], self.twiddles[i - 1 + 8]);
                }

                let (mut transposed0, mut transposed1, mut transposed2) =
                    transpose_5x5_f64(rows1, rows2, rows3);

                transposed0 = self.bf5.exec(transposed0);
                transposed1 = self.bf5.exec(transposed1);

                transposed0[0].write(chunk);
                transposed0[1].write(chunk.get_unchecked_mut(5..));
                transposed0[2].write(chunk.get_unchecked_mut(10..));
                transposed0[3].write(chunk.get_unchecked_mut(15..));
                transposed0[4].write(chunk.get_unchecked_mut(20..));

                transposed2 = self.bf5.exec(transposed2);

                transposed1[0].write(chunk.get_unchecked_mut(2..));
                transposed1[1].write(chunk.get_unchecked_mut(7..));
                transposed1[2].write(chunk.get_unchecked_mut(12..));
                transposed1[3].write(chunk.get_unchecked_mut(17..));
                transposed1[4].write(chunk.get_unchecked_mut(22..));

                transposed2[0].write_lo(chunk.get_unchecked_mut(4..));
                transposed2[1].write_lo(chunk.get_unchecked_mut(9..));
                transposed2[2].write_lo(chunk.get_unchecked_mut(14..));
                transposed2[3].write_lo(chunk.get_unchecked_mut(19..));
                transposed2[4].write_lo(chunk.get_unchecked_mut(24..));
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly25d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly25d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 25 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 25 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        let mut rows1: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];
        let mut rows2: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];
        let mut rows3: [AvxStoreD; 5] = [AvxStoreD::zero(); 5];

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(25).zip(src.chunks_exact(25)) {
                for i in 0..5 {
                    rows1[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 5..));
                }
                for i in 0..5 {
                    rows2[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 5 + 2..));
                }

                rows1 = self.bf5.exec(rows1);
                rows2 = self.bf5.exec(rows2);

                for i in 1..5 {
                    rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1]);
                    rows2[i] = AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 4]);
                }

                for i in 0..5 {
                    rows3[i] = AvxStoreD::from_complex(src.get_unchecked(i * 5 + 4));
                }

                rows3 = self.bf5.exec(rows3);

                for i in 1..5 {
                    rows3[i] = AvxStoreD::mul_by_complex(rows3[i], self.twiddles[i - 1 + 8]);
                }

                let (mut transposed0, mut transposed1, mut transposed2) =
                    transpose_5x5_f64(rows1, rows2, rows3);

                transposed0 = self.bf5.exec(transposed0);
                transposed1 = self.bf5.exec(transposed1);

                transposed0[0].write(dst);
                transposed0[1].write(dst.get_unchecked_mut(5..));
                transposed0[2].write(dst.get_unchecked_mut(10..));
                transposed0[3].write(dst.get_unchecked_mut(15..));
                transposed0[4].write(dst.get_unchecked_mut(20..));

                transposed2 = self.bf5.exec(transposed2);

                transposed1[0].write(dst.get_unchecked_mut(2..));
                transposed1[1].write(dst.get_unchecked_mut(7..));
                transposed1[2].write(dst.get_unchecked_mut(12..));
                transposed1[3].write(dst.get_unchecked_mut(17..));
                transposed1[4].write(dst.get_unchecked_mut(22..));

                transposed2[0].write_lo(dst.get_unchecked_mut(4..));
                transposed2[1].write_lo(dst.get_unchecked_mut(9..));
                transposed2[2].write_lo(dst.get_unchecked_mut(14..));
                transposed2[3].write_lo(dst.get_unchecked_mut(19..));
                transposed2[4].write_lo(dst.get_unchecked_mut(24..));
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly25d {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

pub(crate) struct AvxButterfly25f {
    direction: FftDirection,
    bf5: ColumnButterfly5f,
    twiddles: [AvxStoreF; 8],
}

impl AvxButterfly25f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let mut twiddles = [AvxStoreF::zero(); 8];
            let mut q = 0usize;
            let len_per_row = 5;
            const COMPLEX_PER_VECTOR: usize = 4;
            let quotient = len_per_row / COMPLEX_PER_VECTOR;
            let remainder = len_per_row % COMPLEX_PER_VECTOR;

            let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
            for x in 0..num_twiddle_columns {
                for y in 1..5 {
                    twiddles[q] = AvxStoreF::set_complex4(
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 25, fft_direction),
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 25, fft_direction),
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 2), 25, fft_direction),
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 3), 25, fft_direction),
                    );
                    q += 1;
                }
            }
            Self {
                direction: fft_direction,
                twiddles,
                bf5: ColumnButterfly5f::new(fft_direction),
            }
        }
    }
}

impl FftExecutor<f32> for AvxButterfly25f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        25
    }
}

impl AvxButterfly25f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 25 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows1: [AvxStoreF; 5] = [AvxStoreF::zero(); 5];
            let mut rows2: [AvxStoreF; 5] = [AvxStoreF::zero(); 5];

            for chunk in in_place.chunks_exact_mut(25) {
                for i in 0..5 {
                    rows1[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 5..));
                }
                for i in 0..5 {
                    rows2[i] = AvxStoreF::from_complex(chunk.get_unchecked(i * 5 + 4));
                }

                rows1 = self.bf5.exec(rows1);
                rows2 = self.bf5.exec(rows2);

                for i in 1..5 {
                    rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1]);
                    rows2[i] = AvxStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 4]);
                }

                let (mut transposed0, mut transposed1) = transpose_5x5_f32(rows1, rows2);

                transposed0 = self.bf5.exec(transposed0);
                transposed1 = self.bf5.exec(transposed1);

                transposed0[0].write(chunk);
                transposed0[1].write(chunk.get_unchecked_mut(5..));
                transposed0[2].write(chunk.get_unchecked_mut(10..));
                transposed0[3].write(chunk.get_unchecked_mut(15..));
                transposed0[4].write(chunk.get_unchecked_mut(20..));

                transposed1[0].write_lo1(chunk.get_unchecked_mut(4..));
                transposed1[1].write_lo1(chunk.get_unchecked_mut(9..));
                transposed1[2].write_lo1(chunk.get_unchecked_mut(14..));
                transposed1[3].write_lo1(chunk.get_unchecked_mut(19..));
                transposed1[4].write_lo1(chunk.get_unchecked_mut(24..));
            }
        }
        Ok(())
    }
}
impl FftExecutorOutOfPlace<f32> for AvxButterfly25f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly25f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 25 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 25 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows1: [AvxStoreF; 5] = [AvxStoreF::zero(); 5];
            let mut rows2: [AvxStoreF; 5] = [AvxStoreF::zero(); 5];

            for (dst, src) in dst.chunks_exact_mut(25).zip(src.chunks_exact(25)) {
                for i in 0..5 {
                    rows1[i] = AvxStoreF::from_complex_ref(src.get_unchecked(i * 5..));
                }
                for i in 0..5 {
                    rows2[i] = AvxStoreF::from_complex(src.get_unchecked(i * 5 + 4));
                }

                rows1 = self.bf5.exec(rows1);
                rows2 = self.bf5.exec(rows2);

                for i in 1..5 {
                    rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1]);
                    rows2[i] = AvxStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 4]);
                }

                let (mut transposed0, mut transposed1) = transpose_5x5_f32(rows1, rows2);

                transposed0 = self.bf5.exec(transposed0);
                transposed1 = self.bf5.exec(transposed1);

                transposed0[0].write(dst);
                transposed0[1].write(dst.get_unchecked_mut(5..));
                transposed0[2].write(dst.get_unchecked_mut(10..));
                transposed0[3].write(dst.get_unchecked_mut(15..));
                transposed0[4].write(dst.get_unchecked_mut(20..));

                transposed1[0].write_lo1(dst.get_unchecked_mut(4..));
                transposed1[1].write_lo1(dst.get_unchecked_mut(9..));
                transposed1[2].write_lo1(dst.get_unchecked_mut(14..));
                transposed1[3].write_lo1(dst.get_unchecked_mut(19..));
                transposed1[4].write_lo1(dst.get_unchecked_mut(24..));
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly25f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly25, f32, AvxButterfly25f, 25, 1e-5);
    test_avx_butterfly!(test_avx_butterfly25_f64, f64, AvxButterfly25d, 25, 1e-7);
    test_oof_avx_butterfly!(test_oof_avx_butterfly25, f32, AvxButterfly25f, 25, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly25_f64, f64, AvxButterfly25d, 25, 1e-9);
}
