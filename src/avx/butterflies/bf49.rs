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
use crate::avx::f32x2_7x7::{transpose_7x7_f32, transpose_7x7_f64};
use crate::avx::mixed::{AvxStoreD, AvxStoreF, ColumnButterfly7d, ColumnButterfly7f};
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;

pub(crate) struct AvxButterfly49d {
    direction: FftDirection,
    bf7: ColumnButterfly7d,
    twiddles: [AvxStoreD; 24],
}

impl AvxButterfly49d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let mut twiddles = [AvxStoreD::zero(); 24];
            let mut q = 0usize;
            let len_per_row = 7;
            const COMPLEX_PER_VECTOR: usize = 2;
            let quotient = len_per_row / COMPLEX_PER_VECTOR;
            let remainder = len_per_row % COMPLEX_PER_VECTOR;

            let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
            for x in 0..num_twiddle_columns {
                for y in 1..7 {
                    twiddles[q] = AvxStoreD::set_complex2(
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 49, fft_direction),
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 49, fft_direction),
                    );
                    q += 1;
                }
            }
            Self {
                direction: fft_direction,
                twiddles,
                bf7: ColumnButterfly7d::new(fft_direction),
            }
        }
    }
}

impl FftExecutor<f64> for AvxButterfly49d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        49
    }
}

impl AvxButterfly49d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 49 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let mut rows1: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];
        let mut rows2: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];
        let mut rows3: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];
        let mut rows4: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];

        unsafe {
            for chunk in in_place.chunks_exact_mut(49) {
                // rows - 1

                for i in 0..7 {
                    rows1[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7..));
                }

                rows1 = self.bf7.exec(rows1);

                for i in 1..7 {
                    rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1]);
                }

                // rows - 2

                for i in 0..7 {
                    rows2[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7 + 2..));
                }

                rows2 = self.bf7.exec(rows2);

                for i in 1..7 {
                    rows2[i] = AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 6]);
                }

                // rows - 3

                for i in 0..7 {
                    rows3[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 7 + 4..));
                }

                rows3 = self.bf7.exec(rows3);

                for i in 1..7 {
                    rows3[i] = AvxStoreD::mul_by_complex(rows3[i], self.twiddles[i - 1 + 12]);
                }

                // rows - 4

                for i in 0..7 {
                    rows4[i] = AvxStoreD::from_complex(chunk.get_unchecked(i * 7 + 6));
                }

                rows4 = self.bf7.exec(rows4);

                for i in 1..7 {
                    rows4[i] = AvxStoreD::mul_by_complex(rows4[i], self.twiddles[i - 1 + 18]);
                }

                let (mut transposed0, mut transposed1, mut transposed2, mut transposed3) =
                    transpose_7x7_f64(rows1, rows2, rows3, rows4);

                transposed0 = self.bf7.exec(transposed0);

                for i in 0..7 {
                    transposed0[i].write(chunk.get_unchecked_mut(i * 7..));
                }

                transposed1 = self.bf7.exec(transposed1);

                for i in 0..7 {
                    transposed1[i].write(chunk.get_unchecked_mut(i * 7 + 2..));
                }

                transposed2 = self.bf7.exec(transposed2);

                for i in 0..7 {
                    transposed2[i].write(chunk.get_unchecked_mut(i * 7 + 4..));
                }

                transposed3 = self.bf7.exec(transposed3);

                for i in 0..7 {
                    transposed3[i].write_lo(chunk.get_unchecked_mut(i * 7 + 6..));
                }
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly49d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly49d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 49 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 49 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        let mut rows1: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];
        let mut rows2: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];
        let mut rows3: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];
        let mut rows4: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(49).zip(src.chunks_exact(49)) {
                // rows - 1

                for i in 0..7 {
                    rows1[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 7..));
                }

                rows1 = self.bf7.exec(rows1);

                for i in 1..7 {
                    rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1]);
                }

                // rows - 2

                for i in 0..7 {
                    rows2[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 7 + 2..));
                }

                rows2 = self.bf7.exec(rows2);

                for i in 1..7 {
                    rows2[i] = AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 6]);
                }

                // rows - 3

                for i in 0..7 {
                    rows3[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * 7 + 4..));
                }

                rows3 = self.bf7.exec(rows3);

                for i in 1..7 {
                    rows3[i] = AvxStoreD::mul_by_complex(rows3[i], self.twiddles[i - 1 + 12]);
                }

                // rows - 4

                for i in 0..7 {
                    rows4[i] = AvxStoreD::from_complex(src.get_unchecked(i * 7 + 6));
                }

                rows4 = self.bf7.exec(rows4);

                for i in 1..7 {
                    rows4[i] = AvxStoreD::mul_by_complex(rows4[i], self.twiddles[i - 1 + 18]);
                }

                let (mut transposed0, mut transposed1, mut transposed2, mut transposed3) =
                    transpose_7x7_f64(rows1, rows2, rows3, rows4);

                transposed0 = self.bf7.exec(transposed0);

                for i in 0..7 {
                    transposed0[i].write(dst.get_unchecked_mut(i * 7..));
                }

                transposed1 = self.bf7.exec(transposed1);

                for i in 0..7 {
                    transposed1[i].write(dst.get_unchecked_mut(i * 7 + 2..));
                }

                transposed2 = self.bf7.exec(transposed2);

                for i in 0..7 {
                    transposed2[i].write(dst.get_unchecked_mut(i * 7 + 4..));
                }

                transposed3 = self.bf7.exec(transposed3);

                for i in 0..7 {
                    transposed3[i].write_lo(dst.get_unchecked_mut(i * 7 + 6..));
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly49d {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

pub(crate) struct AvxButterfly49f {
    direction: FftDirection,
    bf7: ColumnButterfly7f,
    twiddles: [AvxStoreF; 12],
}

impl AvxButterfly49f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let mut twiddles = [AvxStoreF::zero(); 12];
            let mut q = 0usize;
            let len_per_row = 7;
            const COMPLEX_PER_VECTOR: usize = 4;
            let quotient = len_per_row / COMPLEX_PER_VECTOR;
            let remainder = len_per_row % COMPLEX_PER_VECTOR;

            let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
            for x in 0..num_twiddle_columns {
                for y in 1..7 {
                    twiddles[q] = AvxStoreF::set_complex4(
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 49, fft_direction),
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 49, fft_direction),
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 2), 49, fft_direction),
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 3), 49, fft_direction),
                    );
                    q += 1;
                }
            }
            Self {
                direction: fft_direction,
                twiddles,
                bf7: ColumnButterfly7f::new(fft_direction),
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn store_transpose_7x7_f32(
    left: [AvxStoreF; 7],
    right: [AvxStoreF; 7],
) -> ([AvxStoreF; 7], [AvxStoreF; 7]) {
    let (q0, q1) = transpose_7x7_f32(
        [
            left[0].v, left[1].v, left[2].v, left[3].v, left[4].v, left[5].v, left[6].v,
        ],
        [
            right[0].v, right[1].v, right[2].v, right[3].v, right[4].v, right[5].v, right[6].v,
        ],
    );
    (
        [
            AvxStoreF::raw(q0[0]),
            AvxStoreF::raw(q0[1]),
            AvxStoreF::raw(q0[2]),
            AvxStoreF::raw(q0[3]),
            AvxStoreF::raw(q0[4]),
            AvxStoreF::raw(q0[5]),
            AvxStoreF::raw(q0[6]),
        ],
        [
            AvxStoreF::raw(q1[0]),
            AvxStoreF::raw(q1[1]),
            AvxStoreF::raw(q1[2]),
            AvxStoreF::raw(q1[3]),
            AvxStoreF::raw(q1[4]),
            AvxStoreF::raw(q1[5]),
            AvxStoreF::raw(q1[6]),
        ],
    )
}

impl FftExecutor<f32> for AvxButterfly49f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        49
    }
}

impl AvxButterfly49f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 49 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows1: [AvxStoreF; 7] = [AvxStoreF::zero(); 7];
            let mut rows2: [AvxStoreF; 7] = [AvxStoreF::zero(); 7];

            for chunk in in_place.chunks_exact_mut(49) {
                for i in 0..7 {
                    rows1[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 7..));
                }
                for i in 0..7 {
                    rows2[i] = AvxStoreF::from_complex3(chunk.get_unchecked(i * 7 + 4..));
                }

                rows1 = self.bf7.exec(rows1);
                rows2 = self.bf7.exec(rows2);

                for i in 1..7 {
                    rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1]);
                    rows2[i] = AvxStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 6]);
                }

                let (mut transposed0, mut transposed1) = store_transpose_7x7_f32(rows1, rows2);

                transposed0 = self.bf7.exec(transposed0);
                transposed1 = self.bf7.exec(transposed1);

                transposed0[0].write(chunk);
                transposed0[1].write(chunk.get_unchecked_mut(7..));
                transposed0[2].write(chunk.get_unchecked_mut(14..));
                transposed0[3].write(chunk.get_unchecked_mut(21..));
                transposed0[4].write(chunk.get_unchecked_mut(28..));
                transposed0[5].write(chunk.get_unchecked_mut(35..));
                transposed0[6].write(chunk.get_unchecked_mut(42..));

                transposed1[0].write_lo3(chunk.get_unchecked_mut(4..));
                transposed1[1].write_lo3(chunk.get_unchecked_mut(11..));
                transposed1[2].write_lo3(chunk.get_unchecked_mut(18..));
                transposed1[3].write_lo3(chunk.get_unchecked_mut(25..));
                transposed1[4].write_lo3(chunk.get_unchecked_mut(32..));
                transposed1[5].write_lo3(chunk.get_unchecked_mut(39..));
                transposed1[6].write_lo3(chunk.get_unchecked_mut(46..));
            }
        }
        Ok(())
    }
}
impl FftExecutorOutOfPlace<f32> for AvxButterfly49f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly49f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 49 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 49 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows1: [AvxStoreF; 7] = [AvxStoreF::zero(); 7];
            let mut rows2: [AvxStoreF; 7] = [AvxStoreF::zero(); 7];

            for (dst, src) in dst.chunks_exact_mut(49).zip(src.chunks_exact(49)) {
                for i in 0..7 {
                    rows1[i] = AvxStoreF::from_complex_ref(src.get_unchecked(i * 7..));
                }
                for i in 0..7 {
                    rows2[i] = AvxStoreF::from_complex3(src.get_unchecked(i * 7 + 4..));
                }

                rows1 = self.bf7.exec(rows1);
                rows2 = self.bf7.exec(rows2);

                for i in 1..7 {
                    rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1]);
                    rows2[i] = AvxStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 6]);
                }

                let (mut transposed0, mut transposed1) = store_transpose_7x7_f32(rows1, rows2);

                transposed0 = self.bf7.exec(transposed0);
                transposed1 = self.bf7.exec(transposed1);

                transposed0[0].write(dst);
                transposed0[1].write(dst.get_unchecked_mut(7..));
                transposed0[2].write(dst.get_unchecked_mut(14..));
                transposed0[3].write(dst.get_unchecked_mut(21..));
                transposed0[4].write(dst.get_unchecked_mut(28..));
                transposed0[5].write(dst.get_unchecked_mut(35..));
                transposed0[6].write(dst.get_unchecked_mut(42..));

                transposed1[0].write_lo3(dst.get_unchecked_mut(4..));
                transposed1[1].write_lo3(dst.get_unchecked_mut(11..));
                transposed1[2].write_lo3(dst.get_unchecked_mut(18..));
                transposed1[3].write_lo3(dst.get_unchecked_mut(25..));
                transposed1[4].write_lo3(dst.get_unchecked_mut(32..));
                transposed1[5].write_lo3(dst.get_unchecked_mut(39..));
                transposed1[6].write_lo3(dst.get_unchecked_mut(46..));
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly49f {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly49, f32, AvxButterfly49f, 49, 1e-3);
    test_avx_butterfly!(test_avx_butterfly49_f64, f64, AvxButterfly49d, 49, 1e-7);
    test_oof_avx_butterfly!(test_oof_avx_butterfly49, f32, AvxButterfly49f, 49, 1e-3);
    test_oof_avx_butterfly!(test_oof_avx_butterfly49_f64, f64, AvxButterfly49d, 49, 1e-9);
}
