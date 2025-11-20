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
use crate::neon::mixed::{ColumnFcmaButterfly7f, NeonStoreF};
use crate::neon::transpose::neon_transpose_f32x2_7x7_aos;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;

pub(crate) struct NeonFcmaButterfly49f {
    direction: FftDirection,
    bf7: ColumnFcmaButterfly7f,
    twiddles: [NeonStoreF; 24],
}

impl NeonFcmaButterfly49f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let mut twiddles = [NeonStoreF::default(); 24];
        let mut q = 0usize;
        let len_per_row = 7;
        const COMPLEX_PER_VECTOR: usize = 2;
        let quotient = len_per_row / COMPLEX_PER_VECTOR;
        let remainder = len_per_row % COMPLEX_PER_VECTOR;

        let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
        for x in 0..num_twiddle_columns {
            for y in 1..7 {
                twiddles[q] = NeonStoreF::from_complex2(
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 49, fft_direction),
                    compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 49, fft_direction),
                );
                q += 1;
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf7: ColumnFcmaButterfly7f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for NeonFcmaButterfly49f {
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

impl NeonFcmaButterfly49f {
    #[target_feature(enable = "fcma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 49 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows1: [NeonStoreF; 7] = [NeonStoreF::default(); 7];
            let mut rows2: [NeonStoreF; 7] = [NeonStoreF::default(); 7];
            let mut rows3: [NeonStoreF; 7] = [NeonStoreF::default(); 7];
            let mut rows4: [NeonStoreF; 7] = [NeonStoreF::default(); 7];

            for chunk in in_place.chunks_exact_mut(49) {
                for i in 0..7 {
                    rows1[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 7..));
                }
                for i in 0..7 {
                    rows2[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 7 + 2..));
                }

                rows1 = self.bf7.exec(rows1);
                rows2 = self.bf7.exec(rows2);

                for i in 1..7 {
                    rows1[i] = NeonStoreF::fcmul_fcma(rows1[i], self.twiddles[i - 1]);
                    rows2[i] = NeonStoreF::fcmul_fcma(rows2[i], self.twiddles[i - 1 + 6]);
                }

                for i in 0..7 {
                    rows3[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 7 + 4..));
                }
                for i in 0..7 {
                    rows4[i] = NeonStoreF::from_complex(chunk.get_unchecked(i * 7 + 6));
                }

                rows3 = self.bf7.exec(rows3);
                rows4 = self.bf7.exec(rows4);

                for i in 1..7 {
                    rows3[i] = NeonStoreF::fcmul_fcma(rows3[i], self.twiddles[i - 1 + 12]);
                    rows4[i] = NeonStoreF::fcmul_fcma(rows4[i], self.twiddles[i - 1 + 18]);
                }

                let (mut transposed0, mut transposed1, mut transposed2, mut transposed3) =
                    neon_transpose_f32x2_7x7_aos(rows1, rows2, rows3, rows4);

                transposed0 = self.bf7.exec(transposed0);
                transposed1 = self.bf7.exec(transposed1);

                transposed0[0].write(chunk);
                transposed0[1].write(chunk.get_unchecked_mut(7..));
                transposed0[2].write(chunk.get_unchecked_mut(14..));
                transposed0[3].write(chunk.get_unchecked_mut(21..));
                transposed0[4].write(chunk.get_unchecked_mut(28..));
                transposed0[5].write(chunk.get_unchecked_mut(35..));
                transposed0[6].write(chunk.get_unchecked_mut(42..));

                transposed1[0].write(chunk.get_unchecked_mut(2..));
                transposed1[1].write(chunk.get_unchecked_mut(9..));
                transposed1[2].write(chunk.get_unchecked_mut(16..));
                transposed1[3].write(chunk.get_unchecked_mut(23..));
                transposed1[4].write(chunk.get_unchecked_mut(30..));
                transposed1[5].write(chunk.get_unchecked_mut(37..));
                transposed1[6].write(chunk.get_unchecked_mut(44..));

                transposed2 = self.bf7.exec(transposed2);
                transposed3 = self.bf7.exec(transposed3);

                transposed2[0].write(chunk.get_unchecked_mut(4..));
                transposed2[1].write(chunk.get_unchecked_mut(11..));
                transposed2[2].write(chunk.get_unchecked_mut(18..));
                transposed2[3].write(chunk.get_unchecked_mut(25..));
                transposed2[4].write(chunk.get_unchecked_mut(32..));
                transposed2[5].write(chunk.get_unchecked_mut(39..));
                transposed2[6].write(chunk.get_unchecked_mut(46..));

                transposed3[0].write_lo(chunk.get_unchecked_mut(6..));
                transposed3[1].write_lo(chunk.get_unchecked_mut(13..));
                transposed3[2].write_lo(chunk.get_unchecked_mut(20..));
                transposed3[3].write_lo(chunk.get_unchecked_mut(27..));
                transposed3[4].write_lo(chunk.get_unchecked_mut(34..));
                transposed3[5].write_lo(chunk.get_unchecked_mut(41..));
                transposed3[6].write_lo(chunk.get_unchecked_mut(48..));
            }
        }
        Ok(())
    }
}
impl FftExecutorOutOfPlace<f32> for NeonFcmaButterfly49f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl NeonFcmaButterfly49f {
    #[target_feature(enable = "fcma")]
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
            let mut rows1: [NeonStoreF; 7] = [NeonStoreF::default(); 7];
            let mut rows2: [NeonStoreF; 7] = [NeonStoreF::default(); 7];
            let mut rows3: [NeonStoreF; 7] = [NeonStoreF::default(); 7];
            let mut rows4: [NeonStoreF; 7] = [NeonStoreF::default(); 7];

            for (dst, src) in dst.chunks_exact_mut(49).zip(src.chunks_exact(49)) {
                for i in 0..7 {
                    rows1[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 7..));
                }
                for i in 0..7 {
                    rows2[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 7 + 2..));
                }

                rows1 = self.bf7.exec(rows1);
                rows2 = self.bf7.exec(rows2);

                for i in 1..7 {
                    rows1[i] = NeonStoreF::fcmul_fcma(rows1[i], self.twiddles[i - 1]);
                    rows2[i] = NeonStoreF::fcmul_fcma(rows2[i], self.twiddles[i - 1 + 6]);
                }

                for i in 0..7 {
                    rows3[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 7 + 4..));
                }
                for i in 0..7 {
                    rows4[i] = NeonStoreF::from_complex(src.get_unchecked(i * 7 + 6));
                }

                rows3 = self.bf7.exec(rows3);
                rows4 = self.bf7.exec(rows4);

                for i in 1..7 {
                    rows3[i] = NeonStoreF::fcmul_fcma(rows3[i], self.twiddles[i - 1 + 12]);
                    rows4[i] = NeonStoreF::fcmul_fcma(rows4[i], self.twiddles[i - 1 + 18]);
                }

                let (mut transposed0, mut transposed1, mut transposed2, mut transposed3) =
                    neon_transpose_f32x2_7x7_aos(rows1, rows2, rows3, rows4);

                transposed0 = self.bf7.exec(transposed0);
                transposed1 = self.bf7.exec(transposed1);

                transposed0[0].write(dst);
                transposed0[1].write(dst.get_unchecked_mut(7..));
                transposed0[2].write(dst.get_unchecked_mut(14..));
                transposed0[3].write(dst.get_unchecked_mut(21..));
                transposed0[4].write(dst.get_unchecked_mut(28..));
                transposed0[5].write(dst.get_unchecked_mut(35..));
                transposed0[6].write(dst.get_unchecked_mut(42..));

                transposed1[0].write(dst.get_unchecked_mut(2..));
                transposed1[1].write(dst.get_unchecked_mut(9..));
                transposed1[2].write(dst.get_unchecked_mut(16..));
                transposed1[3].write(dst.get_unchecked_mut(23..));
                transposed1[4].write(dst.get_unchecked_mut(30..));
                transposed1[5].write(dst.get_unchecked_mut(37..));
                transposed1[6].write(dst.get_unchecked_mut(44..));

                transposed2 = self.bf7.exec(transposed2);
                transposed3 = self.bf7.exec(transposed3);

                transposed2[0].write(dst.get_unchecked_mut(4..));
                transposed2[1].write(dst.get_unchecked_mut(11..));
                transposed2[2].write(dst.get_unchecked_mut(18..));
                transposed2[3].write(dst.get_unchecked_mut(25..));
                transposed2[4].write(dst.get_unchecked_mut(32..));
                transposed2[5].write(dst.get_unchecked_mut(39..));
                transposed2[6].write(dst.get_unchecked_mut(46..));

                transposed3[0].write_lo(dst.get_unchecked_mut(6..));
                transposed3[1].write_lo(dst.get_unchecked_mut(13..));
                transposed3[2].write_lo(dst.get_unchecked_mut(20..));
                transposed3[3].write_lo(dst.get_unchecked_mut(27..));
                transposed3[4].write_lo(dst.get_unchecked_mut(34..));
                transposed3[5].write_lo(dst.get_unchecked_mut(41..));
                transposed3[6].write_lo(dst.get_unchecked_mut(48..));
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for NeonFcmaButterfly49f {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_fcma_butterfly!(test_avx_butterfly49, f32, NeonFcmaButterfly49f, 49, 1e-3);
    test_oof_fcma_butterfly!(
        test_oof_avx_butterfly49,
        f32,
        NeonFcmaButterfly49f,
        49,
        1e-3
    );
}
