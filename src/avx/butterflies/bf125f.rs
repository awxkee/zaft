/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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

use crate::avx::butterflies::shared::{
    gen_butterfly_separate_cols_twiddles_f32, gen_butterfly_twiddles_f32,
};
use crate::avx::mixed::{AvxStoreF, ColumnButterfly5f};
use crate::avx::transpose::transpose_4x5;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;
use std::sync::Arc;

pub(crate) struct ColumnButterfly25f {
    bf5: ColumnButterfly5f,
    twiddles: [AvxStoreF; 20],
}

impl ColumnButterfly25f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            twiddles: gen_butterfly_separate_cols_twiddles_f32(5, 5, fft_direction, 25),
            bf5: ColumnButterfly5f::new(fft_direction),
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn exec(&self, src: &[MaybeUninit<Complex<f32>>], dst: &mut [Complex<f32>]) {
        macro_rules! load {
            ($src: expr, $idx: expr) => {{ unsafe { AvxStoreF::from_complex_refu($src.get_unchecked($idx * 5..)) } }};
        }

        macro_rules! store {
            ($v: expr, $idx: expr, $dst: expr) => {{ unsafe { $v.write($dst.get_unchecked_mut($idx * 5..)) } }};
        }

        let mut s0 = self.bf5.exec([
            load!(src, 0),
            load!(src, 5),
            load!(src, 10),
            load!(src, 15),
            load!(src, 20),
        ]);
        for i in 1..5 {
            s0[i] = AvxStoreF::mul_by_complex(s0[i], self.twiddles[i - 1]);
        }
        let mut s1 = self.bf5.exec([
            load!(src, 1),
            load!(src, 6),
            load!(src, 11),
            load!(src, 16),
            load!(src, 21),
        ]);
        for i in 1..5 {
            s1[i] = AvxStoreF::mul_by_complex(s1[i], self.twiddles[i - 1 + 4]);
        }
        let mut s2 = self.bf5.exec([
            load!(src, 2),
            load!(src, 7),
            load!(src, 12),
            load!(src, 17),
            load!(src, 22),
        ]);
        for i in 1..5 {
            s2[i] = AvxStoreF::mul_by_complex(s2[i], self.twiddles[i - 1 + 8]);
        }
        let mut s3 = self.bf5.exec([
            load!(src, 3),
            load!(src, 8),
            load!(src, 13),
            load!(src, 18),
            load!(src, 23),
        ]);
        for i in 1..5 {
            s3[i] = AvxStoreF::mul_by_complex(s3[i], self.twiddles[i - 1 + 12]);
        }
        let mut s4 = self.bf5.exec([
            load!(src, 4),
            load!(src, 9),
            load!(src, 14),
            load!(src, 19),
            load!(src, 24),
        ]);
        for i in 1..5 {
            s4[i] = AvxStoreF::mul_by_complex(s4[i], self.twiddles[i - 1 + 16]);
        }

        let z0 = self.bf5.exec([s0[0], s1[0], s2[0], s3[0], s4[0]]);
        for i in 0..5 {
            store!(z0[i], i * 5, dst);
        }
        let z1 = self.bf5.exec([s0[1], s1[1], s2[1], s3[1], s4[1]]);
        for i in 0..5 {
            store!(z1[i], i * 5 + 1, dst);
        }
        let z2 = self.bf5.exec([s0[2], s1[2], s2[2], s3[2], s4[2]]);
        for i in 0..5 {
            store!(z2[i], i * 5 + 2, dst);
        }
        let z3 = self.bf5.exec([s0[3], s1[3], s2[3], s3[3], s4[3]]);
        for i in 0..5 {
            store!(z3[i], i * 5 + 3, dst);
        }
        let z4 = self.bf5.exec([s0[4], s1[4], s2[4], s3[4], s4[4]]);
        for i in 0..5 {
            store!(z4[i], i * 5 + 4, dst);
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn exech(&self, src: &[MaybeUninit<Complex<f32>>], dst: &mut [Complex<f32>]) {
        macro_rules! load {
            ($src: expr, $idx: expr) => {{ unsafe { AvxStoreF::from_complexu($src.get_unchecked($idx * 5)) } }};
        }

        macro_rules! store {
            ($v: expr, $idx: expr, $dst: expr) => {{ unsafe { $v.write_lo1($dst.get_unchecked_mut($idx * 5..)) } }};
        }

        let mut s0 = self.bf5.exec([
            load!(src, 0),
            load!(src, 5),
            load!(src, 10),
            load!(src, 15),
            load!(src, 20),
        ]);
        for i in 1..5 {
            s0[i] = AvxStoreF::mul_by_complex(s0[i], self.twiddles[i - 1]);
        }
        let mut s1 = self.bf5.exec([
            load!(src, 1),
            load!(src, 6),
            load!(src, 11),
            load!(src, 16),
            load!(src, 21),
        ]);
        for i in 1..5 {
            s1[i] = AvxStoreF::mul_by_complex(s1[i], self.twiddles[i - 1 + 4]);
        }
        let mut s2 = self.bf5.exec([
            load!(src, 2),
            load!(src, 7),
            load!(src, 12),
            load!(src, 17),
            load!(src, 22),
        ]);
        for i in 1..5 {
            s2[i] = AvxStoreF::mul_by_complex(s2[i], self.twiddles[i - 1 + 8]);
        }
        let mut s3 = self.bf5.exec([
            load!(src, 3),
            load!(src, 8),
            load!(src, 13),
            load!(src, 18),
            load!(src, 23),
        ]);
        for i in 1..5 {
            s3[i] = AvxStoreF::mul_by_complex(s3[i], self.twiddles[i - 1 + 12]);
        }
        let mut s4 = self.bf5.exec([
            load!(src, 4),
            load!(src, 9),
            load!(src, 14),
            load!(src, 19),
            load!(src, 24),
        ]);
        for i in 1..5 {
            s4[i] = AvxStoreF::mul_by_complex(s4[i], self.twiddles[i - 1 + 16]);
        }

        let z0 = self.bf5.exec([s0[0], s1[0], s2[0], s3[0], s4[0]]);
        for i in 0..5 {
            store!(z0[i], i * 5, dst);
        }
        let z1 = self.bf5.exec([s0[1], s1[1], s2[1], s3[1], s4[1]]);
        for i in 0..5 {
            store!(z1[i], i * 5 + 1, dst);
        }
        let z2 = self.bf5.exec([s0[2], s1[2], s2[2], s3[2], s4[2]]);
        for i in 0..5 {
            store!(z2[i], i * 5 + 2, dst);
        }
        let z3 = self.bf5.exec([s0[3], s1[3], s2[3], s3[3], s4[3]]);
        for i in 0..5 {
            store!(z3[i], i * 5 + 3, dst);
        }
        let z4 = self.bf5.exec([s0[4], s1[4], s2[4], s3[4], s4[4]]);
        for i in 0..5 {
            store!(z4[i], i * 5 + 4, dst);
        }
    }
}

pub(crate) struct AvxButterfly125f {
    direction: FftDirection,
    bf25: ColumnButterfly25f,
    twiddles: [AvxStoreF; 28],
}

impl AvxButterfly125f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(25, 5, fft_direction, 125),
            bf25: ColumnButterfly25f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly125f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        125
    }
}

impl AvxButterfly125f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(125) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [AvxStoreF; 5] = [AvxStoreF::zero(); 5];
            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 125];

            for chunk in in_place.chunks_exact_mut(125) {
                // columns
                for k in 0..6 {
                    for i in 0..5 {
                        rows[i] =
                            AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 25 + k * 4..));
                    }

                    rows = self.bf25.bf5.exec(rows);

                    for i in 1..5 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 4 * k]);
                    }

                    let transposed = transpose_4x5(rows);

                    transposed[0].write_u(scratch.get_unchecked_mut(k * 4 * 5..));
                    transposed[1].write_u(scratch.get_unchecked_mut((k * 4 + 1) * 5..));
                    transposed[2].write_u(scratch.get_unchecked_mut((k * 4 + 2) * 5..));
                    transposed[3].write_u(scratch.get_unchecked_mut((k * 4 + 3) * 5..));

                    transposed[4].write_lo1u(scratch.get_unchecked_mut(k * 4 * 5 + 4..));
                    transposed[5].write_lo1u(scratch.get_unchecked_mut((k * 4 + 1) * 5 + 4..));
                    transposed[6].write_lo1u(scratch.get_unchecked_mut((k * 4 + 2) * 5 + 4..));
                    transposed[7].write_lo1u(scratch.get_unchecked_mut((k * 4 + 3) * 5 + 4..));
                }

                {
                    let k = 6;
                    for i in 0..5 {
                        rows[i] = AvxStoreF::from_complex(chunk.get_unchecked(i * 25 + k * 4));
                    }

                    rows = self.bf25.bf5.exec(rows);

                    for i in 1..5 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 4 * k]);
                    }

                    let transposed = transpose_4x5(rows);

                    transposed[0].write_u(scratch.get_unchecked_mut(k * 4 * 5..));
                    transposed[4].write_lo1u(scratch.get_unchecked_mut(k * 4 * 5 + 4..));
                }

                // rows

                self.bf25.exec(&scratch, chunk);

                self.bf25
                    .exech(scratch.get_unchecked(4..), chunk.get_unchecked_mut(4..));
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly125f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly125f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(125) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(125) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        let mut rows: [AvxStoreF; 5] = [AvxStoreF::zero(); 5];
        let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 125];

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(125).zip(src.chunks_exact(125)) {
                // columns
                for k in 0..6 {
                    for i in 0..5 {
                        rows[i] = AvxStoreF::from_complex_ref(src.get_unchecked(i * 25 + k * 4..));
                    }

                    rows = self.bf25.bf5.exec(rows);

                    for i in 1..5 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 4 * k]);
                    }

                    let transposed = transpose_4x5(rows);

                    transposed[0].write_u(scratch.get_unchecked_mut(k * 4 * 5..));
                    transposed[1].write_u(scratch.get_unchecked_mut((k * 4 + 1) * 5..));
                    transposed[2].write_u(scratch.get_unchecked_mut((k * 4 + 2) * 5..));
                    transposed[3].write_u(scratch.get_unchecked_mut((k * 4 + 3) * 5..));

                    transposed[4].write_lo1u(scratch.get_unchecked_mut(k * 4 * 5 + 4..));
                    transposed[5].write_lo1u(scratch.get_unchecked_mut((k * 4 + 1) * 5 + 4..));
                    transposed[6].write_lo1u(scratch.get_unchecked_mut((k * 4 + 2) * 5 + 4..));
                    transposed[7].write_lo1u(scratch.get_unchecked_mut((k * 4 + 3) * 5 + 4..));
                }

                {
                    let k = 6;
                    for i in 0..5 {
                        rows[i] = AvxStoreF::from_complex(src.get_unchecked(i * 25 + k * 4));
                    }

                    rows = self.bf25.bf5.exec(rows);

                    for i in 1..5 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 4 * k]);
                    }

                    let transposed = transpose_4x5(rows);

                    transposed[0].write_u(scratch.get_unchecked_mut(k * 4 * 5..));
                    transposed[4].write_lo1u(scratch.get_unchecked_mut(k * 4 * 5 + 4..));
                }

                // rows

                self.bf25.exec(&scratch, dst);

                self.bf25
                    .exech(scratch.get_unchecked(4..), dst.get_unchecked_mut(4..));
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly125f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_avx_butterfly125, f32, AvxButterfly125f, 125, 1e-3);
    test_oof_butterfly!(test_oof_avx_butterfly125, f32, AvxButterfly125f, 125, 1e-3);
}
