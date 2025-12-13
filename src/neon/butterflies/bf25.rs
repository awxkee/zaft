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
use crate::neon::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::neon::transpose::transpose_2x5;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;
use std::sync::Arc;

macro_rules! gen_bf25d {
    ($name: ident, $features: literal, $internal_bf: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf5: $internal_bf,
            twiddles: [NeonStoreD; 20],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                let mut twiddles = [NeonStoreD::default(); 20];
                for (x, row) in twiddles.chunks_exact_mut(5).enumerate() {
                    for (y, dst) in row.iter_mut().enumerate() {
                        *dst = NeonStoreD::from_complex(&compute_twiddle(
                            (x + 1) * y,
                            25,
                            fft_direction,
                        ));
                    }
                }
                Self {
                    direction: fft_direction,
                    twiddles,
                    bf5: $internal_bf::new(fft_direction),
                }
            }
        }

        impl FftExecutor<f64> for $name {
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

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(25) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    for chunk in in_place.chunks_exact_mut(25) {
                        let u0 = NeonStoreD::raw(vld1q_f64(chunk.as_ptr().cast()));
                        let u1 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast()));
                        let u2 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast()));
                        let u3 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast()));
                        let u4 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast()));
                        let u5 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast()));
                        let u6 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast()));
                        let u7 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast()));
                        let u8 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast()));
                        let u9 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast()));
                        let u10 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast()));
                        let u11 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast()));
                        let u12 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(12..).as_ptr().cast()));
                        let u13 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(13..).as_ptr().cast()));
                        let u14 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(14..).as_ptr().cast()));
                        let u15 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(15..).as_ptr().cast()));
                        let u16 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(16..).as_ptr().cast()));
                        let u17 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(17..).as_ptr().cast()));
                        let u18 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(18..).as_ptr().cast()));
                        let u19 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(19..).as_ptr().cast()));
                        let u20 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(20..).as_ptr().cast()));
                        let u21 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(21..).as_ptr().cast()));
                        let u22 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(22..).as_ptr().cast()));
                        let u23 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(23..).as_ptr().cast()));
                        let u24 =
                            NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(24..).as_ptr().cast()));

                        let s0 = self.bf5.exec([u0, u5, u10, u15, u20]);
                        let mut s1 = self.bf5.exec([u1, u6, u11, u16, u21]);
                        let mut s2 = self.bf5.exec([u2, u7, u12, u17, u22]);
                        for i in 0..5 {
                            s1[i] = NeonStoreD::$mul(s1[i], self.twiddles[i]);
                            s2[i] = NeonStoreD::$mul(s2[i], self.twiddles[5 + i]);
                        }
                        let mut s3 = self.bf5.exec([u3, u8, u13, u18, u23]);
                        let mut s4 = self.bf5.exec([u4, u9, u14, u19, u24]);
                        for i in 0..5 {
                            s3[i] = NeonStoreD::$mul(s3[i], self.twiddles[10 + i]);
                            s4[i] = NeonStoreD::$mul(s4[i], self.twiddles[15 + i]);
                        }

                        let z0 = self.bf5.exec([s0[0], s1[0], s2[0], s3[0], s4[0]]);
                        let z1 = self.bf5.exec([s0[1], s1[1], s2[1], s3[1], s4[1]]);
                        let z2 = self.bf5.exec([s0[2], s1[2], s2[2], s3[2], s4[2]]);
                        for i in 0..5 {
                            z0[i].write(chunk.get_unchecked_mut(i * 5..));
                            z1[i].write(chunk.get_unchecked_mut(i * 5 + 1..));
                            z2[i].write(chunk.get_unchecked_mut(i * 5 + 2..));
                        }
                        let z3 = self.bf5.exec([s0[3], s1[3], s2[3], s3[3], s4[3]]);
                        let z4 = self.bf5.exec([s0[4], s1[4], s2[4], s3[4], s4[4]]);
                        for i in 0..5 {
                            z3[i].write(chunk.get_unchecked_mut(i * 5 + 3..));
                            z4[i].write(chunk.get_unchecked_mut(i * 5 + 4..));
                        }
                    }
                }
                Ok(())
            }
        }

        impl FftExecutorOutOfPlace<f64> for $name {
            fn execute_out_of_place(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_out_of_place_impl(src, dst) }
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_out_of_place_impl(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(25) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(25) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    for (dst, src) in dst.chunks_exact_mut(25).zip(src.chunks_exact(25)) {
                        let u0 = NeonStoreD::raw(vld1q_f64(src.as_ptr().cast()));
                        let u1 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(1..).as_ptr().cast()));
                        let u2 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(2..).as_ptr().cast()));
                        let u3 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(3..).as_ptr().cast()));
                        let u4 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(4..).as_ptr().cast()));
                        let u5 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(5..).as_ptr().cast()));
                        let u6 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(6..).as_ptr().cast()));
                        let u7 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(7..).as_ptr().cast()));
                        let u8 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(8..).as_ptr().cast()));
                        let u9 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(9..).as_ptr().cast()));
                        let u10 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(10..).as_ptr().cast()));
                        let u11 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(11..).as_ptr().cast()));
                        let u12 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(12..).as_ptr().cast()));
                        let u13 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(13..).as_ptr().cast()));
                        let u14 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(14..).as_ptr().cast()));
                        let u15 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(15..).as_ptr().cast()));
                        let u16 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(16..).as_ptr().cast()));
                        let u17 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(17..).as_ptr().cast()));
                        let u18 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(18..).as_ptr().cast()));
                        let u19 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(19..).as_ptr().cast()));
                        let u20 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(20..).as_ptr().cast()));
                        let u21 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(21..).as_ptr().cast()));
                        let u22 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(22..).as_ptr().cast()));
                        let u23 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(23..).as_ptr().cast()));
                        let u24 =
                            NeonStoreD::raw(vld1q_f64(src.get_unchecked(24..).as_ptr().cast()));

                        let s0 = self.bf5.exec([u0, u5, u10, u15, u20]);
                        let mut s1 = self.bf5.exec([u1, u6, u11, u16, u21]);
                        let mut s2 = self.bf5.exec([u2, u7, u12, u17, u22]);
                        for i in 0..5 {
                            s1[i] = NeonStoreD::$mul(s1[i], self.twiddles[i]);
                            s2[i] = NeonStoreD::$mul(s2[i], self.twiddles[5 + i]);
                        }
                        let mut s3 = self.bf5.exec([u3, u8, u13, u18, u23]);
                        let mut s4 = self.bf5.exec([u4, u9, u14, u19, u24]);
                        for i in 0..5 {
                            s3[i] = NeonStoreD::$mul(s3[i], self.twiddles[10 + i]);
                            s4[i] = NeonStoreD::$mul(s4[i], self.twiddles[15 + i]);
                        }

                        let z0 = self.bf5.exec([s0[0], s1[0], s2[0], s3[0], s4[0]]);
                        let z1 = self.bf5.exec([s0[1], s1[1], s2[1], s3[1], s4[1]]);
                        let z2 = self.bf5.exec([s0[2], s1[2], s2[2], s3[2], s4[2]]);
                        for i in 0..5 {
                            z0[i].write(dst.get_unchecked_mut(i * 5..));
                            z1[i].write(dst.get_unchecked_mut(i * 5 + 1..));
                            z2[i].write(dst.get_unchecked_mut(i * 5 + 2..));
                        }
                        let z3 = self.bf5.exec([s0[3], s1[3], s2[3], s3[3], s4[3]]);
                        let z4 = self.bf5.exec([s0[4], s1[4], s2[4], s3[4], s4[4]]);
                        for i in 0..5 {
                            z3[i].write(dst.get_unchecked_mut(i * 5 + 3..));
                            z4[i].write(dst.get_unchecked_mut(i * 5 + 4..));
                        }
                    }
                }
                Ok(())
            }
        }

        impl CompositeFftExecutor<f64> for $name {
            fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
                self
            }
        }
    };
}

gen_bf25d!(NeonButterfly25d, "neon", ColumnButterfly5d, mul_by_complex);
#[cfg(feature = "fcma")]
gen_bf25d!(
    NeonFcmaButterfly25d,
    "fcma",
    ColumnFcmaButterfly5d,
    fcmul_fcma
);

#[inline]
fn transpose_5x5_f32(
    rows0: [NeonStoreF; 5],
    rows1: [NeonStoreF; 5],
    rows2: [NeonStoreF; 5],
) -> ([NeonStoreF; 5], [NeonStoreF; 5], [NeonStoreF; 5]) {
    let transposed00 = transpose_2x5(rows0);
    let transposed01 = transpose_2x5(rows1);
    let transposed10 = transpose_2x5(rows2);

    (
        [
            transposed00[0],
            transposed00[1],
            transposed01[0],
            transposed01[1],
            transposed10[0],
        ],
        [
            transposed00[2],
            transposed00[3],
            transposed01[2],
            transposed01[3],
            transposed10[2],
        ],
        [
            transposed00[4],
            transposed00[5],
            transposed01[4],
            transposed01[5],
            transposed10[4],
        ],
    )
}

macro_rules! gen_bf25f {
    ($name: ident, $features: literal, $internal_bf: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf5: $internal_bf,
            twiddles: [NeonStoreF; 12],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(5, 5, fft_direction, 25),
                    bf5: $internal_bf::new(fft_direction),
                }
            }
        }

        impl FftExecutor<f32> for $name {
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

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(25) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0 = [NeonStoreF::default(); 5];
                    let mut rows1 = [NeonStoreF::default(); 5];
                    let mut rows2 = [NeonStoreF::default(); 5];

                    for chunk in in_place.chunks_exact_mut(25) {
                        for i in 0..5 {
                            rows0[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 5..));
                            rows1[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 5 + 2..));
                            rows2[i] = NeonStoreF::from_complex(chunk.get_unchecked(i * 5 + 4));
                        }

                        rows0 = self.bf5.exec(rows0);
                        rows1 = self.bf5.exec(rows1);
                        rows2 = self.bf5.exec(rows2);

                        for i in 1..5 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 4]);
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 8]);
                        }

                        let (mut q0, mut q1, mut q2) = transpose_5x5_f32(rows0, rows1, rows2);

                        q0 = self.bf5.exec(q0);
                        q1 = self.bf5.exec(q1);
                        q2 = self.bf5.exec(q2);

                        for i in 0..5 {
                            q0[i].write(chunk.get_unchecked_mut(i * 5..));
                            q1[i].write(chunk.get_unchecked_mut(i * 5 + 2..));
                            q2[i].write_lo(chunk.get_unchecked_mut(i * 5 + 4..));
                        }

                        q0[0].write(chunk);
                        q0[1].write(chunk.get_unchecked_mut(5..));
                        q0[2].write(chunk.get_unchecked_mut(10..));
                        q0[3].write(chunk.get_unchecked_mut(15..));
                        q0[4].write(chunk.get_unchecked_mut(20..));

                        q1[0].write(chunk.get_unchecked_mut(2..));
                        q1[1].write(chunk.get_unchecked_mut(7..));
                        q1[2].write(chunk.get_unchecked_mut(12..));
                        q1[3].write(chunk.get_unchecked_mut(17..));
                        q1[4].write(chunk.get_unchecked_mut(22..));

                        q2[0].write_lo(chunk.get_unchecked_mut(4..));
                        q2[1].write_lo(chunk.get_unchecked_mut(9..));
                        q2[2].write_lo(chunk.get_unchecked_mut(14..));
                        q2[3].write_lo(chunk.get_unchecked_mut(19..));
                        q2[4].write_lo(chunk.get_unchecked_mut(24..));
                    }
                }
                Ok(())
            }
        }

        impl FftExecutorOutOfPlace<f32> for $name {
            fn execute_out_of_place(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_out_of_place_impl(src, dst) }
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
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
                    let mut rows0 = [NeonStoreF::default(); 5];
                    let mut rows1 = [NeonStoreF::default(); 5];
                    let mut rows2 = [NeonStoreF::default(); 5];

                    for (dst, src) in dst.chunks_exact_mut(25).zip(src.chunks_exact(25)) {
                        for i in 0..5 {
                            rows0[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 5..));
                            rows1[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 5 + 2..));
                            rows2[i] = NeonStoreF::from_complex(src.get_unchecked(i * 5 + 4));
                        }

                        rows0 = self.bf5.exec(rows0);
                        rows1 = self.bf5.exec(rows1);
                        rows2 = self.bf5.exec(rows2);

                        for i in 1..5 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 4]);
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 8]);
                        }

                        let (mut q0, mut q1, mut q2) = transpose_5x5_f32(rows0, rows1, rows2);

                        q0 = self.bf5.exec(q0);
                        q1 = self.bf5.exec(q1);
                        q2 = self.bf5.exec(q2);

                        q0[0].write(dst);
                        q0[1].write(dst.get_unchecked_mut(5..));
                        q0[2].write(dst.get_unchecked_mut(10..));
                        q0[3].write(dst.get_unchecked_mut(15..));
                        q0[4].write(dst.get_unchecked_mut(20..));

                        q1[0].write(dst.get_unchecked_mut(2..));
                        q1[1].write(dst.get_unchecked_mut(7..));
                        q1[2].write(dst.get_unchecked_mut(12..));
                        q1[3].write(dst.get_unchecked_mut(17..));
                        q1[4].write(dst.get_unchecked_mut(22..));

                        q2[0].write_lo(dst.get_unchecked_mut(4..));
                        q2[1].write_lo(dst.get_unchecked_mut(9..));
                        q2[2].write_lo(dst.get_unchecked_mut(14..));
                        q2[3].write_lo(dst.get_unchecked_mut(19..));
                        q2[4].write_lo(dst.get_unchecked_mut(24..));
                    }
                }
                Ok(())
            }
        }

        impl CompositeFftExecutor<f32> for $name {
            fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
                self
            }
        }
    };
}

gen_bf25f!(NeonButterfly25f, "neon", ColumnButterfly5f, mul_by_complex);
#[cfg(feature = "fcma")]
gen_bf25f!(
    NeonFcmaButterfly25f,
    "fcma",
    ColumnFcmaButterfly5f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly25, f32, NeonButterfly25f, 25, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly25, f32, NeonFcmaButterfly25f, 25, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly25_f64,
        f64,
        NeonFcmaButterfly25d,
        25,
        1e-5
    );
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly25,
        f32,
        NeonFcmaButterfly25f,
        25,
        1e-5
    );
    test_butterfly!(test_neon_butterfly25_f64, f64, NeonButterfly25d, 25, 1e-7);
    test_oof_butterfly!(test_oof_butterfly25, f32, NeonButterfly25f, 25, 1e-5);
    test_oof_butterfly!(test_oof_butterfly25_f64, f64, NeonButterfly25d, 25, 1e-9);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly25_f64,
        f64,
        NeonButterfly25d,
        25,
        1e-9
    );
}
