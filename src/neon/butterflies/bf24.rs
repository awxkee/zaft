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
use crate::neon::transpose::transpose_f32x2_4x4;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

#[inline(always)]
fn transpose_8x3(
    rows0: [NeonStoreF; 3],
    rows1: [NeonStoreF; 3],
    rows2: [NeonStoreF; 3],
    rows3: [NeonStoreF; 3],
) -> ([NeonStoreF; 8], [NeonStoreF; 8]) {
    let a0 = transpose_f32x2_4x4(
        float32x4x2_t(rows0[0].v, rows1[0].v),
        float32x4x2_t(rows0[1].v, rows1[1].v),
        float32x4x2_t(rows0[2].v, rows1[2].v),
        unsafe { float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)) },
    );
    let b0 = transpose_f32x2_4x4(
        float32x4x2_t(rows2[0].v, rows3[0].v),
        float32x4x2_t(rows2[1].v, rows3[1].v),
        float32x4x2_t(rows2[2].v, rows3[2].v),
        unsafe { float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)) },
    );
    (
        [
            // row 0
            NeonStoreF::raw(a0.0.0),
            NeonStoreF::raw(a0.1.0),
            NeonStoreF::raw(a0.2.0),
            NeonStoreF::raw(a0.3.0),
            NeonStoreF::raw(b0.0.0),
            NeonStoreF::raw(b0.1.0),
            NeonStoreF::raw(b0.2.0),
            NeonStoreF::raw(b0.3.0),
        ],
        [
            // row 0
            NeonStoreF::raw(a0.0.1),
            NeonStoreF::raw(a0.1.1),
            NeonStoreF::raw(a0.2.1),
            NeonStoreF::raw(a0.3.1),
            NeonStoreF::raw(b0.0.1),
            NeonStoreF::raw(b0.1.1),
            NeonStoreF::raw(b0.2.1),
            NeonStoreF::raw(b0.3.1),
        ],
    )
}

macro_rules! gen_bf24d {
    ($name: ident, $features: literal, $internal_bf8: ident, $internal_bf3: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf3, $internal_bf8};
        pub(crate) struct $name {
            direction: FftDirection,
            bf8: $internal_bf8,
            bf3: $internal_bf3,
            twiddles: [NeonStoreD; 21],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                let mut twiddles = [NeonStoreD::default(); 21];
                for (x, row) in twiddles.chunks_exact_mut(3).enumerate() {
                    for (y, dst) in row.iter_mut().enumerate() {
                        *dst = NeonStoreD::from_complex(&compute_twiddle(
                            (x + 1) * y,
                            24,
                            fft_direction,
                        ));
                    }
                }
                Self {
                    direction: fft_direction,
                    twiddles,
                    bf8: $internal_bf8::new(fft_direction),
                    bf3: $internal_bf3::new(fft_direction),
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
                24
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if in_place.len() % 24 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    for chunk in in_place.chunks_exact_mut(24) {
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

                        let s0 = self.bf3.exec([u0, u8, u16]);
                        let mut s1 = self.bf3.exec([u1, u9, u17]);
                        let mut s2 = self.bf3.exec([u2, u10, u18]);
                        let mut s3 = self.bf3.exec([u3, u11, u19]);
                        let mut s4 = self.bf3.exec([u4, u12, u20]);
                        let mut s5 = self.bf3.exec([u5, u13, u21]);
                        let mut s6 = self.bf3.exec([u6, u14, u22]);
                        let mut s7 = self.bf3.exec([u7, u15, u23]);
                        for i in 0..3 {
                            s1[i] = NeonStoreD::$mul(s1[i], self.twiddles[i]);
                            s2[i] = NeonStoreD::$mul(s2[i], self.twiddles[3 + i]);
                            s3[i] = NeonStoreD::$mul(s3[i], self.twiddles[6 + i]);
                            s4[i] = NeonStoreD::$mul(s4[i], self.twiddles[9 + i]);
                            s5[i] = NeonStoreD::$mul(s5[i], self.twiddles[12 + i]);
                            s6[i] = NeonStoreD::$mul(s6[i], self.twiddles[15 + i]);
                            s7[i] = NeonStoreD::$mul(s7[i], self.twiddles[18 + i]);
                        }

                        let z0 = self
                            .bf8
                            .exec([s0[0], s1[0], s2[0], s3[0], s4[0], s5[0], s6[0], s7[0]]);
                        let z1 = self
                            .bf8
                            .exec([s0[1], s1[1], s2[1], s3[1], s4[1], s5[1], s6[1], s7[1]]);
                        let z2 = self
                            .bf8
                            .exec([s0[2], s1[2], s2[2], s3[2], s4[2], s5[2], s6[2], s7[2]]);
                        for i in 0..8 {
                            z0[i].write(chunk.get_unchecked_mut(i * 3..));
                            z1[i].write(chunk.get_unchecked_mut(i * 3 + 1..));
                            z2[i].write(chunk.get_unchecked_mut(i * 3 + 2..));
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

macro_rules! gen_bf24f {
    ($name: ident, $features: literal, $internal_bf8: ident, $internal_bf3: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf3, $internal_bf8};
        pub(crate) struct $name {
            direction: FftDirection,
            bf3: $internal_bf3,
            bf8: $internal_bf8,
            twiddles: [NeonStoreF; 8],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(8, 3, fft_direction, 24),
                    bf3: $internal_bf3::new(fft_direction),
                    bf8: $internal_bf8::new(fft_direction),
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
                24
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if in_place.len() % 24 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                    let mut rows1: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                    let mut rows2: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                    let mut rows3: [NeonStoreF; 3] = [NeonStoreF::default(); 3];

                    for chunk in in_place.chunks_exact_mut(24) {
                        // columns
                        {
                            for i in 0..3 {
                                rows0[i] =
                                    NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 8..));
                                rows1[i] =
                                    NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 8 + 2..));
                                rows2[i] =
                                    NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 8 + 4..));
                                rows3[i] =
                                    NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 8 + 6..));
                            }

                            rows0 = self.bf3.exec(rows0);
                            rows1 = self.bf3.exec(rows1);
                            rows2 = self.bf3.exec(rows2);
                            rows3 = self.bf3.exec(rows3);

                            for i in 1..3 {
                                rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                                rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 2]);
                                rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 4]);
                                rows3[i] = NeonStoreF::$mul(rows3[i], self.twiddles[i - 1 + 6]);
                            }

                            let transposed = transpose_8x3(rows0, rows1, rows2, rows3);

                            let q0 = self.bf8.exec(transposed.0);
                            let q1 = self.bf8.exec(transposed.1);

                            for i in 0..8 {
                                q0[i].write(chunk.get_unchecked_mut(i * 3..));
                                q1[i].write_lo(chunk.get_unchecked_mut(i * 3 + 2..));
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf24d!(
    NeonButterfly24d,
    "neon",
    ColumnButterfly8d,
    ColumnButterfly3d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf24d!(
    NeonFcmaButterfly24d,
    "fcma",
    ColumnFcmaButterfly8d,
    ColumnFcmaButterfly3d,
    fcmul_fcma
);

gen_bf24f!(
    NeonButterfly24f,
    "neon",
    ColumnButterfly8f,
    ColumnButterfly3f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf24f!(
    NeonFcmaButterfly24f,
    "fcma",
    ColumnFcmaButterfly8f,
    ColumnFcmaButterfly3f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly24, f32, NeonButterfly24f, 24, 1e-5);
    test_butterfly!(test_neon_butterfly24_f64, f64, NeonButterfly24d, 24, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly24_f64,
        f64,
        NeonFcmaButterfly24d,
        24,
        1e-7
    );
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly24, f32, NeonFcmaButterfly24f, 24, 1e-5);
}
