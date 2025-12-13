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
use crate::neon::butterflies::NeonButterfly;
use crate::neon::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

macro_rules! gen_bf14d {
    ($name: ident, $features: literal, $internal_bf7: ident) => {
        use crate::neon::mixed::$internal_bf7;
        pub(crate) struct $name {
            direction: FftDirection,
            bf7: $internal_bf7,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf7: $internal_bf7::new(fft_direction),
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
                14
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(14) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    for chunk in in_place.chunks_exact_mut(14) {
                        let u0 = vld1q_f64(chunk.as_ptr().cast());
                        let u1 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());
                        let u2 = vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast());
                        let u3 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                        let u4 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                        let u5 = vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast());
                        let u6 = vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast());
                        let u7 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                        let u8 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                        let u9 = vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast());
                        let u10 = vld1q_f64(chunk.get_unchecked(12..).as_ptr().cast());
                        let u11 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                        let u12 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                        let u13 = vld1q_f64(chunk.get_unchecked(13..).as_ptr().cast());

                        let (u0, u1) = NeonButterfly::butterfly2_f64(u0, u1);
                        let (u2, u3) = NeonButterfly::butterfly2_f64(u2, u3);
                        let (u4, u5) = NeonButterfly::butterfly2_f64(u4, u5);
                        let (u6, u7) = NeonButterfly::butterfly2_f64(u6, u7);
                        let (u8, u9) = NeonButterfly::butterfly2_f64(u8, u9);
                        let (u10, u11) = NeonButterfly::butterfly2_f64(u10, u11);
                        let (u12, u13) = NeonButterfly::butterfly2_f64(u12, u13);

                        // Outer 7-point butterflies
                        let [y0, y2, y4, y6, y8, y10, y12] = self.bf7.exec([
                            NeonStoreD::raw(u0),
                            NeonStoreD::raw(u2),
                            NeonStoreD::raw(u4),
                            NeonStoreD::raw(u6),
                            NeonStoreD::raw(u8),
                            NeonStoreD::raw(u10),
                            NeonStoreD::raw(u12),
                        ]); // (v0, v1, v2, v3, v4, v5, v6)
                        let [y7, y9, y11, y13, y1, y3, y5] = self.bf7.exec([
                            NeonStoreD::raw(u1),
                            NeonStoreD::raw(u3),
                            NeonStoreD::raw(u5),
                            NeonStoreD::raw(u7),
                            NeonStoreD::raw(u9),
                            NeonStoreD::raw(u11),
                            NeonStoreD::raw(u13),
                        ]); // (v7, v8, v9, v10, v11, v12, v13)

                        vst1q_f64(chunk.as_mut_ptr().cast(), y0.v);
                        vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1.v);
                        vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2.v);
                        vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3.v);
                        vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4.v);
                        vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y5.v);
                        vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y6.v);
                        vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y7.v);
                        vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y8.v);
                        vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y9.v);
                        vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10.v);
                        vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), y11.v);
                        vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y12.v);
                        vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), y13.v);
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf14d!(NeonButterfly14d, "neon", ColumnButterfly7d);
#[cfg(feature = "fcma")]
gen_bf14d!(NeonFcmaButterfly14d, "fcma", ColumnFcmaButterfly7d);

#[inline(always)]
pub(crate) fn transpose_f32x2_7x2(
    rows0: [NeonStoreF; 2],
    rows1: [NeonStoreF; 2],
    rows2: [NeonStoreF; 2],
    rows3: [NeonStoreF; 2],
) -> [NeonStoreF; 7] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[0].v, rows2[1].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows3[0].v, rows3[1].v));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
    ]
}

use crate::neon::mixed::ColumnButterfly2f;

macro_rules! gen_bf15f {
    ($name: ident, $features: literal, $internal_bf7: ident, $internal_bf2: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf7;
        pub(crate) struct $name {
            direction: FftDirection,
            bf2: $internal_bf2,
            bf7: $internal_bf7,
            twiddles: [NeonStoreF; 4],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(7, 2, fft_direction, 14),
                    bf2: $internal_bf2::new(fft_direction),
                    bf7: $internal_bf7::new(fft_direction),
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
                14
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(14) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 2] = [NeonStoreF::default(); 2];
                    let mut rows1: [NeonStoreF; 2] = [NeonStoreF::default(); 2];
                    let mut rows2: [NeonStoreF; 2] = [NeonStoreF::default(); 2];
                    let mut rows3: [NeonStoreF; 2] = [NeonStoreF::default(); 2];

                    for chunk in in_place.chunks_exact_mut(14) {
                        // columns
                        for i in 0..2 {
                            rows0[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 7..));
                            rows1[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 7 + 2..));
                            rows2[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 7 + 4..));
                            rows3[i] = NeonStoreF::from_complex(chunk.get_unchecked(i * 7 + 6));
                        }

                        rows0 = self.bf2.exec(rows0);
                        rows1 = self.bf2.exec(rows1);
                        rows2 = self.bf2.exec(rows2);
                        rows3 = self.bf2.exec(rows3);

                        rows0[1] = NeonStoreF::$mul(rows0[1], self.twiddles[0]);
                        rows1[1] = NeonStoreF::$mul(rows1[1], self.twiddles[1]);
                        rows2[1] = NeonStoreF::$mul(rows2[1], self.twiddles[2]);
                        rows3[1] = NeonStoreF::$mul(rows3[1], self.twiddles[3]);

                        let transposed = transpose_f32x2_7x2(rows0, rows1, rows2, rows3);

                        let q0 = self.bf7.exec(transposed);

                        for i in 0..7 {
                            q0[i].write(chunk.get_unchecked_mut(i * 2..));
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf15f!(
    NeonButterfly14f,
    "neon",
    ColumnButterfly7f,
    ColumnButterfly2f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf15f!(
    NeonFcmaButterfly14f,
    "fcma",
    ColumnFcmaButterfly7f,
    ColumnButterfly2f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly14, f32, NeonButterfly14f, 14, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly14, f32, NeonFcmaButterfly14f, 14, 1e-5);
    test_butterfly!(test_neon_butterfly14_f64, f64, NeonButterfly14d, 14, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly14_f64,
        f64,
        NeonFcmaButterfly14d,
        14,
        1e-7
    );
}
