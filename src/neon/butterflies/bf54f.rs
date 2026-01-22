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
use crate::neon::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::{transpose_2x2, transpose_f32x2_4x4};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::{float32x4x2_t, vdupq_n_f32};

#[inline(always)]
pub(crate) fn neon_transpose_f32x2_9x6_aos(
    rows1: [NeonStoreF; 6],
    rows2: [NeonStoreF; 6],
    rows3: [NeonStoreF; 6],
    rows4: [NeonStoreF; 6],
    rows5: [NeonStoreF; 6],
) -> ([NeonStoreF; 9], [NeonStoreF; 9], [NeonStoreF; 9]) {
    unsafe {
        let tl = transpose_f32x2_4x4(
            float32x4x2_t(rows1[0].v, rows2[0].v),
            float32x4x2_t(rows1[1].v, rows2[1].v),
            float32x4x2_t(rows1[2].v, rows2[2].v),
            float32x4x2_t(rows1[3].v, rows2[3].v),
        );

        let bl = transpose_f32x2_4x4(
            float32x4x2_t(rows1[4].v, rows2[4].v),
            float32x4x2_t(rows1[5].v, rows2[5].v),
            float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
            float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
        );

        let tr = transpose_f32x2_4x4(
            float32x4x2_t(rows3[0].v, rows4[0].v),
            float32x4x2_t(rows3[1].v, rows4[1].v),
            float32x4x2_t(rows3[2].v, rows4[2].v),
            float32x4x2_t(rows3[3].v, rows4[3].v),
        );

        let br = transpose_f32x2_4x4(
            float32x4x2_t(rows3[4].v, rows4[4].v),
            float32x4x2_t(rows3[5].v, rows4[5].v),
            float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
            float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
        );

        let far_top = transpose_f32x2_4x4(
            float32x4x2_t(rows5[0].v, vdupq_n_f32(0.)),
            float32x4x2_t(rows5[1].v, vdupq_n_f32(0.)),
            float32x4x2_t(rows5[2].v, vdupq_n_f32(0.)),
            float32x4x2_t(rows5[3].v, vdupq_n_f32(0.)),
        );

        let far_bottom = transpose_2x2([rows5[4], rows5[5]]);

        // Reassemble left 6 rows (first 4 columns)
        let output_left = [
            tl.0, tl.1, tl.2, tl.3, // top 4 rows
            tr.0, tr.1, tr.2, tr.3, far_top.0, // bottom 2 rows
        ];

        // Reassemble right 6 rows (last 2 columns)
        let output_right = [
            bl.0, bl.1, bl.2, bl.3, // top 4 rows
            br.0, br.1, br.2, br.3,
        ];

        (
            [
                NeonStoreF::raw(output_left[0].0),
                NeonStoreF::raw(output_left[1].0),
                NeonStoreF::raw(output_left[2].0),
                NeonStoreF::raw(output_left[3].0),
                NeonStoreF::raw(output_left[4].0),
                NeonStoreF::raw(output_left[5].0),
                NeonStoreF::raw(output_left[6].0),
                NeonStoreF::raw(output_left[7].0),
                NeonStoreF::raw(output_left[8].0),
            ],
            [
                NeonStoreF::raw(output_left[0].1),
                NeonStoreF::raw(output_left[1].1),
                NeonStoreF::raw(output_left[2].1),
                NeonStoreF::raw(output_left[3].1),
                NeonStoreF::raw(output_left[4].1),
                NeonStoreF::raw(output_left[5].1),
                NeonStoreF::raw(output_left[6].1),
                NeonStoreF::raw(output_left[7].1),
                NeonStoreF::raw(output_left[8].1),
            ],
            [
                NeonStoreF::raw(output_right[0].0),
                NeonStoreF::raw(output_right[1].0),
                NeonStoreF::raw(output_right[2].0),
                NeonStoreF::raw(output_right[3].0),
                NeonStoreF::raw(output_right[4].0),
                NeonStoreF::raw(output_right[5].0),
                NeonStoreF::raw(output_right[6].0),
                NeonStoreF::raw(output_right[7].0),
                far_bottom[0],
            ],
        )
    }
}

macro_rules! gen_bf54f {
    ($name: ident, $feature: literal, $internal_bf6: ident, $internal_bf9: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf6;
        use crate::neon::mixed::$internal_bf9;
        pub(crate) struct $name {
            direction: FftDirection,
            bf6: $internal_bf6,
            bf9: $internal_bf9,
            twiddles: [NeonStoreF; 25],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(9, 6, fft_direction, 54),
                    bf6: $internal_bf6::new(fft_direction),
                    bf9: $internal_bf9::new(fft_direction),
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
                54
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(54) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 6] = [NeonStoreF::default(); 6];
                    let mut rows1: [NeonStoreF; 6] = [NeonStoreF::default(); 6];
                    let mut rows2: [NeonStoreF; 6] = [NeonStoreF::default(); 6];
                    let mut rows3: [NeonStoreF; 6] = [NeonStoreF::default(); 6];
                    let mut rows4: [NeonStoreF; 6] = [NeonStoreF::default(); 6];

                    for chunk in in_place.chunks_exact_mut(54) {
                        // columns

                        // columns 0-2
                        for i in 0..6 {
                            rows0[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 9..));
                        }

                        rows0 = self.bf6.exec(rows0);

                        for i in 1..6 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                        }

                        // columns 2-4
                        for i in 0..6 {
                            rows1[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 9 + 2..));
                        }

                        rows1 = self.bf6.exec(rows1);

                        for i in 1..6 {
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 5]);
                        }

                        // columns 4-6

                        for i in 0..6 {
                            rows2[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 9 + 4..));
                        }

                        rows2 = self.bf6.exec(rows2);

                        for i in 1..6 {
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 10]);
                        }

                        // columns 6-8

                        for i in 0..6 {
                            rows3[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 9 + 6..));
                        }

                        rows3 = self.bf6.exec(rows3);

                        for i in 1..6 {
                            rows3[i] = NeonStoreF::$mul(rows3[i], self.twiddles[i - 1 + 15]);
                        }

                        // columns 8-9

                        for i in 0..6 {
                            rows4[i] = NeonStoreF::from_complex(chunk.get_unchecked(i * 9 + 8));
                        }

                        rows4 = self.bf6.exec(rows4);

                        for i in 1..6 {
                            rows4[i] = NeonStoreF::$mul(rows4[i], self.twiddles[i - 1 + 20]);
                        }

                        let (mut t0, mut t1, mut t2) =
                            neon_transpose_f32x2_9x6_aos(rows0, rows1, rows2, rows3, rows4);

                        // rows

                        t0 = self.bf9.exec(t0);

                        for i in 0..9 {
                            t0[i].write(chunk.get_unchecked_mut(i * 6..));
                        }

                        t1 = self.bf9.exec(t1);

                        for i in 0..9 {
                            t1[i].write(chunk.get_unchecked_mut(i * 6 + 2..));
                        }

                        t2 = self.bf9.exec(t2);

                        for i in 0..9 {
                            t2[i].write(chunk.get_unchecked_mut(i * 6 + 4..));
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf54f!(
    NeonButterfly54f,
    "neon",
    ColumnButterfly6f,
    ColumnButterfly9f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf54f!(
    NeonFcmaButterfly54f,
    "fcma",
    ColumnFcmaButterfly6f,
    ColumnFcmaButterfly9f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly54, f32, NeonButterfly54f, 54, 1e-3);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly54, f32, NeonFcmaButterfly54f, 54, 1e-3);
}
