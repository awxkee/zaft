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
use crate::neon::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

macro_rules! gen_bf20d {
    ($name: ident, $features: literal, $internal_bf5: ident, $internal_bf4: ident) => {
        use crate::neon::mixed::{$internal_bf4, $internal_bf5};
        pub(crate) struct $name {
            direction: FftDirection,
            bf5: $internal_bf5,
            bf4: $internal_bf4,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf5: $internal_bf5::new(fft_direction),
                    bf4: $internal_bf4::new(fft_direction),
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
                20
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(20) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    for chunk in in_place.chunks_exact_mut(20) {
                        let u0 = NeonStoreD::from_complex_ref(chunk);
                        let u5 = NeonStoreD::from_complex_ref(chunk.get_unchecked(1..));
                        let u10 = NeonStoreD::from_complex_ref(chunk.get_unchecked(2..));
                        let u15 = NeonStoreD::from_complex_ref(chunk.get_unchecked(3..));
                        let u16 = NeonStoreD::from_complex_ref(chunk.get_unchecked(4..));

                        let u1 = NeonStoreD::from_complex_ref(chunk.get_unchecked(5..));
                        let u6 = NeonStoreD::from_complex_ref(chunk.get_unchecked(6..));
                        let u11 = NeonStoreD::from_complex_ref(chunk.get_unchecked(7..));
                        let u12 = NeonStoreD::from_complex_ref(chunk.get_unchecked(8..));
                        let u17 = NeonStoreD::from_complex_ref(chunk.get_unchecked(9..));

                        let u2 = NeonStoreD::from_complex_ref(chunk.get_unchecked(10..));
                        let u7 = NeonStoreD::from_complex_ref(chunk.get_unchecked(11..));
                        let u8 = NeonStoreD::from_complex_ref(chunk.get_unchecked(12..));
                        let u13 = NeonStoreD::from_complex_ref(chunk.get_unchecked(13..));
                        let u18 = NeonStoreD::from_complex_ref(chunk.get_unchecked(14..));

                        let u3 = NeonStoreD::from_complex_ref(chunk.get_unchecked(15..));
                        let u4 = NeonStoreD::from_complex_ref(chunk.get_unchecked(16..));
                        let u9 = NeonStoreD::from_complex_ref(chunk.get_unchecked(17..));
                        let u14 = NeonStoreD::from_complex_ref(chunk.get_unchecked(18..));
                        let u19 = NeonStoreD::from_complex_ref(chunk.get_unchecked(19..));

                        let [t0, t1, t2, t3] = self.bf4.exec([u0, u1, u2, u3]);
                        let [t4, t5, t6, t7] = self.bf4.exec([u4, u5, u6, u7]);
                        let [t8, t9, t10, t11] = self.bf4.exec([u8, u9, u10, u11]);
                        let [t12, t13, t14, t15] = self.bf4.exec([u12, u13, u14, u15]);
                        let [t16, t17, t18, t19] = self.bf4.exec([u16, u17, u18, u19]);

                        let [u0, u4, u8, u12, u16] = self.bf5.exec([t0, t4, t8, t12, t16]);
                        let [u5, u9, u13, u17, u1] = self.bf5.exec([t1, t5, t9, t13, t17]);
                        let [u10, u14, u18, u2, u6] = self.bf5.exec([t2, t6, t10, t14, t18]);
                        let [u15, u19, u3, u7, u11] = self.bf5.exec([t3, t7, t11, t15, t19]);

                        vst1q_f64(chunk.as_mut_ptr().cast(), u0.v);
                        vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), u1.v);

                        vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), u2.v);
                        vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), u3.v);

                        vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), u4.v);
                        vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), u5.v);

                        vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), u6.v);
                        vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), u7.v);

                        vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), u8.v);
                        vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), u9.v);

                        vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), u10.v);
                        vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), u11.v);

                        vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), u12.v);
                        vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), u13.v);

                        vst1q_f64(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), u14.v);
                        vst1q_f64(chunk.get_unchecked_mut(15..).as_mut_ptr().cast(), u15.v);

                        vst1q_f64(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), u16.v);
                        vst1q_f64(chunk.get_unchecked_mut(17..).as_mut_ptr().cast(), u17.v);

                        vst1q_f64(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), u18.v);
                        vst1q_f64(chunk.get_unchecked_mut(19..).as_mut_ptr().cast(), u19.v);
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf20d!(
    NeonButterfly20d,
    "neon",
    ColumnButterfly5d,
    ColumnButterfly4d
);
#[cfg(feature = "fcma")]
gen_bf20d!(
    NeonFcmaButterfly20d,
    "fcma",
    ColumnFcmaButterfly5d,
    ColumnFcmaButterfly4d
);

#[inline(always)]
pub(crate) fn transpose_5x4(
    rows0: [NeonStoreF; 4],
    rows1: [NeonStoreF; 4],
    rows2: [NeonStoreF; 4],
) -> ([NeonStoreF; 5], [NeonStoreF; 5]) {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[2].v, rows0[3].v));

    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    let e0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[2].v, rows1[3].v));

    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[0].v, rows2[1].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[2].v, rows2[3].v));
    (
        [
            NeonStoreF::raw(a0.0),
            NeonStoreF::raw(a0.1),
            NeonStoreF::raw(b0.0),
            NeonStoreF::raw(b0.1),
            NeonStoreF::raw(c0.0),
        ],
        [
            NeonStoreF::raw(d0.0),
            NeonStoreF::raw(d0.1),
            NeonStoreF::raw(e0.0),
            NeonStoreF::raw(e0.1),
            NeonStoreF::raw(f0.0),
        ],
    )
}

macro_rules! gen_bf24f {
    ($name: ident, $features: literal, $internal_bf5: ident, $internal_bf4: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf4, $internal_bf5};
        pub(crate) struct $name {
            direction: FftDirection,
            bf4: $internal_bf4,
            bf5: $internal_bf5,
            twiddles: [NeonStoreF; 9],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(5, 4, fft_direction, 20),
                    bf4: $internal_bf4::new(fft_direction),
                    bf5: $internal_bf5::new(fft_direction),
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
                20
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(20) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows1: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows2: [NeonStoreF; 4] = [NeonStoreF::default(); 4];

                    for chunk in in_place.chunks_exact_mut(20) {
                        // columns
                        for i in 0..4 {
                            rows0[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 5..));
                            rows1[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 5 + 2..));
                            rows2[i] = NeonStoreF::from_complex(chunk.get_unchecked(i * 5 + 4));
                        }

                        rows0 = self.bf4.exec(rows0);
                        rows1 = self.bf4.exec(rows1);
                        rows2 = self.bf4.exec(rows2);

                        for i in 1..4 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 3]);
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 6]);
                        }

                        let transposed = transpose_5x4(rows0, rows1, rows2);

                        let q0 = self.bf5.exec(transposed.0);
                        let q1 = self.bf5.exec(transposed.1);

                        for i in 0..5 {
                            q0[i].write(chunk.get_unchecked_mut(i * 4..));
                            q1[i].write(chunk.get_unchecked_mut(i * 4 + 2..));
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf24f!(
    NeonButterfly20f,
    "neon",
    ColumnButterfly5f,
    ColumnButterfly4f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf24f!(
    NeonFcmaButterfly20f,
    "fcma",
    ColumnFcmaButterfly5f,
    ColumnFcmaButterfly4f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly20f, f32, NeonButterfly20f, 20, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly20, f32, NeonFcmaButterfly20f, 20, 1e-5);
    test_butterfly!(test_neon_butterfly20_f64, f64, NeonButterfly20d, 20, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly20_f64,
        f64,
        NeonFcmaButterfly20d,
        20,
        1e-7
    );
}
