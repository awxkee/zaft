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
use crate::neon::butterflies::NeonButterfly;
use crate::neon::butterflies::fast_bf9d::NeonFastButterfly9d;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

macro_rules! butterfly18d {
    ($name: ident, $bf_name: ident) => {
        pub(crate) struct $name {
            direction: FftDirection,
            bf9: $bf_name,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf9: $bf_name::new(fft_direction),
                }
            }
        }

        impl FftExecutor<f64> for $name {
            fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if in_place.len() % 18 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    for chunk in in_place.chunks_exact_mut(18) {
                        let u0 = vld1q_f64(chunk.as_ptr().cast());
                        let u3 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                        let u4 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                        let u7 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());

                        let u8 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                        let u11 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                        let u12 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                        let u15 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());

                        let u16 = vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast());
                        let u1 = vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast());
                        let u2 = vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast());
                        let u5 = vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast());

                        let u6 = vld1q_f64(chunk.get_unchecked(12..).as_ptr().cast());
                        let u9 = vld1q_f64(chunk.get_unchecked(13..).as_ptr().cast());
                        let u10 = vld1q_f64(chunk.get_unchecked(14..).as_ptr().cast());
                        let u13 = vld1q_f64(chunk.get_unchecked(15..).as_ptr().cast());

                        let u14 = vld1q_f64(chunk.get_unchecked(16..).as_ptr().cast());
                        let u17 = vld1q_f64(chunk.get_unchecked(17..).as_ptr().cast());

                        let (t0, t1) = NeonButterfly::butterfly2_f64(u0, u1);
                        let (t2, t3) = NeonButterfly::butterfly2_f64(u2, u3);
                        let (t4, t5) = NeonButterfly::butterfly2_f64(u4, u5);
                        let (t6, t7) = NeonButterfly::butterfly2_f64(u6, u7);
                        let (t8, t9) = NeonButterfly::butterfly2_f64(u8, u9);
                        let (t10, t11) = NeonButterfly::butterfly2_f64(u10, u11);
                        let (t12, t13) = NeonButterfly::butterfly2_f64(u12, u13);
                        let (t14, t15) = NeonButterfly::butterfly2_f64(u14, u15);
                        let (t16, t17) = NeonButterfly::butterfly2_f64(u16, u17);

                        let (u0, u2, u4, u6, u8, u10, u12, u14, u16) =
                            self.bf9.exec(t0, t2, t4, t6, t8, t10, t12, t14, t16);
                        let (u9, u11, u13, u15, u17, u1, u3, u5, u7) =
                            self.bf9.exec(t1, t3, t5, t7, t9, t11, t13, t15, t17);

                        vst1q_f64(chunk.as_mut_ptr().cast(), u0);
                        vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), u1);

                        vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), u2);
                        vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), u3);

                        vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), u4);
                        vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), u5);

                        vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), u6);
                        vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), u7);

                        vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), u8);
                        vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), u9);

                        vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), u10);
                        vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), u11);

                        vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), u12);
                        vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), u13);

                        vst1q_f64(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), u14);
                        vst1q_f64(chunk.get_unchecked_mut(15..).as_mut_ptr().cast(), u15);

                        vst1q_f64(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), u16);
                        vst1q_f64(chunk.get_unchecked_mut(17..).as_mut_ptr().cast(), u17);
                    }
                }
                Ok(())
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                18
            }
        }
    };
}

butterfly18d!(NeonButterfly18d, NeonFastButterfly9d);
#[cfg(feature = "fcma")]
use crate::neon::butterflies::fast_bf9d::NeonFcmaFastButterfly9d;
use crate::neon::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;

#[inline(always)]
pub(crate) fn transpose_9x2(
    rows0: [NeonStoreF; 2],
    rows1: [NeonStoreF; 2],
    rows2: [NeonStoreF; 2],
    rows3: [NeonStoreF; 2],
    rows4: [NeonStoreF; 2],
) -> [NeonStoreF; 9] {
    let a = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let b = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    let c = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[0].v, rows2[1].v));
    let d = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows3[0].v, rows3[1].v));
    let e = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows4[0].v, rows4[1].v));
    [
        NeonStoreF::raw(a.0),
        NeonStoreF::raw(a.1),
        NeonStoreF::raw(b.0),
        NeonStoreF::raw(b.1),
        NeonStoreF::raw(c.0),
        NeonStoreF::raw(c.1),
        NeonStoreF::raw(d.0),
        NeonStoreF::raw(d.1),
        NeonStoreF::raw(e.0),
    ]
}

#[cfg(feature = "fcma")]
butterfly18d!(NeonFcmaButterfly18d, NeonFcmaFastButterfly9d);

use crate::neon::mixed::ColumnButterfly2f;

macro_rules! gen_bf20f {
    ($name: ident, $features: literal, $internal_bf9: ident, $internal_bf2: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf9;
        pub(crate) struct $name {
            direction: FftDirection,
            bf2: $internal_bf2,
            bf9: $internal_bf9,
            twiddles: [NeonStoreF; 5],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(9, 2, fft_direction, 18),
                    bf2: $internal_bf2::new(fft_direction),
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
                18
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(18) {
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
                    let mut rows4: [NeonStoreF; 2] = [NeonStoreF::default(); 2];

                    for chunk in in_place.chunks_exact_mut(18) {
                        // columns
                        for i in 0..2 {
                            rows0[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 9..));
                            rows1[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 9 + 2..));
                            rows2[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 9 + 4..));
                            rows3[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 9 + 6..));
                            rows4[i] = NeonStoreF::from_complex(chunk.get_unchecked(i * 9 + 8));
                        }

                        rows0 = self.bf2.exec(rows0);
                        rows1 = self.bf2.exec(rows1);
                        rows2 = self.bf2.exec(rows2);
                        rows3 = self.bf2.exec(rows3);
                        rows4 = self.bf2.exec(rows4);

                        rows0[1] = NeonStoreF::$mul(rows0[1], self.twiddles[0]);
                        rows1[1] = NeonStoreF::$mul(rows1[1], self.twiddles[1]);
                        rows2[1] = NeonStoreF::$mul(rows2[1], self.twiddles[2]);
                        rows3[1] = NeonStoreF::$mul(rows3[1], self.twiddles[3]);
                        rows4[1] = NeonStoreF::$mul(rows4[1], self.twiddles[4]);

                        let t = transpose_9x2(rows0, rows1, rows2, rows3, rows4);

                        let q0 = self.bf9.exec(t);

                        for i in 0..9 {
                            q0[i].write(chunk.get_unchecked_mut(i * 2..));
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

gen_bf20f!(
    NeonButterfly18f,
    "neon",
    ColumnButterfly9f,
    ColumnButterfly2f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf20f!(
    NeonFcmaButterfly18f,
    "fcma",
    ColumnFcmaButterfly9f,
    ColumnButterfly2f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly18, f32, NeonButterfly18f, 18, 1e-5);
    test_butterfly!(test_neon_butterfly18_f64, f64, NeonButterfly18d, 18, 1e-7);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly18, f32, NeonFcmaButterfly18f, 18, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly18_f64,
        f64,
        NeonFcmaButterfly18d,
        18,
        1e-7
    );
}
