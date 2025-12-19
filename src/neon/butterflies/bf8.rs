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
use crate::neon::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::neon::mixed::{ColumnButterfly2f, NeonStoreD, NeonStoreF};
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;
use std::sync::Arc;

macro_rules! gen_bf8d {
    ($name: ident, $features: literal, $bf: ident) => {
        use crate::neon::mixed::$bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf: $bf,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf: $bf::new(fft_direction),
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
                8
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(8) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    for chunk in in_place.chunks_exact_mut(8) {
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

                        let [y0, y1, y2, y3, y4, y5, y6, y7] =
                            self.bf.exec([u0, u1, u2, u3, u4, u5, u6, u7]);

                        vst1q_f64(chunk.as_mut_ptr().cast(), y0.v);
                        vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1.v);
                        vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2.v);
                        vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3.v);
                        vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4.v);
                        vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y5.v);
                        vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y6.v);
                        vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y7.v);
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
                if !src.len().is_multiple_of(8) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(8) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    for (dst, src) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
                        let u0 = NeonStoreD::raw(vld1q_f64(src.as_ptr().cast()));
                        let u1 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(1..).as_ptr().cast()));
                        let u2 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(2..).as_ptr().cast()));
                        let u3 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(3..).as_ptr().cast()));
                        let u4 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(4..).as_ptr().cast()));
                        let u5 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(5..).as_ptr().cast()));
                        let u6 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(6..).as_ptr().cast()));
                        let u7 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(7..).as_ptr().cast()));

                        let [y0, y1, y2, y3, y4, y5, y6, y7] =
                            self.bf.exec([u0, u1, u2, u3, u4, u5, u6, u7]);

                        vst1q_f64(dst.as_mut_ptr().cast(), y0.v);
                        vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), y1.v);
                        vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y2.v);
                        vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), y3.v);
                        vst1q_f64(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y4.v);
                        vst1q_f64(dst.get_unchecked_mut(5..).as_mut_ptr().cast(), y5.v);
                        vst1q_f64(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), y6.v);
                        vst1q_f64(dst.get_unchecked_mut(7..).as_mut_ptr().cast(), y7.v);
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

gen_bf8d!(NeonButterfly8d, "neon", ColumnButterfly8d);
#[cfg(feature = "fcma")]
gen_bf8d!(NeonFcmaButterfly8d, "fcma", ColumnFcmaButterfly8d);

#[inline(always)]
pub(crate) fn transpose_f32x2_4x2(
    rows0: [NeonStoreF; 2],
    rows1: [NeonStoreF; 2],
) -> [NeonStoreF; 4] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
    ]
}

macro_rules! gen_bf8f {
    ($name: ident, $features: literal, $internal_bf4: ident, $internal_bf2: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf4;

        pub(crate) struct $name {
            direction: FftDirection,
            bf4: $internal_bf4,
            bf2: $internal_bf2,
            twiddles: [NeonStoreF; 2],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf4: $internal_bf4::new(fft_direction),
                    bf2: $internal_bf2::new(fft_direction),
                    twiddles: gen_butterfly_twiddles_f32(4, 2, fft_direction, 8),
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
                8
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(8) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 2] = [NeonStoreF::default(); 2];
                    let mut rows1: [NeonStoreF; 2] = [NeonStoreF::default(); 2];

                    for chunk in in_place.chunks_exact_mut(8) {
                        // columns
                        for i in 0..2 {
                            rows0[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 4..));
                            rows1[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 4 + 2..));
                        }

                        rows0 = self.bf2.exec(rows0);
                        rows1 = self.bf2.exec(rows1);

                        rows0[1] = NeonStoreF::$mul(rows0[1], self.twiddles[0]);
                        rows1[1] = NeonStoreF::$mul(rows1[1], self.twiddles[1]);

                        let transposed = transpose_f32x2_4x2(rows0, rows1);

                        let q0 = self.bf4.exec(transposed);

                        for i in 0..4 {
                            q0[i].write(chunk.get_unchecked_mut(i * 2..));
                        }
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
                if !src.len().is_multiple_of(8) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(8) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 2] = [NeonStoreF::default(); 2];
                    let mut rows1: [NeonStoreF; 2] = [NeonStoreF::default(); 2];

                    for (dst, src) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
                        // columns
                        for i in 0..2 {
                            rows0[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 4..));
                            rows1[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 4 + 2..));
                        }

                        rows0 = self.bf2.exec(rows0);
                        rows1 = self.bf2.exec(rows1);

                        rows0[1] = NeonStoreF::mul_by_complex(rows0[1], self.twiddles[0]);
                        rows1[1] = NeonStoreF::mul_by_complex(rows1[1], self.twiddles[1]);

                        let transposed = transpose_f32x2_4x2(rows0, rows1);

                        let q0 = self.bf4.exec(transposed);

                        for i in 0..4 {
                            q0[i].write(dst.get_unchecked_mut(i * 2..));
                        }
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

gen_bf8f!(
    NeonButterfly8f,
    "neon",
    ColumnButterfly4f,
    ColumnButterfly2f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf8f!(
    NeonFcmaButterfly8f,
    "fcma",
    ColumnFcmaButterfly4f,
    ColumnButterfly2f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly8, f32, NeonButterfly8f, 8, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly8, f32, NeonFcmaButterfly8f, 8, 1e-5);
    test_butterfly!(test_neon_butterfly8_f64, f64, NeonButterfly8d, 8, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly8_f64, f64, NeonFcmaButterfly8d, 8, 1e-7);
    test_oof_butterfly!(test_oof_butterfly8, f32, NeonButterfly8f, 8, 1e-5);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(test_oof_fcma_butterfly8, f32, NeonFcmaButterfly8f, 8, 1e-5);
    test_oof_butterfly!(test_oof_butterfly8_f64, f64, NeonButterfly8d, 8, 1e-9);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly8_f64,
        f64,
        NeonFcmaButterfly8d,
        8,
        1e-9
    );
}
