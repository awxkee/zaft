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
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;
use std::sync::Arc;

macro_rules! gen_bf9d {
    ($name: ident, $features: literal, $bf: ident) => {
        use crate::neon::mixed::$bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf9: $bf,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf9: $bf::new(fft_direction),
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
                9
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(9) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows = [NeonStoreD::default(); 9];
                    for chunk in in_place.chunks_exact_mut(9) {
                        for i in 0..9 {
                            rows[i] = NeonStoreD::from_complex_ref(chunk.get_unchecked(i..));
                        }

                        rows = self.bf9.exec(rows);

                        for i in 0..9 {
                            rows[i].write(chunk.get_unchecked_mut(i..));
                        }
                    }
                }
                Ok(())
            }

            #[target_feature(enable = $features)]
            fn execute_out_of_place_impl(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(9) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(9) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    let mut rows = [NeonStoreD::default(); 9];

                    for (dst, src) in dst.chunks_exact_mut(9).zip(src.chunks_exact(9)) {
                        for i in 0..9 {
                            rows[i] = NeonStoreD::from_complex_ref(src.get_unchecked(i..));
                        }

                        rows = self.bf9.exec(rows);

                        for i in 0..9 {
                            rows[i].write(dst.get_unchecked_mut(i..));
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

        impl CompositeFftExecutor<f64> for $name {
            fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
                self
            }
        }
    };
}

gen_bf9d!(NeonButterfly9d, "neon", ColumnButterfly9d);
#[cfg(feature = "fcma")]
gen_bf9d!(NeonFcmaButterfly9d, "fcma", ColumnFcmaButterfly9d);

#[inline(always)]
pub(crate) fn transpose_f32x2_3x3(
    rows0: [NeonStoreF; 3],
    rows1: [NeonStoreF; 3],
) -> ([NeonStoreF; 3], [NeonStoreF; 3]) {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[2].v, unsafe { vdupq_n_f32(0.) }));

    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    let e0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[2].v, unsafe { vdupq_n_f32(0.) }));
    (
        [
            NeonStoreF::raw(a0.0),
            NeonStoreF::raw(a0.1),
            NeonStoreF::raw(b0.0),
        ],
        [
            NeonStoreF::raw(d0.0),
            NeonStoreF::raw(d0.1),
            NeonStoreF::raw(e0.0),
        ],
    )
}

macro_rules! gen_bf9f {
    ($name: ident, $features: literal, $bf: ident, $mul: ident) => {
        use crate::neon::mixed::$bf;
        pub(crate) struct $name {
            direction: FftDirection,
            twiddles: [NeonStoreF; 4],
            bf3: $bf,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(3, 3, fft_direction, 9),
                    bf3: $bf::new(fft_direction),
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
                9
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(9) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                    let mut rows1: [NeonStoreF; 3] = [NeonStoreF::default(); 3];

                    for chunk in in_place.chunks_exact_mut(9) {
                        // columns
                        for i in 0..3 {
                            rows0[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 3..));
                            rows1[i] = NeonStoreF::from_complex(chunk.get_unchecked(i * 3 + 2));
                        }

                        rows0 = self.bf3.exec(rows0);
                        rows1 = self.bf3.exec(rows1);

                        for i in 1..3 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 2]);
                        }

                        (rows0, rows1) = transpose_f32x2_3x3(rows0, rows1);

                        // rows

                        rows0 = self.bf3.exec(rows0);
                        rows1 = self.bf3.exec(rows1);

                        for i in 0..3 {
                            rows0[i].write(chunk.get_unchecked_mut(i * 3..));
                            rows1[i].write_lo(chunk.get_unchecked_mut(i * 3 + 2..));
                        }
                    }
                }
                Ok(())
            }

            #[target_feature(enable = $features)]
            fn execute_out_of_place_impl(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(9) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(9) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                    let mut rows1: [NeonStoreF; 3] = [NeonStoreF::default(); 3];

                    for (dst, src) in dst.chunks_exact_mut(9).zip(src.chunks_exact(9)) {
                        // columns
                        for i in 0..3 {
                            rows0[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 3..));
                            rows1[i] = NeonStoreF::from_complex(src.get_unchecked(i * 3 + 2));
                        }

                        rows0 = self.bf3.exec(rows0);
                        rows1 = self.bf3.exec(rows1);

                        for i in 1..3 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 2]);
                        }

                        (rows0, rows1) = transpose_f32x2_3x3(rows0, rows1);

                        // rows

                        rows0 = self.bf3.exec(rows0);
                        rows1 = self.bf3.exec(rows1);

                        for i in 0..3 {
                            rows0[i].write(dst.get_unchecked_mut(i * 3..));
                            rows1[i].write_lo(dst.get_unchecked_mut(i * 3 + 2..));
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

        impl CompositeFftExecutor<f32> for $name {
            fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
                self
            }
        }
    };
}

gen_bf9f!(NeonButterfly9f, "neon", ColumnButterfly3f, mul_by_complex);
#[cfg(feature = "fcma")]
gen_bf9f!(
    NeonFcmaButterfly9f,
    "fcma",
    ColumnFcmaButterfly3f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly9, f32, NeonButterfly9f, 9, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly9, f32, NeonFcmaButterfly9f, 9, 1e-5);
    test_butterfly!(test_neon_butterfly9_f64, f64, NeonButterfly9d, 9, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly9_f64, f64, NeonFcmaButterfly9d, 9, 1e-7);
    test_oof_butterfly!(test_oof_butterfly9, f32, NeonButterfly9f, 9, 1e-5);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(test_oof_fcma_butterfly9, f32, NeonFcmaButterfly9f, 9, 1e-5);
    test_oof_butterfly!(test_oof_butterfly9_f64, f64, NeonButterfly9d, 9, 1e-9);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly9_f64,
        f64,
        NeonFcmaButterfly9d,
        9,
        1e-9
    );
}
