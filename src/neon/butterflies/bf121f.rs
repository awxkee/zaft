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
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::transpose_2x11;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;
use std::sync::Arc;

macro_rules! gen_bf121f {
    ($name: ident, $feature: literal, $internal_bf: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf11: $internal_bf,
            twiddles: [NeonStoreF; 60],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(11, 11, fft_direction, 121),
                    bf11: $internal_bf::new(fft_direction),
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
                121
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if in_place.len() % 121 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows: [NeonStoreF; 11] = [NeonStoreF::default(); 11];
                    let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 121];

                    for chunk in in_place.chunks_exact_mut(121) {
                        // columns
                        for k in 0..5 {
                            for i in 0..11 {
                                rows[i] = NeonStoreF::from_complex_ref(
                                    chunk.get_unchecked(i * 11 + k * 2..),
                                );
                            }

                            rows = self.bf11.exec(rows);

                            for i in 1..11 {
                                rows[i] = NeonStoreF::$mul(rows[i], self.twiddles[i - 1 + 10 * k]);
                            }

                            let transposed = transpose_2x11(rows);

                            for i in 0..5 {
                                transposed[i * 2]
                                    .write_uninit(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                                transposed[i * 2 + 1].write_uninit(
                                    scratch.get_unchecked_mut((k * 2 + 1) * 11 + i * 2..),
                                );
                            }
                            {
                                let i = 5;
                                transposed[i * 2]
                                    .write_lo_u(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                                transposed[i * 2 + 1].write_lo_u(
                                    scratch.get_unchecked_mut((k * 2 + 1) * 11 + i * 2..),
                                );
                            }
                        }

                        {
                            let k = 5;
                            for i in 0..11 {
                                rows[i] =
                                    NeonStoreF::from_complex(chunk.get_unchecked(i * 11 + k * 2));
                            }

                            rows = self.bf11.exec(rows);

                            for i in 1..11 {
                                rows[i] = NeonStoreF::$mul(rows[i], self.twiddles[i - 1 + 10 * k]);
                            }

                            let transposed = transpose_2x11(rows);

                            for i in 0..5 {
                                transposed[i * 2]
                                    .write_uninit(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                            }
                            {
                                let i = 5;
                                transposed[i * 2]
                                    .write_lo_u(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                            }
                        }

                        // rows

                        for k in 0..5 {
                            for i in 0..11 {
                                rows[i] = NeonStoreF::from_complex_refu(
                                    scratch.get_unchecked(i * 11 + k * 2..),
                                );
                            }
                            rows = self.bf11.exec(rows);
                            for i in 0..11 {
                                rows[i].write(chunk.get_unchecked_mut(i * 11 + k * 2..));
                            }
                        }
                        {
                            let k = 5;
                            for i in 0..11 {
                                rows[i] = NeonStoreF::from_complexu(
                                    scratch.get_unchecked(i * 11 + k * 2),
                                );
                            }
                            rows = self.bf11.exec(rows);
                            for i in 0..11 {
                                rows[i].write_lo(chunk.get_unchecked_mut(i * 11 + k * 2..));
                            }
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
            #[target_feature(enable = $feature)]
            fn execute_out_of_place_impl(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                if src.len() % 121 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if dst.len() % 121 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    let mut rows: [NeonStoreF; 11] = [NeonStoreF::default(); 11];
                    let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 121];

                    for (dst, src) in dst.chunks_exact_mut(121).zip(src.chunks_exact(121)) {
                        // columns
                        for k in 0..5 {
                            for i in 0..11 {
                                rows[i] = NeonStoreF::from_complex_ref(
                                    src.get_unchecked(i * 11 + k * 2..),
                                );
                            }

                            rows = self.bf11.exec(rows);

                            for i in 1..11 {
                                rows[i] = NeonStoreF::$mul(rows[i], self.twiddles[i - 1 + 10 * k]);
                            }

                            let transposed = transpose_2x11(rows);

                            for i in 0..5 {
                                transposed[i * 2]
                                    .write_uninit(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                                transposed[i * 2 + 1].write_uninit(
                                    scratch.get_unchecked_mut((k * 2 + 1) * 11 + i * 2..),
                                );
                            }
                            {
                                let i = 5;
                                transposed[i * 2]
                                    .write_lo_u(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                                transposed[i * 2 + 1].write_lo_u(
                                    scratch.get_unchecked_mut((k * 2 + 1) * 11 + i * 2..),
                                );
                            }
                        }

                        {
                            let k = 5;
                            for i in 0..11 {
                                rows[i] =
                                    NeonStoreF::from_complex(src.get_unchecked(i * 11 + k * 2));
                            }

                            rows = self.bf11.exec(rows);

                            for i in 1..11 {
                                rows[i] = NeonStoreF::$mul(rows[i], self.twiddles[i - 1 + 10 * k]);
                            }

                            let transposed = transpose_2x11(rows);

                            for i in 0..5 {
                                transposed[i * 2]
                                    .write_uninit(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                            }
                            {
                                let i = 5;
                                transposed[i * 2]
                                    .write_lo_u(scratch.get_unchecked_mut(k * 2 * 11 + i * 2..));
                            }
                        }

                        // rows

                        for k in 0..5 {
                            for i in 0..11 {
                                rows[i] = NeonStoreF::from_complex_refu(
                                    scratch.get_unchecked(i * 11 + k * 2..),
                                );
                            }
                            rows = self.bf11.exec(rows);
                            for i in 0..11 {
                                rows[i].write(dst.get_unchecked_mut(i * 11 + k * 2..));
                            }
                        }
                        {
                            let k = 5;
                            for i in 0..11 {
                                rows[i] = NeonStoreF::from_complexu(
                                    scratch.get_unchecked(i * 11 + k * 2),
                                );
                            }
                            rows = self.bf11.exec(rows);
                            for i in 0..11 {
                                rows[i].write_lo(dst.get_unchecked_mut(i * 11 + k * 2..));
                            }
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

gen_bf121f!(
    NeonButterfly121f,
    "neon",
    ColumnButterfly11f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf121f!(
    NeonFcmaButterfly121f,
    "fcma",
    ColumnFcmaButterfly11f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly121, f32, NeonButterfly121f, 121, 1e-3);
    test_oof_butterfly!(
        test_oof_neon_butterfly121,
        f32,
        NeonButterfly121f,
        121,
        1e-3
    );

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly121,
        f32,
        NeonFcmaButterfly121f,
        121,
        1e-3
    );
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly121,
        f32,
        NeonFcmaButterfly121f,
        121,
        1e-3
    );
}
