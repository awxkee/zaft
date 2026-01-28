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
use crate::neon::butterflies::shared::{boring_neon_butterfly, gen_butterfly_twiddles_f32};
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

macro_rules! gen_bf16d {
    ($name: ident, $features: literal, $internal_bf: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf16: $internal_bf,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf16: $internal_bf::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f64, 16);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
                let mut rows = [NeonStoreD::default(); 16];
                for i in 0..16 {
                    rows[i] = NeonStoreD::from_complex_ref(chunk.slice_from(i..));
                }
                rows = self.bf16.exec(rows);
                for i in 0..16 {
                    rows[i].write(chunk.slice_from_mut(i..));
                }
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_r2c(&self, src: &[f64], dst: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(16) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        src.len(),
                        self.real_length(),
                    ));
                }
                if !dst.len().is_multiple_of(9) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        dst.len(),
                        self.complex_length(),
                    ));
                }

                unsafe {
                    let mut rows = [NeonStoreD::default(); 16];

                    for (dst, src) in dst.chunks_exact_mut(9).zip(src.chunks_exact(16)) {
                        for i in 0..8 {
                            let [v0, v1] =
                                NeonStoreD::load(src.get_unchecked(i * 2..)).to_complex();
                            rows[i * 2] = v0;
                            rows[i * 2 + 1] = v1;
                        }
                        rows = self.bf16.exec(rows);
                        for i in 0..9 {
                            rows[i].write(dst.get_unchecked_mut(i..));
                        }
                    }
                }
                Ok(())
            }
        }

        impl R2CFftExecutor<f64> for $name {
            fn execute(&self, input: &[f64], output: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                unsafe { self.execute_r2c(input, output) }
            }

            fn execute_with_scratch(
                &self,
                input: &[f64],
                output: &mut [Complex<f64>],
                _: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_r2c(input, output) }
            }

            fn real_length(&self) -> usize {
                16
            }

            fn complex_length(&self) -> usize {
                9
            }

            fn complex_scratch_length(&self) -> usize {
                0
            }
        }
    };
}

gen_bf16d!(NeonButterfly16d, "neon", ColumnButterfly16d);
#[cfg(feature = "fcma")]
gen_bf16d!(NeonFcmaButterfly16d, "fcma", ColumnFcmaButterfly16d);

#[inline(always)]
pub(crate) fn transpose_4x4(
    rows0: [NeonStoreF; 4],
    rows1: [NeonStoreF; 4],
) -> ([NeonStoreF; 4], [NeonStoreF; 4]) {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[2].v, rows0[3].v));

    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    let e0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[2].v, rows1[3].v));
    (
        [
            NeonStoreF::raw(a0.0),
            NeonStoreF::raw(a0.1),
            NeonStoreF::raw(b0.0),
            NeonStoreF::raw(b0.1),
        ],
        [
            NeonStoreF::raw(d0.0),
            NeonStoreF::raw(d0.1),
            NeonStoreF::raw(e0.0),
            NeonStoreF::raw(e0.1),
        ],
    )
}

macro_rules! gen_bf16f {
    ($name: ident, $features: literal, $internal_bf4: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf4;
        pub(crate) struct $name {
            direction: FftDirection,
            bf4: $internal_bf4,
            twiddles: [NeonStoreF; 6],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(4, 4, fft_direction, 16),
                    bf4: $internal_bf4::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 16);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows0: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                let mut rows1: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                // columns
                for i in 0..4 {
                    rows0[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 4..));
                    rows1[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 4 + 2..));
                }

                rows0 = self.bf4.exec(rows0);
                rows1 = self.bf4.exec(rows1);

                for i in 1..4 {
                    rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 3]);
                }

                let transposed = transpose_4x4(rows0, rows1);

                let q0 = self.bf4.exec(transposed.0);
                let q1 = self.bf4.exec(transposed.1);

                for i in 0..4 {
                    q0[i].write(chunk.slice_from_mut(i * 4..));
                    q1[i].write(chunk.slice_from_mut(i * 4 + 2..));
                }
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_r2c(&self, src: &[f32], dst: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(16) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), 16));
                }
                if !dst.len().is_multiple_of(9) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), 9));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows1: [NeonStoreF; 4] = [NeonStoreF::default(); 4];

                    for (dst, src) in dst.chunks_exact_mut(9).zip(src.chunks_exact(16)) {
                        // columns
                        for i in 0..4 {
                            let q = NeonStoreF::load(src.get_unchecked(i * 4..));
                            let [u0, u1] = q.to_complex();
                            rows0[i] = u0;
                            rows1[i] = u1;
                        }

                        rows0 = self.bf4.exec(rows0);
                        rows1 = self.bf4.exec(rows1);

                        for i in 1..4 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 3]);
                        }

                        let transposed = transpose_4x4(rows0, rows1);

                        let q0 = self.bf4.exec(transposed.0);
                        let q1 = self.bf4.exec(transposed.1);

                        for i in 0..2 {
                            q0[i].write(dst.get_unchecked_mut(i * 4..));
                            q1[i].write(dst.get_unchecked_mut(i * 4 + 2..));
                        }
                        q0[2].write_lo(dst.get_unchecked_mut(8..));
                    }
                }
                Ok(())
            }
        }

        impl R2CFftExecutor<f32> for $name {
            fn execute(&self, input: &[f32], output: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                unsafe { self.execute_r2c(input, output) }
            }

            fn execute_with_scratch(
                &self,
                input: &[f32],
                output: &mut [Complex<f32>],
                _: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_r2c(input, output) }
            }

            fn real_length(&self) -> usize {
                16
            }

            fn complex_length(&self) -> usize {
                9
            }

            fn complex_scratch_length(&self) -> usize {
                0
            }
        }
    };
}

gen_bf16f!(NeonButterfly16f, "neon", ColumnButterfly4f, mul_by_complex);
#[cfg(feature = "fcma")]
gen_bf16f!(
    NeonFcmaButterfly16f,
    "fcma",
    ColumnFcmaButterfly4f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};
    use crate::r2c::test_r2c_butterfly;

    test_r2c_butterfly!(test_neon_r2c_butterfly16, f32, NeonButterfly16f, 16, 1e-5);
    test_r2c_butterfly!(test_neon_r2c_butterfly16d, f64, NeonButterfly16d, 16, 1e-5);
    test_butterfly!(test_neon_butterfly16, f32, NeonButterfly16f, 16, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly16, f32, NeonButterfly16f, 16, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly16d, f64, NeonFcmaButterfly16d, 16, 1e-5);
    test_butterfly!(test_neon_butterfly16_f64, f64, NeonButterfly16d, 16, 1e-7);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(test_oof_fcma_butterfly16, f32, NeonButterfly16f, 16, 1e-5);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly16d,
        f64,
        NeonFcmaButterfly16d,
        16,
        1e-5
    );
    test_oof_butterfly!(test_oof_butterfly16, f32, NeonButterfly16f, 16, 1e-5);
    test_oof_butterfly!(test_oof_butterfly16_f64, f64, NeonButterfly16d, 16, 1e-9);
}
