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

        boring_neon_butterfly!($name, $features, f64, 9);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
                let mut rows = [NeonStoreD::default(); 9];
                for i in 0..9 {
                    rows[i] = NeonStoreD::from_complex_ref(chunk.slice_from(i..));
                }

                rows = self.bf9.exec(rows);

                for i in 0..9 {
                    rows[i].write(chunk.slice_from_mut(i..));
                }
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_r2c(&self, src: &[f64], dst: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(9) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        src.len(),
                        self.real_length(),
                    ));
                }
                if !dst.len().is_multiple_of(5) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        dst.len(),
                        self.complex_length(),
                    ));
                }

                unsafe {
                    let mut rows = [NeonStoreD::default(); 9];

                    for (dst, src) in dst.chunks_exact_mut(5).zip(src.chunks_exact(9)) {
                        for i in 0..4 {
                            let q = NeonStoreD::load(src.get_unchecked(i * 2..));
                            let [v0, v1] = q.to_complex();
                            rows[i * 2] = v0;
                            rows[i * 2 + 1] = v1;
                        }

                        rows[8] = NeonStoreD::load1(src.get_unchecked(8..));

                        rows = self.bf9.exec(rows);

                        for i in 0..5 {
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

            #[inline]
            fn real_length(&self) -> usize {
                9
            }

            #[inline]
            fn complex_length(&self) -> usize {
                5
            }

            fn complex_scratch_length(&self) -> usize {
                0
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

        boring_neon_butterfly!($name, $features, f32, 9);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows0: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                let mut rows1: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                // columns
                for i in 0..3 {
                    rows0[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 3..));
                    rows1[i] = NeonStoreF::from_complex(chunk.index(i * 3 + 2));
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
                    rows0[i].write(chunk.slice_from_mut(i * 3..));
                    rows1[i].write_lo(chunk.slice_from_mut(i * 3 + 2..));
                }
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_r2c(&self, src: &[f32], dst: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(9) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        src.len(),
                        self.real_length(),
                    ));
                }
                if !dst.len().is_multiple_of(5) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        dst.len(),
                        self.complex_length(),
                    ));
                }
                if src.len() / 9 != dst.len() / 5 {
                    return Err(ZaftError::InvalidSamplesCount(src.len() / 9, dst.len() / 5));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                    let mut rows1: [NeonStoreF; 3] = [NeonStoreF::default(); 3];

                    for (dst, src) in dst.chunks_exact_mut(5).zip(src.chunks_exact(9)) {
                        // columns
                        for i in 0..3 {
                            let q = NeonStoreF::load3(src.get_unchecked(i * 3..));
                            let [v0, v1] = q.to_complex();
                            rows0[i] = v0;
                            rows1[i] = v1;
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

                        rows0[0].write(dst);
                        rows1[0].write_lo(dst.get_unchecked_mut(2..));
                        rows0[1].write(dst.get_unchecked_mut(3..));
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

            #[inline]
            fn real_length(&self) -> usize {
                9
            }

            #[inline]
            fn complex_length(&self) -> usize {
                5
            }

            fn complex_scratch_length(&self) -> usize {
                0
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
    use crate::r2c::test_r2c_butterfly;

    test_r2c_butterfly!(test_neon_r2c_butterfly9, f32, NeonButterfly9f, 9, 1e-5);
    test_r2c_butterfly!(test_neon_r2c_butterfly9d, f64, NeonButterfly9d, 9, 1e-7);
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
