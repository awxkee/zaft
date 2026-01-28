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
use crate::neon::butterflies::shared::boring_neon_butterfly;
use crate::neon::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::neon::mixed::{ColumnButterfly2f, NeonStoreD, NeonStoreF};
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

macro_rules! gen_bf6d {
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

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
                let u0 = NeonStoreD::from_complex_ref(chunk.slice_from(0..));
                let u1 = NeonStoreD::from_complex_ref(chunk.slice_from(1..));
                let u2 = NeonStoreD::from_complex_ref(chunk.slice_from(2..));
                let u3 = NeonStoreD::from_complex_ref(chunk.slice_from(3..));
                let u4 = NeonStoreD::from_complex_ref(chunk.slice_from(4..));
                let u5 = NeonStoreD::from_complex_ref(chunk.slice_from(5..));

                let [y0, y1, y2, y3, y4, y5] = self.bf.exec([u0, u1, u2, u3, u4, u5]);

                y0.write(chunk.slice_from_mut(0..));
                y1.write(chunk.slice_from_mut(1..));
                y2.write(chunk.slice_from_mut(2..));
                y3.write(chunk.slice_from_mut(3..));
                y4.write(chunk.slice_from_mut(4..));
                y5.write(chunk.slice_from_mut(5..));
            }
        }

        boring_neon_butterfly!($name, $features, f64, 6);

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_r2c(&self, src: &[f64], dst: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(6) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        src.len(),
                        self.real_length(),
                    ));
                }
                if !dst.len().is_multiple_of(4) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        dst.len(),
                        self.complex_length(),
                    ));
                }
                if src.len() / 6 != dst.len() / 4 {
                    return Err(ZaftError::InvalidSamplesCount(src.len() / 6, dst.len() / 4));
                }

                unsafe {
                    for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(6)) {
                        let [u0, u1] = NeonStoreD::load(src).to_complex();
                        let [u2, u3] = NeonStoreD::load(src.get_unchecked(2..)).to_complex();
                        let [u4, u5] = NeonStoreD::load(src.get_unchecked(4..)).to_complex();

                        let [y0, y1, y2, y3, _, _] = self.bf.exec([u0, u1, u2, u3, u4, u5]);

                        y0.write(dst);
                        y1.write(dst.get_unchecked_mut(1..));
                        y2.write(dst.get_unchecked_mut(2..));
                        y3.write(dst.get_unchecked_mut(3..));
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
                6
            }

            #[inline]
            fn complex_length(&self) -> usize {
                4
            }

            fn complex_scratch_length(&self) -> usize {
                0
            }
        }
    };
}

gen_bf6d!(NeonButterfly6d, "neon", ColumnButterfly6d);
#[cfg(feature = "fcma")]
gen_bf6d!(NeonFcmaButterfly6d, "fcma", ColumnFcmaButterfly6d);

#[inline(always)]
pub(crate) fn transpose_f32x2_3x2(
    rows0: [NeonStoreF; 2],
    rows1: [NeonStoreF; 2],
) -> [NeonStoreF; 3] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    [
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
    ]
}

macro_rules! gen_bf6f {
    ($name: ident, $features: literal, $internal_bf3: ident, $internal_bf2: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf3;

        pub(crate) struct $name {
            direction: FftDirection,
            bf3: $internal_bf3,
            bf2: $internal_bf2,
            twiddles: [NeonStoreF; 2],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf3: $internal_bf3::new(fft_direction),
                    bf2: $internal_bf2::new(fft_direction),
                    twiddles: gen_butterfly_twiddles_f32(3, 2, fft_direction, 6),
                }
            }
        }

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows0: [NeonStoreF; 2] = [NeonStoreF::default(); 2];
                let mut rows1: [NeonStoreF; 2] = [NeonStoreF::default(); 2];
                // columns
                for i in 0..2 {
                    rows0[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 3..));
                    rows1[i] = NeonStoreF::from_complex(chunk.index(i * 3 + 2));
                }

                rows0 = self.bf2.exec(rows0);
                rows1 = self.bf2.exec(rows1);

                rows0[1] = NeonStoreF::$mul(rows0[1], self.twiddles[0]);
                rows1[1] = NeonStoreF::$mul(rows1[1], self.twiddles[1]);

                let transposed = transpose_f32x2_3x2(rows0, rows1);

                let q0 = self.bf3.exec(transposed);

                for i in 0..3 {
                    q0[i].write(chunk.slice_from_mut(i * 2..));
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 6);

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_r2c(&self, src: &[f32], dst: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(6) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        src.len(),
                        self.real_length(),
                    ));
                }
                if !dst.len().is_multiple_of(4) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        dst.len(),
                        self.complex_length(),
                    ));
                }
                if src.len() / 6 != dst.len() / 4 {
                    return Err(ZaftError::InvalidSamplesCount(src.len() / 6, dst.len() / 4));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 2] = [NeonStoreF::default(); 2];
                    let mut rows1: [NeonStoreF; 2] = [NeonStoreF::default(); 2];

                    for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(6)) {
                        // columns
                        for i in 0..2 {
                            let [u0, u1] =
                                NeonStoreF::load3(src.get_unchecked(i * 3..)).to_complex();
                            rows0[i] = u0;
                            rows1[i] = u1;
                        }

                        rows0 = self.bf2.exec(rows0);
                        rows1 = self.bf2.exec(rows1);

                        rows0[1] = NeonStoreF::$mul(rows0[1], self.twiddles[0]);
                        rows1[1] = NeonStoreF::$mul(rows1[1], self.twiddles[1]);

                        let transposed = transpose_f32x2_3x2(rows0, rows1);

                        let q0 = self.bf3.exec(transposed);

                        for i in 0..2 {
                            q0[i].write(dst.get_unchecked_mut(i * 2..));
                        }
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
                6
            }

            #[inline]
            fn complex_length(&self) -> usize {
                4
            }

            fn complex_scratch_length(&self) -> usize {
                0
            }
        }
    };
}

gen_bf6f!(
    NeonButterfly6f,
    "neon",
    ColumnButterfly3f,
    ColumnButterfly2f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf6f!(
    NeonFcmaButterfly6f,
    "fcma",
    ColumnFcmaButterfly3f,
    ColumnButterfly2f,
    fcmul_fcma
);

#[cfg(test)]
mod test {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};
    use crate::r2c::test_r2c_butterfly;

    test_butterfly!(test_neon_butterfly6, f32, NeonButterfly6f, 6, 1e-5);
    test_butterfly!(test_neon_butterfly6_f64, f64, NeonButterfly6d, 6, 1e-7);

    test_r2c_butterfly!(test_neon_r2c_butterfly6, f32, NeonButterfly6f, 6, 1e-5);
    test_r2c_butterfly!(test_neon_r2c_butterfly6d, f64, NeonButterfly6d, 6, 1e-5);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly6, f32, NeonFcmaButterfly6f, 6, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly6_f64, f64, NeonFcmaButterfly6d, 6, 1e-7);

    test_oof_butterfly!(test_oof_butterfly6, f32, NeonButterfly6f, 6, 1e-5);
    test_oof_butterfly!(test_oof_butterfly6_f64, f64, NeonButterfly6d, 6, 1e-9);

    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(test_oof_fcma_butterfly6, f32, NeonFcmaButterfly6f, 6, 1e-5);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly6_f64,
        f64,
        NeonFcmaButterfly6d,
        6,
        1e-9
    );
}
