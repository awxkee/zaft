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
use crate::neon::mixed::{ColumnButterfly2f, NeonStoreD, NeonStoreF};
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

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
                let u6 = NeonStoreD::from_complex_ref(chunk.slice_from(6..));
                let u7 = NeonStoreD::from_complex_ref(chunk.slice_from(7..));

                let [y0, y1, y2, y3, y4, y5, y6, y7] =
                    self.bf.exec([u0, u1, u2, u3, u4, u5, u6, u7]);

                unsafe {
                    vst1q_f64(chunk.slice_from_mut(0..).as_mut_ptr().cast(), y0.v);
                    vst1q_f64(chunk.slice_from_mut(1..).as_mut_ptr().cast(), y1.v);
                    vst1q_f64(chunk.slice_from_mut(2..).as_mut_ptr().cast(), y2.v);
                    vst1q_f64(chunk.slice_from_mut(3..).as_mut_ptr().cast(), y3.v);
                    vst1q_f64(chunk.slice_from_mut(4..).as_mut_ptr().cast(), y4.v);
                    vst1q_f64(chunk.slice_from_mut(5..).as_mut_ptr().cast(), y5.v);
                    vst1q_f64(chunk.slice_from_mut(6..).as_mut_ptr().cast(), y6.v);
                    vst1q_f64(chunk.slice_from_mut(7..).as_mut_ptr().cast(), y7.v);
                }
            }
        }

        boring_neon_butterfly!($name, $features, f64, 8);

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_r2c(&self, src: &[f64], dst: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(8) {
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
                    for (dst, src) in dst.chunks_exact_mut(5).zip(src.chunks_exact(8)) {
                        let [u0, u1] = NeonStoreD::load(src).to_complex();
                        let [u2, u3] = NeonStoreD::load(src.get_unchecked(2..)).to_complex();
                        let [u4, u5] = NeonStoreD::load(src.get_unchecked(4..)).to_complex();
                        let [u6, u7] = NeonStoreD::load(src.get_unchecked(6..)).to_complex();

                        let [y0, y1, y2, y3, y4, _, _, _] =
                            self.bf.exec([u0, u1, u2, u3, u4, u5, u6, u7]);

                        y0.write(dst);
                        y1.write(dst.get_unchecked_mut(1..));
                        y2.write(dst.get_unchecked_mut(2..));
                        y3.write(dst.get_unchecked_mut(3..));
                        y4.write(dst.get_unchecked_mut(4..));
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

            fn complex_length(&self) -> usize {
                5
            }

            fn real_length(&self) -> usize {
                8
            }

            fn complex_scratch_length(&self) -> usize {
                0
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

        boring_neon_butterfly!($name, $features, f32, 8);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows0: [NeonStoreF; 2] = [NeonStoreF::default(); 2];
                let mut rows1: [NeonStoreF; 2] = [NeonStoreF::default(); 2];
                // columns
                for i in 0..2 {
                    rows0[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 4..));
                    rows1[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 4 + 2..));
                }

                rows0 = self.bf2.exec(rows0);
                rows1 = self.bf2.exec(rows1);

                rows0[1] = NeonStoreF::$mul(rows0[1], self.twiddles[0]);
                rows1[1] = NeonStoreF::$mul(rows1[1], self.twiddles[1]);

                let transposed = transpose_f32x2_4x2(rows0, rows1);

                let q0 = self.bf4.exec(transposed);

                for i in 0..4 {
                    q0[i].write(chunk.slice_from_mut(i * 2..));
                }
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_r2c(&self, src: &[f32], dst: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(8) {
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
                    let mut rows0: [NeonStoreF; 2] = [NeonStoreF::default(); 2];
                    let mut rows1: [NeonStoreF; 2] = [NeonStoreF::default(); 2];

                    for (dst, src) in dst.chunks_exact_mut(5).zip(src.chunks_exact(8)) {
                        // columns

                        let s0 = NeonStoreF::load(src);
                        let s1 = NeonStoreF::load(src.get_unchecked(4..));

                        let [u0, u1] = s0.to_complex();
                        let [u2, u3] = s1.to_complex();

                        rows0[0] = u0;
                        rows0[1] = u2;
                        rows1[0] = u1;
                        rows1[1] = u3;

                        rows0 = self.bf2.exec(rows0);
                        rows1 = self.bf2.exec(rows1);

                        rows0[1] = NeonStoreF::mul_by_complex(rows0[1], self.twiddles[0]);
                        rows1[1] = NeonStoreF::mul_by_complex(rows1[1], self.twiddles[1]);

                        let transposed = transpose_f32x2_4x2(rows0, rows1);

                        let q0 = self.bf4.exec(transposed);

                        for i in 0..2 {
                            q0[i].write(dst.get_unchecked_mut(i * 2..));
                        }
                        q0[2].write_lo(dst.get_unchecked_mut(4..));
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
                8
            }

            fn complex_length(&self) -> usize {
                5
            }

            fn complex_scratch_length(&self) -> usize {
                0
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
    use crate::r2c::test_r2c_butterfly;

    test_r2c_butterfly!(test_neon_r2c_butterfly8, f32, NeonButterfly8f, 8, 1e-5);
    test_r2c_butterfly!(test_neon_r2c_butterfly8d, f64, NeonButterfly8d, 8, 1e-5);
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
