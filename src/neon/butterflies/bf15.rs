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

macro_rules! gen_bf15d {
    ($name: ident, $features: literal, $internal_bf5: ident, $internal_bf3: ident) => {
        use crate::neon::mixed::{$internal_bf3, $internal_bf5};
        pub(crate) struct $name {
            direction: FftDirection,
            bf5: $internal_bf5,
            bf3: $internal_bf3,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf5: $internal_bf5::new(fft_direction),
                    bf3: $internal_bf3::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f64, 15);

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
                let u8 = NeonStoreD::from_complex_ref(chunk.slice_from(8..));
                let u9 = NeonStoreD::from_complex_ref(chunk.slice_from(9..));
                let u10 = NeonStoreD::from_complex_ref(chunk.slice_from(10..));
                let u11 = NeonStoreD::from_complex_ref(chunk.slice_from(11..));
                let u12 = NeonStoreD::from_complex_ref(chunk.slice_from(12..));
                let u13 = NeonStoreD::from_complex_ref(chunk.slice_from(13..));
                let u14 = NeonStoreD::from_complex_ref(chunk.slice_from(14..));

                let mid0 = self.bf5.exec([u0, u3, u6, u9, u12]);
                let mid1 = self.bf5.exec([u5, u8, u11, u14, u2]);
                let mid2 = self.bf5.exec([u10, u13, u1, u4, u7]);

                // Since this is good-thomas algorithm, we don't need twiddle factors

                // Transpose the data and do size-3 FFTs down the columns
                let [y0, y1, y2] = self.bf3.exec([mid0[0], mid1[0], mid2[0]]);
                let [y3, y4, y5] = self.bf3.exec([mid0[1], mid1[1], mid2[1]]);
                let [y6, y7, y8] = self.bf3.exec([mid0[2], mid1[2], mid2[2]]);
                let [y9, y10, y11] = self.bf3.exec([mid0[3], mid1[3], mid2[3]]);
                let [y12, y13, y14] = self.bf3.exec([mid0[4], mid1[4], mid2[4]]);

                y0.write(chunk.slice_from_mut(0..));
                y4.write(chunk.slice_from_mut(1..));

                y8.write(chunk.slice_from_mut(2..));
                y9.write(chunk.slice_from_mut(3..));

                y13.write(chunk.slice_from_mut(4..));
                y2.write(chunk.slice_from_mut(5..));

                y3.write(chunk.slice_from_mut(6..));
                y7.write(chunk.slice_from_mut(7..));

                y11.write(chunk.slice_from_mut(8..));
                y12.write(chunk.slice_from_mut(9..));

                y1.write(chunk.slice_from_mut(10..));
                y5.write(chunk.slice_from_mut(11..));

                y6.write(chunk.slice_from_mut(12..));
                y10.write(chunk.slice_from_mut(13..));

                y14.write(chunk.slice_from_mut(14..));
            }
        }
    };
}

gen_bf15d!(
    NeonButterfly15d,
    "neon",
    ColumnButterfly5d,
    ColumnButterfly3d
);
#[cfg(feature = "fcma")]
gen_bf15d!(
    NeonFcmaButterfly15d,
    "fcma",
    ColumnFcmaButterfly5d,
    ColumnFcmaButterfly3d
);

#[inline(always)]
pub(crate) fn transpose_f32x2_5x3(
    rows0: [NeonStoreF; 3],
    rows1: [NeonStoreF; 3],
    rows2: [NeonStoreF; 3],
) -> ([NeonStoreF; 5], [NeonStoreF; 5]) {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[2].v, unsafe { vdupq_n_f32(0.) }));

    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    let e0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[2].v, unsafe { vdupq_n_f32(0.) }));

    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[0].v, rows2[1].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[2].v, unsafe { vdupq_n_f32(0.) }));
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

macro_rules! gen_bf15f {
    ($name: ident, $features: literal, $internal_bf5: ident, $internal_bf3: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf3, $internal_bf5};
        pub(crate) struct $name {
            direction: FftDirection,
            bf3: $internal_bf3,
            bf5: $internal_bf5,
            twiddles: [NeonStoreF; 6],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(5, 3, fft_direction, 15),
                    bf3: $internal_bf3::new(fft_direction),
                    bf5: $internal_bf5::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 15);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows0: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                let mut rows1: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                let mut rows2: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                // columns
                for i in 0..3 {
                    rows0[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 5..));
                    rows1[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 5 + 2..));
                    rows2[i] = NeonStoreF::from_complex(chunk.index(i * 5 + 4));
                }

                rows0 = self.bf3.exec(rows0);
                rows1 = self.bf3.exec(rows1);
                rows2 = self.bf3.exec(rows2);

                for i in 1..3 {
                    rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 2]);
                    rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 4]);
                }

                let transposed = transpose_f32x2_5x3(rows0, rows1, rows2);

                let q0 = self.bf5.exec(transposed.0);
                let q1 = self.bf5.exec(transposed.1);

                for i in 0..5 {
                    q0[i].write(chunk.slice_from_mut(i * 3..));
                    q1[i].write_lo(chunk.slice_from_mut(i * 3 + 2..));
                }
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_r2c(&self, src: &[f32], dst: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(15) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        src.len(),
                        self.real_length(),
                    ));
                }
                if !dst.len().is_multiple_of(8) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        dst.len(),
                        self.complex_length(),
                    ));
                }
                if src.len() / 15 != dst.len() / 8 {
                    return Err(ZaftError::InvalidSamplesCount(
                        src.len() / 15,
                        dst.len() / 8,
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                    let mut rows1: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                    let mut rows2: [NeonStoreF; 3] = [NeonStoreF::default(); 3];

                    for (chunk, dst) in src.chunks_exact(15).zip(dst.chunks_exact_mut(8)) {
                        // columns
                        for i in 0..3 {
                            let q0 = NeonStoreF::load(chunk.get_unchecked(i * 5..));
                            let q1 = NeonStoreF::load1(chunk.get_unchecked(i * 5 + 4..));
                            let [v0, v1] = q0.to_complex();
                            rows0[i] = v0;
                            rows1[i] = v1;
                            rows2[i] = q1;
                        }

                        rows0 = self.bf3.exec(rows0);
                        rows1 = self.bf3.exec(rows1);
                        rows2 = self.bf3.exec(rows2);

                        for i in 1..3 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 2]);
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 4]);
                        }

                        let transposed = transpose_f32x2_5x3(rows0, rows1, rows2);

                        let q0 = self.bf5.exec(transposed.0);
                        let q1 = self.bf5.exec(transposed.1);

                        for i in 0..2 {
                            q0[i].write(dst.get_unchecked_mut(i * 3..));
                            q1[i].write_lo(dst.get_unchecked_mut(i * 3 + 2..));
                        }

                        q0[2].write(dst.get_unchecked_mut(6..));
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
                15
            }

            #[inline]
            fn complex_length(&self) -> usize {
                8
            }

            fn complex_scratch_length(&self) -> usize {
                0
            }
        }
    };
}

gen_bf15f!(
    NeonButterfly15f,
    "neon",
    ColumnButterfly5f,
    ColumnButterfly3f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf15f!(
    NeonFcmaButterfly15f,
    "fcma",
    ColumnFcmaButterfly5f,
    ColumnFcmaButterfly3f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;
    use crate::r2c::test_r2c_butterfly;

    test_r2c_butterfly!(test_neon_r2c_butterfly15, f32, NeonButterfly15f, 15, 1e-5);
    test_butterfly!(test_neon_butterfly15, f32, NeonButterfly15f, 15, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly15, f32, NeonFcmaButterfly15f, 15, 1e-5);
    test_butterfly!(test_neon_butterfly15_f64, f64, NeonButterfly15d, 15, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly15_f64, f64, NeonButterfly15d, 15, 1e-7);
}
