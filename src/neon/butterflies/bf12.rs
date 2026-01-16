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
use crate::{FftDirection, FftExecutor, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

macro_rules! get_bf12d {
    ($name: ident, $features: literal, $internal_bf4: ident, $internal_bf3: ident) => {
        use crate::neon::mixed::{$internal_bf3, $internal_bf4};
        pub(crate) struct $name {
            direction: FftDirection,
            bf3: $internal_bf3,
            bf4: $internal_bf4,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf3: $internal_bf3::new(fft_direction),
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
                12
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(12) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    for chunk in in_place.chunks_exact_mut(12) {
                        let u0 = NeonStoreD::from_complex_ref(chunk);
                        let u1 = NeonStoreD::from_complex_ref(chunk.get_unchecked(3..));
                        let u2 = NeonStoreD::from_complex_ref(chunk.get_unchecked(6..));
                        let u3 = NeonStoreD::from_complex_ref(chunk.get_unchecked(9..));

                        let u4 = NeonStoreD::from_complex_ref(chunk.get_unchecked(4..));
                        let u5 = NeonStoreD::from_complex_ref(chunk.get_unchecked(7..));
                        let u6 = NeonStoreD::from_complex_ref(chunk.get_unchecked(10..));
                        let u7 = NeonStoreD::from_complex_ref(chunk.get_unchecked(1..));

                        let u8 = NeonStoreD::from_complex_ref(chunk.get_unchecked(8..));
                        let u9 = NeonStoreD::from_complex_ref(chunk.get_unchecked(11..));
                        let u10 = NeonStoreD::from_complex_ref(chunk.get_unchecked(2..));
                        let u11 = NeonStoreD::from_complex_ref(chunk.get_unchecked(5..));

                        let [u0, u1, u2, u3] = self.bf4.exec([u0, u1, u2, u3]);
                        let [u4, u5, u6, u7] = self.bf4.exec([u4, u5, u6, u7]);
                        let [u8, u9, u10, u11] = self.bf4.exec([u8, u9, u10, u11]);

                        let [y0, y4, y8] = self.bf3.exec([u0, u4, u8]);
                        let [y9, y1, y5] = self.bf3.exec([u1, u5, u9]);
                        let [y6, y10, y2] = self.bf3.exec([u2, u6, u10]);
                        let [y3, y7, y11] = self.bf3.exec([u3, u7, u11]);

                        vst1q_f64(chunk.as_mut_ptr().cast(), y0.v);
                        vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1.v);
                        vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2.v);
                        vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3.v);
                        vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4.v);
                        vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y5.v);
                        vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y6.v);
                        vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y7.v);
                        vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y8.v);
                        vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y9.v);
                        vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10.v);
                        vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), y11.v);
                    }
                }
                Ok(())
            }
        }
    };
}

get_bf12d!(
    NeonButterfly12d,
    "neon",
    ColumnButterfly4d,
    ColumnButterfly3d
);
#[cfg(feature = "fcma")]
get_bf12d!(
    NeonFcmaButterfly12d,
    "fcma",
    ColumnFcmaButterfly4d,
    ColumnFcmaButterfly3d
);

#[inline(always)]
pub(crate) fn transpose_f32x2_4x3(
    rows0: [NeonStoreF; 3],
    rows1: [NeonStoreF; 3],
) -> ([NeonStoreF; 4], [NeonStoreF; 4]) {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[2].v, unsafe { vdupq_n_f32(0.) }));

    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    let e0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[2].v, unsafe { vdupq_n_f32(0.) }));
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

macro_rules! gen_bf12f {
    ($name: ident, $features: literal, $internal_bf4: ident, $internal_bf3: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf3, $internal_bf4};
        pub(crate) struct $name {
            direction: FftDirection,
            bf3: $internal_bf3,
            bf4: $internal_bf4,
            twiddles: [NeonStoreF; 4],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(4, 3, fft_direction, 12),
                    bf3: $internal_bf3::new(fft_direction),
                    bf4: $internal_bf4::new(fft_direction),
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
                12
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(12) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                    let mut rows1: [NeonStoreF; 3] = [NeonStoreF::default(); 3];

                    for chunk in in_place.chunks_exact_mut(12) {
                        // columns
                        for i in 0..3 {
                            rows0[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 4..));
                            rows1[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 4 + 2..));
                        }

                        rows0 = self.bf3.exec(rows0);
                        rows1 = self.bf3.exec(rows1);

                        for i in 1..3 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 2]);
                        }

                        let transposed = transpose_f32x2_4x3(rows0, rows1);

                        let q0 = self.bf4.exec(transposed.0);
                        let q1 = self.bf4.exec(transposed.1);

                        for i in 0..4 {
                            q0[i].write(chunk.get_unchecked_mut(i * 3..));
                            q1[i].write_lo(chunk.get_unchecked_mut(i * 3 + 2..));
                        }
                    }
                }
                Ok(())
            }

            #[target_feature(enable = $features)]
            fn execute_r2c(&self, src: &[f32], dst: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(12) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        src.len(),
                        self.real_length(),
                    ));
                }

                if !dst.len().is_multiple_of(7) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        dst.len(),
                        self.complex_length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                    let mut rows1: [NeonStoreF; 3] = [NeonStoreF::default(); 3];

                    for (chunk, complex) in src.chunks_exact(12).zip(dst.chunks_exact_mut(7)) {
                        // columns
                        for i in 0..3 {
                            let q = NeonStoreF::load(chunk.get_unchecked(i * 4..));
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

                        let transposed = transpose_f32x2_4x3(rows0, rows1);

                        let q0 = self.bf4.exec(transposed.0);
                        let q1 = self.bf4.exec(transposed.1);

                        for i in 0..2 {
                            q0[i].write(complex.get_unchecked_mut(i * 3..));
                            q1[i].write_lo(complex.get_unchecked_mut(i * 3 + 2..));
                        }

                        q0[2].write_lo(complex.get_unchecked_mut(6..));
                    }
                }
                Ok(())
            }
        }

        impl R2CFftExecutor<f32> for $name {
            fn execute(&self, input: &[f32], output: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                unsafe { self.execute_r2c(input, output) }
            }

            #[inline]
            fn real_length(&self) -> usize {
                12
            }

            #[inline]
            fn complex_length(&self) -> usize {
                7
            }
        }
    };
}

gen_bf12f!(
    NeonButterfly12f,
    "neon",
    ColumnButterfly4f,
    ColumnButterfly3f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf12f!(
    NeonFcmaButterfly12f,
    "fcma",
    ColumnFcmaButterfly4f,
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

    test_r2c_butterfly!(test_neon_r2c_butterfly12, f32, NeonButterfly12f, 12, 1e-5);
    test_butterfly!(test_neon_butterfly12, f32, NeonButterfly12f, 12, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly12, f32, NeonFcmaButterfly12f, 12, 1e-5);
    test_butterfly!(test_neon_butterfly12_f64, f64, NeonButterfly12d, 12, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly12_f64,
        f64,
        NeonFcmaButterfly12d,
        12,
        1e-7
    );
}
