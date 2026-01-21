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
use crate::neon::butterflies::shared::{gen_butterfly_twiddles_f32, gen_butterfly_twiddles_f64};
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use crate::{
    CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, R2CFftExecutor,
    ZaftError,
};
use num_complex::Complex;
use std::arch::aarch64::*;
use std::mem::MaybeUninit;
use std::sync::Arc;

macro_rules! gen_bf32d {
    ($name: ident, $feature: literal, $internal_bf8: ident, $internal_bf4: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf4, $internal_bf8};
        pub(crate) struct $name {
            direction: FftDirection,
            bf4: $internal_bf4,
            bf8: $internal_bf8,
            twiddles: [NeonStoreD; 24],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(8, 4, fft_direction, 32),
                    bf8: $internal_bf8::new(fft_direction),
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
                32
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(32) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreD; 4] = [NeonStoreD::default(); 4];
                    let mut rows8: [NeonStoreD; 8] = [NeonStoreD::default(); 8];

                    let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 32];

                    for chunk in in_place.chunks_exact_mut(32) {
                        for k in 0..8 {
                            rows0[0] = NeonStoreD::from_complex_ref(chunk.get_unchecked(k..));
                            rows0[1] = NeonStoreD::from_complex_ref(chunk.get_unchecked(8 + k..));
                            rows0[2] =
                                NeonStoreD::from_complex_ref(chunk.get_unchecked(2 * 8 + k..));
                            rows0[3] =
                                NeonStoreD::from_complex_ref(chunk.get_unchecked(3 * 8 + k..));

                            rows0 = self.bf4.exec(rows0);

                            for i in 1..4 {
                                rows0[i] = NeonStoreD::$mul(rows0[i], self.twiddles[i - 1 + 3 * k]);
                            }

                            rows0[0].write_uninit(scratch.get_unchecked_mut(k * 4..));
                            rows0[1].write_uninit(scratch.get_unchecked_mut(k * 4 + 1..));
                            rows0[2].write_uninit(scratch.get_unchecked_mut(k * 4 + 2..));
                            rows0[3].write_uninit(scratch.get_unchecked_mut(k * 4 + 3..));
                        }

                        // rows

                        for k in 0..4 {
                            for i in 0..8 {
                                rows8[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 4 + k..),
                                );
                            }
                            rows8 = self.bf8.exec(rows8);
                            for i in 0..8 {
                                rows8[i].write(chunk.get_unchecked_mut(i * 4 + k..));
                            }
                        }
                    }
                }
                Ok(())
            }

            #[target_feature(enable = $feature)]
            fn execute_out_of_place_impl(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(32) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(32) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    let mut rows0: [NeonStoreD; 4] = [NeonStoreD::default(); 4];
                    let mut rows8: [NeonStoreD; 8] = [NeonStoreD::default(); 8];

                    let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 32];

                    for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
                        for k in 0..8 {
                            rows0[0] = NeonStoreD::from_complex_ref(src.get_unchecked(k..));
                            rows0[1] = NeonStoreD::from_complex_ref(src.get_unchecked(8 + k..));
                            rows0[2] = NeonStoreD::from_complex_ref(src.get_unchecked(2 * 8 + k..));
                            rows0[3] = NeonStoreD::from_complex_ref(src.get_unchecked(3 * 8 + k..));

                            rows0 = self.bf4.exec(rows0);

                            for i in 1..4 {
                                rows0[i] = NeonStoreD::$mul(rows0[i], self.twiddles[i - 1 + 3 * k]);
                            }

                            rows0[0].write_uninit(scratch.get_unchecked_mut(k * 4..));
                            rows0[1].write_uninit(scratch.get_unchecked_mut(k * 4 + 1..));
                            rows0[2].write_uninit(scratch.get_unchecked_mut(k * 4 + 2..));
                            rows0[3].write_uninit(scratch.get_unchecked_mut(k * 4 + 3..));
                        }

                        // rows

                        for k in 0..4 {
                            for i in 0..8 {
                                rows8[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 4 + k..),
                                );
                            }
                            rows8 = self.bf8.exec(rows8);
                            for i in 0..8 {
                                rows8[i].write(dst.get_unchecked_mut(i * 4 + k..));
                            }
                        }
                    }
                }
                Ok(())
            }

            #[target_feature(enable = $feature)]
            fn execute_r2c(&self, src: &[f64], dst: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(32) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        src.len(),
                        self.real_length(),
                    ));
                }
                if !dst.len().is_multiple_of(17) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        dst.len(),
                        self.complex_length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreD; 4] = [NeonStoreD::default(); 4];
                    let mut rows1: [NeonStoreD; 4] = [NeonStoreD::default(); 4];
                    let mut rows8: [NeonStoreD; 8] = [NeonStoreD::default(); 8];

                    let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 32];

                    for (dst, src) in dst.chunks_exact_mut(17).zip(src.chunks_exact(32)) {
                        for k in 0..4 {
                            let [v0_0, v0_1] =
                                NeonStoreD::load(src.get_unchecked(k * 2..)).to_complex();
                            let [v1_0, v1_1] =
                                NeonStoreD::load(src.get_unchecked(8 + k * 2..)).to_complex();
                            let [v2_0, v2_1] =
                                NeonStoreD::load(src.get_unchecked(2 * 8 + k * 2..)).to_complex();
                            let [v3_0, v3_1] =
                                NeonStoreD::load(src.get_unchecked(3 * 8 + k * 2..)).to_complex();
                            rows0[0] = v0_0;
                            rows1[0] = v0_1;
                            rows0[1] = v1_0;
                            rows1[1] = v1_1;
                            rows0[2] = v2_0;
                            rows1[2] = v2_1;
                            rows0[3] = v3_0;
                            rows1[3] = v3_1;

                            rows0 = self.bf4.exec(rows0);
                            rows1 = self.bf4.exec(rows1);

                            for i in 1..4 {
                                rows0[i] =
                                    NeonStoreD::$mul(rows0[i], self.twiddles[i - 1 + 3 * k * 2]);
                                rows1[i] = NeonStoreD::$mul(
                                    rows1[i],
                                    self.twiddles[i - 1 + 3 * (k * 2 + 1)],
                                );
                            }

                            let qk = k * 2;

                            rows0[0].write_uninit(scratch.get_unchecked_mut(qk * 4..));
                            rows0[1].write_uninit(scratch.get_unchecked_mut(qk * 4 + 1..));
                            rows0[2].write_uninit(scratch.get_unchecked_mut(qk * 4 + 2..));
                            rows0[3].write_uninit(scratch.get_unchecked_mut(qk * 4 + 3..));

                            let qk2 = k * 2 + 1;

                            rows1[0].write_uninit(scratch.get_unchecked_mut(qk2 * 4..));
                            rows1[1].write_uninit(scratch.get_unchecked_mut(qk2 * 4 + 1..));
                            rows1[2].write_uninit(scratch.get_unchecked_mut(qk2 * 4 + 2..));
                            rows1[3].write_uninit(scratch.get_unchecked_mut(qk2 * 4 + 3..));
                        }

                        // rows

                        for k in 0..4 {
                            for i in 0..8 {
                                rows8[i] = NeonStoreD::from_complex_refu(
                                    scratch.get_unchecked(i * 4 + k..),
                                );
                            }
                            rows8 = self.bf8.exec(rows8);
                            for i in 0..4 {
                                rows8[i].write(dst.get_unchecked_mut(i * 4 + k..));
                            }
                            if k == 0 {
                                rows8[4].write(dst.get_unchecked_mut(16..));
                            }
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

            fn real_length(&self) -> usize {
                32
            }

            fn complex_length(&self) -> usize {
                17
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

gen_bf32d!(
    NeonButterfly32d,
    "neon",
    ColumnButterfly8d,
    ColumnButterfly4d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf32d!(
    NeonFcmaButterfly32d,
    "fcma",
    ColumnFcmaButterfly8d,
    ColumnFcmaButterfly4d,
    fcmul_fcma
);

#[inline(always)]
pub(crate) fn transpose_8x4_to_4x8_f32(
    rows0: [NeonStoreF; 4],
    rows1: [NeonStoreF; 4],
    rows2: [NeonStoreF; 4],
    rows3: [NeonStoreF; 4],
) -> ([NeonStoreF; 8], [NeonStoreF; 8]) {
    let output00 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let output01 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    let output02 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[0].v, rows2[1].v));
    let output03 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows3[0].v, rows3[1].v));
    let output10 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[2].v, rows0[3].v));
    let output11 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[2].v, rows1[3].v));
    let output12 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[2].v, rows2[3].v));
    let output13 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows3[2].v, rows3[3].v));

    (
        [
            NeonStoreF::raw(output00.0),
            NeonStoreF::raw(output00.1),
            NeonStoreF::raw(output01.0),
            NeonStoreF::raw(output01.1),
            NeonStoreF::raw(output02.0),
            NeonStoreF::raw(output02.1),
            NeonStoreF::raw(output03.0),
            NeonStoreF::raw(output03.1),
        ],
        [
            NeonStoreF::raw(output10.0),
            NeonStoreF::raw(output10.1),
            NeonStoreF::raw(output11.0),
            NeonStoreF::raw(output11.1),
            NeonStoreF::raw(output12.0),
            NeonStoreF::raw(output12.1),
            NeonStoreF::raw(output13.0),
            NeonStoreF::raw(output13.1),
        ],
    )
}

macro_rules! gen_bf32f {
    ($name: ident, $feature: literal, $internal_bf8: ident, $internal_bf4: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf4, $internal_bf8};
        pub(crate) struct $name {
            direction: FftDirection,
            bf8: $internal_bf8,
            bf4: $internal_bf4,
            twiddles: [NeonStoreF; 12],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf8: $internal_bf8::new(fft_direction),
                    bf4: $internal_bf4::new(fft_direction),
                    twiddles: gen_butterfly_twiddles_f32(8, 4, fft_direction, 32),
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
                32
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(32) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows1: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows2: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows3: [NeonStoreF; 4] = [NeonStoreF::default(); 4];

                    for chunk in in_place.chunks_exact_mut(32) {
                        for i in 0..4 {
                            rows0[i] = NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 8..));
                            rows1[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 8 + 2..));
                            rows2[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 8 + 4..));
                            rows3[i] =
                                NeonStoreF::from_complex_ref(chunk.get_unchecked(i * 8 + 6..));
                        }

                        rows0 = self.bf4.exec(rows0);
                        rows1 = self.bf4.exec(rows1);
                        rows2 = self.bf4.exec(rows2);
                        rows3 = self.bf4.exec(rows3);

                        for i in 1..4 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 3]);
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 6]);
                            rows3[i] = NeonStoreF::$mul(rows3[i], self.twiddles[i - 1 + 9]);
                        }

                        let (mut q0, mut q1) = transpose_8x4_to_4x8_f32(rows0, rows1, rows2, rows3);

                        q0 = self.bf8.exec(q0);
                        q1 = self.bf8.exec(q1);

                        for i in 0..8 {
                            q0[i].write(chunk.get_unchecked_mut(i * 4..));
                            q1[i].write(chunk.get_unchecked_mut(i * 4 + 2..));
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
                if !src.len().is_multiple_of(32) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(32) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows1: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows2: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows3: [NeonStoreF; 4] = [NeonStoreF::default(); 4];

                    for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
                        for i in 0..4 {
                            rows0[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 8..));
                            rows1[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 8 + 2..));
                            rows2[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 8 + 4..));
                            rows3[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * 8 + 6..));
                        }

                        rows0 = self.bf4.exec(rows0);
                        rows1 = self.bf4.exec(rows1);
                        rows2 = self.bf4.exec(rows2);
                        rows3 = self.bf4.exec(rows3);

                        for i in 1..4 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 3]);
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 6]);
                            rows3[i] = NeonStoreF::$mul(rows3[i], self.twiddles[i - 1 + 9]);
                        }

                        let (mut q0, mut q1) = transpose_8x4_to_4x8_f32(rows0, rows1, rows2, rows3);

                        q0 = self.bf8.exec(q0);
                        q1 = self.bf8.exec(q1);

                        for i in 0..8 {
                            q0[i].write(dst.get_unchecked_mut(i * 4..));
                            q1[i].write(dst.get_unchecked_mut(i * 4 + 2..));
                        }
                    }
                }
                Ok(())
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_r2c(&self, src: &[f32], dst: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(32) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(17) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    let mut rows0: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows1: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows2: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                    let mut rows3: [NeonStoreF; 4] = [NeonStoreF::default(); 4];

                    for (dst, src) in dst.chunks_exact_mut(17).zip(src.chunks_exact(32)) {
                        for i in 0..4 {
                            let s0 = NeonStoreF::load(src.get_unchecked(i * 8..));
                            let s1 = NeonStoreF::load(src.get_unchecked(i * 8 + 4..));
                            let [u0, u1] = s0.to_complex();
                            let [u2, u3] = s1.to_complex();
                            rows0[i] = u0;
                            rows1[i] = u1;
                            rows2[i] = u2;
                            rows3[i] = u3;
                        }

                        rows0 = self.bf4.exec(rows0);
                        rows1 = self.bf4.exec(rows1);
                        rows2 = self.bf4.exec(rows2);
                        rows3 = self.bf4.exec(rows3);

                        for i in 1..4 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 3]);
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 6]);
                            rows3[i] = NeonStoreF::$mul(rows3[i], self.twiddles[i - 1 + 9]);
                        }

                        let (mut q0, mut q1) = transpose_8x4_to_4x8_f32(rows0, rows1, rows2, rows3);

                        q0 = self.bf8.exec(q0);
                        q1 = self.bf8.exec(q1);

                        for i in 0..4 {
                            q0[i].write(dst.get_unchecked_mut(i * 4..));
                            q1[i].write(dst.get_unchecked_mut(i * 4 + 2..));
                        }
                        q0[4].write_lo(dst.get_unchecked_mut(16..));
                    }
                }
                Ok(())
            }
        }

        impl R2CFftExecutor<f32> for $name {
            fn execute(&self, input: &[f32], output: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                unsafe { self.execute_r2c(input, output) }
            }

            fn real_length(&self) -> usize {
                32
            }

            fn complex_length(&self) -> usize {
                17
            }
        }

        impl CompositeFftExecutor<f32> for $name {
            fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
                self
            }
        }
    };
}

gen_bf32f!(
    NeonButterfly32f,
    "neon",
    ColumnButterfly8f,
    ColumnButterfly4f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf32f!(
    NeonFcmaButterfly32f,
    "fcma",
    ColumnFcmaButterfly8f,
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

    test_r2c_butterfly!(test_neon_r2c_butterfly32, f32, NeonButterfly32f, 32, 1e-5);
    test_r2c_butterfly!(test_neon_r2c_butterfly32d, f64, NeonButterfly32d, 32, 1e-5);
    test_butterfly!(test_neon_butterfly32, f32, NeonButterfly32f, 32, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly32, f32, NeonFcmaButterfly32f, 32, 1e-5);
    test_butterfly!(test_neon_butterfly32_f64, f64, NeonButterfly32d, 32, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly32_f64,
        f64,
        NeonFcmaButterfly32d,
        32,
        1e-7
    );
    test_oof_butterfly!(test_oof_butterfly32, f32, NeonButterfly32f, 32, 1e-5);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly32,
        f32,
        NeonFcmaButterfly32f,
        32,
        1e-5
    );
    test_oof_butterfly!(test_oof_butterfly32_f64, f64, NeonButterfly32d, 32, 1e-9);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly32_f64,
        f64,
        NeonButterfly32d,
        32,
        1e-9
    );
}
