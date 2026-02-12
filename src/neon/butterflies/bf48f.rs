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

use crate::neon::butterflies::shared::{boring_neon_butterfly, gen_butterfly_twiddles_f32};
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::neon_transpose_f32x2_2x2_impl;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::float32x4x2_t;
use std::mem::MaybeUninit;

#[inline(always)]
pub(crate) fn transpose_6x4(
    rows0: [NeonStoreF; 4],
    rows1: [NeonStoreF; 4],
    rows2: [NeonStoreF; 4],
) -> [NeonStoreF; 12] {
    let a0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[0].v, rows0[1].v));
    let d0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows0[2].v, rows0[3].v));
    let b0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[0].v, rows1[1].v));
    let f0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows1[2].v, rows1[3].v));
    let c0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[0].v, rows2[1].v));
    let g0 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(rows2[2].v, rows2[3].v));
    [
        // row 0
        NeonStoreF::raw(a0.0),
        NeonStoreF::raw(a0.1),
        NeonStoreF::raw(b0.0),
        NeonStoreF::raw(b0.1),
        NeonStoreF::raw(c0.0),
        NeonStoreF::raw(c0.1),
        NeonStoreF::raw(d0.0),
        NeonStoreF::raw(d0.1),
        NeonStoreF::raw(f0.0),
        NeonStoreF::raw(f0.1),
        NeonStoreF::raw(g0.0),
        NeonStoreF::raw(g0.1),
    ]
}

macro_rules! gen_bf48f {
    ($name: ident, $features: literal, $internal_bf4: ident, $internal_bf12: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf4, $internal_bf12};
        pub(crate) struct $name {
            direction: FftDirection,
            bf4: $internal_bf4,
            bf12: $internal_bf12,
            twiddles: [NeonStoreF; 18],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(12, 4, fft_direction, 48),
                    bf12: $internal_bf12::new(fft_direction),
                    bf4: $internal_bf4::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 48);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows0: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                let mut rows1: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                let mut rows2: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                let mut rows12: [NeonStoreF; 12] = [NeonStoreF::default(); 12];

                let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 48];
                unsafe {
                    // columns
                    for k in 0..2 {
                        for i in 0..4 {
                            rows0[i] =
                                NeonStoreF::from_complex_ref(chunk.slice_from(i * 12 + k * 6..));
                            rows1[i] = NeonStoreF::from_complex_ref(
                                chunk.slice_from(i * 12 + k * 6 + 2..),
                            );
                            rows2[i] = NeonStoreF::from_complex_ref(
                                chunk.slice_from(i * 12 + k * 6 + 4..),
                            );
                        }

                        rows0 = self.bf4.exec(rows0);
                        rows1 = self.bf4.exec(rows1);
                        rows2 = self.bf4.exec(rows2);

                        for i in 1..4 {
                            rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1 + 9 * k]);
                            rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 9 * k + 3]);
                            rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 9 * k + 6]);
                        }

                        let transposed = transpose_6x4(rows0, rows1, rows2);

                        for i in 0..6 {
                            transposed[i].write_uninit(scratch.get_unchecked_mut(i * 4 + k * 24..));
                            transposed[i + 6]
                                .write_uninit(scratch.get_unchecked_mut(i * 4 + k * 24 + 2..));
                        }
                    }

                    // rows

                    for k in 0..2 {
                        for i in 0..12 {
                            rows12[i] = NeonStoreF::from_complex_refu(
                                scratch.get_unchecked(i * 4 + k * 2..),
                            );
                        }
                        rows12 = self.bf12.exec(rows12);
                        for i in 0..12 {
                            rows12[i].write(chunk.slice_from_mut(i * 4 + k * 2..));
                        }
                    }
                }
            }
        }
    };
}

gen_bf48f!(
    NeonButterfly48f,
    "neon",
    ColumnButterfly4f,
    ColumnButterfly12f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf48f!(
    NeonFcmaButterfly48f,
    "fcma",
    ColumnFcmaButterfly4f,
    ColumnFcmaButterfly12f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly48, f32, NeonButterfly48f, 48, 1e-3);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly48, f32, NeonFcmaButterfly48f, 48, 1e-3);
}
