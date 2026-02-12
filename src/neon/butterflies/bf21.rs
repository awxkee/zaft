/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::neon::transpose::transpose_f32x2_4x4;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

#[inline(always)]
fn transpose_7x3(
    rows0: [NeonStoreF; 3],
    rows1: [NeonStoreF; 3],
    rows2: [NeonStoreF; 3],
    rows3: [NeonStoreF; 3],
) -> ([NeonStoreF; 7], [NeonStoreF; 7]) {
    let a0 = transpose_f32x2_4x4(
        float32x4x2_t(rows0[0].v, rows1[0].v),
        float32x4x2_t(rows0[1].v, rows1[1].v),
        float32x4x2_t(rows0[2].v, rows1[2].v),
        unsafe { float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)) },
    );
    let b0 = transpose_f32x2_4x4(
        float32x4x2_t(rows2[0].v, rows3[0].v),
        float32x4x2_t(rows2[1].v, rows3[1].v),
        float32x4x2_t(rows2[2].v, rows3[2].v),
        unsafe { float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)) },
    );
    (
        [
            // row 0
            NeonStoreF::raw(a0.0.0),
            NeonStoreF::raw(a0.1.0),
            NeonStoreF::raw(a0.2.0),
            NeonStoreF::raw(a0.3.0),
            NeonStoreF::raw(b0.0.0),
            NeonStoreF::raw(b0.1.0),
            NeonStoreF::raw(b0.2.0),
        ],
        [
            // row 0
            NeonStoreF::raw(a0.0.1),
            NeonStoreF::raw(a0.1.1),
            NeonStoreF::raw(a0.2.1),
            NeonStoreF::raw(a0.3.1),
            NeonStoreF::raw(b0.0.1),
            NeonStoreF::raw(b0.1.1),
            NeonStoreF::raw(b0.2.1),
        ],
    )
}

macro_rules! gen_bf24d {
    ($name: ident, $features: literal, $internal_bf7: ident, $internal_bf3: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf3, $internal_bf7};
        pub(crate) struct $name {
            direction: FftDirection,
            bf7: $internal_bf7,
            bf3: $internal_bf3,
            twiddles: [NeonStoreD; 21],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                let mut twiddles = [NeonStoreD::default(); 21];
                for (x, row) in twiddles.chunks_exact_mut(3).enumerate() {
                    for (y, dst) in row.iter_mut().enumerate() {
                        *dst = NeonStoreD::from_complex(&compute_twiddle(
                            (x + 1) * y,
                            21,
                            fft_direction,
                        ));
                    }
                }
                Self {
                    direction: fft_direction,
                    twiddles,
                    bf7: $internal_bf7::new(fft_direction),
                    bf3: $internal_bf3::new(fft_direction),
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
                let u8 = NeonStoreD::from_complex_ref(chunk.slice_from(8..));
                let u9 = NeonStoreD::from_complex_ref(chunk.slice_from(9..));
                let u10 = NeonStoreD::from_complex_ref(chunk.slice_from(10..));
                let u11 = NeonStoreD::from_complex_ref(chunk.slice_from(11..));
                let u12 = NeonStoreD::from_complex_ref(chunk.slice_from(12..));
                let u13 = NeonStoreD::from_complex_ref(chunk.slice_from(13..));
                let u14 = NeonStoreD::from_complex_ref(chunk.slice_from(14..));
                let u15 = NeonStoreD::from_complex_ref(chunk.slice_from(15..));
                let u16 = NeonStoreD::from_complex_ref(chunk.slice_from(16..));
                let u17 = NeonStoreD::from_complex_ref(chunk.slice_from(17..));
                let u18 = NeonStoreD::from_complex_ref(chunk.slice_from(18..));
                let u19 = NeonStoreD::from_complex_ref(chunk.slice_from(19..));
                let u20 = NeonStoreD::from_complex_ref(chunk.slice_from(20..));

                let s0 = self.bf3.exec([u0, u7, u14]);
                let mut s1 = self.bf3.exec([u1, u8, u15]);
                let mut s2 = self.bf3.exec([u2, u9, u16]);
                let mut s3 = self.bf3.exec([u3, u10, u17]);
                let mut s4 = self.bf3.exec([u4, u11, u18]);
                let mut s5 = self.bf3.exec([u5, u12, u19]);
                let mut s6 = self.bf3.exec([u6, u13, u20]);
                for i in 0..3 {
                    s1[i] = NeonStoreD::$mul(s1[i], self.twiddles[i]);
                    s2[i] = NeonStoreD::$mul(s2[i], self.twiddles[3 + i]);
                    s3[i] = NeonStoreD::$mul(s3[i], self.twiddles[6 + i]);
                    s4[i] = NeonStoreD::$mul(s4[i], self.twiddles[9 + i]);
                    s5[i] = NeonStoreD::$mul(s5[i], self.twiddles[12 + i]);
                    s6[i] = NeonStoreD::$mul(s6[i], self.twiddles[15 + i]);
                }

                let z0 = self
                    .bf7
                    .exec([s0[0], s1[0], s2[0], s3[0], s4[0], s5[0], s6[0]]);
                let z1 = self
                    .bf7
                    .exec([s0[1], s1[1], s2[1], s3[1], s4[1], s5[1], s6[1]]);
                let z2 = self
                    .bf7
                    .exec([s0[2], s1[2], s2[2], s3[2], s4[2], s5[2], s6[2]]);
                for i in 0..7 {
                    z0[i].write(chunk.slice_from_mut(i * 3..));
                    z1[i].write(chunk.slice_from_mut(i * 3 + 1..));
                    z2[i].write(chunk.slice_from_mut(i * 3 + 2..));
                }
            }
        }

        boring_neon_butterfly!($name, $features, f64, 21);
    };
}

macro_rules! gen_bf21f {
    ($name: ident, $features: literal, $internal_bf7: ident, $internal_bf3: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf3, $internal_bf7};
        pub(crate) struct $name {
            direction: FftDirection,
            bf3: $internal_bf3,
            bf7: $internal_bf7,
            twiddles: [NeonStoreF; 8],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(7, 3, fft_direction, 21),
                    bf3: $internal_bf3::new(fft_direction),
                    bf7: $internal_bf7::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 21);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows0: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                let mut rows1: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                let mut rows2: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                let mut rows3: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                for i in 0..3 {
                    rows0[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 7..));
                    rows1[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 7 + 2..));
                    rows2[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 7 + 4..));
                    rows3[i] = NeonStoreF::from_complex(chunk.index(i * 7 + 6));
                }

                rows0 = self.bf3.exec(rows0);
                rows1 = self.bf3.exec(rows1);
                rows2 = self.bf3.exec(rows2);
                rows3 = self.bf3.exec(rows3);

                for i in 1..3 {
                    rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 2]);
                    rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 4]);
                    rows3[i] = NeonStoreF::$mul(rows3[i], self.twiddles[i - 1 + 6]);
                }

                let transposed = transpose_7x3(rows0, rows1, rows2, rows3);

                let q0 = self.bf7.exec(transposed.0);
                let q1 = self.bf7.exec(transposed.1);

                for i in 0..7 {
                    q0[i].write(chunk.slice_from_mut(i * 3..));
                    q1[i].write_lo(chunk.slice_from_mut(i * 3 + 2..));
                }
            }
        }
    };
}

gen_bf24d!(
    NeonButterfly21d,
    "neon",
    ColumnButterfly7d,
    ColumnButterfly3d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf24d!(
    NeonFcmaButterfly21d,
    "fcma",
    ColumnFcmaButterfly7d,
    ColumnFcmaButterfly3d,
    fcmul_fcma
);

gen_bf21f!(
    NeonButterfly21f,
    "neon",
    ColumnButterfly7f,
    ColumnButterfly3f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf21f!(
    NeonFcmaButterfly21f,
    "fcma",
    ColumnFcmaButterfly7f,
    ColumnFcmaButterfly3f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly21, f32, NeonButterfly21f, 21, 1e-5);
    test_butterfly!(test_neon_butterfly21_f64, f64, NeonButterfly21d, 21, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly21_f64,
        f64,
        NeonFcmaButterfly21d,
        21,
        1e-7
    );
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly21, f32, NeonFcmaButterfly21f, 21, 1e-5);
}
