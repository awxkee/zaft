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
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::neon::transpose::transpose_2x5;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

macro_rules! gen_bf25d {
    ($name: ident, $features: literal, $internal_bf: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf5: $internal_bf,
            twiddles: [NeonStoreD; 20],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                let mut twiddles = [NeonStoreD::default(); 20];
                for (x, row) in twiddles.chunks_exact_mut(5).enumerate() {
                    for (y, dst) in row.iter_mut().enumerate() {
                        *dst = NeonStoreD::from_complex(&compute_twiddle(
                            (x + 1) * y,
                            25,
                            fft_direction,
                        ));
                    }
                }
                Self {
                    direction: fft_direction,
                    twiddles,
                    bf5: $internal_bf::new(fft_direction),
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
                let u21 = NeonStoreD::from_complex_ref(chunk.slice_from(21..));
                let u22 = NeonStoreD::from_complex_ref(chunk.slice_from(22..));
                let u23 = NeonStoreD::from_complex_ref(chunk.slice_from(23..));
                let u24 = NeonStoreD::from_complex_ref(chunk.slice_from(24..));

                let s0 = self.bf5.exec([u0, u5, u10, u15, u20]);
                let mut s1 = self.bf5.exec([u1, u6, u11, u16, u21]);
                let mut s2 = self.bf5.exec([u2, u7, u12, u17, u22]);
                for i in 0..5 {
                    s1[i] = NeonStoreD::$mul(s1[i], self.twiddles[i]);
                    s2[i] = NeonStoreD::$mul(s2[i], self.twiddles[5 + i]);
                }
                let mut s3 = self.bf5.exec([u3, u8, u13, u18, u23]);
                let mut s4 = self.bf5.exec([u4, u9, u14, u19, u24]);
                for i in 0..5 {
                    s3[i] = NeonStoreD::$mul(s3[i], self.twiddles[10 + i]);
                    s4[i] = NeonStoreD::$mul(s4[i], self.twiddles[15 + i]);
                }

                let z0 = self.bf5.exec([s0[0], s1[0], s2[0], s3[0], s4[0]]);
                let z1 = self.bf5.exec([s0[1], s1[1], s2[1], s3[1], s4[1]]);
                let z2 = self.bf5.exec([s0[2], s1[2], s2[2], s3[2], s4[2]]);
                for i in 0..5 {
                    z0[i].write(chunk.slice_from_mut(i * 5..));
                    z1[i].write(chunk.slice_from_mut(i * 5 + 1..));
                    z2[i].write(chunk.slice_from_mut(i * 5 + 2..));
                }
                let z3 = self.bf5.exec([s0[3], s1[3], s2[3], s3[3], s4[3]]);
                let z4 = self.bf5.exec([s0[4], s1[4], s2[4], s3[4], s4[4]]);
                for i in 0..5 {
                    z3[i].write(chunk.slice_from_mut(i * 5 + 3..));
                    z4[i].write(chunk.slice_from_mut(i * 5 + 4..));
                }
            }
        }

        boring_neon_butterfly!($name, $features, f64, 25);
    };
}

gen_bf25d!(NeonButterfly25d, "neon", ColumnButterfly5d, mul_by_complex);
#[cfg(feature = "fcma")]
gen_bf25d!(
    NeonFcmaButterfly25d,
    "fcma",
    ColumnFcmaButterfly5d,
    fcmul_fcma
);

#[inline]
fn transpose_5x5_f32(
    rows0: [NeonStoreF; 5],
    rows1: [NeonStoreF; 5],
    rows2: [NeonStoreF; 5],
) -> ([NeonStoreF; 5], [NeonStoreF; 5], [NeonStoreF; 5]) {
    let transposed00 = transpose_2x5(rows0);
    let transposed01 = transpose_2x5(rows1);
    let transposed10 = transpose_2x5(rows2);

    (
        [
            transposed00[0],
            transposed00[1],
            transposed01[0],
            transposed01[1],
            transposed10[0],
        ],
        [
            transposed00[2],
            transposed00[3],
            transposed01[2],
            transposed01[3],
            transposed10[2],
        ],
        [
            transposed00[4],
            transposed00[5],
            transposed01[4],
            transposed01[5],
            transposed10[4],
        ],
    )
}

macro_rules! gen_bf25f {
    ( $ name: ident, $ features: literal, $ internal_bf: ident, $ mul: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf5: $internal_bf,
            twiddles: [NeonStoreF; 12],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(5, 5, fft_direction, 25),
                    bf5: $internal_bf::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 25);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows0 = [NeonStoreF::default(); 5];
                let mut rows1 = [NeonStoreF::default(); 5];
                let mut rows2 = [NeonStoreF::default(); 5];
                for i in 0..5 {
                    rows0[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 5..));
                    rows1[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 5 + 2..));
                    rows2[i] = NeonStoreF::from_complex(chunk.index(i * 5 + 4));
                }

                rows0 = self.bf5.exec(rows0);
                rows1 = self.bf5.exec(rows1);
                rows2 = self.bf5.exec(rows2);

                for i in 1..5 {
                    rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 4]);
                    rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 8]);
                }

                let (mut q0, mut q1, mut q2) = transpose_5x5_f32(rows0, rows1, rows2);

                q0 = self.bf5.exec(q0);
                q1 = self.bf5.exec(q1);
                q2 = self.bf5.exec(q2);

                q0[0].write(chunk.slice_from_mut(0..));
                q0[1].write(chunk.slice_from_mut(5..));
                q0[2].write(chunk.slice_from_mut(10..));
                q0[3].write(chunk.slice_from_mut(15..));
                q0[4].write(chunk.slice_from_mut(20..));

                q1[0].write(chunk.slice_from_mut(2..));
                q1[1].write(chunk.slice_from_mut(7..));
                q1[2].write(chunk.slice_from_mut(12..));
                q1[3].write(chunk.slice_from_mut(17..));
                q1[4].write(chunk.slice_from_mut(22..));

                q2[0].write_lo(chunk.slice_from_mut(4..));
                q2[1].write_lo(chunk.slice_from_mut(9..));
                q2[2].write_lo(chunk.slice_from_mut(14..));
                q2[3].write_lo(chunk.slice_from_mut(19..));
                q2[4].write_lo(chunk.slice_from_mut(24..));
            }
        }
    };
}

gen_bf25f!(NeonButterfly25f, "neon", ColumnButterfly5f, mul_by_complex);
#[cfg(feature = "fcma")]
gen_bf25f!(
    NeonFcmaButterfly25f,
    "fcma",
    ColumnFcmaButterfly5f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly25, f32, NeonButterfly25f, 25, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly25, f32, NeonFcmaButterfly25f, 25, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly25_f64,
        f64,
        NeonFcmaButterfly25d,
        25,
        1e-5
    );
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly25,
        f32,
        NeonFcmaButterfly25f,
        25,
        1e-5
    );
    test_butterfly!(test_neon_butterfly25_f64, f64, NeonButterfly25d, 25, 1e-7);
    test_oof_butterfly!(test_oof_butterfly25, f32, NeonButterfly25f, 25, 1e-5);
    test_oof_butterfly!(test_oof_butterfly25_f64, f64, NeonButterfly25d, 25, 1e-9);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly25_f64,
        f64,
        NeonButterfly25d,
        25,
        1e-9
    );
}
