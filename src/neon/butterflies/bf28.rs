/*
 * // Copyright (c) Radzivon Bartoshyk 02/2026. All rights reserved.
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
use crate::neon::butterflies::shared::{
    boring_neon_butterfly, gen_butterfly_twiddles_f32, gen_butterfly_twiddles_f64,
};
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::neon::transpose::transpose_f32x2_4x4;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;
use std::mem::MaybeUninit;

macro_rules! gen_bf28d {
    ($name: ident, $features: literal, $internal_bf7: ident, $internal_bf4: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf4, $internal_bf7};
        pub(crate) struct $name {
            direction: FftDirection,
            bf4: $internal_bf4,
            bf7: $internal_bf7,
            twiddles: [NeonStoreD; 21],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(7, 4, fft_direction, 28),
                    bf7: $internal_bf7::new(fft_direction),
                    bf4: $internal_bf4::new(fft_direction),
                }
            }
        }

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
                let mut rows: [NeonStoreD; 4] = [NeonStoreD::default(); 4];
                let mut rows7: [NeonStoreD; 7] = [NeonStoreD::default(); 7];

                let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 28];

                unsafe {
                    // columns
                    for k in 0..7 {
                        for i in 0..4 {
                            rows[i] = NeonStoreD::from_complex_ref(chunk.slice_from(i * 7 + k..));
                        }

                        rows = self.bf4.exec(rows);

                        for i in 1..4 {
                            rows[i] = NeonStoreD::$mul(rows[i], self.twiddles[i - 1 + 3 * k]);
                        }

                        for i in 0..4 {
                            rows[i].write_uninit(scratch.get_unchecked_mut(k * 4 + i..));
                        }
                    }

                    // rows

                    for k in 0..4 {
                        for i in 0..7 {
                            rows7[i] =
                                NeonStoreD::from_complex_refu(scratch.get_unchecked(i * 4 + k..));
                        }
                        rows7 = self.bf7.exec(rows7);
                        for i in 0..7 {
                            rows7[i].write(chunk.slice_from_mut(i * 4 + k..));
                        }
                    }
                }
            }
        }

        boring_neon_butterfly!($name, $features, f64, 28);
    };
}

gen_bf28d!(
    NeonButterfly28d,
    "neon",
    ColumnButterfly7d,
    ColumnButterfly4d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf28d!(
    NeonFcmaButterfly28d,
    "fcma",
    ColumnFcmaButterfly7d,
    ColumnFcmaButterfly4d,
    fcmul_fcma
);

#[inline(always)]
fn transpose_7x4(
    rows0: [NeonStoreF; 4],
    rows1: [NeonStoreF; 4],
    rows2: [NeonStoreF; 4],
    rows3: [NeonStoreF; 4],
) -> ([NeonStoreF; 7], [NeonStoreF; 7]) {
    let a0 = transpose_f32x2_4x4(
        float32x4x2_t(rows0[0].v, rows1[0].v),
        float32x4x2_t(rows0[1].v, rows1[1].v),
        float32x4x2_t(rows0[2].v, rows1[2].v),
        float32x4x2_t(rows0[3].v, rows1[3].v),
    );
    let b0 = transpose_f32x2_4x4(
        float32x4x2_t(rows2[0].v, rows3[0].v),
        float32x4x2_t(rows2[1].v, rows3[1].v),
        float32x4x2_t(rows2[2].v, rows3[2].v),
        float32x4x2_t(rows2[3].v, rows3[3].v),
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

macro_rules! gen_bf28f {
    ($name: ident, $features: literal, $internal_bf4: ident, $internal_bf7: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf4, $internal_bf7};
        pub(crate) struct $name {
            direction: FftDirection,
            bf4: $internal_bf4,
            bf7: $internal_bf7,
            twiddles: [NeonStoreF; 12],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(7, 4, fft_direction, 28),
                    bf7: $internal_bf7::new(fft_direction),
                    bf4: $internal_bf4::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 28);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows0: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                let mut rows1: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                let mut rows2: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                let mut rows3: [NeonStoreF; 4] = [NeonStoreF::default(); 4];
                // columns
                for i in 0..4 {
                    rows0[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 7..));
                    rows1[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 7 + 2..));
                    rows2[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 7 + 4..));
                    rows3[i] = NeonStoreF::from_complex(chunk.index(i * 7 + 6));
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

                let t = transpose_7x4(rows0, rows1, rows2, rows3);

                // rows
                let left = self.bf7.exec(t.0);
                let right = self.bf7.exec(t.1);

                for i in 0..7 {
                    left[i].write(chunk.slice_from_mut(i * 4..));
                }
                for i in 0..7 {
                    right[i].write(chunk.slice_from_mut(i * 4 + 2..));
                }
            }
        }
    };
}

gen_bf28f!(
    NeonButterfly28f,
    "neon",
    ColumnButterfly4f,
    ColumnButterfly7f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf28f!(
    NeonFcmaButterfly28f,
    "fcma",
    ColumnFcmaButterfly4f,
    ColumnFcmaButterfly7f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly28, f32, NeonButterfly28f, 28, 1e-5);
    test_butterfly!(test_neon_butterfly28_f64, f64, NeonButterfly28d, 28, 1e-7);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly28_f64,
        f64,
        NeonFcmaButterfly28d,
        28,
        1e-7
    );
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly28, f32, NeonFcmaButterfly28f, 28, 1e-3);
}
