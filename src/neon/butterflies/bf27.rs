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

macro_rules! gen_bf27d {
    ($name: ident, $features: literal, $internal_bf9: ident, $internal_bf3: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf3, $internal_bf9};
        pub(crate) struct $name {
            direction: FftDirection,
            bf3: $internal_bf3,
            bf9: $internal_bf9,
            twiddles: [NeonStoreD; 18],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(9, 3, fft_direction, 27),
                    bf9: $internal_bf9::new(fft_direction),
                    bf3: $internal_bf3::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f64, 27);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
                let mut rows: [NeonStoreD; 3] = [NeonStoreD::default(); 3];
                let mut rows9: [NeonStoreD; 9] = [NeonStoreD::default(); 9];
                let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 27];
                // columns
                unsafe {
                    for k in 0..9 {
                        for i in 0..3 {
                            rows[i] = NeonStoreD::from_complex_ref(chunk.slice_from(i * 9 + k..));
                        }

                        rows = self.bf3.exec(rows);

                        for i in 1..3 {
                            rows[i] = NeonStoreD::$mul(rows[i], self.twiddles[i - 1 + 2 * k]);
                        }

                        for i in 0..3 {
                            rows[i].write_uninit(scratch.get_unchecked_mut(k * 3 + i..));
                        }
                    }

                    // rows

                    for k in 0..3 {
                        for i in 0..9 {
                            rows9[i] =
                                NeonStoreD::from_complex_refu(scratch.get_unchecked(i * 3 + k..));
                        }
                        rows9 = self.bf9.exec(rows9);
                        for i in 0..9 {
                            rows9[i].write(chunk.slice_from_mut(i * 3 + k..));
                        }
                    }
                }
            }
        }
    };
}

gen_bf27d!(
    NeonButterfly27d,
    "neon",
    ColumnButterfly9d,
    ColumnButterfly3d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf27d!(
    NeonFcmaButterfly27d,
    "fcma",
    ColumnFcmaButterfly9d,
    ColumnFcmaButterfly3d,
    fcmul_fcma
);

#[inline(always)]
fn transpose_9x3(
    rows0: [NeonStoreF; 3],
    rows1: [NeonStoreF; 3],
    rows2: [NeonStoreF; 3],
    rows3: [NeonStoreF; 3],
    rows4: [NeonStoreF; 3],
) -> ([NeonStoreF; 9], [NeonStoreF; 9]) {
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
    let c0 = transpose_f32x2_4x4(
        float32x4x2_t(rows4[0].v, unsafe { vdupq_n_f32(0.) }),
        float32x4x2_t(rows4[1].v, unsafe { vdupq_n_f32(0.) }),
        float32x4x2_t(rows4[2].v, unsafe { vdupq_n_f32(0.) }),
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
            NeonStoreF::raw(b0.3.0),
            NeonStoreF::raw(c0.0.0),
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
            NeonStoreF::raw(b0.3.1),
            NeonStoreF::raw(c0.0.1),
        ],
    )
}

macro_rules! gen_bf27f {
    ($name: ident, $features: literal, $internal_bf3: ident, $internal_bf9: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf3, $internal_bf9};
        pub(crate) struct $name {
            direction: FftDirection,
            bf3: $internal_bf3,
            bf9: $internal_bf9,
            twiddles: [NeonStoreF; 10],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(9, 3, fft_direction, 27),
                    bf9: $internal_bf9::new(fft_direction),
                    bf3: $internal_bf3::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 27);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows0: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                let mut rows1: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                let mut rows2: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                let mut rows3: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                let mut rows4: [NeonStoreF; 3] = [NeonStoreF::default(); 3];
                // columns
                for i in 0..3 {
                    rows0[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 9..));
                    rows1[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 9 + 2..));
                    rows2[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 9 + 4..));
                    rows3[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 9 + 6..));
                    rows4[i] = NeonStoreF::from_complex(chunk.index(i * 9 + 8));
                }

                rows0 = self.bf3.exec(rows0);
                rows1 = self.bf3.exec(rows1);
                rows2 = self.bf3.exec(rows2);
                rows3 = self.bf3.exec(rows3);
                rows4 = self.bf3.exec(rows4);

                for i in 1..3 {
                    rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 2]);
                    rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 4]);
                    rows3[i] = NeonStoreF::$mul(rows3[i], self.twiddles[i - 1 + 6]);
                    rows4[i] = NeonStoreF::$mul(rows4[i], self.twiddles[i - 1 + 8]);
                }

                let t = transpose_9x3(rows0, rows1, rows2, rows3, rows4);

                // rows
                let left = self.bf9.exec(t.0);
                let right = self.bf9.exec(t.1);

                for i in 0..9 {
                    left[i].write(chunk.slice_from_mut(i * 3..));
                }
                for i in 0..9 {
                    right[i].write_lo(chunk.slice_from_mut(i * 3 + 2..));
                }
            }
        }
    };
}

gen_bf27f!(
    NeonButterfly27f,
    "neon",
    ColumnButterfly3f,
    ColumnButterfly9f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf27f!(
    NeonFcmaButterfly27f,
    "fcma",
    ColumnFcmaButterfly3f,
    ColumnFcmaButterfly9f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly27, f32, NeonButterfly27f, 27, 1e-5);
    test_butterfly!(test_neon_butterfly27_f64, f64, NeonButterfly27d, 27, 1e-7);
    test_oof_butterfly!(test_oof_butterfly27, f32, NeonButterfly27f, 27, 1e-5);
    test_oof_butterfly!(test_oof_butterfly27_f64, f64, NeonButterfly27d, 27, 1e-9);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly27_f64,
        f64,
        NeonFcmaButterfly27d,
        27,
        1e-7
    );

    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly27_f64,
        f64,
        NeonFcmaButterfly27d,
        27,
        1e-9
    );

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly27, f32, NeonFcmaButterfly27f, 27, 1e-3);

    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly27,
        f32,
        NeonFcmaButterfly27f,
        27,
        1e-3
    );
}
