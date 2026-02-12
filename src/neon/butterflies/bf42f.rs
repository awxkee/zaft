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
use crate::neon::transpose::neon_transpose_f32x2_7x6_aos;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

macro_rules! gen_bf42f {
    ($name: ident, $features: literal, $internal_bf6: ident, $internal_bf7: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf6, $internal_bf7};
        pub(crate) struct $name {
            direction: FftDirection,
            bf6: $internal_bf6,
            bf7: $internal_bf7,
            twiddles: [NeonStoreF; 20],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(7, 6, fft_direction, 42),
                    bf7: $internal_bf7::new(fft_direction),
                    bf6: $internal_bf6::new(fft_direction),
                }
            }
        }

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows0: [NeonStoreF; 6] = [NeonStoreF::default(); 6];
                let mut rows1: [NeonStoreF; 6] = [NeonStoreF::default(); 6];
                let mut rows2: [NeonStoreF; 6] = [NeonStoreF::default(); 6];
                let mut rows3: [NeonStoreF; 6] = [NeonStoreF::default(); 6];
                // columns
                for i in 0..6 {
                    rows0[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 7..));
                    rows1[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 7 + 2..));
                    rows2[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 7 + 4..));
                    rows3[i] = NeonStoreF::from_complex(chunk.index(i * 7 + 6));
                }

                rows0 = self.bf6.exec(rows0);
                rows1 = self.bf6.exec(rows1);
                rows2 = self.bf6.exec(rows2);
                rows3 = self.bf6.exec(rows3);

                for i in 1..6 {
                    rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 5]);
                    rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 10]);
                    rows3[i] = NeonStoreF::$mul(rows3[i], self.twiddles[i - 1 + 15]);
                }

                let (mut v0, mut v1, mut v2) =
                    neon_transpose_f32x2_7x6_aos(rows0, rows1, rows2, rows3);

                v0 = self.bf7.exec(v0);
                for i in 0..7 {
                    v0[i].write(chunk.slice_from_mut(i * 6..));
                }
                v1 = self.bf7.exec(v1);
                for i in 0..7 {
                    v1[i].write(chunk.slice_from_mut(i * 6 + 2..));
                }
                v2 = self.bf7.exec(v2);
                for i in 0..7 {
                    v2[i].write(chunk.slice_from_mut(i * 6 + 4..));
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 42);
    };
}

gen_bf42f!(
    NeonButterfly42f,
    "neon",
    ColumnButterfly6f,
    ColumnButterfly7f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf42f!(
    NeonFcmaButterfly42f,
    "fcma",
    ColumnFcmaButterfly6f,
    ColumnFcmaButterfly7f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly42, f32, NeonButterfly42f, 42, 1e-3);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly42, f32, NeonFcmaButterfly42f, 42, 1e-3);
}
