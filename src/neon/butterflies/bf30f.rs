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
use crate::neon::transpose::transpose_6x5;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

macro_rules! gen_bf30f {
    ($name: ident, $features: literal, $internal_bf5: ident, $internal_bf6: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf5, $internal_bf6};
        pub(crate) struct $name {
            direction: FftDirection,
            bf5: $internal_bf5,
            bf6: $internal_bf6,
            twiddles: [NeonStoreF; 12],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(6, 5, fft_direction, 30),
                    bf6: $internal_bf6::new(fft_direction),
                    bf5: $internal_bf5::new(fft_direction),
                }
            }
        }

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows0: [NeonStoreF; 5] = [NeonStoreF::default(); 5];
                let mut rows1: [NeonStoreF; 5] = [NeonStoreF::default(); 5];
                let mut rows2: [NeonStoreF; 5] = [NeonStoreF::default(); 5];
                // columns
                for i in 0..5 {
                    rows0[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 6..));
                    rows1[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 6 + 2..));
                    rows2[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 6 + 4..));
                }

                rows0 = self.bf5.exec(rows0);
                rows1 = self.bf5.exec(rows1);
                rows2 = self.bf5.exec(rows2);

                for i in 1..5 {
                    rows0[i] = NeonStoreF::$mul(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1 + 4]);
                    rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 8]);
                }

                let t = transpose_6x5(rows0, rows1, rows2);

                let mut r0 = [t[0], t[1], t[2], t[3], t[4], t[5]];
                let mut r1 = [t[6], t[7], t[8], t[9], t[10], t[11]];
                let mut r2 = [t[12], t[13], t[14], t[15], t[16], t[17]];

                // rows

                r0 = self.bf6.exec(r0);
                r1 = self.bf6.exec(r1);
                r2 = self.bf6.exec(r2);

                for i in 0..6 {
                    r0[i].write(chunk.slice_from_mut(i * 5..));
                }
                for i in 0..6 {
                    r1[i].write(chunk.slice_from_mut(i * 5 + 2..));
                }
                for i in 0..6 {
                    r2[i].write_lo(chunk.slice_from_mut(i * 5 + 4..));
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 30);
    };
}

gen_bf30f!(
    NeonButterfly30f,
    "neon",
    ColumnButterfly5f,
    ColumnButterfly6f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf30f!(
    NeonFcmaButterfly30f,
    "fcma",
    ColumnFcmaButterfly5f,
    ColumnFcmaButterfly6f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly30, f32, NeonButterfly30f, 30, 1e-3);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly30, f32, NeonFcmaButterfly30f, 30, 1e-3);
}
