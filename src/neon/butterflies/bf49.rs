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
use crate::neon::transpose::neon_transpose_f32x2_7x7_aos;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

macro_rules! gen_bf49f {
    ($name: ident, $features: literal, $internal_bf: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf7: $internal_bf,
            twiddles: [NeonStoreF; 24],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(7, 7, fft_direction, 49),
                    bf7: $internal_bf::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 49);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows1: [NeonStoreF; 7] = [NeonStoreF::default(); 7];
                let mut rows2: [NeonStoreF; 7] = [NeonStoreF::default(); 7];
                let mut rows3: [NeonStoreF; 7] = [NeonStoreF::default(); 7];
                let mut rows4: [NeonStoreF; 7] = [NeonStoreF::default(); 7];

                for i in 0..7 {
                    rows1[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 7..));
                }
                for i in 0..7 {
                    rows2[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 7 + 2..));
                }

                rows1 = self.bf7.exec(rows1);
                rows2 = self.bf7.exec(rows2);

                for i in 1..7 {
                    rows1[i] = NeonStoreF::$mul(rows1[i], self.twiddles[i - 1]);
                    rows2[i] = NeonStoreF::$mul(rows2[i], self.twiddles[i - 1 + 6]);
                }

                for i in 0..7 {
                    rows3[i] = NeonStoreF::from_complex_ref(chunk.slice_from(i * 7 + 4..));
                }
                for i in 0..7 {
                    rows4[i] = NeonStoreF::from_complex(chunk.index(i * 7 + 6));
                }

                rows3 = self.bf7.exec(rows3);
                rows4 = self.bf7.exec(rows4);

                for i in 1..7 {
                    rows3[i] = NeonStoreF::$mul(rows3[i], self.twiddles[i - 1 + 12]);
                    rows4[i] = NeonStoreF::$mul(rows4[i], self.twiddles[i - 1 + 18]);
                }

                let (mut transposed0, mut transposed1, mut transposed2, mut transposed3) =
                    neon_transpose_f32x2_7x7_aos(rows1, rows2, rows3, rows4);

                transposed0 = self.bf7.exec(transposed0);
                transposed1 = self.bf7.exec(transposed1);

                transposed0[0].write(chunk.slice_from_mut(0..));
                transposed0[1].write(chunk.slice_from_mut(7..));
                transposed0[2].write(chunk.slice_from_mut(14..));
                transposed0[3].write(chunk.slice_from_mut(21..));
                transposed0[4].write(chunk.slice_from_mut(28..));
                transposed0[5].write(chunk.slice_from_mut(35..));
                transposed0[6].write(chunk.slice_from_mut(42..));

                transposed1[0].write(chunk.slice_from_mut(2..));
                transposed1[1].write(chunk.slice_from_mut(9..));
                transposed1[2].write(chunk.slice_from_mut(16..));
                transposed1[3].write(chunk.slice_from_mut(23..));
                transposed1[4].write(chunk.slice_from_mut(30..));
                transposed1[5].write(chunk.slice_from_mut(37..));
                transposed1[6].write(chunk.slice_from_mut(44..));

                transposed2 = self.bf7.exec(transposed2);
                transposed3 = self.bf7.exec(transposed3);

                transposed2[0].write(chunk.slice_from_mut(4..));
                transposed2[1].write(chunk.slice_from_mut(11..));
                transposed2[2].write(chunk.slice_from_mut(18..));
                transposed2[3].write(chunk.slice_from_mut(25..));
                transposed2[4].write(chunk.slice_from_mut(32..));
                transposed2[5].write(chunk.slice_from_mut(39..));
                transposed2[6].write(chunk.slice_from_mut(46..));

                transposed3[0].write_lo(chunk.slice_from_mut(6..));
                transposed3[1].write_lo(chunk.slice_from_mut(13..));
                transposed3[2].write_lo(chunk.slice_from_mut(20..));
                transposed3[3].write_lo(chunk.slice_from_mut(27..));
                transposed3[4].write_lo(chunk.slice_from_mut(34..));
                transposed3[5].write_lo(chunk.slice_from_mut(41..));
                transposed3[6].write_lo(chunk.slice_from_mut(48..));
            }
        }
    };
}

gen_bf49f!(NeonButterfly49f, "neon", ColumnButterfly7f, mul_by_complex);
#[cfg(feature = "fcma")]
gen_bf49f!(
    NeonFcmaButterfly49f,
    "fcma",
    ColumnFcmaButterfly7f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly49, f32, NeonButterfly49f, 49, 1e-3);
    test_oof_butterfly!(test_oof_neon_butterfly49, f32, NeonButterfly49f, 49, 1e-3);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly49, f32, NeonFcmaButterfly49f, 49, 1e-3);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly49,
        f32,
        NeonFcmaButterfly49f,
        49,
        1e-3
    );
}
