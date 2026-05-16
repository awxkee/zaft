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
use crate::neon::butterflies::shared::{boring_neon_butterfly, gen_butterfly_twiddles_f32};
use crate::neon::mixed::NeonStoreF;
use crate::neon::transpose::transpose_2x2;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf512f {
    ($name: ident, $features: literal, $internal_bf32: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf32;
        pub(crate) struct $name {
            direction: FftDirection,
            bf32: $internal_bf32,
            twiddles: [NeonStoreF; 240],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(32, 16, fft_direction, 512),
                    bf32: $internal_bf32::new(fft_direction),
                }
            }
        }

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows: [NeonStoreF; 16] = [NeonStoreF::default(); 16];
                let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 512];

                unsafe {
                    // columns
                    for k in 0..16 {
                        for i in 0..16 {
                            rows[i] =
                                NeonStoreF::from_complex_ref(chunk.slice_from(i * 32 + k * 2..));
                        }

                        rows = self.bf32.bf16.exec(rows);

                        let q1 = NeonStoreF::$mul(rows[1], self.twiddles[15 * k]);
                        let t = transpose_2x2([rows[0], q1]);
                        t[0].write_uninit(scratch.get_unchecked_mut(k * 2 * 16..));
                        t[1].write_uninit(scratch.get_unchecked_mut((k * 2 + 1) * 16..));

                        for i in 1..8 {
                            let q0 = NeonStoreF::$mul(
                                rows[i * 2],
                                self.twiddles[(i - 1) * 2 + 1 + 15 * k],
                            );
                            let q1 = NeonStoreF::$mul(
                                rows[i * 2 + 1],
                                self.twiddles[(i - 1) * 2 + 2 + 15 * k],
                            );
                            let t = transpose_2x2([q0, q1]);
                            t[0].write_uninit(scratch.get_unchecked_mut(k * 2 * 16 + i * 2..));
                            t[1].write_uninit(
                                scratch.get_unchecked_mut((k * 2 + 1) * 16 + i * 2..),
                            );
                        }
                    }
                }

                // rows

                for k in 0..8 {
                    self.bf32.exec_streaming(
                        |i| unsafe {
                            NeonStoreF::from_complex_refu(scratch.get_unchecked(i * 16 + k * 2..))
                        },
                        |i, store| store.write(chunk.slice_from_mut(i * 16 + k * 2..)),
                    );
                }
            }
        }
        boring_neon_butterfly!($name, $features, f32, 512);
    };
}

gen_bf512f!(
    NeonButterfly512f,
    "neon",
    ColumnButterfly32f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf512f!(
    NeonFcmaButterfly512f,
    "fcma",
    ColumnFcmaButterfly32f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly512, f32, NeonButterfly512f, 512, 1e-3);
    test_oof_butterfly!(
        test_oof_neon_butterfly512,
        f32,
        NeonButterfly512f,
        512,
        1e-3
    );

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly512,
        f32,
        NeonFcmaButterfly512f,
        512,
        1e-3
    );

    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly512,
        f32,
        NeonFcmaButterfly512f,
        512,
        1e-3
    );
}
