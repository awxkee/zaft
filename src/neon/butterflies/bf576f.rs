/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

macro_rules! gen_bf576f {
    ($name: ident, $features: literal, $internal_bf16: ident, $internal_bf36: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf16, $internal_bf36};
        pub(crate) struct $name {
            direction: FftDirection,
            bf16: $internal_bf16,
            bf36: $internal_bf36,
            twiddles: [NeonStoreF; 270],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(36, 16, fft_direction, 576),
                    bf36: $internal_bf36::new(fft_direction),
                    bf16: $internal_bf16::new(fft_direction),
                }
            }
        }

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows: [NeonStoreF; 16] = [NeonStoreF::default(); 16];
                let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 576];

                unsafe {
                    // columns
                    for k in 0..18 {
                        for i in 0..16 {
                            rows[i] =
                                NeonStoreF::from_complex_ref(chunk.slice_from(i * 36 + k * 2..));
                        }

                        rows = self.bf16.exec(rows);

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
                    self.bf36.exec_streaming(
                        |i| unsafe {
                            NeonStoreF::from_complex_refu(scratch.get_unchecked(i * 16 + k * 2..))
                        },
                        |i, store| store.write(chunk.slice_from_mut(i * 16 + k * 2..)),
                    );
                }
            }
        }
        boring_neon_butterfly!($name, $features, f32, 576);
    };
}

gen_bf576f!(
    NeonButterfly576f,
    "neon",
    ColumnButterfly16f,
    ColumnButterfly36f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf576f!(
    NeonFcmaButterfly576f,
    "fcma",
    ColumnFcmaButterfly16f,
    ColumnFcmaButterfly36f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    #[test]
    #[allow(unnameable_test_items)]
    fn capsule1() {
        if std::env::var("SHORT_TEST").as_deref() == Ok("yes") {
            return;
        }
        test_butterfly!(test_neon_butterfly576, f32, NeonButterfly576f, 576, 1e-3);
        test_neon_butterfly576();
    }

    #[test]
    #[allow(unnameable_test_items)]
    fn capsule2() {
        if std::env::var("SHORT_TEST").as_deref() == Ok("yes") {
            return;
        }
        test_oof_butterfly!(
            test_oof_neon_butterfly576,
            f32,
            NeonButterfly576f,
            576,
            1e-3
        );
        test_oof_neon_butterfly576();
    }
}
