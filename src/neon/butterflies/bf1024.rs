/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf1024f {
    ($name: ident, $features: literal, $internal_bf32: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf32;
        pub(crate) struct $name {
            direction: FftDirection,
            bf32: $internal_bf32,
            twiddles: [NeonStoreF; 497],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(32, 32, fft_direction, 1024),
                    bf32: $internal_bf32::new(fft_direction),
                }
            }
        }

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 1024];
                // columns
                for k in 0..16 {
                    let tw = k * 31;
                    self.bf32.exec_transpose_streaming(
                        |idx| NeonStoreF::from_complex_ref(chunk.slice_from(k * 2 + idx * 32..)),
                        |idx| self.twiddles[tw + idx],
                        |idx, val| unsafe {
                            let row = k * 2 + idx % 2;
                            let col = (idx / 2) * 2;
                            val.write_uninit(scratch.get_unchecked_mut(row * 32 + col..))
                        },
                    );
                }
                // rows
                for k in 0..16 {
                    self.bf32.exec_streaming(
                        |i| unsafe {
                            NeonStoreF::from_complex_refu(scratch.get_unchecked(i * 32 + k * 2..))
                        },
                        |i, store| store.write(chunk.slice_from_mut(i * 32 + k * 2..)),
                    );
                }
            }
        }

        boring_neon_butterfly!($name, $features, f32, 1024);
    };
}

gen_bf1024f!(
    NeonButterfly1024f,
    "neon",
    ColumnButterfly32f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf1024f!(
    NeonFcmaForwardButterfly1024f,
    "fcma",
    ColumnFcmaForwardButterfly32f,
    fcmul_fcma
);
#[cfg(feature = "fcma")]
gen_bf1024f!(
    NeonFcmaInverseButterfly1024f,
    "fcma",
    ColumnFcmaInverseButterfly32f,
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
        test_butterfly!(test_neon_butterfly1024, f32, NeonButterfly1024f, 1024, 1e-3);
        test_neon_butterfly1024();
    }

    #[test]
    #[allow(unnameable_test_items)]
    fn capsule2() {
        if std::env::var("SHORT_TEST").as_deref() == Ok("yes") {
            return;
        }
        test_oof_butterfly!(
            test_oof_neon_butterfly1024,
            f32,
            NeonButterfly1024f,
            1024,
            1e-3
        );
        test_oof_neon_butterfly1024();
    }
}
