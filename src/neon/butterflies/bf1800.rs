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
use crate::FftDirection;
use crate::FftExecutor;
use crate::ZaftError;
use crate::neon::butterflies::shared::boring_neon_butterfly;
use crate::neon::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::neon::mixed::NeonStoreF;
use crate::store::BidirectionalStore;
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf1800f {
    ($name: ident, $features: literal, $internal_bf40: ident, $internal_bf45: ident, $mul: ident) => {
        use crate::neon::mixed::{$internal_bf40, $internal_bf45};
        pub(crate) struct $name {
            direction: FftDirection,
            bf40: $internal_bf40,
            bf45: $internal_bf45,
            twiddles: [NeonStoreF; 897],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(45, 40, fft_direction, 1800),
                    bf40: $internal_bf40::new(fft_direction),
                    bf45: $internal_bf45::new(fft_direction),
                }
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn exec_bf40(
                &self,
                src: &[Complex<f32>],
                scratch: &mut [MaybeUninit<Complex<f32>>; 1800],
            ) {
                for k in 0..22 {
                    let tw = k * 39;
                    self.bf40.exec_transpose_streaming(
                        |idx| unsafe {
                            NeonStoreF::from_complex_ref(src.get_unchecked(k * 2 + 45 * idx..))
                        },
                        |idx| self.twiddles[tw + idx],
                        |idx, val| unsafe {
                            let row = k * 2 + idx % 2;
                            let col = (idx / 2) * 2;
                            val.write_uninit(scratch.get_unchecked_mut(row * 40 + col..))
                        },
                    );
                }
                {
                    let k = 22;
                    let tw = k * 39;
                    self.bf40.exec_transpose_streaming(
                        |idx| unsafe {
                            NeonStoreF::from_complex(src.get_unchecked(k * 2 + 45 * idx))
                        },
                        |idx| self.twiddles[tw + idx],
                        |idx, val| unsafe {
                            let q = idx % 2;
                            if q == 0 {
                                let row = k * 2 + idx % 2;
                                let col = (idx / 2) * 2;
                                val.write_uninit(scratch.get_unchecked_mut(row * 40 + col..))
                            }
                        },
                    );
                }
            }

            #[target_feature(enable = $features)]
            fn exec_bf45(&self, src: &[MaybeUninit<Complex<f32>>; 1800], dst: &mut [Complex<f32>]) {
                for k in 0..20 {
                    self.bf45.exec_streaming(
                        |i| unsafe {
                            NeonStoreF::from_complex_refu(src.get_unchecked(i * 40 + k * 2..))
                        },
                        |i, store| unsafe { store.write(dst.get_unchecked_mut(i * 40 + k * 2..)) },
                    );
                }
            }

            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 1800];
                // columns
                self.exec_bf40(chunk.slice_from(0..), &mut scratch);
                // rows
                self.exec_bf45(&scratch, chunk.slice_from_mut(0..));
            }
        }

        boring_neon_butterfly!($name, $features, f32, 1800);
    };
}

gen_bf1800f!(
    NeonButterfly1800f,
    "neon",
    ColumnButterfly40f,
    ColumnButterfly45f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf1800f!(
    NeonFcmaButterfly1800f,
    "fcma",
    ColumnFcmaButterfly40f,
    ColumnFcmaButterfly45f,
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
        test_butterfly!(test_neon_butterfly1800, f32, NeonButterfly1800f, 1800, 1e-2);
        test_neon_butterfly1800();
    }

    #[test]
    #[allow(unnameable_test_items)]
    fn capsule2() {
        if std::env::var("SHORT_TEST").as_deref() == Ok("yes") {
            return;
        }
        test_oof_butterfly!(
            test_oof_neon_butterfly1800,
            f32,
            NeonButterfly1800f,
            1800,
            1e-2
        );
        test_oof_neon_butterfly1800();
    }
}
