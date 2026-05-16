/*
 * // Copyright (c) Radzivon Bartoshyk 05/2026. All rights reserved.
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
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf1536f {
    ($name: ident, $features: literal, $internal_bf32: ident, $internal_bf6: ident, $mul: ident) => {
use crate::neon::mixed::$internal_bf32;
pub(crate) struct $name {
    direction: FftDirection,
    bf32: $internal_bf32,
    bf6: $internal_bf6,
    twiddles: [NeonStoreF; 744],
    twiddles48: [NeonStoreF; 5],
}

impl $name {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(48, 32, fft_direction, 1536),
            twiddles48: [
                NeonStoreF::from_complex(&compute_twiddle(1, 48, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(2, 48, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(3, 48, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(4, 48, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(5, 48, fft_direction)),
            ],
            bf32: $internal_bf32::new(fft_direction),
            bf6: $internal_bf6::new(fft_direction),
        }
    }
}

impl $name {
    #[target_feature(enable = $features)]
    fn exec_bf48(&self, src: &[MaybeUninit<Complex<f32>>; 1536], dst: &mut [Complex<f32>]) {
        unsafe {
            for k in 0..16 {
                macro_rules! load {
                    ($src: expr, $k: expr, $idx: expr) => {{ NeonStoreF::from_complex_refu($src.get_unchecked($k * 2 + $idx * 32..)) }};
                }

                macro_rules! store {
                    ($v: expr, $idx: expr, $dst: expr, $k: expr) => {{ $v.write($dst.get_unchecked_mut($k * 2 + $idx * 32..)) }};
                }

                let input1 = std::array::from_fn(|x| load!(src, k, x * 8 + 1));
                let mut mid1 = self.bf6.exec(input1);

                mid1[1] = NeonStoreF::$mul(mid1[1], self.twiddles48[0]);          // W_48^ 1 = T[0]
                mid1[2] = NeonStoreF::$mul(mid1[2], self.twiddles48[1]);          // W_48^ 2 = T[1]
                mid1[3] = NeonStoreF::$mul(mid1[3], self.twiddles48[2]);          // W_48^ 3 = T[2]
                mid1[4] = NeonStoreF::$mul(mid1[4], self.twiddles48[3]);          // W_48^ 4 = T[3]
                mid1[5] = NeonStoreF::$mul(mid1[5], self.twiddles48[4]);          // W_48^ 5 = T[4]

                let input2 = std::array::from_fn(|x| load!(src, k, x * 8 + 2));
                let mut mid2 = self.bf6.exec(input2);

                mid2[1] = NeonStoreF::$mul(mid2[1], self.twiddles48[1]);          // W_48^ 2 = T[1]
                mid2[2] = NeonStoreF::$mul(mid2[2], self.twiddles48[3]);          // W_48^ 4 = T[3]
                mid2[3] = self.bf32.bf16.bf8.rotate45(mid2[3]);                              // W_48^ 6 = rot1
                mid2[4] = NeonStoreF::$mul(mid2[4], self.bf32.bf16.bf8.rotate45(self.twiddles48[1]));
                mid2[5] = NeonStoreF::$mul(mid2[5], self.bf32.bf16.bf8.rotate45(self.twiddles48[3]));

                let input3 = std::array::from_fn(|x| load!(src, k, x * 8 + 3));
                let mut mid3 = self.bf6.exec(input3);

                let tw_mid3_5 = self.bf32.bf16.bf8.rotate(self.twiddles48[2]);

                mid3[1] = NeonStoreF::$mul(mid3[1], self.twiddles48[2]);          // W_48^ 3 = T[2]
                mid3[2] = self.bf32.bf16.bf8.rotate45(mid3[2]);                              // W_48^ 6 = rot1
                mid3[3] = NeonStoreF::$mul(mid3[3], self.bf32.bf16.bf8.rotate45(self.twiddles48[2]));
                mid3[4] = self.bf32.bf16.bf8.rotate(mid3[4]);                               // W_48^12 = rot
                mid3[5] = NeonStoreF::$mul(mid3[5], tw_mid3_5);

                let input4 = std::array::from_fn(|x| load!(src, k, x * 8 + 4));
                let mut mid4 = self.bf6.exec(input4);

                let tw_mid_4_5 = self.bf32.bf16.bf8.rotate135(self.twiddles48[1]);

                mid4[1] = NeonStoreF::$mul(mid4[1], self.twiddles48[3]);          // W_48^ 4 = T[3]
                mid4[2] = NeonStoreF::$mul(mid4[2], self.bf32.bf16.bf8.rotate45(self.twiddles48[1]));
                mid4[3] = self.bf32.bf16.bf8.rotate(mid4[3]);                               // W_48^12 = rot
                mid4[4] = NeonStoreF::$mul(mid4[4], self.bf32.bf16.bf8.rotate(self.twiddles48[3]));
                mid4[5] = NeonStoreF::$mul(mid4[5], tw_mid_4_5);

                let input5 = std::array::from_fn(|x| load!(src, k, x * 8 + 5));
                let mut mid5 = self.bf6.exec(input5);

                mid5[1] = NeonStoreF::$mul(mid5[1], self.twiddles48[4]);          // W_48^ 5 = T[4]
                mid5[2] = NeonStoreF::$mul(mid5[2], self.bf32.bf16.bf8.rotate45(self.twiddles48[3]));
                mid5[3] = NeonStoreF::$mul(mid5[3], tw_mid3_5);
                mid5[4] = NeonStoreF::$mul(mid5[4], tw_mid_4_5);
                mid5[5] = NeonStoreF::$mul(mid5[5], self.twiddles48[0].neg());

                let input6 = std::array::from_fn(|x| load!(src, k, x * 8 + 6));
                let mut mid6 = self.bf6.exec(input6);

                mid6[1] = self.bf32.bf16.bf8.rotate45(mid6[1]);
                mid6[2] = self.bf32.bf16.bf8.rotate(mid6[2]);
                mid6[3] = self.bf32.bf16.bf8.rotate135(mid6[3]);
                mid6[4] = mid6[4].neg();
                mid6[5] = self.bf32.bf16.bf8.rotate45(mid6[5]).neg();

                let input7 = std::array::from_fn(|x| load!(src, k, x * 8 + 7));
                let mut mid7 = self.bf6.exec(input7);

                mid7[1] = NeonStoreF::$mul(mid7[1], self.bf32.bf16.bf8.rotate45(self.twiddles48[0]));
                mid7[2] = NeonStoreF::$mul(mid7[2], self.bf32.bf16.bf8.rotate(self.twiddles48[1]));
                mid7[3] = NeonStoreF::$mul(mid7[3], self.bf32.bf16.bf8.rotate135(self.twiddles48[2]));
                mid7[4] = NeonStoreF::$mul(mid7[4], self.twiddles48[3].neg());
                mid7[5] = NeonStoreF::$mul(mid7[5], self.bf32.bf16.bf8.rotate45(self.twiddles48[4])).neg();

                let input0 = std::array::from_fn(|x| load!(src, k, x * 8));
                let mid0 = self.bf6.exec(input0);

                for i in 0..6 {
                    let output = self.bf32.bf16.bf8.exec([
                        mid0[i], mid1[i], mid2[i], mid3[i], mid4[i], mid5[i], mid6[i], mid7[i],
                    ]);
                    store!(output[0], i, dst, k);
                    store!(output[1], i + 6, dst, k);
                    store!(output[2], i + 12, dst, k);
                    store!(output[3], i + 18, dst, k);
                    store!(output[4], i + 24, dst, k);
                    store!(output[5], i + 30, dst, k);
                    store!(output[6], i + 36, dst, k);
                    store!(output[7], i + 42, dst, k);
                }
            }
        }
    }
}

impl $name {

    #[inline]
    #[target_feature(enable = $features)]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 1536];
        // columns
        for k in 0..24 {
            let tw = k * 31;
            self.bf32.exec_transpose_streaming(
                |idx| NeonStoreF::from_complex_ref(chunk.slice_from(k * 2 + idx * 48..)),
                |idx| self.twiddles[tw + idx],
                |idx, val| unsafe {
                    let row = k * 2 + idx % 2;
                    let col = (idx / 2) * 2;
                    val.write_uninit(scratch.get_unchecked_mut(row * 32 + col..))
                },
            );
        }
        // rows
        self.exec_bf48(&scratch, chunk.slice_from_mut(0..));
    }
}

boring_neon_butterfly!($name, $features, f32, 1536);
    };
}

use crate::neon::mixed::ColumnButterfly6f;
#[cfg(feature = "fcma")]
use crate::neon::mixed::ColumnFcmaButterfly6f;

gen_bf1536f!(
    NeonButterfly1536f,
    "neon",
    ColumnButterfly32f,
    ColumnButterfly6f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf1536f!(
    NeonFcmaForwardButterfly1536f,
    "fcma",
    ColumnFcmaForwardButterfly32f,
    ColumnFcmaButterfly6f,
    fcmul_fcma
);
#[cfg(feature = "fcma")]
gen_bf1536f!(
    NeonFcmaInverseButterfly1536f,
    "fcma",
    ColumnFcmaInverseButterfly32f,
    ColumnFcmaButterfly6f,
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
        test_butterfly!(test_neon_butterfly1536, f32, NeonButterfly1536f, 1536, 1e-2);
        test_neon_butterfly1536();
    }

    #[test]
    #[allow(unnameable_test_items)]
    fn capsule2() {
        if std::env::var("SHORT_TEST").as_deref() == Ok("yes") {
            return;
        }
        test_oof_butterfly!(
            test_oof_neon_butterfly1536,
            f32,
            NeonButterfly1536f,
            1536,
            1e-2
        );
        test_oof_neon_butterfly1536();
    }
}
