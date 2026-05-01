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
use crate::neon::transpose::transpose_2x2;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! gen_bf2048f {
    ($name: ident, $features: literal, $internal_bf8: ident, $mul: ident) => {
use crate::neon::mixed::$internal_bf8;
pub(crate) struct $name {
    direction: FftDirection,
    bf8: $internal_bf8,
    twiddles: [NeonStoreF; 992],
    twiddles32: [NeonStoreF; 6],
    twiddles64: [NeonStoreF; 7],
}

impl $name {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(64, 32, fft_direction, 2048),
            twiddles32: [
                NeonStoreF::from_complex(&compute_twiddle(1, 32, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(2, 32, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(3, 32, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(5, 32, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(6, 32, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(7, 32, fft_direction)),
            ],
            twiddles64: [
                NeonStoreF::from_complex(&compute_twiddle(1, 64, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(2, 64, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(3, 64, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(4, 64, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(5, 64, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(6, 64, fft_direction)),
                NeonStoreF::from_complex(&compute_twiddle(7, 64, fft_direction)),
            ],
            bf8: $internal_bf8::new(fft_direction),
        }
    }
}

impl $name {
    #[target_feature(enable = $features)]
    fn exec_bf64(&self, src: &mut [MaybeUninit<Complex<f32>>; 2048], dst: &mut [Complex<f32>]) {
        unsafe {
            for k in 0..16 {
                macro_rules! load {
                    ($src: expr, $k: expr, $idx: expr) => {{ NeonStoreF::from_complex_refu($src.get_unchecked($k * 2 + $idx * 32..)) }};
                }

                macro_rules! store {
                    ($v: expr, $idx: expr, $dst: expr, $k: expr) => {{ $v.write($dst.get_unchecked_mut($k * 2 + $idx * 32..)) }};
                }

                let input1: [NeonStoreF; 8] = std::array::from_fn(|x| load!(src, k, x * 8 + 1));
                let mut mid1 = self.bf8.exec(input1);

                for i in 0..7 {
                    mid1[i + 1] = NeonStoreF::$mul(mid1[i + 1], self.twiddles64[i]);
                }

                let input2 = std::array::from_fn(|x| load!(src, k, x * 8 + 2));

                let mut mid2 = self.bf8.exec(input2);

                mid2[1] = NeonStoreF::$mul(mid2[1], self.twiddles64[1]);
                mid2[2] = NeonStoreF::$mul(mid2[2], self.twiddles64[3]);
                mid2[3] = NeonStoreF::$mul(mid2[3], self.twiddles64[5]);
                mid2[4] = self.bf8.rotate1(mid2[4]);
                mid2[5] = NeonStoreF::$mul(mid2[5], self.bf8.rotate1(self.twiddles64[1]));
                mid2[6] = NeonStoreF::$mul(mid2[6], self.bf8.rotate1(self.twiddles64[3]));
                mid2[7] = NeonStoreF::$mul(mid2[7], self.bf8.rotate1(self.twiddles64[5]));

                let input3 = std::array::from_fn(|x| load!(src, k, x * 8 + 3));
                let mut mid3 = self.bf8.exec(input3);

                mid3[1] = NeonStoreF::$mul(mid3[1], self.twiddles64[2]);                    // W₆₄^3  = t3
                mid3[2] = NeonStoreF::$mul(mid3[2], self.twiddles64[5]);                    // W₆₄^6  = t6
                mid3[3] = NeonStoreF::$mul(mid3[3], self.bf8.rotate1(self.twiddles64[0])); // W₆₄^9  = W₈¹·t1
                mid3[4] = NeonStoreF::$mul(mid3[4], self.bf8.rotate1(self.twiddles64[3])); // W₆₄^12 = W₈¹·t4
                mid3[5] = NeonStoreF::$mul(mid3[5], self.bf8.rotate1(self.twiddles64[6])); // W₆₄^15 = W₈¹·t7
                mid3[6] = NeonStoreF::$mul(mid3[6], self.bf8.rotate(self.twiddles64[1])); // W₆₄^18 = W₈²·t2
                mid3[7] = NeonStoreF::$mul(mid3[7], self.bf8.rotate(self.twiddles64[4])); // W₆₄^21 = W₈²·t5

                let input4 = std::array::from_fn(|x| load!(src, k, x * 8 + 4));
                let mut mid4 = self.bf8.exec(input4);

                mid4[1] = NeonStoreF::$mul(mid4[1], self.twiddles64[3]);                    // W₆₄^4  = t4
                mid4[2] = self.bf8.rotate1(mid4[2]);                                        // W₆₄^8  = W₈¹
                mid4[3] = NeonStoreF::$mul(mid4[3], self.bf8.rotate1(self.twiddles64[3])); // W₆₄^12 = W₈¹·t4
                mid4[4] = self.bf8.rotate(mid4[4]);                                        // W₆₄^16 = W₈²
                mid4[5] = NeonStoreF::$mul(mid4[5], self.bf8.rotate(self.twiddles64[3])); // W₆₄^20 = W₈²·t4
                mid4[6] = self.bf8.rotate3(mid4[6]);                                        // W₆₄^24 = W₈³
                mid4[7] = NeonStoreF::$mul(mid4[7], self.bf8.rotate3(self.twiddles64[3])); // W₆₄^28 = W₈³·t4

                let input5 = std::array::from_fn(|x| load!(src, k, x * 8 + 5));
                let mut mid5 = self.bf8.exec(input5);

                mid5[1] = NeonStoreF::$mul(mid5[1], self.twiddles64[4]);                    // W₆₄^5  = t5
                mid5[2] = NeonStoreF::$mul(mid5[2], self.bf8.rotate1(self.twiddles64[1])); // W₆₄^10 = W₈¹·t2
                mid5[3] = NeonStoreF::$mul(mid5[3], self.bf8.rotate1(self.twiddles64[6])); // W₆₄^15 = W₈¹·t7
                mid5[4] = NeonStoreF::$mul(mid5[4], self.bf8.rotate(self.twiddles64[3])); // W₆₄^20 = W₈²·t4
                mid5[5] = NeonStoreF::$mul(mid5[5], self.bf8.rotate3(self.twiddles64[0])); // W₆₄^25 = W₈³·t1
                mid5[6] = NeonStoreF::$mul(mid5[6], self.bf8.rotate3(self.twiddles64[5])); // W₆₄^30 = W₈³·t6
                mid5[7] = NeonStoreF::$mul(mid5[7], self.twiddles64[2].neg());             // W₆₄^35 = −t3

                let input6 = std::array::from_fn(|x| load!(src, k, x * 8 + 6));
                let mut mid6 = self.bf8.exec(input6);

                mid6[1] = NeonStoreF::$mul(mid6[1], self.twiddles64[5]);                    // W₆₄^6  = t6
                mid6[2] = NeonStoreF::$mul(mid6[2], self.bf8.rotate1(self.twiddles64[3])); // W₆₄^12 = W₈¹·t4
                mid6[3] = NeonStoreF::$mul(mid6[3], self.bf8.rotate(self.twiddles64[1])); // W₆₄^18 = W₈²·t2
                mid6[4] = self.bf8.rotate3(mid6[4]);                                        // W₆₄^24 = W₈³
                mid6[5] = NeonStoreF::$mul(mid6[5], self.bf8.rotate3(self.twiddles64[5])); // W₆₄^30 = W₈³·t6
                mid6[6] = NeonStoreF::$mul(mid6[6], self.twiddles64[3].neg());             // W₆₄^36 = −t4
                mid6[7] = NeonStoreF::$mul(mid6[7], self.bf8.rotate5(self.twiddles64[1])); // W₆₄^42 = W₈⁵·t2

                let input7 = std::array::from_fn(|x| load!(src, k, x * 8 + 7));
                let mut mid7 = self.bf8.exec(input7);

                mid7[1] = NeonStoreF::$mul(mid7[1], self.twiddles64[6]);                    // W₆₄^7  = t7
                mid7[2] = NeonStoreF::$mul(mid7[2], self.bf8.rotate1(self.twiddles64[5])); // W₆₄^14 = W₈¹·t6
                mid7[3] = NeonStoreF::$mul(mid7[3], self.bf8.rotate(self.twiddles64[4]));  // W₆₄^21 = W₈²·t5
                mid7[4] = NeonStoreF::$mul(mid7[4], self.bf8.rotate3(self.twiddles64[3])); // W₆₄^28 = W₈³·t4
                mid7[5] = NeonStoreF::$mul(mid7[5], self.twiddles64[2].neg());             // W₆₄^35 = −t3
                mid7[6] = NeonStoreF::$mul(mid7[6], self.bf8.rotate5(self.twiddles64[1])); // W₆₄^42 = W₈⁵·t2
                mid7[7] = NeonStoreF::$mul(mid7[7], self.bf8.rotate6(self.twiddles64[0])); // W₆₄^49 = W₈⁶·t1

                let input0: [NeonStoreF; 8] = std::array::from_fn(|x| load!(src, k, x * 8));
                let mid0 = self.bf8.exec(input0);

                for i in 0..8 {
                    let output = self.bf8.exec([
                        mid0[i], mid1[i], mid2[i], mid3[i], mid4[i], mid5[i], mid6[i], mid7[i],
                    ]);
                    store!(output[0], i, dst, k);
                    store!(output[1], i + 8, dst, k);
                    store!(output[2], i + 16, dst, k);
                    store!(output[3], i + 24, dst, k);
                    store!(output[4], i + 32, dst, k);
                    store!(output[5], i + 40, dst, k);
                    store!(output[6], i + 48, dst, k);
                    store!(output[7], i + 56, dst, k);
                }
            }
        }
    }
}

impl $name {
    #[target_feature(enable = $features)]
    fn exec_bf32_1(&self, src: &[Complex<f32>], dst: &mut [MaybeUninit<Complex<f32>>; 2048]) {
        unsafe {
            for k in 0..32 {
                macro_rules! load {
                    ($src: expr, $k: expr, $idx: expr) => {{ NeonStoreF::from_complex_ref($src.get_unchecked($k * 2 + $idx * 64..)) }};
                }

                macro_rules! store0 {
                    ($v: expr, $idx: expr, $dst: expr, $k: expr) => {{ $v.write_uninit($dst.get_unchecked_mut(($k * 2) * 32 + $idx..)) }};
                }

                macro_rules! store1 {
                    ($v: expr, $idx: expr, $dst: expr, $k: expr) => {{
                        $v.write_uninit($dst.get_unchecked_mut(($k * 2 + 1) * 32 + $idx..));
                    }};
                }

                let input1 = [
                    load!(src, k, 1),
                    load!(src, k, 9),
                    load!(src, k, 17),
                    load!(src, k, 25),
                ];
                let mut mid1 = self.bf8.bf4.exec(input1);

                mid1[1] = NeonStoreF::$mul(mid1[1], self.twiddles32[0]);
                mid1[2] = NeonStoreF::$mul(mid1[2], self.twiddles32[1]);
                mid1[3] = NeonStoreF::$mul(mid1[3], self.twiddles32[2]);

                let input2 = [
                    load!(src, k, 2),
                    load!(src, k, 10),
                    load!(src, k, 18),
                    load!(src, k, 26),
                ];
                let mut mid2 = self.bf8.bf4.exec(input2);

                mid2[1] = NeonStoreF::$mul(mid2[1], self.twiddles32[1]);
                mid2[2] = self.bf8.rotate1(mid2[2]);
                mid2[3] = NeonStoreF::$mul(mid2[3], self.twiddles32[4]);

                let input3 = [
                    load!(src, k, 3),
                    load!(src, k, 11),
                    load!(src, k, 19),
                    load!(src, k, 27),
                ];
                let mut mid3 = self.bf8.bf4.exec(input3);

                mid3[1] = NeonStoreF::$mul(mid3[1], self.twiddles32[2]);
                mid3[2] = NeonStoreF::$mul(mid3[2], self.twiddles32[4]);
                mid3[3] = NeonStoreF::$mul(mid3[3], self.bf8.rotate(self.twiddles32[0]));

                let input4 = [
                    load!(src, k, 4),
                    load!(src, k, 12),
                    load!(src, k, 20),
                    load!(src, k, 28),
                ];
                let mut mid4 = self.bf8.bf4.exec(input4);

                mid4[1] = self.bf8.rotate1(mid4[1]);
                mid4[2] = self.bf8.rotate(mid4[2]);
                mid4[3] = self.bf8.rotate3(mid4[3]);

                let input5 = [
                    load!(src, k, 5),
                    load!(src, k, 13),
                    load!(src, k, 21),
                    load!(src, k, 29),
                ];
                let mut mid5 = self.bf8.bf4.exec(input5);

                mid5[1] = NeonStoreF::$mul(mid5[1], self.twiddles32[3]);
                mid5[2] = NeonStoreF::$mul(mid5[2], self.bf8.rotate(self.twiddles32[1]));
                mid5[3] = NeonStoreF::$mul(mid5[3], self.bf8.rotate(self.twiddles32[5]));

                let input6 = [
                    load!(src, k, 6),
                    load!(src, k, 14),
                    load!(src, k, 22),
                    load!(src, k, 30),
                ];
                let mut mid6 = self.bf8.bf4.exec(input6);

                mid6[1] = NeonStoreF::$mul(mid6[1], self.twiddles32[4]);
                mid6[2] = self.bf8.rotate3(mid6[2]);
                mid6[3] = NeonStoreF::$mul(mid6[3], self.twiddles32[1].neg());

                let input7 = [
                    load!(src, k, 7),
                    load!(src, k, 15),
                    load!(src, k, 23),
                    load!(src, k, 31),
                ];
                let mut mid7 = self.bf8.bf4.exec(input7);

                mid7[1] = NeonStoreF::$mul(mid7[1], self.twiddles32[5]);
                mid7[2] = NeonStoreF::$mul(mid7[2], self.bf8.rotate(self.twiddles32[4]));
                mid7[3] = NeonStoreF::$mul(mid7[3], self.twiddles32[3].neg());

                let input0 = [
                    load!(src, k, 0),
                    load!(src, k, 8),
                    load!(src, k, 16),
                    load!(src, k, 24),
                ];
                let mid0 = self.bf8.bf4.exec(input0);

                let tw = k * 31;

                 {
                    let output0 = self.bf8.exec([
                        mid0[0], mid1[0], mid2[0], mid3[0], mid4[0], mid5[0], mid6[0], mid7[0],
                    ]);
                    let output1 = self.bf8.exec([
                        mid0[1], mid1[1], mid2[1], mid3[1], mid4[1], mid5[1], mid6[1], mid7[1],
                    ]);
                    let output2 = self.bf8.exec([
                        mid0[2], mid1[2], mid2[2], mid3[2], mid4[2], mid5[2], mid6[2], mid7[2],
                    ]);
                    let output3 = self.bf8.exec([
                        mid0[3], mid1[3], mid2[3], mid3[3], mid4[3], mid5[3], mid6[3], mid7[3],
                    ]);

                    let q1 = NeonStoreF::$mul(output1[0], self.twiddles[tw]);
                    let q2 = NeonStoreF::$mul(output2[0], self.twiddles[1 + tw]);
                    let q3 = NeonStoreF::$mul(output3[0], self.twiddles[2 + tw]);
                    let t = transpose_2x2([output0[0], q1]);
                    let t1 = transpose_2x2([q2, q3]);
                    store0!(t[0], 0, dst, k);
                    store1!(t[1], 0, dst, k);
                    store0!(t1[0], 2, dst, k);
                    store1!(t1[1], 2, dst, k);

                    for q in 1..8 {
                        let q0 =
                            NeonStoreF::$mul(output0[q], self.twiddles[q * 4 - 1 + tw]);
                        let q1 = NeonStoreF::$mul(output1[q], self.twiddles[q * 4 + tw]);
                        let q2 =
                            NeonStoreF::$mul(output2[q], self.twiddles[q * 4 + 1 + tw]);
                        let q3 =
                            NeonStoreF::$mul(output3[q], self.twiddles[q * 4 + 2 + tw]);
                        let t = transpose_2x2([q0, q1]);
                        let t1 = transpose_2x2([q2, q3]);
                        store0!(t[0], q * 4, dst, k);
                        store1!(t[1], q * 4, dst, k);
                        store0!(t1[0], q * 4 + 2, dst, k);
                        store1!(t1[1], q * 4 + 2, dst, k);
                    }
                }
            }
        }
    }


    #[inline]
    #[target_feature(enable = $features)]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 2048];
        // columns
        self.exec_bf32_1(chunk.slice_from(0..), &mut scratch);
        // rows
        self.exec_bf64(&mut scratch, chunk.slice_from_mut(0..));
    }
}

boring_neon_butterfly!($name, $features, f32, 2048);
    };
}

gen_bf2048f!(
    NeonButterfly2048f,
    "neon",
    ColumnButterfly8f,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf2048f!(
    NeonFcmaForwardButterfly2048f,
    "fcma",
    ColumnFcmaForwardButterfly8f,
    fcmul_fcma
);
#[cfg(feature = "fcma")]
gen_bf2048f!(
    NeonFcmaInverseButterfly2048f,
    "fcma",
    ColumnFcmaInverseButterfly8f,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly_small, test_oof_butterfly_small};

    #[test]
    #[allow(unnameable_test_items)]
    fn capsule1() {
        if std::env::var("SHORT_TEST").as_deref() == Ok("yes") {
            return;
        }
        test_butterfly_small!(test_neon_butterfly2048, f32, NeonButterfly2048f, 2048, 1e-2);
        test_neon_butterfly2048();
    }

    #[test]
    #[allow(unnameable_test_items)]
    fn capsule2() {
        if std::env::var("SHORT_TEST").as_deref() == Ok("yes") {
            return;
        }
        if !std::arch::is_aarch64_feature_detected!("fcma") {
            return;
        }
        test_oof_butterfly_small!(
            test_oof_neon_butterfly2048,
            f32,
            NeonButterfly2048f,
            2048,
            1e-2
        );
        test_oof_neon_butterfly2048();
    }
}
