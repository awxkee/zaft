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

use crate::neon::butterflies::shared::boring_neon_butterfly;
use crate::neon::mixed::NeonStoreD;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::ops::Neg;

macro_rules! gen_bf64d {
    ($name: ident, $features: literal, $internal_bf: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf8: $internal_bf,
            twiddles64: [NeonStoreD; 7],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf8: $internal_bf::new(fft_direction),
                    twiddles64: [
                        NeonStoreD::from_complex(&compute_twiddle(1, 64, fft_direction)),
                        NeonStoreD::from_complex(&compute_twiddle(2, 64, fft_direction)),
                        NeonStoreD::from_complex(&compute_twiddle(3, 64, fft_direction)),
                        NeonStoreD::from_complex(&compute_twiddle(4, 64, fft_direction)),
                        NeonStoreD::from_complex(&compute_twiddle(5, 64, fft_direction)),
                        NeonStoreD::from_complex(&compute_twiddle(6, 64, fft_direction)),
                        NeonStoreD::from_complex(&compute_twiddle(7, 64, fft_direction)),
                    ],
                }
            }
        }

        boring_neon_butterfly!($name, $features, f64, 64);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
                    macro_rules! load {
                    ($src: expr, $k: expr, $idx: expr) => {{ NeonStoreD::from_complex_ref(chunk.slice_from($idx..)) }};
                }

                macro_rules! store {
                    ($v: expr, $idx: expr, $dst: expr, $k: expr) => {{ $v.write(chunk.slice_from_mut($idx..)) }};
                }

                let input1: [NeonStoreD; 8] = std::array::from_fn(|x| load!(src, k, x * 8 + 1));
                let mut mid1 = self.bf8.exec(input1);

                for i in 0..7 {
                    mid1[i + 1] = NeonStoreD::$mul(mid1[i + 1], self.twiddles64[i]);
                }

                let input2 = std::array::from_fn(|x| load!(src, k, x * 8 + 2));

                let mut mid2 = self.bf8.exec(input2);

                mid2[1] = NeonStoreD::$mul(mid2[1], self.twiddles64[1]);
                mid2[2] = NeonStoreD::$mul(mid2[2], self.twiddles64[3]);
                mid2[3] = NeonStoreD::$mul(mid2[3], self.twiddles64[5]);
                mid2[4] = self.bf8.rotate45(mid2[4]);
                mid2[5] = NeonStoreD::$mul(mid2[5], self.bf8.rotate45(self.twiddles64[1]));
                mid2[6] = NeonStoreD::$mul(mid2[6], self.bf8.rotate45(self.twiddles64[3]));
                mid2[7] = NeonStoreD::$mul(mid2[7], self.bf8.rotate45(self.twiddles64[5]));

                let input3 = std::array::from_fn(|x| load!(src, k, x * 8 + 3));
                let mut mid3 = self.bf8.exec(input3);

                mid3[1] = NeonStoreD::$mul(mid3[1], self.twiddles64[2]);                    // W₆₄^3  = t3
                mid3[2] = NeonStoreD::$mul(mid3[2], self.twiddles64[5]);                    // W₆₄^6  = t6
                mid3[3] = NeonStoreD::$mul(mid3[3], self.bf8.rotate45(self.twiddles64[0])); // W₆₄^9  = W₈¹·t1
                mid3[4] = NeonStoreD::$mul(mid3[4], self.bf8.rotate45(self.twiddles64[3])); // W₆₄^12 = W₈¹·t4
                mid3[5] = NeonStoreD::$mul(mid3[5], self.bf8.rotate45(self.twiddles64[6])); // W₆₄^15 = W₈¹·t7
                mid3[6] = NeonStoreD::$mul(mid3[6], self.bf8.rotate(self.twiddles64[1])); // W₆₄^18 = W₈²·t2
                mid3[7] = NeonStoreD::$mul(mid3[7], self.bf8.rotate(self.twiddles64[4])); // W₆₄^21 = W₈²·t5

                let input4 = std::array::from_fn(|x| load!(src, k, x * 8 + 4));
                let mut mid4 = self.bf8.exec(input4);

                mid4[1] = NeonStoreD::$mul(mid4[1], self.twiddles64[3]);                    // W₆₄^4  = t4
                mid4[2] = self.bf8.rotate45(mid4[2]);                                        // W₆₄^8  = W₈¹
                mid4[3] = NeonStoreD::$mul(mid4[3], self.bf8.rotate45(self.twiddles64[3])); // W₆₄^12 = W₈¹·t4
                mid4[4] = self.bf8.rotate(mid4[4]);                                        // W₆₄^16 = W₈²
                mid4[5] = NeonStoreD::$mul(mid4[5], self.bf8.rotate(self.twiddles64[3])); // W₆₄^20 = W₈²·t4
                mid4[6] = self.bf8.rotate135(mid4[6]);                                        // W₆₄^24 = W₈³
                mid4[7] = NeonStoreD::$mul(mid4[7], self.bf8.rotate135(self.twiddles64[3])); // W₆₄^28 = W₈³·t4

                let input5 = std::array::from_fn(|x| load!(src, k, x * 8 + 5));
                let mut mid5 = self.bf8.exec(input5);

                mid5[1] = NeonStoreD::$mul(mid5[1], self.twiddles64[4]);                    // W₆₄^5  = t5
                mid5[2] = NeonStoreD::$mul(mid5[2], self.bf8.rotate45(self.twiddles64[1])); // W₆₄^10 = W₈¹·t2
                mid5[3] = NeonStoreD::$mul(mid5[3], self.bf8.rotate45(self.twiddles64[6])); // W₆₄^15 = W₈¹·t7
                mid5[4] = NeonStoreD::$mul(mid5[4], self.bf8.rotate(self.twiddles64[3])); // W₆₄^20 = W₈²·t4
                mid5[5] = NeonStoreD::$mul(mid5[5], self.bf8.rotate135(self.twiddles64[0])); // W₆₄^25 = W₈³·t1
                mid5[6] = NeonStoreD::$mul(mid5[6], self.bf8.rotate135(self.twiddles64[5])); // W₆₄^30 = W₈³·t6
                mid5[7] = NeonStoreD::$mul(mid5[7], self.twiddles64[2].neg());             // W₆₄^35 = −t3

                let input6 = std::array::from_fn(|x| load!(src, k, x * 8 + 6));
                let mut mid6 = self.bf8.exec(input6);

                mid6[1] = NeonStoreD::$mul(mid6[1], self.twiddles64[5]);                    // W₆₄^6  = t6
                mid6[2] = NeonStoreD::$mul(mid6[2], self.bf8.rotate45(self.twiddles64[3])); // W₆₄^12 = W₈¹·t4
                mid6[3] = NeonStoreD::$mul(mid6[3], self.bf8.rotate(self.twiddles64[1])); // W₆₄^18 = W₈²·t2
                mid6[4] = self.bf8.rotate135(mid6[4]);                                        // W₆₄^24 = W₈³
                mid6[5] = NeonStoreD::$mul(mid6[5], self.bf8.rotate135(self.twiddles64[5])); // W₆₄^30 = W₈³·t6
                mid6[6] = NeonStoreD::$mul(mid6[6], self.twiddles64[3].neg());             // W₆₄^36 = −t4
                mid6[7] = NeonStoreD::$mul(mid6[7], self.bf8.rotate225(self.twiddles64[1])); // W₆₄^42 = W₈⁵·t2

                let input7 = std::array::from_fn(|x| load!(src, k, x * 8 + 7));
                let mut mid7 = self.bf8.exec(input7);

                mid7[1] = NeonStoreD::$mul(mid7[1], self.twiddles64[6]);                    // W₆₄^7  = t7
                mid7[2] = NeonStoreD::$mul(mid7[2], self.bf8.rotate45(self.twiddles64[5])); // W₆₄^14 = W₈¹·t6
                mid7[3] = NeonStoreD::$mul(mid7[3], self.bf8.rotate(self.twiddles64[4]));  // W₆₄^21 = W₈²·t5
                mid7[4] = NeonStoreD::$mul(mid7[4], self.bf8.rotate135(self.twiddles64[3])); // W₆₄^28 = W₈³·t4
                mid7[5] = NeonStoreD::$mul(mid7[5], self.twiddles64[2].neg());             // W₆₄^35 = −t3
                mid7[6] = NeonStoreD::$mul(mid7[6], self.bf8.rotate225(self.twiddles64[1])); // W₆₄^42 = W₈⁵·t2
                mid7[7] = NeonStoreD::$mul(mid7[7], self.bf8.rotate270(self.twiddles64[0])); // W₆₄^49 = W₈⁶·t1

                let input0: [NeonStoreD; 8] = std::array::from_fn(|x| load!(src, k, x * 8));
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
    };
}

gen_bf64d!(NeonButterfly64d, "neon", ColumnButterfly8d, mul_by_complex);
#[cfg(feature = "fcma")]
gen_bf64d!(
    NeonFcmaButterfly64d,
    "fcma",
    ColumnFcmaButterfly8d,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly64_f64, f64, NeonButterfly64d, 64, 1e-7);
    test_oof_butterfly!(
        test_oof_neon_butterfly64_f64,
        f64,
        NeonButterfly64d,
        64,
        1e-7
    );

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly64_f64,
        f64,
        NeonFcmaButterfly64d,
        64,
        1e-7
    );
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly64_f64,
        f64,
        NeonFcmaButterfly64d,
        64,
        1e-7
    );
}
