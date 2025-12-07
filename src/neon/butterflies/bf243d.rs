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
#![allow(clippy::needless_range_loop)]

use crate::neon::butterflies::shared::gen_butterfly_twiddles_f64;
use crate::neon::mixed::NeonStoreD;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;
use std::sync::Arc;

macro_rules! gen_bf243d {
    ($name: ident, $feature: literal, $bf27_name: ident, $internal_bf: ident, $mul: ident) => {
        use crate::neon::mixed::$internal_bf;

        pub(crate) struct $bf27_name {
            bf9: $internal_bf,
            twiddle1: NeonStoreD,
            twiddle2: NeonStoreD,
            twiddle3: NeonStoreD,
            twiddle4: NeonStoreD,
            twiddle5: NeonStoreD,
            twiddle6: NeonStoreD,
            twiddle7: NeonStoreD,
            twiddle8: NeonStoreD,
            twiddle9: NeonStoreD,
            twiddle10: NeonStoreD,
            twiddle11: NeonStoreD,
            twiddle12: NeonStoreD,
        }

        impl $bf27_name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    twiddle1: NeonStoreD::from_complex(&compute_twiddle(1, 27, fft_direction)),
                    twiddle2: NeonStoreD::from_complex(&compute_twiddle(2, 27, fft_direction)),
                    twiddle3: NeonStoreD::from_complex(&compute_twiddle(3, 27, fft_direction)),
                    twiddle4: NeonStoreD::from_complex(&compute_twiddle(4, 27, fft_direction)),
                    twiddle5: NeonStoreD::from_complex(&compute_twiddle(5, 27, fft_direction)),
                    twiddle6: NeonStoreD::from_complex(&compute_twiddle(6, 27, fft_direction)),
                    twiddle7: NeonStoreD::from_complex(&compute_twiddle(7, 27, fft_direction)),
                    twiddle8: NeonStoreD::from_complex(&compute_twiddle(8, 27, fft_direction)),
                    twiddle9: NeonStoreD::from_complex(&compute_twiddle(10, 27, fft_direction)),
                    twiddle10: NeonStoreD::from_complex(&compute_twiddle(12, 27, fft_direction)),
                    twiddle11: NeonStoreD::from_complex(&compute_twiddle(14, 27, fft_direction)),
                    twiddle12: NeonStoreD::from_complex(&compute_twiddle(16, 27, fft_direction)),
                    bf9: $internal_bf::new(fft_direction),
                }
            }

            #[target_feature(enable = $feature)]
            fn exec(&self, src: &[MaybeUninit<Complex<f64>>], dst: &mut [Complex<f64>]) {
                macro_rules! load {
                    ($src: expr, $idx: expr) => {{ unsafe { NeonStoreD::from_complex_refu($src.get_unchecked($idx * 9..)) } }};
                }

                let s0 = self.bf9.exec([
                    load!(src, 0),
                    load!(src, 3),
                    load!(src, 6),
                    load!(src, 9),
                    load!(src, 12),
                    load!(src, 15),
                    load!(src, 18),
                    load!(src, 21),
                    load!(src, 24),
                ]);
                let mut s1 = self.bf9.exec([
                    load!(src, 1),
                    load!(src, 4),
                    load!(src, 7),
                    load!(src, 10),
                    load!(src, 13),
                    load!(src, 16),
                    load!(src, 19),
                    load!(src, 22),
                    load!(src, 25),
                ]);
                let mut s2 = self.bf9.exec([
                    load!(src, 2),
                    load!(src, 5),
                    load!(src, 8),
                    load!(src, 11),
                    load!(src, 14),
                    load!(src, 17),
                    load!(src, 20),
                    load!(src, 23),
                    load!(src, 26),
                ]);

                macro_rules! store {
                    ($v: expr, $idx: expr, $dst: expr) => {{ unsafe { $v.write($dst.get_unchecked_mut($idx * 9..)) } }};
                }

                let z0 = self.bf9.bf3.exec([s0[0], s1[0], s2[0]]);
                store!(z0[0], 0, dst);
                store!(z0[1], 9, dst);
                store!(z0[2], 18, dst);

                s1[1] = NeonStoreD::$mul(s1[1], self.twiddle1);
                s1[2] = NeonStoreD::$mul(s1[2], self.twiddle2);
                s1[3] = NeonStoreD::$mul(s1[3], self.twiddle3);
                s1[4] = NeonStoreD::$mul(s1[4], self.twiddle4);
                s1[5] = NeonStoreD::$mul(s1[5], self.twiddle5);
                s1[6] = NeonStoreD::$mul(s1[6], self.twiddle6);
                s1[7] = NeonStoreD::$mul(s1[7], self.twiddle7);
                s1[8] = NeonStoreD::$mul(s1[8], self.twiddle8);
                s2[1] = NeonStoreD::$mul(s2[1], self.twiddle2);
                s2[2] = NeonStoreD::$mul(s2[2], self.twiddle4);
                s2[3] = NeonStoreD::$mul(s2[3], self.twiddle6);
                s2[4] = NeonStoreD::$mul(s2[4], self.twiddle8);
                s2[5] = NeonStoreD::$mul(s2[5], self.twiddle9);
                s2[6] = NeonStoreD::$mul(s2[6], self.twiddle10);
                s2[7] = NeonStoreD::$mul(s2[7], self.twiddle11);
                s2[8] = NeonStoreD::$mul(s2[8], self.twiddle12);

                let z1 = self.bf9.bf3.exec([s0[1], s1[1], s2[1]]);
                store!(z1[0], 1, dst);
                store!(z1[1], 10, dst);
                store!(z1[2], 19, dst);
                let z2 = self.bf9.bf3.exec([s0[2], s1[2], s2[2]]);
                store!(z2[0], 2, dst);
                store!(z2[1], 11, dst);
                store!(z2[2], 20, dst);
                let z3 = self.bf9.bf3.exec([s0[3], s1[3], s2[3]]);
                store!(z3[0], 3, dst);
                store!(z3[1], 12, dst);
                store!(z3[2], 21, dst);
                let z4 = self.bf9.bf3.exec([s0[4], s1[4], s2[4]]);
                store!(z4[0], 4, dst);
                store!(z4[1], 13, dst);
                store!(z4[2], 22, dst);
                let z5 = self.bf9.bf3.exec([s0[5], s1[5], s2[5]]);
                store!(z5[0], 5, dst);
                store!(z5[1], 14, dst);
                store!(z5[2], 23, dst);
                let z6 = self.bf9.bf3.exec([s0[6], s1[6], s2[6]]);
                store!(z6[0], 6, dst);
                store!(z6[1], 15, dst);
                store!(z6[2], 24, dst);
                let z7 = self.bf9.bf3.exec([s0[7], s1[7], s2[7]]);
                store!(z7[0], 7, dst);
                store!(z7[1], 16, dst);
                store!(z7[2], 25, dst);
                let z8 = self.bf9.bf3.exec([s0[8], s1[8], s2[8]]);
                store!(z8[0], 8, dst);
                store!(z8[1], 17, dst);
                store!(z8[2], 26, dst);
            }
        }

        pub(crate) struct $name {
            direction: FftDirection,
            bf27: $bf27_name,
            twiddles: [NeonStoreD; 216],
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(27, 9, fft_direction, 243),
                    bf27: $bf27_name::new(fft_direction),
                }
            }
        }

        impl FftExecutor<f64> for $name {
            fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                unsafe { self.execute_impl(in_place) }
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                243
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(243) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows: [NeonStoreD; 9] = [NeonStoreD::default(); 9];
                    let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 243];

                    for chunk in in_place.chunks_exact_mut(243) {
                        // columns
                        for k in 0..27 {
                            for i in 0..9 {
                                rows[i] =
                                    NeonStoreD::from_complex_ref(chunk.get_unchecked(i * 27 + k..));
                            }

                            rows = self.bf27.bf9.exec(rows);

                            for i in 1..9 {
                                rows[i] = NeonStoreD::$mul(rows[i], self.twiddles[i - 1 + 8 * k]);
                            }

                            for i in 0..9 {
                                rows[i].write_uninit(scratch.get_unchecked_mut(k * 9 + i..));
                            }
                        }

                        // rows

                        for k in 0..9 {
                            self.bf27
                                .exec(scratch.get_unchecked(k..), chunk.get_unchecked_mut(k..))
                        }
                    }
                }
                Ok(())
            }
        }

        impl FftExecutorOutOfPlace<f64> for $name {
            fn execute_out_of_place(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_out_of_place_impl(src, dst) }
            }
        }

        impl $name {
            #[target_feature(enable = $feature)]
            fn execute_out_of_place_impl(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(243) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(243) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                let mut rows: [NeonStoreD; 9] = [NeonStoreD::default(); 9];
                let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 243];

                unsafe {
                    for (dst, src) in dst.chunks_exact_mut(243).zip(src.chunks_exact(243)) {
                        // columns
                        for k in 0..27 {
                            for i in 0..9 {
                                rows[i] =
                                    NeonStoreD::from_complex_ref(src.get_unchecked(i * 27 + k..));
                            }

                            rows = self.bf27.bf9.exec(rows);

                            for i in 1..9 {
                                rows[i] = NeonStoreD::$mul(rows[i], self.twiddles[i - 1 + 8 * k]);
                            }

                            for i in 0..9 {
                                rows[i].write_uninit(scratch.get_unchecked_mut(k * 9 + i..));
                            }
                        }

                        // rows

                        for k in 0..9 {
                            self.bf27
                                .exec(scratch.get_unchecked(k..), dst.get_unchecked_mut(k..))
                        }
                    }
                }
                Ok(())
            }
        }

        impl CompositeFftExecutor<f64> for $name {
            fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
                self
            }
        }
    };
}

gen_bf243d!(
    NeonButterfly243d,
    "neon",
    ColumnButterfly27d,
    ColumnButterfly9d,
    mul_by_complex
);
#[cfg(feature = "fcma")]
gen_bf243d!(
    NeonFcmaButterfly243d,
    "fcma",
    ColumnFcmaButterfly27d,
    ColumnFcmaButterfly9d,
    fcmul_fcma
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(
        test_neon_butterfly243_f64,
        f64,
        NeonButterfly243d,
        243,
        1e-7
    );
    test_oof_butterfly!(
        test_oof_neon_butterfly243_f64,
        f64,
        NeonButterfly243d,
        243,
        1e-7
    );

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly243_f64,
        f64,
        NeonFcmaButterfly243d,
        243,
        1e-7
    );
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly243_f64,
        f64,
        NeonFcmaButterfly243d,
        243,
        1e-7
    );
}
