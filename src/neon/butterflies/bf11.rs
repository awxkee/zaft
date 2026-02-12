/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use crate::neon::butterflies::shared::{boring_neon_butterfly, boring_neon_butterfly2};
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

macro_rules! gen_bf11d {
    ($name: ident, $features: literal, $internal_bf: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf11: $internal_bf,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf11: $internal_bf::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly!($name, $features, f64, 11);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
                let mut rows = [NeonStoreD::default(); 11];
                for i in 0..11 {
                    rows[i] = NeonStoreD::from_complex_ref(chunk.slice_from(i..));
                }
                rows = self.bf11.exec(rows);
                for i in 0..11 {
                    rows[i].write(chunk.slice_from_mut(i..));
                }
            }
        }
    };
}

gen_bf11d!(NeonButterfly11d, "neon", ColumnButterfly11d);
#[cfg(feature = "fcma")]
gen_bf11d!(NeonFcmaButterfly11d, "fcma", ColumnFcmaButterfly11d);

macro_rules! gen_bf11f {
    ($name: ident, $features: literal, $internal_bf: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf11: $internal_bf,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf11: $internal_bf::new(fft_direction),
                }
            }
        }

        boring_neon_butterfly2!($name, $features, f32, 11);

        impl $name {
            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows = [NeonStoreF::default(); 11];
                for i in 0..11 {
                    rows[i] = NeonStoreF::from_complex(chunk.index(i));
                }
                rows = self.bf11.exec(rows);
                for i in 0..11 {
                    rows[i].write_lo(chunk.slice_from_mut(i..));
                }
            }

            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run2<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut rows = [NeonStoreF::default(); 11];
                unsafe {
                    for i in 0..5 {
                        let q0 = vld1q_f32(chunk.slice_from(i * 2..).as_ptr().cast());
                        let q1 = vld1q_f32(chunk.slice_from(i * 2 + 11..).as_ptr().cast());
                        rows[i * 2] =
                            NeonStoreF::raw(vcombine_f32(vget_low_f32(q0), vget_low_f32(q1)));
                        rows[i * 2 + 1] =
                            NeonStoreF::raw(vcombine_f32(vget_high_f32(q0), vget_high_f32(q1)));
                    }

                    let q0 = vld1_f32(chunk.slice_from(10..).as_ptr().cast());
                    let q1 = vld1_f32(chunk.slice_from(10 + 11..).as_ptr().cast());
                    rows[10] = NeonStoreF::raw(vcombine_f32(q0, q1));

                    rows = self.bf11.exec(rows);
                    for i in 0..5 {
                        let r0 = rows[i * 2];
                        let r1 = rows[i * 2 + 1];
                        let new_row0 =
                            NeonStoreF::raw(vcombine_f32(vget_low_f32(r0.v), vget_low_f32(r1.v)));
                        let new_row1 =
                            NeonStoreF::raw(vcombine_f32(vget_high_f32(r0.v), vget_high_f32(r1.v)));
                        new_row0.write(chunk.slice_from_mut(i * 2..));
                        new_row1.write(chunk.slice_from_mut(i * 2 + 11..));
                    }

                    let r0 = rows[10];
                    r0.write_lo(chunk.slice_from_mut(10..));
                    r0.write_hi(chunk.slice_from_mut(10 + 11..));
                }
            }
        }
    };
}

gen_bf11f!(NeonButterfly11f, "neon", ColumnButterfly11f);
#[cfg(feature = "fcma")]
gen_bf11f!(NeonFcmaButterfly11f, "fcma", ColumnFcmaButterfly11f);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly11, f32, NeonButterfly11f, 11, 1e-5);
    #[cfg(feature = "fcma")]
    test_butterfly!(test_fcma_butterfly11, f32, NeonFcmaButterfly11f, 11, 1e-5);
    test_butterfly!(test_neon_butterfly11_f64, f64, NeonButterfly11d, 11, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly11_f64,
        f64,
        NeonFcmaButterfly11d,
        11,
        1e-7
    );
    test_oof_butterfly!(test_oof_butterfly11, f32, NeonButterfly11f, 11, 1e-5);
    test_oof_butterfly!(test_oof_butterfly11_f64, f64, NeonButterfly11d, 11, 1e-9);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly11_f64,
        f64,
        NeonFcmaButterfly11d,
        11,
        1e-9
    );
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly11,
        f32,
        NeonFcmaButterfly11f,
        11,
        1e-4
    );
}
