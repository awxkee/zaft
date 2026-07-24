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

use crate::avx::butterflies::shared::{
    boring_avx_butterfly, boring_avx512vl_butterfly, gen_butterfly_twiddles_f32,
};
use crate::avx::mixed::{AvxStoreF, ColumnButterfly32f};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! define_bf256 {
    ($bf_name: ident, $features: literal) => {
        pub(crate) struct $bf_name {
            direction: FftDirection,
            bf32: ColumnButterfly32f,
            twiddles: Box<[AvxStoreF; 56]>,
        }

        impl $bf_name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                unsafe { Self::new_init(fft_direction) }
            }

            #[target_feature(enable = $features)]
            pub(crate) fn new_init(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: Box::new(gen_butterfly_twiddles_f32(32, 8, fft_direction, 256)),
                    bf32: ColumnButterfly32f::new(fft_direction),
                }
            }
        }

        impl $bf_name {
            #[inline]
            #[target_feature(enable = $features)]
            fn exec_bf16(&self, src: &[Complex<f32>], dst: &mut [MaybeUninit<Complex<f32>>; 256]) {
                let mut rows: [AvxStoreF; 8] = [AvxStoreF::zero(); 8];
                unsafe {
                    // columns
                    for k in 0..8 {
                        for i in 0..8 {
                            rows[i] =
                                AvxStoreF::from_complex_ref(src.get_unchecked(i * 32 + k * 4..));
                        }

                        rows = self.bf32.bf16.bf8.exec(rows);

                        let q1 = AvxStoreF::mul_by_complex(rows[1], self.twiddles[7 * k]);
                        let q2 = AvxStoreF::mul_by_complex(rows[2], self.twiddles[7 * k + 1]);
                        let q3 = AvxStoreF::mul_by_complex(rows[3], self.twiddles[7 * k + 2]);
                        use crate::avx::transpose::transpose_f32x2_4x4_aos;
                        let t = transpose_f32x2_4x4_aos([rows[0], q1, q2, q3]);
                        t[0].write_u(dst.get_unchecked_mut(k * 4 * 8..));
                        t[1].write_u(dst.get_unchecked_mut((k * 4 + 1) * 8..));
                        t[2].write_u(dst.get_unchecked_mut((k * 4 + 2) * 8..));
                        t[3].write_u(dst.get_unchecked_mut((k * 4 + 3) * 8..));

                        {
                            let i = 1;
                            let q0 = AvxStoreF::mul_by_complex(
                                rows[i * 4],
                                self.twiddles[(i - 1) * 4 + 3 + 7 * k],
                            );
                            let q1 = AvxStoreF::mul_by_complex(
                                rows[i * 4 + 1],
                                self.twiddles[(i - 1) * 4 + 4 + 7 * k],
                            );
                            let q2 = AvxStoreF::mul_by_complex(
                                rows[i * 4 + 2],
                                self.twiddles[(i - 1) * 4 + 5 + 7 * k],
                            );
                            let q3 = AvxStoreF::mul_by_complex(
                                rows[i * 4 + 3],
                                self.twiddles[(i - 1) * 4 + 6 + 7 * k],
                            );
                            let t = transpose_f32x2_4x4_aos([q0, q1, q2, q3]);
                            t[0].write_u(dst.get_unchecked_mut(k * 4 * 8 + i * 4..));
                            t[1].write_u(dst.get_unchecked_mut((k * 4 + 1) * 8 + i * 4..));
                            t[2].write_u(dst.get_unchecked_mut((k * 4 + 2) * 8 + i * 4..));
                            t[3].write_u(dst.get_unchecked_mut((k * 4 + 3) * 8 + i * 4..));
                        }
                    }
                }
            }

            #[inline]
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 256];
                self.exec_bf16(chunk.slice_from(0..), &mut scratch);
                for k in 0..2 {
                    self.bf32.exec_streaming(
                        |i| unsafe {
                            AvxStoreF::from_complex_refu(scratch.get_unchecked(i * 8 + k * 4..))
                        },
                        |i, store| store.write(chunk.slice_from_mut(i * 8 + k * 4..)),
                    );
                }
            }
        }
    };
}

define_bf256!(AvxButterfly256f, "avx2,fma");
define_bf256!(Avx512vlButterfly256f, "avx2,fma,avx512f,avx512vl");

boring_avx_butterfly!(AvxButterfly256f, f32, 256);
boring_avx512vl_butterfly!(Avx512vlButterfly256f, f32, 256);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly256, f32, AvxButterfly256f, 256, 1e-3);
    test_oof_avx_butterfly!(test_oof_avx_butterfly256, f32, AvxButterfly256f, 256, 1e-3);
}
