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
use crate::avx::butterflies::shared::{
    boring_avx_butterfly, boring_avx512vl_butterfly, gen_butterfly_twiddles_f32,
};
use crate::avx::mixed::{AvxStoreF, ColumnButterfly32f, ColumnButterfly36f};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! define_bf1152f {
    ($bf_name: ident, $features: literal) => {
        pub(crate) struct $bf_name {
            direction: FftDirection,
            bf32: ColumnButterfly32f,
            bf36: ColumnButterfly36f,
            twiddles: [AvxStoreF; 279],
        }

        impl $bf_name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                unsafe { Self::new_init(fft_direction) }
            }

            #[target_feature(enable = $features)]
            fn new_init(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f32(36, 32, fft_direction, 1152),
                    bf32: ColumnButterfly32f::new(fft_direction),
                    bf36: ColumnButterfly36f::new(fft_direction),
                }
            }
        }

        impl $bf_name {
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 1152];
                // columns
                for k in 0..9 {
                    let tw = k * 31;
                    self.bf32.exec_transpose_streaming(
                        |idx| AvxStoreF::from_complex_ref(chunk.slice_from(k * 4 + idx * 36..)),
                        |idx| self.twiddles[tw + idx],
                        |idx, val| unsafe {
                            val.write_u(
                                scratch.get_unchecked_mut((k * 4 + idx % 4) * 32 + idx / 4 * 4..),
                            )
                        },
                    );
                }
                // rows
                for k in 0..8 {
                    self.bf36.exec_streaming(
                        |i| unsafe {
                            AvxStoreF::from_complex_refu(scratch.get_unchecked(i * 32 + k * 4..))
                        },
                        |i, store| store.write(chunk.slice_from_mut(i * 32 + k * 4..)),
                    );
                }
            }
        }
    };
}

define_bf1152f!(AvxButterfly1152f, "avx2,fma");
define_bf1152f!(Avx512vlButterfly1152f, "avx2,fma,avx512f,avx512vl");

boring_avx_butterfly!(AvxButterfly1152f, f32, 1152);
boring_avx512vl_butterfly!(Avx512vlButterfly1152f, f32, 1152);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly1152, f32, AvxButterfly1152f, 1152, 1e-2);
    test_oof_avx_butterfly!(
        test_oof_avx_butterfly1152,
        f32,
        AvxButterfly1152f,
        1152,
        1e-2
    );
}
