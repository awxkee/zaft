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
use crate::avx::butterflies::shared::{
    boring_avx_butterfly, boring_avx512vl_butterfly, gen_butterfly_twiddles_f32,
};
use crate::avx::mixed::{AvxStoreF, ColumnButterfly32f};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! define_bf512f {
    ($bf_name: ident, $features: literal) => {
        pub(crate) struct $bf_name {
            direction: FftDirection,
            bf32: ColumnButterfly32f,
            twiddles: Box<[AvxStoreF; 120]>,
        }

        impl $bf_name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                unsafe { Self::new_init(fft_direction) }
            }

            #[target_feature(enable = $features)]
            fn new_init(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: Box::new(gen_butterfly_twiddles_f32(32, 16, fft_direction, 512)),
                    bf32: ColumnButterfly32f::new(fft_direction),
                }
            }
        }

        impl $bf_name {
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
                let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 512];
                // columns
                for k in 0..8 {
                    self.bf32.bf16.exec_transpose_streaming(
                        |idx| AvxStoreF::from_complex_ref(chunk.slice_from(k * 4 + idx * 32..)),
                        |idx| self.twiddles[15 * k + idx],
                        |idx, val| unsafe {
                            val.write_u(
                                scratch.get_unchecked_mut((k * 4 + idx % 4) * 16 + (idx / 4) * 4..),
                            )
                        },
                    )
                }
                // rows
                for k in 0..4 {
                    self.bf32.exec_streaming(
                        |i| unsafe {
                            AvxStoreF::from_complex_refu(scratch.get_unchecked(i * 16 + k * 4..))
                        },
                        |i, store| store.write(chunk.slice_from_mut(i * 16 + k * 4..)),
                    );
                }
            }
        }
    };
}

define_bf512f!(AvxButterfly512f, "avx2,fma");
define_bf512f!(Avx512vlButterfly512f, "avx2,fma,avx512f,avx512vl");

boring_avx_butterfly!(AvxButterfly512f, f32, 512);
boring_avx512vl_butterfly!(Avx512vlButterfly512f, f32, 512);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly512, f32, AvxButterfly512f, 512, 1e-3);
    test_oof_avx_butterfly!(test_oof_neon_butterfly512, f32, AvxButterfly512f, 512, 1e-3);
}
