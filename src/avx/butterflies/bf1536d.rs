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
use crate::avx::butterflies::shared::{
    boring_avx_butterfly, boring_avx512vl_butterfly, gen_butterfly_twiddles_f64,
};
use crate::avx::mixed::{AvxStoreD, ColumnButterfly32d, ColumnButterfly48d};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

macro_rules! define_bf1536 {
    ($bf_name: ident, $features: literal) => {
        pub(crate) struct $bf_name {
            direction: FftDirection,
            bf32: ColumnButterfly32d,
            bf48: ColumnButterfly48d,
            twiddles: [AvxStoreD; 744],
        }

        impl $bf_name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                unsafe { Self::new_init(fft_direction) }
            }

            #[target_feature(enable = $features)]
            fn new_init(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    twiddles: gen_butterfly_twiddles_f64(48, 32, fft_direction, 1536),
                    bf32: ColumnButterfly32d::new(fft_direction),
                    bf48: ColumnButterfly48d::new(fft_direction),
                }
            }
        }

        impl $bf_name {
            #[target_feature(enable = $features)]
            fn exec_bf48(&self, src: &[MaybeUninit<Complex<f64>>; 1536], dst: &mut [Complex<f64>]) {
                for k in 0..16 {
                    self.bf48.exec_streaming(
                        |i| unsafe {
                            AvxStoreD::from_complex_refu(src.get_unchecked(i * 32 + k * 2..))
                        },
                        |i, store| unsafe { store.write(dst.get_unchecked_mut(i * 32 + k * 2..)) },
                    );
                }
            }
        }

        impl $bf_name {
            #[target_feature(enable = $features)]
            pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
                let mut scratch = [MaybeUninit::<Complex<f64>>::uninit(); 1536];
                // columns
                for k in 0..24 {
                    let tw = k * 31;
                    self.bf32.exec_transpose_streaming(
                        |idx| AvxStoreD::from_complex_ref(chunk.slice_from(k * 2 + idx * 48..)),
                        |idx| self.twiddles[tw + idx],
                        |idx, val| unsafe {
                            let row = k * 2 + idx % 2;
                            let col = (idx / 2) * 2;
                            val.write_u(scratch.get_unchecked_mut(row * 32 + col..))
                        },
                    );
                }
                // rows
                self.exec_bf48(&scratch, chunk.slice_from_mut(0..));
            }
        }
    };
}

define_bf1536!(AvxButterfly1536d, "avx2,fma");
define_bf1536!(Avx512vlButterfly1536d, "avx2,fma,avx512f,avx512vl");

boring_avx_butterfly!(AvxButterfly1536d, f64, 1536);
boring_avx512vl_butterfly!(Avx512vlButterfly1536d, f64, 1536);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly1536d, f64, AvxButterfly1536d, 1536, 1e-3);
    test_oof_avx_butterfly!(
        test_oof_avx_butterfly1536d,
        f64,
        AvxButterfly1536d,
        1536,
        1e-3
    );
}
