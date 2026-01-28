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

use crate::avx::butterflies::shared::{boring_avx_butterfly, gen_butterfly_twiddles_f32};
use crate::avx::mixed::{AvxStoreD, AvxStoreF, ColumnButterfly7d, ColumnButterfly7f};
use crate::avx::transpose::{store_transpose_7x7_f32, transpose_7x7_f64};
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

pub(crate) struct AvxButterfly49d {
    direction: FftDirection,
    bf7: ColumnButterfly7d,
    twiddles: [AvxStoreD; 24],
}

impl AvxButterfly49d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let mut twiddles = [AvxStoreD::zero(); 24];
            let mut q = 0usize;
            let len_per_row = 7;
            const COMPLEX_PER_VECTOR: usize = 2;
            let quotient = len_per_row / COMPLEX_PER_VECTOR;
            let remainder = len_per_row % COMPLEX_PER_VECTOR;

            let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
            for x in 0..num_twiddle_columns {
                for y in 1..7 {
                    twiddles[q] = AvxStoreD::set_complex2(
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR), 49, fft_direction),
                        compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), 49, fft_direction),
                    );
                    q += 1;
                }
            }
            Self {
                direction: fft_direction,
                twiddles,
                bf7: ColumnButterfly7d::new(fft_direction),
            }
        }
    }
}

boring_avx_butterfly!(AvxButterfly49d, f64, 49);

impl AvxButterfly49d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows1: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];
        let mut rows2: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];
        let mut rows3: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];
        let mut rows4: [AvxStoreD; 7] = [AvxStoreD::zero(); 7];
        // rows - 1

        for i in 0..7 {
            rows1[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 7..));
        }

        rows1 = self.bf7.exec(rows1);

        for i in 1..7 {
            rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1]);
        }

        // rows - 2

        for i in 0..7 {
            rows2[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 7 + 2..));
        }

        rows2 = self.bf7.exec(rows2);

        for i in 1..7 {
            rows2[i] = AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 6]);
        }

        // rows - 3

        for i in 0..7 {
            rows3[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 7 + 4..));
        }

        rows3 = self.bf7.exec(rows3);

        for i in 1..7 {
            rows3[i] = AvxStoreD::mul_by_complex(rows3[i], self.twiddles[i - 1 + 12]);
        }

        // rows - 4

        for i in 0..7 {
            rows4[i] = AvxStoreD::from_complex(chunk.index(i * 7 + 6));
        }

        rows4 = self.bf7.exec(rows4);

        for i in 1..7 {
            rows4[i] = AvxStoreD::mul_by_complex(rows4[i], self.twiddles[i - 1 + 18]);
        }

        let (mut transposed0, mut transposed1, mut transposed2, mut transposed3) =
            transpose_7x7_f64(rows1, rows2, rows3, rows4);

        transposed0 = self.bf7.exec(transposed0);

        for i in 0..7 {
            transposed0[i].write(chunk.slice_from_mut(i * 7..));
        }

        transposed1 = self.bf7.exec(transposed1);

        for i in 0..7 {
            transposed1[i].write(chunk.slice_from_mut(i * 7 + 2..));
        }

        transposed2 = self.bf7.exec(transposed2);

        for i in 0..7 {
            transposed2[i].write(chunk.slice_from_mut(i * 7 + 4..));
        }

        transposed3 = self.bf7.exec(transposed3);

        for i in 0..7 {
            transposed3[i].write_lo(chunk.slice_from_mut(i * 7 + 6..));
        }
    }
}

pub(crate) struct AvxButterfly49f {
    direction: FftDirection,
    bf7: ColumnButterfly7f,
    twiddles: [AvxStoreF; 12],
}

impl AvxButterfly49f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                direction: fft_direction,
                twiddles: gen_butterfly_twiddles_f32(7, 7, fft_direction, 49),
                bf7: ColumnButterfly7f::new(fft_direction),
            }
        }
    }
}

boring_avx_butterfly!(AvxButterfly49f, f32, 49);

impl AvxButterfly49f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut rows1: [AvxStoreF; 7] = [AvxStoreF::zero(); 7];
        let mut rows2: [AvxStoreF; 7] = [AvxStoreF::zero(); 7];
        for i in 0..7 {
            rows1[i] = AvxStoreF::from_complex_ref(chunk.slice_from(i * 7..));
        }
        for i in 0..7 {
            rows2[i] = AvxStoreF::from_complex3(chunk.slice_from(i * 7 + 4..));
        }

        rows1 = self.bf7.exec(rows1);
        rows2 = self.bf7.exec(rows2);

        for i in 1..7 {
            rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1]);
            rows2[i] = AvxStoreF::mul_by_complex(rows2[i], self.twiddles[i - 1 + 6]);
        }

        let (mut transposed0, mut transposed1) = store_transpose_7x7_f32(rows1, rows2);

        transposed0 = self.bf7.exec(transposed0);
        transposed1 = self.bf7.exec(transposed1);

        transposed0[0].write(chunk.slice_from_mut(0..));
        transposed0[1].write(chunk.slice_from_mut(7..));
        transposed0[2].write(chunk.slice_from_mut(14..));
        transposed0[3].write(chunk.slice_from_mut(21..));
        transposed0[4].write(chunk.slice_from_mut(28..));
        transposed0[5].write(chunk.slice_from_mut(35..));
        transposed0[6].write(chunk.slice_from_mut(42..));

        transposed1[0].write_lo3(chunk.slice_from_mut(4..));
        transposed1[1].write_lo3(chunk.slice_from_mut(11..));
        transposed1[2].write_lo3(chunk.slice_from_mut(18..));
        transposed1[3].write_lo3(chunk.slice_from_mut(25..));
        transposed1[4].write_lo3(chunk.slice_from_mut(32..));
        transposed1[5].write_lo3(chunk.slice_from_mut(39..));
        transposed1[6].write_lo3(chunk.slice_from_mut(46..));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly49, f32, AvxButterfly49f, 49, 1e-3);
    test_avx_butterfly!(test_avx_butterfly49_f64, f64, AvxButterfly49d, 49, 1e-7);
    test_oof_avx_butterfly!(test_oof_avx_butterfly49, f32, AvxButterfly49f, 49, 1e-3);
    test_oof_avx_butterfly!(test_oof_avx_butterfly49_f64, f64, AvxButterfly49d, 49, 1e-9);
}
