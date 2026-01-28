/*
 * // Copyright (c) Radzivon Bartoshyk 02/2025. All rights reserved.
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
use crate::avx::butterflies::bf16::transpose_4x4;
use crate::avx::butterflies::shared::{boring_avx_butterfly, gen_butterfly_twiddles_f64};
use crate::avx::mixed::{AvxStoreD, ColumnButterfly4d, ColumnButterfly7d};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_7x4(
    rows0: [AvxStoreD; 4],
    rows1: [AvxStoreD; 4],
    rows2: [AvxStoreD; 4],
    rows3: [AvxStoreD; 4],
) -> ([AvxStoreD; 7], [AvxStoreD; 7]) {
    let a0 = transpose_4x4(rows0, rows1);
    let b0 = transpose_4x4(rows2, rows3);
    (
        [
            // row 0
            a0.0[0], a0.0[1], a0.0[2], a0.0[3], b0.0[0], b0.0[1], b0.0[2],
        ],
        [
            // row 0
            a0.1[0], a0.1[1], a0.1[2], a0.1[3], b0.1[0], b0.1[1], b0.1[2],
        ],
    )
}

pub(crate) struct AvxButterfly28d {
    direction: FftDirection,
    bf7: ColumnButterfly7d,
    bf4: ColumnButterfly4d,
    twiddles: [AvxStoreD; 12],
}

impl AvxButterfly28d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_f64(7, 4, fft_direction, 28),
            bf7: ColumnButterfly7d::new(fft_direction),
            bf4: ColumnButterfly4d::new(fft_direction),
            direction: fft_direction,
        }
    }
}

impl AvxButterfly28d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows0: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
        let mut rows1: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
        let mut rows2: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
        let mut rows3: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
        // columns
        for i in 0..4 {
            rows0[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 7..));
            rows1[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 7 + 2..));
            rows2[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 7 + 4..));
            rows3[i] = AvxStoreD::from_complex(chunk.index(i * 7 + 6));
        }

        rows0 = self.bf4.exec(rows0);
        rows1 = self.bf4.exec(rows1);
        rows2 = self.bf4.exec(rows2);
        rows3 = self.bf4.exec(rows3);

        for i in 1..4 {
            rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1]);
            rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 3]);
            rows2[i] = AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 6]);
            rows3[i] = AvxStoreD::mul_by_complex(rows3[i], self.twiddles[i - 1 + 9]);
        }

        let t = transpose_7x4(rows0, rows1, rows2, rows3);

        // rows
        let left = self.bf7.exec(t.0);
        let right = self.bf7.exec(t.1);

        for i in 0..7 {
            left[i].write(chunk.slice_from_mut(i * 4..));
        }
        for i in 0..7 {
            right[i].write(chunk.slice_from_mut(i * 4 + 2..));
        }
    }
}

boring_avx_butterfly!(AvxButterfly28d, f64, 28);

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly28d, f64, AvxButterfly28d, 28, 1e-7);
    test_oof_avx_butterfly!(test_avx_oof_butterfly28d, f64, AvxButterfly28d, 28, 1e-7);
}
