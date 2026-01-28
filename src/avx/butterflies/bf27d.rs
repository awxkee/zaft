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
use crate::avx::butterflies::bf16::transpose_4x4;
use crate::avx::butterflies::shared::{boring_avx_butterfly, gen_butterfly_twiddles_f64};
use crate::avx::mixed::{AvxStoreD, ColumnButterfly9d};
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_f64x2_9x3(
    rows0: [AvxStoreD; 3],
    rows1: [AvxStoreD; 3],
    rows2: [AvxStoreD; 3],
    rows3: [AvxStoreD; 3],
    rows4: [AvxStoreD; 3],
) -> ([AvxStoreD; 9], [AvxStoreD; 9]) {
    let a0 = transpose_4x4(
        [rows0[0], rows0[1], rows0[2], AvxStoreD::zero()],
        [rows1[0], rows1[1], rows1[2], AvxStoreD::zero()],
    );
    let b0 = transpose_4x4(
        [rows2[0], rows2[1], rows2[2], AvxStoreD::zero()],
        [rows3[0], rows3[1], rows3[2], AvxStoreD::zero()],
    );
    let c0 = transpose_4x4(
        [rows4[0], rows4[1], rows4[2], AvxStoreD::zero()],
        [
            AvxStoreD::zero(),
            AvxStoreD::zero(),
            AvxStoreD::zero(),
            AvxStoreD::zero(),
        ],
    );
    (
        [
            // row 0
            a0.0[0], a0.0[1], a0.0[2], a0.0[3], b0.0[0], b0.0[1], b0.0[2], b0.0[3], c0.0[0],
        ],
        [
            // row 0
            a0.1[0], a0.1[1], a0.1[2], a0.1[3], b0.1[0], b0.1[1], b0.1[2], b0.1[3], c0.1[0],
        ],
    )
}

pub(crate) struct AvxButterfly27d {
    direction: FftDirection,
    bf9: ColumnButterfly9d,
    twiddles: [AvxStoreD; 10],
}

impl AvxButterfly27d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_f64(9, 3, fft_direction, 27),
            bf9: ColumnButterfly9d::new(fft_direction),
            direction: fft_direction,
        }
    }
}

impl AvxButterfly27d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows0: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
        let mut rows1: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
        let mut rows2: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
        let mut rows3: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
        let mut rows4: [AvxStoreD; 3] = [AvxStoreD::zero(); 3];
        // columns
        for i in 0..3 {
            rows0[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 9..));
            rows1[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 9 + 2..));
            rows2[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 9 + 4..));
            rows3[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 9 + 6..));
            rows4[i] = AvxStoreD::from_complex(chunk.index(i * 9 + 8));
        }

        rows0 = self.bf9.bf3.exec(rows0);
        rows1 = self.bf9.bf3.exec(rows1);
        rows2 = self.bf9.bf3.exec(rows2);
        rows3 = self.bf9.bf3.exec(rows3);
        rows4 = self.bf9.bf3.exec(rows4);

        for i in 1..3 {
            rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1]);
            rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 2]);
            rows2[i] = AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 4]);
            rows3[i] = AvxStoreD::mul_by_complex(rows3[i], self.twiddles[i - 1 + 6]);
            rows4[i] = AvxStoreD::mul_by_complex(rows4[i], self.twiddles[i - 1 + 8]);
        }

        let t = transpose_f64x2_9x3(rows0, rows1, rows2, rows3, rows4);

        // rows
        let left = self.bf9.exec(t.0);
        let right = self.bf9.exec(t.1);

        for i in 0..9 {
            left[i].write(chunk.slice_from_mut(i * 3..));
        }
        for i in 0..9 {
            right[i].write_lo(chunk.slice_from_mut(i * 3 + 2..));
        }
    }
}

boring_avx_butterfly!(AvxButterfly27d, f64, 27);

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly27d, f64, AvxButterfly27d, 27, 1e-7);
    test_oof_avx_butterfly!(test_oof_avx_butterfly27_f64, f64, AvxButterfly27d, 27, 1e-7);
}
