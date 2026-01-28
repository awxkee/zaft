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

use crate::avx::avx_transpose_f64x2_6x6_impl;
use crate::avx::butterflies::shared::boring_avx_butterfly;
use crate::avx::mixed::{AvxStoreD, ColumnButterfly6d};
use crate::avx::util::_mm256_fcmul_pd;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly36d {
    direction: FftDirection,
    twiddles: [Complex<f64>; 30],
    bf6_column: ColumnButterfly6d,
}

impl AvxButterfly36d {
    pub fn new(direction: FftDirection) -> Self {
        let mut twiddles = [Complex::<f64>::default(); 30];
        const ENTRIES: usize = 2;
        let num_twiddle_columns = 3;
        let len = 36;
        let mut k = 0usize;
        for x in 0..num_twiddle_columns {
            for y in 1..6 {
                for i in 0..ENTRIES {
                    twiddles[k] = compute_twiddle(y * (x * ENTRIES + i), len, direction);
                    k += 1;
                }
            }
        }
        AvxButterfly36d {
            direction,
            twiddles,
            bf6_column: unsafe { ColumnButterfly6d::new(direction) },
        }
    }
}

boring_avx_butterfly!(AvxButterfly36d, f64, 36);

impl AvxButterfly36d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn run<S: BidirectionalStore<Complex<f64>>>(&self, chunk: &mut S) {
        let mut rows0 = [AvxStoreD::zero(); 6];
        let mut rows1 = [AvxStoreD::zero(); 6];
        let mut rows2 = [AvxStoreD::zero(); 6];
        unsafe {
            // Mixed Radix 6x6
            for i in 0..6 {
                rows0[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 6..));
            }
            rows0 = self.bf6_column.exec(rows0);
            for i in 1..6 {
                rows0[i] = AvxStoreD::raw(_mm256_fcmul_pd(
                    rows0[i].v,
                    _mm256_loadu_pd(self.twiddles[i * 2 - 2..].as_ptr().cast()),
                ));
            }

            for i in 0..6 {
                rows1[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 6 + 2..));
            }
            rows1 = self.bf6_column.exec(rows1);
            for i in 1..6 {
                rows1[i] = AvxStoreD::raw(_mm256_fcmul_pd(
                    rows1[i].v,
                    _mm256_loadu_pd(self.twiddles[i * 2 + 4 * 2..].as_ptr().cast()),
                ));
            }

            for i in 0..6 {
                rows2[i] = AvxStoreD::from_complex_ref(chunk.slice_from(i * 6 + 4..));
            }
            rows2 = self.bf6_column.exec(rows2);
            for i in 1..6 {
                rows2[i] = AvxStoreD::raw(_mm256_fcmul_pd(
                    rows2[i].v,
                    _mm256_loadu_pd(self.twiddles[i * 2 + 9 * 2..].as_ptr().cast()),
                ));
            }

            let (transposed0, transposed1, transposed2) =
                avx_transpose_f64x2_6x6_impl(rows0, rows1, rows2);

            let output0 = self.bf6_column.exec(transposed0);
            for r in 0..3 {
                _mm256_storeu_pd(
                    chunk.slice_from_mut(12 * r..).as_mut_ptr().cast(),
                    output0[r * 2].v,
                );
                _mm256_storeu_pd(
                    chunk.slice_from_mut(12 * r + 6..).as_mut_ptr().cast(),
                    output0[r * 2 + 1].v,
                );
            }

            let output1 = self.bf6_column.exec(transposed1);
            for r in 0..3 {
                _mm256_storeu_pd(
                    chunk.slice_from_mut(12 * r + 2..).as_mut_ptr().cast(),
                    output1[r * 2].v,
                );
                _mm256_storeu_pd(
                    chunk.slice_from_mut(12 * r + 8..).as_mut_ptr().cast(),
                    output1[r * 2 + 1].v,
                );
            }

            let output2 = self.bf6_column.exec(transposed2);
            for r in 0..3 {
                _mm256_storeu_pd(
                    chunk.slice_from_mut(12 * r + 4..).as_mut_ptr().cast(),
                    output2[r * 2].v,
                );
                _mm256_storeu_pd(
                    chunk.slice_from_mut(12 * r + 10..).as_mut_ptr().cast(),
                    output2[r * 2 + 1].v,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly36, f64, AvxButterfly36d, 36, 1e-4);
    test_oof_avx_butterfly!(test_avx_neon_butterfly36, f64, AvxButterfly36d, 36, 1e-4);
}
