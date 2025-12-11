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
use crate::avx::butterflies::shared::{gen_butterfly_twiddles_f32, gen_butterfly_twiddles_f64};
use crate::avx::mixed::{
    AvxStoreD, AvxStoreF, ColumnButterfly4d, ColumnButterfly4f, ColumnButterfly5d,
    ColumnButterfly5f,
};
use crate::avx::transpose::{avx_transpose_f32x2_4x4_impl, transpose_f64x2_2x2};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;

pub(crate) struct AvxButterfly20d {
    direction: FftDirection,
    bf5: ColumnButterfly5d,
    bf4: ColumnButterfly4d,
    twiddles: [AvxStoreD; 9],
}

impl AvxButterfly20d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf5: ColumnButterfly5d::new(fft_direction),
            bf4: ColumnButterfly4d::new(fft_direction),
            twiddles: gen_butterfly_twiddles_f64(5, 4, fft_direction, 20),
        }
    }
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) fn transpose_f64x2_5x4(
    rows0: [AvxStoreD; 4],
    rows1: [AvxStoreD; 4],
    rows2: [AvxStoreD; 4],
) -> ([AvxStoreD; 5], [AvxStoreD; 5]) {
    let a0 = transpose_f64x2_2x2(rows0[0].v, rows0[1].v);
    let d0 = transpose_f64x2_2x2(rows0[2].v, rows0[3].v);

    let b0 = transpose_f64x2_2x2(rows1[0].v, rows1[1].v);
    let e0 = transpose_f64x2_2x2(rows1[2].v, rows1[3].v);

    let c0 = transpose_f64x2_2x2(rows2[0].v, rows2[1].v);
    let f0 = transpose_f64x2_2x2(rows2[2].v, rows2[3].v);
    (
        [
            AvxStoreD::raw(a0.0),
            AvxStoreD::raw(a0.1),
            AvxStoreD::raw(b0.0),
            AvxStoreD::raw(b0.1),
            AvxStoreD::raw(c0.0),
        ],
        [
            AvxStoreD::raw(d0.0),
            AvxStoreD::raw(d0.1),
            AvxStoreD::raw(e0.0),
            AvxStoreD::raw(e0.1),
            AvxStoreD::raw(f0.0),
        ],
    )
}

impl AvxButterfly20d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(20) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }
        unsafe {
            let mut rows0: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
            let mut rows1: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];
            let mut rows2: [AvxStoreD; 4] = [AvxStoreD::zero(); 4];

            for chunk in in_place.chunks_exact_mut(20) {
                // columns
                for i in 0..4 {
                    rows0[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 5..));
                    rows1[i] = AvxStoreD::from_complex_ref(chunk.get_unchecked(i * 5 + 2..));
                    rows2[i] = AvxStoreD::from_complex(chunk.get_unchecked(i * 5 + 4));
                }

                rows0 = self.bf4.exec(rows0);
                rows1 = self.bf4.exec(rows1);
                rows2 = self.bf4.exec(rows2);

                for i in 1..4 {
                    rows0[i] = AvxStoreD::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = AvxStoreD::mul_by_complex(rows1[i], self.twiddles[i - 1 + 3]);
                    rows2[i] = AvxStoreD::mul_by_complex(rows2[i], self.twiddles[i - 1 + 6]);
                }

                let transposed = transpose_f64x2_5x4(rows0, rows1, rows2);

                // rows

                let q0 = self.bf5.exec(transposed.0);
                let q1 = self.bf5.exec(transposed.1);

                for i in 0..5 {
                    q0[i].write(chunk.get_unchecked_mut(i * 4..));
                    q1[i].write(chunk.get_unchecked_mut(i * 4 + 2..));
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly20d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        20
    }
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) fn transpose_f32x2_5x4(rows0: [AvxStoreF; 4], rows1: [AvxStoreF; 4]) -> [AvxStoreF; 5] {
    let a0 = avx_transpose_f32x2_4x4_impl(rows0[0].v, rows0[1].v, rows0[2].v, rows0[3].v);
    let b0 = avx_transpose_f32x2_4x4_impl(rows1[0].v, rows1[1].v, rows1[2].v, rows1[3].v);

    [
        AvxStoreF::raw(a0.0),
        AvxStoreF::raw(a0.1),
        AvxStoreF::raw(a0.2),
        AvxStoreF::raw(a0.3),
        AvxStoreF::raw(b0.0),
    ]
}

pub(crate) struct AvxButterfly20f {
    direction: FftDirection,
    bf5: ColumnButterfly5f,
    bf4: ColumnButterfly4f,
    twiddles: [AvxStoreF; 6],
}

impl AvxButterfly20f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf5: ColumnButterfly5f::new(fft_direction),
            bf4: ColumnButterfly4f::new(fft_direction),
            twiddles: gen_butterfly_twiddles_f32(5, 4, fft_direction, 20),
        }
    }
}

impl AvxButterfly20f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(20) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }
        unsafe {
            let mut rows0: [AvxStoreF; 4] = [AvxStoreF::zero(); 4];
            let mut rows1: [AvxStoreF; 4] = [AvxStoreF::zero(); 4];

            for chunk in in_place.chunks_exact_mut(20) {
                // columns
                for i in 0..4 {
                    rows0[i] = AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 5..));
                    rows1[i] = AvxStoreF::from_complex(chunk.get_unchecked(i * 5 + 4));
                }

                rows0 = self.bf4.exec(rows0);
                rows1 = self.bf4.exec(rows1);

                for i in 1..4 {
                    rows0[i] = AvxStoreF::mul_by_complex(rows0[i], self.twiddles[i - 1]);
                    rows1[i] = AvxStoreF::mul_by_complex(rows1[i], self.twiddles[i - 1 + 3]);
                }

                let transposed = transpose_f32x2_5x4(rows0, rows1);

                // rows

                let q0 = self.bf5.exec(transposed);

                q0[0].write(chunk);
                q0[1].write(chunk.get_unchecked_mut(4..));
                q0[2].write(chunk.get_unchecked_mut(8..));
                q0[3].write(chunk.get_unchecked_mut(12..));
                q0[4].write(chunk.get_unchecked_mut(16..));
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly20f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        20
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly20, f32, AvxButterfly20f, 20, 1e-5);
    test_avx_butterfly!(test_avx_butterfly20_f64, f64, AvxButterfly20d, 20, 1e-7);
}
