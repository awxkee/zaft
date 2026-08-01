/*
 * // Copyright (c) Radzivon Bartoshyk 01/2026. All rights reserved.
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
use crate::avx::mixed::{AvxMaskF, AvxStoreF};
use crate::avx::util::shuffle;
use crate::transpose::TransposeExecutorReal;
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_8x8_f32(store: [AvxStoreF; 8]) -> [AvxStoreF; 8] {
    let t0 = _mm256_unpacklo_ps(store[0].v, store[1].v);
    let t1 = _mm256_unpackhi_ps(store[0].v, store[1].v);
    let t2 = _mm256_unpacklo_ps(store[2].v, store[3].v);
    let t3 = _mm256_unpackhi_ps(store[2].v, store[3].v);
    let t4 = _mm256_unpacklo_ps(store[4].v, store[5].v);
    let t5 = _mm256_unpackhi_ps(store[4].v, store[5].v);
    let t6 = _mm256_unpacklo_ps(store[6].v, store[7].v);
    let t7 = _mm256_unpackhi_ps(store[6].v, store[7].v);
    let tt0 = _mm256_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(t0, t2);
    let tt1 = _mm256_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(t0, t2);
    let tt2 = _mm256_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(t1, t3);
    let tt3 = _mm256_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(t1, t3);
    let tt4 = _mm256_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(t4, t6);
    let tt5 = _mm256_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(t4, t6);
    let tt6 = _mm256_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(t5, t7);
    let tt7 = _mm256_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(t5, t7);

    [
        AvxStoreF::raw(_mm256_permute2f128_ps::<0x20>(tt0, tt4)),
        AvxStoreF::raw(_mm256_permute2f128_ps::<0x20>(tt1, tt5)),
        AvxStoreF::raw(_mm256_permute2f128_ps::<0x20>(tt2, tt6)),
        AvxStoreF::raw(_mm256_permute2f128_ps::<0x20>(tt3, tt7)),
        AvxStoreF::raw(_mm256_permute2f128_ps::<0x31>(tt0, tt4)),
        AvxStoreF::raw(_mm256_permute2f128_ps::<0x31>(tt1, tt5)),
        AvxStoreF::raw(_mm256_permute2f128_ps::<0x31>(tt2, tt6)),
        AvxStoreF::raw(_mm256_permute2f128_ps::<0x31>(tt3, tt7)),
    ]
}

pub(crate) struct AvxTransposeFReal4x4 {}

impl AvxTransposeFReal4x4 {
    #[target_feature(enable = "avx2")]
    fn transpose_y(
        &self,
        src: &[f32],
        dst: &mut [Complex<f32>],
        y: usize,
        width: usize,
        height: usize,
        rows: usize,
    ) {
        const BLOCK_SIZE: usize = 8;
        debug_assert!((1..=BLOCK_SIZE).contains(&rows));

        let input_stride = width;
        let output_stride = height;
        let src = unsafe { src.get_unchecked(input_stride * y..) };
        let full_load_mask = AvxMaskF::real(BLOCK_SIZE);
        let lo_store_mask = AvxMaskF::complex(rows.min(4));
        let hi_store_mask = (rows > 4).then(|| AvxMaskF::complex(rows - 4));
        let mut x = 0;

        while x < width {
            let columns = (width - x).min(BLOCK_SIZE);
            let load_mask = if columns == BLOCK_SIZE {
                full_load_mask
            } else {
                AvxMaskF::real(columns)
            };
            let block_src = unsafe { src.get_unchecked(x..) };
            let block_dst = unsafe { dst.get_unchecked_mut(y + output_stride * x..) };

            let zbuffer = std::array::from_fn(|row| {
                if row < rows {
                    AvxStoreF::load_partial(
                        unsafe { block_src.get_unchecked(row * input_stride..) },
                        load_mask,
                    )
                } else {
                    AvxStoreF::zero()
                }
            });
            let zbuffer = transpose_8x8_f32(zbuffer);

            for (column, values) in zbuffer.into_iter().take(columns).enumerate() {
                let output = unsafe { block_dst.get_unchecked_mut(output_stride * column..) };
                let [lo, hi] = values.to_complex();
                lo.write_partial(output, lo_store_mask);
                if let Some(hi_store_mask) = hi_store_mask {
                    hi.write_partial(unsafe { output.get_unchecked_mut(4..) }, hi_store_mask);
                }
            }

            x += columns;
        }
    }
}

impl TransposeExecutorReal<f32> for AvxTransposeFReal4x4 {
    fn transpose(&self, input: &[f32], output: &mut [Complex<f32>], width: usize, height: usize) {
        const BLOCK_SIZE: usize = 8;
        let mut y = 0;

        unsafe {
            while y + BLOCK_SIZE <= height {
                self.transpose_y(input, output, y, width, height, BLOCK_SIZE);
                y += BLOCK_SIZE;
            }
            if y < height {
                self.transpose_y(input, output, y, width, height, height - y);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_transpose_handles_every_avx_tail() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let transpose = AvxTransposeFReal4x4 {};
        for height in 1..=10 {
            for width in 1..=10 {
                let input: Vec<_> = (0..width * height).map(|x| x as f32 + 0.25).collect();
                let mut output = vec![Complex::new(f32::NAN, f32::NAN); input.len()];
                transpose.transpose(&input, &mut output, width, height);

                for y in 0..height {
                    for x in 0..width {
                        assert_eq!(
                            output[x * height + y],
                            Complex::new(input[y * width + x], 0.0)
                        );
                    }
                }
            }
        }
    }
}
