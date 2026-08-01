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
use crate::avx::mixed::{AvxMaskD, AvxStoreD};
use crate::avx::transpose::f64x4_4xn::transpose_4x4_f64;
use crate::transpose::TransposeExecutorReal;
use num_complex::Complex;

pub(crate) struct AvxTransposeDReal4x4 {}

impl AvxTransposeDReal4x4 {
    #[target_feature(enable = "avx2")]
    fn transpose_y(
        &self,
        src: &[f64],
        dst: &mut [Complex<f64>],
        y: usize,
        width: usize,
        height: usize,
        rows: usize,
    ) {
        const BLOCK_SIZE: usize = 4;
        debug_assert!((1..=BLOCK_SIZE).contains(&rows));

        let input_stride = width;
        let output_stride = height;
        let src = unsafe { src.get_unchecked(input_stride * y..) };
        let full_load_mask = AvxMaskD::real(BLOCK_SIZE);
        let lo_store_mask = AvxMaskD::complex(rows.min(2));
        let hi_store_mask = (rows > 2).then(|| AvxMaskD::complex(rows - 2));
        let mut x = 0;

        while x < width {
            let columns = (width - x).min(BLOCK_SIZE);
            let load_mask = if columns == BLOCK_SIZE {
                full_load_mask
            } else {
                AvxMaskD::real(columns)
            };
            let block_src = unsafe { src.get_unchecked(x..) };
            let block_dst = unsafe { dst.get_unchecked_mut(y + output_stride * x..) };

            let zbuffer = std::array::from_fn(|row| {
                if row < rows {
                    AvxStoreD::load_partial(
                        unsafe { block_src.get_unchecked(row * input_stride..) },
                        load_mask,
                    )
                } else {
                    AvxStoreD::zero()
                }
            });
            let zbuffer = transpose_4x4_f64(zbuffer);

            for (column, values) in zbuffer.into_iter().take(columns).enumerate() {
                let output = unsafe { block_dst.get_unchecked_mut(output_stride * column..) };
                let [lo, hi] = values.to_complex();
                lo.write_partial(output, lo_store_mask);
                if let Some(hi_store_mask) = hi_store_mask {
                    hi.write_partial(unsafe { output.get_unchecked_mut(2..) }, hi_store_mask);
                }
            }

            x += columns;
        }
    }
}

impl TransposeExecutorReal<f64> for AvxTransposeDReal4x4 {
    fn transpose(&self, input: &[f64], output: &mut [Complex<f64>], width: usize, height: usize) {
        const BLOCK_SIZE: usize = 4;
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

        let transpose = AvxTransposeDReal4x4 {};
        for height in 1..=6 {
            for width in 1..=6 {
                let input: Vec<_> = (0..width * height).map(|x| x as f64 + 0.25).collect();
                let mut output = vec![Complex::new(f64::NAN, f64::NAN); input.len()];
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
