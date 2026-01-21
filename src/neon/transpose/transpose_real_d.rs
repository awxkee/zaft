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
use crate::neon::mixed::NeonStoreD;
use crate::transpose::TransposeExecutorReal;
use num_complex::Complex;
use std::arch::aarch64::*;

#[inline(always)]
fn transpose_2x2_f64(store: [NeonStoreD; 2]) -> [NeonStoreD; 2] {
    unsafe {
        let vl = vcombine_f64(vget_low_f64(store[0].v), vget_low_f64(store[1].v));
        let vh = vcombine_f64(vget_high_f64(store[0].v), vget_high_f64(store[1].v));
        [NeonStoreD::raw(vl), NeonStoreD::raw(vh)]
    }
}

pub(crate) struct NeonTransposeDReal4x4 {}

impl NeonTransposeDReal4x4 {
    fn transpose_y<const REMAINDER_Y: usize>(
        &self,
        src: &[f64],
        dst: &mut [Complex<f64>],
        y: usize,
        width: usize,
        height: usize,
    ) {
        const BLOCK_SIZE_X: usize = 2;
        let input_y = y;

        let input_stride = width;
        let output_stride = height;

        let src = unsafe { src.get_unchecked(input_stride * input_y..) };

        let mut x = 0usize;

        while x + BLOCK_SIZE_X <= width {
            let output_x = x;

            let src = unsafe { src.get_unchecked(x..) };
            let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

            let mut zbuffer: [NeonStoreD; 2] = std::array::from_fn(|x| {
                if x < REMAINDER_Y {
                    NeonStoreD::load(unsafe { src.get_unchecked(x * input_stride..) })
                } else {
                    NeonStoreD::default()
                }
            });

            zbuffer = transpose_2x2_f64(zbuffer);

            for i in 0..2 {
                let [v0, v1] = zbuffer[i].to_complex();
                if REMAINDER_Y == 2 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    v1.write(unsafe { dst.get_unchecked_mut(output_stride * i + 1..) });
                } else if REMAINDER_Y == 1 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                }
            }

            x += BLOCK_SIZE_X;
        }

        let rem = width - x;

        if rem == 1 {
            let output_x = x;

            let src = unsafe { src.get_unchecked(x..) };
            let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

            let mut zbuffer: [NeonStoreD; 2] = std::array::from_fn(|x| {
                if x < REMAINDER_Y {
                    NeonStoreD::load1(unsafe { src.get_unchecked(x * input_stride..) })
                } else {
                    NeonStoreD::default()
                }
            });

            zbuffer = transpose_2x2_f64(zbuffer);

            let [v0, v1] = zbuffer[0].to_complex();
            if REMAINDER_Y == 2 {
                v0.write(dst);
                unsafe {
                    v1.write(dst.get_unchecked_mut(1..));
                }
            } else if REMAINDER_Y == 1 {
                v0.write(dst);
            }
        }
    }
}

impl TransposeExecutorReal<f64> for NeonTransposeDReal4x4 {
    fn transpose(&self, input: &[f64], output: &mut [Complex<f64>], width: usize, height: usize) {
        const BLOCK_SIZE_Y: usize = 2;
        let mut y = 0usize;

        while y + BLOCK_SIZE_Y <= height {
            self.transpose_y::<2>(input, output, y, width, height);
            y += BLOCK_SIZE_Y;
        }

        let rem_y = height - y;
        if rem_y > 0 && rem_y == 1 {
            self.transpose_y::<1>(input, output, y, width, height)
        }
    }
}
