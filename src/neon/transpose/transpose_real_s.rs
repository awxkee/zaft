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
use crate::neon::mixed::NeonStoreF;
use crate::transpose::TransposeExecutorReal;
use num_complex::Complex;
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) fn vtrnq_f64_to_f32(a0: float32x4_t, a1: float32x4_t) -> float32x4x2_t {
    unsafe {
        let b0 = vreinterpretq_f32_f64(vtrn1q_f64(
            vreinterpretq_f64_f32(a0),
            vreinterpretq_f64_f32(a1),
        ));
        let b1 = vreinterpretq_f32_f64(vtrn2q_f64(
            vreinterpretq_f64_f32(a0),
            vreinterpretq_f64_f32(a1),
        ));
        float32x4x2_t(b0, b1)
    }
}

#[inline(always)]
pub(crate) fn transpose_4x4(v0: float32x4x4_t) -> float32x4x4_t {
    unsafe {
        // Swap 32 bit elements. Goes from:
        // a0: 00 01 02 03
        // a1: 10 11 12 13
        // a2: 20 21 22 23
        // a3: 30 31 32 33
        // to:
        // b0.0: 00 10 02 12
        // b0.1: 01 11 03 13
        // b1.0: 20 30 22 32
        // b1.1: 21 31 23 33

        let b0 = vtrnq_f32(v0.0, v0.1);
        let b1 = vtrnq_f32(v0.2, v0.3);

        // Swap 64 bit elements resulting in:
        // c0.0: 00 10 20 30
        // c0.1: 02 12 22 32
        // c1.0: 01 11 21 31
        // c1.1: 03 13 23 33

        let c0 = vtrnq_f64_to_f32(b0.0, b1.0);
        let c1 = vtrnq_f64_to_f32(b0.1, b1.1);

        float32x4x4_t(c0.0, c1.0, c0.1, c1.1)
    }
}

#[inline(always)]
fn transpose_4x4_f32(store: [NeonStoreF; 4]) -> [NeonStoreF; 4] {
    let q = transpose_4x4(float32x4x4_t(
        store[0].v, store[1].v, store[2].v, store[3].v,
    ));
    [
        NeonStoreF::raw(q.0),
        NeonStoreF::raw(q.1),
        NeonStoreF::raw(q.2),
        NeonStoreF::raw(q.3),
    ]
}

pub(crate) struct NeonTransposeReal4x4 {}

impl NeonTransposeReal4x4 {
    fn transpose_y<const REMAINDER_Y: usize>(
        &self,
        src: &[f32],
        dst: &mut [Complex<f32>],
        y: usize,
        width: usize,
        height: usize,
    ) {
        const BLOCK_SIZE_X: usize = 4;
        let input_y = y;

        let input_stride = width;
        let output_stride = height;

        let src = unsafe { src.get_unchecked(input_stride * input_y..) };

        let mut x = 0usize;

        while x + BLOCK_SIZE_X <= width {
            let output_x = x;

            let src = unsafe { src.get_unchecked(x..) };
            let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

            let mut zbuffer: [NeonStoreF; 4] = std::array::from_fn(|x| {
                if x < REMAINDER_Y {
                    NeonStoreF::load(unsafe { src.get_unchecked(x * input_stride..) })
                } else {
                    NeonStoreF::default()
                }
            });

            zbuffer = transpose_4x4_f32(zbuffer);

            for i in 0..4 {
                let [v0, v1] = zbuffer[i].to_complex();
                if REMAINDER_Y == 4 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    v1.write(unsafe { dst.get_unchecked_mut(output_stride * i + 2..) });
                } else if REMAINDER_Y == 3 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    v1.write_lo(unsafe { dst.get_unchecked_mut(output_stride * i + 2..) });
                } else if REMAINDER_Y == 2 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                } else if REMAINDER_Y == 1 {
                    v0.write_lo(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                }
            }

            x += BLOCK_SIZE_X;
        }

        let rem = width - x;

        match rem {
            1 => {
                let output_x = x;

                let src = unsafe { src.get_unchecked(x..) };
                let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

                let mut zbuffer: [NeonStoreF; 4] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        NeonStoreF::load1(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        NeonStoreF::default()
                    }
                });

                zbuffer = transpose_4x4_f32(zbuffer);

                let [v0, v1] = zbuffer[0].to_complex();
                if REMAINDER_Y == 4 {
                    v0.write(dst);
                    unsafe {
                        v1.write(dst.get_unchecked_mut(2..));
                    }
                } else if REMAINDER_Y == 3 {
                    v0.write(dst);
                    unsafe {
                        v1.write_lo(dst.get_unchecked_mut(2..));
                    }
                } else if REMAINDER_Y == 2 {
                    v0.write(dst);
                } else if REMAINDER_Y == 1 {
                    v0.write_lo(dst);
                }
            }
            2 => {
                let output_x = x;

                let src = unsafe { src.get_unchecked(x..) };
                let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

                let zbuffer: [NeonStoreF; 4] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        NeonStoreF::load2(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        NeonStoreF::default()
                    }
                });

                let buffer = transpose_4x4_f32(zbuffer);

                for i in 0..2 {
                    let [v0, v1] = buffer[i].to_complex();
                    if REMAINDER_Y == 4 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write(unsafe { dst.get_unchecked_mut(output_stride * i + 2..) });
                    } else if REMAINDER_Y == 3 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo(unsafe { dst.get_unchecked_mut(output_stride * i + 2..) });
                    } else if REMAINDER_Y == 2 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 1 {
                        v0.write_lo(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    }
                }
            }
            3 => {
                let output_x = x;

                let src = unsafe { src.get_unchecked(x..) };
                let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

                let zbuffer: [NeonStoreF; 4] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        NeonStoreF::load3(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        NeonStoreF::default()
                    }
                });

                let buffer = transpose_4x4_f32(zbuffer);

                for i in 0..3 {
                    let [v0, v1] = buffer[i].to_complex();
                    if REMAINDER_Y == 4 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write(unsafe { dst.get_unchecked_mut(output_stride * i + 2..) });
                    } else if REMAINDER_Y == 3 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo(unsafe { dst.get_unchecked_mut(output_stride * i + 2..) });
                    } else if REMAINDER_Y == 2 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 1 {
                        v0.write_lo(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    }
                }
            }
            _ => {}
        }
    }
}

impl TransposeExecutorReal<f32> for NeonTransposeReal4x4 {
    fn transpose(&self, input: &[f32], output: &mut [Complex<f32>], width: usize, height: usize) {
        const BLOCK_SIZE_Y: usize = 4;
        let mut y = 0usize;

        while y + BLOCK_SIZE_Y <= height {
            self.transpose_y::<4>(input, output, y, width, height);
            y += BLOCK_SIZE_Y;
        }

        let rem_y = height - y;
        if rem_y > 0 {
            match rem_y {
                1 => self.transpose_y::<1>(input, output, y, width, height),
                2 => self.transpose_y::<2>(input, output, y, width, height),
                3 => self.transpose_y::<3>(input, output, y, width, height),
                _ => {}
            }
        }
    }
}
