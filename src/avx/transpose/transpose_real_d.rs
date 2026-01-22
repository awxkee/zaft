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
use crate::avx::mixed::AvxStoreD;
use crate::transpose::TransposeExecutorReal;
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_4x4_f64(store: [AvxStoreD; 4]) -> [AvxStoreD; 4] {
    let tmp0 = _mm256_shuffle_pd::<0x0>(store[0].v, store[1].v);
    let tmp2 = _mm256_shuffle_pd::<0xF>(store[0].v, store[1].v);
    let tmp1 = _mm256_shuffle_pd::<0x0>(store[2].v, store[3].v);
    let tmp3 = _mm256_shuffle_pd::<0xF>(store[2].v, store[3].v);

    let row0 = _mm256_permute2f128_pd::<0x20>(tmp0, tmp1);
    let row1 = _mm256_permute2f128_pd::<0x20>(tmp2, tmp3);
    let row2 = _mm256_permute2f128_pd::<0x31>(tmp0, tmp1);
    let row3 = _mm256_permute2f128_pd::<0x31>(tmp2, tmp3);

    [
        AvxStoreD::raw(row0),
        AvxStoreD::raw(row1),
        AvxStoreD::raw(row2),
        AvxStoreD::raw(row3),
    ]
}

pub(crate) struct AvxTransposeDReal4x4 {}

impl AvxTransposeDReal4x4 {
    #[target_feature(enable = "avx2")]
    fn transpose_y<const REMAINDER_Y: usize>(
        &self,
        src: &[f64],
        dst: &mut [Complex<f64>],
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

            let mut zbuffer: [AvxStoreD; 4] = std::array::from_fn(|x| {
                if x < REMAINDER_Y {
                    AvxStoreD::load(unsafe { src.get_unchecked(x * input_stride..) })
                } else {
                    AvxStoreD::zero()
                }
            });

            zbuffer = transpose_4x4_f64(zbuffer);

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

                let mut zbuffer: [AvxStoreD; 4] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        AvxStoreD::load1(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        AvxStoreD::zero()
                    }
                });

                zbuffer = transpose_4x4_f64(zbuffer);

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

                let zbuffer: [AvxStoreD; 4] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        AvxStoreD::load2(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        AvxStoreD::zero()
                    }
                });

                let buffer = transpose_4x4_f64(zbuffer);

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

                let zbuffer: [AvxStoreD; 4] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        AvxStoreD::load3(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        AvxStoreD::zero()
                    }
                });

                let buffer = transpose_4x4_f64(zbuffer);

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

impl TransposeExecutorReal<f64> for AvxTransposeDReal4x4 {
    fn transpose(&self, input: &[f64], output: &mut [Complex<f64>], width: usize, height: usize) {
        const BLOCK_SIZE_Y: usize = 4;
        let mut y = 0usize;

        unsafe {
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
}
