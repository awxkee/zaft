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
use crate::avx::mixed::AvxStoreF;
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
    const FLAG_1: i32 = shuffle(1, 0, 1, 0);
    let tt0 = _mm256_shuffle_ps::<FLAG_1>(t0, t2);
    const FLAG_2: i32 = shuffle(3, 2, 3, 2);
    let tt1 = _mm256_shuffle_ps::<FLAG_2>(t0, t2);
    const FLAG_3: i32 = shuffle(1, 0, 1, 0);
    let tt2 = _mm256_shuffle_ps::<FLAG_3>(t1, t3);
    const FLAG_4: i32 = shuffle(3, 2, 3, 2);
    let tt3 = _mm256_shuffle_ps::<FLAG_4>(t1, t3);
    const FLAG_5: i32 = shuffle(1, 0, 1, 0);
    let tt4 = _mm256_shuffle_ps::<FLAG_5>(t4, t6);
    const FLAG_6: i32 = shuffle(3, 2, 3, 2);
    let tt5 = _mm256_shuffle_ps::<FLAG_6>(t4, t6);
    const FLAG_7: i32 = shuffle(1, 0, 1, 0);
    let tt6 = _mm256_shuffle_ps::<FLAG_7>(t5, t7);
    const FLAG_8: i32 = shuffle(3, 2, 3, 2);
    let tt7 = _mm256_shuffle_ps::<FLAG_8>(t5, t7);
    let r0 = _mm256_permute2f128_ps::<0x20>(tt0, tt4);
    let r1 = _mm256_permute2f128_ps::<0x20>(tt1, tt5);
    let r2 = _mm256_permute2f128_ps::<0x20>(tt2, tt6);
    let r3 = _mm256_permute2f128_ps::<0x20>(tt3, tt7);
    let r4 = _mm256_permute2f128_ps::<0x31>(tt0, tt4);
    let r5 = _mm256_permute2f128_ps::<0x31>(tt1, tt5);
    let r6 = _mm256_permute2f128_ps::<0x31>(tt2, tt6);
    let r7 = _mm256_permute2f128_ps::<0x31>(tt3, tt7);

    [
        AvxStoreF::raw(r0),
        AvxStoreF::raw(r1),
        AvxStoreF::raw(r2),
        AvxStoreF::raw(r3),
        AvxStoreF::raw(r4),
        AvxStoreF::raw(r5),
        AvxStoreF::raw(r6),
        AvxStoreF::raw(r7),
    ]
}

pub(crate) struct AvxTransposeFReal4x4 {}

impl AvxTransposeFReal4x4 {
    #[target_feature(enable = "avx2")]
    fn transpose_y<const REMAINDER_Y: usize>(
        &self,
        src: &[f32],
        dst: &mut [Complex<f32>],
        y: usize,
        width: usize,
        height: usize,
    ) {
        const BLOCK_SIZE_X: usize = 8;
        let input_y = y;

        let input_stride = width;
        let output_stride = height;

        let src = unsafe { src.get_unchecked(input_stride * input_y..) };

        let mut x = 0usize;

        while x + BLOCK_SIZE_X <= width {
            let output_x = x;

            let src = unsafe { src.get_unchecked(x..) };
            let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

            let mut zbuffer: [AvxStoreF; 8] = std::array::from_fn(|x| {
                if x < REMAINDER_Y {
                    AvxStoreF::load(unsafe { src.get_unchecked(x * input_stride..) })
                } else {
                    AvxStoreF::zero()
                }
            });

            zbuffer = transpose_8x8_f32(zbuffer);

            for i in 0..8 {
                let [v0, v1] = zbuffer[i].to_complex();
                if REMAINDER_Y == 8 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    v1.write(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                } else if REMAINDER_Y == 7 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    v1.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                } else if REMAINDER_Y == 6 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    v1.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                } else if REMAINDER_Y == 5 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    v1.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                } else if REMAINDER_Y == 4 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                } else if REMAINDER_Y == 3 {
                    v0.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                } else if REMAINDER_Y == 2 {
                    v0.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                } else if REMAINDER_Y == 1 {
                    v0.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i..) });
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

                let mut zbuffer: [AvxStoreF; 8] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        AvxStoreF::load1(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        AvxStoreF::zero()
                    }
                });

                zbuffer = transpose_8x8_f32(zbuffer);

                let [v0, v1] = zbuffer[0].to_complex();
                let i = 0;
                if REMAINDER_Y == 8 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    v1.write(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                } else if REMAINDER_Y == 7 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    v1.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                } else if REMAINDER_Y == 6 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    v1.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                } else if REMAINDER_Y == 5 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    v1.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                } else if REMAINDER_Y == 4 {
                    v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                } else if REMAINDER_Y == 3 {
                    v0.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                } else if REMAINDER_Y == 2 {
                    v0.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                } else if REMAINDER_Y == 1 {
                    v0.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                }
            }
            2 => {
                let output_x = x;

                let src = unsafe { src.get_unchecked(x..) };
                let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

                let zbuffer: [AvxStoreF; 8] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        AvxStoreF::load2(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        AvxStoreF::zero()
                    }
                });

                let buffer = transpose_8x8_f32(zbuffer);

                for i in 0..2 {
                    let [v0, v1] = buffer[i].to_complex();
                    if REMAINDER_Y == 8 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 7 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 6 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 5 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 4 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 3 {
                        v0.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 2 {
                        v0.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 1 {
                        v0.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    }
                }
            }
            3 => {
                let output_x = x;

                let src = unsafe { src.get_unchecked(x..) };
                let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

                let zbuffer: [AvxStoreF; 8] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        AvxStoreF::load3(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        AvxStoreF::zero()
                    }
                });

                let buffer = transpose_8x8_f32(zbuffer);

                for i in 0..3 {
                    let [v0, v1] = buffer[i].to_complex();
                    if REMAINDER_Y == 8 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 7 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 6 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 5 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 4 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 3 {
                        v0.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 2 {
                        v0.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 1 {
                        v0.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    }
                }
            }
            4 => {
                let output_x = x;

                let src = unsafe { src.get_unchecked(x..) };
                let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

                let zbuffer: [AvxStoreF; 8] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        AvxStoreF::load4(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        AvxStoreF::zero()
                    }
                });

                let buffer = transpose_8x8_f32(zbuffer);

                for i in 0..4 {
                    let [v0, v1] = buffer[i].to_complex();
                    if REMAINDER_Y == 8 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 7 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 6 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 5 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 4 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 3 {
                        v0.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 2 {
                        v0.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 1 {
                        v0.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    }
                }
            }
            5 => {
                let output_x = x;

                let src = unsafe { src.get_unchecked(x..) };
                let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

                let zbuffer: [AvxStoreF; 8] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        AvxStoreF::load5(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        AvxStoreF::zero()
                    }
                });

                let buffer = transpose_8x8_f32(zbuffer);

                for i in 0..5 {
                    let [v0, v1] = buffer[i].to_complex();
                    if REMAINDER_Y == 8 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 7 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 6 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 5 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 4 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 3 {
                        v0.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 2 {
                        v0.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 1 {
                        v0.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    }
                }
            }
            6 => {
                let output_x = x;

                let src = unsafe { src.get_unchecked(x..) };
                let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

                let zbuffer: [AvxStoreF; 8] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        AvxStoreF::load6(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        AvxStoreF::zero()
                    }
                });

                let buffer = transpose_8x8_f32(zbuffer);

                for i in 0..6 {
                    let [v0, v1] = buffer[i].to_complex();
                    if REMAINDER_Y == 8 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 7 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 6 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 5 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 4 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 3 {
                        v0.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 2 {
                        v0.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 1 {
                        v0.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    }
                }
            }
            7 => {
                let output_x = x;

                let src = unsafe { src.get_unchecked(x..) };
                let dst = unsafe { dst.get_unchecked_mut(y + output_stride * output_x..) };

                let zbuffer: [AvxStoreF; 8] = std::array::from_fn(|x| {
                    if x < REMAINDER_Y {
                        AvxStoreF::load7(unsafe { src.get_unchecked(x * input_stride..) })
                    } else {
                        AvxStoreF::zero()
                    }
                });

                let buffer = transpose_8x8_f32(zbuffer);

                for i in 0..7 {
                    let [v0, v1] = buffer[i].to_complex();
                    if REMAINDER_Y == 8 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 7 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 6 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 5 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                        v1.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i + 4..) });
                    } else if REMAINDER_Y == 4 {
                        v0.write(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 3 {
                        v0.write_lo3(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 2 {
                        v0.write_lo2(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    } else if REMAINDER_Y == 1 {
                        v0.write_lo1(unsafe { dst.get_unchecked_mut(output_stride * i..) });
                    }
                }
            }
            _ => {}
        }
    }
}

impl TransposeExecutorReal<f32> for AvxTransposeFReal4x4 {
    fn transpose(&self, input: &[f32], output: &mut [Complex<f32>], width: usize, height: usize) {
        const BLOCK_SIZE_Y: usize = 8;
        let mut y = 0usize;

        unsafe {
            while y + BLOCK_SIZE_Y <= height {
                self.transpose_y::<8>(input, output, y, width, height);
                y += BLOCK_SIZE_Y;
            }

            let rem_y = height - y;
            if rem_y > 0 {
                match rem_y {
                    1 => self.transpose_y::<1>(input, output, y, width, height),
                    2 => self.transpose_y::<2>(input, output, y, width, height),
                    3 => self.transpose_y::<3>(input, output, y, width, height),
                    4 => self.transpose_y::<4>(input, output, y, width, height),
                    5 => self.transpose_y::<5>(input, output, y, width, height),
                    6 => self.transpose_y::<6>(input, output, y, width, height),
                    7 => self.transpose_y::<7>(input, output, y, width, height),
                    _ => {}
                }
            }
        }
    }
}
