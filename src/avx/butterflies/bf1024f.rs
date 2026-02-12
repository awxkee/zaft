/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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
use crate::avx::butterflies::bf8::transpose_f32x2_4x2;
use crate::avx::butterflies::shared::{boring_avx_butterfly, gen_butterfly_twiddles_f32};
use crate::avx::mixed::{AvxStoreF, ColumnButterfly8f};
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;

pub(crate) struct AvxButterfly1024f {
    direction: FftDirection,
    bf8: ColumnButterfly8f,
    twiddles: [AvxStoreF; 249],
    twiddles32: [AvxStoreF; 6],
}

impl AvxButterfly1024f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                direction: fft_direction,
                twiddles: gen_butterfly_twiddles_f32(32, 32, fft_direction, 1024),
                twiddles32: [
                    AvxStoreF::set_complex(compute_twiddle::<f32>(1, 32, fft_direction)),
                    AvxStoreF::set_complex(compute_twiddle::<f32>(2, 32, fft_direction)),
                    AvxStoreF::set_complex(compute_twiddle::<f32>(3, 32, fft_direction)),
                    AvxStoreF::set_complex(compute_twiddle::<f32>(5, 32, fft_direction)),
                    AvxStoreF::set_complex(compute_twiddle::<f32>(6, 32, fft_direction)),
                    AvxStoreF::set_complex(compute_twiddle::<f32>(7, 32, fft_direction)),
                ],
                bf8: ColumnButterfly8f::new(fft_direction),
            }
        }
    }
}

impl AvxButterfly1024f {
    #[inline(always)]
    fn exec_bf32_2(&self, src: &mut [MaybeUninit<Complex<f32>>], dst: &mut [Complex<f32>]) {
        unsafe {
            for k in 0..8 {
                macro_rules! load {
                    ($src: expr, $k: expr, $idx: expr) => {{ AvxStoreF::from_complex_refu($src.get_unchecked($k * 4 + $idx * 32..)) }};
                }

                macro_rules! store {
                    ($v: expr, $idx: expr, $dst: expr, $k: expr) => {{ $v.write($dst.get_unchecked_mut($k * 4 + $idx * 32..)) }};
                }

                let input1 = [
                    load!(src, k, 1),
                    load!(src, k, 9),
                    load!(src, k, 17),
                    load!(src, k, 25),
                ];
                let mut mid1 = self.bf8.bf4.exec(input1);

                mid1[1] = AvxStoreF::mul_by_complex(mid1[1], self.twiddles32[0]);
                mid1[2] = AvxStoreF::mul_by_complex(mid1[2], self.twiddles32[1]);
                mid1[3] = AvxStoreF::mul_by_complex(mid1[3], self.twiddles32[2]);

                let input2 = [
                    load!(src, k, 2),
                    load!(src, k, 10),
                    load!(src, k, 18),
                    load!(src, k, 26),
                ];
                let mut mid2 = self.bf8.bf4.exec(input2);

                mid2[1] = AvxStoreF::mul_by_complex(mid2[1], self.twiddles32[1]);
                mid2[2] = self.bf8.rotate1(mid2[2]);
                mid2[3] = AvxStoreF::mul_by_complex(mid2[3], self.twiddles32[4]);

                let input3 = [
                    load!(src, k, 3),
                    load!(src, k, 11),
                    load!(src, k, 19),
                    load!(src, k, 27),
                ];
                let mut mid3 = self.bf8.bf4.exec(input3);

                mid3[1] = AvxStoreF::mul_by_complex(mid3[1], self.twiddles32[2]);
                mid3[2] = AvxStoreF::mul_by_complex(mid3[2], self.twiddles32[4]);
                mid3[3] = AvxStoreF::mul_by_complex(mid3[3], self.bf8.rotate(self.twiddles32[0]));

                let input4 = [
                    load!(src, k, 4),
                    load!(src, k, 12),
                    load!(src, k, 20),
                    load!(src, k, 28),
                ];
                let mut mid4 = self.bf8.bf4.exec(input4);

                mid4[1] = self.bf8.rotate1(mid4[1]);
                mid4[2] = self.bf8.rotate(mid4[2]);
                mid4[3] = self.bf8.rotate3(mid4[3]);

                let input5 = [
                    load!(src, k, 5),
                    load!(src, k, 13),
                    load!(src, k, 21),
                    load!(src, k, 29),
                ];
                let mut mid5 = self.bf8.bf4.exec(input5);

                mid5[1] = AvxStoreF::mul_by_complex(mid5[1], self.twiddles32[3]);
                mid5[2] = AvxStoreF::mul_by_complex(mid5[2], self.bf8.rotate(self.twiddles32[1]));
                mid5[3] = AvxStoreF::mul_by_complex(mid5[3], self.bf8.rotate(self.twiddles32[5]));

                let input6 = [
                    load!(src, k, 6),
                    load!(src, k, 14),
                    load!(src, k, 22),
                    load!(src, k, 30),
                ];
                let mut mid6 = self.bf8.bf4.exec(input6);

                mid6[1] = AvxStoreF::mul_by_complex(mid6[1], self.twiddles32[4]);
                mid6[2] = self.bf8.rotate3(mid6[2]);
                mid6[3] = AvxStoreF::mul_by_complex(mid6[3], self.twiddles32[1].neg());

                let input7 = [
                    load!(src, k, 7),
                    load!(src, k, 15),
                    load!(src, k, 23),
                    load!(src, k, 31),
                ];
                let mut mid7 = self.bf8.bf4.exec(input7);

                mid7[1] = AvxStoreF::mul_by_complex(mid7[1], self.twiddles32[5]);
                mid7[2] = AvxStoreF::mul_by_complex(mid7[2], self.bf8.rotate(self.twiddles32[4]));
                mid7[3] = AvxStoreF::mul_by_complex(mid7[3], self.twiddles32[3].neg());

                let input0 = [
                    load!(src, k, 0),
                    load!(src, k, 8),
                    load!(src, k, 16),
                    load!(src, k, 24),
                ];
                let mid0 = self.bf8.bf4.exec(input0);

                for i in 0..4 {
                    let output = self.bf8.exec([
                        mid0[i], mid1[i], mid2[i], mid3[i], mid4[i], mid5[i], mid6[i], mid7[i],
                    ]);
                    store!(output[0], i, dst, k);
                    store!(output[1], i + 4, dst, k);
                    store!(output[2], i + 8, dst, k);
                    store!(output[3], i + 12, dst, k);
                    store!(output[4], i + 16, dst, k);
                    store!(output[5], i + 20, dst, k);
                    store!(output[6], i + 24, dst, k);
                    store!(output[7], i + 28, dst, k);
                }
            }
        }
    }
}

impl AvxButterfly1024f {
    #[inline(always)]
    fn exec_bf32_1(&self, src: &[Complex<f32>], dst: &mut [MaybeUninit<Complex<f32>>]) {
        unsafe {
            for k in 0..8 {
                macro_rules! load {
                    ($src: expr, $k: expr, $idx: expr) => {{ AvxStoreF::from_complex_ref($src.get_unchecked($k * 4 + $idx * 32..)) }};
                }

                macro_rules! store0 {
                    ($v: expr, $idx: expr, $dst: expr, $k: expr) => {{ $v.write_lo2u($dst.get_unchecked_mut(($k * 4) * 32 + $idx..)) }};
                }

                macro_rules! store1 {
                    ($v: expr, $idx: expr, $dst: expr, $k: expr) => {{
                        $v.write_lo2u($dst.get_unchecked_mut(($k * 4 + 1) * 32 + $idx..));
                    }};
                }

                macro_rules! store2 {
                    ($v: expr, $idx: expr, $dst: expr, $k: expr) => {{
                        $v.write_lo2u($dst.get_unchecked_mut(($k * 4 + 2) * 32 + $idx..));
                    }};
                }

                macro_rules! store3 {
                    ($v: expr, $idx: expr, $dst: expr, $k: expr) => {{
                        $v.write_lo2u($dst.get_unchecked_mut(($k * 4 + 3) * 32 + $idx..));
                    }};
                }

                let input1 = [
                    load!(src, k, 1),
                    load!(src, k, 9),
                    load!(src, k, 17),
                    load!(src, k, 25),
                ];
                let mut mid1 = self.bf8.bf4.exec(input1);

                mid1[1] = AvxStoreF::mul_by_complex(mid1[1], self.twiddles32[0]);
                mid1[2] = AvxStoreF::mul_by_complex(mid1[2], self.twiddles32[1]);
                mid1[3] = AvxStoreF::mul_by_complex(mid1[3], self.twiddles32[2]);

                let input2 = [
                    load!(src, k, 2),
                    load!(src, k, 10),
                    load!(src, k, 18),
                    load!(src, k, 26),
                ];
                let mut mid2 = self.bf8.bf4.exec(input2);

                mid2[1] = AvxStoreF::mul_by_complex(mid2[1], self.twiddles32[1]);
                mid2[2] = self.bf8.rotate1(mid2[2]);
                mid2[3] = AvxStoreF::mul_by_complex(mid2[3], self.twiddles32[4]);

                let input3 = [
                    load!(src, k, 3),
                    load!(src, k, 11),
                    load!(src, k, 19),
                    load!(src, k, 27),
                ];
                let mut mid3 = self.bf8.bf4.exec(input3);

                mid3[1] = AvxStoreF::mul_by_complex(mid3[1], self.twiddles32[2]);
                mid3[2] = AvxStoreF::mul_by_complex(mid3[2], self.twiddles32[4]);
                mid3[3] = AvxStoreF::mul_by_complex(mid3[3], self.bf8.rotate(self.twiddles32[0]));

                let input4 = [
                    load!(src, k, 4),
                    load!(src, k, 12),
                    load!(src, k, 20),
                    load!(src, k, 28),
                ];
                let mut mid4 = self.bf8.bf4.exec(input4);

                mid4[1] = self.bf8.rotate1(mid4[1]);
                mid4[2] = self.bf8.rotate(mid4[2]);
                mid4[3] = self.bf8.rotate3(mid4[3]);

                let input5 = [
                    load!(src, k, 5),
                    load!(src, k, 13),
                    load!(src, k, 21),
                    load!(src, k, 29),
                ];
                let mut mid5 = self.bf8.bf4.exec(input5);

                mid5[1] = AvxStoreF::mul_by_complex(mid5[1], self.twiddles32[3]);
                mid5[2] = AvxStoreF::mul_by_complex(mid5[2], self.bf8.rotate(self.twiddles32[1]));
                mid5[3] = AvxStoreF::mul_by_complex(mid5[3], self.bf8.rotate(self.twiddles32[5]));

                let input6 = [
                    load!(src, k, 6),
                    load!(src, k, 14),
                    load!(src, k, 22),
                    load!(src, k, 30),
                ];
                let mut mid6 = self.bf8.bf4.exec(input6);

                mid6[1] = AvxStoreF::mul_by_complex(mid6[1], self.twiddles32[4]);
                mid6[2] = self.bf8.rotate3(mid6[2]);
                mid6[3] = AvxStoreF::mul_by_complex(mid6[3], self.twiddles32[1].neg());

                let input7 = [
                    load!(src, k, 7),
                    load!(src, k, 15),
                    load!(src, k, 23),
                    load!(src, k, 31),
                ];
                let mut mid7 = self.bf8.bf4.exec(input7);

                mid7[1] = AvxStoreF::mul_by_complex(mid7[1], self.twiddles32[5]);
                mid7[2] = AvxStoreF::mul_by_complex(mid7[2], self.bf8.rotate(self.twiddles32[4]));
                mid7[3] = AvxStoreF::mul_by_complex(mid7[3], self.twiddles32[3].neg());

                let input0 = [
                    load!(src, k, 0),
                    load!(src, k, 8),
                    load!(src, k, 16),
                    load!(src, k, 24),
                ];
                let mid0 = self.bf8.bf4.exec(input0);

                let tw = k * 31;

                {
                    let output0 = self.bf8.exec([
                        mid0[0], mid1[0], mid2[0], mid3[0], mid4[0], mid5[0], mid6[0], mid7[0],
                    ]);
                    let output1 = self.bf8.exec([
                        mid0[1], mid1[1], mid2[1], mid3[1], mid4[1], mid5[1], mid6[1], mid7[1],
                    ]);

                    let q1 = AvxStoreF::mul_by_complex(output1[0], self.twiddles[tw]);
                    let t = transpose_f32x2_4x2([output0[0], q1]);
                    store0!(t[0], 0, dst, k);
                    store1!(t[1], 0, dst, k);
                    store2!(t[2], 0, dst, k);
                    store3!(t[3], 0, dst, k);

                    for q in 1..8 {
                        let q0 =
                            AvxStoreF::mul_by_complex(output0[q], self.twiddles[q * 4 - 1 + tw]);
                        let q1 = AvxStoreF::mul_by_complex(output1[q], self.twiddles[q * 4 + tw]);
                        let t = transpose_f32x2_4x2([q0, q1]);
                        store0!(t[0], q * 4, dst, k);
                        store1!(t[1], q * 4, dst, k);
                        store2!(t[2], q * 4, dst, k);
                        store3!(t[3], q * 4, dst, k);
                    }
                }

                {
                    let output0 = self.bf8.exec([
                        mid0[2], mid1[2], mid2[2], mid3[2], mid4[2], mid5[2], mid6[2], mid7[2],
                    ]);
                    let output1 = self.bf8.exec([
                        mid0[3], mid1[3], mid2[3], mid3[3], mid4[3], mid5[3], mid6[3], mid7[3],
                    ]);

                    for q in 0..8 {
                        let q0 =
                            AvxStoreF::mul_by_complex(output0[q], self.twiddles[q * 4 + 1 + tw]);
                        let q1 =
                            AvxStoreF::mul_by_complex(output1[q], self.twiddles[q * 4 + 2 + tw]);
                        let t = transpose_f32x2_4x2([q0, q1]);
                        store0!(t[0], q * 4 + 2, dst, k);
                        store1!(t[1], q * 4 + 2, dst, k);
                        store2!(t[2], q * 4 + 2, dst, k);
                        store3!(t[3], q * 4 + 2, dst, k);
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn run<S: BidirectionalStore<Complex<f32>>>(&self, chunk: &mut S) {
        let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 1024];
        // columns
        self.exec_bf32_1(chunk.slice_from(0..), &mut scratch);
        // rows
        self.exec_bf32_2(&mut scratch, chunk.slice_from_mut(0..));
    }
}

boring_avx_butterfly!(AvxButterfly1024f, f32, 1024);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly1024, f32, AvxButterfly1024f, 1024, 1e-3);
    test_oof_avx_butterfly!(
        test_oof_avx_butterfly1024,
        f32,
        AvxButterfly1024f,
        1024,
        1e-3
    );
}
