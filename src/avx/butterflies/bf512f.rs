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
use crate::avx::butterflies::shared::gen_butterfly_twiddles_f32;
use crate::avx::mixed::{AvxStoreF, ColumnButterfly4f, ColumnButterfly8f, ColumnButterfly16f};
use crate::avx::transpose::transpose_4x16;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::mem::MaybeUninit;
use std::sync::Arc;

pub(crate) struct AvxButterfly512f {
    direction: FftDirection,
    bf16: ColumnButterfly16f,
    bf8: ColumnButterfly8f,
    bf4: ColumnButterfly4f,
    twiddles: [AvxStoreF; 240],
    twiddles32: [AvxStoreF; 6],
}

impl AvxButterfly512f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }

    #[target_feature(enable = "avx")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddles: gen_butterfly_twiddles_f32(32, 16, fft_direction, 512),
            twiddles32: [
                AvxStoreF::set_complex(compute_twiddle(1, 32, fft_direction)),
                AvxStoreF::set_complex(compute_twiddle(2, 32, fft_direction)),
                AvxStoreF::set_complex(compute_twiddle(3, 32, fft_direction)),
                AvxStoreF::set_complex(compute_twiddle(5, 32, fft_direction)),
                AvxStoreF::set_complex(compute_twiddle(6, 32, fft_direction)),
                AvxStoreF::set_complex(compute_twiddle(7, 32, fft_direction)),
            ],
            bf16: ColumnButterfly16f::new(fft_direction),
            bf8: ColumnButterfly8f::new(fft_direction),
            bf4: ColumnButterfly4f::new(fft_direction),
        }
    }
}

impl FftExecutor<f32> for AvxButterfly512f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        512
    }
}

impl AvxButterfly512f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn exec_bf32(&self, src: &mut [MaybeUninit<Complex<f32>>], dst: &mut [Complex<f32>]) {
        unsafe {
            // Size-4 FFTs down the columns
            for k in 0..4 {
                macro_rules! load {
                    ($src: expr, $k: expr, $idx: expr) => {{ AvxStoreF::from_complex_refu($src.get_unchecked($k * 4 + $idx * 16..)) }};
                }

                macro_rules! store {
                    ($v: expr, $idx: expr, $dst: expr, $k: expr) => {{ $v.write($dst.get_unchecked_mut($k * 4 + $idx * 16..)) }};
                }

                let input1 = [
                    load!(src, k, 1),
                    load!(src, k, 9),
                    load!(src, k, 17),
                    load!(src, k, 25),
                ];
                let mut mid1 = self.bf4.exec(input1);

                mid1[1] = AvxStoreF::mul_by_complex(mid1[1], self.twiddles32[0]);
                mid1[2] = AvxStoreF::mul_by_complex(mid1[2], self.twiddles32[1]);
                mid1[3] = AvxStoreF::mul_by_complex(mid1[3], self.twiddles32[2]);

                let input2 = [
                    load!(src, k, 2),
                    load!(src, k, 10),
                    load!(src, k, 18),
                    load!(src, k, 26),
                ];
                let mut mid2 = self.bf4.exec(input2);

                mid2[1] = AvxStoreF::mul_by_complex(mid2[1], self.twiddles32[1]);
                mid2[2] = self.bf8.rotate1(mid2[2]);
                mid2[3] = AvxStoreF::mul_by_complex(mid2[3], self.twiddles32[4]);

                let input3 = [
                    load!(src, k, 3),
                    load!(src, k, 11),
                    load!(src, k, 19),
                    load!(src, k, 27),
                ];
                let mut mid3 = self.bf4.exec(input3);

                mid3[1] = AvxStoreF::mul_by_complex(mid3[1], self.twiddles32[2]);
                mid3[2] = AvxStoreF::mul_by_complex(mid3[2], self.twiddles32[4]);
                mid3[3] = AvxStoreF::mul_by_complex(mid3[3], self.bf8.rotate(self.twiddles32[0]));

                let input4 = [
                    load!(src, k, 4),
                    load!(src, k, 12),
                    load!(src, k, 20),
                    load!(src, k, 28),
                ];
                let mut mid4 = self.bf4.exec(input4);

                mid4[1] = self.bf8.rotate1(mid4[1]);
                mid4[2] = self.bf8.rotate(mid4[2]);
                mid4[3] = self.bf8.rotate3(mid4[3]);

                let input5 = [
                    load!(src, k, 5),
                    load!(src, k, 13),
                    load!(src, k, 21),
                    load!(src, k, 29),
                ];
                let mut mid5 = self.bf4.exec(input5);

                mid5[1] = AvxStoreF::mul_by_complex(mid5[1], self.twiddles32[3]);
                mid5[2] = AvxStoreF::mul_by_complex(mid5[2], self.bf8.rotate(self.twiddles32[1]));
                mid5[3] = AvxStoreF::mul_by_complex(mid5[3], self.bf8.rotate(self.twiddles32[5]));

                let input6 = [
                    load!(src, k, 6),
                    load!(src, k, 14),
                    load!(src, k, 22),
                    load!(src, k, 30),
                ];
                let mut mid6 = self.bf4.exec(input6);

                mid6[1] = AvxStoreF::mul_by_complex(mid6[1], self.twiddles32[4]);
                mid6[2] = self.bf8.rotate3(mid6[2]);
                mid6[3] = AvxStoreF::mul_by_complex(mid6[3], self.twiddles32[1].neg());

                let input7 = [
                    load!(src, k, 7),
                    load!(src, k, 15),
                    load!(src, k, 23),
                    load!(src, k, 31),
                ];
                let mut mid7 = self.bf4.exec(input7);

                mid7[1] = AvxStoreF::mul_by_complex(mid7[1], self.twiddles32[5]);
                mid7[2] = AvxStoreF::mul_by_complex(mid7[2], self.bf8.rotate(self.twiddles32[4]));
                mid7[3] = AvxStoreF::mul_by_complex(mid7[3], self.twiddles32[3].neg());

                let input0 = [
                    load!(src, k, 0),
                    load!(src, k, 8),
                    load!(src, k, 16),
                    load!(src, k, 24),
                ];
                let mid0 = self.bf4.exec(input0);

                // All the data is now in the right format to just do a bunch of butterfly 8's in a loop.
                // Write the data out to the final output as we go so that the compiler can stop worrying about finding stack space for it
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

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 512 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows: [AvxStoreF; 16] = [AvxStoreF::zero(); 16];
            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 512];

            for chunk in in_place.chunks_exact_mut(512) {
                // columns
                for k in 0..8 {
                    for i in 0..16 {
                        rows[i] =
                            AvxStoreF::from_complex_ref(chunk.get_unchecked(i * 32 + k * 4..));
                    }

                    rows = self.bf16.exec(rows);

                    for i in 1..16 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 15 * k]);
                    }

                    let transposed = transpose_4x16(rows);

                    for i in 0..4 {
                        transposed[i * 4].write_u(scratch.get_unchecked_mut(k * 4 * 16 + i * 4..));
                        transposed[i * 4 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 16 + i * 4..));
                        transposed[i * 4 + 2]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 2) * 16 + i * 4..));
                        transposed[i * 4 + 3]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 3) * 16 + i * 4..));
                    }
                }

                // rows

                self.exec_bf32(&mut scratch, chunk);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly512f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly512f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 512 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 512 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows: [AvxStoreF; 16] = [AvxStoreF::zero(); 16];
            let mut scratch = [MaybeUninit::<Complex<f32>>::uninit(); 512];

            for (dst, src) in dst.chunks_exact_mut(512).zip(src.chunks_exact(512)) {
                // columns
                for k in 0..8 {
                    for i in 0..16 {
                        rows[i] = AvxStoreF::from_complex_ref(src.get_unchecked(i * 32 + k * 4..));
                    }

                    rows = self.bf16.exec(rows);

                    for i in 1..16 {
                        rows[i] = AvxStoreF::mul_by_complex(rows[i], self.twiddles[i - 1 + 15 * k]);
                    }

                    let transposed = transpose_4x16(rows);

                    for i in 0..4 {
                        transposed[i * 4].write_u(scratch.get_unchecked_mut(k * 4 * 16 + i * 4..));
                        transposed[i * 4 + 1]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 1) * 16 + i * 4..));
                        transposed[i * 4 + 2]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 2) * 16 + i * 4..));
                        transposed[i * 4 + 3]
                            .write_u(scratch.get_unchecked_mut((k * 4 + 3) * 16 + i * 4..));
                    }
                }

                // rows

                self.exec_bf32(&mut scratch, dst);
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly512f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly512, f32, AvxButterfly512f, 512, 1e-3);
    test_oof_avx_butterfly!(test_oof_neon_butterfly512, f32, AvxButterfly512f, 512, 1e-3);
}
