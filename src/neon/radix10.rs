/*
 * // Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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
use crate::err::try_vec;
use crate::neon::butterflies::{NeonButterfly, NeonFastButterfly5};
use crate::neon::radix4::{complex4_load_f32, complex4_store_f32};
use crate::neon::transpose::{neon_transpose_f32x2_2x2_impl, transpose_f32x2_4x4};
use crate::neon::util::{create_neon_twiddles, vfcmul_f32, vfcmulq_f32, vfcmulq_f64};
use crate::radix10::Radix10Twiddles;
use crate::util::{bitreversed_transpose, int_logarithm, is_power_of_ten, reverse_bits};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::arch::aarch64::*;
use std::sync::Arc;

pub(crate) struct NeonRadix10<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    bf5: NeonFastButterfly5<T>,
    direction: FftDirection,
    butterfly: Arc<dyn CompositeFftExecutor<T> + Send + Sync>,
    butterfly_length: usize,
}

impl<T: FftSample + Radix10Twiddles> NeonRadix10<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonRadix10<T>, ZaftError> {
        assert!(
            is_power_of_ten(size as u64),
            "Input length must be a power of 10"
        );

        let log10 = int_logarithm::<10>(size).unwrap();
        let butterfly = match log10 {
            0 => T::butterfly1(fft_direction)?,
            1 => T::butterfly10(fft_direction)?,
            _ => {
                T::butterfly100(fft_direction).map_or_else(|| T::butterfly10(fft_direction), Ok)?
            }
        };

        let butterfly_length = butterfly.length();

        let twiddles = create_neon_twiddles::<T, 10>(butterfly_length, size, fft_direction)?;

        Ok(NeonRadix10 {
            execution_length: size,
            twiddles,
            bf5: NeonFastButterfly5::new(fft_direction),
            direction: fft_direction,
            butterfly,
            butterfly_length,
        })
    }
}

impl FftExecutor<f64> for NeonRadix10<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }
        unsafe {
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            let rot_sign = vld1q_f64(ROT_90.as_ptr());

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f64>, 10>(
                    self.butterfly_length,
                    chunk,
                    &mut scratch,
                );

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.butterfly_length;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 10;
                    let tenth = len / 10;

                    for data in chunk.chunks_exact_mut(len) {
                        for j in 0..tenth {
                            let td = 9 * j;
                            let tw0 = vld1q_f64(m_twiddles.get_unchecked(td..).as_ptr().cast());
                            let tw1 = vld1q_f64(m_twiddles.get_unchecked(td + 1..).as_ptr().cast());
                            let tw2 = vld1q_f64(m_twiddles.get_unchecked(td + 2..).as_ptr().cast());
                            let tw3 = vld1q_f64(m_twiddles.get_unchecked(td + 3..).as_ptr().cast());
                            let tw4 = vld1q_f64(m_twiddles.get_unchecked(td + 4..).as_ptr().cast());
                            let tw5 = vld1q_f64(m_twiddles.get_unchecked(td + 5..).as_ptr().cast());
                            let tw6 = vld1q_f64(m_twiddles.get_unchecked(td + 6..).as_ptr().cast());
                            let tw7 = vld1q_f64(m_twiddles.get_unchecked(td + 7..).as_ptr().cast());
                            let tw8 = vld1q_f64(m_twiddles.get_unchecked(td + 8..).as_ptr().cast());

                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + tenth..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * tenth..).as_ptr().cast()),
                                tw1,
                            );
                            let u3 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 3 * tenth..).as_ptr().cast()),
                                tw2,
                            );
                            let u4 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 4 * tenth..).as_ptr().cast()),
                                tw3,
                            );
                            let u5 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 5 * tenth..).as_ptr().cast()),
                                tw4,
                            );
                            let u6 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 6 * tenth..).as_ptr().cast()),
                                tw5,
                            );
                            let u7 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 7 * tenth..).as_ptr().cast()),
                                tw6,
                            );
                            let u8 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 8 * tenth..).as_ptr().cast()),
                                tw7,
                            );
                            let u9 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 9 * tenth..).as_ptr().cast()),
                                tw8,
                            );

                            // Radix-10 butterfly

                            let mid0 = self.bf5.exec(u0, u2, u4, u6, u8, rot_sign);
                            let mid1 = self.bf5.exec(u5, u7, u9, u1, u3, rot_sign);

                            // Since this is good-thomas algorithm, we don't need twiddle factors
                            let (y0, y5) = NeonButterfly::butterfly2_f64(mid0.0, mid1.0);
                            let (y6, y1) = NeonButterfly::butterfly2_f64(mid0.1, mid1.1);
                            let (y2, y7) = NeonButterfly::butterfly2_f64(mid0.2, mid1.2);
                            let (y8, y3) = NeonButterfly::butterfly2_f64(mid0.3, mid1.3);
                            let (y4, y9) = NeonButterfly::butterfly2_f64(mid0.4, mid1.4);

                            // Store results
                            vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            vst1q_f64(data.get_unchecked_mut(j + tenth..).as_mut_ptr().cast(), y1);
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * tenth..).as_mut_ptr().cast(),
                                y2,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 3 * tenth..).as_mut_ptr().cast(),
                                y3,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 4 * tenth..).as_mut_ptr().cast(),
                                y4,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 5 * tenth..).as_mut_ptr().cast(),
                                y5,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 6 * tenth..).as_mut_ptr().cast(),
                                y6,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 7 * tenth..).as_mut_ptr().cast(),
                                y7,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 8 * tenth..).as_mut_ptr().cast(),
                                y8,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 9 * tenth..).as_mut_ptr().cast(),
                                y9,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 9..];
                }
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}

#[inline]
fn complex2_load_f32(array: &[Complex<f32>], idx: usize) -> float32x4_t {
    unsafe { vld1q_f32(array.get_unchecked(idx..).as_ptr().cast()) }
}

#[inline]
fn complex2_store_f32(array: &mut [Complex<f32>], idx: usize, v: float32x4_t) {
    unsafe {
        vst1q_f32(array.get_unchecked_mut(idx..).as_mut_ptr().cast(), v);
    }
}

pub(crate) fn neon_bitreversed_transpose_f32_radix10(
    height: usize,
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
) {
    let width = input.len() / height;

    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }

    const WIDTH: usize = 10;
    const HEIGHT: usize = 10;

    let rev_digits = int_logarithm::<10>(width).unwrap();
    let strided_width = width / WIDTH;
    let strided_height = height / HEIGHT;

    for x in 0..strided_width {
        // Graphically, the 10×10 matrix is partitioned as:
        // +--------+--------+------+
        // | 4×4 A0 | 4×4 A1 | 4×2 A2 |
        // +--------+--------+------+
        // | 4×4 B0 | 4×4 B1 | 4×2 B2 |
        // +--------+--------+------+
        // | 2×4 C0 | 2×4 C1 | 2×2 C2 |
        // +--------+--------+------+
        // * T
        // +--------+--------+------+
        // | 4×4 A0ᵀ | 4×4 B0ᵀ | 2×4 C0ᵀ |
        // +--------+--------+------+
        // | 4×4 A1ᵀ | 4×4 B1ᵀ | 2×4 C1ᵀ |
        // +--------+--------+------+
        // | 4×2 A2ᵀ | 4×2 B2ᵀ | 2×2 C2ᵀ |
        // +--------+--------+------+

        let x_rev = [
            reverse_bits::<WIDTH>(WIDTH * x, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 1, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 2, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 3, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 4, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 5, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 6, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 7, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 8, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 9, rev_digits) * height,
        ];

        for y in 0..strided_height {
            let base_input_idx = (WIDTH * x) + y * HEIGHT * width;
            let a0 = [
                complex4_load_f32(input, base_input_idx),
                complex4_load_f32(input, base_input_idx + width),
                complex4_load_f32(input, base_input_idx + width * 2),
                complex4_load_f32(input, base_input_idx + width * 3),
            ];

            let a1 = [
                complex4_load_f32(input, base_input_idx + 4),
                complex4_load_f32(input, base_input_idx + width + 4),
                complex4_load_f32(input, base_input_idx + width * 2 + 4),
                complex4_load_f32(input, base_input_idx + width * 3 + 4),
            ];

            let transposed_a0 = transpose_f32x2_4x4(a0[0], a0[1], a0[2], a0[3]);
            let transposed_a1 = transpose_f32x2_4x4(a1[0], a1[1], a1[2], a1[3]);

            complex4_store_f32(output, HEIGHT * y + x_rev[0], transposed_a0.0);
            complex4_store_f32(output, HEIGHT * y + x_rev[1], transposed_a0.1);
            complex4_store_f32(output, HEIGHT * y + x_rev[2], transposed_a0.2);
            complex4_store_f32(output, HEIGHT * y + x_rev[3], transposed_a0.3);

            complex4_store_f32(output, HEIGHT * y + x_rev[4], transposed_a1.0);
            complex4_store_f32(output, HEIGHT * y + x_rev[5], transposed_a1.1);
            complex4_store_f32(output, HEIGHT * y + x_rev[6], transposed_a1.2);
            complex4_store_f32(output, HEIGHT * y + x_rev[7], transposed_a1.3);

            let b0 = [
                complex4_load_f32(input, base_input_idx + width * 4),
                complex4_load_f32(input, base_input_idx + width * 5),
                complex4_load_f32(input, base_input_idx + width * 6),
                complex4_load_f32(input, base_input_idx + width * 7),
            ];

            let b1 = [
                complex4_load_f32(input, base_input_idx + width * 4 + 4),
                complex4_load_f32(input, base_input_idx + width * 5 + 4),
                complex4_load_f32(input, base_input_idx + width * 6 + 4),
                complex4_load_f32(input, base_input_idx + width * 7 + 4),
            ];

            let transposed_b0 = transpose_f32x2_4x4(b0[0], b0[1], b0[2], b0[3]);
            let transposed_b1 = transpose_f32x2_4x4(b1[0], b1[1], b1[2], b1[3]);

            complex4_store_f32(output, HEIGHT * y + x_rev[0] + 4, transposed_b0.0);
            complex4_store_f32(output, HEIGHT * y + x_rev[1] + 4, transposed_b0.1);
            complex4_store_f32(output, HEIGHT * y + x_rev[2] + 4, transposed_b0.2);
            complex4_store_f32(output, HEIGHT * y + x_rev[3] + 4, transposed_b0.3);

            complex4_store_f32(output, HEIGHT * y + x_rev[4] + 4, transposed_b1.0);
            complex4_store_f32(output, HEIGHT * y + x_rev[5] + 4, transposed_b1.1);
            complex4_store_f32(output, HEIGHT * y + x_rev[6] + 4, transposed_b1.2);
            complex4_store_f32(output, HEIGHT * y + x_rev[7] + 4, transposed_b1.3);

            let a2 = [
                complex2_load_f32(input, base_input_idx + 8),
                complex2_load_f32(input, base_input_idx + width + 8),
                complex2_load_f32(input, base_input_idx + width * 2 + 8),
                complex2_load_f32(input, base_input_idx + width * 3 + 8),
            ];

            let b2 = [
                complex2_load_f32(input, base_input_idx + width * 4 + 8),
                complex2_load_f32(input, base_input_idx + width * 5 + 8),
                complex2_load_f32(input, base_input_idx + width * 6 + 8),
                complex2_load_f32(input, base_input_idx + width * 7 + 8),
            ];

            let transposed_a2 = unsafe {
                transpose_f32x2_4x4(
                    float32x4x2_t(a2[0], vdupq_n_f32(0.)),
                    float32x4x2_t(a2[1], vdupq_n_f32(0.)),
                    float32x4x2_t(a2[2], vdupq_n_f32(0.)),
                    float32x4x2_t(a2[3], vdupq_n_f32(0.)),
                )
            };

            let transposed_b2 = unsafe {
                transpose_f32x2_4x4(
                    float32x4x2_t(b2[0], vdupq_n_f32(0.)),
                    float32x4x2_t(b2[1], vdupq_n_f32(0.)),
                    float32x4x2_t(b2[2], vdupq_n_f32(0.)),
                    float32x4x2_t(b2[3], vdupq_n_f32(0.)),
                )
            };

            complex4_store_f32(output, HEIGHT * y + x_rev[8], transposed_a2.0);
            complex4_store_f32(output, HEIGHT * y + x_rev[9], transposed_a2.1);

            complex4_store_f32(output, HEIGHT * y + x_rev[8] + 4, transposed_b2.0);
            complex4_store_f32(output, HEIGHT * y + x_rev[9] + 4, transposed_b2.1);

            let c0 = [
                complex4_load_f32(input, base_input_idx + width * 8),
                complex4_load_f32(input, base_input_idx + width * 9),
            ];

            let c1 = [
                complex4_load_f32(input, base_input_idx + width * 8 + 4),
                complex4_load_f32(input, base_input_idx + width * 9 + 4),
            ];

            let transposed_c0 = unsafe {
                transpose_f32x2_4x4(
                    c0[0],
                    c0[1],
                    float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
                    float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
                )
            };

            let transposed_c1 = unsafe {
                transpose_f32x2_4x4(
                    c1[0],
                    c1[1],
                    float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
                    float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.)),
                )
            };

            complex2_store_f32(output, HEIGHT * y + x_rev[0] + 8, transposed_c0.0.0);
            complex2_store_f32(output, HEIGHT * y + x_rev[1] + 8, transposed_c0.1.0);
            complex2_store_f32(output, HEIGHT * y + x_rev[2] + 8, transposed_c0.2.0);
            complex2_store_f32(output, HEIGHT * y + x_rev[3] + 8, transposed_c0.3.0);

            complex2_store_f32(output, HEIGHT * y + x_rev[4] + 8, transposed_c1.0.0);
            complex2_store_f32(output, HEIGHT * y + x_rev[5] + 8, transposed_c1.1.0);
            complex2_store_f32(output, HEIGHT * y + x_rev[6] + 8, transposed_c1.2.0);
            complex2_store_f32(output, HEIGHT * y + x_rev[7] + 8, transposed_c1.3.0);

            let c2 = [
                complex2_load_f32(input, base_input_idx + width * 8 + 8),
                complex2_load_f32(input, base_input_idx + width * 9 + 8),
            ];

            let transposed_c2 = neon_transpose_f32x2_2x2_impl(float32x4x2_t(c2[0], c2[1]));

            complex2_store_f32(output, HEIGHT * y + x_rev[8] + 8, transposed_c2.0);
            complex2_store_f32(output, HEIGHT * y + x_rev[9] + 8, transposed_c2.1);
        }
    }
}

impl FftExecutor<f32> for NeonRadix10<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            static ROT_90: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            let rot_sign = vld1q_f32(ROT_90.as_ptr());

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                neon_bitreversed_transpose_f32_radix10(self.butterfly_length, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.butterfly_length;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 10;
                    let tenth = len / 10;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < tenth {
                            let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw = 9 * j;

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(tw..).as_ptr().cast());
                            let tw1 = vld1q_f32(m_twiddles.get_unchecked(tw + 2..).as_ptr().cast());
                            let tw2 = vld1q_f32(m_twiddles.get_unchecked(tw + 4..).as_ptr().cast());
                            let tw3 = vld1q_f32(m_twiddles.get_unchecked(tw + 6..).as_ptr().cast());
                            let tw4 = vld1q_f32(m_twiddles.get_unchecked(tw + 8..).as_ptr().cast());
                            let tw5 =
                                vld1q_f32(m_twiddles.get_unchecked(tw + 10..).as_ptr().cast());
                            let tw6 =
                                vld1q_f32(m_twiddles.get_unchecked(tw + 12..).as_ptr().cast());
                            let tw7 =
                                vld1q_f32(m_twiddles.get_unchecked(tw + 14..).as_ptr().cast());
                            let tw8 =
                                vld1q_f32(m_twiddles.get_unchecked(tw + 16..).as_ptr().cast());

                            let u1 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + tenth..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * tenth..).as_ptr().cast()),
                                tw1,
                            );
                            let u3 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 3 * tenth..).as_ptr().cast()),
                                tw2,
                            );
                            let u4 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 4 * tenth..).as_ptr().cast()),
                                tw3,
                            );
                            let u5 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 5 * tenth..).as_ptr().cast()),
                                tw4,
                            );
                            let u6 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 6 * tenth..).as_ptr().cast()),
                                tw5,
                            );
                            let u7 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 7 * tenth..).as_ptr().cast()),
                                tw6,
                            );
                            let u8 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 8 * tenth..).as_ptr().cast()),
                                tw7,
                            );
                            let u9 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 9 * tenth..).as_ptr().cast()),
                                tw8,
                            );

                            // Radix-10 butterfly

                            let mid0 = self.bf5.exec(u0, u2, u4, u6, u8, rot_sign);
                            let mid1 = self.bf5.exec(u5, u7, u9, u1, u3, rot_sign);

                            // Since this is good-thomas algorithm, we don't need twiddle factors
                            let (y0, y5) = NeonButterfly::butterfly2_f32(mid0.0, mid1.0);
                            let (y6, y1) = NeonButterfly::butterfly2_f32(mid0.1, mid1.1);
                            let (y2, y7) = NeonButterfly::butterfly2_f32(mid0.2, mid1.2);
                            let (y8, y3) = NeonButterfly::butterfly2_f32(mid0.3, mid1.3);
                            let (y4, y9) = NeonButterfly::butterfly2_f32(mid0.4, mid1.4);

                            // Store results
                            vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            vst1q_f32(data.get_unchecked_mut(j + tenth..).as_mut_ptr().cast(), y1);
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * tenth..).as_mut_ptr().cast(),
                                y2,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 3 * tenth..).as_mut_ptr().cast(),
                                y3,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 4 * tenth..).as_mut_ptr().cast(),
                                y4,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 5 * tenth..).as_mut_ptr().cast(),
                                y5,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 6 * tenth..).as_mut_ptr().cast(),
                                y6,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 7 * tenth..).as_mut_ptr().cast(),
                                y7,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 8 * tenth..).as_mut_ptr().cast(),
                                y8,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 9 * tenth..).as_mut_ptr().cast(),
                                y9,
                            );

                            j += 2;
                        }

                        for j in j..tenth {
                            let u0 = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                            let ti = 9 * j;

                            let w0w1 = vld1q_f32(m_twiddles.get_unchecked(ti..).as_ptr().cast());
                            let w2w3 =
                                vld1q_f32(m_twiddles.get_unchecked(ti + 2..).as_ptr().cast());
                            let w4w5 =
                                vld1q_f32(m_twiddles.get_unchecked(ti + 4..).as_ptr().cast());
                            let w6w7 =
                                vld1q_f32(m_twiddles.get_unchecked(ti + 6..).as_ptr().cast());
                            let w8 = vld1_f32(m_twiddles.get_unchecked(ti + 8..).as_ptr().cast());

                            let u1u2 = vfcmulq_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + tenth..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 2 * tenth..).as_ptr().cast()),
                                ),
                                w0w1,
                            );
                            let u3u4 = vfcmulq_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + 3 * tenth..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 4 * tenth..).as_ptr().cast()),
                                ),
                                w2w3,
                            );
                            let u5u6 = vfcmulq_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + 5 * tenth..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 6 * tenth..).as_ptr().cast()),
                                ),
                                w4w5,
                            );
                            let u7u8 = vfcmulq_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + 7 * tenth..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 8 * tenth..).as_ptr().cast()),
                                ),
                                w6w7,
                            );
                            let u9 = vfcmul_f32(
                                vld1_f32(data.get_unchecked(j + 9 * tenth..).as_ptr().cast()),
                                w8,
                            );

                            let u1 = vget_low_f32(u1u2);
                            let u2 = vget_high_f32(u1u2);
                            let u3 = vget_low_f32(u3u4);
                            let u4 = vget_high_f32(u3u4);
                            let u5 = vget_low_f32(u5u6);
                            let u6 = vget_high_f32(u5u6);
                            let u7 = vget_low_f32(u7u8);
                            let u8 = vget_high_f32(u7u8);

                            // Radix-10 butterfly

                            let mid0 = self.bf5.exec(
                                vcombine_f32(u0, u5),
                                vcombine_f32(u2, u7),
                                vcombine_f32(u4, u9),
                                vcombine_f32(u6, u1),
                                vcombine_f32(u8, u3),
                                rot_sign,
                            );

                            // Since this is good-thomas algorithm, we don't need twiddle factors
                            let (y0, y5) = NeonButterfly::butterfly2h_f32(
                                vget_low_f32(mid0.0),
                                vget_high_f32(mid0.0),
                            );
                            let (y6, y1) = NeonButterfly::butterfly2h_f32(
                                vget_low_f32(mid0.1),
                                vget_high_f32(mid0.1),
                            );
                            let (y2, y7) = NeonButterfly::butterfly2h_f32(
                                vget_low_f32(mid0.2),
                                vget_high_f32(mid0.2),
                            );
                            let (y8, y3) = NeonButterfly::butterfly2h_f32(
                                vget_low_f32(mid0.3),
                                vget_high_f32(mid0.3),
                            );
                            let (y4, y9) = NeonButterfly::butterfly2h_f32(
                                vget_low_f32(mid0.4),
                                vget_high_f32(mid0.4),
                            );

                            // Store results
                            vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            vst1_f32(data.get_unchecked_mut(j + tenth..).as_mut_ptr().cast(), y1);
                            vst1_f32(
                                data.get_unchecked_mut(j + 2 * tenth..).as_mut_ptr().cast(),
                                y2,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 3 * tenth..).as_mut_ptr().cast(),
                                y3,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 4 * tenth..).as_mut_ptr().cast(),
                                y4,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 5 * tenth..).as_mut_ptr().cast(),
                                y5,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 6 * tenth..).as_mut_ptr().cast(),
                                y6,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 7 * tenth..).as_mut_ptr().cast(),
                                y7,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 8 * tenth..).as_mut_ptr().cast(),
                                y8,
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 9 * tenth..).as_mut_ptr().cast(),
                                y9,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 9..];
                }
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test_radix;

    test_radix!(test_neon_radix10, f32, NeonRadix10, 4, 10, 1e-3);
    test_radix!(test_neon_radix10_f64, f64, NeonRadix10, 4, 10, 1e-8);
}
