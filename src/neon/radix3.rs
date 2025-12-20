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
use crate::factory::AlgorithmFactory;
use crate::neon::transpose::{neon_transpose_f64x2_4x4_impl, transpose_f32x2_4x4};
use crate::neon::util::{create_neon_twiddles, vfcmulq_f32, vfcmulq_f64};
use crate::radix3::Radix3Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{compute_logarithm, compute_twiddle, is_power_of_three, reverse_bits};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::aarch64::*;
use std::fmt::Display;
use std::sync::Arc;

#[inline]
pub(crate) fn complex3_load_f32(array: &[Complex<f32>], idx: usize) -> float32x4x2_t {
    unsafe {
        float32x4x2_t(
            vld1q_f32(array.get_unchecked(idx..).as_ptr().cast()),
            vcombine_f32(
                vld1_f32(array.get_unchecked(idx + 2..).as_ptr().cast()),
                vdup_n_f32(0.),
            ),
        )
    }
}

#[inline]
pub(crate) fn complex3_store_f32(array: &mut [Complex<f32>], idx: usize, v: float32x4x2_t) {
    unsafe {
        vst1q_f32(array.get_unchecked_mut(idx..).as_mut_ptr().cast(), v.0);
        vst1_f32(
            array.get_unchecked_mut(idx + 2..).as_mut_ptr().cast(),
            vget_low_f32(v.1),
        );
    }
}

#[inline]
fn complex3_load_f64(array: &[Complex<f64>], idx: usize) -> float64x2x4_t {
    unsafe {
        float64x2x4_t(
            vld1q_f64(array.get_unchecked(idx..).as_ptr().cast()),
            vld1q_f64(array.get_unchecked(idx + 1..).as_ptr().cast()),
            vld1q_f64(array.get_unchecked(idx + 2..).as_ptr().cast()),
            vdupq_n_f64(0.),
        )
    }
}

#[inline]
fn complex3_store_f64(array: &mut [Complex<f64>], idx: usize, v: float64x2x4_t) {
    unsafe {
        vst1q_f64(array.get_unchecked_mut(idx..).as_mut_ptr().cast(), v.0);
        vst1q_f64(array.get_unchecked_mut(idx + 1..).as_mut_ptr().cast(), v.1);
        vst1q_f64(array.get_unchecked_mut(idx + 2..).as_mut_ptr().cast(), v.2);
    }
}

pub(crate) fn neon_bitreversed_transpose_f32_radix3(
    height: usize,
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
) {
    let width = input.len() / height;

    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }

    const WIDTH: usize = 3;
    const HEIGHT: usize = 3;

    let rev_digits = compute_logarithm::<3>(width).unwrap();
    let strided_width = width / WIDTH;
    let strided_height = height / HEIGHT;

    for x in 0..strided_width {
        let x_rev = [
            reverse_bits::<WIDTH>(WIDTH * x, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 1, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 2, rev_digits) * height,
        ];

        for y in 0..strided_height {
            let base_input_idx = (WIDTH * x) + y * HEIGHT * width;
            let rows = [
                complex3_load_f32(input, base_input_idx),
                complex3_load_f32(input, base_input_idx + width),
                complex3_load_f32(input, base_input_idx + width * 2),
            ];
            let transposed = transpose_f32x2_4x4(rows[0], rows[1], rows[2], unsafe {
                float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.))
            });

            complex3_store_f32(output, HEIGHT * y + x_rev[0], transposed.0);
            complex3_store_f32(output, HEIGHT * y + x_rev[1], transposed.1);
            complex3_store_f32(output, HEIGHT * y + x_rev[2], transposed.2);
        }
    }
}

pub(crate) fn neon_bitreversed_transpose_f64_radix3(
    height: usize,
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
) {
    let width = input.len() / height;

    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }

    const WIDTH: usize = 3;
    const HEIGHT: usize = 3;

    let rev_digits = compute_logarithm::<3>(width).unwrap();
    let strided_width = width / WIDTH;
    let strided_height = height / HEIGHT;

    for x in 0..strided_width {
        let x_rev = [
            reverse_bits::<WIDTH>(WIDTH * x, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 1, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 2, rev_digits) * height,
        ];

        for y in 0..strided_height {
            let base_input_idx = (WIDTH * x) + y * HEIGHT * width;
            let rows = [
                complex3_load_f64(input, base_input_idx),
                complex3_load_f64(input, base_input_idx + width),
                complex3_load_f64(input, base_input_idx + width * 2),
            ];
            let transposed = neon_transpose_f64x2_4x4_impl(rows[0], rows[1], rows[2], unsafe {
                float64x2x4_t(
                    vdupq_n_f64(0.),
                    vdupq_n_f64(0.),
                    vdupq_n_f64(0.),
                    vdupq_n_f64(0.),
                )
            });

            complex3_store_f64(output, HEIGHT * y + x_rev[0], transposed.0);
            complex3_store_f64(output, HEIGHT * y + x_rev[1], transposed.1);
            complex3_store_f64(output, HEIGHT * y + x_rev[2], transposed.2);
        }
    }
}

pub(crate) struct NeonRadix3<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle_re: T,
    twiddle_im: [T; 4],
    direction: FftDirection,
    base_fft: Arc<dyn CompositeFftExecutor<T> + Send + Sync>,
    base_len: usize,
}

impl<
    T: Default
        + Clone
        + Radix3Twiddles
        + 'static
        + Copy
        + FftTrigonometry
        + Float
        + Send
        + Sync
        + AlgorithmFactory<T>
        + MulAdd<T, Output = T>
        + SpectrumOpsFactory<T>
        + Display
        + TransposeFactory<T>,
> NeonRadix3<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonRadix3<T>, ZaftError> {
        assert!(
            is_power_of_three(size as u64),
            "Input length must be power of 3"
        );

        let exponent = compute_logarithm::<3>(size).unwrap_or_else(|| {
            panic!("Neon Fcma Radix3 length must be power of 3, but got {size}",)
        });

        let base_fft = match exponent {
            0 => T::butterfly1(fft_direction)?,
            1 => T::butterfly3(fft_direction)?,
            2 => T::butterfly9(fft_direction)?,
            3 => T::butterfly27(fft_direction)?,
            4 => T::butterfly81(fft_direction).map_or_else(|| T::butterfly27(fft_direction), Ok)?,
            _ => T::butterfly243(fft_direction).map_or_else(
                || T::butterfly81(fft_direction).map_or_else(|| T::butterfly27(fft_direction), Ok),
                Ok,
            )?,
        };

        let twiddles = create_neon_twiddles::<T, 3>(base_fft.length(), size, fft_direction)?;

        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);

        Ok(NeonRadix3 {
            execution_length: size,
            twiddles,
            twiddle_re: twiddle.re,
            twiddle_im: [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im],
            direction: fft_direction,
            base_len: base_fft.length(),
            base_fft,
        })
    }
}

impl FftExecutor<f64> for NeonRadix3<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let twiddle_re = vdupq_n_f64(self.twiddle_re);
            let twiddle_w_2 = vld1q_f64(self.twiddle_im.as_ptr().cast());

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                neon_bitreversed_transpose_f64_radix3(self.base_len, chunk, &mut scratch);

                self.base_fft.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.base_len;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 3;
                    let third = len / 3;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < third {
                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + third..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                            );
                            let u2 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast()),
                            );

                            let u3 = vld1q_f64(data.get_unchecked(j + 1..).as_ptr().cast());
                            let u4 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + third + 1..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * (j + 1)..).as_ptr().cast()),
                            );
                            let u5 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 1 + 2 * third..).as_ptr().cast()),
                                vld1q_f64(
                                    m_twiddles.get_unchecked(2 * (j + 1) + 1..).as_ptr().cast(),
                                ),
                            );

                            // Radix-3 butterfly
                            let xp0 = vaddq_f64(u1, u2);
                            let xn0 = vsubq_f64(u1, u2);
                            let sum0 = vaddq_f64(u0, xp0);

                            let xp1 = vaddq_f64(u4, u5);
                            let xn1 = vsubq_f64(u4, u5);
                            let sum1 = vaddq_f64(u3, xp1);

                            let w_01 = vfmaq_f64(u0, twiddle_re, xp0);
                            let w_02 = vfmaq_f64(u3, twiddle_re, xp1);

                            let xn0_rot = vextq_f64::<1>(xn0, xn0);
                            let xn1_rot = vextq_f64::<1>(xn1, xn1);

                            let vy0 = sum0;
                            let vy1 = vfmaq_f64(w_01, twiddle_w_2, xn0_rot);
                            let vy2 = vfmsq_f64(w_01, twiddle_w_2, xn0_rot);

                            let vy3 = sum1;
                            let vy4 = vfmaq_f64(w_02, twiddle_w_2, xn1_rot);
                            let vy5 = vfmsq_f64(w_02, twiddle_w_2, xn1_rot);

                            vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            vst1q_f64(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );
                            vst1q_f64(data.get_unchecked_mut(j + 1..).as_mut_ptr().cast(), vy3);
                            vst1q_f64(
                                data.get_unchecked_mut(j + 1 + third..).as_mut_ptr().cast(),
                                vy4,
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 1 + 2 * third..)
                                    .as_mut_ptr()
                                    .cast(),
                                vy5,
                            );
                            j += 2;
                        }

                        for j in j..third {
                            let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let u1 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + third..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                            );
                            let u2 = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast()),
                            );

                            // Radix-3 butterfly
                            let xp = vaddq_f64(u1, u2);
                            let xn = vsubq_f64(u1, u2);
                            let sum = vaddq_f64(u0, xp);

                            let w_1 = vfmaq_f64(u0, twiddle_re, xp);

                            let xn_rot = vextq_f64::<1>(xn, xn);

                            let vy0 = sum;
                            let vy1 = vfmaq_f64(w_1, twiddle_w_2, xn_rot);
                            let vy2 = vfmsq_f64(w_1, twiddle_w_2, xn_rot);

                            vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            vst1q_f64(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 2..];
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

impl FftExecutor<f32> for NeonRadix3<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let twiddle_re = vdupq_n_f32(self.twiddle_re);
            let twiddle_w_2 = vld1q_f32(self.twiddle_im.as_ptr().cast());

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                neon_bitreversed_transpose_f32_radix3(self.base_len, chunk, &mut scratch);

                self.base_fft.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.base_len;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 3;
                    let third = len / 3;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 4 < third {
                            let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(2 * j + 2..).as_ptr().cast());

                            let tw2 =
                                vld1q_f32(m_twiddles.get_unchecked(2 * j + 4..).as_ptr().cast());
                            let tw3 =
                                vld1q_f32(m_twiddles.get_unchecked(2 * j + 6..).as_ptr().cast());

                            let u1 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + third..).as_ptr().cast()),
                                tw0,
                            );

                            let u2 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                tw1,
                            );

                            let u3 = vld1q_f32(data.get_unchecked(j + 2..).as_ptr().cast());

                            let u4 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + third + 2..).as_ptr().cast()),
                                tw2,
                            );
                            let u5 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * third + 2..).as_ptr().cast()),
                                tw3,
                            );

                            // Radix-3 butterfly
                            let xp0 = vaddq_f32(u1, u2);
                            let xn0 = vsubq_f32(u1, u2);
                            let sum0 = vaddq_f32(u0, xp0);

                            let xp1 = vaddq_f32(u4, u5);
                            let xn1 = vsubq_f32(u4, u5);
                            let sum1 = vaddq_f32(u3, xp1);

                            let xn0_rot = vrev64q_f32(xn0);
                            let xn1_rot = vrev64q_f32(xn1);

                            let w_01 = vfmaq_f32(u0, twiddle_re, xp0);
                            let w_02 = vfmaq_f32(u3, twiddle_re, xp1);

                            let vy0 = sum0;
                            let vy1 = vfmaq_f32(w_01, twiddle_w_2, xn0_rot);
                            let vy2 = vfmsq_f32(w_01, twiddle_w_2, xn0_rot);

                            let vy3 = sum1;
                            let vy4 = vfmaq_f32(w_02, twiddle_w_2, xn1_rot);
                            let vy5 = vfmsq_f32(w_02, twiddle_w_2, xn1_rot);

                            vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            vst1q_f32(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );

                            vst1q_f32(data.get_unchecked_mut(j + 2..).as_mut_ptr().cast(), vy3);
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 + third..).as_mut_ptr().cast(),
                                vy4,
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 + 2 * third..)
                                    .as_mut_ptr()
                                    .cast(),
                                vy5,
                            );

                            j += 4;
                        }

                        while j + 2 < third {
                            let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(2 * j + 2..).as_ptr().cast());

                            let u1 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + third..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                tw1,
                            );

                            // Radix-3 butterfly
                            let xp = vaddq_f32(u1, u2);
                            let xn = vsubq_f32(u1, u2);
                            let sum = vaddq_f32(u0, xp);

                            let w_1 = vfmaq_f32(u0, twiddle_re, xp);
                            let xn_rot = vrev64q_f32(xn);

                            let vy0 = sum;
                            let vy1 = vfmaq_f32(w_1, twiddle_w_2, xn_rot);
                            let vy2 = vfmsq_f32(w_1, twiddle_w_2, xn_rot);

                            vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            vst1q_f32(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );

                            j += 2;
                        }

                        for j in j..third {
                            let u0 = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw = vld1q_f32(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());

                            let u1u2 = vfcmulq_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + third..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                ),
                                tw,
                            );

                            let u1 = vget_low_f32(u1u2);
                            let u2 = vget_high_f32(u1u2);

                            // Radix-3 butterfly
                            let xp = vadd_f32(u1, u2);
                            let xn = vsub_f32(u1, u2);
                            let sum = vadd_f32(u0, xp);

                            let w_1 = vfma_f32(u0, vget_low_f32(twiddle_re), xp);

                            let xn_rot = vext_f32::<1>(xn, xn);

                            let vy0 = sum;
                            let vy1 = vfma_f32(w_1, vget_low_f32(twiddle_w_2), xn_rot);
                            let vy2 = vfms_f32(w_1, vget_low_f32(twiddle_w_2), xn_rot);

                            vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            vst1_f32(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                            vst1_f32(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 2..];
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
    use crate::neon::NeonRadix3;
    use crate::util::test_radix;

    test_radix!(test_neon_radix3, f32, NeonRadix3, 6, 3, 1e-3);
    test_radix!(test_neon_radix3_f64, f64, NeonRadix3, 6, 3, 1e-8);
}
