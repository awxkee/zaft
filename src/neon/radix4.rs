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
use crate::neon::f32x2_4x4::transpose_f32x2_4x4;
use crate::neon::f64x2_2x2::neon_transpose_f64x2_4x4_impl;
use crate::neon::util::{create_neon_twiddles, vfcmul_f32, vfcmulq_f32, vfcmulq_f64};
use crate::radix4::Radix4Twiddles;
use crate::traits::FftTrigonometry;
use crate::util::reverse_bits;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

#[inline]
pub(crate) fn complex4_load_f32(array: &[Complex<f32>], idx: usize) -> float32x4x2_t {
    unsafe {
        float32x4x2_t(
            vld1q_f32(array.get_unchecked(idx..).as_ptr().cast()),
            vld1q_f32(array.get_unchecked(idx + 2..).as_ptr().cast()),
        )
    }
}

#[inline]
pub(crate) fn complex4_store_f32(array: &mut [Complex<f32>], idx: usize, v: float32x4x2_t) {
    unsafe {
        vst1q_f32(array.get_unchecked_mut(idx..).as_mut_ptr().cast(), v.0);
        vst1q_f32(array.get_unchecked_mut(idx + 2..).as_mut_ptr().cast(), v.1);
    }
}

#[inline]
fn complex4_load_f64(array: &[Complex<f64>], idx: usize) -> float64x2x4_t {
    unsafe {
        float64x2x4_t(
            vld1q_f64(array.get_unchecked(idx..).as_ptr().cast()),
            vld1q_f64(array.get_unchecked(idx + 1..).as_ptr().cast()),
            vld1q_f64(array.get_unchecked(idx + 2..).as_ptr().cast()),
            vld1q_f64(array.get_unchecked(idx + 3..).as_ptr().cast()),
        )
    }
}

#[inline]
fn complex4_store_f64(array: &mut [Complex<f64>], idx: usize, v: float64x2x4_t) {
    unsafe {
        vst1q_f64(array.get_unchecked_mut(idx..).as_mut_ptr().cast(), v.0);
        vst1q_f64(array.get_unchecked_mut(idx + 1..).as_mut_ptr().cast(), v.1);
        vst1q_f64(array.get_unchecked_mut(idx + 2..).as_mut_ptr().cast(), v.2);
        vst1q_f64(array.get_unchecked_mut(idx + 3..).as_mut_ptr().cast(), v.3);
    }
}

pub(crate) fn neon_bitreversed_transpose_f32_radix4(
    height: usize,
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
) {
    let width = input.len() / height;
    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }
    const WIDTH: usize = 4;

    const HEIGHT: usize = 4;

    let width_bits = width.trailing_zeros();
    let d_bits = WIDTH.trailing_zeros();

    assert!(width_bits % d_bits == 0);
    let rev_digits = width_bits / d_bits;
    let strided_width = width / WIDTH;
    let strided_height = height / HEIGHT;

    for x in 0..strided_width {
        let x_rev = [
            reverse_bits::<WIDTH>(WIDTH * x, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 1, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 2, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 3, rev_digits) * height,
        ];

        for y in 0..strided_height {
            let base_input_idx = (WIDTH * x) + y * HEIGHT * width;
            let rows = [
                complex4_load_f32(input, base_input_idx),
                complex4_load_f32(input, base_input_idx + width),
                complex4_load_f32(input, base_input_idx + width * 2),
                complex4_load_f32(input, base_input_idx + width * 3),
            ];
            let transposed = transpose_f32x2_4x4(rows[0], rows[1], rows[2], rows[3]);

            complex4_store_f32(output, HEIGHT * y + x_rev[0], transposed.0);
            complex4_store_f32(output, HEIGHT * y + x_rev[1], transposed.1);
            complex4_store_f32(output, HEIGHT * y + x_rev[2], transposed.2);
            complex4_store_f32(output, HEIGHT * y + x_rev[3], transposed.3);
        }
    }
}

pub(crate) fn neon_bitreversed_transpose_f64_radix4(
    height: usize,
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
) {
    let width = input.len() / height;
    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }
    const WIDTH: usize = 4;
    const HEIGHT: usize = 4;

    let width_bits = width.trailing_zeros();
    let d_bits = WIDTH.trailing_zeros();

    assert!(width_bits % d_bits == 0);
    let rev_digits = width_bits / d_bits;
    let strided_width = width / WIDTH;
    let strided_height = height / HEIGHT;

    for x in 0..strided_width {
        let x_rev = [
            reverse_bits::<WIDTH>(WIDTH * x, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 1, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 2, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 3, rev_digits) * height,
        ];

        for y in 0..strided_height {
            let base_input_idx = (WIDTH * x) + y * HEIGHT * width;
            let rows = [
                complex4_load_f64(input, base_input_idx),
                complex4_load_f64(input, base_input_idx + width),
                complex4_load_f64(input, base_input_idx + width * 2),
                complex4_load_f64(input, base_input_idx + width * 3),
            ];
            let transposed = neon_transpose_f64x2_4x4_impl(rows[0], rows[1], rows[2], rows[3]);

            complex4_store_f64(output, HEIGHT * y + x_rev[0], transposed.0);
            complex4_store_f64(output, HEIGHT * y + x_rev[1], transposed.1);
            complex4_store_f64(output, HEIGHT * y + x_rev[2], transposed.2);
            complex4_store_f64(output, HEIGHT * y + x_rev[3], transposed.3);
        }
    }
}

pub(crate) struct NeonRadix4<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    direction: FftDirection,
    base_len: usize,
    base_fft: Box<dyn CompositeFftExecutor<T> + Send + Sync>,
}

impl<T: Default + Clone + Radix4Twiddles + AlgorithmFactory<T> + FftTrigonometry + 'static + Float>
    NeonRadix4<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonRadix4<T>, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");
        // assert_eq!(size.trailing_zeros() % 2, 0, "Radix-4 requires power of 4");

        let exponent = size.trailing_zeros();
        let base_fft = match exponent {
            0 => T::butterfly1(fft_direction)?,
            1 => T::butterfly2(fft_direction)?,
            2 => T::butterfly4(fft_direction)?,
            3 => T::butterfly8(fft_direction)?,
            4 => T::butterfly16(fft_direction)?,
            _ => {
                if exponent % 2 == 1 {
                    T::butterfly32(fft_direction)?
                } else {
                    match T::butterfly64(fft_direction) {
                        None => T::butterfly16(fft_direction)?,
                        Some(v) => v,
                    }
                }
            }
        };

        let twiddles = create_neon_twiddles::<T, 4>(base_fft.length(), size, fft_direction)?;

        Ok(NeonRadix4 {
            execution_length: size,
            twiddles,
            direction: fft_direction,
            base_len: base_fft.length(),
            base_fft,
        })
    }
}

impl FftExecutor<f64> for NeonRadix4<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let v_i_multiplier = unsafe {
            vreinterpretq_u64_f64(match self.direction {
                FftDirection::Inverse => vld1q_f64([-0.0, 0.0].as_ptr()),
                FftDirection::Forward => vld1q_f64([0.0, -0.0].as_ptr()),
            })
        };

        let mut scratch = try_vec![Complex::default(); self.execution_length];

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // bit reversal first
            neon_bitreversed_transpose_f64_radix4(self.base_len, chunk, &mut scratch);

            self.base_fft.execute_out_of_place(&scratch, chunk)?;

            let mut len = self.base_len;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 4;
                    let quarter = len / 4;

                    for data in chunk.chunks_exact_mut(len) {
                        for j in 0..quarter {
                            let a = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                            let b = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(3 * j..).as_ptr().cast()),
                            );
                            let c = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 2 * quarter..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(3 * j + 1..).as_ptr().cast()),
                            );
                            let d = vfcmulq_f64(
                                vld1q_f64(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                                vld1q_f64(m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast()),
                            );

                            // radix-4 butterfly
                            let t0 = vaddq_f64(a, c);
                            let t1 = vsubq_f64(a, c);
                            let t2 = vaddq_f64(b, d);
                            let mut t3 = vsubq_f64(b, d);
                            t3 = vreinterpretq_f64_u64(veorq_u64(
                                vreinterpretq_u64_f64(vcombine_f64(
                                    vget_high_f64(t3),
                                    vget_low_f64(t3),
                                )),
                                v_i_multiplier,
                            ));

                            vst1q_f64(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                vaddq_f64(t0, t2),
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                vaddq_f64(t1, t3),
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 2 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                vsubq_f64(t0, t2),
                            );
                            vst1q_f64(
                                data.get_unchecked_mut(j + 3 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                vsubq_f64(t1, t3),
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 3..];
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

impl FftExecutor<f32> for NeonRadix4<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let v_i_multiplier = unsafe {
            vreinterpretq_u32_f32(match self.direction {
                FftDirection::Inverse => vld1q_f32([-0.0, 0.0, -0.0, 0.0].as_ptr()),
                FftDirection::Forward => vld1q_f32([0.0, -0.0, 0.0, -0.0].as_ptr()),
            })
        };

        let mut scratch = try_vec![Complex::default(); self.execution_length];

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // bit reversal first
            neon_bitreversed_transpose_f32_radix4(self.base_len, chunk, &mut scratch);

            self.base_fft.execute_out_of_place(&scratch, chunk)?;

            let mut len = self.base_len;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 4;
                    let quarter = len / 4;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < quarter {
                            let a = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast());
                            let tw2 =
                                vld1q_f32(m_twiddles.get_unchecked(3 * j + 4..).as_ptr().cast());

                            let b = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                tw0,
                            );
                            let c = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * quarter..).as_ptr().cast()),
                                tw1,
                            );
                            let d = vfcmulq_f32(
                                vld1q_f32(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                                tw2,
                            );

                            // radix-4 butterfly
                            let t0 = vaddq_f32(a, c);
                            let t1 = vsubq_f32(a, c);
                            let t2 = vaddq_f32(b, d);
                            let mut t3 = vsubq_f32(b, d);
                            t3 = vreinterpretq_f32_u32(veorq_u32(
                                vrev64q_u32(vreinterpretq_u32_f32(t3)),
                                v_i_multiplier,
                            ));

                            vst1q_f32(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                vaddq_f32(t0, t2),
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                vaddq_f32(t1, t3),
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                vsubq_f32(t0, t2),
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 3 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                vsubq_f32(t1, t3),
                            );

                            j += 2;
                        }

                        for j in j..quarter {
                            let a = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw = vld1q_f32(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());

                            let bc = vfcmulq_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 2 * quarter..).as_ptr().cast()),
                                ),
                                tw,
                            );
                            let d = vfcmul_f32(
                                vld1_f32(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                                vld1_f32(m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast()),
                            );

                            let b = vget_low_f32(bc);
                            let c = vget_high_f32(bc);

                            // radix-4 butterfly
                            let t0 = vadd_f32(a, c);
                            let t1 = vsub_f32(a, c);
                            let t2 = vadd_f32(b, d);
                            let mut t3 = vsub_f32(b, d);
                            t3 = vreinterpret_f32_u32(veor_u32(
                                vrev64_u32(vreinterpret_u32_f32(t3)),
                                vget_low_u32(v_i_multiplier),
                            ));

                            vst1_f32(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                vadd_f32(t0, t2),
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                vadd_f32(t1, t3),
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 2 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                vsub_f32(t0, t2),
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 3 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                vsub_f32(t1, t3),
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 3..];
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

    test_radix!(test_neon_radix4, f32, NeonRadix4, 11, 2, 1e-2);
    test_radix!(test_neon_radix4_f64, f64, NeonRadix4, 11, 2, 1e-8);
}
