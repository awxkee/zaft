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
use crate::avx::mixed::{AvxStoreD, AvxStoreF};
use crate::avx::transpose::{avx_transpose_f32x2_4x4_impl, avx_transpose_f64x2_4x4_impl};
use crate::avx::util::{
    _mm_fcmul_ps, _mm256_create_ps, _mm256_fcmul_pd, _mm256_fcmul_ps, create_avx_twiddles_d,
    create_avx_twiddles_f, shuffle,
};
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::util::{reverse_bits, validate_oof_sizes, validate_scratch};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::Zero;
use std::arch::x86_64::*;
use std::sync::Arc;

pub(crate) struct AvxFmaRadix4d {
    twiddles: Vec<AvxStoreD>,
    execution_length: usize,
    direction: FftDirection,
    base_len: usize,
    base_fft: Arc<dyn FftExecutor<f64> + Send + Sync>,
}

pub(crate) struct AvxFmaRadix4f {
    twiddles: Vec<AvxStoreF>,
    execution_length: usize,
    direction: FftDirection,
    base_len: usize,
    base_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn complex4_load_f32(array: &[Complex<f32>], idx: usize) -> __m256 {
    unsafe { _mm256_loadu_ps(array.get_unchecked(idx..).as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn complex4_store_f32(array: &mut [Complex<f32>], idx: usize, v: __m256) {
    unsafe {
        _mm256_storeu_ps(array.get_unchecked_mut(idx..).as_mut_ptr().cast(), v);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn complex4_load_f64(array: &[Complex<f64>], idx: usize) -> (__m256d, __m256d) {
    unsafe {
        (
            _mm256_loadu_pd(array.get_unchecked(idx..).as_ptr().cast()),
            _mm256_loadu_pd(array.get_unchecked(idx + 2..).as_ptr().cast()),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn complex4_store_f64(array: &mut [Complex<f64>], idx: usize, v: (__m256d, __m256d)) {
    unsafe {
        _mm256_storeu_pd(array.get_unchecked_mut(idx..).as_mut_ptr().cast(), v.0);
        _mm256_storeu_pd(array.get_unchecked_mut(idx + 2..).as_mut_ptr().cast(), v.1);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn avx_bitreversed_transpose_f32_radix4(
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

    assert_eq!(
        width_bits % d_bits,
        0,
        "Radix-4 bit transpose assertion failed on input size {}",
        input.len()
    );
    let rev_digits = width_bits / d_bits;
    let strided_width = width / WIDTH;
    let strided_height = height / HEIGHT;

    if strided_width == 0 {
        output.copy_from_slice(input);
        return;
    }

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
            let transposed = avx_transpose_f32x2_4x4_impl(rows[0], rows[1], rows[2], rows[3]);

            complex4_store_f32(output, HEIGHT * y + x_rev[0], transposed.0);
            complex4_store_f32(output, HEIGHT * y + x_rev[1], transposed.1);
            complex4_store_f32(output, HEIGHT * y + x_rev[2], transposed.2);
            complex4_store_f32(output, HEIGHT * y + x_rev[3], transposed.3);
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn avx_bitreversed_transpose_f64_radix4(
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

    assert_eq!(
        width_bits % d_bits,
        0,
        "Radix-4 bit transpose assertion failed on input size {}",
        input.len()
    );
    let rev_digits = width_bits / d_bits;
    let strided_width = width / WIDTH;
    let strided_height = height / HEIGHT;

    if strided_width == 0 {
        output.copy_from_slice(input);
        return;
    }

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
            let transposed = avx_transpose_f64x2_4x4_impl(rows[0], rows[1], rows[2], rows[3]);

            complex4_store_f64(output, HEIGHT * y + x_rev[0], transposed.0);
            complex4_store_f64(output, HEIGHT * y + x_rev[1], transposed.1);
            complex4_store_f64(output, HEIGHT * y + x_rev[2], transposed.2);
            complex4_store_f64(output, HEIGHT * y + x_rev[3], transposed.3);
        }
    }
}

impl AvxFmaRadix4d {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix4d, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");

        let exponent = size.trailing_zeros();
        let base_fft = match exponent {
            0 => f64::butterfly1(fft_direction)?,
            1 => f64::butterfly2(fft_direction)?,
            2 => f64::butterfly4(fft_direction)?,
            3 => f64::butterfly8(fft_direction)?,
            4 => f64::butterfly16(fft_direction)?,
            _ => {
                if exponent % 2 == 1 {
                    if exponent >= 9 {
                        f64::butterfly512(fft_direction).map_or_else(
                            || {
                                f64::butterfly128(fft_direction)
                                    .map_or_else(|| f64::butterfly32(fft_direction), Ok)
                            },
                            Ok,
                        )?
                    } else if exponent >= 7 {
                        f64::butterfly128(fft_direction)
                            .map_or_else(|| f64::butterfly32(fft_direction), Ok)?
                    } else {
                        f64::butterfly32(fft_direction)?
                    }
                } else {
                    #[allow(clippy::collapsible_else_if)]
                    if exponent >= 10 {
                        f64::butterfly1024(fft_direction).map_or_else(
                            || {
                                f64::butterfly256(fft_direction).map_or_else(
                                    || {
                                        f64::butterfly64(fft_direction)
                                            .map_or_else(|| f64::butterfly16(fft_direction), Ok)
                                    },
                                    Ok,
                                )
                            },
                            Ok,
                        )?
                    } else if exponent >= 8 {
                        f64::butterfly256(fft_direction).map_or_else(
                            || {
                                f64::butterfly64(fft_direction)
                                    .map_or_else(|| f64::butterfly16(fft_direction), Ok)
                            },
                            Ok,
                        )?
                    } else {
                        f64::butterfly64(fft_direction)
                            .map_or_else(|| f64::butterfly16(fft_direction), Ok)?
                    }
                }
            }
        };

        let twiddles =
            unsafe { create_avx_twiddles_d::<4>(base_fft.length(), size, fft_direction) };

        Ok(AvxFmaRadix4d {
            execution_length: size,
            twiddles,
            direction: fft_direction,
            base_len: base_fft.length(),
            base_fft,
        })
    }
}

impl AvxFmaRadix4f {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix4f, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");

        let exponent = size.trailing_zeros();
        let base_fft = match exponent {
            0 => f32::butterfly1(fft_direction)?,
            1 => f32::butterfly2(fft_direction)?,
            2 => f32::butterfly4(fft_direction)?,
            3 => f32::butterfly8(fft_direction)?,
            4 => f32::butterfly16(fft_direction)?,
            _ => {
                if exponent % 2 == 1 {
                    if exponent >= 9 {
                        f32::butterfly512(fft_direction).map_or_else(
                            || {
                                f32::butterfly128(fft_direction)
                                    .map_or_else(|| f32::butterfly32(fft_direction), Ok)
                            },
                            Ok,
                        )?
                    } else if exponent >= 7 {
                        f32::butterfly128(fft_direction)
                            .map_or_else(|| f32::butterfly32(fft_direction), Ok)?
                    } else {
                        f32::butterfly32(fft_direction)?
                    }
                } else {
                    #[allow(clippy::collapsible_else_if)]
                    if exponent >= 10 {
                        f32::butterfly1024(fft_direction).map_or_else(
                            || {
                                f32::butterfly256(fft_direction).map_or_else(
                                    || {
                                        f32::butterfly64(fft_direction)
                                            .map_or_else(|| f32::butterfly16(fft_direction), Ok)
                                    },
                                    Ok,
                                )
                            },
                            Ok,
                        )?
                    } else if exponent >= 8 {
                        f32::butterfly256(fft_direction).map_or_else(
                            || {
                                f32::butterfly64(fft_direction)
                                    .map_or_else(|| f32::butterfly16(fft_direction), Ok)
                            },
                            Ok,
                        )?
                    } else {
                        f32::butterfly64(fft_direction)
                            .map_or_else(|| f32::butterfly16(fft_direction), Ok)?
                    }
                }
            }
        };

        let twiddles =
            unsafe { create_avx_twiddles_f::<4>(base_fft.length(), size, fft_direction) };

        Ok(AvxFmaRadix4f {
            execution_length: size,
            twiddles,
            direction: fft_direction,
            base_len: base_fft.length(),
            base_fft,
        })
    }
}

impl AvxFmaRadix4d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn base_run(&self, chunk: &mut [Complex<f64>]) {
        unsafe {
            let v_i_multiplier = match self.direction {
                FftDirection::Forward => _mm256_loadu_pd([-0.0f64, 0.0, -0.0f64, 0.0].as_ptr()),
                FftDirection::Inverse => _mm256_loadu_pd([0.0f64, -0.0, 0.0f64, -0.0].as_ptr()),
            };

            let mut len = self.base_len;

            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                len *= 4;
                let quarter = len / 4;

                let mut last_twiddle = 0usize;

                for data in chunk.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    let mut tw_idx = 0usize;

                    while j + 4 <= quarter {
                        let a0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());
                        let a1 = _mm256_loadu_pd(data.get_unchecked(j + 2..).as_ptr().cast());

                        let tw0_0 = m_twiddles.get_unchecked(tw_idx).v;
                        let tw1_0 = m_twiddles.get_unchecked(tw_idx + 1).v;
                        let tw2_0 = m_twiddles.get_unchecked(tw_idx + 2).v;

                        let tw0_1 = m_twiddles.get_unchecked(tw_idx + 3).v;
                        let tw1_1 = m_twiddles.get_unchecked(tw_idx + 4).v;
                        let tw2_1 = m_twiddles.get_unchecked(tw_idx + 5).v;

                        let b0 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + quarter..).as_ptr().cast()),
                            tw0_0,
                        );
                        let c0 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + 2 * quarter..).as_ptr().cast()),
                            tw1_0,
                        );
                        let d0 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                            tw2_0,
                        );

                        let b1 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + quarter + 2..).as_ptr().cast()),
                            tw0_1,
                        );
                        let c1 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(
                                data.get_unchecked(j + 2 * quarter + 2..).as_ptr().cast(),
                            ),
                            tw1_1,
                        );
                        let d1 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(
                                data.get_unchecked(j + 3 * quarter + 2..).as_ptr().cast(),
                            ),
                            tw2_1,
                        );

                        // radix-4 butterfly
                        let t0_0 = _mm256_add_pd(a0, c0);
                        let t1_0 = _mm256_sub_pd(a0, c0);
                        let t2_0 = _mm256_add_pd(b0, d0);
                        let mut t3_0 = _mm256_sub_pd(b0, d0);
                        t3_0 = _mm256_xor_pd(t3_0, v_i_multiplier);
                        t3_0 = _mm256_shuffle_pd::<0b0101>(t3_0, t3_0);

                        let t0_1 = _mm256_add_pd(a1, c1);
                        let t1_1 = _mm256_sub_pd(a1, c1);
                        let t2_1 = _mm256_add_pd(b1, d1);
                        let mut t3_1 = _mm256_sub_pd(b1, d1);
                        t3_1 = _mm256_xor_pd(t3_1, v_i_multiplier);
                        t3_1 = _mm256_shuffle_pd::<0b0101>(t3_1, t3_1);

                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                            _mm256_add_pd(t0_0, t2_0),
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                            _mm256_add_pd(t1_0, t3_0),
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 2 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            _mm256_sub_pd(t0_0, t2_0),
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 3 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            _mm256_sub_pd(t1_0, t3_0),
                        );

                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 2..).as_mut_ptr().cast(),
                            _mm256_add_pd(t0_1, t2_1),
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + quarter + 2..)
                                .as_mut_ptr()
                                .cast(),
                            _mm256_add_pd(t1_1, t3_1),
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 2 * quarter + 2..)
                                .as_mut_ptr()
                                .cast(),
                            _mm256_sub_pd(t0_1, t2_1),
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 3 * quarter + 2..)
                                .as_mut_ptr()
                                .cast(),
                            _mm256_sub_pd(t1_1, t3_1),
                        );

                        j += 4;
                        tw_idx += 6;
                    }

                    while j + 2 <= quarter {
                        let a = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                        let tw0 = m_twiddles.get_unchecked(tw_idx).v;
                        let tw1 = m_twiddles.get_unchecked(tw_idx + 1).v;
                        let tw2 = m_twiddles.get_unchecked(tw_idx + 2).v;

                        let b = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + quarter..).as_ptr().cast()),
                            tw0,
                        );
                        let c = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + 2 * quarter..).as_ptr().cast()),
                            tw1,
                        );
                        let d = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                            tw2,
                        );

                        // radix-4 butterfly
                        let t0 = _mm256_add_pd(a, c);
                        let t1 = _mm256_sub_pd(a, c);
                        let t2 = _mm256_add_pd(b, d);
                        let mut t3 = _mm256_sub_pd(b, d);
                        t3 = _mm256_xor_pd(t3, v_i_multiplier);
                        t3 = _mm256_shuffle_pd::<0b0101>(t3, t3);

                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                            _mm256_add_pd(t0, t2),
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                            _mm256_add_pd(t1, t3),
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 2 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            _mm256_sub_pd(t0, t2),
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 3 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            _mm256_sub_pd(t1, t3),
                        );

                        j += 2;
                        tw_idx += 3;
                    }

                    last_twiddle = tw_idx;
                }

                m_twiddles = &m_twiddles[last_twiddle..];
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_f64(
        &self,
        in_place: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let scratch = validate_scratch!(scratch, self.scratch_length());

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // bit reversal first
            avx_bitreversed_transpose_f64_radix4(self.base_len, chunk, scratch);
            self.base_fft.execute_out_of_place(scratch, chunk)?;
            self.base_run(chunk);
        }

        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_oof_f64(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        validate_oof_sizes!(src, dst, self.execution_length);

        for (dst, src) in dst
            .chunks_exact_mut(self.execution_length)
            .zip(src.chunks_exact(self.execution_length))
        {
            // Digit-reversal permutation
            avx_bitreversed_transpose_f64_radix4(self.base_len, src, dst);
            self.base_fft.execute(dst)?;
            self.base_run(dst);
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix4d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        let mut scratch = try_vec![Complex::zero(); self.scratch_length()];
        unsafe { self.execute_f64(in_place, &mut scratch) }
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place, scratch) }
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_oof_f64(src, dst) }
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_oof_f64(src, dst) }
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<f64>],
        dst: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_with_scratch(src, dst, scratch)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_length(&self) -> usize {
        self.execution_length
    }

    fn out_of_place_scratch_length(&self) -> usize {
        0
    }

    fn destructive_scratch_length(&self) -> usize {
        0
    }
}

impl AvxFmaRadix4f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn base_run(&self, chunk: &mut [Complex<f32>]) {
        unsafe {
            let v_i_multiplier = match self.direction {
                FftDirection::Forward => {
                    _mm256_loadu_ps([-0.0f32, 0.0, -0.0, 0.0, -0.0f32, 0.0, -0.0, 0.0].as_ptr())
                }
                FftDirection::Inverse => {
                    _mm256_loadu_ps([0.0f32, -0.0, 0.0, -0.0, 0.0f32, -0.0, 0.0, -0.0].as_ptr())
                }
            };

            let mut len = self.base_len;

            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                len *= 4;
                let quarter = len / 4;

                let mut last_twiddle = 0usize;

                for data in chunk.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    let mut tw_idx = 0usize;

                    macro_rules! make_block {
                        ($data: expr, $twiddles: expr, $quarter: expr, $j: expr, $start: expr, $tw_start: expr) => {{
                            let a0 =
                                _mm256_loadu_ps(data.get_unchecked(j + $start..).as_ptr().cast());

                            let tw0 = $twiddles.get_unchecked($tw_start).v;
                            let tw1 = $twiddles.get_unchecked($tw_start + 1).v;
                            let tw2 = $twiddles.get_unchecked($tw_start + 2).v;

                            let rk1 = _mm256_loadu_ps(
                                $data.get_unchecked(j + $quarter + $start..).as_ptr().cast(),
                            );
                            let rk2 = _mm256_loadu_ps(
                                $data
                                    .get_unchecked(j + 2 * $quarter + $start..)
                                    .as_ptr()
                                    .cast(),
                            );
                            let rk3 = _mm256_loadu_ps(
                                $data
                                    .get_unchecked(j + 3 * $quarter + $start..)
                                    .as_ptr()
                                    .cast(),
                            );

                            let b0 = _mm256_fcmul_ps(rk1, tw0);
                            let c0 = _mm256_fcmul_ps(rk2, tw1);
                            let d0 = _mm256_fcmul_ps(rk3, tw2);

                            // radix-4 butterfly
                            let q0t0 = _mm256_add_ps(a0, c0);
                            let q0t1 = _mm256_sub_ps(a0, c0);
                            let q0t2 = _mm256_add_ps(b0, d0);
                            let mut q0t3 = _mm256_sub_ps(b0, d0);
                            const SH: i32 = shuffle(2, 3, 0, 1);
                            q0t3 = _mm256_xor_ps(q0t3, v_i_multiplier);
                            q0t3 = _mm256_shuffle_ps::<SH>(q0t3, q0t3);

                            let y0 = _mm256_add_ps(q0t0, q0t2);
                            let y1 = _mm256_add_ps(q0t1, q0t3);
                            let y2 = _mm256_sub_ps(q0t0, q0t2);
                            let y3 = _mm256_sub_ps(q0t1, q0t3);
                            (y0, y1, y2, y3)
                        }};
                    }

                    while j + 8 <= quarter {
                        let (y0, y1, y2, y3) = make_block!(data, m_twiddles, quarter, j, 0, tw_idx);
                        let (y4, y5, y6, y7) =
                            make_block!(data, m_twiddles, quarter, j, 4, tw_idx + 3);

                        _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                            y1,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 2 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            y2,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 3 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            y3,
                        );

                        _mm256_storeu_ps(data.get_unchecked_mut(j + 4..).as_mut_ptr().cast(), y4);
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + quarter + 4..)
                                .as_mut_ptr()
                                .cast(),
                            y5,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 2 * quarter + 4..)
                                .as_mut_ptr()
                                .cast(),
                            y6,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 3 * quarter + 4..)
                                .as_mut_ptr()
                                .cast(),
                            y7,
                        );

                        j += 8;
                        tw_idx += 6;
                    }

                    while j + 4 <= quarter {
                        let (y0, y1, y2, y3) = make_block!(data, m_twiddles, quarter, j, 0, tw_idx);
                        _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                            y1,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 2 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            y2,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 3 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            y3,
                        );

                        j += 4;
                        tw_idx += 3;
                    }

                    while j + 2 <= quarter {
                        let a0 = _mm_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                        let tw0 = *m_twiddles.get_unchecked(tw_idx);
                        let tw1 = *m_twiddles.get_unchecked(tw_idx + 1);
                        let tw2 = *m_twiddles.get_unchecked(tw_idx + 2);

                        let rk1 = _mm_loadu_ps(data.get_unchecked(j + quarter..).as_ptr().cast());
                        let rk2 =
                            _mm_loadu_ps(data.get_unchecked(j + 2 * quarter..).as_ptr().cast());
                        let rk3 =
                            _mm_loadu_ps(data.get_unchecked(j + 3 * quarter..).as_ptr().cast());

                        const LO_LO: i32 = 0b0010_0000;
                        let b0c0 = _mm256_fcmul_ps(
                            _mm256_create_ps(rk1, rk2),
                            _mm256_permute2f128_ps::<LO_LO>(tw0.v, tw1.v),
                        );
                        let d0 = _mm_fcmul_ps(rk3, _mm256_castps256_ps128(tw2.v));
                        let b0 = _mm256_castps256_ps128(b0c0);
                        let c0 = _mm256_extractf128_ps::<1>(b0c0);

                        // radix-4 butterfly
                        let q0t0 = _mm_add_ps(a0, c0);
                        let q0t1 = _mm_sub_ps(a0, c0);
                        let q0t2 = _mm_add_ps(b0, d0);
                        let mut q0t3 = _mm_sub_ps(b0, d0);
                        const SH: i32 = shuffle(2, 3, 0, 1);
                        q0t3 = _mm_xor_ps(q0t3, _mm256_castps256_ps128(v_i_multiplier));
                        q0t3 = _mm_shuffle_ps::<SH>(q0t3, q0t3);

                        let y0 = _mm_add_ps(q0t0, q0t2);
                        let y1 = _mm_add_ps(q0t1, q0t3);
                        let y2 = _mm_sub_ps(q0t0, q0t2);
                        let y3 = _mm_sub_ps(q0t1, q0t3);

                        _mm_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                            y1,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 2 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            y2,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 3 * quarter..)
                                .as_mut_ptr()
                                .cast(),
                            y3,
                        );

                        j += 2;
                        tw_idx += 3;
                    }

                    last_twiddle = tw_idx;
                }

                m_twiddles = &m_twiddles[last_twiddle..];
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_f32(
        &self,
        in_place: &mut [Complex<f32>],
        scratch: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let scratch = validate_scratch!(scratch, self.scratch_length());

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // bit reversal first
            avx_bitreversed_transpose_f32_radix4(self.base_len, chunk, scratch);
            self.base_fft.execute_out_of_place(scratch, chunk)?;
            self.base_run(chunk);
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_oof_f32(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        validate_oof_sizes!(src, dst, self.execution_length);

        for (dst, src) in dst
            .chunks_exact_mut(self.execution_length)
            .zip(src.chunks_exact(self.execution_length))
        {
            // Digit-reversal permutation
            avx_bitreversed_transpose_f32_radix4(self.base_len, src, dst);
            self.base_fft.execute(dst)?;
            self.base_run(dst);
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix4f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        let mut scratch = try_vec![Complex::zero(); self.scratch_length()];
        unsafe { self.execute_f32(in_place, &mut scratch) }
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<f32>],
        scratch: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place, scratch) }
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_oof_f32(src, dst) }
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
        _: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_oof_f32(src, dst) }
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<f32>],
        dst: &mut [Complex<f32>],
        scratch: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_with_scratch(src, dst, scratch)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_length(&self) -> usize {
        self.execution_length
    }

    fn out_of_place_scratch_length(&self) -> usize {
        0
    }

    fn destructive_scratch_length(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::{test_avx_radix, test_avx_radix_fast};

    test_avx_radix!(test_avx_radix4, f32, AvxFmaRadix4f, 13, 2, 1e-2);
    test_avx_radix_fast!(
        test_avx_radix4_fast,
        f32,
        AvxFmaRadix4f,
        Radix4,
        13,
        2,
        1e-2
    );
    test_avx_radix_fast!(test_avx_radix4_f64, f64, AvxFmaRadix4d, Radix4, 13, 2, 1e-8);
}
