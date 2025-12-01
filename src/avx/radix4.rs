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
use crate::avx::transpose::{avx_transpose_f32x2_4x4_impl, avx_transpose_f64x2_4x4_impl};
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_fcmul_pd, _mm_fcmul_ps, _mm_unpackhi_ps64,
    _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps, _mm256_fcmul_pd, _mm256_fcmul_ps,
    create_avx4_twiddles, shuffle,
};
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::radix4::Radix4Twiddles;
use crate::traits::FftTrigonometry;
use crate::util::reverse_bits;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;
use std::sync::Arc;

pub(crate) struct AvxFmaRadix4<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    direction: FftDirection,
    base_len: usize,
    base_fft: Arc<dyn CompositeFftExecutor<T> + Send + Sync>,
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

impl<T: Default + Clone + Radix4Twiddles + AlgorithmFactory<T> + FftTrigonometry + Float + 'static>
    AvxFmaRadix4<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix4<T>, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");

        let exponent = size.trailing_zeros();
        let base_fft = match exponent {
            0 => T::butterfly1(fft_direction)?,
            1 => T::butterfly2(fft_direction)?,
            2 => T::butterfly4(fft_direction)?,
            3 => T::butterfly8(fft_direction)?,
            4 => T::butterfly16(fft_direction)?,
            _ => {
                if exponent % 2 == 1 {
                    if exponent >= 7 {
                        T::butterfly128(fft_direction)
                            .map_or_else(|| T::butterfly32(fft_direction), Ok)?
                    } else {
                        T::butterfly32(fft_direction)?
                    }
                } else {
                    #[allow(clippy::collapsible_else_if)]
                    if exponent >= 8 {
                        T::butterfly256(fft_direction).map_or_else(
                            || {
                                T::butterfly64(fft_direction)
                                    .map_or_else(|| T::butterfly16(fft_direction), Ok)
                            },
                            Ok,
                        )?
                    } else {
                        T::butterfly64(fft_direction)
                            .map_or_else(|| T::butterfly16(fft_direction), Ok)?
                    }
                }
            }
        };

        let twiddles = create_avx4_twiddles::<T, 4>(base_fft.length(), size, fft_direction)?;

        Ok(AvxFmaRadix4 {
            execution_length: size,
            twiddles,
            direction: fft_direction,
            base_len: base_fft.length(),
            base_fft,
        })
    }
}

impl AvxFmaRadix4<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let v_i_multiplier = unsafe {
            match self.direction {
                FftDirection::Inverse => _mm256_loadu_pd([-0.0f64, 0.0, -0.0f64, 0.0].as_ptr()),
                FftDirection::Forward => _mm256_loadu_pd([0.0f64, -0.0, 0.0f64, -0.0].as_ptr()),
            }
        };

        let mut scratch = try_vec![Complex::default(); self.execution_length];

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // bit reversal first
            avx_bitreversed_transpose_f64_radix4(self.base_len, chunk, &mut scratch);

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

                        while j + 4 < quarter {
                            let a0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());
                            let a1 = _mm256_loadu_pd(data.get_unchecked(j + 2..).as_ptr().cast());

                            let tw0_0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());
                            let tw1_0 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast(),
                            );
                            let tw2_0 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(3 * j + 4..).as_ptr().cast(),
                            );

                            let tw0_1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(3 * j + 6..).as_ptr().cast(),
                            );
                            let tw1_1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(3 * j + 8..).as_ptr().cast(),
                            );
                            let tw2_1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(3 * j + 10..).as_ptr().cast(),
                            );

                            let b0 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                tw0_0,
                            );
                            let c0 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 2 * quarter..).as_ptr().cast(),
                                ),
                                tw1_0,
                            );
                            let d0 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 3 * quarter..).as_ptr().cast(),
                                ),
                                tw2_0,
                            );

                            let b1 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + quarter + 2..).as_ptr().cast(),
                                ),
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
                            t3_0 = _mm256_xor_pd(_mm256_permute_pd::<0b0101>(t3_0), v_i_multiplier);

                            let t0_1 = _mm256_add_pd(a1, c1);
                            let t1_1 = _mm256_sub_pd(a1, c1);
                            let t2_1 = _mm256_add_pd(b1, d1);
                            let mut t3_1 = _mm256_sub_pd(b1, d1);
                            t3_1 = _mm256_xor_pd(_mm256_permute_pd::<0b0101>(t3_1), v_i_multiplier);

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
                        }

                        while j + 2 < quarter {
                            let a = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(3 * j + 4..).as_ptr().cast(),
                            );

                            let b = _mm256_fcmul_pd(
                                _mm256_loadu_pd(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                tw0,
                            );
                            let c = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 2 * quarter..).as_ptr().cast(),
                                ),
                                tw1,
                            );
                            let d = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 3 * quarter..).as_ptr().cast(),
                                ),
                                tw2,
                            );

                            // radix-4 butterfly
                            let t0 = _mm256_add_pd(a, c);
                            let t1 = _mm256_sub_pd(a, c);
                            let t2 = _mm256_add_pd(b, d);
                            let mut t3 = _mm256_sub_pd(b, d);
                            t3 = _mm256_xor_pd(_mm256_permute_pd::<0b0101>(t3), v_i_multiplier);

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
                        }
                        for j in j..quarter {
                            let a = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());

                            let bc = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 2 * quarter..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let d = _mm_fcmul_pd(
                                _mm_loadu_pd(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                                _mm_loadu_pd(m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast()),
                            );

                            // radix-4 butterfly
                            let b = _mm256_castpd256_pd128(bc);
                            let c = _mm256_extractf128_pd::<1>(bc);
                            let t0 = _mm_add_pd(a, c);
                            let t1 = _mm_sub_pd(a, c);
                            let t2 = _mm_add_pd(b, d);
                            let mut t3 = _mm_sub_pd(b, d);
                            t3 = _mm_xor_pd(
                                _mm_shuffle_pd::<0b01>(t3, t3),
                                _mm256_castpd256_pd128(v_i_multiplier),
                            );

                            _mm_storeu_pd(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                _mm_add_pd(t0, t2),
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                _mm_add_pd(t1, t3),
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 2 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                _mm_sub_pd(t0, t2),
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 3 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                _mm_sub_pd(t1, t3),
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 3..];
                }
            }
        }

        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix4<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }
    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}

impl AvxFmaRadix4<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let v_i_multiplier = unsafe {
            match self.direction {
                FftDirection::Inverse => {
                    _mm256_loadu_ps([-0.0f32, 0.0, -0.0, 0.0, -0.0f32, 0.0, -0.0, 0.0].as_ptr())
                }
                FftDirection::Forward => {
                    _mm256_loadu_ps([0.0f32, -0.0, 0.0, -0.0, 0.0f32, -0.0, 0.0, -0.0].as_ptr())
                }
            }
        };

        let mut scratch = try_vec![Complex::default(); self.execution_length];

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // bit reversal first
            avx_bitreversed_transpose_f32_radix4(self.base_len, chunk, &mut scratch);

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

                        macro_rules! make_block {
                            ($data: expr, $twiddles: expr, $quarter: expr, $j: expr, $start: expr, $tw_start: expr) => {{
                                let a0 = _mm256_loadu_ps(
                                    data.get_unchecked(j + $start..).as_ptr().cast(),
                                );

                                let tw0 = _mm256_loadu_ps(
                                    $twiddles.get_unchecked(3 * j + $tw_start..).as_ptr().cast(),
                                );
                                let tw1 = _mm256_loadu_ps(
                                    $twiddles
                                        .get_unchecked(3 * j + $tw_start + 4..)
                                        .as_ptr()
                                        .cast(),
                                );
                                let tw2 = _mm256_loadu_ps(
                                    $twiddles
                                        .get_unchecked(3 * j + $tw_start + 8..)
                                        .as_ptr()
                                        .cast(),
                                );

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
                                q0t3 = _mm256_xor_ps(
                                    _mm256_shuffle_ps::<SH>(q0t3, q0t3),
                                    v_i_multiplier,
                                );

                                let y0 = _mm256_add_ps(q0t0, q0t2);
                                let y1 = _mm256_add_ps(q0t1, q0t3);
                                let y2 = _mm256_sub_ps(q0t0, q0t2);
                                let y3 = _mm256_sub_ps(q0t1, q0t3);
                                (y0, y1, y2, y3)
                            }};
                        }

                        while j + 12 < quarter {
                            let (y0, y1, y2, y3) = make_block!(data, m_twiddles, quarter, j, 0, 0);
                            let (y4, y5, y6, y7) = make_block!(data, m_twiddles, quarter, j, 4, 12);
                            let (y8, y9, y10, y11) =
                                make_block!(data, m_twiddles, quarter, j, 8, 24);

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

                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 4..).as_mut_ptr().cast(),
                                y4,
                            );
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

                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 8..).as_mut_ptr().cast(),
                                y8,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + quarter + 8..)
                                    .as_mut_ptr()
                                    .cast(),
                                y9,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 2 * quarter + 8..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 3 * quarter + 8..)
                                    .as_mut_ptr()
                                    .cast(),
                                y11,
                            );

                            j += 12;
                        }

                        while j + 8 < quarter {
                            let (y0, y1, y2, y3) = make_block!(data, m_twiddles, quarter, j, 0, 0);
                            let (y4, y5, y6, y7) = make_block!(data, m_twiddles, quarter, j, 4, 12);

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

                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 4..).as_mut_ptr().cast(),
                                y4,
                            );
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
                        }

                        while j + 4 < quarter {
                            let (y0, y1, y2, y3) = make_block!(data, m_twiddles, quarter, j, 0, 0);
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
                        }

                        while j + 2 < quarter {
                            let a0 = _mm_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                            let tw0tw1 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());
                            let tw2 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(3 * j + 4..).as_ptr().cast());

                            let rk1 =
                                _mm_loadu_ps(data.get_unchecked(j + quarter..).as_ptr().cast());
                            let rk2 =
                                _mm_loadu_ps(data.get_unchecked(j + 2 * quarter..).as_ptr().cast());
                            let rk3 =
                                _mm_loadu_ps(data.get_unchecked(j + 3 * quarter..).as_ptr().cast());

                            let b0c0 = _mm256_fcmul_ps(_mm256_create_ps(rk1, rk2), tw0tw1);
                            let d0 = _mm_fcmul_ps(rk3, tw2);
                            let b0 = _mm256_castps256_ps128(b0c0);
                            let c0 = _mm256_extractf128_ps::<1>(b0c0);

                            // radix-4 butterfly
                            let q0t0 = _mm_add_ps(a0, c0);
                            let q0t1 = _mm_sub_ps(a0, c0);
                            let q0t2 = _mm_add_ps(b0, d0);
                            let mut q0t3 = _mm_sub_ps(b0, d0);
                            const SH: i32 = shuffle(2, 3, 0, 1);
                            q0t3 = _mm_xor_ps(
                                _mm_shuffle_ps::<SH>(q0t3, q0t3),
                                _mm256_castps256_ps128(v_i_multiplier),
                            );

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
                        }

                        for j in j..quarter {
                            let a = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());

                            let bc = _mm_fcmul_ps(
                                _mm_unpacklo_ps64(
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + quarter..).as_ptr().cast(),
                                    ),
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 2 * quarter..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let d = _mm_fcmul_ps(
                                _m128s_load_f32x2(
                                    data.get_unchecked(j + 3 * quarter..).as_ptr().cast(),
                                ),
                                _m128s_load_f32x2(
                                    m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast(),
                                ),
                            );

                            let b = bc;
                            let c = _mm_unpackhi_ps64(bc, bc);

                            // radix-4 butterfly
                            let t0 = _mm_add_ps(a, c);
                            let t1 = _mm_sub_ps(a, c);
                            let t2 = _mm_add_ps(b, d);
                            let mut t3 = _mm_sub_ps(b, d);
                            const SH: i32 = shuffle(2, 3, 0, 1);
                            t3 = _mm_xor_ps(
                                _mm_shuffle_ps::<SH>(t3, t3),
                                _mm256_castps256_ps128(v_i_multiplier),
                            );

                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                _mm_add_ps(t0, t2),
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                _mm_add_ps(t1, t3),
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 2 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                _mm_sub_ps(t0, t2),
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 3 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                _mm_sub_ps(t1, t3),
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 3..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix4<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
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
    use crate::avx::{test_avx_radix, test_avx_radix_fast};

    test_avx_radix!(test_avx_radix4, f32, AvxFmaRadix4, 13, 2, 1e-2);
    test_avx_radix_fast!(test_avx_radix4_fast, f32, AvxFmaRadix4, Radix4, 13, 2, 1e-2);
    test_avx_radix_fast!(test_avx_radix4_f64, f64, AvxFmaRadix4, Radix4, 13, 2, 1e-8);
}
