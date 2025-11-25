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
use crate::avx::mixed::AvxStoreD;
use crate::avx::rotate::AvxRotate;
use crate::avx::transpose::{transpose_5x5_f64, transpose_6x6_f32};
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_fcmul_pd, _mm_fcmul_ps, _mm_unpackhi_ps64,
    _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps, _mm256_fcmul_pd, _mm256_fcmul_ps,
    create_avx4_twiddles, shuffle,
};
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::radix5::Radix5Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{compute_logarithm, compute_twiddle, is_power_of_five, reverse_bits};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::x86_64::*;
use std::fmt::Display;
use std::sync::Arc;

pub(crate) struct AvxFmaRadix5<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    direction: FftDirection,
    tw1tw2_im: [T; 8],
    tw2ntw1_im: [T; 8],
    tw1tw2_re: [T; 8],
    tw2tw1_re: [T; 8],
    butterfly: Arc<dyn CompositeFftExecutor<T> + Send + Sync>,
    butterfly_length: usize,
}

impl<
    T: Default
        + Clone
        + Radix5Twiddles
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
> AvxFmaRadix5<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix5<T>, ZaftError> {
        assert!(
            is_power_of_five(size as u64),
            "Input length must be a power of 5"
        );

        let log5 = compute_logarithm::<5>(size).unwrap();
        let butterfly = match log5 {
            0 => T::butterfly1(fft_direction)?,
            1 => T::butterfly5(fft_direction)?,
            _ => T::butterfly25(fft_direction)?,
        };

        let butterfly_length = butterfly.length();

        let twiddles = create_avx4_twiddles::<T, 5>(butterfly_length, size, fft_direction)?;
        let tw1 = compute_twiddle(1, 5, fft_direction);
        let tw2 = compute_twiddle(2, 5, fft_direction);

        Ok(AvxFmaRadix5 {
            execution_length: size,
            twiddles,
            twiddle1: tw1,
            twiddle2: tw2,
            tw1tw2_im: [
                tw1.im, tw1.im, tw2.im, tw2.im, tw1.im, tw1.im, tw2.im, tw2.im,
            ],
            tw2ntw1_im: [
                tw2.im, tw2.im, -tw1.im, -tw1.im, tw2.im, tw2.im, -tw1.im, -tw1.im,
            ],
            tw1tw2_re: [
                tw1.re, tw1.re, tw2.re, tw2.re, tw1.re, tw1.re, tw2.re, tw2.re,
            ],
            tw2tw1_re: [
                tw2.re, tw2.re, tw1.re, tw1.re, tw2.re, tw2.re, tw1.re, tw1.re,
            ],
            direction: fft_direction,
            butterfly,
            butterfly_length,
        })
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn complex5_load_f64(array: &[Complex<f64>], idx: usize) -> (AvxStoreD, AvxStoreD, AvxStoreD) {
    unsafe {
        (
            AvxStoreD::from_complex_ref(array.get_unchecked(idx..)),
            AvxStoreD::from_complex_ref(array.get_unchecked(idx + 2..)),
            AvxStoreD::from_complex(array.get_unchecked(idx + 4)),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn complex5_store_f64(array: &mut [Complex<f64>], idx: usize, v: [AvxStoreD; 3]) {
    unsafe {
        v[0].write(array.get_unchecked_mut(idx..));
        v[1].write(array.get_unchecked_mut(idx + 2..));
        v[2].write_lo(array.get_unchecked_mut(idx + 4..));
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn avx_bitreversed_transpose_f64_radix5(
    height: usize,
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
) {
    let width = input.len() / height;
    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }
    const WIDTH: usize = 5;
    const HEIGHT: usize = 5;

    let rev_digits = compute_logarithm::<5>(width).unwrap();
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
            reverse_bits::<WIDTH>(WIDTH * x + 4, rev_digits) * height,
        ];

        for y in 0..strided_height {
            let base_input_idx = (WIDTH * x) + y * HEIGHT * width;
            let rows = [
                complex5_load_f64(input, base_input_idx),
                complex5_load_f64(input, base_input_idx + width),
                complex5_load_f64(input, base_input_idx + width * 2),
                complex5_load_f64(input, base_input_idx + width * 3),
                complex5_load_f64(input, base_input_idx + width * 4),
            ];
            let transposed = transpose_5x5_f64(
                [rows[0].0, rows[1].0, rows[2].0, rows[3].0, rows[4].0],
                [rows[0].1, rows[1].1, rows[2].1, rows[3].1, rows[4].1],
                [rows[0].2, rows[1].2, rows[2].2, rows[3].2, rows[4].2],
            );

            complex5_store_f64(
                output,
                HEIGHT * y + x_rev[0],
                [transposed.0[0], transposed.1[0], transposed.2[0]],
            );
            complex5_store_f64(
                output,
                HEIGHT * y + x_rev[1],
                [transposed.0[1], transposed.1[1], transposed.2[1]],
            );
            complex5_store_f64(
                output,
                HEIGHT * y + x_rev[2],
                [transposed.0[2], transposed.1[2], transposed.2[2]],
            );
            complex5_store_f64(
                output,
                HEIGHT * y + x_rev[3],
                [transposed.0[3], transposed.1[3], transposed.2[3]],
            );
            complex5_store_f64(
                output,
                HEIGHT * y + x_rev[4],
                [transposed.0[4], transposed.1[4], transposed.2[4]],
            );
        }
    }
}

impl AvxFmaRadix5<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let tw1_re = _mm256_set1_pd(self.twiddle1.re);
            let tw1_im = _mm256_set1_pd(self.twiddle1.im);
            let tw2_re = _mm256_set1_pd(self.twiddle2.re);
            let tw2_im = _mm256_set1_pd(self.twiddle2.im);
            let rot_sign =
                _mm256_loadu_pd([-0.0f64, 0.0, -0.0f64, 0.0, -0.0f64, 0.0, -0.0f64, 0.0].as_ptr());

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                avx_bitreversed_transpose_f64_radix5(self.butterfly_length, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.butterfly_length;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 5;
                    let fifth = len / 5;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;
                        while j + 2 < fifth {
                            let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(4 * j + 4..).as_ptr().cast(),
                            );
                            let tw3 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(4 * j + 6..).as_ptr().cast(),
                            );

                            let u1 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(data.get_unchecked(j + fifth..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 2 * fifth..).as_ptr().cast(),
                                ),
                                tw1,
                            );
                            let u3 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 3 * fifth..).as_ptr().cast(),
                                ),
                                tw2,
                            );
                            let u4 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 4 * fifth..).as_ptr().cast(),
                                ),
                                tw3,
                            );

                            // Radix-5 butterfly

                            let x14p = _mm256_add_pd(u1, u4);
                            let x14n = _mm256_sub_pd(u1, u4);
                            let x23p = _mm256_add_pd(u2, u3);
                            let x23n = _mm256_sub_pd(u2, u3);
                            let y0 = _mm256_add_pd(_mm256_add_pd(u0, x14p), x23p);

                            let temp_b1_1 = _mm256_mul_pd(tw1_im, x14n);
                            let temp_b2_1 = _mm256_mul_pd(tw2_im, x14n);

                            let temp_a1 =
                                _mm256_fmadd_pd(tw2_re, x23p, _mm256_fmadd_pd(tw1_re, x14p, u0));
                            let temp_a2 =
                                _mm256_fmadd_pd(tw1_re, x23p, _mm256_fmadd_pd(tw2_re, x14p, u0));

                            let temp_b1 = _mm256_fmadd_pd(tw2_im, x23n, temp_b1_1);
                            let temp_b2 = _mm256_fnmadd_pd(tw1_im, x23n, temp_b2_1);

                            let temp_b1_rot =
                                _mm256_xor_pd(_mm256_permute_pd::<0b0101>(temp_b1), rot_sign);
                            let temp_b2_rot =
                                _mm256_xor_pd(_mm256_permute_pd::<0b0101>(temp_b2), rot_sign);

                            let y1 = _mm256_add_pd(temp_a1, temp_b1_rot);
                            let y2 = _mm256_add_pd(temp_a2, temp_b2_rot);
                            let y3 = _mm256_sub_pd(temp_a2, temp_b2_rot);
                            let y4 = _mm256_sub_pd(temp_a1, temp_b1_rot);

                            _mm256_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                                y4,
                            );

                            j += 2;
                        }

                        let tw1tw2_im = _mm256_loadu_pd(self.tw1tw2_im.as_ptr().cast());
                        let tw2ntw1_im = _mm256_loadu_pd(self.tw2ntw1_im.as_ptr().cast());
                        let tw1tw2_re = _mm256_loadu_pd(self.tw1tw2_re.as_ptr().cast());
                        let tw2tw1_re = _mm256_loadu_pd(self.tw2tw1_re.as_ptr().cast());
                        let rotate = AvxRotate::<f64>::new(FftDirection::Inverse);

                        for j in j..fifth {
                            let u0 = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast(),
                            );

                            let u1 = _mm_fcmul_pd(
                                _mm_loadu_pd(data.get_unchecked(j + fifth..).as_ptr().cast()),
                                _mm256_castpd256_pd128(tw0),
                            );
                            let u2 = _mm_fcmul_pd(
                                _mm_loadu_pd(data.get_unchecked(j + 2 * fifth..).as_ptr().cast()),
                                _mm256_extractf128_pd::<1>(tw0),
                            );
                            let u3 = _mm_fcmul_pd(
                                _mm_loadu_pd(data.get_unchecked(j + 3 * fifth..).as_ptr().cast()),
                                _mm256_castpd256_pd128(tw1),
                            );
                            let u4 = _mm_fcmul_pd(
                                _mm_loadu_pd(data.get_unchecked(j + 4 * fifth..).as_ptr().cast()),
                                _mm256_extractf128_pd::<1>(tw1),
                            );

                            // Radix-5 butterfly

                            const HI_HI: i32 = 0b0011_0001;
                            const LO_LO: i32 = 0b0010_0000;

                            let u1u2 = _mm256_create_pd(u1, u2);
                            let u4u3 = _mm256_create_pd(u4, u3);

                            let u0u0 = _mm256_create_pd(u0, u0);

                            let x14px23p = _mm256_add_pd(u1u2, u4u3);
                            let x14nx23n = _mm256_sub_pd(u1u2, u4u3);
                            let y0 = _mm_add_pd(
                                _mm_add_pd(u0, _mm256_castpd256_pd128(x14px23p)),
                                _mm256_extractf128_pd::<1>(x14px23p),
                            );

                            let temp_b1_1_b2_1 = _mm256_mul_pd(
                                tw1tw2_im,
                                _mm256_permute2f128_pd::<LO_LO>(x14nx23n, x14nx23n),
                            );

                            let wx23p = _mm256_permute2f128_pd::<HI_HI>(x14px23p, x14px23p);
                            let wx23n = _mm256_permute2f128_pd::<HI_HI>(x14nx23n, x14nx23n);

                            let temp_a1_a2 = _mm256_fmadd_pd(
                                tw2tw1_re,
                                wx23p,
                                _mm256_fmadd_pd(
                                    _mm256_permute2f128_pd::<LO_LO>(x14px23p, x14px23p),
                                    tw1tw2_re,
                                    u0u0,
                                ),
                            );

                            let temp_b1_b2 = _mm256_fmadd_pd(tw2ntw1_im, wx23n, temp_b1_1_b2_1);

                            let temp_b1_b2_rot = rotate.rotate_m256d(temp_b1_b2);

                            let y1y2 = _mm256_add_pd(temp_a1_a2, temp_b1_b2_rot);
                            let y4y3 = _mm256_sub_pd(temp_a1_a2, temp_b1_b2_rot);

                            _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(),
                                _mm256_castpd256_pd128(y1y2),
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                                _mm256_extractf128_pd::<1>(y1y2),
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                                _mm256_extractf128_pd::<1>(y4y3),
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                                _mm256_castpd256_pd128(y4y3),
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 4..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix5<f64> {
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

#[inline]
#[target_feature(enable = "avx2")]
fn complex5_load_f32(array: &[Complex<f32>], idx: usize) -> (__m256, __m256) {
    unsafe {
        (
            _mm256_loadu_ps(array.get_unchecked(idx..).as_ptr().cast()),
            _mm256_castps128_ps256(_m128s_load_f32x2(
                array.get_unchecked(idx + 4..).as_ptr().cast(),
            )),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn complex5_store_f32(array: &mut [Complex<f32>], idx: usize, v: (__m256, __m256)) {
    unsafe {
        _mm256_storeu_ps(array.get_unchecked_mut(idx..).as_mut_ptr().cast(), v.0);
        _m128s_store_f32x2(
            array.get_unchecked_mut(idx + 4..).as_mut_ptr().cast(),
            _mm256_castps256_ps128(v.1),
        );
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn avx_bitreversed_transpose_f32_radix5(
    height: usize,
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
) {
    let width = input.len() / height;
    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }
    const WIDTH: usize = 5;
    const HEIGHT: usize = 5;

    let rev_digits = compute_logarithm::<5>(width).unwrap();
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
            reverse_bits::<WIDTH>(WIDTH * x + 4, rev_digits) * height,
        ];

        for y in 0..strided_height {
            let base_input_idx = (WIDTH * x) + y * HEIGHT * width;
            let rows = [
                complex5_load_f32(input, base_input_idx),
                complex5_load_f32(input, base_input_idx + width),
                complex5_load_f32(input, base_input_idx + width * 2),
                complex5_load_f32(input, base_input_idx + width * 3),
                complex5_load_f32(input, base_input_idx + width * 4),
            ];
            let transposed = transpose_6x6_f32(
                [
                    rows[0].0,
                    rows[1].0,
                    rows[2].0,
                    rows[3].0,
                    rows[4].0,
                    _mm256_setzero_ps(),
                ],
                [
                    rows[0].1,
                    rows[1].1,
                    rows[2].1,
                    rows[3].1,
                    rows[4].1,
                    _mm256_setzero_ps(),
                ],
            );

            complex5_store_f32(
                output,
                HEIGHT * y + x_rev[0],
                (transposed.0[0], transposed.1[0]),
            );
            complex5_store_f32(
                output,
                HEIGHT * y + x_rev[1],
                (transposed.0[1], transposed.1[1]),
            );
            complex5_store_f32(
                output,
                HEIGHT * y + x_rev[2],
                (transposed.0[2], transposed.1[2]),
            );
            complex5_store_f32(
                output,
                HEIGHT * y + x_rev[3],
                (transposed.0[3], transposed.1[3]),
            );
            complex5_store_f32(
                output,
                HEIGHT * y + x_rev[4],
                (transposed.0[4], transposed.1[4]),
            );
        }
    }
}

impl AvxFmaRadix5<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let tw1_re = _mm256_set1_ps(self.twiddle1.re);
            let tw1_im = _mm256_set1_ps(self.twiddle1.im);
            let tw2_re = _mm256_set1_ps(self.twiddle2.re);
            let tw2_im = _mm256_set1_ps(self.twiddle2.im);
            static ROT_90: [f32; 8] = [-0.0f32, 0.0, -0.0, 0.0, -0.0f32, 0.0, -0.0, 0.0];
            let rot_sign = _mm256_loadu_ps(ROT_90.as_ptr());

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                avx_bitreversed_transpose_f32_radix5(self.butterfly_length, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.butterfly_length;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 5;
                    let fifth = len / 5;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 4 < fifth {
                            let u0 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(4 * j + 4..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(4 * j + 8..).as_ptr().cast(),
                            );
                            let tw3 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(4 * j + 12..).as_ptr().cast(),
                            );

                            let u1 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(data.get_unchecked(j + fifth..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 2 * fifth..).as_ptr().cast(),
                                ),
                                tw1,
                            );
                            let u3 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 3 * fifth..).as_ptr().cast(),
                                ),
                                tw2,
                            );
                            let u4 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 4 * fifth..).as_ptr().cast(),
                                ),
                                tw3,
                            );

                            // Radix-5 butterfly

                            let x14p = _mm256_add_ps(u1, u4);
                            let x14n = _mm256_sub_ps(u1, u4);
                            let x23p = _mm256_add_ps(u2, u3);
                            let x23n = _mm256_sub_ps(u2, u3);
                            let y0 = _mm256_add_ps(_mm256_add_ps(u0, x14p), x23p);

                            let temp_b1_1 = _mm256_mul_ps(tw1_im, x14n);
                            let temp_b2_1 = _mm256_mul_ps(tw2_im, x14n);

                            let temp_a1 =
                                _mm256_fmadd_ps(tw2_re, x23p, _mm256_fmadd_ps(tw1_re, x14p, u0));
                            let temp_a2 =
                                _mm256_fmadd_ps(tw1_re, x23p, _mm256_fmadd_ps(tw2_re, x14p, u0));

                            let temp_b1 = _mm256_fmadd_ps(tw2_im, x23n, temp_b1_1);
                            let temp_b2 = _mm256_fnmadd_ps(tw1_im, x23n, temp_b2_1);

                            const SH: i32 = shuffle(2, 3, 0, 1);
                            let temp_b1_rot =
                                _mm256_xor_ps(_mm256_shuffle_ps::<SH>(temp_b1, temp_b1), rot_sign);
                            let temp_b2_rot =
                                _mm256_xor_ps(_mm256_shuffle_ps::<SH>(temp_b2, temp_b2), rot_sign);

                            let y1 = _mm256_add_ps(temp_a1, temp_b1_rot);
                            let y2 = _mm256_add_ps(temp_a2, temp_b2_rot);
                            let y3 = _mm256_sub_ps(temp_a2, temp_b2_rot);
                            let y4 = _mm256_sub_ps(temp_a1, temp_b1_rot);

                            _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                                y4,
                            );
                            j += 4;
                        }

                        while j + 2 < fifth {
                            let u0 = _mm_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());

                            let tw1 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(4 * j + 4..).as_ptr().cast(),
                            );

                            const SH: i32 = shuffle(3, 1, 2, 0);

                            let u1u2 = _mm256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(data.get_unchecked(j + fifth..).as_ptr().cast()),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 2 * fifth..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let u3u4 = _mm256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 3 * fifth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 4 * fifth..).as_ptr().cast(),
                                    ),
                                ),
                                tw1,
                            );

                            let u1 = _mm256_castps256_ps128(u1u2);
                            let u2 = _mm256_extractf128_ps::<1>(u1u2);
                            let u3 = _mm256_castps256_ps128(u3u4);
                            let u4 = _mm256_extractf128_ps::<1>(u3u4);

                            // Radix-5 butterfly

                            let x14p = _mm_add_ps(u1, u4);
                            let x14n = _mm_sub_ps(u1, u4);
                            let x23p = _mm_add_ps(u2, u3);
                            let x23n = _mm_sub_ps(u2, u3);
                            let y0 = _mm_add_ps(_mm_add_ps(u0, x14p), x23p);

                            let temp_b1_1 = _mm_mul_ps(_mm256_castps256_ps128(tw1_im), x14n);
                            let temp_b2_1 = _mm_mul_ps(_mm256_castps256_ps128(tw2_im), x14n);

                            let temp_a1 = _mm_fmadd_ps(
                                _mm256_castps256_ps128(tw2_re),
                                x23p,
                                _mm_fmadd_ps(_mm256_castps256_ps128(tw1_re), x14p, u0),
                            );
                            let temp_a2 = _mm_fmadd_ps(
                                _mm256_castps256_ps128(tw1_re),
                                x23p,
                                _mm_fmadd_ps(_mm256_castps256_ps128(tw2_re), x14p, u0),
                            );

                            let temp_b1 =
                                _mm_fmadd_ps(_mm256_castps256_ps128(tw2_im), x23n, temp_b1_1);
                            let temp_b2 =
                                _mm_fnmadd_ps(_mm256_castps256_ps128(tw1_im), x23n, temp_b2_1);

                            let temp_b1_rot = _mm_xor_ps(
                                _mm_shuffle_ps::<SH>(temp_b1, temp_b1),
                                _mm256_castps256_ps128(rot_sign),
                            );
                            let temp_b2_rot = _mm_xor_ps(
                                _mm_shuffle_ps::<SH>(temp_b2, temp_b2),
                                _mm256_castps256_ps128(rot_sign),
                            );

                            let y1 = _mm_add_ps(temp_a1, temp_b1_rot);
                            let y2 = _mm_add_ps(temp_a2, temp_b2_rot);
                            let y3 = _mm_sub_ps(temp_a2, temp_b2_rot);
                            let y4 = _mm_sub_ps(temp_a1, temp_b1_rot);

                            _mm_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                                y4,
                            );
                            j += 2;
                        }

                        for j in j..fifth {
                            let u0 = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                            let tw2 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast());

                            let u1u2 = _mm_fcmul_ps(
                                _mm_unpacklo_ps64(
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + fifth..).as_ptr().cast(),
                                    ),
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 2 * fifth..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let u3u4 = _mm_fcmul_ps(
                                _mm_unpacklo_ps64(
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 3 * fifth..).as_ptr().cast(),
                                    ),
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 4 * fifth..).as_ptr().cast(),
                                    ),
                                ),
                                tw2,
                            );

                            let u1 = u1u2;
                            let u2 = _mm_unpackhi_ps64(u1u2, u1u2);
                            let u3 = u3u4;
                            let u4 = _mm_unpackhi_ps64(u3u4, u3u4);

                            // Radix-5 butterfly

                            let x14p = _mm_add_ps(u1, u4);
                            let x14n = _mm_sub_ps(u1, u4);
                            let x23p = _mm_add_ps(u2, u3);
                            let x23n = _mm_sub_ps(u2, u3);
                            let y0 = _mm_add_ps(_mm_add_ps(u0, x14p), x23p);

                            let temp_b1_1 = _mm_mul_ps(_mm256_castps256_ps128(tw1_im), x14n);
                            let temp_b2_1 = _mm_mul_ps(_mm256_castps256_ps128(tw2_im), x14n);

                            let temp_a1 = _mm_fmadd_ps(
                                _mm256_castps256_ps128(tw2_re),
                                x23p,
                                _mm_fmadd_ps(_mm256_castps256_ps128(tw1_re), x14p, u0),
                            );
                            let temp_a2 = _mm_fmadd_ps(
                                _mm256_castps256_ps128(tw1_re),
                                x23p,
                                _mm_fmadd_ps(_mm256_castps256_ps128(tw2_re), x14p, u0),
                            );

                            let temp_b1 =
                                _mm_fmadd_ps(_mm256_castps256_ps128(tw2_im), x23n, temp_b1_1);
                            let temp_b2 =
                                _mm_fnmadd_ps(_mm256_castps256_ps128(tw1_im), x23n, temp_b2_1);

                            const SH: i32 = shuffle(2, 3, 0, 1);
                            let temp_b1_rot = _mm_xor_ps(
                                _mm_shuffle_ps::<SH>(temp_b1, temp_b1),
                                _mm256_castps256_ps128(rot_sign),
                            );
                            let temp_b2_rot = _mm_xor_ps(
                                _mm_shuffle_ps::<SH>(temp_b2, temp_b2),
                                _mm256_castps256_ps128(rot_sign),
                            );

                            let y1 = _mm_add_ps(temp_a1, temp_b1_rot);
                            let y2 = _mm_add_ps(temp_a2, temp_b2_rot);
                            let y3 = _mm_sub_ps(temp_a2, temp_b2_rot);
                            let y4 = _mm_sub_ps(temp_a1, temp_b1_rot);

                            _m128s_store_f32x2(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                                y4,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 4..];
                }
            }
        }

        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix5<f32> {
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
    use crate::avx::test_avx_radix;

    test_avx_radix!(test_avx_radix5, f32, AvxFmaRadix5, 5, 5, 1e-3);
    test_avx_radix!(test_avx_radix5_f64, f64, AvxFmaRadix5, 5, 5, 1e-8);
}
