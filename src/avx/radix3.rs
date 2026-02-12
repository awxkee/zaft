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
    _mm_unpacklo_ps64, _mm256_fcmul_pd, _mm256_fcmul_ps, shuffle,
};
use crate::err::try_vec;
use crate::radix3::Radix3Twiddles;
use crate::util::{
    compute_twiddle, int_logarithm, reverse_bits, validate_oof_sizes, validate_scratch,
};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::any::TypeId;
use std::arch::x86_64::*;
use std::sync::Arc;

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn complex3_load_f32(array: &[Complex<f32>], idx: usize) -> __m256 {
    unsafe {
        _mm256_setr_m128(
            _mm_loadu_ps(array.get_unchecked(idx..).as_ptr().cast()),
            _m128s_load_f32x2(array.get_unchecked(idx + 2..).as_ptr().cast()),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn complex3_store_f32(array: &mut [Complex<f32>], idx: usize, v: __m256) {
    unsafe {
        _mm_storeu_ps(
            array.get_unchecked_mut(idx..).as_mut_ptr().cast(),
            _mm256_castps256_ps128(v),
        );
        _m128s_store_f32x2(
            array.get_unchecked_mut(idx + 2..).as_mut_ptr().cast(),
            _mm256_extractf128_ps::<1>(v),
        );
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn complex3_load_f64(array: &[Complex<f64>], idx: usize) -> (__m256d, __m256d) {
    unsafe {
        (
            _mm256_loadu_pd(array.get_unchecked(idx..).as_ptr().cast()),
            _mm256_castpd128_pd256(_mm_loadu_pd(array.get_unchecked(idx + 2..).as_ptr().cast())),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn complex3_store_f64(array: &mut [Complex<f64>], idx: usize, v: (__m256d, __m256d)) {
    unsafe {
        _mm256_storeu_pd(array.get_unchecked_mut(idx..).as_mut_ptr().cast(), v.0);
        _mm_storeu_pd(
            array.get_unchecked_mut(idx + 2..).as_mut_ptr().cast(),
            _mm256_castpd256_pd128(v.1),
        );
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn avx_bitreversed_transpose_f32_radix3(
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

    let rev_digits = int_logarithm::<3>(width).unwrap();
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
        ];

        for y in 0..strided_height {
            let base_input_idx = (WIDTH * x) + y * HEIGHT * width;
            let rows = [
                complex3_load_f32(input, base_input_idx),
                complex3_load_f32(input, base_input_idx + width),
                complex3_load_f32(input, base_input_idx + width * 2),
            ];
            let transposed =
                avx_transpose_f32x2_4x4_impl(rows[0], rows[1], rows[2], _mm256_setzero_ps());

            complex3_store_f32(output, HEIGHT * y + x_rev[0], transposed.0);
            complex3_store_f32(output, HEIGHT * y + x_rev[1], transposed.1);
            complex3_store_f32(output, HEIGHT * y + x_rev[2], transposed.2);
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn avx_bitreversed_transpose_f64_radix3(
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

    let rev_digits = int_logarithm::<3>(width).unwrap();
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
        ];

        for y in 0..strided_height {
            let base_input_idx = (WIDTH * x) + y * HEIGHT * width;
            let rows = [
                complex3_load_f64(input, base_input_idx),
                complex3_load_f64(input, base_input_idx + width),
                complex3_load_f64(input, base_input_idx + width * 2),
            ];
            let transposed = avx_transpose_f64x2_4x4_impl(
                rows[0],
                rows[1],
                rows[2],
                (_mm256_setzero_pd(), _mm256_setzero_pd()),
            );

            complex3_store_f64(output, HEIGHT * y + x_rev[0], transposed.0);
            complex3_store_f64(output, HEIGHT * y + x_rev[1], transposed.1);
            complex3_store_f64(output, HEIGHT * y + x_rev[2], transposed.2);
        }
    }
}

pub(crate) struct AvxFmaRadix3<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle_re: T,
    twiddle_im: [T; 8],
    direction: FftDirection,
    base_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    base_len: usize,
}

impl<T: FftSample + Radix3Twiddles> AvxFmaRadix3<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix3<T>, ZaftError> {
        assert!(
            size.is_power_of_two() || size.is_multiple_of(3),
            "Input length must be divisible by 3"
        );

        let exponent = int_logarithm::<3>(size).unwrap_or_else(|| {
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

        let mut twiddles = Vec::new();
        twiddles
            .try_reserve_exact(size - 1)
            .map_err(|_| ZaftError::OutOfMemory(size - 1))?;

        const N: usize = 3;
        let mut cross_fft_len = base_fft.length();
        while cross_fft_len < size {
            let num_columns = cross_fft_len;
            cross_fft_len *= N;

            let mut i = 0usize;

            if TypeId::of::<T>() == TypeId::of::<f32>() {
                while i + 4 <= num_columns {
                    for k in 1..N {
                        let twiddle0 = compute_twiddle(i * k, cross_fft_len, fft_direction);
                        let twiddle1 = compute_twiddle((i + 1) * k, cross_fft_len, fft_direction);
                        let twiddle2 = compute_twiddle((i + 2) * k, cross_fft_len, fft_direction);
                        let twiddle3 = compute_twiddle((i + 3) * k, cross_fft_len, fft_direction);
                        twiddles.push(twiddle0);
                        twiddles.push(twiddle1);
                        twiddles.push(twiddle2);
                        twiddles.push(twiddle3);
                    }
                    i += 4;
                }
            }

            if TypeId::of::<T>() == TypeId::of::<f64>() {
                while i + 2 <= num_columns {
                    for k in 1..N {
                        let twiddle0 = compute_twiddle(i * k, cross_fft_len, fft_direction);
                        let twiddle1 = compute_twiddle((i + 1) * k, cross_fft_len, fft_direction);
                        twiddles.push(twiddle0);
                        twiddles.push(twiddle1);
                    }
                    i += 2;
                }
            }

            for i in i..num_columns {
                for k in 1..N {
                    let twiddle = compute_twiddle(i * k, cross_fft_len, fft_direction);
                    twiddles.push(twiddle);
                }
            }
        }

        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);

        Ok(AvxFmaRadix3 {
            execution_length: size,
            twiddles,
            twiddle_re: twiddle.re,
            twiddle_im: [
                -twiddle.im,
                twiddle.im,
                -twiddle.im,
                twiddle.im,
                -twiddle.im,
                twiddle.im,
                -twiddle.im,
                twiddle.im,
            ],
            direction: fft_direction,
            base_len: base_fft.length(),
            base_fft,
        })
    }
}

impl AvxFmaRadix3<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn base_run(&self, chunk: &mut [Complex<f64>]) {
        unsafe {
            let twiddle_re = _mm256_set1_pd(self.twiddle_re);
            let twiddle_w_2 = _mm256_loadu_pd(self.twiddle_im.as_ptr().cast());

            let mut len = self.base_len;

            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                let columns = len;
                len *= 3;
                let third = len / 3;

                for data in chunk.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 4 <= third {
                        let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());
                        let u0_1 = _mm256_loadu_pd(data.get_unchecked(j + 2..).as_ptr().cast());

                        let tw0 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());
                        let tw1 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(2 * j + 2..).as_ptr().cast());

                        let tw2 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(2 * j + 4..).as_ptr().cast());
                        let tw3 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(2 * j + 6..).as_ptr().cast());

                        let rk1 = _mm256_loadu_pd(data.get_unchecked(j + third..).as_ptr().cast());
                        let rk2 =
                            _mm256_loadu_pd(data.get_unchecked(j + 2 * third..).as_ptr().cast());

                        let rk1_1 =
                            _mm256_loadu_pd(data.get_unchecked(j + third + 2..).as_ptr().cast());
                        let rk2_1 = _mm256_loadu_pd(
                            data.get_unchecked(j + 2 * third + 2..).as_ptr().cast(),
                        );

                        let u1 = _mm256_fcmul_pd(rk1, tw0);
                        let u2 = _mm256_fcmul_pd(rk2, tw1);

                        let u1_1 = _mm256_fcmul_pd(rk1_1, tw2);
                        let u2_1 = _mm256_fcmul_pd(rk2_1, tw3);

                        // Radix-3 butterfly
                        let xp_0 = _mm256_add_pd(u1, u2);
                        let xn_0 = _mm256_sub_pd(u1, u2);
                        let sum_0 = _mm256_add_pd(u0, xp_0);

                        let xp_1 = _mm256_add_pd(u1_1, u2_1);
                        let xn_1 = _mm256_sub_pd(u1_1, u2_1);
                        let sum_1 = _mm256_add_pd(u0_1, xp_1);

                        let w_0 = _mm256_fmadd_pd(twiddle_re, xp_0, u0);
                        let xn_rot_0 = _mm256_shuffle_pd::<0b0101>(xn_0, xn_0);

                        let w_1 = _mm256_fmadd_pd(twiddle_re, xp_1, u0_1);
                        let xn_rot_1 = _mm256_shuffle_pd::<0b0101>(xn_1, xn_1);

                        let vy0 = sum_0;
                        let vy1 = _mm256_fmadd_pd(twiddle_w_2, xn_rot_0, w_0);
                        let vy2 = _mm256_fnmadd_pd(twiddle_w_2, xn_rot_0, w_0);

                        let vy0_1 = sum_1;
                        let vy1_1 = _mm256_fmadd_pd(twiddle_w_2, xn_rot_1, w_1);
                        let vy2_1 = _mm256_fnmadd_pd(twiddle_w_2, xn_rot_1, w_1);

                        _mm256_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + third..).as_mut_ptr().cast(),
                            vy1,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                            vy2,
                        );

                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 2..).as_mut_ptr().cast(),
                            vy0_1,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + third + 2..).as_mut_ptr().cast(),
                            vy1_1,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 2 * third + 2..)
                                .as_mut_ptr()
                                .cast(),
                            vy2_1,
                        );

                        j += 4;
                    }

                    while j + 2 <= third {
                        let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                        let tw0 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());
                        let tw1 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(2 * j + 2..).as_ptr().cast());

                        let rk1 = _mm256_loadu_pd(data.get_unchecked(j + third..).as_ptr().cast());
                        let rk2 =
                            _mm256_loadu_pd(data.get_unchecked(j + 2 * third..).as_ptr().cast());

                        let u1 = _mm256_fcmul_pd(rk1, tw0);
                        let u2 = _mm256_fcmul_pd(rk2, tw1);

                        // Radix-3 butterfly
                        let xp = _mm256_add_pd(u1, u2);
                        let xn = _mm256_sub_pd(u1, u2);
                        let sum = _mm256_add_pd(u0, xp);

                        let w_1 = _mm256_fmadd_pd(twiddle_re, xp, u0);
                        let xn_rot = _mm256_shuffle_pd::<0b0101>(xn, xn);

                        let vy0 = sum;
                        let vy1 = _mm256_fmadd_pd(twiddle_w_2, xn_rot, w_1);
                        let vy2 = _mm256_fnmadd_pd(twiddle_w_2, xn_rot, w_1);

                        _mm256_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + third..).as_mut_ptr().cast(),
                            vy1,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                            vy2,
                        );

                        j += 2;
                    }

                    for j in j..third {
                        let u0 = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());
                        let u1 = _mm_fcmul_pd(
                            _mm_loadu_pd(data.get_unchecked(j + third..).as_ptr().cast()),
                            _mm_loadu_pd(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                        );
                        let u2 = _mm_fcmul_pd(
                            _mm_loadu_pd(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                            _mm_loadu_pd(m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast()),
                        );

                        // Radix-3 butterfly
                        let xp = _mm_add_pd(u1, u2);
                        let xn = _mm_sub_pd(u1, u2);
                        let sum = _mm_add_pd(u0, xp);

                        let w_1 = _mm_fmadd_pd(_mm256_castpd256_pd128(twiddle_re), xp, u0);
                        let xn_rot = _mm_shuffle_pd::<0b01>(xn, xn);

                        let vy0 = sum;
                        let vy1 = _mm_fmadd_pd(_mm256_castpd256_pd128(twiddle_w_2), xn_rot, w_1);
                        let vy2 = _mm_fnmadd_pd(_mm256_castpd256_pd128(twiddle_w_2), xn_rot, w_1);

                        _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                        _mm_storeu_pd(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                            vy2,
                        );
                    }
                }

                m_twiddles = &m_twiddles[columns * 2..];
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
            // Digit-reversal permutation
            avx_bitreversed_transpose_f64_radix3(self.base_len, chunk, scratch);
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
            avx_bitreversed_transpose_f64_radix3(self.base_len, src, dst);
            self.base_fft.execute(dst)?;
            self.base_run(dst);
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix3<f64> {
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

impl AvxFmaRadix3<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn base_run(&self, chunk: &mut [Complex<f32>]) {
        unsafe {
            let twiddle_re = _mm256_set1_ps(self.twiddle_re);
            let twiddle_w_2 = _mm256_loadu_ps(self.twiddle_im.as_ptr().cast());

            let mut len = self.base_len;

            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                let columns = len;
                len *= 3;
                let third = len / 3;

                for data in chunk.chunks_exact_mut(len) {
                    macro_rules! make_block {
                        ($data: expr, $twiddles: expr, $third: expr, $j: expr, $start: expr, $tw_start: expr) => {{
                            let u0 =
                                _mm256_loadu_ps($data.get_unchecked($j + $start..).as_ptr().cast());

                            let tw0 = _mm256_loadu_ps(
                                $twiddles
                                    .get_unchecked(2 * $j + $tw_start..)
                                    .as_ptr()
                                    .cast(),
                            );
                            let tw1 = _mm256_loadu_ps(
                                $twiddles
                                    .get_unchecked(2 * $j + $tw_start + 4..)
                                    .as_ptr()
                                    .cast(),
                            );

                            let rk1 = _mm256_loadu_ps(
                                $data.get_unchecked($j + $third + $start..).as_ptr().cast(),
                            );
                            let rk2 = _mm256_loadu_ps(
                                $data
                                    .get_unchecked($j + 2 * $third + $start..)
                                    .as_ptr()
                                    .cast(),
                            );

                            let u1 = _mm256_fcmul_ps(rk1, tw0);
                            let u2 = _mm256_fcmul_ps(rk2, tw1);

                            // Radix-3 butterfly
                            let xp_0 = _mm256_add_ps(u1, u2);
                            let xn_0 = _mm256_sub_ps(u1, u2);
                            let sum_0 = _mm256_add_ps(u0, xp_0);

                            const SH: i32 = shuffle(2, 3, 0, 1);

                            let vw_1_1 = _mm256_fmadd_ps(twiddle_re, xp_0, u0);
                            let xn_rot = _mm256_shuffle_ps::<SH>(xn_0, xn_0);

                            let vy0 = sum_0;
                            let vy1 = _mm256_fmadd_ps(twiddle_w_2, xn_rot, vw_1_1);
                            let vy2 = _mm256_fnmadd_ps(twiddle_w_2, xn_rot, vw_1_1);
                            (vy0, vy1, vy2)
                        }};
                    }

                    let mut j = 0usize;

                    while j + 8 <= third {
                        let (vy0, vy1, vy2) = make_block!(data, m_twiddles, third, j, 0, 0);
                        let (vy0_1, vy1_1, vy2_1) = make_block!(data, m_twiddles, third, j, 4, 8);

                        _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + third..).as_mut_ptr().cast(),
                            vy1,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                            vy2,
                        );

                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 4..).as_mut_ptr().cast(),
                            vy0_1,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + third + 4..).as_mut_ptr().cast(),
                            vy1_1,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 2 * third + 4..)
                                .as_mut_ptr()
                                .cast(),
                            vy2_1,
                        );

                        j += 8;
                    }

                    while j + 4 <= third {
                        let (vy0, vy1, vy2) = make_block!(data, m_twiddles, third, j, 0, 0);

                        _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + third..).as_mut_ptr().cast(),
                            vy1,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                            vy2,
                        );

                        j += 4;
                    }

                    for j in j..third {
                        let u0 = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());

                        let tw = _mm_loadu_ps(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());

                        let rk1 =
                            _m128s_load_f32x2(data.get_unchecked(j + third..).as_ptr().cast());
                        let rk2 =
                            _m128s_load_f32x2(data.get_unchecked(j + 2 * third..).as_ptr().cast());

                        let u1u2 = _mm_fcmul_ps(_mm_unpacklo_ps64(rk1, rk2), tw);

                        let u1 = u1u2;
                        let u2 = _mm_unpackhi_ps64(u1u2, u1u2);

                        // Radix-3 butterfly
                        let xp = _mm_add_ps(u1, u2);
                        let xn = _mm_sub_ps(u1, u2);
                        let sum = _mm_add_ps(u0, xp);

                        const SH: i32 = shuffle(2, 3, 0, 1);

                        let w_1 = _mm_fmadd_ps(_mm256_castps256_ps128(twiddle_re), xp, u0);
                        let xn_rot = _mm_shuffle_ps::<SH>(xn, xn);

                        let vy0 = sum;
                        let vy1 = _mm_fmadd_ps(_mm256_castps256_ps128(twiddle_w_2), xn_rot, w_1);
                        let vy2 = _mm_fnmadd_ps(_mm256_castps256_ps128(twiddle_w_2), xn_rot, w_1);

                        _m128s_store_f32x2(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + third..).as_mut_ptr().cast(),
                            vy1,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                            vy2,
                        );
                    }
                }

                m_twiddles = &m_twiddles[columns * 2..];
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
            // Digit-reversal permutation
            avx_bitreversed_transpose_f32_radix3(self.base_len, chunk, scratch);
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
            avx_bitreversed_transpose_f32_radix3(self.base_len, src, dst);
            self.base_fft.execute(dst)?;
            self.base_run(dst);
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix3<f32> {
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
    use crate::avx::test_avx_radix;

    test_avx_radix!(test_avx_radix3, f32, AvxFmaRadix3, 6, 3, 1e-3);
    test_avx_radix!(test_avx_radix3_f64, f64, AvxFmaRadix3, 6, 3, 1e-8);
}
