/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use crate::avx::butterflies::{AvxButterfly, AvxFastButterfly5d, AvxFastButterfly5f};
use crate::avx::radix4::{
    complex4_load_f32, complex4_load_f64, complex4_store_f32, complex4_store_f64,
};
use crate::avx::transpose::{
    avx_transpose_f32x2_4x4_impl, avx_transpose_f64x2_4x4_impl, transpose_f32_2x2_impl,
    transpose_f64x2_2x2,
};
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_fcmul_pd, _mm_fcmul_ps, _mm_unpackhi_ps64,
    _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps, _mm256_fcmul_pd, _mm256_fcmul_ps,
    _mm256_load4_f32x2, create_avx4_twiddles, shuffle,
};
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::util::{compute_logarithm, is_power_of_ten, reverse_bits};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;
use std::sync::Arc;

pub(crate) struct AvxFmaRadix10d {
    twiddles: Vec<Complex<f64>>,
    execution_length: usize,
    bf5: AvxFastButterfly5d,
    direction: FftDirection,
    butterfly: Arc<dyn CompositeFftExecutor<f64> + Send + Sync>,
    butterfly_length: usize,
}

impl AvxFmaRadix10d {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix10d, ZaftError> {
        assert!(
            is_power_of_ten(size as u64),
            "Input length must be a power of 10"
        );

        let log10 = compute_logarithm::<10>(size).unwrap();
        let butterfly = match log10 {
            0 => f64::butterfly1(fft_direction)?,
            1 => f64::butterfly10(fft_direction)?,
            _ => f64::butterfly100(fft_direction)
                .map_or_else(|| f64::butterfly10(fft_direction), Ok)?,
        };

        let butterfly_length = butterfly.length();

        let twiddles = create_avx4_twiddles::<f64, 10>(butterfly_length, size, fft_direction)?;

        Ok(AvxFmaRadix10d {
            execution_length: size,
            twiddles,
            bf5: unsafe { AvxFastButterfly5d::new(fft_direction) },
            direction: fft_direction,
            butterfly,
            butterfly_length,
        })
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn complex2_load_f64(array: &[Complex<f64>], idx: usize) -> __m256d {
    unsafe { _mm256_loadu_pd(array.get_unchecked(idx..).as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn complex2_store_f64(array: &mut [Complex<f64>], idx: usize, v: __m256d) {
    unsafe {
        _mm256_storeu_pd(array.get_unchecked_mut(idx..).as_mut_ptr().cast(), v);
    }
}

#[target_feature(enable = "avx2")]
fn avx_bitreversed_transpose_f64_radix10(
    height: usize,
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
) {
    let width = input.len() / height;

    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }

    const WIDTH: usize = 10;
    const HEIGHT: usize = 10;

    let rev_digits = compute_logarithm::<10>(width).unwrap();
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
        // ^ T
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
                complex4_load_f64(input, base_input_idx),
                complex4_load_f64(input, base_input_idx + width),
                complex4_load_f64(input, base_input_idx + width * 2),
                complex4_load_f64(input, base_input_idx + width * 3),
            ];

            let a1 = [
                complex4_load_f64(input, base_input_idx + 4),
                complex4_load_f64(input, base_input_idx + width + 4),
                complex4_load_f64(input, base_input_idx + width * 2 + 4),
                complex4_load_f64(input, base_input_idx + width * 3 + 4),
            ];

            let transposed_a0 = avx_transpose_f64x2_4x4_impl(a0[0], a0[1], a0[2], a0[3]);
            let transposed_a1 = avx_transpose_f64x2_4x4_impl(a1[0], a1[1], a1[2], a1[3]);

            complex4_store_f64(output, HEIGHT * y + x_rev[0], transposed_a0.0);
            complex4_store_f64(output, HEIGHT * y + x_rev[1], transposed_a0.1);
            complex4_store_f64(output, HEIGHT * y + x_rev[2], transposed_a0.2);
            complex4_store_f64(output, HEIGHT * y + x_rev[3], transposed_a0.3);

            complex4_store_f64(output, HEIGHT * y + x_rev[4], transposed_a1.0);
            complex4_store_f64(output, HEIGHT * y + x_rev[5], transposed_a1.1);
            complex4_store_f64(output, HEIGHT * y + x_rev[6], transposed_a1.2);
            complex4_store_f64(output, HEIGHT * y + x_rev[7], transposed_a1.3);

            let b0 = [
                complex4_load_f64(input, base_input_idx + width * 4),
                complex4_load_f64(input, base_input_idx + width * 5),
                complex4_load_f64(input, base_input_idx + width * 6),
                complex4_load_f64(input, base_input_idx + width * 7),
            ];

            let b1 = [
                complex4_load_f64(input, base_input_idx + width * 4 + 4),
                complex4_load_f64(input, base_input_idx + width * 5 + 4),
                complex4_load_f64(input, base_input_idx + width * 6 + 4),
                complex4_load_f64(input, base_input_idx + width * 7 + 4),
            ];

            let transposed_b0 = avx_transpose_f64x2_4x4_impl(b0[0], b0[1], b0[2], b0[3]);
            let transposed_b1 = avx_transpose_f64x2_4x4_impl(b1[0], b1[1], b1[2], b1[3]);

            complex4_store_f64(output, HEIGHT * y + x_rev[0] + 4, transposed_b0.0);
            complex4_store_f64(output, HEIGHT * y + x_rev[1] + 4, transposed_b0.1);
            complex4_store_f64(output, HEIGHT * y + x_rev[2] + 4, transposed_b0.2);
            complex4_store_f64(output, HEIGHT * y + x_rev[3] + 4, transposed_b0.3);

            complex4_store_f64(output, HEIGHT * y + x_rev[4] + 4, transposed_b1.0);
            complex4_store_f64(output, HEIGHT * y + x_rev[5] + 4, transposed_b1.1);
            complex4_store_f64(output, HEIGHT * y + x_rev[6] + 4, transposed_b1.2);
            complex4_store_f64(output, HEIGHT * y + x_rev[7] + 4, transposed_b1.3);

            let a2 = [
                complex2_load_f64(input, base_input_idx + 8),
                complex2_load_f64(input, base_input_idx + width + 8),
                complex2_load_f64(input, base_input_idx + width * 2 + 8),
                complex2_load_f64(input, base_input_idx + width * 3 + 8),
            ];

            let b2 = [
                complex2_load_f64(input, base_input_idx + width * 4 + 8),
                complex2_load_f64(input, base_input_idx + width * 5 + 8),
                complex2_load_f64(input, base_input_idx + width * 6 + 8),
                complex2_load_f64(input, base_input_idx + width * 7 + 8),
            ];

            let transposed_a2 = avx_transpose_f64x2_4x4_impl(
                (a2[0], _mm256_setzero_pd()),
                (a2[1], _mm256_setzero_pd()),
                (a2[2], _mm256_setzero_pd()),
                (a2[3], _mm256_setzero_pd()),
            );

            let transposed_b2 = avx_transpose_f64x2_4x4_impl(
                (b2[0], _mm256_setzero_pd()),
                (b2[1], _mm256_setzero_pd()),
                (b2[2], _mm256_setzero_pd()),
                (b2[3], _mm256_setzero_pd()),
            );

            complex4_store_f64(output, HEIGHT * y + x_rev[8], transposed_a2.0);
            complex4_store_f64(output, HEIGHT * y + x_rev[9], transposed_a2.1);

            complex4_store_f64(output, HEIGHT * y + x_rev[8] + 4, transposed_b2.0);
            complex4_store_f64(output, HEIGHT * y + x_rev[9] + 4, transposed_b2.1);

            let c0 = [
                complex4_load_f64(input, base_input_idx + width * 8),
                complex4_load_f64(input, base_input_idx + width * 9),
            ];

            let c1 = [
                complex4_load_f64(input, base_input_idx + width * 8 + 4),
                complex4_load_f64(input, base_input_idx + width * 9 + 4),
            ];

            let transposed_c0 = avx_transpose_f64x2_4x4_impl(
                c0[0],
                c0[1],
                (_mm256_setzero_pd(), _mm256_setzero_pd()),
                (_mm256_setzero_pd(), _mm256_setzero_pd()),
            );

            let transposed_c1 = avx_transpose_f64x2_4x4_impl(
                c1[0],
                c1[1],
                (_mm256_setzero_pd(), _mm256_setzero_pd()),
                (_mm256_setzero_pd(), _mm256_setzero_pd()),
            );

            complex2_store_f64(output, HEIGHT * y + x_rev[0] + 8, transposed_c0.0.0);
            complex2_store_f64(output, HEIGHT * y + x_rev[1] + 8, transposed_c0.1.0);
            complex2_store_f64(output, HEIGHT * y + x_rev[2] + 8, transposed_c0.2.0);
            complex2_store_f64(output, HEIGHT * y + x_rev[3] + 8, transposed_c0.3.0);

            complex2_store_f64(output, HEIGHT * y + x_rev[4] + 8, transposed_c1.0.0);
            complex2_store_f64(output, HEIGHT * y + x_rev[5] + 8, transposed_c1.1.0);
            complex2_store_f64(output, HEIGHT * y + x_rev[6] + 8, transposed_c1.2.0);
            complex2_store_f64(output, HEIGHT * y + x_rev[7] + 8, transposed_c1.3.0);

            let c2 = [
                complex2_load_f64(input, base_input_idx + width * 8 + 8),
                complex2_load_f64(input, base_input_idx + width * 9 + 8),
            ];

            let transposed_c2 = transpose_f64x2_2x2(c2[0], c2[1]);

            complex2_store_f64(output, HEIGHT * y + x_rev[8] + 8, transposed_c2.0);
            complex2_store_f64(output, HEIGHT * y + x_rev[9] + 8, transposed_c2.1);
        }
    }
}

impl AvxFmaRadix10d {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                avx_bitreversed_transpose_f64_radix10(self.butterfly_length, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.butterfly_length;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 10;
                    let tenth = len / 10;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        const HI_LO: i32 = 0b0010_0001;
                        const LO_LO: i32 = 0b0010_0000;

                        while j + 2 < tenth {
                            let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let twi = 9 * j;
                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 2..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );
                            let tw3 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 6..).as_ptr().cast(),
                            );
                            let tw4 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 8..).as_ptr().cast(),
                            );
                            let tw5 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 10..).as_ptr().cast(),
                            );
                            let tw6 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 12..).as_ptr().cast(),
                            );
                            let tw7 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 14..).as_ptr().cast(),
                            );
                            let tw8 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 16..).as_ptr().cast(),
                            );

                            let u1 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(data.get_unchecked(j + tenth..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 2 * tenth..).as_ptr().cast(),
                                ),
                                tw1,
                            );
                            let u3 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 3 * tenth..).as_ptr().cast(),
                                ),
                                tw2,
                            );
                            let u4 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 4 * tenth..).as_ptr().cast(),
                                ),
                                tw3,
                            );
                            let u5 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 5 * tenth..).as_ptr().cast(),
                                ),
                                tw4,
                            );
                            let u6 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 6 * tenth..).as_ptr().cast(),
                                ),
                                tw5,
                            );
                            let u7 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 7 * tenth..).as_ptr().cast(),
                                ),
                                tw6,
                            );
                            let u8 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 8 * tenth..).as_ptr().cast(),
                                ),
                                tw7,
                            );
                            let u9 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 9 * tenth..).as_ptr().cast(),
                                ),
                                tw8,
                            );

                            let mid0 = self.bf5.exec(u0, u2, u4, u6, u8);
                            let mid1 = self.bf5.exec(u5, u7, u9, u1, u3);

                            let (y0, y1) = AvxButterfly::butterfly2_f64(mid0.0, mid1.0);
                            let (y2, y3) = AvxButterfly::butterfly2_f64(mid0.1, mid1.1);
                            let (y4, y5) = AvxButterfly::butterfly2_f64(mid0.2, mid1.2);
                            let (y6, y7) = AvxButterfly::butterfly2_f64(mid0.3, mid1.3);
                            let (y8, y9) = AvxButterfly::butterfly2_f64(mid0.4, mid1.4);

                            // Store results
                            _mm256_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + tenth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 2 * tenth..).as_mut_ptr().cast(),
                                y4,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 3 * tenth..).as_mut_ptr().cast(),
                                y7,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 4 * tenth..).as_mut_ptr().cast(),
                                y8,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 5 * tenth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 6 * tenth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 7 * tenth..).as_mut_ptr().cast(),
                                y5,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 8 * tenth..).as_mut_ptr().cast(),
                                y6,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 9 * tenth..).as_mut_ptr().cast(),
                                y9,
                            );

                            j += 2;
                        }

                        for j in j..tenth {
                            let u0 = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let twi = 9 * j;
                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 2..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );
                            let tw3 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 6..).as_ptr().cast(),
                            );
                            let tw4 =
                                _mm_loadu_pd(m_twiddles.get_unchecked(twi + 8..).as_ptr().cast());

                            let u1u2 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(data.get_unchecked(j + tenth..).as_ptr().cast()),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 2 * tenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let u3u4 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 3 * tenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 4 * tenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw1,
                            );
                            let u5u6 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 5 * tenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 6 * tenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw2,
                            );
                            let u7u8 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 7 * tenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 8 * tenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw3,
                            );
                            let u9 = _mm_fcmul_pd(
                                _mm_loadu_pd(data.get_unchecked(j + 9 * tenth..).as_ptr().cast()),
                                tw4,
                            );

                            let u0u5 =
                                _mm256_permute2f128_pd::<LO_LO>(_mm256_castpd128_pd256(u0), u5u6);
                            let u2u7 = _mm256_permute2f128_pd::<HI_LO>(u1u2, u7u8);
                            let u4u9 =
                                _mm256_permute2f128_pd::<HI_LO>(u3u4, _mm256_castpd128_pd256(u9));
                            let u6u1 = _mm256_permute2f128_pd::<HI_LO>(u5u6, u1u2);
                            let u8u3 = _mm256_permute2f128_pd::<HI_LO>(u7u8, u3u4);

                            let mid0mid1 = self.bf5.exec(u0u5, u2u7, u4u9, u6u1, u8u3);

                            let (y0, y1) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(mid0mid1.0),
                                _mm256_extractf128_pd::<1>(mid0mid1.0),
                            );
                            let (y2, y3) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(mid0mid1.1),
                                _mm256_extractf128_pd::<1>(mid0mid1.1),
                            );
                            let (y4, y5) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(mid0mid1.2),
                                _mm256_extractf128_pd::<1>(mid0mid1.2),
                            );
                            let (y6, y7) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(mid0mid1.3),
                                _mm256_extractf128_pd::<1>(mid0mid1.3),
                            );
                            let (y8, y9) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(mid0mid1.4),
                                _mm256_extractf128_pd::<1>(mid0mid1.4),
                            );

                            // Store results
                            _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + tenth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 2 * tenth..).as_mut_ptr().cast(),
                                y4,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 3 * tenth..).as_mut_ptr().cast(),
                                y7,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 4 * tenth..).as_mut_ptr().cast(),
                                y8,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 5 * tenth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 6 * tenth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 7 * tenth..).as_mut_ptr().cast(),
                                y5,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 8 * tenth..).as_mut_ptr().cast(),
                                y6,
                            );
                            _mm_storeu_pd(
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
}

impl FftExecutor<f64> for AvxFmaRadix10d {
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
fn complex2_load_f32(array: &[Complex<f32>], idx: usize) -> __m256 {
    unsafe { _mm256_castps128_ps256(_mm_loadu_ps(array.get_unchecked(idx..).as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn complex2_store_f32(array: &mut [Complex<f32>], idx: usize, v: __m256) {
    unsafe {
        _mm_storeu_ps(
            array.get_unchecked_mut(idx..).as_mut_ptr().cast(),
            _mm256_castps256_ps128(v),
        );
    }
}

#[target_feature(enable = "avx2")]
fn avx_bitreversed_transpose_f32_radix10(
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

    let rev_digits = compute_logarithm::<10>(width).unwrap();
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
        // ^ T
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

            let transposed_a0 = avx_transpose_f32x2_4x4_impl(a0[0], a0[1], a0[2], a0[3]);
            let transposed_a1 = avx_transpose_f32x2_4x4_impl(a1[0], a1[1], a1[2], a1[3]);

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

            let transposed_b0 = avx_transpose_f32x2_4x4_impl(b0[0], b0[1], b0[2], b0[3]);
            let transposed_b1 = avx_transpose_f32x2_4x4_impl(b1[0], b1[1], b1[2], b1[3]);

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

            let transposed_a2 = avx_transpose_f32x2_4x4_impl(a2[0], a2[1], a2[2], a2[3]);

            let transposed_b2 = avx_transpose_f32x2_4x4_impl(b2[0], b2[1], b2[2], b2[3]);

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

            let transposed_c0 = avx_transpose_f32x2_4x4_impl(
                c0[0],
                c0[1],
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            );

            let transposed_c1 = avx_transpose_f32x2_4x4_impl(
                c1[0],
                c1[1],
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            );

            complex2_store_f32(output, HEIGHT * y + x_rev[0] + 8, transposed_c0.0);
            complex2_store_f32(output, HEIGHT * y + x_rev[1] + 8, transposed_c0.1);
            complex2_store_f32(output, HEIGHT * y + x_rev[2] + 8, transposed_c0.2);
            complex2_store_f32(output, HEIGHT * y + x_rev[3] + 8, transposed_c0.3);

            complex2_store_f32(output, HEIGHT * y + x_rev[4] + 8, transposed_c1.0);
            complex2_store_f32(output, HEIGHT * y + x_rev[5] + 8, transposed_c1.1);
            complex2_store_f32(output, HEIGHT * y + x_rev[6] + 8, transposed_c1.2);
            complex2_store_f32(output, HEIGHT * y + x_rev[7] + 8, transposed_c1.3);

            let c2 = [
                complex2_load_f32(input, base_input_idx + width * 8 + 8),
                complex2_load_f32(input, base_input_idx + width * 9 + 8),
            ];

            let transposed_c2 = transpose_f32_2x2_impl((c2[0], c2[1]));

            complex2_store_f32(output, HEIGHT * y + x_rev[8] + 8, transposed_c2.0);
            complex2_store_f32(output, HEIGHT * y + x_rev[9] + 8, transposed_c2.1);
        }
    }
}

pub(crate) struct AvxFmaRadix10f {
    twiddles: Vec<Complex<f32>>,
    execution_length: usize,
    bf5: AvxFastButterfly5f,
    direction: FftDirection,
    butterfly: Arc<dyn CompositeFftExecutor<f32> + Send + Sync>,
    butterfly_length: usize,
}

impl AvxFmaRadix10f {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix10f, ZaftError> {
        assert!(
            is_power_of_ten(size as u64),
            "Input length must be a power of 10"
        );

        let log10 = compute_logarithm::<10>(size).unwrap();
        let butterfly = match log10 {
            0 => f32::butterfly1(fft_direction)?,
            1 => f32::butterfly10(fft_direction)?,
            _ => f32::butterfly100(fft_direction)
                .map_or_else(|| f32::butterfly10(fft_direction), Ok)?,
        };

        let butterfly_length = butterfly.length();

        let twiddles = create_avx4_twiddles::<f32, 10>(butterfly_length, size, fft_direction)?;

        Ok(AvxFmaRadix10f {
            execution_length: size,
            twiddles,
            bf5: unsafe { AvxFastButterfly5f::new(fft_direction) },
            direction: fft_direction,
            butterfly,
            butterfly_length,
        })
    }
}

impl AvxFmaRadix10f {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                avx_bitreversed_transpose_f32_radix10(self.butterfly_length, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = self.butterfly_length;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 10;
                    let tenth = len / 10;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        const HI_LO: i32 = 0b0010_0001;
                        const LO_LO: i32 = 0b0010_0000;

                        while j + 4 < tenth {
                            let twi = 9 * j;
                            let tw0 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw1 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 8..).as_ptr().cast(),
                            );
                            let tw3 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 12..).as_ptr().cast(),
                            );
                            let tw4 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 16..).as_ptr().cast(),
                            );
                            let tw5 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 20..).as_ptr().cast(),
                            );

                            let mut u1111 =
                                _mm256_loadu_ps(data.get_unchecked(j + tenth..).as_ptr().cast());
                            let mut u2222 = _mm256_loadu_ps(
                                data.get_unchecked(j + 2 * tenth..).as_ptr().cast(),
                            );
                            let mut u3333 = _mm256_loadu_ps(
                                data.get_unchecked(j + 3 * tenth..).as_ptr().cast(),
                            );
                            let mut u4444 = _mm256_loadu_ps(
                                data.get_unchecked(j + 4 * tenth..).as_ptr().cast(),
                            );
                            let mut u5555 = _mm256_loadu_ps(
                                data.get_unchecked(j + 5 * tenth..).as_ptr().cast(),
                            );
                            let mut u6666 = _mm256_loadu_ps(
                                data.get_unchecked(j + 6 * tenth..).as_ptr().cast(),
                            );
                            let mut u7777 = _mm256_loadu_ps(
                                data.get_unchecked(j + 7 * tenth..).as_ptr().cast(),
                            );
                            let mut u8888 = _mm256_loadu_ps(
                                data.get_unchecked(j + 8 * tenth..).as_ptr().cast(),
                            );

                            u1111 = _mm256_fcmul_ps(u1111, tw0);
                            u2222 = _mm256_fcmul_ps(u2222, tw1);
                            u3333 = _mm256_fcmul_ps(u3333, tw2);
                            u4444 = _mm256_fcmul_ps(u4444, tw3);
                            u5555 = _mm256_fcmul_ps(u5555, tw4);

                            let tw6 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 24..).as_ptr().cast(),
                            );
                            let tw7 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 28..).as_ptr().cast(),
                            );
                            let tw8 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 32..).as_ptr().cast(),
                            );

                            u6666 = _mm256_fcmul_ps(u6666, tw5);
                            u7777 = _mm256_fcmul_ps(u7777, tw6);
                            u8888 = _mm256_fcmul_ps(u8888, tw7);

                            let u9 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 9 * tenth..).as_ptr().cast(),
                                ),
                                tw8,
                            );

                            let u0000 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                            let mid0 = self.bf5._m256_exec(u0000, u2222, u4444, u6666, u8888);
                            let mid1 = self.bf5._m256_exec(u5555, u7777, u9, u1111, u3333);

                            // Since this is good-thomas algorithm, we don't need twiddle factors
                            let (y0, y1) = AvxButterfly::butterfly2_f32(mid0.0, mid1.0);
                            let (y2, y3) = AvxButterfly::butterfly2_f32(mid0.1, mid1.1);
                            let (y4, y5) = AvxButterfly::butterfly2_f32(mid0.2, mid1.2);
                            let (y6, y7) = AvxButterfly::butterfly2_f32(mid0.3, mid1.3);
                            let (y8, y9) = AvxButterfly::butterfly2_f32(mid0.4, mid1.4);

                            // Store results
                            _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + tenth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 2 * tenth..).as_mut_ptr().cast(),
                                y4,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 3 * tenth..).as_mut_ptr().cast(),
                                y7,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 4 * tenth..).as_mut_ptr().cast(),
                                y8,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 5 * tenth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 6 * tenth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 7 * tenth..).as_mut_ptr().cast(),
                                y5,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 8 * tenth..).as_mut_ptr().cast(),
                                y6,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 9 * tenth..).as_mut_ptr().cast(),
                                y9,
                            );

                            j += 4;
                        }

                        while j + 2 < tenth {
                            let u0 = _mm_loadu_ps(data.get_unchecked(j..).as_ptr().cast());
                            let twi = 9 * j;
                            let tw0 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw2 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );
                            let tw3 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 8..).as_ptr().cast(),
                            );
                            let tw4 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 12..).as_ptr().cast(),
                            );
                            let tw5 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(twi + 16..).as_ptr().cast());

                            let u1u2 = _mm256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(data.get_unchecked(j + tenth..).as_ptr().cast()),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 2 * tenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let u3u4 = _mm256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 3 * tenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 4 * tenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw2,
                            );
                            let u5u6 = _mm256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 5 * tenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 6 * tenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw3,
                            );
                            let u7u8 = _mm256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 7 * tenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 8 * tenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw4,
                            );

                            let u9 = _mm_fcmul_ps(
                                _mm_loadu_ps(data.get_unchecked(j + 9 * tenth..).as_ptr().cast()),
                                tw5,
                            );

                            let u0u5 =
                                _mm256_permute2f128_ps::<LO_LO>(_mm256_castps128_ps256(u0), u5u6);
                            let u2u7 = _mm256_permute2f128_ps::<HI_LO>(u1u2, u7u8);
                            let u4u9 =
                                _mm256_permute2f128_ps::<HI_LO>(u3u4, _mm256_castps128_ps256(u9));
                            let u6u1 = _mm256_permute2f128_ps::<HI_LO>(u5u6, u1u2);
                            let u8u3 = _mm256_permute2f128_ps::<HI_LO>(u7u8, u3u4);

                            let mid0mid1 = self.bf5._m256_exec(u0u5, u2u7, u4u9, u6u1, u8u3);

                            // Since this is good-thomas algorithm, we don't need twiddle factors
                            let (y0, y1) = AvxButterfly::butterfly2_f32_m128(
                                _mm256_castps256_ps128(mid0mid1.0),
                                _mm256_extractf128_ps::<1>(mid0mid1.0),
                            );
                            let (y2, y3) = AvxButterfly::butterfly2_f32_m128(
                                _mm256_castps256_ps128(mid0mid1.1),
                                _mm256_extractf128_ps::<1>(mid0mid1.1),
                            );
                            let (y4, y5) = AvxButterfly::butterfly2_f32_m128(
                                _mm256_castps256_ps128(mid0mid1.2),
                                _mm256_extractf128_ps::<1>(mid0mid1.2),
                            );
                            let (y6, y7) = AvxButterfly::butterfly2_f32_m128(
                                _mm256_castps256_ps128(mid0mid1.3),
                                _mm256_extractf128_ps::<1>(mid0mid1.3),
                            );
                            let (y8, y9) = AvxButterfly::butterfly2_f32_m128(
                                _mm256_castps256_ps128(mid0mid1.4),
                                _mm256_extractf128_ps::<1>(mid0mid1.4),
                            );

                            // Store results
                            _mm_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + tenth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 2 * tenth..).as_mut_ptr().cast(),
                                y4,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 3 * tenth..).as_mut_ptr().cast(),
                                y7,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 4 * tenth..).as_mut_ptr().cast(),
                                y8,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 5 * tenth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 6 * tenth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 7 * tenth..).as_mut_ptr().cast(),
                                y5,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 8 * tenth..).as_mut_ptr().cast(),
                                y6,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 9 * tenth..).as_mut_ptr().cast(),
                                y9,
                            );

                            j += 2;
                        }

                        for j in j..tenth {
                            let u0 = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());
                            let twi = 9 * j;
                            let tw0tw1tw2tw3 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw4tw5tw6tw7 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );
                            let tw8 = _mm_castsi128_ps(_mm_loadu_si64(
                                m_twiddles.get_unchecked(twi + 8..).as_ptr().cast(),
                            ));

                            let u1u2u3u4 = _mm256_fcmul_ps(
                                _mm256_load4_f32x2(
                                    data.get_unchecked(j + tenth..),
                                    data.get_unchecked(j + 2 * tenth..),
                                    data.get_unchecked(j + 3 * tenth..),
                                    data.get_unchecked(j + 4 * tenth..),
                                ),
                                tw0tw1tw2tw3,
                            );
                            let u5u6u7u8 = _mm256_fcmul_ps(
                                _mm256_load4_f32x2(
                                    data.get_unchecked(j + 5 * tenth..),
                                    data.get_unchecked(j + 6 * tenth..),
                                    data.get_unchecked(j + 7 * tenth..),
                                    data.get_unchecked(j + 8 * tenth..),
                                ),
                                tw4tw5tw6tw7,
                            );

                            let u9 = _mm_fcmul_ps(
                                _m128s_load_f32x2(
                                    data.get_unchecked(j + 9 * tenth..).as_ptr().cast(),
                                ),
                                tw8,
                            );

                            let u1u2 = _mm256_castps256_ps128(u1u2u3u4);
                            let u7u8 = _mm256_extractf128_ps::<1>(u5u6u7u8);
                            let u5u6 = _mm256_castps256_ps128(u5u6u7u8);
                            let u3u4 = _mm256_extractf128_ps::<1>(u1u2u3u4);

                            let u0u5 = _mm_unpacklo_ps64(u0, u5u6);
                            let u2u7 = _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(u1u2, u7u8);
                            let u4u9 = _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(u3u4, u9);
                            let u6u1 = _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(u5u6, u1u2);
                            let u8u3 = _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(u7u8, u3u4);

                            let mid0mid1 = self.bf5.exec(u0u5, u2u7, u4u9, u6u1, u8u3);

                            // Since this is good-thomas algorithm, we don't need twiddle factors
                            let (y0, y1) = AvxButterfly::butterfly2_f32_m128(
                                mid0mid1.0,
                                _mm_unpackhi_ps64(mid0mid1.0, mid0mid1.0),
                            );
                            let (y2, y3) = AvxButterfly::butterfly2_f32_m128(
                                mid0mid1.1,
                                _mm_unpackhi_ps64(mid0mid1.1, mid0mid1.1),
                            );
                            let (y4, y5) = AvxButterfly::butterfly2_f32_m128(
                                mid0mid1.2,
                                _mm_unpackhi_ps64(mid0mid1.2, mid0mid1.2),
                            );
                            let (y6, y7) = AvxButterfly::butterfly2_f32_m128(
                                mid0mid1.3,
                                _mm_unpackhi_ps64(mid0mid1.3, mid0mid1.3),
                            );
                            let (y8, y9) = AvxButterfly::butterfly2_f32_m128(
                                mid0mid1.4,
                                _mm_unpackhi_ps64(mid0mid1.4, mid0mid1.4),
                            );

                            // Store results
                            _m128s_store_f32x2(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + tenth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 2 * tenth..).as_mut_ptr().cast(),
                                y4,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 3 * tenth..).as_mut_ptr().cast(),
                                y7,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 4 * tenth..).as_mut_ptr().cast(),
                                y8,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 5 * tenth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 6 * tenth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 7 * tenth..).as_mut_ptr().cast(),
                                y5,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 8 * tenth..).as_mut_ptr().cast(),
                                y6,
                            );
                            _m128s_store_f32x2(
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
}

impl FftExecutor<f32> for AvxFmaRadix10f {
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

    test_avx_radix!(test_avx_radix10, f32, AvxFmaRadix10f, 3, 10, 1e-3);
    test_avx_radix!(test_avx_radix10_f64, f64, AvxFmaRadix10d, 3, 10, 1e-8);
}
