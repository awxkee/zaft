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
use crate::avx::butterflies::AvxButterfly;
use crate::avx::radix3::{
    complex3_load_f32, complex3_load_f64, complex3_store_f32, complex3_store_f64,
};
use crate::avx::radix4::{
    complex4_load_f32, complex4_load_f64, complex4_store_f32, complex4_store_f64,
};
use crate::avx::rotate::AvxRotate;
use crate::avx::transpose::{avx_transpose_f32x2_4x4_impl, avx_transpose_f64x2_4x4_impl};
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_fcmul_ps, _mm_unpackhi_ps64, _mm_unpacklo_ps64,
    _mm256_create_pd, _mm256_create_ps, _mm256_fcmul_pd, _mm256_fcmul_ps, _mm256_load4_f32x2,
    create_avx4_twiddles,
};
use crate::err::try_vec;
use crate::radix11::Radix11Twiddles;
use crate::util::{
    compute_twiddle, int_logarithm, is_power_of_eleven, reverse_bits, validate_oof_sizes,
    validate_scratch,
};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::arch::x86_64::*;
use std::sync::Arc;

pub(crate) struct AvxFmaRadix11<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    direction: FftDirection,
    butterfly: Arc<dyn FftExecutor<T> + Send + Sync>,
    butterfly_length: usize,
}

impl<T: FftSample + Radix11Twiddles> AvxFmaRadix11<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix11<T>, ZaftError> {
        assert!(
            is_power_of_eleven(size as u64),
            "Input length must be a power of 11"
        );

        let log11 = int_logarithm::<11>(size).unwrap();
        let butterfly = match log11 {
            0 => T::butterfly1(fft_direction)?,
            1 => T::butterfly11(fft_direction)?,
            _ => {
                T::butterfly121(fft_direction).map_or_else(|| T::butterfly11(fft_direction), Ok)?
            }
        };

        let butterfly_length = butterfly.length();

        let twiddles = create_avx4_twiddles::<T, 11>(butterfly_length, size, fft_direction)?;

        Ok(AvxFmaRadix11 {
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 11, fft_direction),
            twiddle2: compute_twiddle(2, 11, fft_direction),
            twiddle3: compute_twiddle(3, 11, fft_direction),
            twiddle4: compute_twiddle(4, 11, fft_direction),
            twiddle5: compute_twiddle(5, 11, fft_direction),
            direction: fft_direction,
            butterfly,
            butterfly_length,
        })
    }
}

#[target_feature(enable = "avx2")]
fn avx_bitreversed_transpose_f64_radix11(
    height: usize,
    input: &[Complex<f64>],
    output: &mut [Complex<f64>],
) {
    let width = input.len() / height;

    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }

    const WIDTH: usize = 11;
    const HEIGHT: usize = 11;

    let rev_digits = int_logarithm::<11>(width).unwrap();
    let strided_width = width / WIDTH;
    let strided_height = height / HEIGHT;

    for x in 0..strided_width {
        // Graphically, the 11×11 matrix is partitioned as:
        // +--------+--------+--------+
        // | 4×4 A0 | 4×4 A1 | 4×3 A2 |
        // +--------+--------+--------+
        // | 4×4 B0 | 4×4 B1 | 4×3 B2 |
        // +--------+--------+--------+
        // | 3×4 C0 | 3×4 C1 | 3×3 C2 |
        // +--------+--------+--------+
        // ^ T
        // +--------+--------+--------+
        // | 4×4 A0ᵀ | 4×4 B0ᵀ | 3×4 C0ᵀ |
        // +--------+--------+--------+
        // | 4×4 A1ᵀ | 4×4 B1ᵀ | 3×4 C1ᵀ |
        // +--------+--------+--------+
        // | 4×3 A2ᵀ | 4×3 B2ᵀ | 3×3 C2ᵀ |
        // +--------+--------+--------+
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
            reverse_bits::<WIDTH>(WIDTH * x + 10, rev_digits) * height,
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
                complex3_load_f64(input, base_input_idx + 8),
                complex3_load_f64(input, base_input_idx + width + 8),
                complex3_load_f64(input, base_input_idx + width * 2 + 8),
                complex3_load_f64(input, base_input_idx + width * 3 + 8),
            ];

            let b2 = [
                complex3_load_f64(input, base_input_idx + width * 4 + 8),
                complex3_load_f64(input, base_input_idx + width * 5 + 8),
                complex3_load_f64(input, base_input_idx + width * 6 + 8),
                complex3_load_f64(input, base_input_idx + width * 7 + 8),
            ];

            let transposed_a2 = avx_transpose_f64x2_4x4_impl(a2[0], a2[1], a2[2], a2[3]);
            let transposed_b2 = avx_transpose_f64x2_4x4_impl(b2[0], b2[1], b2[2], b2[3]);

            complex4_store_f64(output, HEIGHT * y + x_rev[8], transposed_a2.0);
            complex4_store_f64(output, HEIGHT * y + x_rev[9], transposed_a2.1);
            complex4_store_f64(output, HEIGHT * y + x_rev[10], transposed_a2.2);

            complex4_store_f64(output, HEIGHT * y + x_rev[8] + 4, transposed_b2.0);
            complex4_store_f64(output, HEIGHT * y + x_rev[9] + 4, transposed_b2.1);
            complex4_store_f64(output, HEIGHT * y + x_rev[10] + 4, transposed_b2.2);

            let c0 = [
                complex4_load_f64(input, base_input_idx + width * 8),
                complex4_load_f64(input, base_input_idx + width * 9),
                complex4_load_f64(input, base_input_idx + width * 10),
            ];

            let c1 = [
                complex4_load_f64(input, base_input_idx + width * 8 + 4),
                complex4_load_f64(input, base_input_idx + width * 9 + 4),
                complex4_load_f64(input, base_input_idx + width * 10 + 4),
            ];

            let transposed_c0 = avx_transpose_f64x2_4x4_impl(
                c0[0],
                c0[1],
                c0[2],
                (_mm256_setzero_pd(), _mm256_setzero_pd()),
            );

            let transposed_c1 = avx_transpose_f64x2_4x4_impl(
                c1[0],
                c1[1],
                c1[2],
                (_mm256_setzero_pd(), _mm256_setzero_pd()),
            );

            complex3_store_f64(output, HEIGHT * y + x_rev[0] + 8, transposed_c0.0);
            complex3_store_f64(output, HEIGHT * y + x_rev[1] + 8, transposed_c0.1);
            complex3_store_f64(output, HEIGHT * y + x_rev[2] + 8, transposed_c0.2);
            complex3_store_f64(output, HEIGHT * y + x_rev[3] + 8, transposed_c0.3);

            complex3_store_f64(output, HEIGHT * y + x_rev[4] + 8, transposed_c1.0);
            complex3_store_f64(output, HEIGHT * y + x_rev[5] + 8, transposed_c1.1);
            complex3_store_f64(output, HEIGHT * y + x_rev[6] + 8, transposed_c1.2);
            complex3_store_f64(output, HEIGHT * y + x_rev[7] + 8, transposed_c1.3);

            let c2 = [
                complex3_load_f64(input, base_input_idx + width * 8 + 8),
                complex3_load_f64(input, base_input_idx + width * 9 + 8),
                complex3_load_f64(input, base_input_idx + width * 10 + 8),
            ];

            let transposed_c2 = avx_transpose_f64x2_4x4_impl(
                c2[0],
                c2[1],
                c2[2],
                (_mm256_setzero_pd(), _mm256_setzero_pd()),
            );

            complex3_store_f64(output, HEIGHT * y + x_rev[8] + 8, transposed_c2.0);
            complex3_store_f64(output, HEIGHT * y + x_rev[9] + 8, transposed_c2.1);
            complex3_store_f64(output, HEIGHT * y + x_rev[10] + 8, transposed_c2.2);
        }
    }
}

impl AvxFmaRadix11<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn base_run(&self, chunk: &mut [Complex<f64>]) {
        unsafe {
            let rotate = AvxRotate::<f64>::new(FftDirection::Inverse);

            let mut len = self.butterfly_length;

            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                let columns = len;
                len *= 11;
                let eleventh = len / 11;

                for data in chunk.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 2 <= eleventh {
                        let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                        let twi = 10 * j;
                        let tw0 = _mm256_loadu_pd(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                        let tw1 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 2..).as_ptr().cast());
                        let tw2 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 4..).as_ptr().cast());
                        let tw3 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 6..).as_ptr().cast());
                        let tw4 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 8..).as_ptr().cast());

                        let tw5 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 10..).as_ptr().cast());
                        let tw6 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 12..).as_ptr().cast());
                        let tw7 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 14..).as_ptr().cast());
                        let tw8 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 16..).as_ptr().cast());
                        let tw9 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 18..).as_ptr().cast());

                        let u1 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + eleventh..).as_ptr().cast()),
                            tw0,
                        );
                        let u2 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + 2 * eleventh..).as_ptr().cast()),
                            tw1,
                        );
                        let u3 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + 3 * eleventh..).as_ptr().cast()),
                            tw2,
                        );
                        let u4 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + 4 * eleventh..).as_ptr().cast()),
                            tw3,
                        );
                        let u5 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + 5 * eleventh..).as_ptr().cast()),
                            tw4,
                        );
                        let u6 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + 6 * eleventh..).as_ptr().cast()),
                            tw5,
                        );
                        let u7 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + 7 * eleventh..).as_ptr().cast()),
                            tw6,
                        );
                        let u8 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + 8 * eleventh..).as_ptr().cast()),
                            tw7,
                        );
                        let u9 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(data.get_unchecked(j + 9 * eleventh..).as_ptr().cast()),
                            tw8,
                        );
                        let u10 = _mm256_fcmul_pd(
                            _mm256_loadu_pd(
                                data.get_unchecked(j + 10 * eleventh..).as_ptr().cast(),
                            ),
                            tw9,
                        );

                        let y00 = u0;
                        let (x1p10, x1m10) = AvxButterfly::butterfly2_f64(u1, u10); // u1, u10
                        let x1m10 = rotate.rotate_m256d(x1m10);
                        let y00 = _mm256_add_pd(y00, x1p10);
                        let (x2p9, x2m9) = AvxButterfly::butterfly2_f64(u2, u9); // u2, u9
                        let x2m9 = rotate.rotate_m256d(x2m9);
                        let y00 = _mm256_add_pd(y00, x2p9);
                        let (x3p8, x3m8) = AvxButterfly::butterfly2_f64(u3, u8); // u3, u8
                        let x3m8 = rotate.rotate_m256d(x3m8);
                        let y00 = _mm256_add_pd(y00, x3p8);
                        let (x4p7, x4m7) = AvxButterfly::butterfly2_f64(u4, u7); // u4, u7
                        let x4m7 = rotate.rotate_m256d(x4m7);
                        let y00 = _mm256_add_pd(y00, x4p7);
                        let (x5p6, x5m6) = AvxButterfly::butterfly2_f64(u5, u6); // u5, u6
                        let x5m6 = rotate.rotate_m256d(x5m6);
                        let y00 = _mm256_add_pd(y00, x5p6);

                        let m0110a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle1.re), u0);
                        let m0110a =
                            _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle2.re), x2p9, m0110a);
                        let m0110a =
                            _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle3.re), x3p8, m0110a);
                        let m0110a =
                            _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle4.re), x4p7, m0110a);
                        let m0110a =
                            _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle5.re), x5p6, m0110a);
                        let m0110b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle1.im));
                        let m0110b =
                            _mm256_fmadd_pd(x2m9, _mm256_set1_pd(self.twiddle2.im), m0110b);
                        let m0110b =
                            _mm256_fmadd_pd(x3m8, _mm256_set1_pd(self.twiddle3.im), m0110b);
                        let m0110b =
                            _mm256_fmadd_pd(x4m7, _mm256_set1_pd(self.twiddle4.im), m0110b);
                        let m0110b =
                            _mm256_fmadd_pd(x5m6, _mm256_set1_pd(self.twiddle5.im), m0110b);
                        let (y01, y10) = AvxButterfly::butterfly2_f64(m0110a, m0110b);

                        let m0209a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle2.re), u0);
                        let m0209a =
                            _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle4.re), m0209a);
                        let m0209a =
                            _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle5.re), m0209a);
                        let m0209a =
                            _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle3.re), m0209a);
                        let m0209a =
                            _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle1.re), m0209a);
                        let m0209b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle2.im));
                        let m0209b =
                            _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle4.im), x2m9, m0209b);
                        let m0209b =
                            _mm256_fnmadd_pd(x3m8, _mm256_set1_pd(self.twiddle5.im), m0209b);
                        let m0209b =
                            _mm256_fnmadd_pd(x4m7, _mm256_set1_pd(self.twiddle3.im), m0209b);
                        let m0209b =
                            _mm256_fnmadd_pd(x5m6, _mm256_set1_pd(self.twiddle1.im), m0209b);
                        let (y02, y09) = AvxButterfly::butterfly2_f64(m0209a, m0209b);

                        let m0308a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle3.re), u0);
                        let m0308a =
                            _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle5.re), m0308a);
                        let m0308a =
                            _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle2.re), m0308a);
                        let m0308a =
                            _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle1.re), m0308a);
                        let m0308a =
                            _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle4.re), m0308a);
                        let m0308b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle3.im));
                        let m0308b =
                            _mm256_fnmadd_pd(x2m9, _mm256_set1_pd(self.twiddle5.im), m0308b);
                        let m0308b =
                            _mm256_fnmadd_pd(x3m8, _mm256_set1_pd(self.twiddle2.im), m0308b);
                        let m0308b =
                            _mm256_fmadd_pd(x4m7, _mm256_set1_pd(self.twiddle1.im), m0308b);
                        let m0308b =
                            _mm256_fmadd_pd(x5m6, _mm256_set1_pd(self.twiddle4.im), m0308b);
                        let (y03, y08) = AvxButterfly::butterfly2_f64(m0308a, m0308b);

                        let m0407a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle4.re), u0);
                        let m0407a =
                            _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle3.re), m0407a);
                        let m0407a =
                            _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle1.re), m0407a);
                        let m0407a =
                            _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle5.re), m0407a);
                        let m0407a =
                            _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle2.re), m0407a);
                        let m0407b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle4.im));
                        let m0407b =
                            _mm256_fnmadd_pd(x2m9, _mm256_set1_pd(self.twiddle3.im), m0407b);
                        let m0407b =
                            _mm256_fmadd_pd(x3m8, _mm256_set1_pd(self.twiddle1.im), m0407b);
                        let m0407b =
                            _mm256_fmadd_pd(x4m7, _mm256_set1_pd(self.twiddle5.im), m0407b);
                        let m0407b =
                            _mm256_fnmadd_pd(x5m6, _mm256_set1_pd(self.twiddle2.im), m0407b);
                        let (y04, y07) = AvxButterfly::butterfly2_f64(m0407a, m0407b);

                        let m0506a = _mm256_fmadd_pd(x1p10, _mm256_set1_pd(self.twiddle5.re), u0);
                        let m0506a =
                            _mm256_fmadd_pd(x2p9, _mm256_set1_pd(self.twiddle1.re), m0506a);
                        let m0506a =
                            _mm256_fmadd_pd(x3p8, _mm256_set1_pd(self.twiddle4.re), m0506a);
                        let m0506a =
                            _mm256_fmadd_pd(x4p7, _mm256_set1_pd(self.twiddle2.re), m0506a);
                        let m0506a =
                            _mm256_fmadd_pd(x5p6, _mm256_set1_pd(self.twiddle3.re), m0506a);
                        let m0506b = _mm256_mul_pd(x1m10, _mm256_set1_pd(self.twiddle5.im));
                        let m0506b =
                            _mm256_fnmadd_pd(x2m9, _mm256_set1_pd(self.twiddle1.im), m0506b);
                        let m0506b =
                            _mm256_fmadd_pd(x3m8, _mm256_set1_pd(self.twiddle4.im), m0506b);
                        let m0506b =
                            _mm256_fnmadd_pd(x4m7, _mm256_set1_pd(self.twiddle2.im), m0506b);
                        let m0506b =
                            _mm256_fmadd_pd(x5m6, _mm256_set1_pd(self.twiddle3.im), m0506b);
                        let (y05, y06) = AvxButterfly::butterfly2_f64(m0506a, m0506b);

                        // // Store results
                        _mm256_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + eleventh..).as_mut_ptr().cast(),
                            y01,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 2 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y02,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 3 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 4 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 5 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y05,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 6 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y06,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 7 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y07,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 8 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y08,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 9 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y09,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 10 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y10,
                        );

                        j += 2;
                    }

                    for j in j..eleventh {
                        let u0 = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                        let twi = 10 * j;
                        let tw0 = _mm256_loadu_pd(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                        let tw1 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 2..).as_ptr().cast());
                        let tw2 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 4..).as_ptr().cast());
                        let tw3 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 6..).as_ptr().cast());
                        let tw4 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 8..).as_ptr().cast());

                        let u1u2 = _mm256_fcmul_pd(
                            _mm256_create_pd(
                                _mm_loadu_pd(data.get_unchecked(j + eleventh..).as_ptr().cast()),
                                _mm_loadu_pd(
                                    data.get_unchecked(j + 2 * eleventh..).as_ptr().cast(),
                                ),
                            ),
                            tw0,
                        );
                        let u3u4 = _mm256_fcmul_pd(
                            _mm256_create_pd(
                                _mm_loadu_pd(
                                    data.get_unchecked(j + 3 * eleventh..).as_ptr().cast(),
                                ),
                                _mm_loadu_pd(
                                    data.get_unchecked(j + 4 * eleventh..).as_ptr().cast(),
                                ),
                            ),
                            tw1,
                        );
                        let u5u6 = _mm256_fcmul_pd(
                            _mm256_create_pd(
                                _mm_loadu_pd(
                                    data.get_unchecked(j + 5 * eleventh..).as_ptr().cast(),
                                ),
                                _mm_loadu_pd(
                                    data.get_unchecked(j + 6 * eleventh..).as_ptr().cast(),
                                ),
                            ),
                            tw2,
                        );
                        let u7u8 = _mm256_fcmul_pd(
                            _mm256_create_pd(
                                _mm_loadu_pd(
                                    data.get_unchecked(j + 7 * eleventh..).as_ptr().cast(),
                                ),
                                _mm_loadu_pd(
                                    data.get_unchecked(j + 8 * eleventh..).as_ptr().cast(),
                                ),
                            ),
                            tw3,
                        );
                        let u9u10 = _mm256_fcmul_pd(
                            _mm256_create_pd(
                                _mm_loadu_pd(
                                    data.get_unchecked(j + 9 * eleventh..).as_ptr().cast(),
                                ),
                                _mm_loadu_pd(
                                    data.get_unchecked(j + 10 * eleventh..).as_ptr().cast(),
                                ),
                            ),
                            tw4,
                        );

                        let y00 = u0;
                        let (x1p10, x1m10) = AvxButterfly::butterfly2_f64_m128(
                            _mm256_castpd256_pd128(u1u2),
                            _mm256_extractf128_pd::<1>(u9u10),
                        ); // u1, u10
                        let x1m10 = rotate.rotate_m128d(x1m10);
                        let y00 = _mm_add_pd(y00, x1p10);
                        let (x2p9, x2m9) = AvxButterfly::butterfly2_f64_m128(
                            _mm256_extractf128_pd::<1>(u1u2),
                            _mm256_castpd256_pd128(u9u10),
                        ); // u2, u9
                        let x2m9 = rotate.rotate_m128d(x2m9);
                        let y00 = _mm_add_pd(y00, x2p9);
                        let (x3p8, x3m8) = AvxButterfly::butterfly2_f64_m128(
                            _mm256_castpd256_pd128(u3u4),
                            _mm256_extractf128_pd::<1>(u7u8),
                        ); // u3, u8
                        let x3m8 = rotate.rotate_m128d(x3m8);
                        let y00 = _mm_add_pd(y00, x3p8);
                        let (x4p7, x4m7) = AvxButterfly::butterfly2_f64_m128(
                            _mm256_extractf128_pd::<1>(u3u4),
                            _mm256_castpd256_pd128(u7u8),
                        ); // u4, u7
                        let x4m7 = rotate.rotate_m128d(x4m7);
                        let y00 = _mm_add_pd(y00, x4p7);
                        let (x5p6, x5m6) = AvxButterfly::butterfly2_f64_m128(
                            _mm256_castpd256_pd128(u5u6),
                            _mm256_extractf128_pd::<1>(u5u6),
                        ); // u5, u6
                        let x5m6 = rotate.rotate_m128d(x5m6);
                        let y00 = _mm_add_pd(y00, x5p6);

                        let m0110a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle1.re), u0);
                        let m0110a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle2.re), x2p9, m0110a);
                        let m0110a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle3.re), x3p8, m0110a);
                        let m0110a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle4.re), x4p7, m0110a);
                        let m0110a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle5.re), x5p6, m0110a);
                        let m0110b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle1.im));
                        let m0110b = _mm_fmadd_pd(x2m9, _mm_set1_pd(self.twiddle2.im), m0110b);
                        let m0110b = _mm_fmadd_pd(x3m8, _mm_set1_pd(self.twiddle3.im), m0110b);
                        let m0110b = _mm_fmadd_pd(x4m7, _mm_set1_pd(self.twiddle4.im), m0110b);
                        let m0110b = _mm_fmadd_pd(x5m6, _mm_set1_pd(self.twiddle5.im), m0110b);
                        let (y01, y10) = AvxButterfly::butterfly2_f64_m128(m0110a, m0110b);

                        let m0209a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle2.re), u0);
                        let m0209a = _mm_fmadd_pd(x2p9, _mm_set1_pd(self.twiddle4.re), m0209a);
                        let m0209a = _mm_fmadd_pd(x3p8, _mm_set1_pd(self.twiddle5.re), m0209a);
                        let m0209a = _mm_fmadd_pd(x4p7, _mm_set1_pd(self.twiddle3.re), m0209a);
                        let m0209a = _mm_fmadd_pd(x5p6, _mm_set1_pd(self.twiddle1.re), m0209a);
                        let m0209b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle2.im));
                        let m0209b = _mm_fmadd_pd(_mm_set1_pd(self.twiddle4.im), x2m9, m0209b);
                        let m0209b = _mm_fnmadd_pd(x3m8, _mm_set1_pd(self.twiddle5.im), m0209b);
                        let m0209b = _mm_fnmadd_pd(x4m7, _mm_set1_pd(self.twiddle3.im), m0209b);
                        let m0209b = _mm_fnmadd_pd(x5m6, _mm_set1_pd(self.twiddle1.im), m0209b);
                        let (y02, y09) = AvxButterfly::butterfly2_f64_m128(m0209a, m0209b);

                        let m0308a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle3.re), u0);
                        let m0308a = _mm_fmadd_pd(x2p9, _mm_set1_pd(self.twiddle5.re), m0308a);
                        let m0308a = _mm_fmadd_pd(x3p8, _mm_set1_pd(self.twiddle2.re), m0308a);
                        let m0308a = _mm_fmadd_pd(x4p7, _mm_set1_pd(self.twiddle1.re), m0308a);
                        let m0308a = _mm_fmadd_pd(x5p6, _mm_set1_pd(self.twiddle4.re), m0308a);
                        let m0308b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle3.im));
                        let m0308b = _mm_fnmadd_pd(x2m9, _mm_set1_pd(self.twiddle5.im), m0308b);
                        let m0308b = _mm_fnmadd_pd(x3m8, _mm_set1_pd(self.twiddle2.im), m0308b);
                        let m0308b = _mm_fmadd_pd(x4m7, _mm_set1_pd(self.twiddle1.im), m0308b);
                        let m0308b = _mm_fmadd_pd(x5m6, _mm_set1_pd(self.twiddle4.im), m0308b);
                        let (y03, y08) = AvxButterfly::butterfly2_f64_m128(m0308a, m0308b);

                        let m0407a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle4.re), u0);
                        let m0407a = _mm_fmadd_pd(x2p9, _mm_set1_pd(self.twiddle3.re), m0407a);
                        let m0407a = _mm_fmadd_pd(x3p8, _mm_set1_pd(self.twiddle1.re), m0407a);
                        let m0407a = _mm_fmadd_pd(x4p7, _mm_set1_pd(self.twiddle5.re), m0407a);
                        let m0407a = _mm_fmadd_pd(x5p6, _mm_set1_pd(self.twiddle2.re), m0407a);
                        let m0407b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle4.im));
                        let m0407b = _mm_fnmadd_pd(x2m9, _mm_set1_pd(self.twiddle3.im), m0407b);
                        let m0407b = _mm_fmadd_pd(x3m8, _mm_set1_pd(self.twiddle1.im), m0407b);
                        let m0407b = _mm_fmadd_pd(x4m7, _mm_set1_pd(self.twiddle5.im), m0407b);
                        let m0407b = _mm_fnmadd_pd(x5m6, _mm_set1_pd(self.twiddle2.im), m0407b);
                        let (y04, y07) = AvxButterfly::butterfly2_f64_m128(m0407a, m0407b);

                        let m0506a = _mm_fmadd_pd(x1p10, _mm_set1_pd(self.twiddle5.re), u0);
                        let m0506a = _mm_fmadd_pd(x2p9, _mm_set1_pd(self.twiddle1.re), m0506a);
                        let m0506a = _mm_fmadd_pd(x3p8, _mm_set1_pd(self.twiddle4.re), m0506a);
                        let m0506a = _mm_fmadd_pd(x4p7, _mm_set1_pd(self.twiddle2.re), m0506a);
                        let m0506a = _mm_fmadd_pd(x5p6, _mm_set1_pd(self.twiddle3.re), m0506a);
                        let m0506b = _mm_mul_pd(x1m10, _mm_set1_pd(self.twiddle5.im));
                        let m0506b = _mm_fnmadd_pd(x2m9, _mm_set1_pd(self.twiddle1.im), m0506b);
                        let m0506b = _mm_fmadd_pd(x3m8, _mm_set1_pd(self.twiddle4.im), m0506b);
                        let m0506b = _mm_fnmadd_pd(x4m7, _mm_set1_pd(self.twiddle2.im), m0506b);
                        let m0506b = _mm_fmadd_pd(x5m6, _mm_set1_pd(self.twiddle3.im), m0506b);
                        let (y05, y06) = AvxButterfly::butterfly2_f64_m128(m0506a, m0506b);

                        // // Store results
                        _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + eleventh..).as_mut_ptr().cast(),
                            y01,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 2 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y02,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 3 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 4 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 5 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y05,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 6 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y06,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 7 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y07,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 8 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y08,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 9 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y09,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 10 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y10,
                        );
                    }
                }

                m_twiddles = &m_twiddles[columns * 10..];
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
            avx_bitreversed_transpose_f64_radix11(self.butterfly_length, chunk, scratch);
            self.butterfly.execute_out_of_place(scratch, chunk)?;
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
            avx_bitreversed_transpose_f64_radix11(self.butterfly_length, src, dst);
            self.butterfly.execute(dst)?;
            self.base_run(dst);
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix11<f64> {
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

#[target_feature(enable = "avx2")]
fn avx_bitreversed_transpose_f32_radix11(
    height: usize,
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
) {
    let width = input.len() / height;

    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }

    const WIDTH: usize = 11;
    const HEIGHT: usize = 11;

    let rev_digits = int_logarithm::<11>(width).unwrap();
    let strided_width = width / WIDTH;
    let strided_height = height / HEIGHT;

    for x in 0..strided_width {
        // Graphically, the 11×11 matrix is partitioned as:
        // +--------+--------+--------+
        // | 4×4 A0 | 4×4 A1 | 4×3 A2 |
        // +--------+--------+--------+
        // | 4×4 B0 | 4×4 B1 | 4×3 B2 |
        // +--------+--------+--------+
        // | 3×4 C0 | 3×4 C1 | 3×3 C2 |
        // +--------+--------+--------+
        // ^ T
        // +--------+--------+--------+
        // | 4×4 A0ᵀ | 4×4 B0ᵀ | 3×4 C0ᵀ |
        // +--------+--------+--------+
        // | 4×4 A1ᵀ | 4×4 B1ᵀ | 3×4 C1ᵀ |
        // +--------+--------+--------+
        // | 4×3 A2ᵀ | 4×3 B2ᵀ | 3×3 C2ᵀ |
        // +--------+--------+--------+
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
            reverse_bits::<WIDTH>(WIDTH * x + 10, rev_digits) * height,
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
                complex3_load_f32(input, base_input_idx + 8),
                complex3_load_f32(input, base_input_idx + width + 8),
                complex3_load_f32(input, base_input_idx + width * 2 + 8),
                complex3_load_f32(input, base_input_idx + width * 3 + 8),
            ];

            let b2 = [
                complex3_load_f32(input, base_input_idx + width * 4 + 8),
                complex3_load_f32(input, base_input_idx + width * 5 + 8),
                complex3_load_f32(input, base_input_idx + width * 6 + 8),
                complex3_load_f32(input, base_input_idx + width * 7 + 8),
            ];

            let transposed_a2 = avx_transpose_f32x2_4x4_impl(a2[0], a2[1], a2[2], a2[3]);
            let transposed_b2 = avx_transpose_f32x2_4x4_impl(b2[0], b2[1], b2[2], b2[3]);

            complex4_store_f32(output, HEIGHT * y + x_rev[8], transposed_a2.0);
            complex4_store_f32(output, HEIGHT * y + x_rev[9], transposed_a2.1);
            complex4_store_f32(output, HEIGHT * y + x_rev[10], transposed_a2.2);

            complex4_store_f32(output, HEIGHT * y + x_rev[8] + 4, transposed_b2.0);
            complex4_store_f32(output, HEIGHT * y + x_rev[9] + 4, transposed_b2.1);
            complex4_store_f32(output, HEIGHT * y + x_rev[10] + 4, transposed_b2.2);

            let c0 = [
                complex4_load_f32(input, base_input_idx + width * 8),
                complex4_load_f32(input, base_input_idx + width * 9),
                complex4_load_f32(input, base_input_idx + width * 10),
            ];

            let c1 = [
                complex4_load_f32(input, base_input_idx + width * 8 + 4),
                complex4_load_f32(input, base_input_idx + width * 9 + 4),
                complex4_load_f32(input, base_input_idx + width * 10 + 4),
            ];

            let transposed_c0 =
                avx_transpose_f32x2_4x4_impl(c0[0], c0[1], c0[2], _mm256_setzero_ps());

            let transposed_c1 =
                avx_transpose_f32x2_4x4_impl(c1[0], c1[1], c1[2], _mm256_setzero_ps());

            complex3_store_f32(output, HEIGHT * y + x_rev[0] + 8, transposed_c0.0);
            complex3_store_f32(output, HEIGHT * y + x_rev[1] + 8, transposed_c0.1);
            complex3_store_f32(output, HEIGHT * y + x_rev[2] + 8, transposed_c0.2);
            complex3_store_f32(output, HEIGHT * y + x_rev[3] + 8, transposed_c0.3);

            complex3_store_f32(output, HEIGHT * y + x_rev[4] + 8, transposed_c1.0);
            complex3_store_f32(output, HEIGHT * y + x_rev[5] + 8, transposed_c1.1);
            complex3_store_f32(output, HEIGHT * y + x_rev[6] + 8, transposed_c1.2);
            complex3_store_f32(output, HEIGHT * y + x_rev[7] + 8, transposed_c1.3);

            let c2 = [
                complex3_load_f32(input, base_input_idx + width * 8 + 8),
                complex3_load_f32(input, base_input_idx + width * 9 + 8),
                complex3_load_f32(input, base_input_idx + width * 10 + 8),
            ];

            let transposed_c2 =
                avx_transpose_f32x2_4x4_impl(c2[0], c2[1], c2[2], _mm256_setzero_ps());

            complex3_store_f32(output, HEIGHT * y + x_rev[8] + 8, transposed_c2.0);
            complex3_store_f32(output, HEIGHT * y + x_rev[9] + 8, transposed_c2.1);
            complex3_store_f32(output, HEIGHT * y + x_rev[10] + 8, transposed_c2.2);
        }
    }
}

impl AvxFmaRadix11<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn base_run(&self, chunk: &mut [Complex<f32>]) {
        unsafe {
            let rotate = AvxRotate::<f32>::new(FftDirection::Inverse);

            let mut len = self.butterfly_length;

            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                let columns = len;
                len *= 11;
                let eleventh = len / 11;

                for data in chunk.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 4 <= eleventh {
                        let twi = 10 * j;
                        let tw0 = _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                        let tw1 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 4..).as_ptr().cast());
                        let tw2 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 8..).as_ptr().cast());
                        let tw3 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 12..).as_ptr().cast());
                        let tw4 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 16..).as_ptr().cast());

                        let u0000 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());
                        let mut u1111 =
                            _mm256_loadu_ps(data.get_unchecked(j + eleventh..).as_ptr().cast());
                        let mut u2222 =
                            _mm256_loadu_ps(data.get_unchecked(j + 2 * eleventh..).as_ptr().cast());
                        let mut u3333 =
                            _mm256_loadu_ps(data.get_unchecked(j + 3 * eleventh..).as_ptr().cast());
                        let mut u4444 =
                            _mm256_loadu_ps(data.get_unchecked(j + 4 * eleventh..).as_ptr().cast());
                        let mut u5555 =
                            _mm256_loadu_ps(data.get_unchecked(j + 5 * eleventh..).as_ptr().cast());
                        let mut u6666 =
                            _mm256_loadu_ps(data.get_unchecked(j + 6 * eleventh..).as_ptr().cast());
                        let mut u7777 =
                            _mm256_loadu_ps(data.get_unchecked(j + 7 * eleventh..).as_ptr().cast());
                        let mut u8888 =
                            _mm256_loadu_ps(data.get_unchecked(j + 8 * eleventh..).as_ptr().cast());
                        let mut u9999 =
                            _mm256_loadu_ps(data.get_unchecked(j + 9 * eleventh..).as_ptr().cast());
                        let mut u101010 = _mm256_loadu_ps(
                            data.get_unchecked(j + 10 * eleventh..).as_ptr().cast(),
                        );

                        u1111 = _mm256_fcmul_ps(u1111, tw0);
                        u2222 = _mm256_fcmul_ps(u2222, tw1);
                        u3333 = _mm256_fcmul_ps(u3333, tw2);
                        u4444 = _mm256_fcmul_ps(u4444, tw3);
                        u5555 = _mm256_fcmul_ps(u5555, tw4);

                        let tw5 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 20..).as_ptr().cast());
                        let tw6 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 24..).as_ptr().cast());
                        let tw7 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 28..).as_ptr().cast());
                        let tw8 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 32..).as_ptr().cast());
                        let tw9 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 36..).as_ptr().cast());

                        u6666 = _mm256_fcmul_ps(u6666, tw5);
                        u7777 = _mm256_fcmul_ps(u7777, tw6);
                        u8888 = _mm256_fcmul_ps(u8888, tw7);
                        u9999 = _mm256_fcmul_ps(u9999, tw8);
                        u101010 = _mm256_fcmul_ps(u101010, tw9);

                        let y00 = u0000;
                        let (x1p10, x1m10) = AvxButterfly::butterfly2_f32(u1111, u101010); // u1, u10
                        let x1m10 = rotate.rotate_m256(x1m10);
                        let y00 = _mm256_add_ps(y00, x1p10);
                        let (x2p9, x2m9) = AvxButterfly::butterfly2_f32(u2222, u9999); // u2, u9
                        let x2m9 = rotate.rotate_m256(x2m9);
                        let y00 = _mm256_add_ps(y00, x2p9);
                        let (x3p8, x3m8) = AvxButterfly::butterfly2_f32(u3333, u8888); // u3, u8
                        let x3m8 = rotate.rotate_m256(x3m8);
                        let y00 = _mm256_add_ps(y00, x3p8);
                        let (x4p7, x4m7) = AvxButterfly::butterfly2_f32(u4444, u7777); // u4, u7
                        let x4m7 = rotate.rotate_m256(x4m7);
                        let y00 = _mm256_add_ps(y00, x4p7);
                        let (x5p6, x5m6) = AvxButterfly::butterfly2_f32(u5555, u6666); // u5, u6
                        let x5m6 = rotate.rotate_m256(x5m6);
                        let y00 = _mm256_add_ps(y00, x5p6);

                        let m0110a =
                            _mm256_fmadd_ps(x1p10, _mm256_set1_ps(self.twiddle1.re), u0000);
                        let m0110a =
                            _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle2.re), x2p9, m0110a);
                        let m0110a =
                            _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle3.re), x3p8, m0110a);
                        let m0110a =
                            _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle4.re), x4p7, m0110a);
                        let m0110a =
                            _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle5.re), x5p6, m0110a);
                        let m0110b = _mm256_mul_ps(x1m10, _mm256_set1_ps(self.twiddle1.im));
                        let m0110b =
                            _mm256_fmadd_ps(x2m9, _mm256_set1_ps(self.twiddle2.im), m0110b);
                        let m0110b =
                            _mm256_fmadd_ps(x3m8, _mm256_set1_ps(self.twiddle3.im), m0110b);
                        let m0110b =
                            _mm256_fmadd_ps(x4m7, _mm256_set1_ps(self.twiddle4.im), m0110b);
                        let m0110b =
                            _mm256_fmadd_ps(x5m6, _mm256_set1_ps(self.twiddle5.im), m0110b);
                        let (y01, y10) = AvxButterfly::butterfly2_f32(m0110a, m0110b);

                        let m0209a =
                            _mm256_fmadd_ps(x1p10, _mm256_set1_ps(self.twiddle2.re), u0000);
                        let m0209a =
                            _mm256_fmadd_ps(x2p9, _mm256_set1_ps(self.twiddle4.re), m0209a);
                        let m0209a =
                            _mm256_fmadd_ps(x3p8, _mm256_set1_ps(self.twiddle5.re), m0209a);
                        let m0209a =
                            _mm256_fmadd_ps(x4p7, _mm256_set1_ps(self.twiddle3.re), m0209a);
                        let m0209a =
                            _mm256_fmadd_ps(x5p6, _mm256_set1_ps(self.twiddle1.re), m0209a);
                        let m0209b = _mm256_mul_ps(x1m10, _mm256_set1_ps(self.twiddle2.im));
                        let m0209b =
                            _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle4.im), x2m9, m0209b);
                        let m0209b =
                            _mm256_fnmadd_ps(x3m8, _mm256_set1_ps(self.twiddle5.im), m0209b);
                        let m0209b =
                            _mm256_fnmadd_ps(x4m7, _mm256_set1_ps(self.twiddle3.im), m0209b);
                        let m0209b =
                            _mm256_fnmadd_ps(x5m6, _mm256_set1_ps(self.twiddle1.im), m0209b);
                        let (y02, y09) = AvxButterfly::butterfly2_f32(m0209a, m0209b);

                        let m0308a =
                            _mm256_fmadd_ps(x1p10, _mm256_set1_ps(self.twiddle3.re), u0000);
                        let m0308a =
                            _mm256_fmadd_ps(x2p9, _mm256_set1_ps(self.twiddle5.re), m0308a);
                        let m0308a =
                            _mm256_fmadd_ps(x3p8, _mm256_set1_ps(self.twiddle2.re), m0308a);
                        let m0308a =
                            _mm256_fmadd_ps(x4p7, _mm256_set1_ps(self.twiddle1.re), m0308a);
                        let m0308a =
                            _mm256_fmadd_ps(x5p6, _mm256_set1_ps(self.twiddle4.re), m0308a);
                        let m0308b = _mm256_mul_ps(x1m10, _mm256_set1_ps(self.twiddle3.im));
                        let m0308b =
                            _mm256_fnmadd_ps(x2m9, _mm256_set1_ps(self.twiddle5.im), m0308b);
                        let m0308b =
                            _mm256_fnmadd_ps(x3m8, _mm256_set1_ps(self.twiddle2.im), m0308b);
                        let m0308b =
                            _mm256_fmadd_ps(x4m7, _mm256_set1_ps(self.twiddle1.im), m0308b);
                        let m0308b =
                            _mm256_fmadd_ps(x5m6, _mm256_set1_ps(self.twiddle4.im), m0308b);
                        let (y03, y08) = AvxButterfly::butterfly2_f32(m0308a, m0308b);

                        let m0407a =
                            _mm256_fmadd_ps(x1p10, _mm256_set1_ps(self.twiddle4.re), u0000);
                        let m0407a =
                            _mm256_fmadd_ps(x2p9, _mm256_set1_ps(self.twiddle3.re), m0407a);
                        let m0407a =
                            _mm256_fmadd_ps(x3p8, _mm256_set1_ps(self.twiddle1.re), m0407a);
                        let m0407a =
                            _mm256_fmadd_ps(x4p7, _mm256_set1_ps(self.twiddle5.re), m0407a);
                        let m0407a =
                            _mm256_fmadd_ps(x5p6, _mm256_set1_ps(self.twiddle2.re), m0407a);
                        let m0407b = _mm256_mul_ps(x1m10, _mm256_set1_ps(self.twiddle4.im));
                        let m0407b =
                            _mm256_fnmadd_ps(x2m9, _mm256_set1_ps(self.twiddle3.im), m0407b);
                        let m0407b =
                            _mm256_fmadd_ps(x3m8, _mm256_set1_ps(self.twiddle1.im), m0407b);
                        let m0407b =
                            _mm256_fmadd_ps(x4m7, _mm256_set1_ps(self.twiddle5.im), m0407b);
                        let m0407b =
                            _mm256_fnmadd_ps(x5m6, _mm256_set1_ps(self.twiddle2.im), m0407b);
                        let (y04, y07) = AvxButterfly::butterfly2_f32(m0407a, m0407b);

                        let m0506a =
                            _mm256_fmadd_ps(x1p10, _mm256_set1_ps(self.twiddle5.re), u0000);
                        let m0506a =
                            _mm256_fmadd_ps(x2p9, _mm256_set1_ps(self.twiddle1.re), m0506a);
                        let m0506a =
                            _mm256_fmadd_ps(x3p8, _mm256_set1_ps(self.twiddle4.re), m0506a);
                        let m0506a =
                            _mm256_fmadd_ps(x4p7, _mm256_set1_ps(self.twiddle2.re), m0506a);
                        let m0506a =
                            _mm256_fmadd_ps(x5p6, _mm256_set1_ps(self.twiddle3.re), m0506a);
                        let m0506b = _mm256_mul_ps(x1m10, _mm256_set1_ps(self.twiddle5.im));
                        let m0506b =
                            _mm256_fnmadd_ps(x2m9, _mm256_set1_ps(self.twiddle1.im), m0506b);
                        let m0506b =
                            _mm256_fmadd_ps(x3m8, _mm256_set1_ps(self.twiddle4.im), m0506b);
                        let m0506b =
                            _mm256_fnmadd_ps(x4m7, _mm256_set1_ps(self.twiddle2.im), m0506b);
                        let m0506b =
                            _mm256_fmadd_ps(x5m6, _mm256_set1_ps(self.twiddle3.im), m0506b);
                        let (y05, y06) = AvxButterfly::butterfly2_f32(m0506a, m0506b);

                        // Store results
                        _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + eleventh..).as_mut_ptr().cast(),
                            y01,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 2 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y02,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 3 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 4 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 5 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y05,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 6 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y06,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 7 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y07,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 8 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y08,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 9 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y09,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 10 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y10,
                        );

                        j += 4;
                    }

                    while j + 2 <= eleventh {
                        let u0 = _mm_loadu_ps(data.get_unchecked(j..).as_ptr().cast());
                        let twi = 10 * j;
                        let tw0 = _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                        let tw1 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 4..).as_ptr().cast());
                        let tw2 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 8..).as_ptr().cast());
                        let tw3 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 12..).as_ptr().cast());
                        let tw4 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 16..).as_ptr().cast());

                        let u1u2 = _mm256_fcmul_ps(
                            _mm256_create_ps(
                                _mm_loadu_ps(data.get_unchecked(j + eleventh..).as_ptr().cast()),
                                _mm_loadu_ps(
                                    data.get_unchecked(j + 2 * eleventh..).as_ptr().cast(),
                                ),
                            ),
                            tw0,
                        );
                        let u3u4 = _mm256_fcmul_ps(
                            _mm256_create_ps(
                                _mm_loadu_ps(
                                    data.get_unchecked(j + 3 * eleventh..).as_ptr().cast(),
                                ),
                                _mm_loadu_ps(
                                    data.get_unchecked(j + 4 * eleventh..).as_ptr().cast(),
                                ),
                            ),
                            tw1,
                        );
                        let u5u6 = _mm256_fcmul_ps(
                            _mm256_create_ps(
                                _mm_loadu_ps(
                                    data.get_unchecked(j + 5 * eleventh..).as_ptr().cast(),
                                ),
                                _mm_loadu_ps(
                                    data.get_unchecked(j + 6 * eleventh..).as_ptr().cast(),
                                ),
                            ),
                            tw2,
                        );
                        let u7u8 = _mm256_fcmul_ps(
                            _mm256_create_ps(
                                _mm_loadu_ps(
                                    data.get_unchecked(j + 7 * eleventh..).as_ptr().cast(),
                                ),
                                _mm_loadu_ps(
                                    data.get_unchecked(j + 8 * eleventh..).as_ptr().cast(),
                                ),
                            ),
                            tw3,
                        );
                        let u9u10 = _mm256_fcmul_ps(
                            _mm256_create_ps(
                                _mm_loadu_ps(
                                    data.get_unchecked(j + 9 * eleventh..).as_ptr().cast(),
                                ),
                                _mm_loadu_ps(
                                    data.get_unchecked(j + 10 * eleventh..).as_ptr().cast(),
                                ),
                            ),
                            tw4,
                        );

                        let y00 = u0;
                        let (x1p10, x1m10) = AvxButterfly::butterfly2_f32_m128(
                            _mm256_castps256_ps128(u1u2),
                            _mm256_extractf128_ps::<1>(u9u10),
                        ); // u1, u10
                        let x1m10 = rotate.rotate_m128(x1m10);
                        let y00 = _mm_add_ps(y00, x1p10);
                        let (x2p9, x2m9) = AvxButterfly::butterfly2_f32_m128(
                            _mm256_extractf128_ps::<1>(u1u2),
                            _mm256_castps256_ps128(u9u10),
                        ); // u2, u9
                        let x2m9 = rotate.rotate_m128(x2m9);
                        let y00 = _mm_add_ps(y00, x2p9);
                        let (x3p8, x3m8) = AvxButterfly::butterfly2_f32_m128(
                            _mm256_castps256_ps128(u3u4),
                            _mm256_extractf128_ps::<1>(u7u8),
                        ); // u3, u8
                        let x3m8 = rotate.rotate_m128(x3m8);
                        let y00 = _mm_add_ps(y00, x3p8);
                        let (x4p7, x4m7) = AvxButterfly::butterfly2_f32_m128(
                            _mm256_extractf128_ps::<1>(u3u4),
                            _mm256_castps256_ps128(u7u8),
                        ); // u4, u7
                        let x4m7 = rotate.rotate_m128(x4m7);
                        let y00 = _mm_add_ps(y00, x4p7);
                        let (x5p6, x5m6) = AvxButterfly::butterfly2_f32_m128(
                            _mm256_castps256_ps128(u5u6),
                            _mm256_extractf128_ps::<1>(u5u6),
                        ); // u5, u6
                        let x5m6 = rotate.rotate_m128(x5m6);
                        let y00 = _mm_add_ps(y00, x5p6);

                        let m0110a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle1.re), u0);
                        let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle2.re), x2p9, m0110a);
                        let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle3.re), x3p8, m0110a);
                        let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.re), x4p7, m0110a);
                        let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle5.re), x5p6, m0110a);
                        let m0110b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle1.im));
                        let m0110b = _mm_fmadd_ps(x2m9, _mm_set1_ps(self.twiddle2.im), m0110b);
                        let m0110b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle3.im), m0110b);
                        let m0110b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle4.im), m0110b);
                        let m0110b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle5.im), m0110b);
                        let (y01, y10) = AvxButterfly::butterfly2_f32_m128(m0110a, m0110b);

                        let m0209a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle2.re), u0);
                        let m0209a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle4.re), m0209a);
                        let m0209a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle5.re), m0209a);
                        let m0209a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle3.re), m0209a);
                        let m0209a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle1.re), m0209a);
                        let m0209b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle2.im));
                        let m0209b = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.im), x2m9, m0209b);
                        let m0209b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle5.im), m0209b);
                        let m0209b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle3.im), m0209b);
                        let m0209b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle1.im), m0209b);
                        let (y02, y09) = AvxButterfly::butterfly2_f32_m128(m0209a, m0209b);

                        let m0308a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle3.re), u0);
                        let m0308a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle5.re), m0308a);
                        let m0308a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle2.re), m0308a);
                        let m0308a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle1.re), m0308a);
                        let m0308a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle4.re), m0308a);
                        let m0308b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle3.im));
                        let m0308b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle5.im), m0308b);
                        let m0308b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle2.im), m0308b);
                        let m0308b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle1.im), m0308b);
                        let m0308b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle4.im), m0308b);
                        let (y03, y08) = AvxButterfly::butterfly2_f32_m128(m0308a, m0308b);

                        let m0407a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle4.re), u0);
                        let m0407a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle3.re), m0407a);
                        let m0407a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle1.re), m0407a);
                        let m0407a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle5.re), m0407a);
                        let m0407a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle2.re), m0407a);
                        let m0407b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle4.im));
                        let m0407b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle3.im), m0407b);
                        let m0407b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle1.im), m0407b);
                        let m0407b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle5.im), m0407b);
                        let m0407b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle2.im), m0407b);
                        let (y04, y07) = AvxButterfly::butterfly2_f32_m128(m0407a, m0407b);

                        let m0506a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle5.re), u0);
                        let m0506a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle1.re), m0506a);
                        let m0506a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle4.re), m0506a);
                        let m0506a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle2.re), m0506a);
                        let m0506a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle3.re), m0506a);
                        let m0506b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle5.im));
                        let m0506b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle1.im), m0506b);
                        let m0506b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle4.im), m0506b);
                        let m0506b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle2.im), m0506b);
                        let m0506b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle3.im), m0506b);
                        let (y05, y06) = AvxButterfly::butterfly2_f32_m128(m0506a, m0506b);

                        // Store results
                        _mm_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + eleventh..).as_mut_ptr().cast(),
                            y01,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 2 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y02,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 3 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 4 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 5 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y05,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 6 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y06,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 7 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y07,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 8 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y08,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 9 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y09,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 10 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y10,
                        );

                        j += 2;
                    }

                    for j in j..eleventh {
                        let u0 = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());
                        let twi = 10 * j;
                        let tw0tw1tw2tw3 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                        let tw4tw5tw6tw7 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 4..).as_ptr().cast());
                        let tw8tw9 =
                            _mm_loadu_ps(m_twiddles.get_unchecked(twi + 8..).as_ptr().cast());

                        let u1u2u3u4 = _mm256_fcmul_ps(
                            _mm256_load4_f32x2(
                                data.get_unchecked(j + eleventh..),
                                data.get_unchecked(j + 2 * eleventh..),
                                data.get_unchecked(j + 3 * eleventh..),
                                data.get_unchecked(j + 4 * eleventh..),
                            ),
                            tw0tw1tw2tw3,
                        );
                        let u5u6u7u8 = _mm256_fcmul_ps(
                            _mm256_load4_f32x2(
                                data.get_unchecked(j + 5 * eleventh..),
                                data.get_unchecked(j + 6 * eleventh..),
                                data.get_unchecked(j + 7 * eleventh..),
                                data.get_unchecked(j + 8 * eleventh..),
                            ),
                            tw4tw5tw6tw7,
                        );

                        let u9u10 = _mm_fcmul_ps(
                            _mm_unpacklo_ps64(
                                _m128s_load_f32x2(
                                    data.get_unchecked(j + 9 * eleventh..).as_ptr().cast(),
                                ),
                                _m128s_load_f32x2(
                                    data.get_unchecked(j + 10 * eleventh..).as_ptr().cast(),
                                ),
                            ),
                            tw8tw9,
                        );

                        let u3u4 = _mm256_extractf128_ps::<1>(u1u2u3u4);
                        let u7u8 = _mm256_extractf128_ps::<1>(u5u6u7u8);
                        let u5u6 = _mm256_castps256_ps128(u5u6u7u8);

                        let y00 = u0;
                        let (x1p10, x1m10) = AvxButterfly::butterfly2_f32_m128(
                            _mm256_castps256_ps128(u1u2u3u4),
                            _mm_unpackhi_ps64(u9u10, u9u10),
                        ); // u1, u10
                        let x1m10 = rotate.rotate_m128(x1m10);
                        let y00 = _mm_add_ps(y00, x1p10);
                        let (x2p9, x2m9) = AvxButterfly::butterfly2_f32_m128(
                            _mm_unpackhi_ps64(
                                _mm256_castps256_ps128(u1u2u3u4),
                                _mm256_castps256_ps128(u1u2u3u4),
                            ),
                            u9u10,
                        ); // u2, u9
                        let x2m9 = rotate.rotate_m128(x2m9);
                        let y00 = _mm_add_ps(y00, x2p9);
                        let (x3p8, x3m8) =
                            AvxButterfly::butterfly2_f32_m128(u3u4, _mm_unpackhi_ps64(u7u8, u7u8)); // u3, u8
                        let x3m8 = rotate.rotate_m128(x3m8);
                        let y00 = _mm_add_ps(y00, x3p8);
                        let (x4p7, x4m7) =
                            AvxButterfly::butterfly2_f32_m128(_mm_unpackhi_ps64(u3u4, u3u4), u7u8); // u4, u7
                        let x4m7 = rotate.rotate_m128(x4m7);
                        let y00 = _mm_add_ps(y00, x4p7);
                        let (x5p6, x5m6) =
                            AvxButterfly::butterfly2_f32_m128(u5u6, _mm_unpackhi_ps64(u5u6, u5u6)); // u5, u6
                        let x5m6 = rotate.rotate_m128(x5m6);
                        let y00 = _mm_add_ps(y00, x5p6);

                        let m0110a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle1.re), u0);
                        let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle2.re), x2p9, m0110a);
                        let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle3.re), x3p8, m0110a);
                        let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.re), x4p7, m0110a);
                        let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle5.re), x5p6, m0110a);
                        let m0110b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle1.im));
                        let m0110b = _mm_fmadd_ps(x2m9, _mm_set1_ps(self.twiddle2.im), m0110b);
                        let m0110b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle3.im), m0110b);
                        let m0110b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle4.im), m0110b);
                        let m0110b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle5.im), m0110b);
                        let (y01, y10) = AvxButterfly::butterfly2_f32_m128(m0110a, m0110b);

                        let m0209a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle2.re), u0);
                        let m0209a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle4.re), m0209a);
                        let m0209a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle5.re), m0209a);
                        let m0209a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle3.re), m0209a);
                        let m0209a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle1.re), m0209a);
                        let m0209b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle2.im));
                        let m0209b = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.im), x2m9, m0209b);
                        let m0209b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle5.im), m0209b);
                        let m0209b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle3.im), m0209b);
                        let m0209b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle1.im), m0209b);
                        let (y02, y09) = AvxButterfly::butterfly2_f32_m128(m0209a, m0209b);

                        let m0308a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle3.re), u0);
                        let m0308a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle5.re), m0308a);
                        let m0308a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle2.re), m0308a);
                        let m0308a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle1.re), m0308a);
                        let m0308a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle4.re), m0308a);
                        let m0308b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle3.im));
                        let m0308b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle5.im), m0308b);
                        let m0308b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle2.im), m0308b);
                        let m0308b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle1.im), m0308b);
                        let m0308b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle4.im), m0308b);
                        let (y03, y08) = AvxButterfly::butterfly2_f32_m128(m0308a, m0308b);

                        let m0407a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle4.re), u0);
                        let m0407a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle3.re), m0407a);
                        let m0407a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle1.re), m0407a);
                        let m0407a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle5.re), m0407a);
                        let m0407a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle2.re), m0407a);
                        let m0407b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle4.im));
                        let m0407b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle3.im), m0407b);
                        let m0407b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle1.im), m0407b);
                        let m0407b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle5.im), m0407b);
                        let m0407b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle2.im), m0407b);
                        let (y04, y07) = AvxButterfly::butterfly2_f32_m128(m0407a, m0407b);

                        let m0506a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle5.re), u0);
                        let m0506a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle1.re), m0506a);
                        let m0506a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle4.re), m0506a);
                        let m0506a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle2.re), m0506a);
                        let m0506a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle3.re), m0506a);
                        let m0506b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle5.im));
                        let m0506b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle1.im), m0506b);
                        let m0506b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle4.im), m0506b);
                        let m0506b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle2.im), m0506b);
                        let m0506b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle3.im), m0506b);
                        let (y05, y06) = AvxButterfly::butterfly2_f32_m128(m0506a, m0506b);

                        // Store results
                        _m128s_store_f32x2(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + eleventh..).as_mut_ptr().cast(),
                            y01,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 2 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y02,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 3 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 4 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 5 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y05,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 6 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y06,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 7 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y07,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 8 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y08,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 9 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y09,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 10 * eleventh..)
                                .as_mut_ptr()
                                .cast(),
                            y10,
                        );
                    }
                }

                m_twiddles = &m_twiddles[columns * 10..];
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
            avx_bitreversed_transpose_f32_radix11(self.butterfly_length, chunk, scratch);
            self.butterfly.execute_out_of_place(scratch, chunk)?;
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
            avx_bitreversed_transpose_f32_radix11(self.butterfly_length, src, dst);
            self.butterfly.execute(dst)?;
            self.base_run(dst);
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix11<f32> {
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
    use crate::avx::test_avx_radix;

    test_avx_radix!(test_avx_radix11, f32, AvxFmaRadix11, 4, 11, 1e-3);
    test_avx_radix!(test_avx_radix11_f64, f64, AvxFmaRadix11, 4, 11, 1e-8);
}
