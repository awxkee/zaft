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
use crate::neon::butterflies::NeonButterfly;
use crate::neon::radix3::{complex3_load_f32, complex3_store_f32};
use crate::neon::radix4::{complex4_load_f32, complex4_store_f32};
use crate::neon::transpose::transpose_f32x2_4x4;
use crate::neon::util::{
    create_neon_twiddles, v_rotate90_f32, v_rotate90_f64, vfcmulq_f32, vfcmulq_f64, vh_rotate90_f32,
};
use crate::radix7::Radix7Twiddles;
use crate::util::{
    bitreversed_transpose, compute_twiddle, int_logarithm, is_power_of_seven, reverse_bits,
    validate_oof_sizes, validate_scratch,
};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::arch::aarch64::*;
use std::sync::Arc;

pub(crate) struct NeonRadix7<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    direction: FftDirection,
    butterfly: Arc<dyn FftExecutor<T> + Send + Sync>,
    butterfly_length: usize,
}

impl<T: FftSample + Radix7Twiddles> NeonRadix7<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonRadix7<T>, ZaftError> {
        assert!(
            is_power_of_seven(size as u64),
            "Input length must be a power of 7"
        );

        let log7 = int_logarithm::<7>(size).unwrap();
        let butterfly = match log7 {
            0 => T::butterfly1(fft_direction)?,
            1 => T::butterfly7(fft_direction)?,
            _ => T::butterfly49(fft_direction).map_or_else(|| T::butterfly7(fft_direction), Ok)?,
        };

        let butterfly_length = butterfly.length();

        let twiddles = create_neon_twiddles::<T, 7>(butterfly_length, size, fft_direction)?;

        Ok(NeonRadix7 {
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 7, fft_direction),
            twiddle2: compute_twiddle(2, 7, fft_direction),
            twiddle3: compute_twiddle(3, 7, fft_direction),
            direction: fft_direction,
            butterfly,
            butterfly_length,
        })
    }
}

impl NeonRadix7<f64> {
    fn base_run(&self, chunk: &mut [Complex<f64>]) {
        unsafe {
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            let rot_sign = vld1q_f64(ROT_90.as_ptr());

            let mut len = self.butterfly_length;

            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                let columns = len;
                len *= 7;
                let seventh = len / 7;

                for data in chunk.chunks_exact_mut(len) {
                    for j in 0..seventh {
                        let tw0 = vld1q_f64(m_twiddles.get_unchecked(6 * j..).as_ptr().cast());
                        let tw1 = vld1q_f64(m_twiddles.get_unchecked(6 * j + 1..).as_ptr().cast());
                        let tw2 = vld1q_f64(m_twiddles.get_unchecked(6 * j + 2..).as_ptr().cast());
                        let tw3 = vld1q_f64(m_twiddles.get_unchecked(6 * j + 3..).as_ptr().cast());
                        let tw4 = vld1q_f64(m_twiddles.get_unchecked(6 * j + 4..).as_ptr().cast());
                        let tw5 = vld1q_f64(m_twiddles.get_unchecked(6 * j + 5..).as_ptr().cast());

                        let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                        let u1 = vfcmulq_f64(
                            vld1q_f64(data.get_unchecked(j + seventh..).as_ptr().cast()),
                            tw0,
                        );
                        let u2 = vfcmulq_f64(
                            vld1q_f64(data.get_unchecked(j + 2 * seventh..).as_ptr().cast()),
                            tw1,
                        );
                        let u3 = vfcmulq_f64(
                            vld1q_f64(data.get_unchecked(j + 3 * seventh..).as_ptr().cast()),
                            tw2,
                        );
                        let u4 = vfcmulq_f64(
                            vld1q_f64(data.get_unchecked(j + 4 * seventh..).as_ptr().cast()),
                            tw3,
                        );
                        let u5 = vfcmulq_f64(
                            vld1q_f64(data.get_unchecked(j + 5 * seventh..).as_ptr().cast()),
                            tw4,
                        );
                        let u6 = vfcmulq_f64(
                            vld1q_f64(data.get_unchecked(j + 6 * seventh..).as_ptr().cast()),
                            tw5,
                        );

                        // Radix-7 butterfly

                        let (x1p6, x1m6) = NeonButterfly::butterfly2_f64(u1, u6);
                        let x1m6 = v_rotate90_f64(x1m6, rot_sign);
                        let y00 = vaddq_f64(u0, x1p6);
                        let (x2p5, x2m5) = NeonButterfly::butterfly2_f64(u2, u5);
                        let x2m5 = v_rotate90_f64(x2m5, rot_sign);
                        let y00 = vaddq_f64(y00, x2p5);
                        let (x3p4, x3m4) = NeonButterfly::butterfly2_f64(u3, u4);
                        let x3m4 = v_rotate90_f64(x3m4, rot_sign);
                        let y00 = vaddq_f64(y00, x3p4);

                        let m0106a = vfmaq_n_f64(u0, x1p6, self.twiddle1.re);
                        let m0106a = vfmaq_n_f64(m0106a, x2p5, self.twiddle2.re);
                        let m0106a = vfmaq_n_f64(m0106a, x3p4, self.twiddle3.re);
                        let m0106b = vmulq_n_f64(x1m6, self.twiddle1.im);
                        let m0106b = vfmaq_n_f64(m0106b, x2m5, self.twiddle2.im);
                        let m0106b = vfmaq_n_f64(m0106b, x3m4, self.twiddle3.im);
                        let (y01, y06) = NeonButterfly::butterfly2_f64(m0106a, m0106b);

                        let m0205a = vfmaq_n_f64(u0, x1p6, self.twiddle2.re);
                        let m0205a = vfmaq_n_f64(m0205a, x2p5, self.twiddle3.re);
                        let m0205a = vfmaq_n_f64(m0205a, x3p4, self.twiddle1.re);
                        let m0205b = vmulq_n_f64(x1m6, self.twiddle2.im);
                        let m0205b = vfmsq_n_f64(m0205b, x2m5, self.twiddle3.im);
                        let m0205b = vfmsq_n_f64(m0205b, x3m4, self.twiddle1.im);
                        let (y02, y05) = NeonButterfly::butterfly2_f64(m0205a, m0205b);

                        let m0304a = vfmaq_n_f64(u0, x1p6, self.twiddle3.re);
                        let m0304a = vfmaq_n_f64(m0304a, x2p5, self.twiddle1.re);
                        let m0304a = vfmaq_n_f64(m0304a, x3p4, self.twiddle2.re);
                        let m0304b = vmulq_n_f64(x1m6, self.twiddle3.im);
                        let m0304b = vfmsq_n_f64(m0304b, x2m5, self.twiddle1.im);
                        let m0304b = vfmaq_n_f64(m0304b, x3m4, self.twiddle2.im);
                        let (y03, y04) = NeonButterfly::butterfly2_f64(m0304a, m0304b);

                        // Store results
                        vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        vst1q_f64(
                            data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                            y01,
                        );
                        vst1q_f64(
                            data.get_unchecked_mut(j + 2 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y02,
                        );
                        vst1q_f64(
                            data.get_unchecked_mut(j + 3 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        vst1q_f64(
                            data.get_unchecked_mut(j + 4 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        vst1q_f64(
                            data.get_unchecked_mut(j + 5 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y05,
                        );
                        vst1q_f64(
                            data.get_unchecked_mut(j + 6 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y06,
                        );
                    }
                }

                m_twiddles = &m_twiddles[columns * 6..];
            }
        }
    }
}

impl FftExecutor<f64> for NeonRadix7<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        let mut scratch = try_vec![Complex::zero(); self.scratch_length()];
        self.execute_with_scratch(in_place, &mut scratch)
    }

    fn execute_with_scratch(
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
            bitreversed_transpose::<Complex<f64>, 7>(self.butterfly_length, chunk, scratch);

            self.butterfly.execute_out_of_place(scratch, chunk)?;
            self.base_run(chunk);
        }
        Ok(())
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_with_scratch(src, dst, &mut [])
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        validate_oof_sizes!(src, dst, self.execution_length);

        for (dst, src) in dst
            .chunks_exact_mut(self.execution_length)
            .zip(src.chunks_exact(self.execution_length))
        {
            // Digit-reversal permutation
            bitreversed_transpose::<Complex<f64>, 7>(self.butterfly_length, src, dst);
            self.butterfly.execute(dst)?;
            self.base_run(dst);
        }
        Ok(())
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

pub(crate) fn neon_bitreversed_transpose_f32_radix7(
    height: usize,
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
) {
    let width = input.len() / height;

    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }

    const WIDTH: usize = 7;
    const HEIGHT: usize = 7;

    let rev_digits = int_logarithm::<7>(width).unwrap();
    let strided_width = width / WIDTH;
    let strided_height = height / HEIGHT;

    for x in 0..strided_width {
        let x_rev = [
            reverse_bits::<WIDTH>(WIDTH * x, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 1, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 2, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 3, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 4, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 5, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 6, rev_digits) * height,
        ];

        for y in 0..strided_height {
            // Graphically, the 7×7 matrix is partitioned as:
            // +--------+--------+
            // | 4×4 A0 | 3×4 A1 |
            // +--------+--------+
            // | 4×3 B0 | 3×3 B1 |
            // +--------+--------+
            // ^ T
            // +--------+--------+
            // | 4×4 A0ᵀ | 3×4 B0ᵀ |
            // +--------+--------+
            // | 4×3 A1ᵀ | 3×3 B1ᵀ |
            // +--------+--------+
            let base_input_idx = (WIDTH * x) + y * HEIGHT * width;
            let a0 = [
                complex4_load_f32(input, base_input_idx),
                complex4_load_f32(input, base_input_idx + width),
                complex4_load_f32(input, base_input_idx + width * 2),
                complex4_load_f32(input, base_input_idx + width * 3),
            ];
            let transposed_a0 = transpose_f32x2_4x4(a0[0], a0[1], a0[2], a0[3]);

            complex4_store_f32(output, HEIGHT * y + x_rev[0], transposed_a0.0);
            complex4_store_f32(output, HEIGHT * y + x_rev[1], transposed_a0.1);
            complex4_store_f32(output, HEIGHT * y + x_rev[2], transposed_a0.2);
            complex4_store_f32(output, HEIGHT * y + x_rev[3], transposed_a0.3);

            let a1 = [
                complex3_load_f32(input, base_input_idx + 4),
                complex3_load_f32(input, base_input_idx + width + 4),
                complex3_load_f32(input, base_input_idx + width * 2 + 4),
                complex3_load_f32(input, base_input_idx + width * 3 + 4),
            ];
            let transposed_a1 = transpose_f32x2_4x4(a1[0], a1[1], a1[2], a1[3]);

            complex4_store_f32(output, HEIGHT * y + x_rev[4], transposed_a1.0);
            complex4_store_f32(output, HEIGHT * y + x_rev[5], transposed_a1.1);
            complex4_store_f32(output, HEIGHT * y + x_rev[6], transposed_a1.2);

            let b0 = [
                complex4_load_f32(input, base_input_idx + width * 4),
                complex4_load_f32(input, base_input_idx + width * 5),
                complex4_load_f32(input, base_input_idx + width * 6),
            ];

            let transposed_b0 = transpose_f32x2_4x4(b0[0], b0[1], b0[2], unsafe {
                float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.))
            });

            complex3_store_f32(output, HEIGHT * y + x_rev[0] + 4, transposed_b0.0);
            complex3_store_f32(output, HEIGHT * y + x_rev[1] + 4, transposed_b0.1);
            complex3_store_f32(output, HEIGHT * y + x_rev[2] + 4, transposed_b0.2);
            complex3_store_f32(output, HEIGHT * y + x_rev[3] + 4, transposed_b0.3);

            let b1 = [
                complex3_load_f32(input, base_input_idx + width * 4 + 4),
                complex3_load_f32(input, base_input_idx + width * 5 + 4),
                complex3_load_f32(input, base_input_idx + width * 6 + 4),
            ];

            let transposed_b1 = transpose_f32x2_4x4(b1[0], b1[1], b1[2], unsafe {
                float32x4x2_t(vdupq_n_f32(0.), vdupq_n_f32(0.))
            });

            complex3_store_f32(output, HEIGHT * y + x_rev[4] + 4, transposed_b1.0);
            complex3_store_f32(output, HEIGHT * y + x_rev[5] + 4, transposed_b1.1);
            complex3_store_f32(output, HEIGHT * y + x_rev[6] + 4, transposed_b1.2);
        }
    }
}

impl NeonRadix7<f32> {
    fn base_run(&self, chunk: &mut [Complex<f32>]) {
        unsafe {
            static ROT_90: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            let rot_sign = vld1q_f32(ROT_90.as_ptr());

            let mut len = self.butterfly_length;

            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                let columns = len;
                len *= 7;
                let seventh = len / 7;

                for data in chunk.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 2 <= seventh {
                        let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                        let tw0 = vld1q_f32(m_twiddles.get_unchecked(6 * j..).as_ptr().cast());
                        let tw1 = vld1q_f32(m_twiddles.get_unchecked(6 * j + 2..).as_ptr().cast());
                        let tw2 = vld1q_f32(m_twiddles.get_unchecked(6 * j + 4..).as_ptr().cast());
                        let tw3 = vld1q_f32(m_twiddles.get_unchecked(6 * j + 6..).as_ptr().cast());
                        let tw4 = vld1q_f32(m_twiddles.get_unchecked(6 * j + 8..).as_ptr().cast());
                        let tw5 = vld1q_f32(m_twiddles.get_unchecked(6 * j + 10..).as_ptr().cast());

                        let u1 = vfcmulq_f32(
                            vld1q_f32(data.get_unchecked(j + seventh..).as_ptr().cast()),
                            tw0,
                        );
                        let u2 = vfcmulq_f32(
                            vld1q_f32(data.get_unchecked(j + 2 * seventh..).as_ptr().cast()),
                            tw1,
                        );
                        let u3 = vfcmulq_f32(
                            vld1q_f32(data.get_unchecked(j + 3 * seventh..).as_ptr().cast()),
                            tw2,
                        );
                        let u4 = vfcmulq_f32(
                            vld1q_f32(data.get_unchecked(j + 4 * seventh..).as_ptr().cast()),
                            tw3,
                        );
                        let u5 = vfcmulq_f32(
                            vld1q_f32(data.get_unchecked(j + 5 * seventh..).as_ptr().cast()),
                            tw4,
                        );
                        let u6 = vfcmulq_f32(
                            vld1q_f32(data.get_unchecked(j + 6 * seventh..).as_ptr().cast()),
                            tw5,
                        );

                        // Radix-7 butterfly

                        let (x1p6, x1m6) = NeonButterfly::butterfly2_f32(u1, u6);
                        let x1m6 = v_rotate90_f32(x1m6, rot_sign);
                        let y00 = vaddq_f32(u0, x1p6);
                        let (x2p5, x2m5) = NeonButterfly::butterfly2_f32(u2, u5);
                        let x2m5 = v_rotate90_f32(x2m5, rot_sign);
                        let y00 = vaddq_f32(y00, x2p5);
                        let (x3p4, x3m4) = NeonButterfly::butterfly2_f32(u3, u4);
                        let x3m4 = v_rotate90_f32(x3m4, rot_sign);
                        let y00 = vaddq_f32(y00, x3p4);

                        let m0106a = vfmaq_n_f32(u0, x1p6, self.twiddle1.re);
                        let m0106a = vfmaq_n_f32(m0106a, x2p5, self.twiddle2.re);
                        let m0106a = vfmaq_n_f32(m0106a, x3p4, self.twiddle3.re);
                        let m0106b = vmulq_n_f32(x1m6, self.twiddle1.im);
                        let m0106b = vfmaq_n_f32(m0106b, x2m5, self.twiddle2.im);
                        let m0106b = vfmaq_n_f32(m0106b, x3m4, self.twiddle3.im);
                        let (y01, y06) = NeonButterfly::butterfly2_f32(m0106a, m0106b);

                        let m0205a = vfmaq_n_f32(u0, x1p6, self.twiddle2.re);
                        let m0205a = vfmaq_n_f32(m0205a, x2p5, self.twiddle3.re);
                        let m0205a = vfmaq_n_f32(m0205a, x3p4, self.twiddle1.re);
                        let m0205b = vmulq_n_f32(x1m6, self.twiddle2.im);
                        let m0205b = vfmsq_n_f32(m0205b, x2m5, self.twiddle3.im);
                        let m0205b = vfmsq_n_f32(m0205b, x3m4, self.twiddle1.im);
                        let (y02, y05) = NeonButterfly::butterfly2_f32(m0205a, m0205b);

                        let m0304a = vfmaq_n_f32(u0, x1p6, self.twiddle3.re);
                        let m0304a = vfmaq_n_f32(m0304a, x2p5, self.twiddle1.re);
                        let m0304a = vfmaq_n_f32(m0304a, x3p4, self.twiddle2.re);
                        let m0304b = vmulq_n_f32(x1m6, self.twiddle3.im);
                        let m0304b = vfmsq_n_f32(m0304b, x2m5, self.twiddle1.im);
                        let m0304b = vfmaq_n_f32(m0304b, x3m4, self.twiddle2.im);
                        let (y03, y04) = NeonButterfly::butterfly2_f32(m0304a, m0304b);

                        // Store results
                        vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        vst1q_f32(
                            data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                            y01,
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 2 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y02,
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 3 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 4 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 5 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y05,
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 6 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y06,
                        );

                        j += 2;
                    }

                    for j in j..seventh {
                        let u0 = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                        let w0w1 = vld1q_f32(m_twiddles.get_unchecked(6 * j..).as_ptr().cast());
                        let w2w3 = vld1q_f32(m_twiddles.get_unchecked(6 * j + 2..).as_ptr().cast());
                        let w4w5 = vld1q_f32(m_twiddles.get_unchecked(6 * j + 4..).as_ptr().cast());

                        let u1u2 = vfcmulq_f32(
                            vcombine_f32(
                                vld1_f32(data.get_unchecked(j + seventh..).as_ptr().cast()),
                                vld1_f32(data.get_unchecked(j + 2 * seventh..).as_ptr().cast()),
                            ),
                            w0w1,
                        );
                        let u3u4 = vfcmulq_f32(
                            vcombine_f32(
                                vld1_f32(data.get_unchecked(j + 3 * seventh..).as_ptr().cast()),
                                vld1_f32(data.get_unchecked(j + 4 * seventh..).as_ptr().cast()),
                            ),
                            w2w3,
                        );
                        let u5u6 = vfcmulq_f32(
                            vcombine_f32(
                                vld1_f32(data.get_unchecked(j + 5 * seventh..).as_ptr().cast()),
                                vld1_f32(data.get_unchecked(j + 6 * seventh..).as_ptr().cast()),
                            ),
                            w4w5,
                        );

                        let u1 = vget_low_f32(u1u2);
                        let u2 = vget_high_f32(u1u2);
                        let u3 = vget_low_f32(u3u4);
                        let u4 = vget_high_f32(u3u4);
                        let u5 = vget_low_f32(u5u6);
                        let u6 = vget_high_f32(u5u6);

                        // Radix-7 butterfly

                        let (x1p6, x1m6) = NeonButterfly::butterfly2h_f32(u1, u6);
                        let x1m6 = vh_rotate90_f32(x1m6, vget_low_f32(rot_sign));
                        let y00 = vadd_f32(u0, x1p6);
                        let (x2p5, x2m5) = NeonButterfly::butterfly2h_f32(u2, u5);
                        let x2m5 = vh_rotate90_f32(x2m5, vget_low_f32(rot_sign));
                        let y00 = vadd_f32(y00, x2p5);
                        let (x3p4, x3m4) = NeonButterfly::butterfly2h_f32(u3, u4);
                        let x3m4 = vh_rotate90_f32(x3m4, vget_low_f32(rot_sign));
                        let y00 = vadd_f32(y00, x3p4);

                        let m0106a = vfma_n_f32(u0, x1p6, self.twiddle1.re);
                        let m0106a = vfma_n_f32(m0106a, x2p5, self.twiddle2.re);
                        let m0106a = vfma_n_f32(m0106a, x3p4, self.twiddle3.re);
                        let m0106b = vmul_n_f32(x1m6, self.twiddle1.im);
                        let m0106b = vfma_n_f32(m0106b, x2m5, self.twiddle2.im);
                        let m0106b = vfma_n_f32(m0106b, x3m4, self.twiddle3.im);
                        let (y01, y06) = NeonButterfly::butterfly2h_f32(m0106a, m0106b);

                        let m0205a = vfma_n_f32(u0, x1p6, self.twiddle2.re);
                        let m0205a = vfma_n_f32(m0205a, x2p5, self.twiddle3.re);
                        let m0205a = vfma_n_f32(m0205a, x3p4, self.twiddle1.re);
                        let m0205b = vmul_n_f32(x1m6, self.twiddle2.im);
                        let m0205b = vfms_n_f32(m0205b, x2m5, self.twiddle3.im);
                        let m0205b = vfms_n_f32(m0205b, x3m4, self.twiddle1.im);
                        let (y02, y05) = NeonButterfly::butterfly2h_f32(m0205a, m0205b);

                        let m0304a = vfma_n_f32(u0, x1p6, self.twiddle3.re);
                        let m0304a = vfma_n_f32(m0304a, x2p5, self.twiddle1.re);
                        let m0304a = vfma_n_f32(m0304a, x3p4, self.twiddle2.re);
                        let m0304b = vmul_n_f32(x1m6, self.twiddle3.im);
                        let m0304b = vfms_n_f32(m0304b, x2m5, self.twiddle1.im);
                        let m0304b = vfma_n_f32(m0304b, x3m4, self.twiddle2.im);
                        let (y03, y04) = NeonButterfly::butterfly2h_f32(m0304a, m0304b);

                        // Store results
                        vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        vst1_f32(
                            data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                            y01,
                        );
                        vst1_f32(
                            data.get_unchecked_mut(j + 2 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y02,
                        );
                        vst1_f32(
                            data.get_unchecked_mut(j + 3 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        vst1_f32(
                            data.get_unchecked_mut(j + 4 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        vst1_f32(
                            data.get_unchecked_mut(j + 5 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y05,
                        );
                        vst1_f32(
                            data.get_unchecked_mut(j + 6 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y06,
                        );
                    }
                }

                m_twiddles = &m_twiddles[columns * 6..];
            }
        }
    }
}

impl FftExecutor<f32> for NeonRadix7<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        let mut scratch = try_vec![Complex::zero(); self.scratch_length()];
        self.execute_with_scratch(in_place, &mut scratch)
    }

    fn execute_with_scratch(
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
            neon_bitreversed_transpose_f32_radix7(self.butterfly_length, chunk, scratch);
            self.butterfly.execute_out_of_place(scratch, chunk)?;
            self.base_run(chunk);
        }
        Ok(())
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_with_scratch(src, dst, &mut [])
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
        _: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        validate_oof_sizes!(src, dst, self.execution_length);

        for (dst, src) in dst
            .chunks_exact_mut(self.execution_length)
            .zip(src.chunks_exact(self.execution_length))
        {
            // Digit-reversal permutation
            neon_bitreversed_transpose_f32_radix7(self.butterfly_length, src, dst);
            self.butterfly.execute(dst)?;
            self.base_run(dst);
        }
        Ok(())
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
    use crate::util::test_radix;

    test_radix!(test_neon_radix7, f32, NeonRadix7, 4, 7, 1e-3);
    test_radix!(test_neon_radix7_f64, f64, NeonRadix7, 4, 7, 1e-8);
}
