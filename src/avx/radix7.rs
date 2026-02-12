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
use crate::avx::rotate::AvxRotate;
use crate::avx::transpose::transpose_7x7_f32;
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_fcmul_ps, _mm_unpackhi_ps64, _mm_unpacklo_ps64,
    _mm256_create_pd, _mm256_create_ps, _mm256_fcmul_pd, _mm256_fcmul_ps, _mm256_load4_f32x2,
    avx_bitreversed_transpose, create_avx4_twiddles,
};
use crate::err::try_vec;
use crate::radix7::Radix7Twiddles;
use crate::util::{
    compute_twiddle, int_logarithm, is_power_of_seven, reverse_bits, validate_oof_sizes,
    validate_scratch,
};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::arch::x86_64::*;
use std::sync::Arc;

pub(crate) struct AvxFmaRadix7<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    direction: FftDirection,
    butterfly: Arc<dyn FftExecutor<T> + Send + Sync>,
    butterfly_length: usize,
}

impl<T: FftSample + Radix7Twiddles> AvxFmaRadix7<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix7<T>, ZaftError> {
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

        let twiddles = create_avx4_twiddles::<T, 7>(butterfly_length, size, fft_direction)?;

        Ok(AvxFmaRadix7 {
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

impl AvxFmaRadix7<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn base_run(&self, chunk: &mut [Complex<f64>]) {
        unsafe {
            let rotate = AvxRotate::<f64>::new(FftDirection::Inverse);

            let tw1r = _mm256_set1_pd(self.twiddle1.re);
            let tw2r = _mm256_set1_pd(self.twiddle2.re);
            let tw3r = _mm256_set1_pd(self.twiddle3.re);
            let tw1i = _mm256_set1_pd(self.twiddle1.im);
            let tw2i = _mm256_set1_pd(self.twiddle2.im);
            let tw3i = _mm256_set1_pd(self.twiddle3.im);

            let mut len = self.butterfly_length;

            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                let columns = len;
                len *= 7;
                let seventh = len / 7;

                for data in chunk.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 2 <= seventh {
                        let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                        let twi = 6 * j;
                        let tw0 = _mm256_loadu_pd(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                        let tw1 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 2..).as_ptr().cast());
                        let tw2 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 4..).as_ptr().cast());

                        let tw0_2 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 6..).as_ptr().cast());
                        let tw1_2 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 8..).as_ptr().cast());
                        let tw2_2 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 10..).as_ptr().cast());

                        let wu1 =
                            _mm256_loadu_pd(data.get_unchecked(j + seventh..).as_ptr().cast());
                        let wu2 =
                            _mm256_loadu_pd(data.get_unchecked(j + 2 * seventh..).as_ptr().cast());
                        let wu3 =
                            _mm256_loadu_pd(data.get_unchecked(j + 3 * seventh..).as_ptr().cast());

                        let u1 = _mm256_fcmul_pd(wu1, tw0);
                        let u2 = _mm256_fcmul_pd(wu2, tw1);
                        let u3 = _mm256_fcmul_pd(wu3, tw2);

                        let wu4 =
                            _mm256_loadu_pd(data.get_unchecked(j + 4 * seventh..).as_ptr().cast());
                        let wu5 =
                            _mm256_loadu_pd(data.get_unchecked(j + 5 * seventh..).as_ptr().cast());
                        let wu6 =
                            _mm256_loadu_pd(data.get_unchecked(j + 6 * seventh..).as_ptr().cast());

                        let u4 = _mm256_fcmul_pd(wu4, tw0_2);
                        let u5 = _mm256_fcmul_pd(wu5, tw1_2);
                        let u6 = _mm256_fcmul_pd(wu6, tw2_2);

                        let (x1p6, x1m6) = AvxButterfly::butterfly2_f64(u1, u6);
                        let x1m6 = rotate.rotate_m256d(x1m6);
                        let y00 = _mm256_add_pd(u0, x1p6);
                        let (x2p5, x2m5) = AvxButterfly::butterfly2_f64(u2, u5);
                        let x2m5 = rotate.rotate_m256d(x2m5);
                        let y00 = _mm256_add_pd(y00, x2p5);
                        let (x3p4, x3m4) = AvxButterfly::butterfly2_f64(u3, u4);
                        let x3m4 = rotate.rotate_m256d(x3m4);
                        let y00 = _mm256_add_pd(y00, x3p4);

                        let m0106a = _mm256_fmadd_pd(x1p6, tw1r, u0);
                        let m0106a = _mm256_fmadd_pd(x2p5, tw2r, m0106a);
                        let m0106a = _mm256_fmadd_pd(x3p4, tw3r, m0106a);
                        let m0106b = _mm256_mul_pd(x1m6, tw1i);
                        let m0106b = _mm256_fmadd_pd(x2m5, tw2i, m0106b);
                        let m0106b = _mm256_fmadd_pd(x3m4, tw3i, m0106b);
                        let (y01, y06) = AvxButterfly::butterfly2_f64(m0106a, m0106b);

                        let m0205a = _mm256_fmadd_pd(x1p6, tw2r, u0);
                        let m0205a = _mm256_fmadd_pd(x2p5, tw3r, m0205a);
                        let m0205a = _mm256_fmadd_pd(x3p4, tw1r, m0205a);
                        let m0205b = _mm256_mul_pd(x1m6, tw2i);
                        let m0205b = _mm256_fnmadd_pd(x2m5, tw3i, m0205b);
                        let m0205b = _mm256_fnmadd_pd(x3m4, tw1i, m0205b);
                        let (y02, y05) = AvxButterfly::butterfly2_f64(m0205a, m0205b);

                        let m0304a = _mm256_fmadd_pd(x1p6, tw3r, u0);
                        let m0304a = _mm256_fmadd_pd(x2p5, tw1r, m0304a);
                        let m0304a = _mm256_fmadd_pd(x3p4, tw2r, m0304a);
                        let m0304b = _mm256_mul_pd(x1m6, tw3i);
                        let m0304b = _mm256_fnmadd_pd(x2m5, tw1i, m0304b);
                        let m0304b = _mm256_fmadd_pd(x3m4, tw2i, m0304b);
                        let (y03, y04) = AvxButterfly::butterfly2_f64(m0304a, m0304b);

                        // // Store results
                        _mm256_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                            y01,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 2 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y02,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 3 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 4 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 5 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y05,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 6 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y06,
                        );

                        j += 2;
                    }

                    let tw1tw2r = _mm256_setr_pd(
                        self.twiddle1.re,
                        self.twiddle1.re,
                        self.twiddle2.re,
                        self.twiddle2.re,
                    );
                    let tw2tw3r = _mm256_setr_pd(
                        self.twiddle2.re,
                        self.twiddle2.re,
                        self.twiddle3.re,
                        self.twiddle3.re,
                    );
                    let tw3tw1r = _mm256_setr_pd(
                        self.twiddle3.re,
                        self.twiddle3.re,
                        self.twiddle1.re,
                        self.twiddle1.re,
                    );

                    let tw1tw2i = _mm256_setr_pd(
                        self.twiddle1.im,
                        self.twiddle1.im,
                        self.twiddle2.im,
                        self.twiddle2.im,
                    );
                    let tw2tw3i = _mm256_setr_pd(
                        self.twiddle2.im,
                        self.twiddle2.im,
                        -self.twiddle3.im,
                        -self.twiddle3.im,
                    );
                    let tw3tw1i = _mm256_setr_pd(
                        self.twiddle3.im,
                        self.twiddle3.im,
                        -self.twiddle1.im,
                        -self.twiddle1.im,
                    );

                    for j in j..seventh {
                        let u0 = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                        let twi = 6 * j;
                        let tw0 = _mm256_loadu_pd(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                        let tw1 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 2..).as_ptr().cast());
                        let tw2 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(twi + 4..).as_ptr().cast());

                        let wu1u2 = _mm256_create_pd(
                            _mm_loadu_pd(data.get_unchecked(j + seventh..).as_ptr().cast()),
                            _mm_loadu_pd(data.get_unchecked(j + 2 * seventh..).as_ptr().cast()),
                        );
                        let wu3u4 = _mm256_create_pd(
                            _mm_loadu_pd(data.get_unchecked(j + 3 * seventh..).as_ptr().cast()),
                            _mm_loadu_pd(data.get_unchecked(j + 4 * seventh..).as_ptr().cast()),
                        );
                        let wu5u6 = _mm256_create_pd(
                            _mm_loadu_pd(data.get_unchecked(j + 5 * seventh..).as_ptr().cast()),
                            _mm_loadu_pd(data.get_unchecked(j + 6 * seventh..).as_ptr().cast()),
                        );

                        let u1u2 = _mm256_fcmul_pd(wu1u2, tw0);
                        let u3u4 = _mm256_fcmul_pd(wu3u4, tw1);
                        let u5u6 = _mm256_fcmul_pd(wu5u6, tw2);

                        const HI_HI: i32 = 0b0011_0001;
                        const LO_LO: i32 = 0b0010_0000;
                        const HI_LO: i32 = 0b0010_0001;

                        let (x1p6x2p5, x1m6x2m5) = AvxButterfly::butterfly2_f64(
                            u1u2,
                            _mm256_permute2f128_pd::<HI_LO>(u5u6, u5u6),
                        );
                        let x1m6x2m5 = rotate.rotate_m256d(x1m6x2m5);

                        let x1p6 = _mm256_castpd256_pd128(x1p6x2p5);
                        let x2p5 = _mm256_extractf128_pd::<1>(x1p6x2p5);
                        let x1m6 = _mm256_castpd256_pd128(x1m6x2m5);
                        let x2m5 = _mm256_extractf128_pd::<1>(x1m6x2m5);

                        let y00 = _mm_add_pd(
                            _mm_add_pd(u0, _mm256_castpd256_pd128(x1p6x2p5)),
                            _mm256_extractf128_pd::<1>(x1p6x2p5),
                        );
                        let (x3p4, x3m4) = AvxButterfly::butterfly2_f64_m128(
                            _mm256_castpd256_pd128(u3u4),
                            _mm256_extractf128_pd::<1>(u3u4),
                        );
                        let x3m4 = rotate.rotate_m128d(x3m4);
                        let y00 = _mm_add_pd(y00, x3p4);

                        let x1p6d = _mm256_permute2f128_pd::<LO_LO>(x1p6x2p5, x1p6x2p5);
                        let x2p5d = _mm256_permute2f128_pd::<HI_HI>(x1p6x2p5, x1p6x2p5);
                        let x3p4d = _mm256_create_pd(x3p4, x3p4);

                        let x1m6d = _mm256_permute2f128_pd::<LO_LO>(x1m6x2m5, x1m6x2m5);
                        let x2m5d = _mm256_permute2f128_pd::<HI_HI>(x1m6x2m5, x1m6x2m5);
                        let x3m4d = _mm256_create_pd(x3m4, x3m4);

                        let m0106am0205a =
                            _mm256_fmadd_pd(x1p6d, tw1tw2r, _mm256_create_pd(u0, u0));
                        let m0106am0205a = _mm256_fmadd_pd(x2p5d, tw2tw3r, m0106am0205a);
                        let m0106am0205a = _mm256_fmadd_pd(x3p4d, tw3tw1r, m0106am0205a);
                        let m0106bm0205b = _mm256_mul_pd(x1m6d, tw1tw2i);
                        let m0106bm0205b = _mm256_fmadd_pd(x2m5d, tw2tw3i, m0106bm0205b);
                        let m0106bm0205b = _mm256_fmadd_pd(x3m4d, tw3tw1i, m0106bm0205b);
                        let (y01y02, y06y05) =
                            AvxButterfly::butterfly2_f64(m0106am0205a, m0106bm0205b);

                        let m0304a = _mm_fmadd_pd(x1p6, _mm256_castpd256_pd128(tw3tw1r), u0);
                        let m0304a = _mm_fmadd_pd(x2p5, _mm256_castpd256_pd128(tw1tw2r), m0304a);
                        let m0304a = _mm_fmadd_pd(x3p4, _mm256_castpd256_pd128(tw2tw3r), m0304a);
                        let m0304b = _mm_mul_pd(x1m6, _mm256_castpd256_pd128(tw3tw1i));
                        let m0304b = _mm_fnmadd_pd(x2m5, _mm256_castpd256_pd128(tw1tw2i), m0304b);
                        let m0304b = _mm_fmadd_pd(x3m4, _mm256_castpd256_pd128(tw2tw3i), m0304b);
                        let (y03, y04) = AvxButterfly::butterfly2_f64_m128(m0304a, m0304b);

                        // // Store results
                        _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                            _mm256_castpd256_pd128(y01y02),
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 2 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            _mm256_extractf128_pd::<1>(y01y02),
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 3 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 4 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 5 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            _mm256_extractf128_pd::<1>(y06y05),
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 6 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            _mm256_castpd256_pd128(y06y05),
                        );
                    }
                }

                m_twiddles = &m_twiddles[columns * 6..];
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
            avx_bitreversed_transpose::<Complex<f64>, 7>(self.butterfly_length, chunk, scratch);
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
            avx_bitreversed_transpose::<Complex<f64>, 7>(self.butterfly_length, src, dst);
            self.butterfly.execute(dst)?;
            self.base_run(dst);
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix7<f64> {
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

#[inline]
#[target_feature(enable = "avx2")]
fn complex7_load_f32(array: &[Complex<f32>], idx: usize) -> (__m256, __m256) {
    unsafe {
        (
            _mm256_loadu_ps(array.get_unchecked(idx..).as_ptr().cast()),
            _mm256_setr_m128(
                _mm_loadu_ps(array.get_unchecked(idx + 4..).as_ptr().cast()),
                _m128s_load_f32x2(array.get_unchecked(idx + 6..).as_ptr().cast()),
            ),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn complex7_store_f32(array: &mut [Complex<f32>], idx: usize, v: (__m256, __m256)) {
    unsafe {
        _mm256_storeu_ps(array.get_unchecked_mut(idx..).as_mut_ptr().cast(), v.0);
        _mm_storeu_ps(
            array.get_unchecked_mut(idx + 4..).as_mut_ptr().cast(),
            _mm256_castps256_ps128(v.1),
        );
        _m128s_store_f32x2(
            array.get_unchecked_mut(idx + 6..).as_mut_ptr().cast(),
            _mm256_extractf128_ps::<1>(v.1),
        );
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn avx_bitreversed_transpose_f32_radix7(
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
            reverse_bits::<WIDTH>(WIDTH * x + 5, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 6, rev_digits) * height,
        ];

        for y in 0..strided_height {
            let base_input_idx = (WIDTH * x) + y * HEIGHT * width;
            let rows = [
                complex7_load_f32(input, base_input_idx),
                complex7_load_f32(input, base_input_idx + width),
                complex7_load_f32(input, base_input_idx + width * 2),
                complex7_load_f32(input, base_input_idx + width * 3),
                complex7_load_f32(input, base_input_idx + width * 4),
                complex7_load_f32(input, base_input_idx + width * 5),
                complex7_load_f32(input, base_input_idx + width * 6),
            ];
            let transposed = transpose_7x7_f32(
                [
                    rows[0].0, rows[1].0, rows[2].0, rows[3].0, rows[4].0, rows[5].0, rows[6].0,
                ],
                [
                    rows[0].1, rows[1].1, rows[2].1, rows[3].1, rows[4].1, rows[5].1, rows[6].1,
                ],
            );

            complex7_store_f32(
                output,
                HEIGHT * y + x_rev[0],
                (transposed.0[0], transposed.1[0]),
            );
            complex7_store_f32(
                output,
                HEIGHT * y + x_rev[1],
                (transposed.0[1], transposed.1[1]),
            );
            complex7_store_f32(
                output,
                HEIGHT * y + x_rev[2],
                (transposed.0[2], transposed.1[2]),
            );
            complex7_store_f32(
                output,
                HEIGHT * y + x_rev[3],
                (transposed.0[3], transposed.1[3]),
            );
            complex7_store_f32(
                output,
                HEIGHT * y + x_rev[4],
                (transposed.0[4], transposed.1[4]),
            );
            complex7_store_f32(
                output,
                HEIGHT * y + x_rev[5],
                (transposed.0[5], transposed.1[5]),
            );
            complex7_store_f32(
                output,
                HEIGHT * y + x_rev[6],
                (transposed.0[6], transposed.1[6]),
            );
        }
    }
}

impl AvxFmaRadix7<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn base_run(&self, chunk: &mut [Complex<f32>]) {
        unsafe {
            let rotate = AvxRotate::<f32>::new(FftDirection::Inverse);

            let tw1r = _mm256_set1_ps(self.twiddle1.re);
            let tw2r = _mm256_set1_ps(self.twiddle2.re);
            let tw3r = _mm256_set1_ps(self.twiddle3.re);
            let tw1i = _mm256_set1_ps(self.twiddle1.im);
            let tw2i = _mm256_set1_ps(self.twiddle2.im);
            let tw3i = _mm256_set1_ps(self.twiddle3.im);

            let mut len = self.butterfly_length;

            let mut m_twiddles = self.twiddles.as_slice();

            while len < self.execution_length {
                let columns = len;
                len *= 7;
                let seventh = len / 7;

                for data in chunk.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 4 <= seventh {
                        let u0 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());
                        let twi = 6 * j;
                        let tw0 = _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                        let tw1 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 4..).as_ptr().cast());
                        let tw2 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 8..).as_ptr().cast());

                        let tw3 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 12..).as_ptr().cast());
                        let tw4 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 16..).as_ptr().cast());
                        let tw5 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 20..).as_ptr().cast());

                        let a1 = _mm256_loadu_ps(data.get_unchecked(j + seventh..).as_ptr().cast());
                        let a2 =
                            _mm256_loadu_ps(data.get_unchecked(j + 2 * seventh..).as_ptr().cast());
                        let a3 =
                            _mm256_loadu_ps(data.get_unchecked(j + 3 * seventh..).as_ptr().cast());
                        let a4 =
                            _mm256_loadu_ps(data.get_unchecked(j + 4 * seventh..).as_ptr().cast());
                        let a5 =
                            _mm256_loadu_ps(data.get_unchecked(j + 5 * seventh..).as_ptr().cast());
                        let a6 =
                            _mm256_loadu_ps(data.get_unchecked(j + 6 * seventh..).as_ptr().cast());

                        let u1 = _mm256_fcmul_ps(a1, tw0);
                        let u2 = _mm256_fcmul_ps(a2, tw1);
                        let u3 = _mm256_fcmul_ps(a3, tw2);
                        let u4 = _mm256_fcmul_ps(a4, tw3);
                        let u5 = _mm256_fcmul_ps(a5, tw4);
                        let u6 = _mm256_fcmul_ps(a6, tw5);

                        let (x1p6, x1m6) = AvxButterfly::butterfly2_f32(u1, u6);
                        let x1m6 = rotate.rotate_m256(x1m6);
                        let y00 = _mm256_add_ps(u0, x1p6);
                        let (x2p5, x2m5) = AvxButterfly::butterfly2_f32(u2, u5);
                        let x2m5 = rotate.rotate_m256(x2m5);
                        let y00 = _mm256_add_ps(y00, x2p5);
                        let (x3p4, x3m4) = AvxButterfly::butterfly2_f32(u3, u4);
                        let x3m4 = rotate.rotate_m256(x3m4);
                        let y00 = _mm256_add_ps(y00, x3p4);

                        let m0106a = _mm256_fmadd_ps(x1p6, tw1r, u0);
                        let m0106a = _mm256_fmadd_ps(x2p5, tw2r, m0106a);
                        let m0106a = _mm256_fmadd_ps(x3p4, tw3r, m0106a);
                        let m0106b = _mm256_mul_ps(x1m6, tw1i);
                        let m0106b = _mm256_fmadd_ps(x2m5, tw2i, m0106b);
                        let m0106b = _mm256_fmadd_ps(x3m4, tw3i, m0106b);
                        let (y01, y06) = AvxButterfly::butterfly2_f32(m0106a, m0106b);

                        let m0205a = _mm256_fmadd_ps(x1p6, tw2r, u0);
                        let m0205a = _mm256_fmadd_ps(x2p5, tw3r, m0205a);
                        let m0205a = _mm256_fmadd_ps(x3p4, tw1r, m0205a);
                        let m0205b = _mm256_mul_ps(x1m6, tw2i);
                        let m0205b = _mm256_fnmadd_ps(x2m5, tw3i, m0205b);
                        let m0205b = _mm256_fnmadd_ps(x3m4, tw1i, m0205b);
                        let (y02, y05) = AvxButterfly::butterfly2_f32(m0205a, m0205b);

                        let m0304a = _mm256_fmadd_ps(x1p6, tw3r, u0);
                        let m0304a = _mm256_fmadd_ps(x2p5, tw1r, m0304a);
                        let m0304a = _mm256_fmadd_ps(x3p4, tw2r, m0304a);
                        let m0304b = _mm256_mul_ps(x1m6, tw3i);
                        let m0304b = _mm256_fnmadd_ps(x2m5, tw1i, m0304b);
                        let m0304b = _mm256_fmadd_ps(x3m4, tw2i, m0304b);
                        let (y03, y04) = AvxButterfly::butterfly2_f32(m0304a, m0304b);

                        // Store results
                        _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                            y01,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 2 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y02,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 3 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 4 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 5 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y05,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 6 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y06,
                        );

                        j += 4;
                    }

                    while j + 2 < seventh {
                        let u0 = _mm_loadu_ps(data.get_unchecked(j..).as_ptr().cast());
                        let twi = 6 * j;
                        let tw0 = _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                        let tw1 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 4..).as_ptr().cast());
                        let tw2 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi + 8..).as_ptr().cast());

                        let u1u2 = _mm256_fcmul_ps(
                            _mm256_create_ps(
                                _mm_loadu_ps(data.get_unchecked(j + seventh..).as_ptr().cast()),
                                _mm_loadu_ps(data.get_unchecked(j + 2 * seventh..).as_ptr().cast()),
                            ),
                            tw0,
                        );

                        let u3u4 = _mm256_fcmul_ps(
                            _mm256_create_ps(
                                _mm_loadu_ps(data.get_unchecked(j + 3 * seventh..).as_ptr().cast()),
                                _mm_loadu_ps(data.get_unchecked(j + 4 * seventh..).as_ptr().cast()),
                            ),
                            tw1,
                        );

                        let u5u6 = _mm256_fcmul_ps(
                            _mm256_create_ps(
                                _mm_loadu_ps(data.get_unchecked(j + 5 * seventh..).as_ptr().cast()),
                                _mm_loadu_ps(data.get_unchecked(j + 6 * seventh..).as_ptr().cast()),
                            ),
                            tw2,
                        );

                        let (x1p6, x1m6) = AvxButterfly::butterfly2_f32_m128(
                            _mm256_castps256_ps128(u1u2),
                            _mm256_extractf128_ps::<1>(u5u6),
                        );
                        let x1m6 = rotate.rotate_m128(x1m6);
                        let y00 = _mm_add_ps(u0, x1p6);
                        let (x2p5, x2m5) = AvxButterfly::butterfly2_f32_m128(
                            _mm256_extractf128_ps::<1>(u1u2),
                            _mm256_castps256_ps128(u5u6),
                        );
                        let x2m5 = rotate.rotate_m128(x2m5);
                        let y00 = _mm_add_ps(y00, x2p5);
                        let (x3p4, x3m4) = AvxButterfly::butterfly2_f32_m128(
                            _mm256_castps256_ps128(u3u4),
                            _mm256_extractf128_ps::<1>(u3u4),
                        );
                        let x3m4 = rotate.rotate_m128(x3m4);
                        let y00 = _mm_add_ps(y00, x3p4);

                        let m0106a = _mm_fmadd_ps(x1p6, _mm256_castps256_ps128(tw1r), u0);
                        let m0106a = _mm_fmadd_ps(x2p5, _mm256_castps256_ps128(tw2r), m0106a);
                        let m0106a = _mm_fmadd_ps(x3p4, _mm256_castps256_ps128(tw3r), m0106a);
                        let m0106b = _mm_mul_ps(x1m6, _mm256_castps256_ps128(tw1i));
                        let m0106b = _mm_fmadd_ps(x2m5, _mm256_castps256_ps128(tw2i), m0106b);
                        let m0106b = _mm_fmadd_ps(x3m4, _mm256_castps256_ps128(tw3i), m0106b);
                        let (y01, y06) = AvxButterfly::butterfly2_f32_m128(m0106a, m0106b);

                        let m0205a = _mm_fmadd_ps(x1p6, _mm256_castps256_ps128(tw2r), u0);
                        let m0205a = _mm_fmadd_ps(x2p5, _mm256_castps256_ps128(tw3r), m0205a);
                        let m0205a = _mm_fmadd_ps(x3p4, _mm256_castps256_ps128(tw1r), m0205a);
                        let m0205b = _mm_mul_ps(x1m6, _mm256_castps256_ps128(tw2i));
                        let m0205b = _mm_fnmadd_ps(x2m5, _mm256_castps256_ps128(tw3i), m0205b);
                        let m0205b = _mm_fnmadd_ps(x3m4, _mm256_castps256_ps128(tw1i), m0205b);
                        let (y02, y05) = AvxButterfly::butterfly2_f32_m128(m0205a, m0205b);

                        let m0304a = _mm_fmadd_ps(x1p6, _mm256_castps256_ps128(tw3r), u0);
                        let m0304a = _mm_fmadd_ps(x2p5, _mm256_castps256_ps128(tw1r), m0304a);
                        let m0304a = _mm_fmadd_ps(x3p4, _mm256_castps256_ps128(tw2r), m0304a);
                        let m0304b = _mm_mul_ps(x1m6, _mm256_castps256_ps128(tw3i));
                        let m0304b = _mm_fnmadd_ps(x2m5, _mm256_castps256_ps128(tw1i), m0304b);
                        let m0304b = _mm_fmadd_ps(x3m4, _mm256_castps256_ps128(tw2i), m0304b);
                        let (y03, y04) = AvxButterfly::butterfly2_f32_m128(m0304a, m0304b);

                        // Store results
                        _mm_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                            y01,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 2 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y02,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 3 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 4 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 5 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y05,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 6 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y06,
                        );

                        j += 2;
                    }

                    for j in j..seventh {
                        let u0 = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());
                        let twi = 6 * j;
                        let tw0tw1tw2tw3 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                        let tw4tw5 =
                            _mm_loadu_ps(m_twiddles.get_unchecked(twi + 4..).as_ptr().cast());

                        let u1u2u3u4 = _mm256_fcmul_ps(
                            _mm256_load4_f32x2(
                                data.get_unchecked(j + seventh..),
                                data.get_unchecked(j + 2 * seventh..),
                                data.get_unchecked(j + 3 * seventh..),
                                data.get_unchecked(j + 4 * seventh..),
                            ),
                            tw0tw1tw2tw3,
                        );
                        let u5u6 = _mm_fcmul_ps(
                            _mm_unpacklo_ps64(
                                _m128s_load_f32x2(
                                    data.get_unchecked(j + 5 * seventh..).as_ptr().cast(),
                                ),
                                _m128s_load_f32x2(
                                    data.get_unchecked(j + 6 * seventh..).as_ptr().cast(),
                                ),
                            ),
                            tw4tw5,
                        );

                        let u1u2 = _mm256_castps256_ps128(u1u2u3u4);
                        let u3u4 = _mm256_extractf128_ps::<1>(u1u2u3u4);
                        let u6 = _mm_unpackhi_ps64(u5u6, u5u6);
                        let u2 = _mm_unpackhi_ps64(u1u2, u1u2);
                        let u4 = _mm_unpackhi_ps64(u3u4, u3u4);

                        let (x1p6, x1m6) = AvxButterfly::butterfly2_f32_m128(u1u2, u6);
                        let x1m6 = rotate.rotate_m128(x1m6);
                        let y00 = _mm_add_ps(u0, x1p6);
                        let (x2p5, x2m5) = AvxButterfly::butterfly2_f32_m128(u2, u5u6);
                        let x2m5 = rotate.rotate_m128(x2m5);
                        let y00 = _mm_add_ps(y00, x2p5);
                        let (x3p4, x3m4) = AvxButterfly::butterfly2_f32_m128(u3u4, u4);
                        let x3m4 = rotate.rotate_m128(x3m4);
                        let y00 = _mm_add_ps(y00, x3p4);

                        let m0106a = _mm_fmadd_ps(x1p6, _mm256_castps256_ps128(tw1r), u0);
                        let m0106a = _mm_fmadd_ps(x2p5, _mm256_castps256_ps128(tw2r), m0106a);
                        let m0106a = _mm_fmadd_ps(x3p4, _mm256_castps256_ps128(tw3r), m0106a);
                        let m0106b = _mm_mul_ps(x1m6, _mm256_castps256_ps128(tw1i));
                        let m0106b = _mm_fmadd_ps(x2m5, _mm256_castps256_ps128(tw2i), m0106b);
                        let m0106b = _mm_fmadd_ps(x3m4, _mm256_castps256_ps128(tw3i), m0106b);
                        let (y01, y06) = AvxButterfly::butterfly2_f32_m128(m0106a, m0106b);

                        let m0205a = _mm_fmadd_ps(x1p6, _mm256_castps256_ps128(tw2r), u0);
                        let m0205a = _mm_fmadd_ps(x2p5, _mm256_castps256_ps128(tw3r), m0205a);
                        let m0205a = _mm_fmadd_ps(x3p4, _mm256_castps256_ps128(tw1r), m0205a);
                        let m0205b = _mm_mul_ps(x1m6, _mm256_castps256_ps128(tw2i));
                        let m0205b = _mm_fnmadd_ps(x2m5, _mm256_castps256_ps128(tw3i), m0205b);
                        let m0205b = _mm_fnmadd_ps(x3m4, _mm256_castps256_ps128(tw1i), m0205b);
                        let (y02, y05) = AvxButterfly::butterfly2_f32_m128(m0205a, m0205b);

                        let m0304a = _mm_fmadd_ps(x1p6, _mm256_castps256_ps128(tw3r), u0);
                        let m0304a = _mm_fmadd_ps(x2p5, _mm256_castps256_ps128(tw1r), m0304a);
                        let m0304a = _mm_fmadd_ps(x3p4, _mm256_castps256_ps128(tw2r), m0304a);
                        let m0304b = _mm_mul_ps(x1m6, _mm256_castps256_ps128(tw3i));
                        let m0304b = _mm_fnmadd_ps(x2m5, _mm256_castps256_ps128(tw1i), m0304b);
                        let m0304b = _mm_fmadd_ps(x3m4, _mm256_castps256_ps128(tw2i), m0304b);
                        let (y03, y04) = AvxButterfly::butterfly2_f32_m128(m0304a, m0304b);

                        // Store results
                        _m128s_store_f32x2(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                            y01,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 2 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y02,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 3 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y03,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 4 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y04,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 5 * seventh..)
                                .as_mut_ptr()
                                .cast(),
                            y05,
                        );
                        _m128s_store_f32x2(
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
            avx_bitreversed_transpose_f32_radix7(self.butterfly_length, chunk, scratch);
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
            avx_bitreversed_transpose_f32_radix7(self.butterfly_length, src, dst);
            self.butterfly.execute(dst)?;
            self.base_run(dst);
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix7<f32> {
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

    test_avx_radix!(test_avx_radix7, f32, AvxFmaRadix7, 4, 7, 1e-3);
    test_avx_radix!(test_avx_radix7_f64, f64, AvxFmaRadix7, 4, 7, 1e-8);
}
