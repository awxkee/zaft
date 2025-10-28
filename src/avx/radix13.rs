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
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_unpackhi_ps64, _mm256_create_pd, _mm256_fcmul_pd,
    _mm256_fcmul_ps, _mm256_load4_f32x2,
};
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::radix13::Radix13Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{bitreversed_transpose, compute_twiddle, is_power_of_thirteen};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::x86_64::*;
use std::fmt::Display;

pub(crate) struct AvxFmaRadix13<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    direction: FftDirection,
    butterfly: Box<dyn CompositeFftExecutor<T> + Send + Sync>,
}

impl<
    T: Default
        + Clone
        + Radix13Twiddles
        + 'static
        + Copy
        + FftTrigonometry
        + Float
        + Send
        + Sync
        + AlgorithmFactory<T>
        + SpectrumOpsFactory<T>
        + TransposeFactory<T>
        + MulAdd<T, Output = T>
        + Display,
> AvxFmaRadix13<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix13<T>, ZaftError> {
        assert!(
            is_power_of_thirteen(size as u64),
            "Input length must be a power of 13"
        );

        let twiddles = T::make_twiddles_with_base(13, size, fft_direction)?;

        Ok(AvxFmaRadix13 {
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 13, fft_direction),
            twiddle2: compute_twiddle(2, 13, fft_direction),
            twiddle3: compute_twiddle(3, 13, fft_direction),
            twiddle4: compute_twiddle(4, 13, fft_direction),
            twiddle5: compute_twiddle(5, 13, fft_direction),
            twiddle6: compute_twiddle(6, 13, fft_direction),
            direction: fft_direction,
            butterfly: T::butterfly13(fft_direction)?,
        })
    }
}

impl AvxFmaRadix13<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let rotate = AvxRotate::<f64>::new(FftDirection::Inverse);

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f64>, 13>(13, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = 13;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 13;
                    let thirteenth = len / 13;

                    for data in chunk.chunks_exact_mut(len) {
                        let j = 0usize;

                        for j in j..thirteenth {
                            let u0 = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let twi = 12 * j;
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

                            let u1u2 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + thirteenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 2 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let u3u4 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 3 * thirteenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 4 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw1,
                            );
                            let u5u6 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 5 * thirteenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 6 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw2,
                            );
                            let u7u8 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 7 * thirteenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 8 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw3,
                            );
                            let u9u10 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 9 * thirteenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 10 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw4,
                            );
                            let u11u12 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 11 * thirteenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 12 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw5,
                            );

                            let y00 = u0;
                            let (x1p12, x1m12) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(u1u2),
                                _mm256_extractf128_pd::<1>(u11u12),
                            ); // u1, u12
                            let x1m12 = rotate.rotate_m128d(x1m12);
                            let y00 = _mm_add_pd(y00, x1p12);
                            let (x2p11, x2m11) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_extractf128_pd::<1>(u1u2),
                                _mm256_castpd256_pd128(u11u12),
                            ); // u2, u11
                            let x2m11 = rotate.rotate_m128d(x2m11);
                            let y00 = _mm_add_pd(y00, x2p11);
                            let (x3p10, x3m10) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(u3u4),
                                _mm256_extractf128_pd::<1>(u9u10),
                            ); // u3, u10
                            let x3m10 = rotate.rotate_m128d(x3m10);
                            let y00 = _mm_add_pd(y00, x3p10);
                            let (x4p9, x4m9) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_extractf128_pd::<1>(u3u4),
                                _mm256_castpd256_pd128(u9u10),
                            ); // u4, u9
                            let x4m9 = rotate.rotate_m128d(x4m9);
                            let y00 = _mm_add_pd(y00, x4p9);
                            let (x5p8, x5m8) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(u5u6),
                                _mm256_extractf128_pd::<1>(u7u8),
                            ); // u5, u8
                            let x5m8 = rotate.rotate_m128d(x5m8);
                            let y00 = _mm_add_pd(y00, x5p8);
                            let (x6p7, x6m7) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_extractf128_pd::<1>(u5u6),
                                _mm256_castpd256_pd128(u7u8),
                            ); // u6, u7
                            let x6m7 = rotate.rotate_m128d(x6m7);
                            let y00 = _mm_add_pd(y00, x6p7);

                            let m0112a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle1.re), u0);
                            let m0112a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle2.re), x2p11, m0112a);
                            let m0112a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle3.re), x3p10, m0112a);
                            let m0112a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle4.re), m0112a);
                            let m0112a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle5.re), m0112a);
                            let m0112a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle6.re), m0112a);
                            let m0112b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle1.im));
                            let m0112b = _mm_fmadd_pd(x2m11, _mm_set1_pd(self.twiddle2.im), m0112b);
                            let m0112b = _mm_fmadd_pd(x3m10, _mm_set1_pd(self.twiddle3.im), m0112b);
                            let m0112b = _mm_fmadd_pd(x4m9, _mm_set1_pd(self.twiddle4.im), m0112b);
                            let m0112b = _mm_fmadd_pd(x5m8, _mm_set1_pd(self.twiddle5.im), m0112b);
                            let m0112b = _mm_fmadd_pd(x6m7, _mm_set1_pd(self.twiddle6.im), m0112b);
                            let (y01, y12) = AvxButterfly::butterfly2_f64_m128(m0112a, m0112b);

                            let m0211a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle2.re), u0);
                            let m0211a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle4.re), m0211a);
                            let m0211a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle6.re), m0211a);
                            let m0211a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle5.re), m0211a);
                            let m0211a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle3.re), m0211a);
                            let m0211a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle1.re), m0211a);
                            let m0211b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle2.im));
                            let m0211b = _mm_fmadd_pd(x2m11, _mm_set1_pd(self.twiddle4.im), m0211b);
                            let m0211b = _mm_fmadd_pd(x3m10, _mm_set1_pd(self.twiddle6.im), m0211b);
                            let m0211b = _mm_fnmadd_pd(x4m9, _mm_set1_pd(self.twiddle5.im), m0211b);
                            let m0211b = _mm_fnmadd_pd(x5m8, _mm_set1_pd(self.twiddle3.im), m0211b);
                            let m0211b = _mm_fnmadd_pd(x6m7, _mm_set1_pd(self.twiddle1.im), m0211b);
                            let (y02, y11) = AvxButterfly::butterfly2_f64_m128(m0211a, m0211b);

                            let m0310a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle3.re), u0);
                            let m0310a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle6.re), m0310a);
                            let m0310a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle4.re), m0310a);
                            let m0310a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle1.re), m0310a);
                            let m0310a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle2.re), m0310a);
                            let m0310a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle5.re), m0310a);
                            let m0310b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle3.im));
                            let m0310b = _mm_fmadd_pd(x2m11, _mm_set1_pd(self.twiddle6.im), m0310b);
                            let m0310b =
                                _mm_fnmadd_pd(x3m10, _mm_set1_pd(self.twiddle4.im), m0310b);
                            let m0310b = _mm_fnmadd_pd(x4m9, _mm_set1_pd(self.twiddle1.im), m0310b);
                            let m0310b = _mm_fmadd_pd(x5m8, _mm_set1_pd(self.twiddle2.im), m0310b);
                            let m0310b = _mm_fmadd_pd(x6m7, _mm_set1_pd(self.twiddle5.im), m0310b);
                            let (y03, y10) = AvxButterfly::butterfly2_f64_m128(m0310a, m0310b);

                            let m0409a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle4.re), u0);
                            let m0409a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle5.re), m0409a);
                            let m0409a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle1.re), m0409a);
                            let m0409a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle3.re), m0409a);
                            let m0409a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle6.re), m0409a);
                            let m0409a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle2.re), m0409a);
                            let m0409b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle4.im));
                            let m0409b =
                                _mm_fnmadd_pd(x2m11, _mm_set1_pd(self.twiddle5.im), m0409b);
                            let m0409b =
                                _mm_fnmadd_pd(x3m10, _mm_set1_pd(self.twiddle1.im), m0409b);
                            let m0409b = _mm_fmadd_pd(x4m9, _mm_set1_pd(self.twiddle3.im), m0409b);
                            let m0409b = _mm_fnmadd_pd(x5m8, _mm_set1_pd(self.twiddle6.im), m0409b);
                            let m0409b = _mm_fnmadd_pd(x6m7, _mm_set1_pd(self.twiddle2.im), m0409b);
                            let (y04, y09) = AvxButterfly::butterfly2_f64_m128(m0409a, m0409b);

                            let m0508a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle5.re), u0);
                            let m0508a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle3.re), m0508a);
                            let m0508a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle2.re), m0508a);
                            let m0508a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle6.re), m0508a);
                            let m0508a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle1.re), m0508a);
                            let m0508a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle4.re), m0508a);
                            let m0508b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle5.im));
                            let m0508b =
                                _mm_fnmadd_pd(x2m11, _mm_set1_pd(self.twiddle3.im), m0508b);
                            let m0508b = _mm_fmadd_pd(x3m10, _mm_set1_pd(self.twiddle2.im), m0508b);
                            let m0508b = _mm_fnmadd_pd(x4m9, _mm_set1_pd(self.twiddle6.im), m0508b);
                            let m0508b = _mm_fnmadd_pd(x5m8, _mm_set1_pd(self.twiddle1.im), m0508b);
                            let m0508b = _mm_fmadd_pd(x6m7, _mm_set1_pd(self.twiddle4.im), m0508b);
                            let (y05, y08) = AvxButterfly::butterfly2_f64_m128(m0508a, m0508b);

                            let m0607a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle6.re), u0);
                            let m0607a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle1.re), m0607a);
                            let m0607a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle5.re), m0607a);
                            let m0607a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle2.re), m0607a);
                            let m0607a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle4.re), m0607a);
                            let m0607a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle3.re), m0607a);
                            let m0607b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle6.im));
                            let m0607b =
                                _mm_fnmadd_pd(x2m11, _mm_set1_pd(self.twiddle1.im), m0607b);
                            let m0607b = _mm_fmadd_pd(x3m10, _mm_set1_pd(self.twiddle5.im), m0607b);
                            let m0607b = _mm_fnmadd_pd(x4m9, _mm_set1_pd(self.twiddle2.im), m0607b);
                            let m0607b = _mm_fmadd_pd(x5m8, _mm_set1_pd(self.twiddle4.im), m0607b);
                            let m0607b = _mm_fnmadd_pd(x6m7, _mm_set1_pd(self.twiddle3.im), m0607b);
                            let (y06, y07) = AvxButterfly::butterfly2_f64_m128(m0607a, m0607b);

                            // // Store results
                            _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + thirteenth..).as_mut_ptr().cast(),
                                y01,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 2 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 3 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 4 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 5 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 6 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 7 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y07,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 8 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y08,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 9 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y09,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 10 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 11 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y11,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 12 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y12,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 12..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix13<f64> {
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

impl AvxFmaRadix13<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let rotate = AvxRotate::<f32>::new(FftDirection::Inverse);

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f32>, 13>(13, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = 13;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 13;
                    let thirteenth = len / 13;

                    for data in chunk.chunks_exact_mut(len) {
                        let j = 0usize;

                        for j in j..thirteenth {
                            let u0 = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());
                            let twi = 12 * j;
                            let tw0tw1tw2tw3 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw4tw5tw6tw7 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );
                            let tw8tw9tw10tw11 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 8..).as_ptr().cast(),
                            );

                            let u1u2u3u4 = _mm256_fcmul_ps(
                                _mm256_load4_f32x2(
                                    data.get_unchecked(j + thirteenth..),
                                    data.get_unchecked(j + 2 * thirteenth..),
                                    data.get_unchecked(j + 3 * thirteenth..),
                                    data.get_unchecked(j + 4 * thirteenth..),
                                ),
                                tw0tw1tw2tw3,
                            );
                            let u5u6u7u8 = _mm256_fcmul_ps(
                                _mm256_load4_f32x2(
                                    data.get_unchecked(j + 5 * thirteenth..),
                                    data.get_unchecked(j + 6 * thirteenth..),
                                    data.get_unchecked(j + 7 * thirteenth..),
                                    data.get_unchecked(j + 8 * thirteenth..),
                                ),
                                tw4tw5tw6tw7,
                            );

                            let u9u10u11u12 = _mm256_fcmul_ps(
                                _mm256_load4_f32x2(
                                    data.get_unchecked(j + 9 * thirteenth..),
                                    data.get_unchecked(j + 10 * thirteenth..),
                                    data.get_unchecked(j + 11 * thirteenth..),
                                    data.get_unchecked(j + 12 * thirteenth..),
                                ),
                                tw8tw9tw10tw11,
                            );

                            let u1u2 = _mm256_castps256_ps128(u1u2u3u4);
                            let u3u4 = _mm256_extractf128_ps::<1>(u1u2u3u4);
                            let u9u10 = _mm256_castps256_ps128(u9u10u11u12);
                            let u5u6 = _mm256_castps256_ps128(u5u6u7u8);
                            let u7u8 = _mm256_extractf128_ps::<1>(u5u6u7u8);
                            let u11u12 = _mm256_extractf128_ps::<1>(u9u10u11u12);

                            let y00 = u0;
                            let (x1p12, x1m12) = AvxButterfly::butterfly2_f32_m128(
                                u1u2,
                                _mm_unpackhi_ps64(u11u12, u11u12),
                            ); // u1, u12
                            let x1m12 = rotate.rotate_m128(x1m12);
                            let y00 = _mm_add_ps(y00, x1p12);
                            let (x2p11, x2m11) = AvxButterfly::butterfly2_f32_m128(
                                _mm_unpackhi_ps64(u1u2, u1u2),
                                u11u12,
                            ); // u2, u11
                            let x2m11 = rotate.rotate_m128(x2m11);
                            let y00 = _mm_add_ps(y00, x2p11);
                            let (x3p10, x3m10) = AvxButterfly::butterfly2_f32_m128(
                                u3u4,
                                _mm_unpackhi_ps64(u9u10, u9u10),
                            ); // u3, u10
                            let x3m10 = rotate.rotate_m128(x3m10);
                            let y00 = _mm_add_ps(y00, x3p10);
                            let (x4p9, x4m9) = AvxButterfly::butterfly2_f32_m128(
                                _mm_unpackhi_ps64(u3u4, u3u4),
                                u9u10,
                            ); // u4, u9
                            let x4m9 = rotate.rotate_m128(x4m9);
                            let y00 = _mm_add_ps(y00, x4p9);
                            let (x5p8, x5m8) = AvxButterfly::butterfly2_f32_m128(
                                u5u6,
                                _mm_unpackhi_ps64(u7u8, u7u8),
                            ); // u5, u8
                            let x5m8 = rotate.rotate_m128(x5m8);
                            let y00 = _mm_add_ps(y00, x5p8);
                            let (x6p7, x6m7) = AvxButterfly::butterfly2_f32_m128(
                                _mm_unpackhi_ps64(u5u6, u5u6),
                                u7u8,
                            ); // u6, u7
                            let x6m7 = rotate.rotate_m128(x6m7);
                            let y00 = _mm_add_ps(y00, x6p7);

                            let m0112a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle1.re), u0);
                            let m0112a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle2.re), x2p11, m0112a);
                            let m0112a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle3.re), x3p10, m0112a);
                            let m0112a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle4.re), m0112a);
                            let m0112a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle5.re), m0112a);
                            let m0112a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle6.re), m0112a);
                            let m0112b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle1.im));
                            let m0112b = _mm_fmadd_ps(x2m11, _mm_set1_ps(self.twiddle2.im), m0112b);
                            let m0112b = _mm_fmadd_ps(x3m10, _mm_set1_ps(self.twiddle3.im), m0112b);
                            let m0112b = _mm_fmadd_ps(x4m9, _mm_set1_ps(self.twiddle4.im), m0112b);
                            let m0112b = _mm_fmadd_ps(x5m8, _mm_set1_ps(self.twiddle5.im), m0112b);
                            let m0112b = _mm_fmadd_ps(x6m7, _mm_set1_ps(self.twiddle6.im), m0112b);
                            let (y01, y12) = AvxButterfly::butterfly2_f32_m128(m0112a, m0112b);

                            let m0211a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle2.re), u0);
                            let m0211a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle4.re), m0211a);
                            let m0211a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle6.re), m0211a);
                            let m0211a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle5.re), m0211a);
                            let m0211a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle3.re), m0211a);
                            let m0211a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle1.re), m0211a);
                            let m0211b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle2.im));
                            let m0211b = _mm_fmadd_ps(x2m11, _mm_set1_ps(self.twiddle4.im), m0211b);
                            let m0211b = _mm_fmadd_ps(x3m10, _mm_set1_ps(self.twiddle6.im), m0211b);
                            let m0211b = _mm_fnmadd_ps(x4m9, _mm_set1_ps(self.twiddle5.im), m0211b);
                            let m0211b = _mm_fnmadd_ps(x5m8, _mm_set1_ps(self.twiddle3.im), m0211b);
                            let m0211b = _mm_fnmadd_ps(x6m7, _mm_set1_ps(self.twiddle1.im), m0211b);
                            let (y02, y11) = AvxButterfly::butterfly2_f32_m128(m0211a, m0211b);

                            let m0310a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle3.re), u0);
                            let m0310a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle6.re), m0310a);
                            let m0310a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle4.re), m0310a);
                            let m0310a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle1.re), m0310a);
                            let m0310a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle2.re), m0310a);
                            let m0310a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle5.re), m0310a);
                            let m0310b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle3.im));
                            let m0310b = _mm_fmadd_ps(x2m11, _mm_set1_ps(self.twiddle6.im), m0310b);
                            let m0310b =
                                _mm_fnmadd_ps(x3m10, _mm_set1_ps(self.twiddle4.im), m0310b);
                            let m0310b = _mm_fnmadd_ps(x4m9, _mm_set1_ps(self.twiddle1.im), m0310b);
                            let m0310b = _mm_fmadd_ps(x5m8, _mm_set1_ps(self.twiddle2.im), m0310b);
                            let m0310b = _mm_fmadd_ps(x6m7, _mm_set1_ps(self.twiddle5.im), m0310b);
                            let (y03, y10) = AvxButterfly::butterfly2_f32_m128(m0310a, m0310b);

                            let m0409a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle4.re), u0);
                            let m0409a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle5.re), m0409a);
                            let m0409a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle1.re), m0409a);
                            let m0409a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle3.re), m0409a);
                            let m0409a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle6.re), m0409a);
                            let m0409a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle2.re), m0409a);
                            let m0409b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle4.im));
                            let m0409b =
                                _mm_fnmadd_ps(x2m11, _mm_set1_ps(self.twiddle5.im), m0409b);
                            let m0409b =
                                _mm_fnmadd_ps(x3m10, _mm_set1_ps(self.twiddle1.im), m0409b);
                            let m0409b = _mm_fmadd_ps(x4m9, _mm_set1_ps(self.twiddle3.im), m0409b);
                            let m0409b = _mm_fnmadd_ps(x5m8, _mm_set1_ps(self.twiddle6.im), m0409b);
                            let m0409b = _mm_fnmadd_ps(x6m7, _mm_set1_ps(self.twiddle2.im), m0409b);
                            let (y04, y09) = AvxButterfly::butterfly2_f32_m128(m0409a, m0409b);

                            let m0508a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle5.re), u0);
                            let m0508a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle3.re), m0508a);
                            let m0508a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle2.re), m0508a);
                            let m0508a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle6.re), m0508a);
                            let m0508a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle1.re), m0508a);
                            let m0508a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle4.re), m0508a);
                            let m0508b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle5.im));
                            let m0508b =
                                _mm_fnmadd_ps(x2m11, _mm_set1_ps(self.twiddle3.im), m0508b);
                            let m0508b = _mm_fmadd_ps(x3m10, _mm_set1_ps(self.twiddle2.im), m0508b);
                            let m0508b = _mm_fnmadd_ps(x4m9, _mm_set1_ps(self.twiddle6.im), m0508b);
                            let m0508b = _mm_fnmadd_ps(x5m8, _mm_set1_ps(self.twiddle1.im), m0508b);
                            let m0508b = _mm_fmadd_ps(x6m7, _mm_set1_ps(self.twiddle4.im), m0508b);
                            let (y05, y08) = AvxButterfly::butterfly2_f32_m128(m0508a, m0508b);

                            let m0607a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle6.re), u0);
                            let m0607a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle1.re), m0607a);
                            let m0607a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle5.re), m0607a);
                            let m0607a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle2.re), m0607a);
                            let m0607a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle4.re), m0607a);
                            let m0607a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle3.re), m0607a);
                            let m0607b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle6.im));
                            let m0607b =
                                _mm_fnmadd_ps(x2m11, _mm_set1_ps(self.twiddle1.im), m0607b);
                            let m0607b = _mm_fmadd_ps(x3m10, _mm_set1_ps(self.twiddle5.im), m0607b);
                            let m0607b = _mm_fnmadd_ps(x4m9, _mm_set1_ps(self.twiddle2.im), m0607b);
                            let m0607b = _mm_fmadd_ps(x5m8, _mm_set1_ps(self.twiddle4.im), m0607b);
                            let m0607b = _mm_fnmadd_ps(x6m7, _mm_set1_ps(self.twiddle3.im), m0607b);
                            let (y06, y07) = AvxButterfly::butterfly2_f32_m128(m0607a, m0607b);

                            // Store results
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                y00,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + thirteenth..).as_mut_ptr().cast(),
                                y01,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 2 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 3 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 4 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 5 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 6 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 7 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y07,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 8 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y08,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 9 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y09,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 10 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 11 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y11,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 12 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y12,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 12..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix13<f32> {
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
    use crate::radix13::Radix13;
    use rand::Rng;

    #[test]
    fn test_avx_radix13() {
        for i in 1..4 {
            let size = 13usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let radix3_reference = Radix13::new(size, FftDirection::Forward).unwrap();

            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix_forward = AvxFmaRadix13::new(size, FftDirection::Forward).unwrap();
            radix3_reference.execute(&mut z_ref).unwrap();
            radix_forward.execute(&mut input).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 0.01,
                        "a_re {} != b_re {} for size {}, reference failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 0.01,
                        "a_im {} != b_im {} for size {}, reference failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            let radix_inverse = AvxFmaRadix13::new(size, FftDirection::Inverse).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input
                .iter()
                .map(|&x| Complex::new(x.re as f64, x.im as f64) * (1.0f64 / input.len() as f64))
                .map(|x| Complex::new(x.re as f32, x.im as f32))
                .collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_avx_radix13_f64() {
        for i in 1..4 {
            let size = 13usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }

            let radix7_reference = Radix13::new(size, FftDirection::Forward).unwrap();

            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix_forward = AvxFmaRadix13::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = AvxFmaRadix13::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();
            radix7_reference.execute(&mut z_ref).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {}, reference failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {}, reference failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input
                .iter()
                .map(|&x| x * (1.0f64 / input.len() as f64))
                .collect();

            input
                .iter()
                .zip(src.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {}",
                        a.im,
                        b.im,
                        size
                    );
                });
        }
    }
}
