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
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _m256_fcmul_ps, _mm_fcmul_pd, _mm_fcmul_ps,
    _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_fcmul_pd, _mm256_load4_f32x2,
    shuffle,
};
use crate::err::try_vec;
use crate::radix10::Radix10Twiddles;
use crate::util::{bitreversed_transpose, is_power_of_ten};
use crate::{FftDirection, FftExecutor, Zaft, ZaftError};
use num_complex::Complex;
use num_traits::{Float, MulAdd};
use std::arch::x86_64::*;

fn create_twiddles<T: Radix10Twiddles + Sized>(
    size: usize,
    fft_direction: FftDirection,
) -> Result<Vec<Complex<T>>, ZaftError> {
    T::make_twiddles_with_base(10, size, fft_direction)
}

pub(crate) struct AvxFmaRadix10d {
    twiddles: Vec<Complex<f64>>,
    execution_length: usize,
    bf5: AvxFastButterfly5d,
    direction: FftDirection,
    butterfly: Box<dyn FftExecutor<f64> + Send + Sync>,
}

impl AvxFmaRadix10d {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix10d, ZaftError> {
        assert!(
            is_power_of_ten(size as u64),
            "Input length must be a power of 10"
        );

        let twiddles = create_twiddles(size, fft_direction)?;

        Ok(AvxFmaRadix10d {
            execution_length: size,
            twiddles,
            bf5: unsafe { AvxFastButterfly5d::new(fft_direction) },
            direction: fft_direction,
            butterfly: Zaft::strategy(10, fft_direction)?,
        })
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
                bitreversed_transpose::<Complex<f64>, 10>(10, chunk, &mut scratch);

                self.butterfly.execute(&mut scratch)?;

                let mut len = 10;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 10;
                    let tenth = len / 10;

                    for data in scratch.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        const LO_HI: i32 = 0b0011_0000;
                        const HI_LO: i32 = 0b0010_0001;
                        const HI_HI: i32 = 0b0011_0001;
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
                            let tw4_tw0_1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 8..).as_ptr().cast(),
                            );
                            let tw1_2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 10..).as_ptr().cast(),
                            );
                            let tw2_2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 12..).as_ptr().cast(),
                            );
                            let tw3_2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 14..).as_ptr().cast(),
                            );
                            let tw4_2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 16..).as_ptr().cast(),
                            );

                            let u1 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(data.get_unchecked(j + tenth..).as_ptr().cast()),
                                _mm256_permute2f128_pd::<LO_HI>(tw0, tw4_tw0_1),
                            );
                            let u2 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 2 * tenth..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<HI_LO>(tw0, tw1_2),
                            );
                            let u3 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 3 * tenth..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<LO_HI>(tw1, tw1_2),
                            );
                            let u4 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 4 * tenth..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<HI_LO>(tw1, tw2_2),
                            );
                            let u5 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 5 * tenth..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<LO_HI>(tw2, tw2_2),
                            );
                            let u6 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 6 * tenth..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<HI_LO>(tw2, tw3_2),
                            );
                            let u7 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 7 * tenth..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<LO_HI>(tw3, tw3_2),
                            );
                            let u8 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 8 * tenth..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<HI_LO>(tw3, tw4_2),
                            );
                            let u9 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 9 * tenth..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<LO_HI>(tw4_tw0_1, tw4_2),
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
                chunk.copy_from_slice(&scratch);
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

pub(crate) struct AvxFmaRadix10f {
    twiddles: Vec<Complex<f32>>,
    execution_length: usize,
    bf5: AvxFastButterfly5f,
    direction: FftDirection,
    butterfly: Box<dyn FftExecutor<f32> + Send + Sync>,
}

impl AvxFmaRadix10f {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix10f, ZaftError> {
        assert!(
            is_power_of_ten(size as u64),
            "Input length must be a power of 10"
        );

        let twiddles = create_twiddles(size, fft_direction)?;

        Ok(AvxFmaRadix10f {
            execution_length: size,
            twiddles,
            bf5: unsafe { AvxFastButterfly5f::new(fft_direction) },
            direction: fft_direction,
            butterfly: Zaft::strategy(10, fft_direction)?,
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
                bitreversed_transpose::<Complex<f32>, 10>(10, chunk, &mut scratch);

                self.butterfly.execute(&mut scratch)?;

                let mut len = 10;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 10;
                    let tenth = len / 10;

                    for data in scratch.chunks_exact_mut(len) {
                        let mut j = 0usize;

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

                            let u1u2u3u4 = _m256_fcmul_ps(
                                _mm256_load4_f32x2(
                                    data.get_unchecked(j + tenth..),
                                    data.get_unchecked(j + 2 * tenth..),
                                    data.get_unchecked(j + 3 * tenth..),
                                    data.get_unchecked(j + 4 * tenth..),
                                ),
                                tw0tw1tw2tw3,
                            );
                            let u5u6u7u8 = _m256_fcmul_ps(
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
                chunk.copy_from_slice(&scratch);
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
    use crate::radix10::Radix10;
    use rand::Rng;

    #[test]
    fn test_avx_radix10() {
        for i in 1..4 {
            let size = 10usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let radix10_reference = Radix10::new(size, FftDirection::Forward).unwrap();

            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix_forward = AvxFmaRadix10f::new(size, FftDirection::Forward).unwrap();
            radix10_reference.execute(&mut z_ref).unwrap();
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

            let radix_inverse = AvxFmaRadix10f::new(size, FftDirection::Inverse).unwrap();
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
    fn test_avx_radix10_f64() {
        for i in 1..4 {
            let size = 10usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }

            let radix7_reference = Radix10::new(size, FftDirection::Forward).unwrap();

            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix_forward = AvxFmaRadix10d::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = AvxFmaRadix10d::new(size, FftDirection::Inverse).unwrap();
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
