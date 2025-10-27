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
use crate::avx::butterflies::AvxButterfly;
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_fcmul_pd, _mm_fcmul_ps, _mm_unpackhi_ps64,
    _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps, _mm256_fcmul_pd, _mm256_fcmul_ps,
    create_avx4_twiddles,
};
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::radix6::Radix6Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{bitreversed_transpose, compute_twiddle, is_power_of_six};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::x86_64::*;
use std::fmt::Display;

pub(crate) struct AvxFmaRadix6<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle_re: T,
    twiddle_im: [T; 8],
    direction: FftDirection,
    butterfly: Box<dyn CompositeFftExecutor<T> + Send + Sync>,
}

impl<
    T: Default
        + Clone
        + Radix6Twiddles
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
> AvxFmaRadix6<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix6<T>, ZaftError> {
        assert!(
            is_power_of_six(size as u64),
            "Input length must be a power of 6"
        );

        let twiddles = create_avx4_twiddles::<T, 6>(6, size, fft_direction)?;

        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);

        Ok(AvxFmaRadix6 {
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
            butterfly: T::butterfly6(fft_direction)?,
        })
    }
}

impl AvxFmaRadix6<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let twiddle_re = _mm256_set1_pd(self.twiddle_re);
            let twiddle_w_2 = _mm256_loadu_pd(self.twiddle_im.as_ptr().cast());

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f64>, 6>(6, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = 6;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 6;
                    let sixth = len / 6;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < sixth {
                            let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(5 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(5 * j + 2..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(5 * j + 4..).as_ptr().cast(),
                            );
                            let tw3 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(5 * j + 6..).as_ptr().cast(),
                            );
                            let tw4 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(5 * j + 8..).as_ptr().cast(),
                            );

                            let u1 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(data.get_unchecked(j + sixth..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 2 * sixth..).as_ptr().cast(),
                                ),
                                tw1,
                            );
                            let u3 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 3 * sixth..).as_ptr().cast(),
                                ),
                                tw2,
                            );
                            let u4 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 4 * sixth..).as_ptr().cast(),
                                ),
                                tw3,
                            );
                            let u5 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 5 * sixth..).as_ptr().cast(),
                                ),
                                tw4,
                            );

                            let (t0, t2, t4) =
                                AvxButterfly::butterfly3_f64(u0, u2, u4, twiddle_re, twiddle_w_2);
                            let (t1, t3, t5) =
                                AvxButterfly::butterfly3_f64(u3, u5, u1, twiddle_re, twiddle_w_2);
                            let (y0, y3) = AvxButterfly::butterfly2_f64(t0, t1);
                            let (y4, y1) = AvxButterfly::butterfly2_f64(t2, t3);
                            let (y2, y5) = AvxButterfly::butterfly2_f64(t4, t5);

                            // Store results
                            _mm256_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + sixth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 2 * sixth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 3 * sixth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 4 * sixth..).as_mut_ptr().cast(),
                                y4,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 5 * sixth..).as_mut_ptr().cast(),
                                y5,
                            );

                            j += 2;
                        }

                        for j in j..sixth {
                            let u0 = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(5 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(5 * j + 2..).as_ptr().cast(),
                            );

                            let u1u2 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(data.get_unchecked(j + sixth..).as_ptr().cast()),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 2 * sixth..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let u3u4 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 3 * sixth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 4 * sixth..).as_ptr().cast(),
                                    ),
                                ),
                                tw1,
                            );
                            let u5 = _mm_fcmul_pd(
                                _mm_loadu_pd(data.get_unchecked(j + 5 * sixth..).as_ptr().cast()),
                                _mm_loadu_pd(m_twiddles.get_unchecked(5 * j + 4..).as_ptr().cast()),
                            );

                            let (t0, t2, t4) = AvxButterfly::butterfly3_f64_m128(
                                u0,
                                _mm256_extractf128_pd::<1>(u1u2),
                                _mm256_extractf128_pd::<1>(u3u4),
                                _mm256_castpd256_pd128(twiddle_re),
                                _mm256_castpd256_pd128(twiddle_w_2),
                            );
                            let (t1, t3, t5) = AvxButterfly::butterfly3_f64_m128(
                                _mm256_castpd256_pd128(u3u4),
                                u5,
                                _mm256_castpd256_pd128(u1u2),
                                _mm256_castpd256_pd128(twiddle_re),
                                _mm256_castpd256_pd128(twiddle_w_2),
                            );
                            let (y0, y3) = AvxButterfly::butterfly2_f64_m128(t0, t1);
                            let (y4, y1) = AvxButterfly::butterfly2_f64_m128(t2, t3);
                            let (y2, y5) = AvxButterfly::butterfly2_f64_m128(t4, t5);

                            // Store results
                            _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + sixth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 2 * sixth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 3 * sixth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 4 * sixth..).as_mut_ptr().cast(),
                                y4,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 5 * sixth..).as_mut_ptr().cast(),
                                y5,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 5..];
                }
            }
        }

        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix6<f64> {
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

impl AvxFmaRadix6<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let twiddle_re = _mm256_set1_ps(self.twiddle_re);
            let twiddle_w_2 = _mm256_loadu_ps(self.twiddle_im.as_ptr().cast());

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f32>, 6>(6, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = 6;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 6;
                    let sixth = len / 6;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 4 < sixth {
                            let u0 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(5 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(5 * j + 4..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(5 * j + 8..).as_ptr().cast(),
                            );
                            let tw3 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(5 * j + 12..).as_ptr().cast(),
                            );
                            let tw4 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(5 * j + 16..).as_ptr().cast(),
                            );

                            let rk1 =
                                _mm256_loadu_ps(data.get_unchecked(j + sixth..).as_ptr().cast());
                            let rk2 = _mm256_loadu_ps(
                                data.get_unchecked(j + 2 * sixth..).as_ptr().cast(),
                            );
                            let rk3 = _mm256_loadu_ps(
                                data.get_unchecked(j + 3 * sixth..).as_ptr().cast(),
                            );
                            let rk4 = _mm256_loadu_ps(
                                data.get_unchecked(j + 4 * sixth..).as_ptr().cast(),
                            );

                            let u1 = _mm256_fcmul_ps(rk1, tw0);
                            let u2 = _mm256_fcmul_ps(rk2, tw1);
                            let u3 = _mm256_fcmul_ps(rk3, tw2);
                            let u4 = _mm256_fcmul_ps(rk4, tw3);
                            let u5 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 5 * sixth..).as_ptr().cast(),
                                ),
                                tw4,
                            );

                            let (t0, t2, t4) =
                                AvxButterfly::butterfly3_f32(u0, u2, u4, twiddle_re, twiddle_w_2);
                            let (t1, t3, t5) =
                                AvxButterfly::butterfly3_f32(u3, u5, u1, twiddle_re, twiddle_w_2);
                            let (y0, y3) = AvxButterfly::butterfly2_f32(t0, t1);
                            let (y4, y1) = AvxButterfly::butterfly2_f32(t2, t3);
                            let (y2, y5) = AvxButterfly::butterfly2_f32(t4, t5);

                            // Store results
                            _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + sixth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 2 * sixth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 3 * sixth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 4 * sixth..).as_mut_ptr().cast(),
                                y4,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 5 * sixth..).as_mut_ptr().cast(),
                                y5,
                            );

                            j += 4;
                        }

                        while j + 2 < sixth {
                            let u0 = _mm_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(5 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(5 * j + 4..).as_ptr().cast(),
                            );
                            let tw2 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(5 * j + 8..).as_ptr().cast());

                            let u1u2 = _mm256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(data.get_unchecked(j + sixth..).as_ptr().cast()),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 2 * sixth..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let u3u4 = _mm256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 3 * sixth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 4 * sixth..).as_ptr().cast(),
                                    ),
                                ),
                                tw1,
                            );
                            let u5 = _mm_fcmul_ps(
                                _mm_loadu_ps(data.get_unchecked(j + 5 * sixth..).as_ptr().cast()),
                                tw2,
                            );

                            let u1 = _mm256_castps256_ps128(u1u2);
                            let u2 = _mm256_extractf128_ps::<1>(u1u2);
                            let u3 = _mm256_castps256_ps128(u3u4);
                            let u4 = _mm256_extractf128_ps::<1>(u3u4);

                            let (t0, t2, t4) = AvxButterfly::butterfly3_f32_m128(
                                u0,
                                u2,
                                u4,
                                _mm256_castps256_ps128(twiddle_re),
                                _mm256_castps256_ps128(twiddle_w_2),
                            );
                            let (t1, t3, t5) = AvxButterfly::butterfly3_f32_m128(
                                u3,
                                u5,
                                u1,
                                _mm256_castps256_ps128(twiddle_re),
                                _mm256_castps256_ps128(twiddle_w_2),
                            );
                            let (y0, y3) = AvxButterfly::butterfly2_f32_m128(t0, t1);
                            let (y4, y1) = AvxButterfly::butterfly2_f32_m128(t2, t3);
                            let (y2, y5) = AvxButterfly::butterfly2_f32_m128(t4, t5);

                            // Store results
                            _mm_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + sixth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 2 * sixth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 3 * sixth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 4 * sixth..).as_mut_ptr().cast(),
                                y4,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 5 * sixth..).as_mut_ptr().cast(),
                                y5,
                            );

                            j += 2;
                        }

                        for j in j..sixth {
                            let u0 = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(5 * j..).as_ptr().cast());
                            let tw1 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(5 * j + 2..).as_ptr().cast());

                            let u1u2 = _mm_fcmul_ps(
                                _mm_unpacklo_ps64(
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + sixth..).as_ptr().cast(),
                                    ),
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 2 * sixth..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let u3u4 = _mm_fcmul_ps(
                                _mm_unpacklo_ps64(
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 3 * sixth..).as_ptr().cast(),
                                    ),
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 4 * sixth..).as_ptr().cast(),
                                    ),
                                ),
                                tw1,
                            );
                            let u5 = _mm_fcmul_ps(
                                _m128s_load_f32x2(
                                    data.get_unchecked(j + 5 * sixth..).as_ptr().cast(),
                                ),
                                _m128s_load_f32x2(
                                    m_twiddles.get_unchecked(5 * j + 4..).as_ptr().cast(),
                                ),
                            );

                            let u1 = u1u2;
                            let u2 = _mm_unpackhi_ps64(u1u2, u1u2);
                            let u3 = u3u4;
                            let u4 = _mm_unpackhi_ps64(u3u4, u3u4);

                            let (t0, t2, t4) = AvxButterfly::butterfly3_f32_m128(
                                u0,
                                u2,
                                u4,
                                _mm256_castps256_ps128(twiddle_re),
                                _mm256_castps256_ps128(twiddle_w_2),
                            );
                            let (t1, t3, t5) = AvxButterfly::butterfly3_f32_m128(
                                u3,
                                u5,
                                u1,
                                _mm256_castps256_ps128(twiddle_re),
                                _mm256_castps256_ps128(twiddle_w_2),
                            );
                            let (y0, y3) = AvxButterfly::butterfly2_f32_m128(t0, t1);
                            let (y4, y1) = AvxButterfly::butterfly2_f32_m128(t2, t3);
                            let (y2, y5) = AvxButterfly::butterfly2_f32_m128(t4, t5);

                            // Store results
                            _m128s_store_f32x2(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + sixth..).as_mut_ptr().cast(),
                                y1,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 2 * sixth..).as_mut_ptr().cast(),
                                y2,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 3 * sixth..).as_mut_ptr().cast(),
                                y3,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 4 * sixth..).as_mut_ptr().cast(),
                                y4,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 5 * sixth..).as_mut_ptr().cast(),
                                y5,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 5..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix6<f32> {
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
    use crate::dft::Dft;
    use rand::Rng;

    #[test]
    fn test_neon_radix6() {
        for i in 1..4 {
            let size = 6usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref_input = input.to_vec();
            let radix_forward = AvxFmaRadix6::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = AvxFmaRadix6::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();

            let reference_dft = Dft::new(size, FftDirection::Forward).unwrap();
            reference_dft.execute(&mut ref_input).unwrap();

            input
                .iter()
                .zip(ref_input.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-2,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-2,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

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
    fn test_avx_radix6_f64() {
        for i in 1..4 {
            let size = 6usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref_input = input.to_vec();
            let radix_forward = AvxFmaRadix6::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = AvxFmaRadix6::new(size, FftDirection::Inverse).unwrap();

            let reference_dft = Dft::new(size, FftDirection::Forward).unwrap();
            reference_dft.execute(&mut ref_input).unwrap();

            radix_forward.execute(&mut input).unwrap();

            input
                .iter()
                .zip(ref_input.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-7,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-7,
                        "a_im {} != b_im {} for size {} at {idx}",
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

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for size {}",
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
