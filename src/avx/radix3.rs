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
use crate::avx::util::{
    _m128d_fma_mul_complex, _m128s_fma_mul_complex, _m128s_load_f32x2, _m128s_store_f32x2,
    _m256d_mul_complex, _m256s_mul_complex, shuffle,
};
use crate::radix3::Radix3Twiddles;
use crate::traits::FftTrigonometry;
use crate::util::{compute_twiddle, digit_reverse_indices, permute_inplace};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxFmaRadix3<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    twiddle_re: T,
    twiddle_im: [T; 8],
}

impl<T: Default + Clone + Radix3Twiddles + 'static + Copy + FftTrigonometry + Float> AvxFmaRadix3<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix3<T>, ZaftError> {
        assert!(
            size.is_power_of_two() || size % 3 == 0,
            "Input length must be divisible by 3"
        );

        let twiddles = T::make_twiddles(size, fft_direction)?;
        let rev = digit_reverse_indices(size, 3)?;

        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);

        Ok(AvxFmaRadix3 {
            permutations: rev,
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
        })
    }
}

impl AvxFmaRadix3<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        // Trit-reversal permutation
        permute_inplace(in_place, &self.permutations);

        let mut len = 3;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            let twiddle_re = _mm256_set1_pd(self.twiddle_re);
            let twiddle_w_2 = _mm256_loadu_pd(self.twiddle_im.as_ptr().cast());

            while len <= self.execution_length {
                let third = len / 3;
                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 2 < third {
                        let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());
                        let u1 = _m256d_mul_complex(
                            _mm256_loadu_pd(data.get_unchecked(j + third..).as_ptr().cast()),
                            _mm256_loadu2_m128d(
                                m_twiddles.get_unchecked(2 * (j + 1)..).as_ptr().cast(),
                                m_twiddles.get_unchecked(2 * j..).as_ptr().cast(),
                            ),
                        );
                        let u2 = _m256d_mul_complex(
                            _mm256_loadu_pd(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                            _mm256_loadu2_m128d(
                                m_twiddles.get_unchecked(2 * (j + 1) + 1..).as_ptr().cast(),
                                m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast(),
                            ),
                        );

                        // Radix-3 butterfly
                        let xp = _mm256_add_pd(u1, u2);
                        let xn = _mm256_sub_pd(u1, u2);
                        let sum = _mm256_add_pd(u0, xp);

                        let w_1 = _mm256_fmadd_pd(twiddle_re, xp, u0);
                        let perm = _mm256_permute_pd::<0b0101>(xn);
                        let vw_2 = _mm256_mul_pd(twiddle_w_2, perm);

                        let vy0 = sum;
                        let vy1 = _mm256_add_pd(w_1, vw_2);
                        let vy2 = _mm256_sub_pd(w_1, vw_2);

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
                        let u1 = _m128d_fma_mul_complex(
                            _mm_loadu_pd(data.get_unchecked(j + third..).as_ptr().cast()),
                            _mm_loadu_pd(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                        );
                        let u2 = _m128d_fma_mul_complex(
                            _mm_loadu_pd(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                            _mm_loadu_pd(m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast()),
                        );

                        // Radix-3 butterfly
                        let xp = _mm_add_pd(u1, u2);
                        let xn = _mm_sub_pd(u1, u2);
                        let sum = _mm_add_pd(u0, xp);

                        let w_1 = _mm_fmadd_pd(_mm256_castpd256_pd128(twiddle_re), xp, u0);
                        let vw_2 = _mm_mul_pd(
                            _mm256_castpd256_pd128(twiddle_w_2),
                            _mm_shuffle_pd::<0b01>(xn, xn),
                        );

                        let vy0 = sum;
                        let vy1 = _mm_add_pd(w_1, vw_2);
                        let vy2 = _mm_sub_pd(w_1, vw_2);

                        _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                        _mm_storeu_pd(data.get_unchecked_mut(j + third..).as_mut_ptr().cast(), vy1);
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                            vy2,
                        );
                    }
                }

                m_twiddles = &m_twiddles[third * 2..];
                len *= 3;
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix3<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }
}

impl AvxFmaRadix3<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        // Trit-reversal permutation
        permute_inplace(in_place, &self.permutations);

        let mut len = 3;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            let twiddle_re = _mm256_set1_ps(self.twiddle_re);
            let twiddle_w_2 = _mm256_loadu_ps(self.twiddle_im.as_ptr().cast());

            while len <= self.execution_length {
                let third = len / 3;
                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 4 < third {
                        let u0 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());
                        let u1 = _m256s_mul_complex(
                            _mm256_loadu_ps(data.get_unchecked(j + third..).as_ptr().cast()),
                            _mm256_loadu_ps(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                        );
                        let u2 = _m256s_mul_complex(
                            _mm256_loadu_ps(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                            _mm256_loadu_ps(m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast()),
                        );

                        // Radix-3 butterfly
                        let xp_0 = _mm256_add_ps(u1, u2);
                        let xn_0 = _mm256_sub_ps(u1, u2);
                        let sum_0 = _mm256_add_ps(u0, xp_0);

                        const SH: i32 = shuffle(2, 3, 0, 1);

                        let vw_1_1 = _mm256_fmadd_ps(twiddle_re, xp_0, u0);
                        let vw_2_1 = _mm256_mul_ps(twiddle_w_2, _mm256_permute_ps::<SH>(xn_0));

                        let vy0 = sum_0;
                        let vy1 = _mm256_add_ps(vw_1_1, vw_2_1);
                        let vy2 = _mm256_sub_ps(vw_1_1, vw_2_1);

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

                    while j + 3 < third {
                        let u0 = _mm256_insertf128_ps::<1>(
                            _mm256_castps128_ps256(_mm_loadu_ps(
                                data.get_unchecked(j..).as_ptr().cast(),
                            )),
                            _m128s_load_f32x2(data.get_unchecked(j + 2..).as_ptr().cast()),
                        );
                        let u1 = _m256s_mul_complex(
                            _mm256_insertf128_ps::<1>(
                                _mm256_castps128_ps256(_mm_loadu_ps(
                                    data.get_unchecked(j + third..).as_ptr().cast(),
                                )),
                                _m128s_load_f32x2(
                                    data.get_unchecked(j + third + 2..).as_ptr().cast(),
                                ),
                            ),
                            _mm256_insertf128_ps::<1>(
                                _mm256_castps128_ps256(_mm_loadu_ps(
                                    m_twiddles.get_unchecked(2 * j..).as_ptr().cast(),
                                )),
                                _m128s_load_f32x2(
                                    m_twiddles.get_unchecked(2 * j + 2..).as_ptr().cast(),
                                ),
                            ),
                        );
                        let u2 = _m256s_mul_complex(
                            _mm256_insertf128_ps::<1>(
                                _mm256_castps128_ps256(_mm_loadu_ps(
                                    data.get_unchecked(j + 2 * third..).as_ptr().cast(),
                                )),
                                _m128s_load_f32x2(
                                    data.get_unchecked(j + 2 * third + 2..).as_ptr().cast(),
                                ),
                            ),
                            _mm256_insertf128_ps::<1>(
                                _mm256_castps128_ps256(_mm_loadu_ps(
                                    m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast(),
                                )),
                                _m128s_load_f32x2(
                                    m_twiddles.get_unchecked(2 * j + 1 + 2..).as_ptr().cast(),
                                ),
                            ),
                        );

                        // Radix-3 butterfly
                        let xp_0 = _mm256_add_ps(u1, u2);
                        let xn_0 = _mm256_sub_ps(u1, u2);
                        let sum_0 = _mm256_add_ps(u0, xp_0);

                        const SH: i32 = shuffle(2, 3, 0, 1);

                        let vw_1_1 = _mm256_fmadd_ps(twiddle_re, xp_0, u0);
                        let vw_2_1 = _mm256_mul_ps(twiddle_w_2, _mm256_permute_ps::<SH>(xn_0));

                        let vy0 = sum_0;
                        let vy1 = _mm256_add_ps(vw_1_1, vw_2_1);
                        let vy2 = _mm256_sub_ps(vw_1_1, vw_2_1);

                        _mm_storeu_ps(
                            data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                            _mm256_castps256_ps128(vy0),
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 2..).as_mut_ptr().cast(),
                            _mm256_extractf128_ps::<1>(vy0),
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + third..).as_mut_ptr().cast(),
                            _mm256_castps256_ps128(vy1),
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + third + 2..).as_mut_ptr().cast(),
                            _mm256_extractf128_ps::<1>(vy1),
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                            _mm256_castps256_ps128(vy2),
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 2 * third + 2..)
                                .as_mut_ptr()
                                .cast(),
                            _mm256_extractf128_ps::<1>(vy2),
                        );

                        j += 3;
                    }

                    for j in j..third {
                        let u0 = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());
                        let u1 = _m128s_fma_mul_complex(
                            _m128s_load_f32x2(data.get_unchecked(j + third..).as_ptr().cast()),
                            _m128s_load_f32x2(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                        );
                        let u2 = _m128s_fma_mul_complex(
                            _m128s_load_f32x2(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                            _m128s_load_f32x2(
                                m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast(),
                            ),
                        );

                        // Radix-3 butterfly
                        let xp = _mm_add_ps(u1, u2);
                        let xn = _mm_sub_ps(u1, u2);
                        let sum = _mm_add_ps(u0, xp);

                        const SH: i32 = shuffle(2, 3, 0, 1);

                        let w_1 = _mm_fmadd_ps(_mm256_castps256_ps128(twiddle_re), xp, u0);
                        let vw_2 = _mm_mul_ps(
                            _mm256_castps256_ps128(twiddle_w_2),
                            _mm_shuffle_ps::<SH>(xn, xn),
                        );

                        let vy0 = sum;
                        let vy1 = _mm_add_ps(w_1, vw_2);
                        let vy2 = _mm_sub_ps(w_1, vw_2);

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

                m_twiddles = &m_twiddles[third * 2..];
                len *= 3;
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix3<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }
}
