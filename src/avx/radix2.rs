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
    _m256d_mul_complex, _m256s_mul_complex,
};
use crate::radix2::Radix2Twiddles;
use crate::util::{digit_reverse_indices, permute_inplace};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxFmaRadix2<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    direction: FftDirection,
}

impl<T: Default + Clone + Radix2Twiddles> AvxFmaRadix2<T> {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix2<T>, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");

        let twiddles = T::make_twiddles(size, fft_direction)?;

        // Bit-reversal permutation
        let rev = digit_reverse_indices(size, 2)?;

        Ok(AvxFmaRadix2 {
            permutations: rev,
            execution_length: size,
            twiddles,
            direction: fft_direction,
        })
    }
}

impl AvxFmaRadix2<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        permute_inplace(in_place, &self.permutations);

        let mut len = 2;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();
            while len <= self.execution_length {
                let half = len / 2;
                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;
                    while j + 2 < half {
                        let u0_0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());
                        let u1_0 = _mm256_loadu_pd(data.get_unchecked(j + half..).as_ptr().cast());
                        let tw0 = _mm256_loadu_pd(m_twiddles.get_unchecked(j..).as_ptr().cast());

                        let t_0 = _m256d_mul_complex(u1_0, tw0);
                        let y0 = _mm256_add_pd(u0_0, t_0);
                        let y1 = _mm256_sub_pd(u0_0, t_0);

                        _mm256_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + half..).as_mut_ptr().cast(),
                            y1,
                        );
                        j += 2;
                    }
                    for j in j..half {
                        let u = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());
                        let tw = _mm_loadu_pd(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let u1 = _mm_loadu_pd(data.get_unchecked(j + half..).as_ptr().cast());
                        let t = _m128d_fma_mul_complex(u1, tw);
                        let y0 = _mm_add_pd(u, t);
                        let y1 = _mm_sub_pd(u, t);
                        _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _mm_storeu_pd(data.get_unchecked_mut(j + half..).as_mut_ptr().cast(), y1);
                    }
                }

                len *= 2;
                m_twiddles = &m_twiddles[half..];
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix2<f64> {
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

impl AvxFmaRadix2<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        permute_inplace(in_place, &self.permutations);

        let mut len = 2;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            while len <= self.execution_length {
                let half = len / 2;
                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 4 < half {
                        let u0_0 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());
                        let u1_0 = _mm256_loadu_ps(data.get_unchecked(j + half..).as_ptr().cast());
                        let tw0 = _mm256_loadu_ps(m_twiddles.get_unchecked(j..).as_ptr().cast());

                        let t_0 = _m256s_mul_complex(tw0, u1_0);

                        let y0 = _mm256_add_ps(u0_0, t_0);
                        let y1 = _mm256_sub_ps(u0_0, t_0);

                        _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + half..).as_mut_ptr().cast(),
                            y1,
                        );

                        j += 4;
                    }

                    while j + 2 < half {
                        let u = _mm_loadu_ps(data.get_unchecked(j..).as_ptr().cast());
                        let tw = _mm_loadu_ps(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let u1 = _mm_loadu_ps(data.get_unchecked(j + half..).as_ptr().cast());
                        let t = _m128s_fma_mul_complex(tw, u1);
                        let y0 = _mm_add_ps(u, t);
                        let y1 = _mm_sub_ps(u, t);
                        _mm_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _mm_storeu_ps(data.get_unchecked_mut(j + half..).as_mut_ptr().cast(), y1);

                        j += 2;
                    }

                    for j in j..half {
                        let u = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());
                        let tw = _m128s_load_f32x2(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let u1 = _m128s_load_f32x2(data.get_unchecked(j + half..).as_ptr().cast());
                        let t = _m128s_fma_mul_complex(tw, u1);
                        let y0 = _mm_add_ps(u, t);
                        let y1 = _mm_sub_ps(u, t);
                        _m128s_store_f32x2(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + half..).as_mut_ptr().cast(),
                            y1,
                        );
                    }
                }

                len *= 2;
                m_twiddles = &m_twiddles[half..];
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix2<f32> {
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
