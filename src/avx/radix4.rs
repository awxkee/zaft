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
    _m256d_mul_complex, _m256s_mul_complex, _mm_unpackhi_ps64, _mm_unpacklo_ps64,
    _mm128s_deinterleave3_epi64, _mm256_create_ps, _mm256_unpackhi_pd2, _mm256_unpacklo_pd2,
    _mm256s_deinterleave3_epi64, shuffle,
};
use crate::radix4::Radix4Twiddles;
use crate::util::{digit_reverse_indices, permute_inplace};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxFmaRadix4<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    direction: FftDirection,
}

impl<T: Default + Clone + Radix4Twiddles> AvxFmaRadix4<T> {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix4<T>, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");
        assert_eq!(size.trailing_zeros() % 2, 0, "Radix-4 requires power of 4");

        let twiddles = T::make_twiddles(size, fft_direction)?;
        let rev = digit_reverse_indices(size, 4)?;

        Ok(AvxFmaRadix4 {
            permutations: rev,
            execution_length: size,
            twiddles,
            direction: fft_direction,
        })
    }
}

impl AvxFmaRadix4<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let v_i_multiplier = unsafe {
            match self.direction {
                FftDirection::Inverse => _mm256_loadu_pd([-0.0f64, 0.0, -0.0f64, 0.0].as_ptr()),
                FftDirection::Forward => _mm256_loadu_pd([0.0f64, -0.0, 0.0f64, -0.0].as_ptr()),
            }
        };

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // bit reversal first
            permute_inplace(chunk, &self.permutations);

            let mut len = 4;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len <= self.execution_length {
                    let quarter = len / 4;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;
                        while j + 2 < quarter {
                            let a = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(3 * (j + 1)..).as_ptr().cast(),
                            );

                            let b = _m256d_mul_complex(
                                _mm256_loadu_pd(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                _mm256_unpacklo_pd2(tw0, tw1),
                            );
                            let c = _m256d_mul_complex(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 2 * quarter..).as_ptr().cast(),
                                ),
                                _mm256_unpackhi_pd2(tw0, tw1),
                            );
                            let d = _m256d_mul_complex(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 3 * quarter..).as_ptr().cast(),
                                ),
                                _mm256_loadu2_m128d(
                                    m_twiddles.get_unchecked(3 * (j + 1) + 2..).as_ptr().cast(),
                                    m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast(),
                                ),
                            );

                            // radix-4 butterfly
                            let t0 = _mm256_add_pd(a, c);
                            let t1 = _mm256_sub_pd(a, c);
                            let t2 = _mm256_add_pd(b, d);
                            let mut t3 = _mm256_sub_pd(b, d);
                            t3 = _mm256_xor_pd(_mm256_permute_pd::<0b0101>(t3), v_i_multiplier);

                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                _mm256_add_pd(t0, t2),
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                _mm256_add_pd(t1, t3),
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 2 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                _mm256_sub_pd(t0, t2),
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 3 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                _mm256_sub_pd(t1, t3),
                            );

                            j += 2;
                        }
                        for j in j..quarter {
                            let a = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());

                            let b = _m128d_fma_mul_complex(
                                _mm_loadu_pd(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                _mm256_castpd256_pd128(tw0),
                            );
                            let c = _m128d_fma_mul_complex(
                                _mm_loadu_pd(data.get_unchecked(j + 2 * quarter..).as_ptr().cast()),
                                _mm256_extractf128_pd::<1>(tw0),
                            );
                            let d = _m128d_fma_mul_complex(
                                _mm_loadu_pd(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                                _mm_loadu_pd(m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast()),
                            );

                            // radix-4 butterfly
                            let t0 = _mm_add_pd(a, c);
                            let t1 = _mm_sub_pd(a, c);
                            let t2 = _mm_add_pd(b, d);
                            let mut t3 = _mm_sub_pd(b, d);
                            t3 = _mm_xor_pd(
                                _mm_shuffle_pd::<0b01>(t3, t3),
                                _mm256_castpd256_pd128(v_i_multiplier),
                            );

                            _mm_storeu_pd(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                _mm_add_pd(t0, t2),
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                _mm_add_pd(t1, t3),
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 2 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                _mm_sub_pd(t0, t2),
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 3 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                _mm_sub_pd(t1, t3),
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[quarter * 3..];
                    len *= 4;
                }
            }
        }

        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix4<f64> {
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

impl AvxFmaRadix4<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let v_i_multiplier = unsafe {
            match self.direction {
                FftDirection::Inverse => {
                    _mm256_loadu_ps([-0.0f32, 0.0, -0.0, 0.0, -0.0f32, 0.0, -0.0, 0.0].as_ptr())
                }
                FftDirection::Forward => {
                    _mm256_loadu_ps([0.0f32, -0.0, 0.0, -0.0, 0.0f32, -0.0, 0.0, -0.0].as_ptr())
                }
            }
        };

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // bit reversal first
            permute_inplace(chunk, &self.permutations);

            let mut len = 4;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len <= self.execution_length {
                    let quarter = len / 4;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 4 < quarter {
                            let a0 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(3 * j + 4..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(3 * j + 8..).as_ptr().cast(),
                            );

                            let rk1 =
                                _mm256_loadu_ps(data.get_unchecked(j + quarter..).as_ptr().cast());
                            let rk2 = _mm256_loadu_ps(
                                data.get_unchecked(j + 2 * quarter..).as_ptr().cast(),
                            );
                            let rk3 = _mm256_loadu_ps(
                                data.get_unchecked(j + 3 * quarter..).as_ptr().cast(),
                            );

                            let (xw0, xw1, xw2) = _mm256s_deinterleave3_epi64(tw0, tw1, tw2);

                            let b0 = _m256s_mul_complex(rk1, xw0);
                            let c0 = _m256s_mul_complex(rk2, xw1);
                            let d0 = _m256s_mul_complex(rk3, xw2);

                            // radix-4 butterfly
                            let q0t0 = _mm256_add_ps(a0, c0);
                            let q0t1 = _mm256_sub_ps(a0, c0);
                            let q0t2 = _mm256_add_ps(b0, d0);
                            let mut q0t3 = _mm256_sub_ps(b0, d0);
                            const SH: i32 = shuffle(2, 3, 0, 1);
                            q0t3 =
                                _mm256_xor_ps(_mm256_shuffle_ps::<SH>(q0t3, q0t3), v_i_multiplier);

                            let y0 = _mm256_add_ps(q0t0, q0t2);
                            let y1 = _mm256_add_ps(q0t1, q0t3);
                            let y2 = _mm256_sub_ps(q0t0, q0t2);
                            let y3 = _mm256_sub_ps(q0t1, q0t3);

                            _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 2 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                y2,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 3 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                y3,
                            );

                            j += 4;
                        }

                        while j + 2 < quarter {
                            let a0 = _mm_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());
                            let tw1 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast());
                            let tw2 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(3 * j + 4..).as_ptr().cast());

                            let rk1 =
                                _mm_loadu_ps(data.get_unchecked(j + quarter..).as_ptr().cast());
                            let rk2 =
                                _mm_loadu_ps(data.get_unchecked(j + 2 * quarter..).as_ptr().cast());
                            let rk3 =
                                _mm_loadu_ps(data.get_unchecked(j + 3 * quarter..).as_ptr().cast());

                            let (xw0, xw1, xw2) = _mm128s_deinterleave3_epi64(tw0, tw1, tw2);

                            let b0c0 = _m256s_mul_complex(
                                _mm256_create_ps(rk1, rk2),
                                _mm256_create_ps(xw0, xw1),
                            );
                            let d0 = _m128s_fma_mul_complex(rk3, xw2);
                            let b0 = _mm256_castps256_ps128(b0c0);
                            let c0 = _mm256_extractf128_ps::<1>(b0c0);

                            // radix-4 butterfly
                            let q0t0 = _mm_add_ps(a0, c0);
                            let q0t1 = _mm_sub_ps(a0, c0);
                            let q0t2 = _mm_add_ps(b0, d0);
                            let mut q0t3 = _mm_sub_ps(b0, d0);
                            const SH: i32 = shuffle(2, 3, 0, 1);
                            q0t3 = _mm_xor_ps(
                                _mm_shuffle_ps::<SH>(q0t3, q0t3),
                                _mm256_castps256_ps128(v_i_multiplier),
                            );

                            let y0 = _mm_add_ps(q0t0, q0t2);
                            let y1 = _mm_add_ps(q0t1, q0t3);
                            let y2 = _mm_sub_ps(q0t0, q0t2);
                            let y3 = _mm_sub_ps(q0t1, q0t3);

                            _mm_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                y1,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 2 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                y2,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 3 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                y3,
                            );

                            j += 2;
                        }

                        for j in j..quarter {
                            let a = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());

                            let bc = _m128s_fma_mul_complex(
                                _mm_unpacklo_ps64(
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + quarter..).as_ptr().cast(),
                                    ),
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 2 * quarter..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let d = _m128s_fma_mul_complex(
                                _m128s_load_f32x2(
                                    data.get_unchecked(j + 3 * quarter..).as_ptr().cast(),
                                ),
                                _m128s_load_f32x2(
                                    m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast(),
                                ),
                            );

                            let b = bc;
                            let c = _mm_unpackhi_ps64(bc, bc);

                            // radix-4 butterfly
                            let t0 = _mm_add_ps(a, c);
                            let t1 = _mm_sub_ps(a, c);
                            let t2 = _mm_add_ps(b, d);
                            let mut t3 = _mm_sub_ps(b, d);
                            const SH: i32 = shuffle(2, 3, 0, 1);
                            t3 = _mm_xor_ps(
                                _mm_shuffle_ps::<SH>(t3, t3),
                                _mm256_castps256_ps128(v_i_multiplier),
                            );

                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                _mm_add_ps(t0, t2),
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                _mm_add_ps(t1, t3),
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 2 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                _mm_sub_ps(t0, t2),
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 3 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                _mm_sub_ps(t1, t3),
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[quarter * 3..];
                    len *= 4;
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix4<f32> {
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
    use rand::Rng;

    #[test]
    fn test_neon_radix4() {
        for i in 1..7 {
            let size = 4usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxFmaRadix4::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = AvxFmaRadix4::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();
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
    fn test_neon_radix4_f64() {
        for i in 1..7 {
            let size = 4usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxFmaRadix4::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = AvxFmaRadix4::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();
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
