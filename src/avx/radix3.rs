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
    _m128s_load_f32x2, _m128s_store_f32x2, _mm256_fcmul_ps, _mm_fcmul_pd, _mm_fcmul_ps,
    _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_fcmul_pd, _mm256_unpackhi_pd2,
    _mm256_unpacklo_pd2, _mm256s_deinterleave2_epi64, shuffle,
};
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::radix3::Radix3Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{bitreversed_transpose, compute_logarithm, compute_twiddle};
use crate::{FftDirection, FftExecutor, Zaft, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::x86_64::*;
use std::fmt::Display;

pub(crate) struct AvxFmaRadix3<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle_re: T,
    twiddle_im: [T; 8],
    direction: FftDirection,
    base_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    base_len: usize,
}

impl<
    T: Default
        + Clone
        + Radix3Twiddles
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
> AvxFmaRadix3<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix3<T>, ZaftError> {
        assert!(
            size.is_power_of_two() || size % 3 == 0,
            "Input length must be divisible by 3"
        );

        let exponent = compute_logarithm::<3>(size).unwrap_or_else(|| {
            panic!("Neon Fcma Radix3 length must be power of 3, but got {size}",)
        });

        let base_fft = match exponent {
            0 => Zaft::strategy(1, fft_direction)?,
            1 => Zaft::strategy(3, fft_direction)?,
            _ => Zaft::strategy(9, fft_direction)?,
        };

        let twiddles = T::make_twiddles_with_base(base_fft.length(), size, fft_direction)?;

        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);

        Ok(AvxFmaRadix3 {
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
            base_len: base_fft.length(),
            base_fft,
        })
    }
}

impl AvxFmaRadix3<f64> {
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
                bitreversed_transpose::<Complex<f64>, 3>(self.base_len, chunk, &mut scratch);

                self.base_fft.execute(&mut scratch)?;

                let mut len = self.base_len;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 3;
                    let third = len / 3;

                    for data in scratch.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < third {
                            let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(2 * j + 2..).as_ptr().cast(),
                            );

                            let rk1 =
                                _mm256_loadu_pd(data.get_unchecked(j + third..).as_ptr().cast());
                            let rk2 = _mm256_loadu_pd(
                                data.get_unchecked(j + 2 * third..).as_ptr().cast(),
                            );

                            let u1 = _mm256_fcmul_pd(rk1, _mm256_unpacklo_pd2(tw0, tw1));
                            let u2 = _mm256_fcmul_pd(rk2, _mm256_unpackhi_pd2(tw0, tw1));

                            // Radix-3 butterfly
                            let xp = _mm256_add_pd(u1, u2);
                            let xn = _mm256_sub_pd(u1, u2);
                            let sum = _mm256_add_pd(u0, xp);

                            let w_1 = _mm256_fmadd_pd(twiddle_re, xp, u0);
                            let xn_rot = _mm256_permute_pd::<0b0101>(xn);

                            let vy0 = sum;
                            let vy1 = _mm256_fmadd_pd(twiddle_w_2, xn_rot, w_1);
                            let vy2 = _mm256_fnmadd_pd(twiddle_w_2, xn_rot, w_1);

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
                            let u1 = _mm_fcmul_pd(
                                _mm_loadu_pd(data.get_unchecked(j + third..).as_ptr().cast()),
                                _mm_loadu_pd(m_twiddles.get_unchecked(2 * j..).as_ptr().cast()),
                            );
                            let u2 = _mm_fcmul_pd(
                                _mm_loadu_pd(data.get_unchecked(j + 2 * third..).as_ptr().cast()),
                                _mm_loadu_pd(m_twiddles.get_unchecked(2 * j + 1..).as_ptr().cast()),
                            );

                            // Radix-3 butterfly
                            let xp = _mm_add_pd(u1, u2);
                            let xn = _mm_sub_pd(u1, u2);
                            let sum = _mm_add_pd(u0, xp);

                            let w_1 = _mm_fmadd_pd(_mm256_castpd256_pd128(twiddle_re), xp, u0);
                            let xn_rot = _mm_shuffle_pd::<0b01>(xn, xn);

                            let vy0 = sum;
                            let vy1 =
                                _mm_fmadd_pd(_mm256_castpd256_pd128(twiddle_w_2), xn_rot, w_1);
                            let vy2 =
                                _mm_fnmadd_pd(_mm256_castpd256_pd128(twiddle_w_2), xn_rot, w_1);

                            _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), vy0);
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + third..).as_mut_ptr().cast(),
                                vy1,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 2 * third..).as_mut_ptr().cast(),
                                vy2,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 2..];
                }
                chunk.copy_from_slice(&scratch);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix3<f64> {
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

impl AvxFmaRadix3<f32> {
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
                bitreversed_transpose::<Complex<f32>, 3>(self.base_len, chunk, &mut scratch);

                self.base_fft.execute(&mut scratch)?;

                let mut len = self.base_len;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 3;
                    let third = len / 3;

                    for data in scratch.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 4 < third {
                            let u0 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(2 * (j + 2)..).as_ptr().cast(),
                            );

                            let rk1 =
                                _mm256_loadu_ps(data.get_unchecked(j + third..).as_ptr().cast());
                            let rk2 = _mm256_loadu_ps(
                                data.get_unchecked(j + 2 * third..).as_ptr().cast(),
                            );

                            let (xw0, xw1) = _mm256s_deinterleave2_epi64(tw0, tw1);

                            let u1 = _mm256_fcmul_ps(rk1, xw0);
                            let u2 = _mm256_fcmul_ps(rk2, xw1);

                            // Radix-3 butterfly
                            let xp_0 = _mm256_add_ps(u1, u2);
                            let xn_0 = _mm256_sub_ps(u1, u2);
                            let sum_0 = _mm256_add_ps(u0, xp_0);

                            const SH: i32 = shuffle(2, 3, 0, 1);

                            let vw_1_1 = _mm256_fmadd_ps(twiddle_re, xp_0, u0);
                            let xn_rot = _mm256_permute_ps::<SH>(xn_0);

                            let vy0 = sum_0;
                            let vy1 = _mm256_fmadd_ps(twiddle_w_2, xn_rot, vw_1_1);
                            let vy2 = _mm256_fnmadd_ps(twiddle_w_2, xn_rot, vw_1_1);

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

                        for j in j..third {
                            let u0 = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());

                            let tw =
                                _mm_loadu_ps(m_twiddles.get_unchecked(2 * j..).as_ptr().cast());

                            let rk1 =
                                _m128s_load_f32x2(data.get_unchecked(j + third..).as_ptr().cast());
                            let rk2 = _m128s_load_f32x2(
                                data.get_unchecked(j + 2 * third..).as_ptr().cast(),
                            );

                            let u1u2 = _mm_fcmul_ps(_mm_unpacklo_ps64(rk1, rk2), tw);

                            let u1 = u1u2;
                            let u2 = _mm_unpackhi_ps64(u1u2, u1u2);

                            // Radix-3 butterfly
                            let xp = _mm_add_ps(u1, u2);
                            let xn = _mm_sub_ps(u1, u2);
                            let sum = _mm_add_ps(u0, xp);

                            const SH: i32 = shuffle(2, 3, 0, 1);

                            let w_1 = _mm_fmadd_ps(_mm256_castps256_ps128(twiddle_re), xp, u0);
                            let xn_rot = _mm_shuffle_ps::<SH>(xn, xn);

                            let vy0 = sum;
                            let vy1 =
                                _mm_fmadd_ps(_mm256_castps256_ps128(twiddle_w_2), xn_rot, w_1);
                            let vy2 =
                                _mm_fnmadd_ps(_mm256_castps256_ps128(twiddle_w_2), xn_rot, w_1);

                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                vy0,
                            );
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

                    m_twiddles = &m_twiddles[columns * 2..];
                }
                chunk.copy_from_slice(&scratch);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix3<f32> {
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
    use crate::radix3::Radix3;
    use rand::Rng;

    #[test]
    fn test_avx_radix3() {
        for i in 1..5 {
            let size = 3usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix3_forward_ref = Radix3::new(size, FftDirection::Forward).unwrap();

            let radix_forward = AvxFmaRadix3::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = AvxFmaRadix3::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();
            radix3_forward_ref.execute(&mut z_ref).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-4,
                        "a_re {} != b_re {} for size {}, reference failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-4,
                        "a_im {} != b_im {} for size {}, reference failed at {idx}",
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

            input
                .iter()
                .zip(src.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-5,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-5,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });
        }
    }

    #[test]
    fn test_avx_radix3_f64() {
        for i in 1..5 {
            let size = 3usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxFmaRadix3::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = AvxFmaRadix3::new(size, FftDirection::Inverse).unwrap();
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
