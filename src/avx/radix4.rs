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
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_fcmul_pd, _mm_fcmul_ps, _mm_unpackhi_ps64,
    _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps, _mm256_fcmul_pd, _mm256_fcmul_ps,
    shuffle,
};
use crate::factory::AlgorithmFactory;
use crate::radix4::Radix4Twiddles;
use crate::traits::FftTrigonometry;
use crate::util::{bitreversed_transpose, compute_twiddle};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::any::TypeId;
use std::arch::x86_64::*;

pub(crate) struct AvxFmaRadix4<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    direction: FftDirection,
    base_len: usize,
    base_fft: Box<dyn FftExecutor<T> + Send + Sync>,
}

impl<T: Default + Clone + Radix4Twiddles + AlgorithmFactory<T> + FftTrigonometry + Float + 'static>
    AvxFmaRadix4<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix4<T>, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");

        let exponent = size.trailing_zeros();
        let base_fft = match exponent {
            0 => T::butterfly1(fft_direction)?,
            1 => T::butterfly2(fft_direction)?,
            2 => T::butterfly4(fft_direction)?,
            3 => T::butterfly8(fft_direction)?,
            _ => {
                if exponent % 2 == 1 {
                    T::butterfly8(fft_direction)?
                } else {
                    T::butterfly16(fft_direction)?
                }
            }
        };

        let mut twiddles = Vec::new();
        twiddles
            .try_reserve_exact(size - 1)
            .map_err(|_| ZaftError::OutOfMemory(size - 1))?;

        const N: usize = 4;
        let mut cross_fft_len = base_fft.length();
        while cross_fft_len < size {
            let num_columns = cross_fft_len;
            cross_fft_len *= N;

            let mut i = 0usize;

            if TypeId::of::<T>() == TypeId::of::<f32>() {
                while i + 4 < num_columns {
                    for k in 1..N {
                        let twiddle0 = compute_twiddle(i * k, cross_fft_len, fft_direction);
                        let twiddle1 = compute_twiddle((i + 1) * k, cross_fft_len, fft_direction);
                        let twiddle2 = compute_twiddle((i + 2) * k, cross_fft_len, fft_direction);
                        let twiddle3 = compute_twiddle((i + 3) * k, cross_fft_len, fft_direction);
                        twiddles.push(twiddle0);
                        twiddles.push(twiddle1);
                        twiddles.push(twiddle2);
                        twiddles.push(twiddle3);
                    }
                    i += 4;
                }
            }

            while i + 2 < num_columns {
                for k in 1..N {
                    let twiddle0 = compute_twiddle(i * k, cross_fft_len, fft_direction);
                    let twiddle1 = compute_twiddle((i + 1) * k, cross_fft_len, fft_direction);
                    twiddles.push(twiddle0);
                    twiddles.push(twiddle1);
                }
                i += 2;
            }

            for i in i..num_columns {
                for k in 1..N {
                    let twiddle = compute_twiddle(i * k, cross_fft_len, fft_direction);
                    twiddles.push(twiddle);
                }
            }
        }

        Ok(AvxFmaRadix4 {
            execution_length: size,
            twiddles,
            direction: fft_direction,
            base_len: base_fft.length(),
            base_fft,
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

        let mut scratch = vec![Complex::default(); self.execution_length];

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            scratch.copy_from_slice(chunk);
            // bit reversal first
            bitreversed_transpose::<Complex<f64>, 4>(self.base_len, &scratch, chunk);

            self.base_fft.execute(chunk)?;

            let mut len = self.base_len;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 4;
                    let quarter = len / 4;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;
                        while j + 2 < quarter {
                            let a = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(3 * j + 4..).as_ptr().cast(),
                            );

                            let b = _mm256_fcmul_pd(
                                _mm256_loadu_pd(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                tw0,
                            );
                            let c = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 2 * quarter..).as_ptr().cast(),
                                ),
                                tw1,
                            );
                            let d = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 3 * quarter..).as_ptr().cast(),
                                ),
                                tw2,
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

                            let bc = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 2 * quarter..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let d = _mm_fcmul_pd(
                                _mm_loadu_pd(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                                _mm_loadu_pd(m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast()),
                            );

                            // radix-4 butterfly
                            let b = _mm256_castpd256_pd128(bc);
                            let c = _mm256_extractf128_pd::<1>(bc);
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

                    m_twiddles = &m_twiddles[columns * 3..];
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

        let mut scratch = vec![Complex::default(); self.execution_length];

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            scratch.copy_from_slice(chunk);
            // bit reversal first
            bitreversed_transpose::<Complex<f32>, 4>(self.base_len, &scratch, chunk);

            self.base_fft.execute(chunk)?;

            let mut len = self.base_len;

            unsafe {
                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 4;
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

                            let b0 = _mm256_fcmul_ps(rk1, tw0);
                            let c0 = _mm256_fcmul_ps(rk2, tw1);
                            let d0 = _mm256_fcmul_ps(rk3, tw2);

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

                            let tw0tw1 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());
                            let tw2 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(3 * j + 4..).as_ptr().cast());

                            let rk1 =
                                _mm_loadu_ps(data.get_unchecked(j + quarter..).as_ptr().cast());
                            let rk2 =
                                _mm_loadu_ps(data.get_unchecked(j + 2 * quarter..).as_ptr().cast());
                            let rk3 =
                                _mm_loadu_ps(data.get_unchecked(j + 3 * quarter..).as_ptr().cast());

                            let b0c0 = _mm256_fcmul_ps(_mm256_create_ps(rk1, rk2), tw0tw1);
                            let d0 = _mm_fcmul_ps(rk3, tw2);
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

                            let bc = _mm_fcmul_ps(
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
                            let d = _mm_fcmul_ps(
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

                    m_twiddles = &m_twiddles[columns * 3..];
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
        for i in 1..5 {
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
        for i in 1..5 {
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
