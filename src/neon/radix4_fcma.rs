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
use crate::factory::AlgorithmFactory;
use crate::neon::util::fcma_complex_f32;
use crate::neon::util::fcma_complex_f64;
use crate::neon::util::fcmah_complex_f32;
use crate::radix4::Radix4Twiddles;
use crate::util::bitreversed_transpose;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaRadix4<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    direction: FftDirection,
    base_len: usize,
    base_fft: Box<dyn FftExecutor<T> + Send + Sync>,
}

impl<T: Default + Clone + Radix4Twiddles + AlgorithmFactory<T>> NeonFcmaRadix4<T> {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonFcmaRadix4<T>, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");
        // assert_eq!(size.trailing_zeros() % 2, 0, "Radix-4 requires power of 4");

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

        let twiddles = T::make_twiddles(base_fft.length(), size, fft_direction)?;

        Ok(NeonFcmaRadix4 {
            execution_length: size,
            twiddles,
            direction: fft_direction,
            base_len: base_fft.length(),
            base_fft,
        })
    }
}

impl NeonFcmaRadix4<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

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

                    if self.direction == FftDirection::Inverse {
                        for data in chunk.chunks_exact_mut(len) {
                            for j in 0..quarter {
                                let a = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                                let b = fcma_complex_f64(
                                    vld1q_f64(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                    vld1q_f64(m_twiddles.get_unchecked(3 * j..).as_ptr().cast()),
                                );
                                let c = fcma_complex_f64(
                                    vld1q_f64(
                                        data.get_unchecked(j + 2 * quarter..).as_ptr().cast(),
                                    ),
                                    vld1q_f64(
                                        m_twiddles.get_unchecked(3 * j + 1..).as_ptr().cast(),
                                    ),
                                );
                                let d = fcma_complex_f64(
                                    vld1q_f64(
                                        data.get_unchecked(j + 3 * quarter..).as_ptr().cast(),
                                    ),
                                    vld1q_f64(
                                        m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast(),
                                    ),
                                );

                                // radix-4 butterfly
                                let t0 = vaddq_f64(a, c);
                                let t1 = vsubq_f64(a, c);
                                let t2 = vaddq_f64(b, d);
                                let t3 = vsubq_f64(b, d);

                                vst1q_f64(
                                    data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                    vaddq_f64(t0, t2),
                                );
                                vst1q_f64(
                                    data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                    vcaddq_rot90_f64(t1, t3),
                                );
                                vst1q_f64(
                                    data.get_unchecked_mut(j + 2 * quarter..)
                                        .as_mut_ptr()
                                        .cast(),
                                    vsubq_f64(t0, t2),
                                );
                                vst1q_f64(
                                    data.get_unchecked_mut(j + 3 * quarter..)
                                        .as_mut_ptr()
                                        .cast(),
                                    vcaddq_rot270_f64(t1, t3),
                                );
                            }
                        }
                    } else {
                        for data in chunk.chunks_exact_mut(len) {
                            for j in 0..quarter {
                                let a = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                                let b = fcma_complex_f64(
                                    vld1q_f64(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                    vld1q_f64(m_twiddles.get_unchecked(3 * j..).as_ptr().cast()),
                                );
                                let c = fcma_complex_f64(
                                    vld1q_f64(
                                        data.get_unchecked(j + 2 * quarter..).as_ptr().cast(),
                                    ),
                                    vld1q_f64(
                                        m_twiddles.get_unchecked(3 * j + 1..).as_ptr().cast(),
                                    ),
                                );
                                let d = fcma_complex_f64(
                                    vld1q_f64(
                                        data.get_unchecked(j + 3 * quarter..).as_ptr().cast(),
                                    ),
                                    vld1q_f64(
                                        m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast(),
                                    ),
                                );

                                // radix-4 butterfly
                                let t0 = vaddq_f64(a, c);
                                let t1 = vsubq_f64(a, c);
                                let t2 = vaddq_f64(b, d);
                                let t3 = vsubq_f64(b, d);

                                vst1q_f64(
                                    data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                    vaddq_f64(t0, t2),
                                );
                                vst1q_f64(
                                    data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                    vcaddq_rot270_f64(t1, t3),
                                );
                                vst1q_f64(
                                    data.get_unchecked_mut(j + 2 * quarter..)
                                        .as_mut_ptr()
                                        .cast(),
                                    vsubq_f64(t0, t2),
                                );
                                vst1q_f64(
                                    data.get_unchecked_mut(j + 3 * quarter..)
                                        .as_mut_ptr()
                                        .cast(),
                                    vcaddq_rot90_f64(t1, t3),
                                );
                            }
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 3..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for NeonFcmaRadix4<f64> {
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

impl NeonFcmaRadix4<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let v_i_multiplier = unsafe {
            vreinterpretq_u32_f32(match self.direction {
                FftDirection::Inverse => vld1q_f32([-0.0, 0.0, -0.0, 0.0].as_ptr()),
                FftDirection::Forward => vld1q_f32([0.0, -0.0, 0.0, -0.0].as_ptr()),
            })
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

                        while j + 2 < quarter {
                            let a = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 = vld1q_f32(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());
                            let tw1 =
                                vld1q_f32(m_twiddles.get_unchecked(3 * (j + 1)..).as_ptr().cast());

                            let b = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                vcombine_f32(vget_low_f32(tw0), vget_low_f32(tw1)),
                            );
                            let c = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 2 * quarter..).as_ptr().cast()),
                                vcombine_f32(vget_high_f32(tw0), vget_high_f32(tw1)),
                            );
                            let d = fcma_complex_f32(
                                vld1q_f32(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                                vcombine_f32(
                                    vld1_f32(m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast()),
                                    vld1_f32(
                                        m_twiddles.get_unchecked(3 * (j + 1) + 2..).as_ptr().cast(),
                                    ),
                                ),
                            );

                            // radix-4 butterfly
                            let t0 = vaddq_f32(a, c);
                            let t1 = vsubq_f32(a, c);
                            let t2 = vaddq_f32(b, d);
                            let mut t3 = vsubq_f32(b, d);
                            t3 = vreinterpretq_f32_u32(veorq_u32(
                                vrev64q_u32(vreinterpretq_u32_f32(t3)),
                                v_i_multiplier,
                            ));

                            vst1q_f32(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                vaddq_f32(t0, t2),
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                vaddq_f32(t1, t3),
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 2 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                vsubq_f32(t0, t2),
                            );
                            vst1q_f32(
                                data.get_unchecked_mut(j + 3 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                vsubq_f32(t1, t3),
                            );

                            j += 2;
                        }

                        for j in j..quarter {
                            let a = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                            let tw = vld1q_f32(m_twiddles.get_unchecked(3 * j..).as_ptr().cast());

                            let bc = fcma_complex_f32(
                                vcombine_f32(
                                    vld1_f32(data.get_unchecked(j + quarter..).as_ptr().cast()),
                                    vld1_f32(data.get_unchecked(j + 2 * quarter..).as_ptr().cast()),
                                ),
                                tw,
                            );
                            let d = fcmah_complex_f32(
                                vld1_f32(data.get_unchecked(j + 3 * quarter..).as_ptr().cast()),
                                vld1_f32(m_twiddles.get_unchecked(3 * j + 2..).as_ptr().cast()),
                            );

                            let b = vget_low_f32(bc);
                            let c = vget_high_f32(bc);

                            // radix-4 butterfly
                            let t0 = vadd_f32(a, c);
                            let t1 = vsub_f32(a, c);
                            let t2 = vadd_f32(b, d);
                            let mut t3 = vsub_f32(b, d);
                            t3 = vreinterpret_f32_u32(veor_u32(
                                vrev64_u32(vreinterpret_u32_f32(t3)),
                                vget_low_u32(v_i_multiplier),
                            ));

                            vst1_f32(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                vadd_f32(t0, t2),
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + quarter..).as_mut_ptr().cast(),
                                vadd_f32(t1, t3),
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 2 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                vsub_f32(t0, t2),
                            );
                            vst1_f32(
                                data.get_unchecked_mut(j + 3 * quarter..)
                                    .as_mut_ptr()
                                    .cast(),
                                vsub_f32(t1, t3),
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

impl FftExecutor<f32> for NeonFcmaRadix4<f32> {
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
    fn test_neon_fcma_radix4() {
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
            let radix_forward = NeonFcmaRadix4::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix4::new(size, FftDirection::Inverse).unwrap();
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
    fn test_neon_fcma_radix4_f64() {
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
            let radix_forward = NeonFcmaRadix4::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix4::new(size, FftDirection::Inverse).unwrap();
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
