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
use crate::radix2::Radix2Twiddles;
use crate::util::{digit_reverse_indices, permute_inplace};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaRadix2<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    direction: FftDirection,
}

impl<T: Default + Clone + Radix2Twiddles> NeonFcmaRadix2<T> {
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonFcmaRadix2<T>, ZaftError> {
        assert!(size.is_power_of_two(), "Input length must be a power of 2");

        let twiddles = T::make_twiddles(size, fft_direction)?;

        // Bit-reversal permutation
        let rev = digit_reverse_indices(size, 2)?;

        Ok(NeonFcmaRadix2 {
            permutations: rev,
            execution_length: size,
            twiddles,
            direction: fft_direction,
        })
    }
}

impl NeonFcmaRadix2<f64> {
    #[target_feature(enable = "fcma")]
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
            let zero = vdupq_n_f64(0.);
            while len <= self.execution_length {
                let half = len / 2;
                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;
                    while j + 2 < half {
                        let u01 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                        let u02 = vld1q_f64(data.get_unchecked(j + 1..).as_ptr().cast());
                        let tw0 = vld1q_f64(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let tw1 = vld1q_f64(m_twiddles.get_unchecked(j + 1..).as_ptr().cast());
                        let u11 = vld1q_f64(data.get_unchecked(j + half..).as_ptr().cast());
                        let u12 = vld1q_f64(data.get_unchecked(j + half + 1..).as_ptr().cast());
                        let t0 = vcmlaq_rot90_f64(vcmlaq_f64(zero, tw0, u11), tw0, u11);
                        let t1 = vcmlaq_rot90_f64(vcmlaq_f64(zero, tw1, u12), tw1, u12);
                        let y0_1 = vaddq_f64(u01, t0);
                        let y1_1 = vsubq_f64(u01, t0);
                        let y0_2 = vaddq_f64(u02, t1);
                        let y1_2 = vsubq_f64(u02, t1);
                        vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0_1);
                        vst1q_f64(data.get_unchecked_mut(j + 1..).as_mut_ptr().cast(), y0_2);
                        vst1q_f64(data.get_unchecked_mut(j + half..).as_mut_ptr().cast(), y1_1);
                        vst1q_f64(
                            data.get_unchecked_mut(j + half + 1..).as_mut_ptr().cast(),
                            y1_2,
                        );
                        j += 2;
                    }
                    for j in j..half {
                        let u = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                        let tw = vld1q_f64(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let u1 = vld1q_f64(data.get_unchecked(j + half..).as_ptr().cast());
                        let t = vcmlaq_rot90_f64(vcmlaq_f64(zero, tw, u1), tw, u1);
                        let y0 = vaddq_f64(u, t);
                        let y1 = vsubq_f64(u, t);
                        vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        vst1q_f64(data.get_unchecked_mut(j + half..).as_mut_ptr().cast(), y1);
                    }
                }

                len *= 2;
                m_twiddles = &m_twiddles[half..];
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for NeonFcmaRadix2<f64> {
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

impl NeonFcmaRadix2<f32> {
    #[target_feature(enable = "fcma")]
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
            let zero = vdupq_n_f32(0.);
            while len <= self.execution_length {
                let half = len / 2;
                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 4 < half {
                        let u0_0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());
                        let u1_0 = vld1q_f32(data.get_unchecked(j + half..).as_ptr().cast());

                        let u0_1 = vld1q_f32(data.get_unchecked(j + 2..).as_ptr().cast());
                        let u1_1 = vld1q_f32(data.get_unchecked(j + half + 2..).as_ptr().cast());

                        let tw0 = vld1q_f32(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let tw1 = vld1q_f32(m_twiddles.get_unchecked(j + 2..).as_ptr().cast());

                        let t_0 = vcmlaq_rot90_f32(vcmlaq_f32(zero, tw0, u1_0), tw0, u1_0);
                        let t_1 = vcmlaq_rot90_f32(vcmlaq_f32(zero, tw1, u1_1), tw1, u1_1);

                        let y0 = vaddq_f32(u0_0, t_0);
                        let y1 = vsubq_f32(u0_0, t_0);
                        let y2 = vaddq_f32(u0_1, t_1);
                        let y3 = vsubq_f32(u0_1, t_1);

                        vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        vst1q_f32(data.get_unchecked_mut(j + half..).as_mut_ptr().cast(), y1);
                        vst1q_f32(data.get_unchecked_mut(j + 2..).as_mut_ptr().cast(), y2);
                        vst1q_f32(
                            data.get_unchecked_mut(j + half + 2..).as_mut_ptr().cast(),
                            y3,
                        );

                        j += 4;
                    }

                    while j + 2 < half {
                        let u = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());
                        let tw = vld1q_f32(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let u1 = vld1q_f32(data.get_unchecked(j + half..).as_ptr().cast());
                        let t = vcmlaq_rot90_f32(vcmlaq_f32(zero, tw, u1), tw, u1);
                        let y0 = vaddq_f32(u, t);
                        let y1 = vsubq_f32(u, t);
                        vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        vst1q_f32(data.get_unchecked_mut(j + half..).as_mut_ptr().cast(), y1);

                        j += 2;
                    }

                    for j in j..half {
                        let u = vld1_f32(data.get_unchecked(j..).as_ptr().cast());
                        let tw = vld1_f32(m_twiddles.get_unchecked(j..).as_ptr().cast());
                        let u1 = vld1_f32(data.get_unchecked(j + half..).as_ptr().cast());
                        let t = vcmla_rot90_f32(vcmla_f32(vdup_n_f32(0.), tw, u1), tw, u1);
                        let y0 = vadd_f32(u, t);
                        let y1 = vsub_f32(u, t);
                        vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        vst1_f32(data.get_unchecked_mut(j + half..).as_mut_ptr().cast(), y1);
                    }
                }

                len *= 2;
                m_twiddles = &m_twiddles[half..];
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaRadix2<f32> {
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
    fn test_neon_fcma_radix2() {
        for i in 1..14 {
            let size = 2usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaRadix2::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix2::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input
                .iter()
                .map(|&x| x * (1.0 / input.len() as f32))
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
    fn test_neon_radix2_f64() {
        for i in 1..14 {
            let size = 2usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaRadix2::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix2::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input
                .iter()
                .map(|&x| x * (1.0 / input.len() as f64))
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
