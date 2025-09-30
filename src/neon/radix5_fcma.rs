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
use crate::neon::util::{
    fcma_complex_f32, fcma_complex_f64, fcmah_complex_f32, v_rotate90_f32, v_rotate90_f64,
    vh_rotate90_f32,
};
use crate::radix5::Radix5Twiddles;
use crate::traits::FftTrigonometry;
use crate::util::{compute_twiddle, digit_reverse_indices, is_power_of_five, permute_inplace};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaRadix5<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    direction: FftDirection,
}

impl<T: Default + Clone + Radix5Twiddles + 'static + Copy + FftTrigonometry + Float>
    NeonFcmaRadix5<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<NeonFcmaRadix5<T>, ZaftError> {
        assert!(is_power_of_five(size), "Input length must be a power of 5");

        let twiddles = T::make_twiddles(size, fft_direction)?;
        let rev = digit_reverse_indices(size, 5)?;

        Ok(NeonFcmaRadix5 {
            permutations: rev,
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 5, fft_direction),
            twiddle2: compute_twiddle(2, 5, fft_direction),
            direction: fft_direction,
        })
    }
}

impl NeonFcmaRadix5<f64> {
    #[target_feature(enable = "fcma")]
    fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        // Digit-reversal permutation
        permute_inplace(in_place, &self.permutations);

        let mut len = 5;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            let tw1_re = vdupq_n_f64(self.twiddle1.re);
            let tw1_im = vdupq_n_f64(self.twiddle1.im);
            let tw2_re = vdupq_n_f64(self.twiddle2.re);
            let tw2_im = vdupq_n_f64(self.twiddle2.im);
            let rot_sign = vld1q_f64([-0.0, 0.0].as_ptr());

            while len <= self.execution_length {
                let fifth = len / 5;

                for data in in_place.chunks_exact_mut(len) {
                    for j in 0..fifth {
                        let u0 = vld1q_f64(data.get_unchecked(j..).as_ptr().cast());
                        let u1 = fcma_complex_f64(
                            vld1q_f64(data.get_unchecked(j + fifth..).as_ptr().cast()),
                            vld1q_f64(m_twiddles.get_unchecked(4 * j..).as_ptr().cast()),
                        );
                        let u2 = fcma_complex_f64(
                            vld1q_f64(data.get_unchecked(j + 2 * fifth..).as_ptr().cast()),
                            vld1q_f64(m_twiddles.get_unchecked(4 * j + 1..).as_ptr().cast()),
                        );
                        let u3 = fcma_complex_f64(
                            vld1q_f64(data.get_unchecked(j + 3 * fifth..).as_ptr().cast()),
                            vld1q_f64(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast()),
                        );
                        let u4 = fcma_complex_f64(
                            vld1q_f64(data.get_unchecked(j + 4 * fifth..).as_ptr().cast()),
                            vld1q_f64(m_twiddles.get_unchecked(4 * j + 3..).as_ptr().cast()),
                        );

                        // Radix-5 butterfly

                        let x14p = vaddq_f64(u1, u4);
                        let x14n = vsubq_f64(u1, u4);
                        let x23p = vaddq_f64(u2, u3);
                        let x23n = vsubq_f64(u2, u3);
                        let y0 = vaddq_f64(vaddq_f64(u0, x14p), x23p);

                        let temp_b1_1 = vmulq_f64(tw1_im, x14n);
                        let temp_b2_1 = vmulq_f64(tw2_im, x14n);

                        let temp_a1 = vfmaq_f64(vfmaq_f64(u0, tw1_re, x14p), tw2_re, x23p);
                        let temp_a2 = vfmaq_f64(vfmaq_f64(u0, tw2_re, x14p), tw1_re, x23p);

                        let temp_b1 = vfmaq_f64(temp_b1_1, tw2_im, x23n);
                        let temp_b2 = vfmsq_f64(temp_b2_1, tw1_im, x23n);

                        let temp_b1_rot = v_rotate90_f64(temp_b1, rot_sign);
                        let temp_b2_rot = v_rotate90_f64(temp_b2, rot_sign);

                        let y1 = vaddq_f64(temp_a1, temp_b1_rot);
                        let y2 = vaddq_f64(temp_a2, temp_b2_rot);
                        let y3 = vsubq_f64(temp_a2, temp_b2_rot);
                        let y4 = vsubq_f64(temp_a1, temp_b1_rot);

                        vst1q_f64(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        vst1q_f64(data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(), y1);
                        vst1q_f64(
                            data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                            y2,
                        );
                        vst1q_f64(
                            data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                            y3,
                        );
                        vst1q_f64(
                            data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                            y4,
                        );
                    }
                }

                m_twiddles = &m_twiddles[fifth * 4..];
                len *= 5;
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for NeonFcmaRadix5<f64> {
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

impl NeonFcmaRadix5<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        // Digit-reversal permutation
        permute_inplace(in_place, &self.permutations);

        let mut len = 5;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            let tw1_re = vdupq_n_f32(self.twiddle1.re);
            let tw1_im = vdupq_n_f32(self.twiddle1.im);
            let tw2_re = vdupq_n_f32(self.twiddle2.re);
            let tw2_im = vdupq_n_f32(self.twiddle2.im);
            let rot_sign = vld1q_f32([-0.0, 0.0, -0.0, 0.0].as_ptr());

            while len <= self.execution_length {
                let fifth = len / 5;

                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 2 < fifth {
                        let u0 = vld1q_f32(data.get_unchecked(j..).as_ptr().cast());

                        let tw0 = vld1q_f32(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                        let tw1 =
                            vld1q_f32(m_twiddles.get_unchecked(4 * (j + 1)..).as_ptr().cast());
                        let tw2 = vld1q_f32(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast());
                        let tw3 =
                            vld1q_f32(m_twiddles.get_unchecked(4 * (j + 1) + 2..).as_ptr().cast());

                        let u1 = fcma_complex_f32(
                            vld1q_f32(data.get_unchecked(j + fifth..).as_ptr().cast()),
                            vcombine_f32(vget_low_f32(tw0), vget_low_f32(tw1)),
                        );
                        let u2 = fcma_complex_f32(
                            vld1q_f32(data.get_unchecked(j + 2 * fifth..).as_ptr().cast()),
                            vcombine_f32(vget_high_f32(tw0), vget_high_f32(tw1)),
                        );
                        let u3 = fcma_complex_f32(
                            vld1q_f32(data.get_unchecked(j + 3 * fifth..).as_ptr().cast()),
                            vcombine_f32(vget_low_f32(tw2), vget_low_f32(tw3)),
                        );
                        let u4 = fcma_complex_f32(
                            vld1q_f32(data.get_unchecked(j + 4 * fifth..).as_ptr().cast()),
                            vcombine_f32(vget_high_f32(tw2), vget_high_f32(tw3)),
                        );

                        // Radix-5 butterfly

                        let x14p = vaddq_f32(u1, u4);
                        let x14n = vsubq_f32(u1, u4);
                        let x23p = vaddq_f32(u2, u3);
                        let x23n = vsubq_f32(u2, u3);
                        let y0 = vaddq_f32(vaddq_f32(u0, x14p), x23p);

                        let temp_b1_1 = vmulq_f32(tw1_im, x14n);
                        let temp_b2_1 = vmulq_f32(tw2_im, x14n);

                        let temp_a1 = vfmaq_f32(vfmaq_f32(u0, tw1_re, x14p), tw2_re, x23p);
                        let temp_a2 = vfmaq_f32(vfmaq_f32(u0, tw2_re, x14p), tw1_re, x23p);

                        let temp_b1 = vfmaq_f32(temp_b1_1, tw2_im, x23n);
                        let temp_b2 = vfmsq_f32(temp_b2_1, tw1_im, x23n);

                        let temp_b1_rot = v_rotate90_f32(temp_b1, rot_sign);
                        let temp_b2_rot = v_rotate90_f32(temp_b2, rot_sign);

                        let y1 = vaddq_f32(temp_a1, temp_b1_rot);
                        let y2 = vaddq_f32(temp_a2, temp_b2_rot);
                        let y3 = vsubq_f32(temp_a2, temp_b2_rot);
                        let y4 = vsubq_f32(temp_a1, temp_b1_rot);

                        vst1q_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        vst1q_f32(data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(), y1);
                        vst1q_f32(
                            data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                            y2,
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                            y3,
                        );
                        vst1q_f32(
                            data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                            y4,
                        );

                        j += 2;
                    }

                    for j in j..fifth {
                        let u0 = vld1_f32(data.get_unchecked(j..).as_ptr().cast());

                        let tw0 = vld1q_f32(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                        let tw1 = vld1q_f32(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast());

                        let u1 = fcmah_complex_f32(
                            vld1_f32(data.get_unchecked(j + fifth..).as_ptr().cast()),
                            vget_low_f32(tw0),
                        );
                        let u2 = fcmah_complex_f32(
                            vld1_f32(data.get_unchecked(j + 2 * fifth..).as_ptr().cast()),
                            vget_high_f32(tw0),
                        );
                        let u3 = fcmah_complex_f32(
                            vld1_f32(data.get_unchecked(j + 3 * fifth..).as_ptr().cast()),
                            vget_low_f32(tw1),
                        );
                        let u4 = fcmah_complex_f32(
                            vld1_f32(data.get_unchecked(j + 4 * fifth..).as_ptr().cast()),
                            vget_high_f32(tw1),
                        );

                        // Radix-5 butterfly

                        let x14p = vadd_f32(u1, u4);
                        let x14n = vsub_f32(u1, u4);
                        let x23p = vadd_f32(u2, u3);
                        let x23n = vsub_f32(u2, u3);
                        let y0 = vadd_f32(vadd_f32(u0, x14p), x23p);

                        let temp_b1_1 = vmul_f32(vget_low_f32(tw1_im), x14n);
                        let temp_b2_1 = vmul_f32(vget_low_f32(tw2_im), x14n);

                        let temp_a1 = vfma_f32(
                            vfma_f32(u0, vget_low_f32(tw1_re), x14p),
                            vget_low_f32(tw2_re),
                            x23p,
                        );
                        let temp_a2 = vfma_f32(
                            vfma_f32(u0, vget_low_f32(tw2_re), x14p),
                            vget_low_f32(tw1_re),
                            x23p,
                        );

                        let temp_b1 = vfma_f32(temp_b1_1, vget_low_f32(tw2_im), x23n);
                        let temp_b2 = vfms_f32(temp_b2_1, vget_low_f32(tw1_im), x23n);

                        let temp_b1_rot = vh_rotate90_f32(temp_b1, vget_low_f32(rot_sign));
                        let temp_b2_rot = vh_rotate90_f32(temp_b2, vget_low_f32(rot_sign));

                        let y1 = vadd_f32(temp_a1, temp_b1_rot);
                        let y2 = vadd_f32(temp_a2, temp_b2_rot);
                        let y3 = vsub_f32(temp_a2, temp_b2_rot);
                        let y4 = vsub_f32(temp_a1, temp_b1_rot);

                        vst1_f32(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        vst1_f32(data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(), y1);
                        vst1_f32(
                            data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                            y2,
                        );
                        vst1_f32(
                            data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                            y3,
                        );
                        vst1_f32(
                            data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                            y4,
                        );
                    }
                }

                m_twiddles = &m_twiddles[fifth * 4..];
                len *= 5;
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaRadix5<f32> {
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
    fn test_neon_fcma_radix5() {
        for i in 1..7 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaRadix5::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix5::new(size, FftDirection::Inverse).unwrap();
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
    fn test_neon_fcma_radix5_f64() {
        for i in 1..7 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaRadix5::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = NeonFcmaRadix5::new(size, FftDirection::Inverse).unwrap();
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
