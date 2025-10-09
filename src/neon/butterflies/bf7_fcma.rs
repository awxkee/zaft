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
use crate::neon::butterflies::NeonButterfly;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaButterfly7<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonFcmaButterfly7<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 7, fft_direction),
            twiddle2: compute_twiddle(2, 7, fft_direction),
            twiddle3: compute_twiddle(3, 7, fft_direction),
        }
    }
}

impl FftExecutor<f32> for NeonFcmaButterfly7<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        7
    }
}

impl NeonFcmaButterfly7<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 7 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(14) {
                let u0u1 = vld1q_f32(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());

                let u0 = vcombine_f32(vget_low_f32(u0u1), vget_high_f32(u6u7));
                let u1 = vcombine_f32(vget_high_f32(u0u1), vget_low_f32(u8u9));
                let u2 = vcombine_f32(vget_low_f32(u2u3), vget_high_f32(u8u9));
                let u3 = vcombine_f32(vget_high_f32(u2u3), vget_low_f32(u10u11));
                let u4 = vcombine_f32(vget_low_f32(u4u5), vget_high_f32(u10u11));
                let u5 = vcombine_f32(vget_high_f32(u4u5), vget_low_f32(u12u13));
                let u6 = vcombine_f32(vget_low_f32(u6u7), vget_high_f32(u12u13));

                // Radix-7 butterfly

                let (x1p6, x1m6) = NeonButterfly::butterfly2_f32(u1, u6);
                let x1m6 = vcaddq_rot90_f32(vdupq_n_f32(0.), x1m6);
                let y00 = vaddq_f32(u0, x1p6);
                let (x2p5, x2m5) = NeonButterfly::butterfly2_f32(u2, u5);
                let x2m5 = vcaddq_rot90_f32(vdupq_n_f32(0.), x2m5);
                let y00 = vaddq_f32(y00, x2p5);
                let (x3p4, x3m4) = NeonButterfly::butterfly2_f32(u3, u4);
                let x3m4 = vcaddq_rot90_f32(vdupq_n_f32(0.), x3m4);
                let y00 = vaddq_f32(y00, x3p4);

                let m0106a = vfmaq_n_f32(u0, x1p6, self.twiddle1.re);
                let m0106a = vfmaq_n_f32(m0106a, x2p5, self.twiddle2.re);
                let m0106a = vfmaq_n_f32(m0106a, x3p4, self.twiddle3.re);
                let m0106b = vmulq_n_f32(x1m6, self.twiddle1.im);
                let m0106b = vfmaq_n_f32(m0106b, x2m5, self.twiddle2.im);
                let m0106b = vfmaq_n_f32(m0106b, x3m4, self.twiddle3.im);
                let (y01, y06) = NeonButterfly::butterfly2_f32(m0106a, m0106b);

                let m0205a = vfmaq_n_f32(u0, x1p6, self.twiddle2.re);
                let m0205a = vfmaq_n_f32(m0205a, x2p5, self.twiddle3.re);
                let m0205a = vfmaq_n_f32(m0205a, x3p4, self.twiddle1.re);
                let m0205b = vmulq_n_f32(x1m6, self.twiddle2.im);
                let m0205b = vfmsq_n_f32(m0205b, x2m5, self.twiddle3.im);
                let m0205b = vfmsq_n_f32(m0205b, x3m4, self.twiddle1.im);
                let (y02, y05) = NeonButterfly::butterfly2_f32(m0205a, m0205b);

                let m0304a = vfmaq_n_f32(u0, x1p6, self.twiddle3.re);
                let m0304a = vfmaq_n_f32(m0304a, x2p5, self.twiddle1.re);
                let m0304a = vfmaq_n_f32(m0304a, x3p4, self.twiddle2.re);
                let m0304b = vmulq_n_f32(x1m6, self.twiddle3.im);
                let m0304b = vfmsq_n_f32(m0304b, x2m5, self.twiddle1.im);
                let m0304b = vfmaq_n_f32(m0304b, x3m4, self.twiddle2.im);
                let (y03, y04) = NeonButterfly::butterfly2_f32(m0304a, m0304b);

                let y0y1 = vreinterpretq_f32_f64(vtrn1q_f64(
                    vreinterpretq_f64_f32(y00),
                    vreinterpretq_f64_f32(y01),
                ));
                let y2y3 = vreinterpretq_f32_f64(vtrn1q_f64(
                    vreinterpretq_f64_f32(y02),
                    vreinterpretq_f64_f32(y03),
                ));
                let y4y5 = vreinterpretq_f32_f64(vtrn1q_f64(
                    vreinterpretq_f64_f32(y04),
                    vreinterpretq_f64_f32(y05),
                ));
                let y6y7 = vcombine_f32(vget_low_f32(y06), vget_high_f32(y00));
                let y8y9 = vreinterpretq_f32_f64(vtrn2q_f64(
                    vreinterpretq_f64_f32(y01),
                    vreinterpretq_f64_f32(y02),
                ));
                let y10y11 = vreinterpretq_f32_f64(vtrn2q_f64(
                    vreinterpretq_f64_f32(y03),
                    vreinterpretq_f64_f32(y04),
                ));
                let y12y13 = vreinterpretq_f32_f64(vtrn2q_f64(
                    vreinterpretq_f64_f32(y05),
                    vreinterpretq_f64_f32(y06),
                ));

                vst1q_f32(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                vst1q_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2y3);
                vst1q_f32(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4y5);
                vst1q_f32(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y6y7);
                vst1q_f32(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y8y9);
                vst1q_f32(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10y11);
                vst1q_f32(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y12y13);
            }

            let rem = in_place.chunks_exact_mut(14).into_remainder();

            for chunk in rem.chunks_exact_mut(7) {
                let uz0 = vld1q_f32(chunk.get_unchecked(0..).as_ptr().cast());
                let uz1 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let uz2 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let uz3 = vld1_f32(chunk.get_unchecked(6..).as_ptr().cast());

                let u0 = vget_low_f32(uz0);
                let u1 = vget_high_f32(uz0);
                let u2 = vget_low_f32(uz1);
                let u3 = vget_high_f32(uz1);
                let u4 = vget_low_f32(uz2);
                let u5 = vget_high_f32(uz2);
                let u6 = uz3;

                // Radix-7 butterfly

                let (x1p6, x1m6) = NeonButterfly::butterfly2h_f32(u1, u6);
                let x1m6 = vcadd_rot90_f32(vdup_n_f32(0.), x1m6);
                let y00 = vadd_f32(u0, x1p6);
                let (x2p5, x2m5) = NeonButterfly::butterfly2h_f32(u2, u5);
                let x2m5 = vcadd_rot90_f32(vdup_n_f32(0.), x2m5);
                let y00 = vadd_f32(y00, x2p5);
                let (x3p4, x3m4) = NeonButterfly::butterfly2h_f32(u3, u4);
                let x3m4 = vcadd_rot90_f32(vdup_n_f32(0.), x3m4);
                let y00 = vadd_f32(y00, x3p4);

                let m0106a = vfma_n_f32(u0, x1p6, self.twiddle1.re);
                let m0106a = vfma_n_f32(m0106a, x2p5, self.twiddle2.re);
                let m0106a = vfma_n_f32(m0106a, x3p4, self.twiddle3.re);
                let m0106b = vmul_n_f32(x1m6, self.twiddle1.im);
                let m0106b = vfma_n_f32(m0106b, x2m5, self.twiddle2.im);
                let m0106b = vfma_n_f32(m0106b, x3m4, self.twiddle3.im);
                let (y01, y06) = NeonButterfly::butterfly2h_f32(m0106a, m0106b);

                let m0205a = vfma_n_f32(u0, x1p6, self.twiddle2.re);
                let m0205a = vfma_n_f32(m0205a, x2p5, self.twiddle3.re);
                let m0205a = vfma_n_f32(m0205a, x3p4, self.twiddle1.re);
                let m0205b = vmul_n_f32(x1m6, self.twiddle2.im);
                let m0205b = vfms_n_f32(m0205b, x2m5, self.twiddle3.im);
                let m0205b = vfms_n_f32(m0205b, x3m4, self.twiddle1.im);
                let (y02, y05) = NeonButterfly::butterfly2h_f32(m0205a, m0205b);

                let m0304a = vfma_n_f32(u0, x1p6, self.twiddle3.re);
                let m0304a = vfma_n_f32(m0304a, x2p5, self.twiddle1.re);
                let m0304a = vfma_n_f32(m0304a, x3p4, self.twiddle2.re);
                let m0304b = vmul_n_f32(x1m6, self.twiddle3.im);
                let m0304b = vfms_n_f32(m0304b, x2m5, self.twiddle1.im);
                let m0304b = vfma_n_f32(m0304b, x3m4, self.twiddle2.im);
                let (y03, y04) = NeonButterfly::butterfly2h_f32(m0304a, m0304b);

                vst1q_f32(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(y00, y01),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y02, y03),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y04, y05),
                );
                vst1_f32(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y06);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly7<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        7
    }
}

impl NeonFcmaButterfly7<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 7 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(7) {
                let u0 = vld1q_f64(chunk.get_unchecked(0..).as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                let u6 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());

                // Radix-7 butterfly

                let (x1p6, x1m6) = NeonButterfly::butterfly2_f64(u1, u6);
                let x1m6 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m6);
                let y00 = vaddq_f64(u0, x1p6);
                let (x2p5, x2m5) = NeonButterfly::butterfly2_f64(u2, u5);
                let x2m5 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m5);
                let y00 = vaddq_f64(y00, x2p5);
                let (x3p4, x3m4) = NeonButterfly::butterfly2_f64(u3, u4);
                let x3m4 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m4);
                let y00 = vaddq_f64(y00, x3p4);

                let m0106a = vfmaq_n_f64(u0, x1p6, self.twiddle1.re);
                let m0106a = vfmaq_n_f64(m0106a, x2p5, self.twiddle2.re);
                let m0106a = vfmaq_n_f64(m0106a, x3p4, self.twiddle3.re);
                let m0106b = vmulq_n_f64(x1m6, self.twiddle1.im);
                let m0106b = vfmaq_n_f64(m0106b, x2m5, self.twiddle2.im);
                let m0106b = vfmaq_n_f64(m0106b, x3m4, self.twiddle3.im);
                let (y01, y06) = NeonButterfly::butterfly2_f64(m0106a, m0106b);

                let m0205a = vfmaq_n_f64(u0, x1p6, self.twiddle2.re);
                let m0205a = vfmaq_n_f64(m0205a, x2p5, self.twiddle3.re);
                let m0205a = vfmaq_n_f64(m0205a, x3p4, self.twiddle1.re);
                let m0205b = vmulq_n_f64(x1m6, self.twiddle2.im);
                let m0205b = vfmsq_n_f64(m0205b, x2m5, self.twiddle3.im);
                let m0205b = vfmsq_n_f64(m0205b, x3m4, self.twiddle1.im);
                let (y02, y05) = NeonButterfly::butterfly2_f64(m0205a, m0205b);

                let m0304a = vfmaq_n_f64(u0, x1p6, self.twiddle3.re);
                let m0304a = vfmaq_n_f64(m0304a, x2p5, self.twiddle1.re);
                let m0304a = vfmaq_n_f64(m0304a, x3p4, self.twiddle2.re);
                let m0304b = vmulq_n_f64(x1m6, self.twiddle3.im);
                let m0304b = vfmsq_n_f64(m0304b, x2m5, self.twiddle1.im);
                let m0304b = vfmaq_n_f64(m0304b, x3m4, self.twiddle2.im);
                let (y03, y04) = NeonButterfly::butterfly2_f64(m0304a, m0304b);

                vst1q_f64(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y00);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y01);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y02);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y03);
                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y04);
                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y05);
                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y06);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Butterfly7;
    use rand::Rng;

    #[test]
    fn test_butterfly7_f32() {
        for i in 1..6 {
            let size = 7usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix5_reference = Butterfly7::new(FftDirection::Forward);
            let radix5_inv_reference = Butterfly7::new(FftDirection::Inverse);

            let radix_forward = NeonFcmaButterfly7::new(FftDirection::Forward);
            let radix_inverse = NeonFcmaButterfly7::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix5_reference.execute(&mut z_ref).unwrap();

            input.iter().zip(z_ref.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}, reference failed",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}, reference failed",
                    a.im,
                    b.im,
                    size
                );
            });

            radix_inverse.execute(&mut input).unwrap();
            radix5_inv_reference.execute(&mut z_ref).unwrap();

            input.iter().zip(z_ref.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}, reference inv failed",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}, reference inv failed",
                    a.im,
                    b.im,
                    size
                );
            });

            input = input.iter().map(|&x| x * (1.0 / 7f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly5_f64() {
        for i in 1..6 {
            let size = 7usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonFcmaButterfly7::new(FftDirection::Forward);
            let radix_inverse = NeonFcmaButterfly7::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 7f64)).collect();

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
