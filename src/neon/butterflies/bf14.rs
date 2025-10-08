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
use crate::neon::butterflies::fast_bf7::NeonFastButterfly7;
use crate::neon::util::vqtrnq_f32;
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonButterfly14<T> {
    direction: FftDirection,
    bf7: NeonFastButterfly7<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonButterfly14<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf7: NeonFastButterfly7::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for NeonButterfly14<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 14 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            let rot_sign = vld1q_f64(ROT_90.as_ptr());

            for chunk in in_place.chunks_exact_mut(14) {
                let u0 = vld1q_f64(chunk.as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u5 = vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast());
                let u6 = vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast());
                let u7 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u8 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                let u9 = vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast());
                let u10 = vld1q_f64(chunk.get_unchecked(12..).as_ptr().cast());
                let u11 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                let u12 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                let u13 = vld1q_f64(chunk.get_unchecked(13..).as_ptr().cast());

                let (u0, u1) = NeonButterfly::butterfly2_f64(u0, u1);
                let (u2, u3) = NeonButterfly::butterfly2_f64(u2, u3);
                let (u4, u5) = NeonButterfly::butterfly2_f64(u4, u5);
                let (u6, u7) = NeonButterfly::butterfly2_f64(u6, u7);
                let (u8, u9) = NeonButterfly::butterfly2_f64(u8, u9);
                let (u10, u11) = NeonButterfly::butterfly2_f64(u10, u11);
                let (u12, u13) = NeonButterfly::butterfly2_f64(u12, u13);

                // Outer 7-point butterflies
                let (y0, y2, y4, y6, y8, y10, y12) =
                    self.bf7.exec(u0, u2, u4, u6, u8, u10, u12, rot_sign); // (v0, v1, v2, v3, v4, v5, v6)
                let (y7, y9, y11, y13, y1, y3, y5) =
                    self.bf7.exec(u1, u3, u5, u7, u9, u11, u13, rot_sign); // (v7, v8, v9, v10, v11, v12, v13)

                vst1q_f64(chunk.as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3);
                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y5);
                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y6);
                vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y7);
                vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y8);
                vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y9);
                vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10);
                vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), y11);
                vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y12);
                vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), y13);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        14
    }
}

impl FftExecutor<f32> for NeonButterfly14<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 14 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            static ROT_90: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            let rot_sign = vld1q_f32(ROT_90.as_ptr());

            for chunk in in_place.chunks_exact_mut(28) {
                let u0u7 = vld1q_f32(chunk.as_mut_ptr().cast());
                let u8u1 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u2u9 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u10u3 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u4u11 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u5 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u6u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());

                let u0u7_2 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());
                let u8u1_2 = vld1q_f32(chunk.get_unchecked(16..).as_ptr().cast());
                let u2u9_2 = vld1q_f32(chunk.get_unchecked(18..).as_ptr().cast());
                let u10u3_2 = vld1q_f32(chunk.get_unchecked(20..).as_ptr().cast());
                let u4u11_2 = vld1q_f32(chunk.get_unchecked(22..).as_ptr().cast());
                let u12u5_2 = vld1q_f32(chunk.get_unchecked(24..).as_ptr().cast());
                let u6u13_2 = vld1q_f32(chunk.get_unchecked(26..).as_ptr().cast());

                let (u0, u7) = vqtrnq_f32(u0u7, u0u7_2);
                let (u8, u1) = vqtrnq_f32(u8u1, u8u1_2);
                let (u2, u9) = vqtrnq_f32(u2u9, u2u9_2);
                let (u10, u3) = vqtrnq_f32(u10u3, u10u3_2);
                let (u4, u11) = vqtrnq_f32(u4u11, u4u11_2);
                let (u12, u5) = vqtrnq_f32(u12u5, u12u5_2);
                let (u6, u13) = vqtrnq_f32(u6u13, u6u13_2);

                let (u0, u1) = NeonButterfly::butterfly2_f32(u0, u1);
                let (u2, u3) = NeonButterfly::butterfly2_f32(u2, u3);
                let (u4, u5) = NeonButterfly::butterfly2_f32(u4, u5);
                let (u6, u7) = NeonButterfly::butterfly2_f32(u6, u7);
                let (u8, u9) = NeonButterfly::butterfly2_f32(u8, u9);
                let (u10, u11) = NeonButterfly::butterfly2_f32(u10, u11);
                let (u12, u13) = NeonButterfly::butterfly2_f32(u12, u13);

                // Outer 7-point butterflies
                let (y0, y2, y4, y6, y8, y10, y12) =
                    self.bf7.exec(u0, u2, u4, u6, u8, u10, u12, rot_sign); // (v0, v1, v2, v3, v4, v5, v6)
                let (y7, y9, y11, y13, y1, y3, y5) =
                    self.bf7.exec(u1, u3, u5, u7, u9, u11, u13, rot_sign); // (v7, v8, v9, v10, v11, v12, v13)

                let (q0, q7) = vqtrnq_f32(y0, y1);
                let (q1, q8) = vqtrnq_f32(y2, y3);
                let (q2, q9) = vqtrnq_f32(y4, y5);
                let (q3, q10) = vqtrnq_f32(y6, y7);
                let (q4, q11) = vqtrnq_f32(y8, y9);
                let (q5, q12) = vqtrnq_f32(y10, y11);
                let (q6, q13) = vqtrnq_f32(y12, y13);

                vst1q_f32(chunk.as_mut_ptr().cast(), q0);
                vst1q_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), q1);
                vst1q_f32(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), q2);
                vst1q_f32(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), q3);
                vst1q_f32(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), q4);
                vst1q_f32(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), q5);
                vst1q_f32(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), q6);
                vst1q_f32(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), q7);
                vst1q_f32(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), q8);
                vst1q_f32(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), q9);
                vst1q_f32(chunk.get_unchecked_mut(20..).as_mut_ptr().cast(), q10);
                vst1q_f32(chunk.get_unchecked_mut(22..).as_mut_ptr().cast(), q11);
                vst1q_f32(chunk.get_unchecked_mut(24..).as_mut_ptr().cast(), q12);
                vst1q_f32(chunk.get_unchecked_mut(26..).as_mut_ptr().cast(), q13);
            }

            let rem = in_place.chunks_exact_mut(28).into_remainder();

            for chunk in rem.chunks_exact_mut(14) {
                let u0u7 = vld1q_f32(chunk.as_mut_ptr().cast());
                let u8u1 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u2u9 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u10u3 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u4u11 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u5 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u6u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());

                let u0 = vget_low_f32(u0u7);
                let u1 = vget_high_f32(u8u1);
                let u2 = vget_low_f32(u2u9);
                let u3 = vget_high_f32(u10u3);
                let u4 = vget_low_f32(u4u11);
                let u5 = vget_high_f32(u12u5);
                let u6 = vget_low_f32(u6u13);
                let u7 = vget_high_f32(u0u7);
                let u8 = vget_low_f32(u8u1);
                let u9 = vget_high_f32(u2u9);
                let u10 = vget_low_f32(u10u3);
                let u11 = vget_high_f32(u4u11);
                let u12 = vget_low_f32(u12u5);
                let u13 = vget_high_f32(u6u13);

                let (u0, u1) = NeonButterfly::butterfly2h_f32(u0, u1);
                let (u2, u3) = NeonButterfly::butterfly2h_f32(u2, u3);
                let (u4, u5) = NeonButterfly::butterfly2h_f32(u4, u5);
                let (u6, u7) = NeonButterfly::butterfly2h_f32(u6, u7);
                let (u8, u9) = NeonButterfly::butterfly2h_f32(u8, u9);
                let (u10, u11) = NeonButterfly::butterfly2h_f32(u10, u11);
                let (u12, u13) = NeonButterfly::butterfly2h_f32(u12, u13);

                // Outer 7-point butterflies
                let (y0y7, y2y9, y4y11, y6y13, y8y1, y10y3, y12y5) = self.bf7.exec(
                    vcombine_f32(u0, u1),
                    vcombine_f32(u2, u3),
                    vcombine_f32(u4, u5),
                    vcombine_f32(u6, u7),
                    vcombine_f32(u8, u9),
                    vcombine_f32(u10, u11),
                    vcombine_f32(u12, u13),
                    rot_sign,
                ); // (v0, v1, v2, v3, v4, v5, v6)

                vst1q_f32(
                    chunk.as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y0y7), vget_high_f32(y8y1)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y2y9), vget_high_f32(y10y3)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y4y11), vget_high_f32(y12y5)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y6y13), vget_high_f32(y0y7)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y8y1), vget_high_f32(y2y9)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y10y3), vget_high_f32(y4y11)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    vcombine_f32(vget_low_f32(y12y5), vget_high_f32(y6y13)),
                );
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        14
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Butterfly14;
    use rand::Rng;

    #[test]
    fn test_butterfly14_f32() {
        for i in 1..5 {
            let size = 14usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix14_reference = Butterfly14::new(FftDirection::Forward);
            let radix14_inv_reference = Butterfly14::new(FftDirection::Inverse);

            let radix_forward = NeonButterfly14::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly14::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix14_reference.execute(&mut z_ref).unwrap();
            println!("forward {:?}", input);
            println!("reference {:?}", z_ref);

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-5,
                        "a_re {} != b_re {} for size {}, reference failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-5,
                        "a_im {} != b_im {} for size {}, reference failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();
            radix14_inv_reference.execute(&mut z_ref).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-5,
                        "a_re {} != b_re {} for size {}, reference inv failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-5,
                        "a_im {} != b_im {} for size {}, reference inv failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            input = input.iter().map(|&x| x * (1.0 / 14f32)).collect();

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
    fn test_butterfly14_f64() {
        for i in 1..5 {
            let size = 14usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();

            let mut z_ref = input.to_vec();

            let radix14_reference = Butterfly14::new(FftDirection::Forward);

            let radix_forward = NeonButterfly14::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly14::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix14_reference.execute(&mut z_ref).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {}, reference failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {}, reference failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 14f64)).collect();

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
