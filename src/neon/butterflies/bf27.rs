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
use crate::neon::butterflies::fast_bf9d::NeonFastButterfly9d;
use crate::neon::butterflies::fast_bf9f::NeonFastButterfly9f;
use crate::neon::util::{mul_complex_f32, mul_complex_f64, vdup_complex_f32, vdup_complex_f64};
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) struct NeonButterfly27d {
    direction: FftDirection,
    bf9: NeonFastButterfly9d,
    twiddle1: float64x2_t,
    twiddle2: float64x2_t,
    twiddle3: float64x2_t,
    twiddle4: float64x2_t,
    twiddle5: float64x2_t,
    twiddle6: float64x2_t,
    twiddle7: float64x2_t,
    twiddle8: float64x2_t,
    twiddle9: float64x2_t,
    twiddle10: float64x2_t,
    twiddle11: float64x2_t,
    twiddle12: float64x2_t,
}

impl NeonButterfly27d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                direction: fft_direction,
                twiddle1: vdup_complex_f64(compute_twiddle(1, 27, fft_direction)),
                twiddle2: vdup_complex_f64(compute_twiddle(2, 27, fft_direction)),
                twiddle3: vdup_complex_f64(compute_twiddle(3, 27, fft_direction)),
                twiddle4: vdup_complex_f64(compute_twiddle(4, 27, fft_direction)),
                twiddle5: vdup_complex_f64(compute_twiddle(5, 27, fft_direction)),
                twiddle6: vdup_complex_f64(compute_twiddle(6, 27, fft_direction)),
                twiddle7: vdup_complex_f64(compute_twiddle(7, 27, fft_direction)),
                twiddle8: vdup_complex_f64(compute_twiddle(8, 27, fft_direction)),
                twiddle9: vdup_complex_f64(compute_twiddle(10, 27, fft_direction)),
                twiddle10: vdup_complex_f64(compute_twiddle(12, 27, fft_direction)),
                twiddle11: vdup_complex_f64(compute_twiddle(14, 27, fft_direction)),
                twiddle12: vdup_complex_f64(compute_twiddle(16, 27, fft_direction)),
                bf9: NeonFastButterfly9d::new(fft_direction),
            }
        }
    }
}

impl FftExecutor<f64> for NeonButterfly27d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 27 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(27) {
                let u0 = vld1q_f64(chunk.as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                let u6 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                let u7 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());
                let u8 = vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast());
                let u9 = vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast());
                let u10 = vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast());
                let u11 = vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast());
                let u12 = vld1q_f64(chunk.get_unchecked(12..).as_ptr().cast());
                let u13 = vld1q_f64(chunk.get_unchecked(13..).as_ptr().cast());
                let u14 = vld1q_f64(chunk.get_unchecked(14..).as_ptr().cast());
                let u15 = vld1q_f64(chunk.get_unchecked(15..).as_ptr().cast());
                let u16 = vld1q_f64(chunk.get_unchecked(16..).as_ptr().cast());
                let u17 = vld1q_f64(chunk.get_unchecked(17..).as_ptr().cast());
                let u18 = vld1q_f64(chunk.get_unchecked(18..).as_ptr().cast());
                let u19 = vld1q_f64(chunk.get_unchecked(19..).as_ptr().cast());
                let u20 = vld1q_f64(chunk.get_unchecked(20..).as_ptr().cast());
                let u21 = vld1q_f64(chunk.get_unchecked(21..).as_ptr().cast());
                let u22 = vld1q_f64(chunk.get_unchecked(22..).as_ptr().cast());
                let u23 = vld1q_f64(chunk.get_unchecked(23..).as_ptr().cast());
                let u24 = vld1q_f64(chunk.get_unchecked(24..).as_ptr().cast());
                let u25 = vld1q_f64(chunk.get_unchecked(25..).as_ptr().cast());
                let u26 = vld1q_f64(chunk.get_unchecked(26..).as_ptr().cast());

                let s0 = self.bf9.exec(u0, u3, u6, u9, u12, u15, u18, u21, u24);
                let mut s1 = self.bf9.exec(u1, u4, u7, u10, u13, u16, u19, u22, u25);
                let mut s2 = self.bf9.exec(u2, u5, u8, u11, u14, u17, u20, u23, u26);

                let z0 = self.bf9.bf3(s0.0, s1.0, s2.0);
                vst1q_f64(chunk.as_mut_ptr().cast(), z0.0);
                vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), z0.1);
                vst1q_f64(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), z0.2);

                s1.1 = mul_complex_f64(s1.1, self.twiddle1);
                s1.2 = mul_complex_f64(s1.2, self.twiddle2);
                s1.3 = mul_complex_f64(s1.3, self.twiddle3);
                s1.4 = mul_complex_f64(s1.4, self.twiddle4);
                s1.5 = mul_complex_f64(s1.5, self.twiddle5);
                s1.6 = mul_complex_f64(s1.6, self.twiddle6);
                s1.7 = mul_complex_f64(s1.7, self.twiddle7);
                s1.8 = mul_complex_f64(s1.8, self.twiddle8);
                s2.1 = mul_complex_f64(s2.1, self.twiddle2);
                s2.2 = mul_complex_f64(s2.2, self.twiddle4);
                s2.3 = mul_complex_f64(s2.3, self.twiddle6);
                s2.4 = mul_complex_f64(s2.4, self.twiddle8);
                s2.5 = mul_complex_f64(s2.5, self.twiddle9);
                s2.6 = mul_complex_f64(s2.6, self.twiddle10);
                s2.7 = mul_complex_f64(s2.7, self.twiddle11);
                s2.8 = mul_complex_f64(s2.8, self.twiddle12);

                let z1 = self.bf9.bf3(s0.1, s1.1, s2.1);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), z1.0);
                vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), z1.1);
                vst1q_f64(chunk.get_unchecked_mut(19..).as_mut_ptr().cast(), z1.2);
                let z2 = self.bf9.bf3(s0.2, s1.2, s2.2);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), z2.0);
                vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), z2.1);
                vst1q_f64(chunk.get_unchecked_mut(20..).as_mut_ptr().cast(), z2.2);
                let z3 = self.bf9.bf3(s0.3, s1.3, s2.3);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), z3.0);
                vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), z3.1);
                vst1q_f64(chunk.get_unchecked_mut(21..).as_mut_ptr().cast(), z3.2);
                let z4 = self.bf9.bf3(s0.4, s1.4, s2.4);
                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), z4.0);
                vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), z4.1);
                vst1q_f64(chunk.get_unchecked_mut(22..).as_mut_ptr().cast(), z4.2);
                let z5 = self.bf9.bf3(s0.5, s1.5, s2.5);
                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), z5.0);
                vst1q_f64(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), z5.1);
                vst1q_f64(chunk.get_unchecked_mut(23..).as_mut_ptr().cast(), z5.2);
                let z6 = self.bf9.bf3(s0.6, s1.6, s2.6);
                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), z6.0);
                vst1q_f64(chunk.get_unchecked_mut(15..).as_mut_ptr().cast(), z6.1);
                vst1q_f64(chunk.get_unchecked_mut(24..).as_mut_ptr().cast(), z6.2);
                let z7 = self.bf9.bf3(s0.7, s1.7, s2.7);
                vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), z7.0);
                vst1q_f64(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), z7.1);
                vst1q_f64(chunk.get_unchecked_mut(25..).as_mut_ptr().cast(), z7.2);
                let z8 = self.bf9.bf3(s0.8, s1.8, s2.8);
                vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), z8.0);
                vst1q_f64(chunk.get_unchecked_mut(17..).as_mut_ptr().cast(), z8.1);
                vst1q_f64(chunk.get_unchecked_mut(26..).as_mut_ptr().cast(), z8.2);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        27
    }
}

pub(crate) struct NeonButterfly27f {
    direction: FftDirection,
    bf9: NeonFastButterfly9f,
    twiddle1: float32x4_t,
    twiddle2: float32x4_t,
    twiddle3: float32x4_t,
    twiddle4: float32x4_t,
    twiddle5: float32x4_t,
    twiddle6: float32x4_t,
    twiddle7: float32x4_t,
    twiddle8: float32x4_t,
    twiddle9: float32x4_t,
    twiddle10: float32x4_t,
    twiddle11: float32x4_t,
    twiddle12: float32x4_t,
}

impl NeonButterfly27f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                direction: fft_direction,
                twiddle1: vdup_complex_f32(compute_twiddle(1, 27, fft_direction)),
                twiddle2: vdup_complex_f32(compute_twiddle(2, 27, fft_direction)),
                twiddle3: vdup_complex_f32(compute_twiddle(3, 27, fft_direction)),
                twiddle4: vdup_complex_f32(compute_twiddle(4, 27, fft_direction)),
                twiddle5: vdup_complex_f32(compute_twiddle(5, 27, fft_direction)),
                twiddle6: vdup_complex_f32(compute_twiddle(6, 27, fft_direction)),
                twiddle7: vdup_complex_f32(compute_twiddle(7, 27, fft_direction)),
                twiddle8: vdup_complex_f32(compute_twiddle(8, 27, fft_direction)),
                twiddle9: vdup_complex_f32(compute_twiddle(10, 27, fft_direction)),
                twiddle10: vdup_complex_f32(compute_twiddle(12, 27, fft_direction)),
                twiddle11: vdup_complex_f32(compute_twiddle(14, 27, fft_direction)),
                twiddle12: vdup_complex_f32(compute_twiddle(16, 27, fft_direction)),
                bf9: NeonFastButterfly9f::new(fft_direction),
            }
        }
    }
}

impl FftExecutor<f32> for NeonButterfly27f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 27 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let tw1 = vcombine_f32(vget_low_f32(self.twiddle1), vget_low_f32(self.twiddle2));
            let tw2 = vcombine_f32(vget_low_f32(self.twiddle3), vget_low_f32(self.twiddle4));
            let tw3 = vcombine_f32(vget_low_f32(self.twiddle5), vget_low_f32(self.twiddle6));
            let tw4 = vcombine_f32(vget_low_f32(self.twiddle7), vget_low_f32(self.twiddle8));
            let tw5 = vcombine_f32(vget_low_f32(self.twiddle2), vget_low_f32(self.twiddle4));
            let tw6 = vcombine_f32(vget_low_f32(self.twiddle6), vget_low_f32(self.twiddle8));
            let tw7 = vcombine_f32(vget_low_f32(self.twiddle9), vget_low_f32(self.twiddle10));
            let tw8 = vcombine_f32(vget_low_f32(self.twiddle11), vget_low_f32(self.twiddle12));

            for chunk in in_place.chunks_exact_mut(27) {
                let u0u1 = vld1q_f32(chunk.as_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());
                let u16u17 = vld1q_f32(chunk.get_unchecked(16..).as_ptr().cast());
                let u18u19 = vld1q_f32(chunk.get_unchecked(18..).as_ptr().cast());
                let u20u21 = vld1q_f32(chunk.get_unchecked(20..).as_ptr().cast());
                let u22u23 = vld1q_f32(chunk.get_unchecked(22..).as_ptr().cast());
                let u24u25 = vld1q_f32(chunk.get_unchecked(24..).as_ptr().cast());
                let u26 = vld1_f32(chunk.get_unchecked(26..).as_ptr().cast());

                let s0s1 = self.bf9.exec(
                    u0u1,
                    vcombine_f32(vget_high_f32(u2u3), vget_low_f32(u4u5)),
                    u6u7,
                    vcombine_f32(vget_high_f32(u8u9), vget_low_f32(u10u11)),
                    u12u13,
                    vcombine_f32(vget_high_f32(u14u15), vget_low_f32(u16u17)),
                    u18u19,
                    vcombine_f32(vget_high_f32(u20u21), vget_low_f32(u22u23)),
                    u24u25,
                );
                let s2 = self.bf9.exech(
                    vget_low_f32(u2u3),
                    vget_high_f32(u4u5),
                    vget_low_f32(u8u9),
                    vget_high_f32(u10u11),
                    vget_low_f32(u14u15),
                    vget_high_f32(u16u17),
                    vget_low_f32(u20u21),
                    vget_high_f32(u22u23),
                    u26,
                );

                let s1_1xs1_2 = mul_complex_f32(
                    vcombine_f32(vget_high_f32(s0s1.1), vget_high_f32(s0s1.2)),
                    tw1,
                );
                let s1_3xs1_4 = mul_complex_f32(
                    vcombine_f32(vget_high_f32(s0s1.3), vget_high_f32(s0s1.4)),
                    tw2,
                );
                let s1_5xs1_6 = mul_complex_f32(
                    vcombine_f32(vget_high_f32(s0s1.5), vget_high_f32(s0s1.6)),
                    tw3,
                );
                let s1_7xs1_8 = mul_complex_f32(
                    vcombine_f32(vget_high_f32(s0s1.7), vget_high_f32(s0s1.8)),
                    tw4,
                );
                let s2_1xs2_2 = mul_complex_f32(vcombine_f32(s2.1, s2.2), tw5);
                let s2_3xs2_4 = mul_complex_f32(vcombine_f32(s2.3, s2.4), tw6);
                let s2_5xs2_6 = mul_complex_f32(vcombine_f32(s2.5, s2.6), tw7);
                let s2_7xs2_8 = mul_complex_f32(vcombine_f32(s2.7, s2.8), tw8);

                let z0 = self
                    .bf9
                    .bf3h(vget_low_f32(s0s1.0), vget_high_f32(s0s1.0), s2.0);

                vst1_f32(chunk.as_mut_ptr().cast(), z0.0);
                vst1_f32(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), z0.1);
                vst1_f32(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), z0.2);

                let z1z2 = self.bf9.bf3(
                    vcombine_f32(vget_low_f32(s0s1.1), vget_low_f32(s0s1.2)),
                    s1_1xs1_2,
                    s2_1xs2_2,
                );
                vst1q_f32(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), z1z2.0);
                vst1q_f32(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), z1z2.1);
                vst1q_f32(chunk.get_unchecked_mut(19..).as_mut_ptr().cast(), z1z2.2);

                let z3z4 = self.bf9.bf3(
                    vcombine_f32(vget_low_f32(s0s1.3), vget_low_f32(s0s1.4)),
                    s1_3xs1_4,
                    s2_3xs2_4,
                );
                vst1q_f32(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), z3z4.0);
                vst1q_f32(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), z3z4.1);
                vst1q_f32(chunk.get_unchecked_mut(21..).as_mut_ptr().cast(), z3z4.2);
                let z5z6 = self.bf9.bf3(
                    vcombine_f32(vget_low_f32(s0s1.5), vget_low_f32(s0s1.6)),
                    s1_5xs1_6,
                    s2_5xs2_6,
                );

                vst1q_f32(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), z5z6.0);
                vst1q_f32(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), z5z6.1);
                vst1q_f32(chunk.get_unchecked_mut(23..).as_mut_ptr().cast(), z5z6.2);

                let z7z8 = self.bf9.bf3(
                    vcombine_f32(vget_low_f32(s0s1.7), vget_low_f32(s0s1.8)),
                    s1_7xs1_8,
                    s2_7xs2_8,
                );
                vst1q_f32(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), z7z8.0);
                vst1q_f32(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), z7z8.1);
                vst1q_f32(chunk.get_unchecked_mut(25..).as_mut_ptr().cast(), z7z8.2);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        27
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Butterfly27;
    use crate::dft::Dft;
    use rand::Rng;

    #[test]
    fn test_butterfly27_f32() {
        for i in 1..5 {
            let size = 27usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix23_reference = Dft::new(27, FftDirection::Forward).unwrap();
            let radix23_inv_reference = Dft::new(27, FftDirection::Inverse).unwrap();

            let radix_forward = NeonButterfly27f::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly27f::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix23_reference.execute(&mut z_ref).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-3,
                        "a_re {} != b_re {} for size {}, reference failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-3,
                        "a_im {} != b_im {} for size {}, reference failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();
            radix23_inv_reference.execute(&mut z_ref).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-3,
                        "a_re {} != b_re {} for size {}, reference inv failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-3,
                        "a_im {} != b_im {} for size {}, reference inv failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            input = input.iter().map(|&x| x * (1.0 / 27f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-3,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-3,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly27_f64() {
        for i in 1..5 {
            let size = 27usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();

            let mut z_ref = input.to_vec();

            let bf23_reference = Butterfly27::new(FftDirection::Forward);

            let radix_forward = NeonButterfly27d::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly27d::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            bf23_reference.execute(&mut z_ref).unwrap();

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

            input = input.iter().map(|&x| x * (1.0 / 27f64)).collect();

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
