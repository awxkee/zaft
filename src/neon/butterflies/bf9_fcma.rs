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
use crate::neon::util::{fcma_complex_f32, fcma_complex_f64};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaButterfly9<T> {
    direction: FftDirection,
    twiddle1: [T; 4],
    twiddle2: [T; 4],
    twiddle4: [T; 4],
    twiddle3_re: T,
    twiddle3_im: [T; 4],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonFcmaButterfly9<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle3 = compute_twiddle::<T>(1, 3, fft_direction);
        let tw1 = compute_twiddle(1, 9, fft_direction);
        let tw2 = compute_twiddle(2, 9, fft_direction);
        let tw4 = compute_twiddle(4, 9, fft_direction);
        Self {
            direction: fft_direction,
            twiddle3_re: twiddle3.re,
            twiddle3_im: [-twiddle3.im, twiddle3.im, -twiddle3.im, twiddle3.im],
            twiddle1: [tw1.re, tw1.im, tw1.re, tw1.im],
            twiddle2: [tw2.re, tw2.im, tw2.re, tw2.im],
            twiddle4: [tw4.re, tw4.im, tw4.re, tw4.im],
        }
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly9<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        9
    }
}

impl NeonFcmaButterfly9<f64> {
    #[target_feature(enable = "fcma")]
    fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 9 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let tw3_re = vdupq_n_f64(self.twiddle3_re);
            let tw3_im = vld1q_f64(self.twiddle3_im.as_ptr().cast());
            let tw1 = vld1q_f64(self.twiddle1.as_ptr().cast());
            let tw2 = vld1q_f64(self.twiddle2.as_ptr().cast());
            let tw4 = vld1q_f64(self.twiddle4.as_ptr().cast());

            for chunk in in_place.chunks_exact_mut(9) {
                let u0 = vld1q_f64(chunk.as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                let u5 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                let u6 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                let u7 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());
                let u8 = vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast());

                let (u0, u3, u6) = NeonButterfly::butterfly3_f64(u0, u3, u6, tw3_re, tw3_im);
                let (u1, mut u4, mut u7) =
                    NeonButterfly::butterfly3_f64(u1, u4, u7, tw3_re, tw3_im);
                let (u2, mut u5, mut u8) =
                    NeonButterfly::butterfly3_f64(u2, u5, u8, tw3_re, tw3_im);

                u4 = fcma_complex_f64(u4, tw1);
                u7 = fcma_complex_f64(u7, tw2);
                u5 = fcma_complex_f64(u5, tw2);
                u8 = fcma_complex_f64(u8, tw4);

                let (y0, y3, y6) = NeonButterfly::butterfly3_f64(u0, u1, u2, tw3_re, tw3_im);
                let (y1, y4, y7) = NeonButterfly::butterfly3_f64(u3, u4, u5, tw3_re, tw3_im);
                let (y2, y5, y8) = NeonButterfly::butterfly3_f64(u6, u7, u8, tw3_re, tw3_im);

                vst1q_f64(chunk.as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3);
                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), y5);
                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y6);
                vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y7);
                vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y8);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for NeonFcmaButterfly9<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        9
    }
}

impl NeonFcmaButterfly9<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 9 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let tw3_re = vdupq_n_f32(self.twiddle3_re);
            let tw3_im = vld1q_f32(self.twiddle3_im.as_ptr().cast());
            let tw1 = vld1q_f32(self.twiddle1.as_ptr().cast());
            let tw2 = vld1q_f32(self.twiddle2.as_ptr().cast());
            let tw4 = vld1q_f32(self.twiddle4.as_ptr().cast());

            for chunk in in_place.chunks_exact_mut(18) {
                let u0u1 = vld1q_f32(chunk.as_mut_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());
                let u16u17 = vld1q_f32(chunk.get_unchecked(16..).as_ptr().cast());

                let u0 = vcombine_f32(vget_low_f32(u0u1), vget_high_f32(u8u9));
                let u1 = vcombine_f32(vget_high_f32(u0u1), vget_low_f32(u10u11));
                let u2 = vcombine_f32(vget_low_f32(u2u3), vget_high_f32(u10u11));
                let u3 = vcombine_f32(vget_high_f32(u2u3), vget_low_f32(u12u13));
                let u4 = vcombine_f32(vget_low_f32(u4u5), vget_high_f32(u12u13));
                let u5 = vcombine_f32(vget_high_f32(u4u5), vget_low_f32(u14u15));
                let u6 = vcombine_f32(vget_low_f32(u6u7), vget_high_f32(u14u15));
                let u7 = vcombine_f32(vget_high_f32(u6u7), vget_low_f32(u16u17));
                let u8 = vcombine_f32(vget_low_f32(u8u9), vget_high_f32(u16u17));

                let (u0, u3, u6) = NeonButterfly::butterfly3_f32(u0, u3, u6, tw3_re, tw3_im);
                let (u1, mut u4, mut u7) =
                    NeonButterfly::butterfly3_f32(u1, u4, u7, tw3_re, tw3_im);
                let (u2, mut u5, mut u8) =
                    NeonButterfly::butterfly3_f32(u2, u5, u8, tw3_re, tw3_im);

                u4 = fcma_complex_f32(u4, tw1);
                u7 = fcma_complex_f32(u7, tw2);
                u5 = fcma_complex_f32(u5, tw2);
                u8 = fcma_complex_f32(u8, tw4);

                let (y0, y3, y6) = NeonButterfly::butterfly3_f32(u0, u1, u2, tw3_re, tw3_im);
                let (y1, y4, y7) = NeonButterfly::butterfly3_f32(u3, u4, u5, tw3_re, tw3_im);
                let (y2, y5, y8) = NeonButterfly::butterfly3_f32(u6, u7, u8, tw3_re, tw3_im);

                let qy0 = vcombine_f32(vget_low_f32(y0), vget_low_f32(y1));
                let qy1 = vcombine_f32(vget_low_f32(y2), vget_low_f32(y3));
                let qy2 = vcombine_f32(vget_low_f32(y4), vget_low_f32(y5));
                let qy3 = vcombine_f32(vget_low_f32(y6), vget_low_f32(y7));
                let qy4 = vcombine_f32(vget_low_f32(y8), vget_high_f32(y0));
                let qy5 = vcombine_f32(vget_high_f32(y1), vget_high_f32(y2));
                let qy6 = vcombine_f32(vget_high_f32(y3), vget_high_f32(y4));
                let qy7 = vcombine_f32(vget_high_f32(y5), vget_high_f32(y6));
                let qy8 = vcombine_f32(vget_high_f32(y7), vget_high_f32(y8));

                vst1q_f32(chunk.as_mut_ptr().cast(), qy0);
                vst1q_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), qy1);
                vst1q_f32(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), qy2);
                vst1q_f32(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), qy3);
                vst1q_f32(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), qy4);
                vst1q_f32(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), qy5);
                vst1q_f32(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), qy6);
                vst1q_f32(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), qy7);
                vst1q_f32(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), qy8);
            }

            let rem = in_place.chunks_exact_mut(18).into_remainder();

            for chunk in rem.chunks_exact_mut(9) {
                let u0u1 = vld1q_f32(chunk.as_mut_ptr().cast());
                let u2u3 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());
                let u8 = vld1_f32(chunk.get_unchecked(8..).as_ptr().cast());

                let u0 = vget_low_f32(u0u1);
                let u1 = vget_high_f32(u0u1);
                let u2 = vget_low_f32(u2u3);
                let u3 = vget_high_f32(u2u3);
                let u4 = vget_low_f32(u4u5);
                let u5 = vget_high_f32(u4u5);
                let u6 = vget_low_f32(u6u7);
                let u7 = vget_high_f32(u6u7);

                let (u0, u3, u6) = NeonButterfly::butterfly3h_f32(
                    u0,
                    u3,
                    u6,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_im),
                );
                let (u1, mut u4, mut u7) = NeonButterfly::butterfly3h_f32(
                    u1,
                    u4,
                    u7,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_im),
                );
                let (u2, mut u5, mut u8) = NeonButterfly::butterfly3h_f32(
                    u2,
                    u5,
                    u8,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_im),
                );

                let hu4u7 = fcma_complex_f32(
                    vcombine_f32(u4, u7),
                    vcombine_f32(vget_low_f32(tw1), vget_low_f32(tw2)),
                );
                let hu5u8 = fcma_complex_f32(
                    vcombine_f32(u5, u8),
                    vcombine_f32(vget_low_f32(tw2), vget_low_f32(tw4)),
                );
                u4 = vget_low_f32(hu4u7);
                u7 = vget_high_f32(hu4u7);
                u5 = vget_low_f32(hu5u8);
                u8 = vget_high_f32(hu5u8);

                let (y0, y3, y6) = NeonButterfly::butterfly3h_f32(
                    u0,
                    u1,
                    u2,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_im),
                );
                let (y1, y4, y7) = NeonButterfly::butterfly3h_f32(
                    u3,
                    u4,
                    u5,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_im),
                );
                let (y2, y5, y8) = NeonButterfly::butterfly3h_f32(
                    u6,
                    u7,
                    u8,
                    vget_low_f32(tw3_re),
                    vget_low_f32(tw3_im),
                );

                vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(y0, y1));
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y2, y3),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    vcombine_f32(y4, y5),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    vcombine_f32(y6, y7),
                );
                vst1_f32(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y8);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Butterfly9;
    use rand::Rng;

    #[test]
    fn test_fcma_butterfly9_f32() {
        for i in 1..5 {
            let size = 9usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix5_reference = Butterfly9::new(FftDirection::Forward);
            let radix5_inv_reference = Butterfly9::new(FftDirection::Inverse);

            let radix_forward = NeonFcmaButterfly9::new(FftDirection::Forward);
            let radix_inverse = NeonFcmaButterfly9::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix5_reference.execute(&mut z_ref).unwrap();

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
            radix5_inv_reference.execute(&mut z_ref).unwrap();

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

            input = input.iter().map(|&x| x * (1.0 / 9f32)).collect();

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
    fn test_fcma_butterfly9_f64() {
        for i in 1..5 {
            let size = 9usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();

            let mut z_ref = input.to_vec();

            let radix9_reference = Butterfly9::new(FftDirection::Forward);

            let radix_forward = NeonFcmaButterfly9::new(FftDirection::Forward);
            let radix_inverse = NeonFcmaButterfly9::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix9_reference.execute(&mut z_ref).unwrap();

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

            input = input.iter().map(|&x| x * (1.0 / 9f64)).collect();

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
