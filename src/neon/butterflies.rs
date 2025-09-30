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
use crate::radix6::Radix6Twiddles;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;
use std::marker::PhantomData;

pub(crate) struct NeonButterfly {}

impl NeonButterfly {
    #[inline]
    pub(crate) fn butterfly3h_f32(
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
        tw_re: float32x2_t,
        tw_w_2: float32x2_t,
    ) -> (float32x2_t, float32x2_t, float32x2_t) {
        unsafe {
            let xp = vadd_f32(u1, u2);
            let xn = vsub_f32(u1, u2);
            let sum = vadd_f32(u0, xp);

            let w_1 = vfma_f32(u0, tw_re, xp);
            let vw_2 = vmul_f32(tw_w_2, vext_f32::<1>(xn, xn));

            let y0 = sum;
            let y1 = vadd_f32(w_1, vw_2);
            let y2 = vsub_f32(w_1, vw_2);
            (y0, y1, y2)
        }
    }

    #[inline]
    pub(crate) fn butterfly3_f32(
        u0: float32x4_t,
        u1: float32x4_t,
        u2: float32x4_t,
        tw_re: float32x4_t,
        tw_w_2: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t) {
        unsafe {
            let xp = vaddq_f32(u1, u2);
            let xn = vsubq_f32(u1, u2);
            let sum = vaddq_f32(u0, xp);

            let w_1 = vfmaq_f32(u0, tw_re, xp);
            let vw_2 = vmulq_f32(tw_w_2, vrev64q_f32(xn));

            let y0 = sum;
            let y1 = vaddq_f32(w_1, vw_2);
            let y2 = vsubq_f32(w_1, vw_2);
            (y0, y1, y2)
        }
    }

    #[inline]
    pub(crate) fn butterfly2h_f32(u0: float32x2_t, u1: float32x2_t) -> (float32x2_t, float32x2_t) {
        unsafe {
            let t = vadd_f32(u0, u1);

            let y1 = vsub_f32(u0, u1);
            let y0 = t;
            (y0, y1)
        }
    }

    #[inline]
    pub(crate) fn butterfly2_f32(u0: float32x4_t, u1: float32x4_t) -> (float32x4_t, float32x4_t) {
        unsafe {
            let t = vaddq_f32(u0, u1);

            let y1 = vsubq_f32(u0, u1);
            let y0 = t;
            (y0, y1)
        }
    }

    #[inline]
    pub(crate) fn butterfly3_f64(
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        tw_re: float64x2_t,
        tw_w_2: float64x2_t,
    ) -> (float64x2_t, float64x2_t, float64x2_t) {
        unsafe {
            let xp = vaddq_f64(u1, u2);
            let xn = vsubq_f64(u1, u2);
            let sum = vaddq_f64(u0, xp);

            let w_1 = vfmaq_f64(u0, tw_re, xp);
            let vw_2 = vmulq_f64(tw_w_2, vextq_f64::<1>(xn, xn));

            let y0 = sum;
            let y1 = vaddq_f64(w_1, vw_2);
            let y2 = vsubq_f64(w_1, vw_2);
            (y0, y1, y2)
        }
    }

    #[inline]
    pub(crate) fn butterfly2_f64(u0: float64x2_t, u1: float64x2_t) -> (float64x2_t, float64x2_t) {
        unsafe {
            let t = vaddq_f64(u0, u1);

            let y1 = vsubq_f64(u0, u1);
            let y0 = t;
            (y0, y1)
        }
    }
}

pub(crate) struct NeonButterfly3<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
    twiddle: Complex<T>,
    tw1: [T; 4],
    tw2: [T; 4],
}

impl<T: Default + Clone + Radix6Twiddles + 'static + Copy + FftTrigonometry + Float>
    NeonButterfly3<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle = compute_twiddle(1, 3, fft_direction);
        Self {
            direction: fft_direction,
            phantom_data: PhantomData,
            twiddle,
            tw1: [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im],
            tw2: [twiddle.im, -twiddle.im, twiddle.im, -twiddle.im],
        }
    }
}

impl FftExecutor<f64> for NeonButterfly3<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() != 3 {
            return Err(ZaftError::InvalidInPlaceLength(3, in_place.len()));
        }

        unsafe {
            let twiddle_re = vdupq_n_f64(self.twiddle.re);
            let tw1 = vld1q_f64(self.tw1.as_ptr());
            let tw2 = vld1q_f64(self.tw2.as_ptr());

            let u0 = vld1q_f64(in_place.get_unchecked(0..).as_ptr().cast());
            let u1 = vld1q_f64(in_place.get_unchecked(1..).as_ptr().cast());
            let u2 = vld1q_f64(in_place.get_unchecked(2..).as_ptr().cast());

            let xp = vaddq_f64(u1, u2);
            let xn = vsubq_f64(u1, u2);
            let sum = vaddq_f64(u0, xp);

            let w_1 = vfmaq_f64(u0, twiddle_re, xp);

            let xn_rot = vextq_f64::<1>(xn, xn);

            let y0 = sum;
            let y1 = vfmaq_f64(w_1, tw1, xn_rot);
            let y2 = vfmaq_f64(w_1, tw2, xn_rot);

            vst1q_f64(in_place.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
            vst1q_f64(in_place.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
            vst1q_f64(in_place.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        3
    }
}

impl FftExecutor<f32> for NeonButterfly3<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() != 3 {
            return Err(ZaftError::InvalidInPlaceLength(3, in_place.len()));
        }

        unsafe {
            let twiddle_re = vdup_n_f32(self.twiddle.re);

            let tw1 = vld1_f32(self.tw1.as_ptr());
            let tw2 = vld1_f32(self.tw2.as_ptr());
            let uz0 = vld1q_f32(in_place.get_unchecked(0..).as_ptr().cast());

            let u0 = vget_low_f32(uz0);
            let u1 = vget_high_f32(uz0);
            let u2 = vld1_f32(in_place.get_unchecked(2..).as_ptr().cast());

            let xp = vadd_f32(u1, u2);
            let xn = vsub_f32(u1, u2);
            let sum = vadd_f32(u0, xp);

            let w_1 = vfma_f32(u0, twiddle_re, xp);

            let xn_rot = vext_f32::<1>(xn, xn);

            let y0 = sum;
            let y1 = vfma_f32(w_1, tw1, xn_rot);
            let y2 = vfma_f32(w_1, tw2, xn_rot);

            vst1q_f32(
                in_place.get_unchecked_mut(0..).as_mut_ptr().cast(),
                vcombine_f32(y0, y1),
            );
            vst1_f32(in_place.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        3
    }
}

pub(crate) struct NeonButterfly4<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
    multiplier: [T; 4],
}

impl<T: Default + Clone + Radix6Twiddles + 'static + Copy + FftTrigonometry + Float>
    NeonButterfly4<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            phantom_data: PhantomData,
            multiplier: match fft_direction {
                FftDirection::Forward => [-0.0.as_(), 0.0.as_(), -0.0.as_(), 0.0.as_()],
                FftDirection::Inverse => [0.0.as_(), -0.0.as_(), 0.0.as_(), -0.0.as_()],
            },
        }
    }
}

impl FftExecutor<f32> for NeonButterfly4<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() != 4 {
            return Err(ZaftError::InvalidInPlaceLength(
                self.length(),
                in_place.len(),
            ));
        }
        unsafe {
            let v_i_multiplier = vreinterpret_u32_f32(vld1_f32(self.multiplier.as_ptr()));

            let uz0 = vld1q_f32(in_place.get_unchecked(0..).as_ptr().cast());
            let uz1 = vld1q_f32(in_place.get_unchecked(2..).as_ptr().cast());

            let a = vget_low_f32(uz0);
            let b = vget_high_f32(uz0);
            let c = vget_low_f32(uz1);
            let d = vget_high_f32(uz1);

            let t0 = vadd_f32(a, c);
            let t1 = vsub_f32(a, c);
            let t2 = vadd_f32(b, d);
            let mut t3 = vsub_f32(b, d);
            t3 = vreinterpret_f32_u32(veor_u32(
                vreinterpret_u32_f32(vext_f32::<1>(t3, t3)),
                v_i_multiplier,
            ));

            vst1q_f32(
                in_place.get_unchecked_mut(0..).as_mut_ptr().cast(),
                vcombine_f32(vadd_f32(t0, t2), vadd_f32(t1, t3)),
            );
            vst1q_f32(
                in_place.get_unchecked_mut(2..).as_mut_ptr().cast(),
                vcombine_f32(vsub_f32(t0, t2), vsub_f32(t1, t3)),
            );
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        4
    }
}

impl FftExecutor<f64> for NeonButterfly4<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() != 4 {
            return Err(ZaftError::InvalidInPlaceLength(
                self.length(),
                in_place.len(),
            ));
        }
        unsafe {
            let v_i_multiplier = vreinterpretq_u64_f64(vld1q_f64(self.multiplier.as_ptr()));

            let a = vld1q_f64(in_place.get_unchecked(0..).as_ptr().cast());
            let b = vld1q_f64(in_place.get_unchecked(1..).as_ptr().cast());
            let c = vld1q_f64(in_place.get_unchecked(2..).as_ptr().cast());
            let d = vld1q_f64(in_place.get_unchecked(3..).as_ptr().cast());

            let t0 = vaddq_f64(a, c);
            let t1 = vsubq_f64(a, c);
            let t2 = vaddq_f64(b, d);
            let mut t3 = vsubq_f64(b, d);
            t3 = vreinterpretq_f64_u64(veorq_u64(
                vreinterpretq_u64_f64(vextq_f64::<1>(t3, t3)),
                v_i_multiplier,
            ));

            vst1q_f64(
                in_place.get_unchecked_mut(0..).as_mut_ptr().cast(),
                vaddq_f64(t0, t2),
            );
            vst1q_f64(
                in_place.get_unchecked_mut(1..).as_mut_ptr().cast(),
                vaddq_f64(t1, t3),
            );
            vst1q_f64(
                in_place.get_unchecked_mut(2..).as_mut_ptr().cast(),
                vsubq_f64(t0, t2),
            );
            vst1q_f64(
                in_place.get_unchecked_mut(3..).as_mut_ptr().cast(),
                vsubq_f64(t1, t3),
            );
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_butterfly3_f32() {
        let size = 3usize;
        let mut input = vec![Complex::<f32>::default(); size];
        for z in input.iter_mut() {
            *z = Complex {
                re: rand::rng().random(),
                im: rand::rng().random(),
            };
        }
        let src = input.to_vec();
        let radix_forward = NeonButterfly3::new(FftDirection::Forward);
        let radix_inverse = NeonButterfly3::new(FftDirection::Inverse);
        radix_forward.execute(&mut input).unwrap();
        radix_inverse.execute(&mut input).unwrap();

        input = input
            .iter()
            .map(|&x| x * (1.0 / input.len() as f32))
            .collect();

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

    #[test]
    fn test_butterfly3_f64() {
        let size = 3usize;
        let mut input = vec![Complex::<f64>::default(); size];
        for z in input.iter_mut() {
            *z = Complex {
                re: rand::rng().random(),
                im: rand::rng().random(),
            };
        }
        let src = input.to_vec();
        let radix_forward = NeonButterfly3::new(FftDirection::Forward);
        let radix_inverse = NeonButterfly3::new(FftDirection::Inverse);
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

    #[test]
    fn test_butterfly4_f32() {
        let size = 4usize;
        let mut input = vec![Complex::<f32>::default(); size];
        for z in input.iter_mut() {
            *z = Complex {
                re: rand::rng().random(),
                im: rand::rng().random(),
            };
        }
        let src = input.to_vec();
        let radix_forward = NeonButterfly4::new(FftDirection::Forward);
        let radix_inverse = NeonButterfly4::new(FftDirection::Inverse);
        radix_forward.execute(&mut input).unwrap();
        radix_inverse.execute(&mut input).unwrap();

        input = input
            .iter()
            .map(|&x| x * (1.0 / input.len() as f32))
            .collect();

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

    #[test]
    fn test_butterfly4_f64() {
        let size = 4usize;
        let mut input = vec![Complex::<f64>::default(); size];
        for z in input.iter_mut() {
            *z = Complex {
                re: rand::rng().random(),
                im: rand::rng().random(),
            };
        }
        let src = input.to_vec();
        let radix_forward = NeonButterfly4::new(FftDirection::Forward);
        let radix_inverse = NeonButterfly4::new(FftDirection::Inverse);
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
