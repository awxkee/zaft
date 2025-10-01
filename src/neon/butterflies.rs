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
use crate::neon::util::{v_rotate90_f64, v_transpose_complex_f32, vh_rotate90_f32};
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

pub(crate) struct NeonButterfly2<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
}

impl<T: Default + Clone + Radix6Twiddles + 'static + Copy + FftTrigonometry + Float>
    NeonButterfly2<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            phantom_data: PhantomData,
        }
    }
}

impl FftExecutor<f32> for NeonButterfly2<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 2 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(8) {
            unsafe {
                let zu0_0 = vld1q_f32(chunk.get_unchecked(0..).as_ptr().cast());
                let zu1_0 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());
                let zu2_0 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                let zu3_0 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());

                let (u0_0, u1_0) = v_transpose_complex_f32(zu0_0, zu1_0);
                let (u2_0, u3_0) = v_transpose_complex_f32(zu2_0, zu3_0);

                let zy0 = vaddq_f32(u0_0, u1_0);
                let zy1 = vsubq_f32(u0_0, u1_0);
                let zy2 = vaddq_f32(u2_0, u3_0);
                let zy3 = vsubq_f32(u2_0, u3_0);

                let (y0, y1) = v_transpose_complex_f32(zy0, zy1);
                let (y2, y3) = v_transpose_complex_f32(zy2, zy3);

                vst1q_f32(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y1);
                vst1q_f32(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y2);
                vst1q_f32(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y3);
            }
        }

        let rem = in_place.chunks_exact_mut(8).into_remainder();

        for chunk in rem.chunks_exact_mut(4) {
            unsafe {
                let zu0_0 = vld1q_f32(chunk.get_unchecked(0..).as_ptr().cast());
                let zu1_0 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());

                let (u0_0, u1_0) = v_transpose_complex_f32(zu0_0, zu1_0);

                let zy0 = vaddq_f32(u0_0, u1_0);
                let zy1 = vsubq_f32(u0_0, u1_0);

                let (y0, y1) = v_transpose_complex_f32(zy0, zy1);

                vst1q_f32(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y1);
            }
        }

        let rem = in_place.chunks_exact_mut(4).into_remainder();

        for chunk in rem.chunks_exact_mut(2) {
            unsafe {
                let u0_0 = vld1_f32(chunk.get_unchecked(0..).as_ptr().cast());
                let u1_0 = vld1_f32(chunk.get_unchecked(1..).as_ptr().cast());

                let y0 = vadd_f32(u0_0, u1_0);
                let y1 = vsub_f32(u0_0, u1_0);

                vst1_f32(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1_f32(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        2
    }
}

impl FftExecutor<f64> for NeonButterfly2<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 2 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(4) {
            unsafe {
                let u0_0 = vld1q_f64(chunk.get_unchecked(0..).as_ptr().cast());
                let u1_0 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u0_1 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u1_1 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());

                let y0 = vaddq_f64(u0_0, u1_0);
                let y1 = vsubq_f64(u0_0, u1_0);
                let y2 = vaddq_f64(u0_1, u1_1);
                let y3 = vsubq_f64(u0_1, u1_1);

                vst1q_f64(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3);
            }
        }

        let remainer = in_place.chunks_exact_mut(4).into_remainder();

        for chunk in remainer.chunks_exact_mut(2) {
            unsafe {
                let u0_0 = vld1q_f64(chunk.get_unchecked(0..).as_ptr().cast());
                let u1_0 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());

                let y0 = vaddq_f64(u0_0, u1_0);
                let y1 = vsubq_f64(u0_0, u1_0);

                vst1q_f64(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
            }
        }

        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        2
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
        if in_place.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), 3));
        }

        let twiddle_re = unsafe { vdupq_n_f64(self.twiddle.re) };

        for chunk in in_place.chunks_exact_mut(3) {
            unsafe {
                let tw1 = vld1q_f64(self.tw1.as_ptr());
                let tw2 = vld1q_f64(self.tw2.as_ptr());

                let u0 = vld1q_f64(chunk.get_unchecked(0..).as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());

                let xp = vaddq_f64(u1, u2);
                let xn = vsubq_f64(u1, u2);
                let sum = vaddq_f64(u0, xp);

                let w_1 = vfmaq_f64(u0, twiddle_re, xp);

                let xn_rot = vextq_f64::<1>(xn, xn);

                let y0 = sum;
                let y1 = vfmaq_f64(w_1, tw1, xn_rot);
                let y2 = vfmaq_f64(w_1, tw2, xn_rot);

                vst1q_f64(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
            }
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
        if in_place.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), 3));
        }
        let twiddle_re = unsafe { vdup_n_f32(self.twiddle.re) };

        for chunk in in_place.chunks_exact_mut(3) {
            unsafe {
                let tw1 = vld1_f32(self.tw1.as_ptr());
                let tw2 = vld1_f32(self.tw2.as_ptr());
                let uz0 = vld1q_f32(chunk.get_unchecked(0..).as_ptr().cast());

                let u0 = vget_low_f32(uz0);
                let u1 = vget_high_f32(uz0);
                let u2 = vld1_f32(chunk.get_unchecked(2..).as_ptr().cast());

                let xp = vadd_f32(u1, u2);
                let xn = vsub_f32(u1, u2);
                let sum = vadd_f32(u0, xp);

                let w_1 = vfma_f32(u0, twiddle_re, xp);

                let xn_rot = vext_f32::<1>(xn, xn);

                let y0 = sum;
                let y1 = vfma_f32(w_1, tw1, xn_rot);
                let y2 = vfma_f32(w_1, tw2, xn_rot);

                vst1q_f32(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(y0, y1),
                );
                vst1_f32(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
            }
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
                FftDirection::Inverse => [-0.0f64.as_(), 0.0.as_(), -0.0.as_(), 0.0.as_()],
                FftDirection::Forward => [0.0f64.as_(), -0.0.as_(), 0.0.as_(), -0.0.as_()],
            },
        }
    }
}

impl FftExecutor<f32> for NeonButterfly4<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let z_mul = unsafe { vld1_f32(self.multiplier.as_ptr()) };
        let v_i_multiplier = unsafe { vreinterpret_u32_f32(z_mul) };
        for chunk in in_place.chunks_exact_mut(4) {
            unsafe {
                let uz0 = vld1q_f32(chunk.get_unchecked(0..).as_ptr().cast());
                let uz1 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());

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
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(vadd_f32(t0, t2), vadd_f32(t1, t3)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(vsub_f32(t0, t2), vsub_f32(t1, t3)),
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
        4
    }
}

impl FftExecutor<f64> for NeonButterfly4<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }
        let v_i_multiplier = unsafe { vreinterpretq_u64_f64(vld1q_f64(self.multiplier.as_ptr())) };

        for chunk in in_place.chunks_exact_mut(4) {
            unsafe {
                let a = vld1q_f64(chunk.get_unchecked(0..).as_ptr().cast());
                let b = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let c = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let d = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());

                let t0 = vaddq_f64(a, c);
                let t1 = vsubq_f64(a, c);
                let t2 = vaddq_f64(b, d);
                let mut t3 = vsubq_f64(b, d);
                t3 = vreinterpretq_f64_u64(veorq_u64(
                    vreinterpretq_u64_f64(vextq_f64::<1>(t3, t3)),
                    v_i_multiplier,
                ));

                vst1q_f64(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vaddq_f64(t0, t2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vaddq_f64(t1, t3),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vsubq_f64(t0, t2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vsubq_f64(t1, t3),
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
        4
    }
}

pub(crate) struct NeonButterfly5<T> {
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    direction: FftDirection,
}

impl<T: Default + Clone + Radix6Twiddles + 'static + Copy + FftTrigonometry + Float>
    NeonButterfly5<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 5, fft_direction),
            twiddle2: compute_twiddle(2, 5, fft_direction),
        }
    }
}

impl FftExecutor<f32> for NeonButterfly5<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 5 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }
        let tw1_re = unsafe { vdupq_n_f32(self.twiddle1.re) };
        let tw1_im = unsafe { vdupq_n_f32(self.twiddle1.im) };
        let tw2_re = unsafe { vdupq_n_f32(self.twiddle2.re) };
        let tw2_im = unsafe { vdupq_n_f32(self.twiddle2.im) };
        let rot_sign = unsafe { vld1q_f32([-0.0, 0.0, -0.0, 0.0].as_ptr()) };

        for chunk in in_place.chunks_exact_mut(5) {
            unsafe {
                let uz0 = vld1q_f32(chunk.get_unchecked(0..).as_ptr().cast());
                let uz1 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());

                let u0 = vget_low_f32(uz0);
                let u1 = vget_high_f32(uz0);
                let u2 = vget_low_f32(uz1);
                let u3 = vget_high_f32(uz1);
                let u4 = vld1_f32(chunk.get_unchecked(4..).as_ptr().cast());

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

                vst1q_f32(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(y0, y1),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(y2, y3),
                );
                vst1_f32(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        5
    }
}

impl FftExecutor<f64> for NeonButterfly5<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 5 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }
        let tw1_re = unsafe { vdupq_n_f64(self.twiddle1.re) };
        let tw1_im = unsafe { vdupq_n_f64(self.twiddle1.im) };
        let tw2_re = unsafe { vdupq_n_f64(self.twiddle2.re) };
        let tw2_im = unsafe { vdupq_n_f64(self.twiddle2.im) };
        let rot_sign = unsafe { vld1q_f64([-0.0, 0.0].as_ptr()) };

        for chunk in in_place.chunks_exact_mut(5) {
            unsafe {
                let u0 = vld1q_f64(chunk.get_unchecked(0..).as_ptr().cast());
                let u1 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u2 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u3 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());

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

                vst1q_f64(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3);
                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        5
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::butterflies::{Butterfly4, Butterfly5};
    use rand::Rng;

    #[test]
    fn test_butterfly2_f32() {
        for i in 1..6 {
            let size = 2usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonButterfly2::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly2::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 2f32)).collect();

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
    fn test_butterfly2_f64() {
        for i in 1..6 {
            let size = 2usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonButterfly2::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly2::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 2f64)).collect();

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

    #[test]
    fn test_butterfly3_f32() {
        for i in 1..6 {
            let size = 3usize.pow(i);
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

            input = input.iter().map(|&x| x * (1.0 / 3f32)).collect();

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
    fn test_butterfly3_f64() {
        for i in 1..6 {
            let size = 3usize.pow(i);
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

            input = input.iter().map(|&x| x * (1.0 / 3f64)).collect();

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

    #[test]
    fn test_butterfly4_f32() {
        for i in 1..6 {
            let size = 4usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix_forward = NeonButterfly4::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly4::new(FftDirection::Inverse);

            let radix4_reference = Butterfly4::new(FftDirection::Forward);
            let radix4_inv_reference = Butterfly4::new(FftDirection::Inverse);

            radix_forward.execute(&mut input).unwrap();
            radix4_reference.execute(&mut z_ref).unwrap();

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
            radix4_inv_reference.execute(&mut z_ref).unwrap();

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

            input = input.iter().map(|&x| x * (1.0 / 4f32)).collect();

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
    fn test_butterfly4_f64() {
        for i in 1..6 {
            let size = 4usize.pow(i);
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

            input = input.iter().map(|&x| x * (1.0 / 4f64)).collect();

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

    #[test]
    fn test_butterfly5_f32() {
        for i in 1..6 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix5_reference = Butterfly5::new(FftDirection::Forward);
            let radix5_inv_reference = Butterfly5::new(FftDirection::Inverse);

            let radix_forward = NeonButterfly5::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly5::new(FftDirection::Inverse);
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

            input = input.iter().map(|&x| x * (1.0 / 5f32)).collect();

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
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonButterfly5::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly5::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 5f64)).collect();

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
