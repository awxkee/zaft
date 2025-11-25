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
use crate::traits::FftTrigonometry;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;
use std::marker::PhantomData;
use std::sync::Arc;

pub(crate) struct NeonFcmaButterfly4<T> {
    direction: FftDirection,
    phantom_data: PhantomData<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonFcmaButterfly4<T>
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

impl FftExecutor<f32> for NeonFcmaButterfly4<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            match self.direction {
                FftDirection::Forward => self.execute_forward(in_place),
                FftDirection::Inverse => self.execute_backward(in_place),
            }
        }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        4
    }
}

impl NeonFcmaButterfly4<f32> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_forward(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(16) {
            unsafe {
                let uzp0 = vld4q_f64(chunk.as_ptr().cast());
                let uzp1 = vld4q_f64(chunk.get_unchecked(8..).as_ptr().cast());

                let u0 = vreinterpretq_f32_f64(uzp0.0);
                let u1 = vreinterpretq_f32_f64(uzp0.1);
                let u2 = vreinterpretq_f32_f64(uzp0.2);
                let u3 = vreinterpretq_f32_f64(uzp0.3);

                let u4 = vreinterpretq_f32_f64(uzp1.0);
                let u5 = vreinterpretq_f32_f64(uzp1.1);
                let u6 = vreinterpretq_f32_f64(uzp1.2);
                let u7 = vreinterpretq_f32_f64(uzp1.3);

                let t0 = vaddq_f32(u0, u2);
                let t4 = vaddq_f32(u4, u6);
                let t1 = vsubq_f32(u0, u2);
                let t5 = vsubq_f32(u4, u6);
                let t2 = vaddq_f32(u1, u3);
                let t6 = vaddq_f32(u5, u7);
                let t3 = vsubq_f32(u1, u3);
                let t7 = vsubq_f32(u5, u7);

                let rw0 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot270_f32(t1, t3)),
                    vreinterpretq_f64_f32(vsubq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot90_f32(t1, t3)),
                );
                let rw1 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t4, t6)),
                    vreinterpretq_f64_f32(vcaddq_rot270_f32(t5, t7)),
                    vreinterpretq_f64_f32(vsubq_f32(t4, t6)),
                    vreinterpretq_f64_f32(vcaddq_rot90_f32(t5, t7)),
                );

                vst4q_f64(chunk.as_mut_ptr().cast(), rw0);
                vst4q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), rw1);
            }
        }

        let rem = in_place.chunks_exact_mut(16).into_remainder();

        for chunk in rem.chunks_exact_mut(8) {
            unsafe {
                let uzp = vld4q_f64(chunk.as_ptr().cast());

                let a = vreinterpretq_f32_f64(uzp.0);
                let b = vreinterpretq_f32_f64(uzp.1);
                let c = vreinterpretq_f32_f64(uzp.2);
                let d = vreinterpretq_f32_f64(uzp.3);

                let t0 = vaddq_f32(a, c);
                let t1 = vsubq_f32(a, c);
                let t2 = vaddq_f32(b, d);
                let t3 = vsubq_f32(b, d);

                let rw0 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot270_f32(t1, t3)),
                    vreinterpretq_f64_f32(vsubq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot90_f32(t1, t3)),
                );

                vst4q_f64(chunk.as_mut_ptr().cast(), rw0);
            }
        }

        let rem = rem.chunks_exact_mut(8).into_remainder();

        for chunk in rem.chunks_exact_mut(4) {
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
                let t3 = vsub_f32(b, d);

                vst1q_f32(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(vadd_f32(t0, t2), vcadd_rot270_f32(t1, t3)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(vsub_f32(t0, t2), vcadd_rot90_f32(t1, t3)),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_forward_out(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
            unsafe {
                let uzp0 = vld4q_f64(src.as_ptr().cast());
                let uzp1 = vld4q_f64(src.get_unchecked(8..).as_ptr().cast());

                let u0 = vreinterpretq_f32_f64(uzp0.0);
                let u1 = vreinterpretq_f32_f64(uzp0.1);
                let u2 = vreinterpretq_f32_f64(uzp0.2);
                let u3 = vreinterpretq_f32_f64(uzp0.3);

                let u4 = vreinterpretq_f32_f64(uzp1.0);
                let u5 = vreinterpretq_f32_f64(uzp1.1);
                let u6 = vreinterpretq_f32_f64(uzp1.2);
                let u7 = vreinterpretq_f32_f64(uzp1.3);

                let t0 = vaddq_f32(u0, u2);
                let t4 = vaddq_f32(u4, u6);
                let t1 = vsubq_f32(u0, u2);
                let t5 = vsubq_f32(u4, u6);
                let t2 = vaddq_f32(u1, u3);
                let t6 = vaddq_f32(u5, u7);
                let t3 = vsubq_f32(u1, u3);
                let t7 = vsubq_f32(u5, u7);

                let rw0 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot270_f32(t1, t3)),
                    vreinterpretq_f64_f32(vsubq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot90_f32(t1, t3)),
                );
                let rw1 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t4, t6)),
                    vreinterpretq_f64_f32(vcaddq_rot270_f32(t5, t7)),
                    vreinterpretq_f64_f32(vsubq_f32(t4, t6)),
                    vreinterpretq_f64_f32(vcaddq_rot90_f32(t5, t7)),
                );

                vst4q_f64(dst.as_mut_ptr().cast(), rw0);
                vst4q_f64(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), rw1);
            }
        }

        let rem_dst = dst.chunks_exact_mut(16).into_remainder();
        let rem_src = src.chunks_exact(16).remainder();

        for (dst, src) in rem_dst.chunks_exact_mut(8).zip(rem_src.chunks_exact(8)) {
            unsafe {
                let uzp = vld4q_f64(src.as_ptr().cast());

                let a = vreinterpretq_f32_f64(uzp.0);
                let b = vreinterpretq_f32_f64(uzp.1);
                let c = vreinterpretq_f32_f64(uzp.2);
                let d = vreinterpretq_f32_f64(uzp.3);

                let t0 = vaddq_f32(a, c);
                let t1 = vsubq_f32(a, c);
                let t2 = vaddq_f32(b, d);
                let t3 = vsubq_f32(b, d);

                let rw0 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot270_f32(t1, t3)),
                    vreinterpretq_f64_f32(vsubq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot90_f32(t1, t3)),
                );

                vst4q_f64(dst.as_mut_ptr().cast(), rw0);
            }
        }

        let rem_dst = rem_dst.chunks_exact_mut(16).into_remainder();
        let rem_src = rem_src.chunks_exact(16).remainder();

        for (dst, src) in rem_dst.chunks_exact_mut(4).zip(rem_src.chunks_exact(4)) {
            unsafe {
                let uz0 = vld1q_f32(src.get_unchecked(0..).as_ptr().cast());
                let uz1 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());

                let a = vget_low_f32(uz0);
                let b = vget_high_f32(uz0);
                let c = vget_low_f32(uz1);
                let d = vget_high_f32(uz1);

                let t0 = vadd_f32(a, c);
                let t1 = vsub_f32(a, c);
                let t2 = vadd_f32(b, d);
                let t3 = vsub_f32(b, d);

                vst1q_f32(
                    dst.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(vadd_f32(t0, t2), vcadd_rot270_f32(t1, t3)),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(vsub_f32(t0, t2), vcadd_rot90_f32(t1, t3)),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_backward(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(16) {
            unsafe {
                let uzp0 = vld4q_f64(chunk.as_ptr().cast());
                let uzp1 = vld4q_f64(chunk.get_unchecked(8..).as_ptr().cast());

                let u0 = vreinterpretq_f32_f64(uzp0.0);
                let u1 = vreinterpretq_f32_f64(uzp0.1);
                let u2 = vreinterpretq_f32_f64(uzp0.2);
                let u3 = vreinterpretq_f32_f64(uzp0.3);

                let u4 = vreinterpretq_f32_f64(uzp1.0);
                let u5 = vreinterpretq_f32_f64(uzp1.1);
                let u6 = vreinterpretq_f32_f64(uzp1.2);
                let u7 = vreinterpretq_f32_f64(uzp1.3);

                let t0 = vaddq_f32(u0, u2);
                let t4 = vaddq_f32(u4, u6);
                let t1 = vsubq_f32(u0, u2);
                let t5 = vsubq_f32(u4, u6);
                let t2 = vaddq_f32(u1, u3);
                let t6 = vaddq_f32(u5, u7);
                let t3 = vsubq_f32(u1, u3);
                let t7 = vsubq_f32(u5, u7);

                let rw0 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot90_f32(t1, t3)),
                    vreinterpretq_f64_f32(vsubq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot270_f32(t1, t3)),
                );
                let rw1 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t4, t6)),
                    vreinterpretq_f64_f32(vcaddq_rot90_f32(t5, t7)),
                    vreinterpretq_f64_f32(vsubq_f32(t4, t6)),
                    vreinterpretq_f64_f32(vcaddq_rot270_f32(t5, t7)),
                );

                vst4q_f64(chunk.as_mut_ptr().cast(), rw0);
                vst4q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), rw1);
            }
        }

        let rem = in_place.chunks_exact_mut(16).into_remainder();

        for chunk in rem.chunks_exact_mut(8) {
            unsafe {
                let uzp = vld4q_f64(chunk.as_ptr().cast());

                let a = vreinterpretq_f32_f64(uzp.0);
                let b = vreinterpretq_f32_f64(uzp.1);
                let c = vreinterpretq_f32_f64(uzp.2);
                let d = vreinterpretq_f32_f64(uzp.3);

                let t0 = vaddq_f32(a, c);
                let t1 = vsubq_f32(a, c);
                let t2 = vaddq_f32(b, d);
                let t3 = vsubq_f32(b, d);

                let rw0 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot90_f32(t1, t3)),
                    vreinterpretq_f64_f32(vsubq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot270_f32(t1, t3)),
                );

                vst4q_f64(chunk.as_mut_ptr().cast(), rw0);
            }
        }

        let rem = rem.chunks_exact_mut(8).into_remainder();

        for chunk in rem.chunks_exact_mut(4) {
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
                let t3 = vsub_f32(b, d);

                vst1q_f32(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(vadd_f32(t0, t2), vcadd_rot90_f32(t1, t3)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(vsub_f32(t0, t2), vcadd_rot270_f32(t1, t3)),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_backward_out(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
            unsafe {
                let uzp0 = vld4q_f64(src.as_ptr().cast());
                let uzp1 = vld4q_f64(src.get_unchecked(8..).as_ptr().cast());

                let u0 = vreinterpretq_f32_f64(uzp0.0);
                let u1 = vreinterpretq_f32_f64(uzp0.1);
                let u2 = vreinterpretq_f32_f64(uzp0.2);
                let u3 = vreinterpretq_f32_f64(uzp0.3);

                let u4 = vreinterpretq_f32_f64(uzp1.0);
                let u5 = vreinterpretq_f32_f64(uzp1.1);
                let u6 = vreinterpretq_f32_f64(uzp1.2);
                let u7 = vreinterpretq_f32_f64(uzp1.3);

                let t0 = vaddq_f32(u0, u2);
                let t4 = vaddq_f32(u4, u6);
                let t1 = vsubq_f32(u0, u2);
                let t5 = vsubq_f32(u4, u6);
                let t2 = vaddq_f32(u1, u3);
                let t6 = vaddq_f32(u5, u7);
                let t3 = vsubq_f32(u1, u3);
                let t7 = vsubq_f32(u5, u7);

                let rw0 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot90_f32(t1, t3)),
                    vreinterpretq_f64_f32(vsubq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot270_f32(t1, t3)),
                );
                let rw1 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t4, t6)),
                    vreinterpretq_f64_f32(vcaddq_rot90_f32(t5, t7)),
                    vreinterpretq_f64_f32(vsubq_f32(t4, t6)),
                    vreinterpretq_f64_f32(vcaddq_rot270_f32(t5, t7)),
                );

                vst4q_f64(dst.as_mut_ptr().cast(), rw0);
                vst4q_f64(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), rw1);
            }
        }

        let rem_dst = dst.chunks_exact_mut(16).into_remainder();
        let rem_src = src.chunks_exact(16).remainder();

        for (dst, src) in rem_dst.chunks_exact_mut(8).zip(rem_src.chunks_exact(8)) {
            unsafe {
                let uzp = vld4q_f64(src.as_ptr().cast());

                let a = vreinterpretq_f32_f64(uzp.0);
                let b = vreinterpretq_f32_f64(uzp.1);
                let c = vreinterpretq_f32_f64(uzp.2);
                let d = vreinterpretq_f32_f64(uzp.3);

                let t0 = vaddq_f32(a, c);
                let t1 = vsubq_f32(a, c);
                let t2 = vaddq_f32(b, d);
                let t3 = vsubq_f32(b, d);

                let rw0 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot90_f32(t1, t3)),
                    vreinterpretq_f64_f32(vsubq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vcaddq_rot270_f32(t1, t3)),
                );

                vst4q_f64(dst.as_mut_ptr().cast(), rw0);
            }
        }

        let rem_dst = rem_dst.chunks_exact_mut(16).into_remainder();
        let rem_src = rem_src.chunks_exact(16).remainder();

        for (dst, src) in rem_dst.chunks_exact_mut(4).zip(rem_src.chunks_exact(4)) {
            unsafe {
                let uz0 = vld1q_f32(src.get_unchecked(0..).as_ptr().cast());
                let uz1 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());

                let a = vget_low_f32(uz0);
                let b = vget_high_f32(uz0);
                let c = vget_low_f32(uz1);
                let d = vget_high_f32(uz1);

                let t0 = vadd_f32(a, c);
                let t1 = vsub_f32(a, c);
                let t2 = vadd_f32(b, d);
                let t3 = vsub_f32(b, d);

                vst1q_f32(
                    dst.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(vadd_f32(t0, t2), vcadd_rot90_f32(t1, t3)),
                );
                vst1q_f32(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(vsub_f32(t0, t2), vcadd_rot270_f32(t1, t3)),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly4<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            match self.direction {
                FftDirection::Forward => self.execute_forward(in_place),
                FftDirection::Inverse => self.execute_backward(in_place),
            }
        }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        4
    }
}

impl FftExecutorOutOfPlace<f32> for NeonFcmaButterfly4<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe {
            match self.direction {
                FftDirection::Forward => self.execute_forward_out(src, dst),
                FftDirection::Inverse => self.execute_backward_out(src, dst),
            }
        }
    }
}

impl CompositeFftExecutor<f32> for NeonFcmaButterfly4<f32> {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl NeonFcmaButterfly4<f64> {
    #[target_feature(enable = "fcma")]
    unsafe fn execute_forward(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(4) {
            unsafe {
                let a = vld1q_f64(chunk.get_unchecked(0..).as_ptr().cast());
                let b = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let c = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let d = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());

                let t0 = vaddq_f64(a, c);
                let t1 = vsubq_f64(a, c);
                let t2 = vaddq_f64(b, d);
                let t3 = vsubq_f64(b, d);

                vst1q_f64(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vaddq_f64(t0, t2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vcaddq_rot270_f64(t1, t3),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vsubq_f64(t0, t2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vcaddq_rot90_f64(t1, t3),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_forward_out(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            unsafe {
                let a = vld1q_f64(src.get_unchecked(0..).as_ptr().cast());
                let b = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());
                let c = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());
                let d = vld1q_f64(src.get_unchecked(3..).as_ptr().cast());

                let t0 = vaddq_f64(a, c);
                let t1 = vsubq_f64(a, c);
                let t2 = vaddq_f64(b, d);
                let t3 = vsubq_f64(b, d);

                vst1q_f64(
                    dst.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vaddq_f64(t0, t2),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vcaddq_rot270_f64(t1, t3),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vsubq_f64(t0, t2),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vcaddq_rot90_f64(t1, t3),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_backward(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(4) {
            unsafe {
                let a = vld1q_f64(chunk.get_unchecked(0..).as_ptr().cast());
                let b = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let c = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let d = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());

                let t0 = vaddq_f64(a, c);
                let t1 = vsubq_f64(a, c);
                let t2 = vaddq_f64(b, d);
                let t3 = vsubq_f64(b, d);

                vst1q_f64(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vaddq_f64(t0, t2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vcaddq_rot90_f64(t1, t3),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vsubq_f64(t0, t2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vcaddq_rot270_f64(t1, t3),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    unsafe fn execute_backward_out(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            unsafe {
                let a = vld1q_f64(src.get_unchecked(0..).as_ptr().cast());
                let b = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());
                let c = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());
                let d = vld1q_f64(src.get_unchecked(3..).as_ptr().cast());

                let t0 = vaddq_f64(a, c);
                let t1 = vsubq_f64(a, c);
                let t2 = vaddq_f64(b, d);
                let t3 = vsubq_f64(b, d);

                vst1q_f64(
                    dst.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vaddq_f64(t0, t2),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vcaddq_rot90_f64(t1, t3),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vsubq_f64(t0, t2),
                );
                vst1q_f64(
                    dst.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vcaddq_rot270_f64(t1, t3),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for NeonFcmaButterfly4<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe {
            match self.direction {
                FftDirection::Forward => self.execute_forward_out(src, dst),
                FftDirection::Inverse => self.execute_backward_out(src, dst),
            }
        }
    }
}

impl CompositeFftExecutor<f64> for NeonFcmaButterfly4<f64> {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_fcma_butterfly!(test_neon_butterfly4, f32, NeonFcmaButterfly4, 4, 1e-5);
    test_fcma_butterfly!(test_neon_butterfly4_f64, f64, NeonFcmaButterfly4, 4, 1e-7);
    test_oof_fcma_butterfly!(test_oof_butterfly4, f32, NeonFcmaButterfly4, 4, 1e-5);
    test_oof_fcma_butterfly!(test_oof_butterfly4_f64, f64, NeonFcmaButterfly4, 4, 1e-9);
}
