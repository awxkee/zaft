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
use crate::neon::util::v_transpose_complex_f32;
use crate::traits::FftTrigonometry;
use crate::util::validate_oof_sizes;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;
use std::marker::PhantomData;

pub(crate) struct NeonButterfly2<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonButterfly2<T>
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
        if !in_place.len().is_multiple_of(2) {
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

        let rem = rem.chunks_exact_mut(4).into_remainder();

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

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<f32>],
        _: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        self.execute(in_place)
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        validate_oof_sizes!(src, dst, 2);

        for (dst, src) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
            unsafe {
                let zu0_0 = vld1q_f32(src.get_unchecked(0..).as_ptr().cast());
                let zu1_0 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());
                let zu2_0 = vld1q_f32(src.get_unchecked(4..).as_ptr().cast());
                let zu3_0 = vld1q_f32(src.get_unchecked(6..).as_ptr().cast());

                let (u0_0, u1_0) = v_transpose_complex_f32(zu0_0, zu1_0);
                let (u2_0, u3_0) = v_transpose_complex_f32(zu2_0, zu3_0);

                let zy0 = vaddq_f32(u0_0, u1_0);
                let zy1 = vsubq_f32(u0_0, u1_0);
                let zy2 = vaddq_f32(u2_0, u3_0);
                let zy3 = vsubq_f32(u2_0, u3_0);

                let (y0, y1) = v_transpose_complex_f32(zy0, zy1);
                let (y2, y3) = v_transpose_complex_f32(zy2, zy3);

                vst1q_f32(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y1);
                vst1q_f32(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y2);
                vst1q_f32(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), y3);
            }
        }

        let rem_src = src.chunks_exact(8).remainder();
        let rem_dst = dst.chunks_exact_mut(8).into_remainder();

        for (dst, src) in rem_dst.chunks_exact_mut(4).zip(rem_src.chunks_exact(4)) {
            unsafe {
                let zu0_0 = vld1q_f32(src.get_unchecked(0..).as_ptr().cast());
                let zu1_0 = vld1q_f32(src.get_unchecked(2..).as_ptr().cast());

                let (u0_0, u1_0) = v_transpose_complex_f32(zu0_0, zu1_0);

                let zy0 = vaddq_f32(u0_0, u1_0);
                let zy1 = vsubq_f32(u0_0, u1_0);

                let (y0, y1) = v_transpose_complex_f32(zy0, zy1);

                vst1q_f32(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y1);
            }
        }

        let rem_src = rem_src.chunks_exact(4).remainder();
        let rem_dst = rem_dst.chunks_exact_mut(4).into_remainder();

        for (dst, src) in rem_dst.chunks_exact_mut(2).zip(rem_src.chunks_exact(2)) {
            unsafe {
                let u0_0 = vld1_f32(src.get_unchecked(0..).as_ptr().cast());
                let u1_0 = vld1_f32(src.get_unchecked(1..).as_ptr().cast());

                let y0 = vadd_f32(u0_0, u1_0);
                let y1 = vsub_f32(u0_0, u1_0);

                vst1_f32(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1_f32(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
            }
        }
        Ok(())
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
        _: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place(src, dst)
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<f32>],
        dst: &mut [Complex<f32>],
        _: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place(src, dst)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        2
    }

    fn scratch_length(&self) -> usize {
        0
    }

    fn out_of_place_scratch_length(&self) -> usize {
        0
    }

    fn destructive_scratch_length(&self) -> usize {
        0
    }
}

impl FftExecutor<f64> for NeonButterfly2<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(2) {
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

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        self.execute(in_place)
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            unsafe {
                let u0_0 = vld1q_f64(src.get_unchecked(0..).as_ptr().cast());
                let u1_0 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());
                let u0_1 = vld1q_f64(src.get_unchecked(2..).as_ptr().cast());
                let u1_1 = vld1q_f64(src.get_unchecked(3..).as_ptr().cast());

                let y0 = vaddq_f64(u0_0, u1_0);
                let y1 = vsubq_f64(u0_0, u1_0);
                let y2 = vaddq_f64(u0_1, u1_1);
                let y3 = vsubq_f64(u0_1, u1_1);

                vst1q_f64(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
                vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), y3);
            }
        }

        let rem_src = src.chunks_exact(4).remainder();
        let rem_dst = dst.chunks_exact_mut(4).into_remainder();

        for (dst, src) in rem_dst.chunks_exact_mut(2).zip(rem_src.chunks_exact(2)) {
            unsafe {
                let u0_0 = vld1q_f64(src.get_unchecked(0..).as_ptr().cast());
                let u1_0 = vld1q_f64(src.get_unchecked(1..).as_ptr().cast());

                let y0 = vaddq_f64(u0_0, u1_0);
                let y1 = vsubq_f64(u0_0, u1_0);

                vst1q_f64(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), y1);
            }
        }

        Ok(())
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place(src, dst)
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<f64>],
        dst: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place(src, dst)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        2
    }

    fn scratch_length(&self) -> usize {
        0
    }

    fn out_of_place_scratch_length(&self) -> usize {
        0
    }

    fn destructive_scratch_length(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_neon_butterfly2, f32, NeonButterfly2, 2, 1e-5);
    test_butterfly!(test_neon_butterfly2_f64, f64, NeonButterfly2, 2, 1e-7);
    test_oof_butterfly!(test_oof_butterfly2, f32, NeonButterfly2, 2, 1e-5);
    test_oof_butterfly!(test_oof_butterfly2_f64, f64, NeonButterfly2, 2, 1e-9);
}
