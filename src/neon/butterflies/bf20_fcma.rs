/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
use crate::neon::butterflies::{FastFcmaBf4d, NeonFastButterfly5};
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonFcmaButterfly20<T> {
    direction: FftDirection,
    bf5: NeonFastButterfly5<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonFcmaButterfly20<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf5: NeonFastButterfly5::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly20<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        20
    }
}

impl NeonFcmaButterfly20<f64> {
    #[target_feature(enable = "fcma")]
    fn execute_impl_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 20 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let bf4 = FastFcmaBf4d::new(self.direction);
            for chunk in in_place.chunks_exact_mut(20) {
                let u0 = vld1q_f64(chunk.as_ptr().cast());
                let u5 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let u10 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let u15 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());
                let u16 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());

                let u1 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                let u6 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                let u11 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());
                let u12 = vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast());
                let u17 = vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast());

                let u2 = vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast());
                let u7 = vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast());
                let u8 = vld1q_f64(chunk.get_unchecked(12..).as_ptr().cast());
                let u13 = vld1q_f64(chunk.get_unchecked(13..).as_ptr().cast());
                let u18 = vld1q_f64(chunk.get_unchecked(14..).as_ptr().cast());

                let u3 = vld1q_f64(chunk.get_unchecked(15..).as_ptr().cast());
                let u4 = vld1q_f64(chunk.get_unchecked(16..).as_ptr().cast());
                let u9 = vld1q_f64(chunk.get_unchecked(17..).as_ptr().cast());
                let u14 = vld1q_f64(chunk.get_unchecked(18..).as_ptr().cast());
                let u19 = vld1q_f64(chunk.get_unchecked(19..).as_ptr().cast());

                let (t0, t1, t2, t3) = bf4.exec(u0, u1, u2, u3);
                let (t4, t5, t6, t7) = bf4.exec(u4, u5, u6, u7);
                let (t8, t9, t10, t11) = bf4.exec(u8, u9, u10, u11);
                let (t12, t13, t14, t15) = bf4.exec(u12, u13, u14, u15);
                let (t16, t17, t18, t19) = bf4.exec(u16, u17, u18, u19);

                let (u0, u4, u8, u12, u16) = self.bf5.exec_fcma(t0, t4, t8, t12, t16);
                let (u5, u9, u13, u17, u1) = self.bf5.exec_fcma(t1, t5, t9, t13, t17);
                let (u10, u14, u18, u2, u6) = self.bf5.exec_fcma(t2, t6, t10, t14, t18);
                let (u15, u19, u3, u7, u11) = self.bf5.exec_fcma(t3, t7, t11, t15, t19);

                vst1q_f64(chunk.as_mut_ptr().cast(), u0);
                vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), u1);

                vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), u2);
                vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), u3);

                vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), u4);
                vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), u5);

                vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), u6);
                vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), u7);

                vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), u8);
                vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), u9);

                vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), u10);
                vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), u11);

                vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), u12);
                vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), u13);

                vst1q_f64(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), u14);
                vst1q_f64(chunk.get_unchecked_mut(15..).as_mut_ptr().cast(), u15);

                vst1q_f64(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), u16);
                vst1q_f64(chunk.get_unchecked_mut(17..).as_mut_ptr().cast(), u17);

                vst1q_f64(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), u18);
                vst1q_f64(chunk.get_unchecked_mut(19..).as_mut_ptr().cast(), u19);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neon::butterflies::test_fcma_butterfly;

    test_fcma_butterfly!(
        test_neon_butterfly20_f64,
        f64,
        NeonFcmaButterfly20,
        20,
        1e-7
    );
}
