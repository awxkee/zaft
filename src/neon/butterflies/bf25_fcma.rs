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
use crate::neon::mixed::{ColumnFcmaButterfly5d, NeonStoreD};
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;
use std::sync::Arc;

pub(crate) struct NeonFcmaButterfly25d {
    direction: FftDirection,
    bf5: ColumnFcmaButterfly5d,
    twiddles: [NeonStoreD; 20],
}

impl NeonFcmaButterfly25d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let mut twiddles = [NeonStoreD::default(); 20];
        for (x, row) in twiddles.chunks_exact_mut(5).enumerate() {
            for (y, dst) in row.iter_mut().enumerate() {
                *dst = NeonStoreD::from_complex(&compute_twiddle((x + 1) * y, 25, fft_direction));
            }
        }
        Self {
            direction: fft_direction,
            twiddles,
            bf5: ColumnFcmaButterfly5d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for NeonFcmaButterfly25d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        25
    }
}

impl NeonFcmaButterfly25d {
    #[target_feature(enable = "fcma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(25) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(25) {
                let u0 = NeonStoreD::raw(vld1q_f64(chunk.as_ptr().cast()));
                let u1 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast()));
                let u2 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast()));
                let u3 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast()));
                let u4 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast()));
                let u5 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast()));
                let u6 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast()));
                let u7 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast()));
                let u8 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast()));
                let u9 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast()));
                let u10 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast()));
                let u11 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast()));
                let u12 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(12..).as_ptr().cast()));
                let u13 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(13..).as_ptr().cast()));
                let u14 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(14..).as_ptr().cast()));
                let u15 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(15..).as_ptr().cast()));
                let u16 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(16..).as_ptr().cast()));
                let u17 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(17..).as_ptr().cast()));
                let u18 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(18..).as_ptr().cast()));
                let u19 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(19..).as_ptr().cast()));
                let u20 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(20..).as_ptr().cast()));
                let u21 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(21..).as_ptr().cast()));
                let u22 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(22..).as_ptr().cast()));
                let u23 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(23..).as_ptr().cast()));
                let u24 = NeonStoreD::raw(vld1q_f64(chunk.get_unchecked(24..).as_ptr().cast()));

                let s0 = self.bf5.exec([u0, u5, u10, u15, u20]);
                let mut s1 = self.bf5.exec([u1, u6, u11, u16, u21]);
                let mut s2 = self.bf5.exec([u2, u7, u12, u17, u22]);
                for i in 0..5 {
                    s1[i] = NeonStoreD::fcmul_fcma(s1[i], self.twiddles[i]);
                    s2[i] = NeonStoreD::fcmul_fcma(s2[i], self.twiddles[5 + i]);
                }
                let mut s3 = self.bf5.exec([u3, u8, u13, u18, u23]);
                let mut s4 = self.bf5.exec([u4, u9, u14, u19, u24]);
                for i in 0..5 {
                    s3[i] = NeonStoreD::fcmul_fcma(s3[i], self.twiddles[10 + i]);
                    s4[i] = NeonStoreD::fcmul_fcma(s4[i], self.twiddles[15 + i]);
                }

                let z0 = self.bf5.exec([s0[0], s1[0], s2[0], s3[0], s4[0]]);
                let z1 = self.bf5.exec([s0[1], s1[1], s2[1], s3[1], s4[1]]);
                let z2 = self.bf5.exec([s0[2], s1[2], s2[2], s3[2], s4[2]]);
                for i in 0..5 {
                    z0[i].write(chunk.get_unchecked_mut(i * 5..));
                    z1[i].write(chunk.get_unchecked_mut(i * 5 + 1..));
                    z2[i].write(chunk.get_unchecked_mut(i * 5 + 2..));
                }
                let z3 = self.bf5.exec([s0[3], s1[3], s2[3], s3[3], s4[3]]);
                let z4 = self.bf5.exec([s0[4], s1[4], s2[4], s3[4], s4[4]]);
                for i in 0..5 {
                    z3[i].write(chunk.get_unchecked_mut(i * 5 + 3..));
                    z4[i].write(chunk.get_unchecked_mut(i * 5 + 4..));
                }
            }
        }
        Ok(())
    }

    #[target_feature(enable = "fcma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(25) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(25) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(25).zip(src.chunks_exact(25)) {
                let u0 = NeonStoreD::raw(vld1q_f64(src.as_ptr().cast()));
                let u1 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(1..).as_ptr().cast()));
                let u2 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(2..).as_ptr().cast()));
                let u3 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(3..).as_ptr().cast()));
                let u4 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(4..).as_ptr().cast()));
                let u5 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(5..).as_ptr().cast()));
                let u6 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(6..).as_ptr().cast()));
                let u7 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(7..).as_ptr().cast()));
                let u8 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(8..).as_ptr().cast()));
                let u9 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(9..).as_ptr().cast()));
                let u10 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(10..).as_ptr().cast()));
                let u11 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(11..).as_ptr().cast()));
                let u12 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(12..).as_ptr().cast()));
                let u13 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(13..).as_ptr().cast()));
                let u14 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(14..).as_ptr().cast()));
                let u15 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(15..).as_ptr().cast()));
                let u16 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(16..).as_ptr().cast()));
                let u17 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(17..).as_ptr().cast()));
                let u18 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(18..).as_ptr().cast()));
                let u19 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(19..).as_ptr().cast()));
                let u20 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(20..).as_ptr().cast()));
                let u21 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(21..).as_ptr().cast()));
                let u22 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(22..).as_ptr().cast()));
                let u23 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(23..).as_ptr().cast()));
                let u24 = NeonStoreD::raw(vld1q_f64(src.get_unchecked(24..).as_ptr().cast()));

                let s0 = self.bf5.exec([u0, u5, u10, u15, u20]);
                let mut s1 = self.bf5.exec([u1, u6, u11, u16, u21]);
                let mut s2 = self.bf5.exec([u2, u7, u12, u17, u22]);
                for i in 0..5 {
                    s1[i] = NeonStoreD::fcmul_fcma(s1[i], self.twiddles[i]);
                    s2[i] = NeonStoreD::fcmul_fcma(s2[i], self.twiddles[5 + i]);
                }
                let mut s3 = self.bf5.exec([u3, u8, u13, u18, u23]);
                let mut s4 = self.bf5.exec([u4, u9, u14, u19, u24]);
                for i in 0..5 {
                    s3[i] = NeonStoreD::fcmul_fcma(s3[i], self.twiddles[10 + i]);
                    s4[i] = NeonStoreD::fcmul_fcma(s4[i], self.twiddles[15 + i]);
                }

                let z0 = self.bf5.exec([s0[0], s1[0], s2[0], s3[0], s4[0]]);
                let z1 = self.bf5.exec([s0[1], s1[1], s2[1], s3[1], s4[1]]);
                let z2 = self.bf5.exec([s0[2], s1[2], s2[2], s3[2], s4[2]]);
                for i in 0..5 {
                    z0[i].write(dst.get_unchecked_mut(i * 5..));
                    z1[i].write(dst.get_unchecked_mut(i * 5 + 1..));
                    z2[i].write(dst.get_unchecked_mut(i * 5 + 2..));
                }
                let z3 = self.bf5.exec([s0[3], s1[3], s2[3], s3[3], s4[3]]);
                let z4 = self.bf5.exec([s0[4], s1[4], s2[4], s3[4], s4[4]]);
                for i in 0..5 {
                    z3[i].write(dst.get_unchecked_mut(i * 5 + 3..));
                    z4[i].write(dst.get_unchecked_mut(i * 5 + 4..));
                }
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for NeonFcmaButterfly25d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for NeonFcmaButterfly25d {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_fcma_butterfly!(
        test_neon_butterfly25_f64,
        f64,
        NeonFcmaButterfly25d,
        25,
        1e-7
    );
    test_oof_fcma_butterfly!(
        test_oof_butterfly25_f64,
        f64,
        NeonFcmaButterfly25d,
        25,
        1e-9
    );
}
