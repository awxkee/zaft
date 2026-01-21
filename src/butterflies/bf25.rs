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
#![allow(clippy::needless_range_loop)]
use crate::butterflies::short_butterflies::FastButterfly5;
use crate::complex_fma::c_mul_fast;
use crate::util::compute_twiddle;
use crate::{
    CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, FftSample, ZaftError,
};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;

#[allow(unused)]
pub(crate) struct Butterfly25<T> {
    direction: FftDirection,
    twiddles: [Complex<T>; 20],
    bf5: FastButterfly5<T>,
}

#[allow(unused)]
impl<T: FftSample> Butterfly25<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        let mut twiddles = [Complex::default(); 20];
        for (x, row) in twiddles.chunks_exact_mut(5).enumerate() {
            for (y, dst) in row.iter_mut().enumerate() {
                *dst = compute_twiddle((x + 1) * y, 25, fft_direction);
            }
        }

        Butterfly25 {
            direction: fft_direction,
            twiddles,
            bf5: FastButterfly5::new(fft_direction),
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Butterfly25<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.length()) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(25) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];

            let u4 = chunk[4];
            let u5 = chunk[5];
            let u6 = chunk[6];
            let u7 = chunk[7];

            let u8 = chunk[8];
            let u9 = chunk[9];
            let u10 = chunk[10];
            let u11 = chunk[11];
            let u12 = chunk[12];

            let u13 = chunk[13];
            let u14 = chunk[14];
            let u15 = chunk[15];
            let u16 = chunk[16];

            let u17 = chunk[17];
            let u18 = chunk[18];

            let u19 = chunk[19];
            let u20 = chunk[20];

            let u21 = chunk[21];
            let u22 = chunk[22];

            let u23 = chunk[23];
            let u24 = chunk[24];

            let s0 = self.bf5.exec5(u0, u5, u10, u15, u20);
            let mut s1 = self.bf5.exec5(u1, u6, u11, u16, u21);
            for i in 0..5 {
                s1[i] = c_mul_fast(s1[i], self.twiddles[i]);
            }
            let mut s2 = self.bf5.exec5(u2, u7, u12, u17, u22);
            for i in 0..5 {
                s2[i] = c_mul_fast(s2[i], self.twiddles[5 + i]);
            }
            let mut s3 = self.bf5.exec5(u3, u8, u13, u18, u23);
            for i in 0..5 {
                s3[i] = c_mul_fast(s3[i], self.twiddles[10 + i]);
            }
            let mut s4 = self.bf5.exec5(u4, u9, u14, u19, u24);
            for i in 0..5 {
                s4[i] = c_mul_fast(s4[i], self.twiddles[15 + i]);
            }

            let z0 = self.bf5.exec5(s0[0], s1[0], s2[0], s3[0], s4[0]);
            for i in 0..5 {
                chunk[i * 5] = z0[i];
            }
            let z1 = self.bf5.exec5(s0[1], s1[1], s2[1], s3[1], s4[1]);
            for i in 0..5 {
                chunk[i * 5 + 1] = z1[i];
            }
            let z2 = self.bf5.exec5(s0[2], s1[2], s2[2], s3[2], s4[2]);
            for i in 0..5 {
                chunk[i * 5 + 2] = z2[i];
            }
            let z3 = self.bf5.exec5(s0[3], s1[3], s2[3], s3[3], s4[3]);
            for i in 0..5 {
                chunk[i * 5 + 3] = z3[i];
            }
            let z4 = self.bf5.exec5(s0[4], s1[4], s2[4], s3[4], s4[4]);
            for i in 0..5 {
                chunk[i * 5 + 4] = z4[i];
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        25
    }
}

impl<T: FftSample> FftExecutorOutOfPlace<T> for Butterfly25<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(self.length()) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(self.length()) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(25).zip(src.chunks_exact(25)) {
            let u0 = src[0];
            let u1 = src[1];
            let u2 = src[2];
            let u3 = src[3];

            let u4 = src[4];
            let u5 = src[5];
            let u6 = src[6];
            let u7 = src[7];

            let u8 = src[8];
            let u9 = src[9];
            let u10 = src[10];
            let u11 = src[11];
            let u12 = src[12];

            let u13 = src[13];
            let u14 = src[14];
            let u15 = src[15];
            let u16 = src[16];

            let u17 = src[17];
            let u18 = src[18];

            let u19 = src[19];
            let u20 = src[20];

            let u21 = src[21];
            let u22 = src[22];

            let u23 = src[23];
            let u24 = src[24];

            let s0 = self.bf5.exec5(u0, u5, u10, u15, u20);
            let mut s1 = self.bf5.exec5(u1, u6, u11, u16, u21);
            for i in 0..5 {
                s1[i] = c_mul_fast(s1[i], self.twiddles[i]);
            }
            let mut s2 = self.bf5.exec5(u2, u7, u12, u17, u22);
            for i in 0..5 {
                s2[i] = c_mul_fast(s2[i], self.twiddles[5 + i]);
            }
            let mut s3 = self.bf5.exec5(u3, u8, u13, u18, u23);
            for i in 0..5 {
                s3[i] = c_mul_fast(s3[i], self.twiddles[10 + i]);
            }
            let mut s4 = self.bf5.exec5(u4, u9, u14, u19, u24);
            for i in 0..5 {
                s4[i] = c_mul_fast(s4[i], self.twiddles[15 + i]);
            }

            let z0 = self.bf5.exec5(s0[0], s1[0], s2[0], s3[0], s4[0]);
            for i in 0..5 {
                dst[i * 5] = z0[i];
            }
            let z1 = self.bf5.exec5(s0[1], s1[1], s2[1], s3[1], s4[1]);
            for i in 0..5 {
                dst[i * 5 + 1] = z1[i];
            }
            let z2 = self.bf5.exec5(s0[2], s1[2], s2[2], s3[2], s4[2]);
            for i in 0..5 {
                dst[i * 5 + 2] = z2[i];
            }
            let z3 = self.bf5.exec5(s0[3], s1[3], s2[3], s3[3], s4[3]);
            for i in 0..5 {
                dst[i * 5 + 3] = z3[i];
            }
            let z4 = self.bf5.exec5(s0[4], s1[4], s2[4], s3[4], s4[4]);
            for i in 0..5 {
                dst[i * 5 + 4] = z4[i];
            }
        }
        Ok(())
    }
}

impl<T: FftSample> CompositeFftExecutor<T> for Butterfly25<T>
where
    f64: AsPrimitive<T>,
{
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<T> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_butterfly25, f32, Butterfly25, 25, 1e-5);
    test_oof_butterfly!(test_oof_butterfly25, f32, Butterfly25, 25, 1e-5);
}
