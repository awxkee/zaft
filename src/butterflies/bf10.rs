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
#![allow(unused)]
use crate::butterflies::short_butterflies::{FastButterfly2, FastButterfly5};
use crate::butterflies::util::boring_scalar_butterfly;
use crate::mla::fmla;
use crate::store::BidirectionalStore;
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};

pub(crate) struct Butterfly10<T> {
    direction: FftDirection,
    bf5: FastButterfly5<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> Butterfly10<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly10 {
            direction: fft_direction,
            bf5: FastButterfly5::new(fft_direction),
        }
    }
}

impl<T: FftSample> Butterfly10<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn run<S: BidirectionalStore<Complex<T>>>(&self, chunk: &mut S) {
        let bf2 = FastButterfly2::new(self.direction);
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

        // Good-thomas butterfly-10
        let mid0 = self.bf5.bf5(u0, u2, u4, u6, u8);
        let mid1 = self.bf5.bf5(u5, u7, u9, u1, u3);

        // Since this is good-thomas algorithm, we don't need twiddle factors
        let (y0, y1) = bf2.butterfly2(mid0.0, mid1.0);
        let (y2, y3) = bf2.butterfly2(mid0.1, mid1.1);
        let (y4, y5) = bf2.butterfly2(mid0.2, mid1.2);
        let (y6, y7) = bf2.butterfly2(mid0.3, mid1.3);
        let (y8, y9) = bf2.butterfly2(mid0.4, mid1.4);

        chunk[0] = y0;
        chunk[1] = y3;
        chunk[2] = y4;

        chunk[3] = y7;
        chunk[4] = y8;
        chunk[5] = y1;

        chunk[6] = y2;
        chunk[7] = y5;
        chunk[8] = y6;

        chunk[9] = y9;
    }
}

boring_scalar_butterfly!(Butterfly10, 10);

pub(crate) struct RdftButterfly10<T> {
    c1: T,
    c2: T,
    s1: T,
    s2: T,
}

impl<T: FftTrigonometry + Float + 'static + Default> RdftButterfly10<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        let tw0 = (1f64.as_() / 5f64.as_()).sincos_pi();
        let tw1 = (2f64.as_() / 5f64.as_()).sincos_pi();
        RdftButterfly10 {
            c1: tw0.1,
            c2: tw1.1,
            s1: tw0.0,
            s2: tw1.0,
        }
    }
}

impl<T: FftSample> R2CFftExecutor<T> for RdftButterfly10<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !input.len().is_multiple_of(10) {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), 10));
        }
        if !output.len().is_multiple_of(6) {
            return Err(ZaftError::InvalidSizeMultiplier(output.len(), 6));
        }

        for (input, complex) in input
            .as_chunks::<10>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<6>().0.iter_mut())
        {
            let s1p = input[1] + input[9];
            let d1p = input[1] - input[9];
            let s2p = input[2] + input[8];
            let d2p = input[2] - input[8];
            let s3p = input[3] + input[7];
            let d3p = input[3] - input[7];
            let s4p = input[4] + input[6];
            let d4p = input[4] - input[6];
            let s5p = input[5]; // Nyquist

            // DC
            let s5x0 = s5p + input[0];
            let x0ms5p = input[0] - s5p;

            let s1p_ps4p = s1p + s4p;
            let s2p_ps3p = s2p + s3p;

            let y0 = s5x0 + s1p_ps4p + s2p_ps3p;

            let s1p_ms4p = s1p - s4p;
            let s2p_ms3p = s2p - s3p;

            let c1_s1ms4 = self.c1 * s1p_ms4p;
            let c2_s2ms3 = self.c2 * s2p_ms3p;
            let c2_s1ms4 = self.c2 * s1p_ms4p;
            let c1_s2ms3 = self.c1 * s2p_ms3p;
            let c1_s1ps4 = self.c1 * s1p_ps4p;
            let c2_s2ps3 = self.c2 * s2p_ps3p;
            let c2_s1ps4 = self.c2 * s1p_ps4p;
            let c1_s2ps3 = self.c1 * s2p_ps3p;

            let x0_ns5 = x0ms5p;
            let x0_ps5 = s5x0;

            let y1r = x0_ns5 + c1_s1ms4 + c2_s2ms3;
            let y2r = x0_ps5 + c2_s1ps4 - c1_s2ps3;
            let y3r = x0_ns5 - c2_s1ms4 - c1_s2ms3;
            let y4r = x0_ps5 - c1_s1ps4 + c2_s2ps3;
            let y5r = x0ms5p - s1p + s2p - s3p + s4p; // Nyquist

            let d1p_pd4p = d1p + d4p;
            let d2p_pd3p = d2p + d3p;
            let d1p_nd4p = d1p - d4p;
            let d2p_nd3p = d2p - d3p;

            let s1_d1pd4 = self.s1 * d1p_pd4p;
            let s2_d2pd3 = self.s2 * d2p_pd3p;
            let s2_d1pd4 = self.s2 * d1p_pd4p;
            let s1_d2pd3 = self.s1 * d2p_pd3p;
            let s2_d1nd4 = self.s2 * d1p_nd4p;
            let s1_d2nd3 = self.s1 * d2p_nd3p;
            let s1_d1nd4 = self.s1 * d1p_nd4p;
            let s2_d2nd3 = self.s2 * d2p_nd3p;

            let y1i = -(s1_d1pd4 + s2_d2pd3);
            let y2i = -(s2_d1nd4 + s1_d2nd3);
            let y3i = -(s2_d1pd4 - s1_d2pd3);
            let y4i = -(s1_d1nd4 - s2_d2nd3);

            complex[0] = Complex::new(y0, T::zero());
            complex[1] = Complex::new(y1r, y1i);
            complex[2] = Complex::new(y2r, y2i);
            complex[3] = Complex::new(y3r, y3i);
            complex[4] = Complex::new(y4r, y4i);
            complex[5] = Complex::new(y5r, T::zero());
        }
        Ok(())
    }

    fn execute_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        _: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        R2CFftExecutor::execute(self, input, output)
    }

    fn real_length(&self) -> usize {
        10
    }

    fn complex_length(&self) -> usize {
        6
    }

    fn complex_scratch_length(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    use crate::r2c::test_r2c_butterfly;

    test_butterfly!(test_butterfly10, f32, Butterfly10, 10, 1e-5);
    test_oof_butterfly!(test_oof_butterfly10, f32, Butterfly10, 10, 1e-5);
    test_r2c_butterfly!(test_butterfly10_r2c, f32, RdftButterfly10, 10, 1e-5);
}
