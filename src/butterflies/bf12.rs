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
use crate::butterflies::short_butterflies::{FastButterfly3, FastButterfly4};
use crate::butterflies::util::boring_scalar_butterfly;
use crate::store::BidirectionalStore;
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::marker::PhantomData;

#[allow(unused)]
pub(crate) struct Butterfly12<T> {
    direction: FftDirection,
    phantom_data: PhantomData<T>,
    bf3: FastButterfly3<T>,
    bf4: FastButterfly4<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly12 {
            direction: fft_direction,
            phantom_data: PhantomData,
            bf3: FastButterfly3::new(fft_direction),
            bf4: FastButterfly4::new(fft_direction),
        }
    }
}

impl<T: FftSample> Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    #[allow(dead_code)] // live in scalar builds; displaced by SIMD butterflies
    #[inline(always)]
    pub(crate) fn run<S: BidirectionalStore<Complex<T>>>(&self, chunk: &mut S) {
        let u0 = chunk[0];
        let u1 = chunk[3];
        let u2 = chunk[6];
        let u3 = chunk[9];

        let u4 = chunk[4];
        let u5 = chunk[7];
        let u6 = chunk[10];
        let u7 = chunk[1];

        let u8 = chunk[8];
        let u9 = chunk[11];
        let u10 = chunk[2];
        let u11 = chunk[5];

        let (u0, u1, u2, u3) = self.bf4.butterfly4(u0, u1, u2, u3);
        let (u4, u5, u6, u7) = self.bf4.butterfly4(u4, u5, u6, u7);
        let (u8, u9, u10, u11) = self.bf4.butterfly4(u8, u9, u10, u11);

        let (v0, v4, v8) = self.bf3.butterfly3(u0, u4, u8); // (v0, v4, v8)
        let (v9, v1, v5) = self.bf3.butterfly3(u1, u5, u9); // (v9, v1, v5)
        let (v6, v10, v2) = self.bf3.butterfly3(u2, u6, u10); // (v6, v10, v2)
        let (v3, v7, v11) = self.bf3.butterfly3(u3, u7, u11); // (v3, v7, v11)

        chunk[0] = v0;
        chunk[1] = v1;
        chunk[2] = v2;
        chunk[3] = v3;

        chunk[4] = v4;
        chunk[5] = v5;
        chunk[6] = v6;
        chunk[7] = v7;

        chunk[8] = v8;
        chunk[9] = v9;
        chunk[10] = v10;
        chunk[11] = v11;
    }
}

boring_scalar_butterfly!(Butterfly12, 12);

impl<T: FftSample> R2CFftExecutor<T> for Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.real_length(),
            output.len(),
            self.complex_length(),
        )?;

        for (chunk, complex) in input
            .as_chunks::<12>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<7>().0.iter_mut())
        {
            let s1 = chunk[1] + chunk[11];
            let d1 = chunk[1] - chunk[11];
            let s2 = chunk[2] + chunk[10];
            let d2 = chunk[2] - chunk[10];
            let s3 = chunk[3] + chunk[9];
            let d3 = chunk[3] - chunk[9];
            let s4 = chunk[4] + chunk[8];
            let d4 = chunk[4] - chunk[8];
            let s5 = chunk[5] + chunk[7];
            let d5 = chunk[5] - chunk[7];
            let s6 = chunk[6];

            let sqrt_s1 = T::SQRT_3_OVER_2 * s1;
            let sqrt_s5 = T::SQRT_3_OVER_2 * s5;
            let hs1 = T::HALF * s1;
            let hs2 = T::HALF * s2;
            let hs4 = T::HALF * s4;
            let hs5 = T::HALF * s5;
            let p_hs2_hs4 = hs2 - hs4;
            let n_hs2_hs4 = -hs2 - hs4;
            let x0_ns6 = chunk[0] - s6;
            let x0_ps6 = chunk[0] + s6;

            let hd1 = T::HALF * d1;
            let hd5 = T::HALF * d5;
            let hd1_hd5 = hd1 + hd5;
            let hd1_hd5_d3 = hd1_hd5 + d3;
            let sqrt_d2pd4 = T::SQRT_3_OVER_2 * (d2 + d4);
            let sqrt3_d1nd5 = T::SQRT_3_OVER_2 * (d1 - d5);
            let sqrt3_d2nd4 = T::SQRT_3_OVER_2 * (d2 - d4);

            // DC
            let s2_s6 = s2 + s6;
            let y0 = chunk[0] + s1 + s2_s6 + s3 + s4 + s5;

            let j0 = sqrt_s1 - sqrt_s5;
            let j1 = s3 - hs5;
            let j1_m_hs1 = j1 - hs1;

            let p_hs2_x0_ns = p_hs2_hs4 + x0_ns6;
            let x0_ps6_p_n_hs2_hs4 = x0_ps6 + n_hs2_hs4;

            let y1r = j0 + p_hs2_x0_ns;
            let y2r = x0_ps6_p_n_hs2_hs4 - j1_m_hs1;
            let y3r = chunk[0] + s4 - s2_s6;
            let y4r = x0_ps6_p_n_hs2_hs4 + j1_m_hs1;
            let y5r = p_hs2_x0_ns - j0;
            let y6r = chunk[0] - s1 + s2_s6 - s3 + s4 - s5;

            let sqrt3_sum = sqrt3_d1nd5 + sqrt3_d2nd4;
            let sqrt3_dif = sqrt3_d1nd5 - sqrt3_d2nd4;
            let neg_hd1_hd5_d3 = -hd1_hd5_d3;
            let y1i = neg_hd1_hd5_d3 - sqrt_d2pd4;
            let y2i = -sqrt3_sum;
            let y3i = -(d1 - d3 + d5);
            let y4i = -sqrt3_dif;
            let y5i = neg_hd1_hd5_d3 + sqrt_d2pd4;

            complex[0] = Complex::new(y0, T::zero());
            complex[1] = Complex::new(y1r, y1i);
            complex[2] = Complex::new(y2r, y2i);
            complex[3] = Complex::new(y3r, y3i);
            complex[4] = Complex::new(y4r, y4i);
            complex[5] = Complex::new(y5r, y5i);
            complex[6] = Complex::new(y6r, T::zero());
        }
        Ok(())
    }

    fn execute_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        _: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.real_length(),
            output.len(),
            self.complex_length(),
        )?;
        R2CFftExecutor::execute(self, input, output)
    }

    #[inline]
    fn real_length(&self) -> usize {
        12
    }

    #[inline]
    fn complex_length(&self) -> usize {
        7
    }

    fn complex_scratch_length(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    use crate::r2c::test_r2c_butterfly;

    test_butterfly!(test_butterfly12, f32, Butterfly12, 12, 1e-5);
    test_r2c_butterfly!(test_r2c_butterfly12, f32, Butterfly12, 12, 1e-5);
    test_r2c_butterfly!(test_r2c_butterfly12_f64, f64, Butterfly12, 12, 1e-9);
}
