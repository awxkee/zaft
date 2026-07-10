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

use crate::butterflies::Butterfly3;
use crate::butterflies::short_butterflies::FastButterfly2;
use crate::butterflies::util::boring_scalar_butterfly;
use crate::mla::fmla;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::f64::consts::PI;

pub(crate) struct Butterfly6<T> {
    direction: FftDirection,
    twiddle: Complex<T>,
    bf3: Butterfly3<T>,
}

impl<T: FftSample> Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Butterfly6 {
            direction: fft_direction,
            twiddle: compute_twiddle(1, 3, fft_direction),
            bf3: Butterfly3::new(fft_direction),
        }
    }
}

impl<T: FftSample> Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn run<S: BidirectionalStore<Complex<T>>>(&self, chunk: &mut S) {
        let fast_butterfly2 = FastButterfly2::new(self.direction);
        let u0 = chunk[0];
        let u1 = chunk[1];
        let u2 = chunk[2];
        let u3 = chunk[3];
        let u4 = chunk[4];
        let u5 = chunk[5];

        // Radix-6 butterfly

        let [t0, t2, t4] = self.bf3.exec(&[u0, u2, u4]);
        let [t1, t3, t5] = self.bf3.exec(&[u3, u5, u1]);
        let (y0, y3) = fast_butterfly2.butterfly2(t0, t1);
        let (y4, y1) = fast_butterfly2.butterfly2(t2, t3);
        let (y2, y5) = fast_butterfly2.butterfly2(t4, t5);

        chunk[0] = y0;
        chunk[1] = y1;
        chunk[2] = y2;
        chunk[3] = y3;
        chunk[4] = y4;
        chunk[5] = y5;
    }
}

boring_scalar_butterfly!(Butterfly6, 6);

impl<T: FftSample> R2CFftExecutor<T> for Butterfly6<T>
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

        for (dst, src) in output
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(input.as_chunks::<6>().0.iter())
        {
            let s03 = src[0] + src[3];
            let s14 = src[1] + src[4];
            let s25 = src[2] + src[5];
            let d03 = src[0] - src[3];
            let d14 = src[1] - src[4];
            let d25 = src[2] - src[5];
            let d15 = src[1] - src[5];
            let d24 = src[2] - src[4];

            let q0 = d14 - d25;
            let q1 = s14 + s25;
            let y0 = s03 + q1;
            let y3 = d03 - q0;
            let y1 = fmla(q0, T::HALF, d03);
            let y2 = fmla(q1, -T::HALF, s03);
            let y4 = -T::SQRT_3_OVER_2 * (d14 + d25);
            let y5 = T::SQRT_3_OVER_2 * (d24 - d15);

            dst[0] = Complex::new(y0, T::zero());
            dst[1] = Complex::new(y1, y4);
            dst[2] = Complex::new(y2, y5);
            dst[3] = Complex::new(y3, T::zero());
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
        6
    }

    #[inline]
    fn complex_length(&self) -> usize {
        4
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

    test_r2c_butterfly!(test_r2c_butterfly6, f32, Butterfly6, 6, 1e-5);
    test_r2c_butterfly!(test_r2c_butterfly6_f64, f64, Butterfly6, 6, 1e-5);
    test_butterfly!(test_butterfly6, f32, Butterfly6, 6, 1e-5);
    test_oof_butterfly!(test_oof_butterfly6, f32, Butterfly6, 6, 1e-5);
}
