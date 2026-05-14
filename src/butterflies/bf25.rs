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
#![allow(clippy::needless_range_loop, unused)]
use crate::butterflies::short_butterflies::FastButterfly5;
use crate::butterflies::util::boring_scalar_butterfly;
use crate::complex_fma::c_mul_fast;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::AsPrimitive;

#[allow(unused)]
pub(crate) struct Butterfly25<T> {
    direction: FftDirection,
    twiddles: [Complex<T>; 9],
    bf5: FastButterfly5<T>,
}

#[allow(unused)]
impl<T: FftSample> Butterfly25<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        const EXPONENTS: [usize; 9] = [1, 2, 3, 4, 6, 8, 9, 12, 16];
        let twiddles = EXPONENTS.map(|e| compute_twiddle(e, 25, fft_direction));
        Butterfly25 {
            direction: fft_direction,
            twiddles,
            bf5: FastButterfly5::new(fft_direction),
        }
    }
}

impl<T: FftSample> Butterfly25<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn run<S: BidirectionalStore<Complex<T>>>(&self, chunk: &mut S) {
        let [t0, t1, t2, t3, t4, t5, t6, t7, t8] = self.twiddles;

        let s0 = self
            .bf5
            .exec5(chunk[0], chunk[5], chunk[10], chunk[15], chunk[20]);

        let mut s1 = self
            .bf5
            .exec5(chunk[1], chunk[6], chunk[11], chunk[16], chunk[21]);
        s1[1] = c_mul_fast(s1[1], t0);
        s1[2] = c_mul_fast(s1[2], t1);
        s1[3] = c_mul_fast(s1[3], t2);
        s1[4] = c_mul_fast(s1[4], t3);

        let mut s2 = self
            .bf5
            .exec5(chunk[2], chunk[7], chunk[12], chunk[17], chunk[22]);
        s2[1] = c_mul_fast(s2[1], t1);
        s2[2] = c_mul_fast(s2[2], t3);
        s2[3] = c_mul_fast(s2[3], t4);
        s2[4] = c_mul_fast(s2[4], t5);

        let mut s3 = self
            .bf5
            .exec5(chunk[3], chunk[8], chunk[13], chunk[18], chunk[23]);
        s3[1] = c_mul_fast(s3[1], t2);
        s3[2] = c_mul_fast(s3[2], t4);
        s3[3] = c_mul_fast(s3[3], t6);
        s3[4] = c_mul_fast(s3[4], t7);

        let mut s4 = self
            .bf5
            .exec5(chunk[4], chunk[9], chunk[14], chunk[19], chunk[24]);
        s4[1] = c_mul_fast(s4[1], t3);
        s4[2] = c_mul_fast(s4[2], t5);
        s4[3] = c_mul_fast(s4[3], t7);
        s4[4] = c_mul_fast(s4[4], t8);

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
}

boring_scalar_butterfly!(Butterfly25, 25);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_butterfly25, f32, Butterfly25, 25, 1e-5);
    test_oof_butterfly!(test_oof_butterfly25, f32, Butterfly25, 25, 1e-5);
}
