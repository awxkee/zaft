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
use crate::butterflies::fast_bf9::FastButterfly9;
use crate::butterflies::util::boring_scalar_butterfly;
use crate::complex_fma::c_mul_fast;
use crate::store::BidirectionalStore;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::AsPrimitive;

pub(crate) struct Butterfly27<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    twiddle7: Complex<T>,
    twiddle8: Complex<T>,
    twiddle9: Complex<T>,
    twiddle10: Complex<T>,
    twiddle11: Complex<T>,
    twiddle12: Complex<T>,
    bf9: FastButterfly9<T>,
}

impl<T: FftSample> Butterfly27<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly27 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 27, fft_direction),
            twiddle2: compute_twiddle(2, 27, fft_direction),
            twiddle3: compute_twiddle(3, 27, fft_direction),
            twiddle4: compute_twiddle(4, 27, fft_direction),
            twiddle5: compute_twiddle(5, 27, fft_direction),
            twiddle6: compute_twiddle(6, 27, fft_direction),
            twiddle7: compute_twiddle(7, 27, fft_direction),
            twiddle8: compute_twiddle(8, 27, fft_direction),
            twiddle9: compute_twiddle(10, 27, fft_direction),
            twiddle10: compute_twiddle(12, 27, fft_direction),
            twiddle11: compute_twiddle(14, 27, fft_direction),
            twiddle12: compute_twiddle(16, 27, fft_direction),
            bf9: FastButterfly9::new(fft_direction),
        }
    }
}

impl<T: FftSample> Butterfly27<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn run<S: BidirectionalStore<Complex<T>>>(&self, chunk: &mut S) {
        let s0 = self.bf9.exec(
            chunk[0], chunk[3], chunk[6], chunk[9], chunk[12], chunk[15], chunk[18], chunk[21],
            chunk[24],
        );
        let s1 = self.bf9.exec(
            chunk[1], chunk[4], chunk[7], chunk[10], chunk[13], chunk[16], chunk[19], chunk[22],
            chunk[25],
        );
        let s2 = self.bf9.exec(
            chunk[2], chunk[5], chunk[8], chunk[11], chunk[14], chunk[17], chunk[20], chunk[23],
            chunk[26],
        );

        // lane 0 — no twiddles
        let z = self.bf9.bf3.butterfly3(s0.0, s1.0, s2.0);
        chunk[0] = z.0;
        chunk[9] = z.1;
        chunk[18] = z.2;

        // lane 1
        let z = self.bf9.bf3.butterfly3(
            s0.1,
            c_mul_fast(s1.1, self.twiddle1),
            c_mul_fast(s2.1, self.twiddle2),
        );
        chunk[1] = z.0;
        chunk[10] = z.1;
        chunk[19] = z.2;

        // lane 2
        let z = self.bf9.bf3.butterfly3(
            s0.2,
            c_mul_fast(s1.2, self.twiddle2),
            c_mul_fast(s2.2, self.twiddle4),
        );
        chunk[2] = z.0;
        chunk[11] = z.1;
        chunk[20] = z.2;

        // lane 3
        let z = self.bf9.bf3.butterfly3(
            s0.3,
            c_mul_fast(s1.3, self.twiddle3),
            c_mul_fast(s2.3, self.twiddle6),
        );
        chunk[3] = z.0;
        chunk[12] = z.1;
        chunk[21] = z.2;

        // lane 4
        let z = self.bf9.bf3.butterfly3(
            s0.4,
            c_mul_fast(s1.4, self.twiddle4),
            c_mul_fast(s2.4, self.twiddle8),
        );
        chunk[4] = z.0;
        chunk[13] = z.1;
        chunk[22] = z.2;

        // lane 5
        let z = self.bf9.bf3.butterfly3(
            s0.5,
            c_mul_fast(s1.5, self.twiddle5),
            c_mul_fast(s2.5, self.twiddle9),
        );
        chunk[5] = z.0;
        chunk[14] = z.1;
        chunk[23] = z.2;

        // lane 6
        let z = self.bf9.bf3.butterfly3(
            s0.6,
            c_mul_fast(s1.6, self.twiddle6),
            c_mul_fast(s2.6, self.twiddle10),
        );
        chunk[6] = z.0;
        chunk[15] = z.1;
        chunk[24] = z.2;

        // lane 7
        let z = self.bf9.bf3.butterfly3(
            s0.7,
            c_mul_fast(s1.7, self.twiddle7),
            c_mul_fast(s2.7, self.twiddle11),
        );
        chunk[7] = z.0;
        chunk[16] = z.1;
        chunk[25] = z.2;

        // lane 8
        let z = self.bf9.bf3.butterfly3(
            s0.8,
            c_mul_fast(s1.8, self.twiddle8),
            c_mul_fast(s2.8, self.twiddle12),
        );
        chunk[8] = z.0;
        chunk[17] = z.1;
        chunk[26] = z.2;
    }
}

boring_scalar_butterfly!(Butterfly27, 27);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};

    test_butterfly!(test_butterfly27, f32, Butterfly27, 27, 1e-5);
    test_oof_butterfly!(test_oof_butterfly27, f32, Butterfly27, 27, 1e-5);
}
