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
use crate::FftDirection;
use crate::butterflies::short_butterflies::FastButterfly3;
use crate::complex_fma::c_mul_fast;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct FastButterfly9<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle4: Complex<T>,
    pub(crate) bf3: FastButterfly3<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> FastButterfly9<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        FastButterfly9 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 9, fft_direction),
            twiddle2: compute_twiddle(2, 9, fft_direction),
            twiddle4: compute_twiddle(4, 9, fft_direction),
            bf3: FastButterfly3::new(fft_direction),
        }
    }
}

impl<
    T: Copy
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Num
        + 'static
        + Neg<Output = T>
        + MulAdd<T, Output = T>
        + Float
        + Default,
> FastButterfly9<T>
where
    f64: AsPrimitive<T>,
{
    #[inline]
    pub(crate) fn exec(
        &self,
        u0: Complex<T>,
        u1: Complex<T>,
        u2: Complex<T>,
        u3: Complex<T>,
        u4: Complex<T>,
        u5: Complex<T>,
        u6: Complex<T>,
        u7: Complex<T>,
        u8: Complex<T>,
    ) -> (
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
    ) {
        let (u0, u3, u6) = self.bf3.butterfly3(u0, u3, u6);
        let (u1, mut u4, mut u7) = self.bf3.butterfly3(u1, u4, u7);
        let (u2, mut u5, mut u8) = self.bf3.butterfly3(u2, u5, u8);

        u4 = c_mul_fast(u4, self.twiddle1);
        u7 = c_mul_fast(u7, self.twiddle2);
        u5 = c_mul_fast(u5, self.twiddle2);
        u8 = c_mul_fast(u8, self.twiddle4);

        let (zu0, zu3, zu6) = self.bf3.butterfly3(u0, u1, u2);
        let (zu1, zu4, zu7) = self.bf3.butterfly3(u3, u4, u5);
        let (zu2, zu5, zu8) = self.bf3.butterfly3(u6, u7, u8);

        (zu0, zu1, zu2, zu3, zu4, zu5, zu6, zu7, zu8)
    }
}
