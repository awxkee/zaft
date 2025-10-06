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
use crate::butterflies::rotate_90;
use crate::butterflies::short_butterflies::{FastButterfly2, FastButterfly4};
use crate::traits::FftTrigonometry;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct FastButterfly8<T> {
    direction: FftDirection,
    root2: T,
    bf4: FastButterfly4<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> FastButterfly8<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        FastButterfly8 {
            direction: fft_direction,
            root2: (0.5f64.sqrt()).as_(),
            bf4: FastButterfly4::new(fft_direction),
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
> FastButterfly8<T>
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
    ) -> (
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
        Complex<T>,
    ) {
        let bf2 = FastButterfly2::new(self.direction);

        let (u0, u2, u4, u6) = self.bf4.butterfly4(u0, u2, u4, u6);
        let (u1, mut u3, mut u5, mut u7) = self.bf4.butterfly4(u1, u3, u5, u7);

        u3 = (rotate_90(u3, self.direction) + u3) * self.root2;
        u5 = rotate_90(u5, self.direction);
        u7 = (rotate_90(u7, self.direction) - u7) * self.root2;

        let (u0, u1) = bf2.butterfly2(u0, u1);
        let (u2, u3) = bf2.butterfly2(u2, u3);
        let (u4, u5) = bf2.butterfly2(u4, u5);
        let (u6, u7) = bf2.butterfly2(u6, u7);

        (u0, u2, u4, u6, u1, u3, u5, u7)
    }
}
