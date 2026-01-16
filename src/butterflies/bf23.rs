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

use crate::mla::fmla;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::AsPrimitive;

#[allow(unused)]
pub(crate) struct Butterfly23<T> {
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
}

#[allow(unused)]
impl<T: FftSample> Butterfly23<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly23 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 23, fft_direction),
            twiddle2: compute_twiddle(2, 23, fft_direction),
            twiddle3: compute_twiddle(3, 23, fft_direction),
            twiddle4: compute_twiddle(4, 23, fft_direction),
            twiddle5: compute_twiddle(5, 23, fft_direction),
            twiddle6: compute_twiddle(6, 23, fft_direction),
            twiddle7: compute_twiddle(7, 23, fft_direction),
            twiddle8: compute_twiddle(8, 23, fft_direction),
            twiddle9: compute_twiddle(9, 23, fft_direction),
            twiddle10: compute_twiddle(10, 23, fft_direction),
            twiddle11: compute_twiddle(11, 23, fft_direction),
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Butterfly23<T>
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

        for chunk in in_place.chunks_exact_mut(23) {
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

            let x122p = u1 + u22;
            let x122n = u1 - u22;
            let x221p = u2 + u21;
            let x221n = u2 - u21;
            let x320p = u3 + u20;
            let x320n = u3 - u20;
            let x419p = u4 + u19;
            let x419n = u4 - u19;
            let x518p = u5 + u18;
            let x518n = u5 - u18;
            let x617p = u6 + u17;
            let x617n = u6 - u17;
            let x716p = u7 + u16;
            let x716n = u7 - u16;
            let x815p = u8 + u15;
            let x815n = u8 - u15;
            let x914p = u9 + u14;
            let x914n = u9 - u14;
            let x1013p = u10 + u13;
            let x1013n = u10 - u13;
            let x1112p = u11 + u12;
            let x1112n = u11 - u12;
            let sum = u0
                + x122p
                + x221p
                + x320p
                + x419p
                + x518p
                + x617p
                + x716p
                + x815p
                + x914p
                + x1013p
                + x1112p;
            chunk[0] = sum;
            let b122re_a = fmla(self.twiddle1.re, x122p.re, u0.re)
                + fmla(self.twiddle2.re, x221p.re, self.twiddle3.re * x320p.re)
                + fmla(self.twiddle4.re, x419p.re, self.twiddle5.re * x518p.re)
                + self.twiddle6.re * x617p.re
                + self.twiddle7.re * x716p.re
                + self.twiddle8.re * x815p.re
                + self.twiddle9.re * x914p.re
                + self.twiddle10.re * x1013p.re
                + self.twiddle11.re * x1112p.re;
            let b122re_b = fmla(self.twiddle1.im, x122n.im, self.twiddle2.im * x221n.im)
                + self.twiddle3.im * x320n.im
                + self.twiddle4.im * x419n.im
                + self.twiddle5.im * x518n.im
                + self.twiddle6.im * x617n.im
                + self.twiddle7.im * x716n.im
                + self.twiddle8.im * x815n.im
                + self.twiddle9.im * x914n.im
                + self.twiddle10.im * x1013n.im
                + self.twiddle11.im * x1112n.im;
            let b221re_a = fmla(self.twiddle2.re, x122p.re, u0.re)
                + fmla(self.twiddle4.re, x221p.re, self.twiddle6.re * x320p.re)
                + self.twiddle8.re * x419p.re
                + self.twiddle10.re * x518p.re
                + self.twiddle11.re * x617p.re
                + self.twiddle9.re * x716p.re
                + self.twiddle7.re * x815p.re
                + self.twiddle5.re * x914p.re
                + self.twiddle3.re * x1013p.re
                + self.twiddle1.re * x1112p.re;
            let b221re_b = fmla(self.twiddle2.im, x122n.im, self.twiddle4.im * x221n.im)
                + fmla(self.twiddle6.im, x320n.im, self.twiddle8.im * x419n.im)
                + self.twiddle10.im * x518n.im
                + -self.twiddle11.im * x617n.im
                + -self.twiddle9.im * x716n.im
                + -self.twiddle7.im * x815n.im
                + -self.twiddle5.im * x914n.im
                + -self.twiddle3.im * x1013n.im
                + -self.twiddle1.im * x1112n.im;
            let b320re_a = fmla(self.twiddle3.re, x122p.re, u0.re)
                + fmla(self.twiddle6.re, x221p.re, self.twiddle9.re * x320p.re)
                + self.twiddle11.re * x419p.re
                + self.twiddle8.re * x518p.re
                + self.twiddle5.re * x617p.re
                + self.twiddle2.re * x716p.re
                + self.twiddle1.re * x815p.re
                + self.twiddle4.re * x914p.re
                + self.twiddle7.re * x1013p.re
                + self.twiddle10.re * x1112p.re;
            let b320re_b = fmla(self.twiddle3.im, x122n.im, self.twiddle6.im * x221n.im)
                + fmla(self.twiddle9.im, x320n.im, -self.twiddle11.im * x419n.im)
                + -self.twiddle8.im * x518n.im
                + -self.twiddle5.im * x617n.im
                + -self.twiddle2.im * x716n.im
                + self.twiddle1.im * x815n.im
                + self.twiddle4.im * x914n.im
                + self.twiddle7.im * x1013n.im
                + self.twiddle10.im * x1112n.im;
            let b419re_a = fmla(self.twiddle4.re, x122p.re, u0.re)
                + fmla(self.twiddle8.re, x221p.re, self.twiddle11.re * x320p.re)
                + self.twiddle7.re * x419p.re
                + self.twiddle3.re * x518p.re
                + self.twiddle1.re * x617p.re
                + self.twiddle5.re * x716p.re
                + self.twiddle9.re * x815p.re
                + self.twiddle10.re * x914p.re
                + self.twiddle6.re * x1013p.re
                + self.twiddle2.re * x1112p.re;
            let b419re_b = fmla(self.twiddle4.im, x122n.im, self.twiddle8.im * x221n.im)
                + fmla(-self.twiddle11.im, x320n.im, -self.twiddle7.im * x419n.im)
                + -self.twiddle3.im * x518n.im
                + self.twiddle1.im * x617n.im
                + self.twiddle5.im * x716n.im
                + self.twiddle9.im * x815n.im
                + -self.twiddle10.im * x914n.im
                + -self.twiddle6.im * x1013n.im
                + -self.twiddle2.im * x1112n.im;
            let b518re_a = fmla(self.twiddle5.re, x122p.re, u0.re)
                + fmla(self.twiddle10.re, x221p.re, self.twiddle8.re * x320p.re)
                + self.twiddle3.re * x419p.re
                + self.twiddle2.re * x518p.re
                + self.twiddle7.re * x617p.re
                + self.twiddle11.re * x716p.re
                + self.twiddle6.re * x815p.re
                + self.twiddle1.re * x914p.re
                + self.twiddle4.re * x1013p.re
                + self.twiddle9.re * x1112p.re;
            let b518re_b = fmla(self.twiddle5.im, x122n.im, self.twiddle10.im * x221n.im)
                + fmla(-self.twiddle8.im, x320n.im, -self.twiddle3.im * x419n.im)
                + self.twiddle2.im * x518n.im
                + self.twiddle7.im * x617n.im
                + -self.twiddle11.im * x716n.im
                + -self.twiddle6.im * x815n.im
                + -self.twiddle1.im * x914n.im
                + self.twiddle4.im * x1013n.im
                + self.twiddle9.im * x1112n.im;
            let b617re_a = fmla(self.twiddle6.re, x122p.re, u0.re)
                + fmla(self.twiddle11.re, x221p.re, self.twiddle5.re * x320p.re)
                + self.twiddle1.re * x419p.re
                + self.twiddle7.re * x518p.re
                + self.twiddle10.re * x617p.re
                + self.twiddle4.re * x716p.re
                + self.twiddle2.re * x815p.re
                + self.twiddle8.re * x914p.re
                + self.twiddle9.re * x1013p.re
                + self.twiddle3.re * x1112p.re;
            let b617re_b = fmla(self.twiddle6.im, x122n.im, -self.twiddle11.im * x221n.im)
                + fmla(-self.twiddle5.im, x320n.im, self.twiddle1.im * x419n.im)
                + self.twiddle7.im * x518n.im
                + -self.twiddle10.im * x617n.im
                + -self.twiddle4.im * x716n.im
                + self.twiddle2.im * x815n.im
                + self.twiddle8.im * x914n.im
                + -self.twiddle9.im * x1013n.im
                + -self.twiddle3.im * x1112n.im;
            let b716re_a = fmla(self.twiddle7.re, x122p.re, u0.re)
                + fmla(self.twiddle9.re, x221p.re, self.twiddle2.re * x320p.re)
                + fmla(self.twiddle5.re, x419p.re, self.twiddle11.re * x518p.re)
                + self.twiddle4.re * x617p.re
                + self.twiddle3.re * x716p.re
                + self.twiddle10.re * x815p.re
                + self.twiddle6.re * x914p.re
                + self.twiddle1.re * x1013p.re
                + self.twiddle8.re * x1112p.re;
            let b716re_b = fmla(self.twiddle7.im, x122n.im, -self.twiddle9.im * x221n.im)
                + fmla(-self.twiddle2.im, x320n.im, self.twiddle5.im * x419n.im)
                + -self.twiddle11.im * x518n.im
                + -self.twiddle4.im * x617n.im
                + self.twiddle3.im * x716n.im
                + self.twiddle10.im * x815n.im
                + -self.twiddle6.im * x914n.im
                + self.twiddle1.im * x1013n.im
                + self.twiddle8.im * x1112n.im;
            let b815re_a = fmla(self.twiddle8.re, x122p.re, u0.re)
                + fmla(self.twiddle7.re, x221p.re, self.twiddle1.re * x320p.re)
                + self.twiddle9.re * x419p.re
                + self.twiddle6.re * x518p.re
                + self.twiddle2.re * x617p.re
                + self.twiddle10.re * x716p.re
                + self.twiddle5.re * x815p.re
                + self.twiddle3.re * x914p.re
                + self.twiddle11.re * x1013p.re
                + self.twiddle4.re * x1112p.re;
            let b815re_b = fmla(self.twiddle8.im, x122n.im, -self.twiddle7.im * x221n.im)
                + self.twiddle1.im * x320n.im
                + self.twiddle9.im * x419n.im
                + -self.twiddle6.im * x518n.im
                + self.twiddle2.im * x617n.im
                + self.twiddle10.im * x716n.im
                + -self.twiddle5.im * x815n.im
                + self.twiddle3.im * x914n.im
                + self.twiddle11.im * x1013n.im
                + -self.twiddle4.im * x1112n.im;
            let b914re_a = fmla(self.twiddle9.re, x122p.re, u0.re)
                + fmla(self.twiddle5.re, x221p.re, self.twiddle4.re * x320p.re)
                + self.twiddle10.re * x419p.re
                + self.twiddle1.re * x518p.re
                + self.twiddle8.re * x617p.re
                + self.twiddle6.re * x716p.re
                + self.twiddle3.re * x815p.re
                + self.twiddle11.re * x914p.re
                + self.twiddle2.re * x1013p.re
                + self.twiddle7.re * x1112p.re;
            let b914re_b = fmla(self.twiddle9.im, x122n.im, -self.twiddle5.im * x221n.im)
                + self.twiddle4.im * x320n.im
                + -self.twiddle10.im * x419n.im
                + -self.twiddle1.im * x518n.im
                + self.twiddle8.im * x617n.im
                + -self.twiddle6.im * x716n.im
                + self.twiddle3.im * x815n.im
                + -self.twiddle11.im * x914n.im
                + -self.twiddle2.im * x1013n.im
                + self.twiddle7.im * x1112n.im;
            let b1013re_a = fmla(self.twiddle10.re, x122p.re, u0.re)
                + self.twiddle3.re * x221p.re
                + self.twiddle7.re * x320p.re
                + self.twiddle6.re * x419p.re
                + self.twiddle4.re * x518p.re
                + self.twiddle9.re * x617p.re
                + self.twiddle1.re * x716p.re
                + self.twiddle11.re * x815p.re
                + self.twiddle2.re * x914p.re
                + self.twiddle8.re * x1013p.re
                + self.twiddle5.re * x1112p.re;
            let b1013re_b = fmla(self.twiddle10.im, x122n.im, -self.twiddle3.im * x221n.im)
                + self.twiddle7.im * x320n.im
                + -self.twiddle6.im * x419n.im
                + self.twiddle4.im * x518n.im
                + -self.twiddle9.im * x617n.im
                + self.twiddle1.im * x716n.im
                + self.twiddle11.im * x815n.im
                + -self.twiddle2.im * x914n.im
                + self.twiddle8.im * x1013n.im
                + -self.twiddle5.im * x1112n.im;
            let b1112re_a = fmla(self.twiddle11.re, x122p.re, u0.re)
                + self.twiddle1.re * x221p.re
                + self.twiddle10.re * x320p.re
                + self.twiddle2.re * x419p.re
                + self.twiddle9.re * x518p.re
                + self.twiddle3.re * x617p.re
                + self.twiddle8.re * x716p.re
                + self.twiddle4.re * x815p.re
                + self.twiddle7.re * x914p.re
                + self.twiddle5.re * x1013p.re
                + self.twiddle6.re * x1112p.re;
            let b1112re_b = fmla(self.twiddle11.im, x122n.im, -self.twiddle1.im * x221n.im)
                + self.twiddle10.im * x320n.im
                + -self.twiddle2.im * x419n.im
                + self.twiddle9.im * x518n.im
                + -self.twiddle3.im * x617n.im
                + self.twiddle8.im * x716n.im
                + -self.twiddle4.im * x815n.im
                + self.twiddle7.im * x914n.im
                + -self.twiddle5.im * x1013n.im
                + self.twiddle6.im * x1112n.im;

            let b122im_a = fmla(self.twiddle1.re, x122p.im, u0.im)
                + fmla(self.twiddle2.re, x221p.im, self.twiddle3.re * x320p.im)
                + self.twiddle4.re * x419p.im
                + self.twiddle5.re * x518p.im
                + self.twiddle6.re * x617p.im
                + self.twiddle7.re * x716p.im
                + self.twiddle8.re * x815p.im
                + self.twiddle9.re * x914p.im
                + self.twiddle10.re * x1013p.im
                + self.twiddle11.re * x1112p.im;
            let b122im_b = fmla(self.twiddle1.im, x122n.re, self.twiddle2.im * x221n.re)
                + self.twiddle3.im * x320n.re
                + self.twiddle4.im * x419n.re
                + self.twiddle5.im * x518n.re
                + self.twiddle6.im * x617n.re
                + self.twiddle7.im * x716n.re
                + self.twiddle8.im * x815n.re
                + self.twiddle9.im * x914n.re
                + self.twiddle10.im * x1013n.re
                + self.twiddle11.im * x1112n.re;
            let b221im_a = fmla(self.twiddle2.re, x122p.im, u0.im)
                + fmla(self.twiddle4.re, x221p.im, self.twiddle6.re * x320p.im)
                + self.twiddle8.re * x419p.im
                + self.twiddle10.re * x518p.im
                + self.twiddle11.re * x617p.im
                + self.twiddle9.re * x716p.im
                + self.twiddle7.re * x815p.im
                + self.twiddle5.re * x914p.im
                + self.twiddle3.re * x1013p.im
                + self.twiddle1.re * x1112p.im;
            let b221im_b = fmla(self.twiddle2.im, x122n.re, self.twiddle4.im * x221n.re)
                + fmla(self.twiddle6.im, x320n.re, self.twiddle8.im * x419n.re)
                + self.twiddle10.im * x518n.re
                + -self.twiddle11.im * x617n.re
                + -self.twiddle9.im * x716n.re
                + -self.twiddle7.im * x815n.re
                + -self.twiddle5.im * x914n.re
                + -self.twiddle3.im * x1013n.re
                + -self.twiddle1.im * x1112n.re;
            let b320im_a = fmla(self.twiddle3.re, x122p.im, u0.im)
                + self.twiddle6.re * x221p.im
                + self.twiddle9.re * x320p.im
                + self.twiddle11.re * x419p.im
                + self.twiddle8.re * x518p.im
                + self.twiddle5.re * x617p.im
                + self.twiddle2.re * x716p.im
                + self.twiddle1.re * x815p.im
                + self.twiddle4.re * x914p.im
                + self.twiddle7.re * x1013p.im
                + self.twiddle10.re * x1112p.im;
            let b320im_b = fmla(self.twiddle3.im, x122n.re, self.twiddle6.im * x221n.re)
                + fmla(self.twiddle9.im, x320n.re, -self.twiddle11.im * x419n.re)
                + -self.twiddle8.im * x518n.re
                + -self.twiddle5.im * x617n.re
                + -self.twiddle2.im * x716n.re
                + self.twiddle1.im * x815n.re
                + self.twiddle4.im * x914n.re
                + self.twiddle7.im * x1013n.re
                + self.twiddle10.im * x1112n.re;
            let b419im_a = fmla(self.twiddle4.re, x122p.im, u0.im)
                + fmla(self.twiddle8.re, x221p.im, self.twiddle11.re * x320p.im)
                + self.twiddle7.re * x419p.im
                + self.twiddle3.re * x518p.im
                + self.twiddle1.re * x617p.im
                + self.twiddle5.re * x716p.im
                + self.twiddle9.re * x815p.im
                + self.twiddle10.re * x914p.im
                + self.twiddle6.re * x1013p.im
                + self.twiddle2.re * x1112p.im;
            let b419im_b = self.twiddle4.im * x122n.re
                + fmla(self.twiddle8.im, x221n.re, -self.twiddle11.im * x320n.re)
                + fmla(-self.twiddle7.im, x419n.re, -self.twiddle3.im * x518n.re)
                + self.twiddle1.im * x617n.re
                + self.twiddle5.im * x716n.re
                + self.twiddle9.im * x815n.re
                + -self.twiddle10.im * x914n.re
                + -self.twiddle6.im * x1013n.re
                + -self.twiddle2.im * x1112n.re;
            let b518im_a = fmla(self.twiddle5.re, x122p.im, u0.im)
                + fmla(self.twiddle10.re, x221p.im, self.twiddle8.re * x320p.im)
                + self.twiddle3.re * x419p.im
                + self.twiddle2.re * x518p.im
                + self.twiddle7.re * x617p.im
                + self.twiddle11.re * x716p.im
                + self.twiddle6.re * x815p.im
                + self.twiddle1.re * x914p.im
                + self.twiddle4.re * x1013p.im
                + self.twiddle9.re * x1112p.im;
            let b518im_b = fmla(self.twiddle5.im, x122n.re, self.twiddle10.im * x221n.re)
                + fmla(-self.twiddle8.im, x320n.re, -self.twiddle3.im * x419n.re)
                + self.twiddle2.im * x518n.re
                + self.twiddle7.im * x617n.re
                + -self.twiddle11.im * x716n.re
                + -self.twiddle6.im * x815n.re
                + -self.twiddle1.im * x914n.re
                + self.twiddle4.im * x1013n.re
                + self.twiddle9.im * x1112n.re;
            let b617im_a = fmla(self.twiddle6.re, x122p.im, u0.im)
                + fmla(self.twiddle11.re, x221p.im, self.twiddle5.re * x320p.im)
                + self.twiddle1.re * x419p.im
                + self.twiddle7.re * x518p.im
                + self.twiddle10.re * x617p.im
                + self.twiddle4.re * x716p.im
                + self.twiddle2.re * x815p.im
                + self.twiddle8.re * x914p.im
                + self.twiddle9.re * x1013p.im
                + self.twiddle3.re * x1112p.im;
            let b617im_b = fmla(self.twiddle6.im, x122n.re, -self.twiddle11.im * x221n.re)
                + fmla(-self.twiddle5.im, x320n.re, self.twiddle1.im * x419n.re)
                + self.twiddle7.im * x518n.re
                + -self.twiddle10.im * x617n.re
                + -self.twiddle4.im * x716n.re
                + self.twiddle2.im * x815n.re
                + self.twiddle8.im * x914n.re
                + -self.twiddle9.im * x1013n.re
                + -self.twiddle3.im * x1112n.re;
            let b716im_a = fmla(self.twiddle7.re, x122p.im, u0.im)
                + fmla(self.twiddle9.re, x221p.im, self.twiddle2.re * x320p.im)
                + fmla(self.twiddle5.re, x419p.im, self.twiddle11.re * x518p.im)
                + self.twiddle4.re * x617p.im
                + self.twiddle3.re * x716p.im
                + self.twiddle10.re * x815p.im
                + self.twiddle6.re * x914p.im
                + self.twiddle1.re * x1013p.im
                + self.twiddle8.re * x1112p.im;
            let b716im_b = fmla(self.twiddle7.im, x122n.re, -self.twiddle9.im * x221n.re)
                + fmla(-self.twiddle2.im, x320n.re, self.twiddle5.im * x419n.re)
                + -self.twiddle11.im * x518n.re
                + -self.twiddle4.im * x617n.re
                + self.twiddle3.im * x716n.re
                + self.twiddle10.im * x815n.re
                + -self.twiddle6.im * x914n.re
                + self.twiddle1.im * x1013n.re
                + self.twiddle8.im * x1112n.re;
            let b815im_a = fmla(self.twiddle8.re, x122p.im, u0.im)
                + fmla(self.twiddle7.re, x221p.im, self.twiddle1.re * x320p.im)
                + self.twiddle9.re * x419p.im
                + self.twiddle6.re * x518p.im
                + self.twiddle2.re * x617p.im
                + self.twiddle10.re * x716p.im
                + self.twiddle5.re * x815p.im
                + self.twiddle3.re * x914p.im
                + self.twiddle11.re * x1013p.im
                + self.twiddle4.re * x1112p.im;
            let b815im_b = fmla(self.twiddle8.im, x122n.re, -self.twiddle7.im * x221n.re)
                + fmla(self.twiddle1.im, x320n.re, self.twiddle9.im * x419n.re)
                + -self.twiddle6.im * x518n.re
                + self.twiddle2.im * x617n.re
                + self.twiddle10.im * x716n.re
                + -self.twiddle5.im * x815n.re
                + self.twiddle3.im * x914n.re
                + self.twiddle11.im * x1013n.re
                + -self.twiddle4.im * x1112n.re;
            let b914im_a = fmla(self.twiddle9.re, x122p.im, u0.im)
                + fmla(self.twiddle5.re, x221p.im, self.twiddle4.re * x320p.im)
                + self.twiddle10.re * x419p.im
                + self.twiddle1.re * x518p.im
                + self.twiddle8.re * x617p.im
                + self.twiddle6.re * x716p.im
                + self.twiddle3.re * x815p.im
                + self.twiddle11.re * x914p.im
                + self.twiddle2.re * x1013p.im
                + self.twiddle7.re * x1112p.im;
            let b914im_b = fmla(self.twiddle9.im, x122n.re, -self.twiddle5.im * x221n.re)
                + fmla(self.twiddle4.im, x320n.re, -self.twiddle10.im * x419n.re)
                + -self.twiddle1.im * x518n.re
                + self.twiddle8.im * x617n.re
                + -self.twiddle6.im * x716n.re
                + self.twiddle3.im * x815n.re
                + -self.twiddle11.im * x914n.re
                + -self.twiddle2.im * x1013n.re
                + self.twiddle7.im * x1112n.re;
            let b1013im_a = fmla(self.twiddle10.re, x122p.im, u0.im)
                + fmla(self.twiddle3.re, x221p.im, self.twiddle7.re * x320p.im)
                + fmla(self.twiddle6.re, x419p.im, self.twiddle4.re * x518p.im)
                + self.twiddle9.re * x617p.im
                + self.twiddle1.re * x716p.im
                + self.twiddle11.re * x815p.im
                + self.twiddle2.re * x914p.im
                + self.twiddle8.re * x1013p.im
                + self.twiddle5.re * x1112p.im;
            let b1013im_b = fmla(self.twiddle10.im, x122n.re, -self.twiddle3.im * x221n.re)
                + fmla(self.twiddle7.im, x320n.re, -self.twiddle6.im * x419n.re)
                + fmla(self.twiddle4.im, x518n.re, -self.twiddle9.im * x617n.re)
                + self.twiddle1.im * x716n.re
                + self.twiddle11.im * x815n.re
                + -self.twiddle2.im * x914n.re
                + self.twiddle8.im * x1013n.re
                + -self.twiddle5.im * x1112n.re;
            let b1112im_a = fmla(self.twiddle11.re, x122p.im, u0.im)
                + fmla(self.twiddle1.re, x221p.im, self.twiddle10.re * x320p.im)
                + fmla(self.twiddle2.re, x419p.im, self.twiddle9.re * x518p.im)
                + self.twiddle3.re * x617p.im
                + self.twiddle8.re * x716p.im
                + self.twiddle4.re * x815p.im
                + self.twiddle7.re * x914p.im
                + self.twiddle5.re * x1013p.im
                + self.twiddle6.re * x1112p.im;
            let b1112im_b = fmla(self.twiddle11.im, x122n.re, -self.twiddle1.im * x221n.re)
                + fmla(self.twiddle10.im, x320n.re, -self.twiddle2.im * x419n.re)
                + fmla(self.twiddle9.im, x518n.re, -self.twiddle3.im * x617n.re)
                + fmla(self.twiddle8.im, x716n.re, -self.twiddle4.im * x815n.re)
                + fmla(self.twiddle7.im, x914n.re, -self.twiddle5.im * x1013n.re)
                + self.twiddle6.im * x1112n.re;

            let out1re = b122re_a - b122re_b;
            let out1im = b122im_a + b122im_b;
            let out2re = b221re_a - b221re_b;
            let out2im = b221im_a + b221im_b;
            let out3re = b320re_a - b320re_b;
            let out3im = b320im_a + b320im_b;
            let out4re = b419re_a - b419re_b;
            let out4im = b419im_a + b419im_b;
            let out5re = b518re_a - b518re_b;
            let out5im = b518im_a + b518im_b;
            let out6re = b617re_a - b617re_b;
            let out6im = b617im_a + b617im_b;
            let out7re = b716re_a - b716re_b;
            let out7im = b716im_a + b716im_b;
            let out8re = b815re_a - b815re_b;
            let out8im = b815im_a + b815im_b;
            let out9re = b914re_a - b914re_b;
            let out9im = b914im_a + b914im_b;
            let out10re = b1013re_a - b1013re_b;
            let out10im = b1013im_a + b1013im_b;
            let out11re = b1112re_a - b1112re_b;
            let out11im = b1112im_a + b1112im_b;
            let out12re = b1112re_a + b1112re_b;
            let out12im = b1112im_a - b1112im_b;
            let out13re = b1013re_a + b1013re_b;
            let out13im = b1013im_a - b1013im_b;
            let out14re = b914re_a + b914re_b;
            let out14im = b914im_a - b914im_b;
            let out15re = b815re_a + b815re_b;
            let out15im = b815im_a - b815im_b;
            let out16re = b716re_a + b716re_b;
            let out16im = b716im_a - b716im_b;
            let out17re = b617re_a + b617re_b;
            let out17im = b617im_a - b617im_b;
            let out18re = b518re_a + b518re_b;
            let out18im = b518im_a - b518im_b;
            let out19re = b419re_a + b419re_b;
            let out19im = b419im_a - b419im_b;
            let out20re = b320re_a + b320re_b;
            let out20im = b320im_a - b320im_b;
            let out21re = b221re_a + b221re_b;
            let out21im = b221im_a - b221im_b;
            let out22re = b122re_a + b122re_b;
            let out22im = b122im_a - b122im_b;
            chunk[1] = Complex {
                re: out1re,
                im: out1im,
            };
            chunk[2] = Complex {
                re: out2re,
                im: out2im,
            };
            chunk[3] = Complex {
                re: out3re,
                im: out3im,
            };
            chunk[4] = Complex {
                re: out4re,
                im: out4im,
            };
            chunk[5] = Complex {
                re: out5re,
                im: out5im,
            };
            chunk[6] = Complex {
                re: out6re,
                im: out6im,
            };
            chunk[7] = Complex {
                re: out7re,
                im: out7im,
            };
            chunk[8] = Complex {
                re: out8re,
                im: out8im,
            };
            chunk[9] = Complex {
                re: out9re,
                im: out9im,
            };
            chunk[10] = Complex {
                re: out10re,
                im: out10im,
            };
            chunk[11] = Complex {
                re: out11re,
                im: out11im,
            };
            chunk[12] = Complex {
                re: out12re,
                im: out12im,
            };
            chunk[13] = Complex {
                re: out13re,
                im: out13im,
            };
            chunk[14] = Complex {
                re: out14re,
                im: out14im,
            };
            chunk[15] = Complex {
                re: out15re,
                im: out15im,
            };
            chunk[16] = Complex {
                re: out16re,
                im: out16im,
            };
            chunk[17] = Complex {
                re: out17re,
                im: out17im,
            };
            chunk[18] = Complex {
                re: out18re,
                im: out18im,
            };
            chunk[19] = Complex {
                re: out19re,
                im: out19im,
            };
            chunk[20] = Complex {
                re: out20re,
                im: out20im,
            };
            chunk[21] = Complex {
                re: out21re,
                im: out21im,
            };
            chunk[22] = Complex {
                re: out22re,
                im: out22im,
            };
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        23
    }
}

impl<T: FftSample> R2CFftExecutor<T> for Butterfly23<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !input.len().is_multiple_of(23) {
            return Err(ZaftError::InvalidSizeMultiplier(
                input.len(),
                self.real_length(),
            ));
        }
        if !output.len().is_multiple_of(12) {
            return Err(ZaftError::InvalidSizeMultiplier(
                input.len(),
                self.complex_length(),
            ));
        }

        for (input, complex) in input.chunks_exact(23).zip(output.chunks_exact_mut(12)) {
            let u0 = input[0];
            let u1 = input[1];
            let u2 = input[2];
            let u3 = input[3];

            let u4 = input[4];
            let u5 = input[5];
            let u6 = input[6];
            let u7 = input[7];

            let u8 = input[8];
            let u9 = input[9];
            let u10 = input[10];
            let u11 = input[11];
            let u12 = input[12];

            let u13 = input[13];
            let u14 = input[14];
            let u15 = input[15];
            let u16 = input[16];

            let u17 = input[17];
            let u18 = input[18];

            let u19 = input[19];
            let u20 = input[20];

            let u21 = input[21];
            let u22 = input[22];

            let x122p = u1 + u22;
            let x122n = u1 - u22;
            let x221p = u2 + u21;
            let x221n = u2 - u21;
            let x320p = u3 + u20;
            let x320n = u3 - u20;
            let x419p = u4 + u19;
            let x419n = u4 - u19;
            let x518p = u5 + u18;
            let x518n = u5 - u18;
            let x617p = u6 + u17;
            let x617n = u6 - u17;
            let x716p = u7 + u16;
            let x716n = u7 - u16;
            let x815p = u8 + u15;
            let x815n = u8 - u15;
            let x914p = u9 + u14;
            let x914n = u9 - u14;
            let x1013p = u10 + u13;
            let x1013n = u10 - u13;
            let x1112p = u11 + u12;
            let x1112n = u11 - u12;
            let sum = u0
                + x122p
                + x221p
                + x320p
                + x419p
                + x518p
                + x617p
                + x716p
                + x815p
                + x914p
                + x1013p
                + x1112p;
            complex[0] = Complex::new(sum, T::zero());
            let b122re_a = fmla(self.twiddle1.re, x122p, u0)
                + fmla(self.twiddle2.re, x221p, self.twiddle3.re * x320p)
                + fmla(self.twiddle4.re, x419p, self.twiddle5.re * x518p)
                + self.twiddle6.re * x617p
                + self.twiddle7.re * x716p
                + self.twiddle8.re * x815p
                + self.twiddle9.re * x914p
                + self.twiddle10.re * x1013p
                + self.twiddle11.re * x1112p;
            let b221re_a = fmla(self.twiddle2.re, x122p, u0)
                + fmla(self.twiddle4.re, x221p, self.twiddle6.re * x320p)
                + self.twiddle8.re * x419p
                + self.twiddle10.re * x518p
                + self.twiddle11.re * x617p
                + self.twiddle9.re * x716p
                + self.twiddle7.re * x815p
                + self.twiddle5.re * x914p
                + self.twiddle3.re * x1013p
                + self.twiddle1.re * x1112p;
            let b320re_a = fmla(self.twiddle3.re, x122p, u0)
                + fmla(self.twiddle6.re, x221p, self.twiddle9.re * x320p)
                + self.twiddle11.re * x419p
                + self.twiddle8.re * x518p
                + self.twiddle5.re * x617p
                + self.twiddle2.re * x716p
                + self.twiddle1.re * x815p
                + self.twiddle4.re * x914p
                + self.twiddle7.re * x1013p
                + self.twiddle10.re * x1112p;
            let b419re_a = fmla(self.twiddle4.re, x122p, u0)
                + fmla(self.twiddle8.re, x221p, self.twiddle11.re * x320p)
                + self.twiddle7.re * x419p
                + self.twiddle3.re * x518p
                + self.twiddle1.re * x617p
                + self.twiddle5.re * x716p
                + self.twiddle9.re * x815p
                + self.twiddle10.re * x914p
                + self.twiddle6.re * x1013p
                + self.twiddle2.re * x1112p;
            let b518re_a = fmla(self.twiddle5.re, x122p, u0)
                + fmla(self.twiddle10.re, x221p, self.twiddle8.re * x320p)
                + self.twiddle3.re * x419p
                + self.twiddle2.re * x518p
                + self.twiddle7.re * x617p
                + self.twiddle11.re * x716p
                + self.twiddle6.re * x815p
                + self.twiddle1.re * x914p
                + self.twiddle4.re * x1013p
                + self.twiddle9.re * x1112p;
            let b617re_a = fmla(self.twiddle6.re, x122p, u0)
                + fmla(self.twiddle11.re, x221p, self.twiddle5.re * x320p)
                + self.twiddle1.re * x419p
                + self.twiddle7.re * x518p
                + self.twiddle10.re * x617p
                + self.twiddle4.re * x716p
                + self.twiddle2.re * x815p
                + self.twiddle8.re * x914p
                + self.twiddle9.re * x1013p
                + self.twiddle3.re * x1112p;
            let b716re_a = fmla(self.twiddle7.re, x122p, u0)
                + fmla(self.twiddle9.re, x221p, self.twiddle2.re * x320p)
                + fmla(self.twiddle5.re, x419p, self.twiddle11.re * x518p)
                + self.twiddle4.re * x617p
                + self.twiddle3.re * x716p
                + self.twiddle10.re * x815p
                + self.twiddle6.re * x914p
                + self.twiddle1.re * x1013p
                + self.twiddle8.re * x1112p;
            let b815re_a = fmla(self.twiddle8.re, x122p, u0)
                + fmla(self.twiddle7.re, x221p, self.twiddle1.re * x320p)
                + self.twiddle9.re * x419p
                + self.twiddle6.re * x518p
                + self.twiddle2.re * x617p
                + self.twiddle10.re * x716p
                + self.twiddle5.re * x815p
                + self.twiddle3.re * x914p
                + self.twiddle11.re * x1013p
                + self.twiddle4.re * x1112p;
            let b914re_a = fmla(self.twiddle9.re, x122p, u0)
                + fmla(self.twiddle5.re, x221p, self.twiddle4.re * x320p)
                + self.twiddle10.re * x419p
                + self.twiddle1.re * x518p
                + self.twiddle8.re * x617p
                + self.twiddle6.re * x716p
                + self.twiddle3.re * x815p
                + self.twiddle11.re * x914p
                + self.twiddle2.re * x1013p
                + self.twiddle7.re * x1112p;
            let b1013re_a = fmla(self.twiddle10.re, x122p, u0)
                + self.twiddle3.re * x221p
                + self.twiddle7.re * x320p
                + self.twiddle6.re * x419p
                + self.twiddle4.re * x518p
                + self.twiddle9.re * x617p
                + self.twiddle1.re * x716p
                + self.twiddle11.re * x815p
                + self.twiddle2.re * x914p
                + self.twiddle8.re * x1013p
                + self.twiddle5.re * x1112p;
            let b1112re_a = fmla(self.twiddle11.re, x122p, u0)
                + self.twiddle1.re * x221p
                + self.twiddle10.re * x320p
                + self.twiddle2.re * x419p
                + self.twiddle9.re * x518p
                + self.twiddle3.re * x617p
                + self.twiddle8.re * x716p
                + self.twiddle4.re * x815p
                + self.twiddle7.re * x914p
                + self.twiddle5.re * x1013p
                + self.twiddle6.re * x1112p;

            let b122im_b = fmla(self.twiddle1.im, x122n, self.twiddle2.im * x221n)
                + self.twiddle3.im * x320n
                + self.twiddle4.im * x419n
                + self.twiddle5.im * x518n
                + self.twiddle6.im * x617n
                + self.twiddle7.im * x716n
                + self.twiddle8.im * x815n
                + self.twiddle9.im * x914n
                + self.twiddle10.im * x1013n
                + self.twiddle11.im * x1112n;
            let b221im_b = fmla(self.twiddle2.im, x122n, self.twiddle4.im * x221n)
                + fmla(self.twiddle6.im, x320n, self.twiddle8.im * x419n)
                + self.twiddle10.im * x518n
                + -self.twiddle11.im * x617n
                + -self.twiddle9.im * x716n
                + -self.twiddle7.im * x815n
                + -self.twiddle5.im * x914n
                + -self.twiddle3.im * x1013n
                + -self.twiddle1.im * x1112n;
            let b320im_b = fmla(self.twiddle3.im, x122n, self.twiddle6.im * x221n)
                + fmla(self.twiddle9.im, x320n, -self.twiddle11.im * x419n)
                + -self.twiddle8.im * x518n
                + -self.twiddle5.im * x617n
                + -self.twiddle2.im * x716n
                + self.twiddle1.im * x815n
                + self.twiddle4.im * x914n
                + self.twiddle7.im * x1013n
                + self.twiddle10.im * x1112n;
            let b419im_b = self.twiddle4.im * x122n
                + fmla(self.twiddle8.im, x221n, -self.twiddle11.im * x320n)
                + fmla(-self.twiddle7.im, x419n, -self.twiddle3.im * x518n)
                + self.twiddle1.im * x617n
                + self.twiddle5.im * x716n
                + self.twiddle9.im * x815n
                + -self.twiddle10.im * x914n
                + -self.twiddle6.im * x1013n
                + -self.twiddle2.im * x1112n;
            let b518im_b = fmla(self.twiddle5.im, x122n, self.twiddle10.im * x221n)
                + fmla(-self.twiddle8.im, x320n, -self.twiddle3.im * x419n)
                + self.twiddle2.im * x518n
                + self.twiddle7.im * x617n
                + -self.twiddle11.im * x716n
                + -self.twiddle6.im * x815n
                + -self.twiddle1.im * x914n
                + self.twiddle4.im * x1013n
                + self.twiddle9.im * x1112n;
            let b617im_b = fmla(self.twiddle6.im, x122n, -self.twiddle11.im * x221n)
                + fmla(-self.twiddle5.im, x320n, self.twiddle1.im * x419n)
                + self.twiddle7.im * x518n
                + -self.twiddle10.im * x617n
                + -self.twiddle4.im * x716n
                + self.twiddle2.im * x815n
                + self.twiddle8.im * x914n
                + -self.twiddle9.im * x1013n
                + -self.twiddle3.im * x1112n;
            let b716im_b = fmla(self.twiddle7.im, x122n, -self.twiddle9.im * x221n)
                + fmla(-self.twiddle2.im, x320n, self.twiddle5.im * x419n)
                + -self.twiddle11.im * x518n
                + -self.twiddle4.im * x617n
                + self.twiddle3.im * x716n
                + self.twiddle10.im * x815n
                + -self.twiddle6.im * x914n
                + self.twiddle1.im * x1013n
                + self.twiddle8.im * x1112n;
            let b815im_b = fmla(self.twiddle8.im, x122n, -self.twiddle7.im * x221n)
                + fmla(self.twiddle1.im, x320n, self.twiddle9.im * x419n)
                + -self.twiddle6.im * x518n
                + self.twiddle2.im * x617n
                + self.twiddle10.im * x716n
                + -self.twiddle5.im * x815n
                + self.twiddle3.im * x914n
                + self.twiddle11.im * x1013n
                + -self.twiddle4.im * x1112n;
            let b914im_b = fmla(self.twiddle9.im, x122n, -self.twiddle5.im * x221n)
                + fmla(self.twiddle4.im, x320n, -self.twiddle10.im * x419n)
                + -self.twiddle1.im * x518n
                + self.twiddle8.im * x617n
                + -self.twiddle6.im * x716n
                + self.twiddle3.im * x815n
                + -self.twiddle11.im * x914n
                + -self.twiddle2.im * x1013n
                + self.twiddle7.im * x1112n;
            let b1013im_b = fmla(self.twiddle10.im, x122n, -self.twiddle3.im * x221n)
                + fmla(self.twiddle7.im, x320n, -self.twiddle6.im * x419n)
                + fmla(self.twiddle4.im, x518n, -self.twiddle9.im * x617n)
                + self.twiddle1.im * x716n
                + self.twiddle11.im * x815n
                + -self.twiddle2.im * x914n
                + self.twiddle8.im * x1013n
                + -self.twiddle5.im * x1112n;
            let b1112im_b = fmla(self.twiddle11.im, x122n, -self.twiddle1.im * x221n)
                + fmla(self.twiddle10.im, x320n, -self.twiddle2.im * x419n)
                + fmla(self.twiddle9.im, x518n, -self.twiddle3.im * x617n)
                + fmla(self.twiddle8.im, x716n, -self.twiddle4.im * x815n)
                + fmla(self.twiddle7.im, x914n, -self.twiddle5.im * x1013n)
                + self.twiddle6.im * x1112n;

            let out1re = b122re_a;
            let out1im = b122im_b;
            let out2re = b221re_a;
            let out2im = b221im_b;
            let out3re = b320re_a;
            let out3im = b320im_b;
            let out4re = b419re_a;
            let out4im = b419im_b;
            let out5re = b518re_a;
            let out5im = b518im_b;
            let out6re = b617re_a;
            let out6im = b617im_b;
            let out7re = b716re_a;
            let out7im = b716im_b;
            let out8re = b815re_a;
            let out8im = b815im_b;
            let out9re = b914re_a;
            let out9im = b914im_b;
            let out10re = b1013re_a;
            let out10im = b1013im_b;
            let out11re = b1112re_a;
            let out11im = b1112im_b;
            complex[1] = Complex {
                re: out1re,
                im: out1im,
            };
            complex[2] = Complex {
                re: out2re,
                im: out2im,
            };
            complex[3] = Complex {
                re: out3re,
                im: out3im,
            };
            complex[4] = Complex {
                re: out4re,
                im: out4im,
            };
            complex[5] = Complex {
                re: out5re,
                im: out5im,
            };
            complex[6] = Complex {
                re: out6re,
                im: out6im,
            };
            complex[7] = Complex {
                re: out7re,
                im: out7im,
            };
            complex[8] = Complex {
                re: out8re,
                im: out8im,
            };
            complex[9] = Complex {
                re: out9re,
                im: out9im,
            };
            complex[10] = Complex {
                re: out10re,
                im: out10im,
            };
            complex[11] = Complex {
                re: out11re,
                im: out11im,
            };
        }
        Ok(())
    }

    fn real_length(&self) -> usize {
        23
    }

    fn complex_length(&self) -> usize {
        12
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    use crate::r2c::test_r2c_butterfly;

    test_r2c_butterfly!(test_r2c_butterfly23, f32, Butterfly23, 23, 1e-5);
    test_butterfly!(test_butterfly23, f32, Butterfly23, 23, 1e-5);
}
