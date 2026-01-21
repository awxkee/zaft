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
pub(crate) struct Butterfly29<T> {
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
    twiddle13: Complex<T>,
    twiddle14: Complex<T>,
}

#[allow(unused)]
impl<T: FftSample> Butterfly29<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly29 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 29, fft_direction),
            twiddle2: compute_twiddle(2, 29, fft_direction),
            twiddle3: compute_twiddle(3, 29, fft_direction),
            twiddle4: compute_twiddle(4, 29, fft_direction),
            twiddle5: compute_twiddle(5, 29, fft_direction),
            twiddle6: compute_twiddle(6, 29, fft_direction),
            twiddle7: compute_twiddle(7, 29, fft_direction),
            twiddle8: compute_twiddle(8, 29, fft_direction),
            twiddle9: compute_twiddle(9, 29, fft_direction),
            twiddle10: compute_twiddle(10, 29, fft_direction),
            twiddle11: compute_twiddle(11, 29, fft_direction),
            twiddle12: compute_twiddle(12, 29, fft_direction),
            twiddle13: compute_twiddle(13, 29, fft_direction),
            twiddle14: compute_twiddle(14, 29, fft_direction),
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Butterfly29<T>
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

        for chunk in in_place.chunks_exact_mut(29) {
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

            let u25 = chunk[25];
            let u26 = chunk[26];

            let u27 = chunk[27];
            let u28 = chunk[28];

            let x128p = u1 + u28;
            let x128n = u1 - u28;
            let x227p = u2 + u27;
            let x227n = u2 - u27;
            let x326p = u3 + u26;
            let x326n = u3 - u26;
            let x425p = u4 + u25;
            let x425n = u4 - u25;
            let x524p = u5 + u24;
            let x524n = u5 - u24;
            let x623p = u6 + u23;
            let x623n = u6 - u23;
            let x722p = u7 + u22;
            let x722n = u7 - u22;
            let x821p = u8 + u21;
            let x821n = u8 - u21;
            let x920p = u9 + u20;
            let x920n = u9 - u20;
            let x1019p = u10 + u19;
            let x1019n = u10 - u19;
            let x1118p = u11 + u18;
            let x1118n = u11 - u18;
            let x1217p = u12 + u17;
            let x1217n = u12 - u17;
            let x1316p = u13 + u16;
            let x1316n = u13 - u16;
            let x1415p = u14 + u15;
            let x1415n = u14 - u15;
            let sum = u0
                + x128p
                + x227p
                + x326p
                + x425p
                + x524p
                + x623p
                + x722p
                + x821p
                + x920p
                + x1019p
                + x1118p
                + x1217p
                + x1316p
                + x1415p;
            chunk[0] = sum;
            let b128re_a = fmla(self.twiddle1.re, x128p.re, u0.re)
                + self.twiddle2.re * x227p.re
                + self.twiddle3.re * x326p.re
                + self.twiddle4.re * x425p.re
                + self.twiddle5.re * x524p.re
                + self.twiddle6.re * x623p.re
                + self.twiddle7.re * x722p.re
                + self.twiddle8.re * x821p.re
                + self.twiddle9.re * x920p.re
                + self.twiddle10.re * x1019p.re
                + self.twiddle11.re * x1118p.re
                + self.twiddle12.re * x1217p.re
                + self.twiddle13.re * x1316p.re
                + self.twiddle14.re * x1415p.re;
            let b128re_b = self.twiddle1.im * x128n.im
                + self.twiddle2.im * x227n.im
                + self.twiddle3.im * x326n.im
                + self.twiddle4.im * x425n.im
                + self.twiddle5.im * x524n.im
                + self.twiddle6.im * x623n.im
                + self.twiddle7.im * x722n.im
                + self.twiddle8.im * x821n.im
                + self.twiddle9.im * x920n.im
                + self.twiddle10.im * x1019n.im
                + self.twiddle11.im * x1118n.im
                + self.twiddle12.im * x1217n.im
                + self.twiddle13.im * x1316n.im
                + self.twiddle14.im * x1415n.im;
            let b227re_a = u0.re
                + self.twiddle2.re * x128p.re
                + self.twiddle4.re * x227p.re
                + self.twiddle6.re * x326p.re
                + self.twiddle8.re * x425p.re
                + self.twiddle10.re * x524p.re
                + self.twiddle12.re * x623p.re
                + self.twiddle14.re * x722p.re
                + self.twiddle13.re * x821p.re
                + self.twiddle11.re * x920p.re
                + self.twiddle9.re * x1019p.re
                + self.twiddle7.re * x1118p.re
                + self.twiddle5.re * x1217p.re
                + self.twiddle3.re * x1316p.re
                + self.twiddle1.re * x1415p.re;
            let b227re_b = self.twiddle2.im * x128n.im
                + self.twiddle4.im * x227n.im
                + self.twiddle6.im * x326n.im
                + self.twiddle8.im * x425n.im
                + self.twiddle10.im * x524n.im
                + self.twiddle12.im * x623n.im
                + self.twiddle14.im * x722n.im
                + -self.twiddle13.im * x821n.im
                + -self.twiddle11.im * x920n.im
                + -self.twiddle9.im * x1019n.im
                + -self.twiddle7.im * x1118n.im
                + -self.twiddle5.im * x1217n.im
                + -self.twiddle3.im * x1316n.im
                + -self.twiddle1.im * x1415n.im;
            let b326re_a = fmla(self.twiddle3.re, x128p.re, u0.re)
                + self.twiddle6.re * x227p.re
                + self.twiddle9.re * x326p.re
                + self.twiddle12.re * x425p.re
                + self.twiddle14.re * x524p.re
                + self.twiddle11.re * x623p.re
                + self.twiddle8.re * x722p.re
                + self.twiddle5.re * x821p.re
                + self.twiddle2.re * x920p.re
                + self.twiddle1.re * x1019p.re
                + self.twiddle4.re * x1118p.re
                + self.twiddle7.re * x1217p.re
                + self.twiddle10.re * x1316p.re
                + self.twiddle13.re * x1415p.re;
            let b326re_b = fmla(self.twiddle3.im, x128n.im, self.twiddle6.im * x227n.im)
                + self.twiddle9.im * x326n.im
                + self.twiddle12.im * x425n.im
                + -self.twiddle14.im * x524n.im
                + -self.twiddle11.im * x623n.im
                + -self.twiddle8.im * x722n.im
                + -self.twiddle5.im * x821n.im
                + -self.twiddle2.im * x920n.im
                + self.twiddle1.im * x1019n.im
                + self.twiddle4.im * x1118n.im
                + self.twiddle7.im * x1217n.im
                + self.twiddle10.im * x1316n.im
                + self.twiddle13.im * x1415n.im;
            let b425re_a = fmla(self.twiddle4.re, x128p.re, u0.re)
                + self.twiddle8.re * x227p.re
                + self.twiddle12.re * x326p.re
                + self.twiddle13.re * x425p.re
                + self.twiddle9.re * x524p.re
                + self.twiddle5.re * x623p.re
                + self.twiddle1.re * x722p.re
                + self.twiddle3.re * x821p.re
                + self.twiddle7.re * x920p.re
                + self.twiddle11.re * x1019p.re
                + self.twiddle14.re * x1118p.re
                + self.twiddle10.re * x1217p.re
                + self.twiddle6.re * x1316p.re
                + self.twiddle2.re * x1415p.re;
            let b425re_b = self.twiddle4.im * x128n.im
                + self.twiddle8.im * x227n.im
                + self.twiddle12.im * x326n.im
                + -self.twiddle13.im * x425n.im
                + -self.twiddle9.im * x524n.im
                + -self.twiddle5.im * x623n.im
                + -self.twiddle1.im * x722n.im
                + self.twiddle3.im * x821n.im
                + self.twiddle7.im * x920n.im
                + self.twiddle11.im * x1019n.im
                + -self.twiddle14.im * x1118n.im
                + -self.twiddle10.im * x1217n.im
                + -self.twiddle6.im * x1316n.im
                + -self.twiddle2.im * x1415n.im;
            let b524re_a = fmla(self.twiddle5.re, x128p.re, u0.re)
                + self.twiddle10.re * x227p.re
                + self.twiddle14.re * x326p.re
                + self.twiddle9.re * x425p.re
                + self.twiddle4.re * x524p.re
                + self.twiddle1.re * x623p.re
                + self.twiddle6.re * x722p.re
                + self.twiddle11.re * x821p.re
                + self.twiddle13.re * x920p.re
                + self.twiddle8.re * x1019p.re
                + self.twiddle3.re * x1118p.re
                + self.twiddle2.re * x1217p.re
                + self.twiddle7.re * x1316p.re
                + self.twiddle12.re * x1415p.re;
            let b524re_b = fmla(self.twiddle5.im, x128n.im, self.twiddle10.im * x227n.im)
                + -self.twiddle14.im * x326n.im
                + -self.twiddle9.im * x425n.im
                + -self.twiddle4.im * x524n.im
                + self.twiddle1.im * x623n.im
                + self.twiddle6.im * x722n.im
                + self.twiddle11.im * x821n.im
                + -self.twiddle13.im * x920n.im
                + -self.twiddle8.im * x1019n.im
                + -self.twiddle3.im * x1118n.im
                + self.twiddle2.im * x1217n.im
                + self.twiddle7.im * x1316n.im
                + self.twiddle12.im * x1415n.im;
            let b623re_a = fmla(self.twiddle6.re, x128p.re, u0.re)
                + self.twiddle12.re * x227p.re
                + self.twiddle11.re * x326p.re
                + self.twiddle5.re * x425p.re
                + self.twiddle1.re * x524p.re
                + self.twiddle7.re * x623p.re
                + self.twiddle13.re * x722p.re
                + self.twiddle10.re * x821p.re
                + self.twiddle4.re * x920p.re
                + self.twiddle2.re * x1019p.re
                + self.twiddle8.re * x1118p.re
                + self.twiddle14.re * x1217p.re
                + self.twiddle9.re * x1316p.re
                + self.twiddle3.re * x1415p.re;
            let b623re_b = self.twiddle6.im * x128n.im
                + self.twiddle12.im * x227n.im
                + -self.twiddle11.im * x326n.im
                + -self.twiddle5.im * x425n.im
                + self.twiddle1.im * x524n.im
                + self.twiddle7.im * x623n.im
                + self.twiddle13.im * x722n.im
                + -self.twiddle10.im * x821n.im
                + -self.twiddle4.im * x920n.im
                + self.twiddle2.im * x1019n.im
                + self.twiddle8.im * x1118n.im
                + self.twiddle14.im * x1217n.im
                + -self.twiddle9.im * x1316n.im
                + -self.twiddle3.im * x1415n.im;
            let b722re_a = fmla(self.twiddle7.re, x128p.re, u0.re)
                + self.twiddle14.re * x227p.re
                + self.twiddle8.re * x326p.re
                + self.twiddle1.re * x425p.re
                + self.twiddle6.re * x524p.re
                + self.twiddle13.re * x623p.re
                + self.twiddle9.re * x722p.re
                + self.twiddle2.re * x821p.re
                + self.twiddle5.re * x920p.re
                + self.twiddle12.re * x1019p.re
                + self.twiddle10.re * x1118p.re
                + self.twiddle3.re * x1217p.re
                + self.twiddle4.re * x1316p.re
                + self.twiddle11.re * x1415p.re;
            let b722re_b = self.twiddle7.im * x128n.im
                + self.twiddle14.im * x227n.im
                + -self.twiddle8.im * x326n.im
                + -self.twiddle1.im * x425n.im
                + self.twiddle6.im * x524n.im
                + self.twiddle13.im * x623n.im
                + -self.twiddle9.im * x722n.im
                + -self.twiddle2.im * x821n.im
                + self.twiddle5.im * x920n.im
                + self.twiddle12.im * x1019n.im
                + -self.twiddle10.im * x1118n.im
                + -self.twiddle3.im * x1217n.im
                + self.twiddle4.im * x1316n.im
                + self.twiddle11.im * x1415n.im;
            let b821re_a = fmla(self.twiddle8.re, x128p.re, u0.re)
                + self.twiddle13.re * x227p.re
                + self.twiddle5.re * x326p.re
                + self.twiddle3.re * x425p.re
                + self.twiddle11.re * x524p.re
                + self.twiddle10.re * x623p.re
                + self.twiddle2.re * x722p.re
                + self.twiddle6.re * x821p.re
                + self.twiddle14.re * x920p.re
                + self.twiddle7.re * x1019p.re
                + self.twiddle1.re * x1118p.re
                + self.twiddle9.re * x1217p.re
                + self.twiddle12.re * x1316p.re
                + self.twiddle4.re * x1415p.re;
            let b821re_b = self.twiddle8.im * x128n.im
                + -self.twiddle13.im * x227n.im
                + -self.twiddle5.im * x326n.im
                + self.twiddle3.im * x425n.im
                + self.twiddle11.im * x524n.im
                + -self.twiddle10.im * x623n.im
                + -self.twiddle2.im * x722n.im
                + self.twiddle6.im * x821n.im
                + self.twiddle14.im * x920n.im
                + -self.twiddle7.im * x1019n.im
                + self.twiddle1.im * x1118n.im
                + self.twiddle9.im * x1217n.im
                + -self.twiddle12.im * x1316n.im
                + -self.twiddle4.im * x1415n.im;
            let b920re_a = fmla(self.twiddle9.re, x128p.re, u0.re)
                + self.twiddle11.re * x227p.re
                + self.twiddle2.re * x326p.re
                + self.twiddle7.re * x425p.re
                + self.twiddle13.re * x524p.re
                + self.twiddle4.re * x623p.re
                + self.twiddle5.re * x722p.re
                + self.twiddle14.re * x821p.re
                + self.twiddle6.re * x920p.re
                + self.twiddle3.re * x1019p.re
                + self.twiddle12.re * x1118p.re
                + self.twiddle8.re * x1217p.re
                + self.twiddle1.re * x1316p.re
                + self.twiddle10.re * x1415p.re;
            let b920re_b = self.twiddle9.im * x128n.im
                + -self.twiddle11.im * x227n.im
                + -self.twiddle2.im * x326n.im
                + self.twiddle7.im * x425n.im
                + -self.twiddle13.im * x524n.im
                + -self.twiddle4.im * x623n.im
                + self.twiddle5.im * x722n.im
                + self.twiddle14.im * x821n.im
                + -self.twiddle6.im * x920n.im
                + self.twiddle3.im * x1019n.im
                + self.twiddle12.im * x1118n.im
                + -self.twiddle8.im * x1217n.im
                + self.twiddle1.im * x1316n.im
                + self.twiddle10.im * x1415n.im;
            let b1019re_a = u0.re
                + self.twiddle10.re * x128p.re
                + self.twiddle9.re * x227p.re
                + self.twiddle1.re * x326p.re
                + self.twiddle11.re * x425p.re
                + self.twiddle8.re * x524p.re
                + self.twiddle2.re * x623p.re
                + self.twiddle12.re * x722p.re
                + self.twiddle7.re * x821p.re
                + self.twiddle3.re * x920p.re
                + self.twiddle13.re * x1019p.re
                + self.twiddle6.re * x1118p.re
                + self.twiddle4.re * x1217p.re
                + self.twiddle14.re * x1316p.re
                + self.twiddle5.re * x1415p.re;
            let b1019re_b = self.twiddle10.im * x128n.im
                + -self.twiddle9.im * x227n.im
                + self.twiddle1.im * x326n.im
                + self.twiddle11.im * x425n.im
                + -self.twiddle8.im * x524n.im
                + self.twiddle2.im * x623n.im
                + self.twiddle12.im * x722n.im
                + -self.twiddle7.im * x821n.im
                + self.twiddle3.im * x920n.im
                + self.twiddle13.im * x1019n.im
                + -self.twiddle6.im * x1118n.im
                + self.twiddle4.im * x1217n.im
                + self.twiddle14.im * x1316n.im
                + -self.twiddle5.im * x1415n.im;
            let b1118re_a = u0.re
                + self.twiddle11.re * x128p.re
                + self.twiddle7.re * x227p.re
                + self.twiddle4.re * x326p.re
                + self.twiddle14.re * x425p.re
                + self.twiddle3.re * x524p.re
                + self.twiddle8.re * x623p.re
                + self.twiddle10.re * x722p.re
                + self.twiddle1.re * x821p.re
                + self.twiddle12.re * x920p.re
                + self.twiddle6.re * x1019p.re
                + self.twiddle5.re * x1118p.re
                + self.twiddle13.re * x1217p.re
                + self.twiddle2.re * x1316p.re
                + self.twiddle9.re * x1415p.re;
            let b1118re_b = self.twiddle11.im * x128n.im
                + -self.twiddle7.im * x227n.im
                + self.twiddle4.im * x326n.im
                + -self.twiddle14.im * x425n.im
                + -self.twiddle3.im * x524n.im
                + self.twiddle8.im * x623n.im
                + -self.twiddle10.im * x722n.im
                + self.twiddle1.im * x821n.im
                + self.twiddle12.im * x920n.im
                + -self.twiddle6.im * x1019n.im
                + self.twiddle5.im * x1118n.im
                + -self.twiddle13.im * x1217n.im
                + -self.twiddle2.im * x1316n.im
                + self.twiddle9.im * x1415n.im;
            let b1217re_a = u0.re
                + self.twiddle12.re * x128p.re
                + self.twiddle5.re * x227p.re
                + self.twiddle7.re * x326p.re
                + self.twiddle10.re * x425p.re
                + self.twiddle2.re * x524p.re
                + self.twiddle14.re * x623p.re
                + self.twiddle3.re * x722p.re
                + self.twiddle9.re * x821p.re
                + self.twiddle8.re * x920p.re
                + self.twiddle4.re * x1019p.re
                + self.twiddle13.re * x1118p.re
                + self.twiddle1.re * x1217p.re
                + self.twiddle11.re * x1316p.re
                + self.twiddle6.re * x1415p.re;
            let b1217re_b = self.twiddle12.im * x128n.im
                + -self.twiddle5.im * x227n.im
                + self.twiddle7.im * x326n.im
                + -self.twiddle10.im * x425n.im
                + self.twiddle2.im * x524n.im
                + self.twiddle14.im * x623n.im
                + -self.twiddle3.im * x722n.im
                + self.twiddle9.im * x821n.im
                + -self.twiddle8.im * x920n.im
                + self.twiddle4.im * x1019n.im
                + -self.twiddle13.im * x1118n.im
                + -self.twiddle1.im * x1217n.im
                + self.twiddle11.im * x1316n.im
                + -self.twiddle6.im * x1415n.im;
            let b1316re_a = u0.re
                + self.twiddle13.re * x128p.re
                + self.twiddle3.re * x227p.re
                + self.twiddle10.re * x326p.re
                + self.twiddle6.re * x425p.re
                + self.twiddle7.re * x524p.re
                + self.twiddle9.re * x623p.re
                + self.twiddle4.re * x722p.re
                + self.twiddle12.re * x821p.re
                + self.twiddle1.re * x920p.re
                + self.twiddle14.re * x1019p.re
                + self.twiddle2.re * x1118p.re
                + self.twiddle11.re * x1217p.re
                + self.twiddle5.re * x1316p.re
                + self.twiddle8.re * x1415p.re;
            let b1316re_b = self.twiddle13.im * x128n.im
                + -self.twiddle3.im * x227n.im
                + self.twiddle10.im * x326n.im
                + -self.twiddle6.im * x425n.im
                + self.twiddle7.im * x524n.im
                + -self.twiddle9.im * x623n.im
                + self.twiddle4.im * x722n.im
                + -self.twiddle12.im * x821n.im
                + self.twiddle1.im * x920n.im
                + self.twiddle14.im * x1019n.im
                + -self.twiddle2.im * x1118n.im
                + self.twiddle11.im * x1217n.im
                + -self.twiddle5.im * x1316n.im
                + self.twiddle8.im * x1415n.im;
            let b1415re_a = u0.re
                + self.twiddle14.re * x128p.re
                + self.twiddle1.re * x227p.re
                + self.twiddle13.re * x326p.re
                + self.twiddle2.re * x425p.re
                + self.twiddle12.re * x524p.re
                + self.twiddle3.re * x623p.re
                + self.twiddle11.re * x722p.re
                + self.twiddle4.re * x821p.re
                + self.twiddle10.re * x920p.re
                + self.twiddle5.re * x1019p.re
                + self.twiddle9.re * x1118p.re
                + self.twiddle6.re * x1217p.re
                + self.twiddle8.re * x1316p.re
                + self.twiddle7.re * x1415p.re;
            let b1415re_b = self.twiddle14.im * x128n.im
                + -self.twiddle1.im * x227n.im
                + self.twiddle13.im * x326n.im
                + -self.twiddle2.im * x425n.im
                + self.twiddle12.im * x524n.im
                + -self.twiddle3.im * x623n.im
                + self.twiddle11.im * x722n.im
                + -self.twiddle4.im * x821n.im
                + self.twiddle10.im * x920n.im
                + -self.twiddle5.im * x1019n.im
                + self.twiddle9.im * x1118n.im
                + -self.twiddle6.im * x1217n.im
                + self.twiddle8.im * x1316n.im
                + -self.twiddle7.im * x1415n.im;

            let b128im_a = u0.im
                + self.twiddle1.re * x128p.im
                + self.twiddle2.re * x227p.im
                + self.twiddle3.re * x326p.im
                + self.twiddle4.re * x425p.im
                + self.twiddle5.re * x524p.im
                + self.twiddle6.re * x623p.im
                + self.twiddle7.re * x722p.im
                + self.twiddle8.re * x821p.im
                + self.twiddle9.re * x920p.im
                + self.twiddle10.re * x1019p.im
                + self.twiddle11.re * x1118p.im
                + self.twiddle12.re * x1217p.im
                + self.twiddle13.re * x1316p.im
                + self.twiddle14.re * x1415p.im;
            let b128im_b = self.twiddle1.im * x128n.re
                + self.twiddle2.im * x227n.re
                + self.twiddle3.im * x326n.re
                + self.twiddle4.im * x425n.re
                + self.twiddle5.im * x524n.re
                + self.twiddle6.im * x623n.re
                + self.twiddle7.im * x722n.re
                + self.twiddle8.im * x821n.re
                + self.twiddle9.im * x920n.re
                + self.twiddle10.im * x1019n.re
                + self.twiddle11.im * x1118n.re
                + self.twiddle12.im * x1217n.re
                + self.twiddle13.im * x1316n.re
                + self.twiddle14.im * x1415n.re;
            let b227im_a = u0.im
                + self.twiddle2.re * x128p.im
                + self.twiddle4.re * x227p.im
                + self.twiddle6.re * x326p.im
                + self.twiddle8.re * x425p.im
                + self.twiddle10.re * x524p.im
                + self.twiddle12.re * x623p.im
                + self.twiddle14.re * x722p.im
                + self.twiddle13.re * x821p.im
                + self.twiddle11.re * x920p.im
                + self.twiddle9.re * x1019p.im
                + self.twiddle7.re * x1118p.im
                + self.twiddle5.re * x1217p.im
                + self.twiddle3.re * x1316p.im
                + self.twiddle1.re * x1415p.im;
            let b227im_b = self.twiddle2.im * x128n.re
                + self.twiddle4.im * x227n.re
                + self.twiddle6.im * x326n.re
                + self.twiddle8.im * x425n.re
                + self.twiddle10.im * x524n.re
                + self.twiddle12.im * x623n.re
                + self.twiddle14.im * x722n.re
                + -self.twiddle13.im * x821n.re
                + -self.twiddle11.im * x920n.re
                + -self.twiddle9.im * x1019n.re
                + -self.twiddle7.im * x1118n.re
                + -self.twiddle5.im * x1217n.re
                + -self.twiddle3.im * x1316n.re
                + -self.twiddle1.im * x1415n.re;
            let b326im_a = u0.im
                + self.twiddle3.re * x128p.im
                + self.twiddle6.re * x227p.im
                + self.twiddle9.re * x326p.im
                + self.twiddle12.re * x425p.im
                + self.twiddle14.re * x524p.im
                + self.twiddle11.re * x623p.im
                + self.twiddle8.re * x722p.im
                + self.twiddle5.re * x821p.im
                + self.twiddle2.re * x920p.im
                + self.twiddle1.re * x1019p.im
                + self.twiddle4.re * x1118p.im
                + self.twiddle7.re * x1217p.im
                + self.twiddle10.re * x1316p.im
                + self.twiddle13.re * x1415p.im;
            let b326im_b = self.twiddle3.im * x128n.re
                + self.twiddle6.im * x227n.re
                + self.twiddle9.im * x326n.re
                + self.twiddle12.im * x425n.re
                + -self.twiddle14.im * x524n.re
                + -self.twiddle11.im * x623n.re
                + -self.twiddle8.im * x722n.re
                + -self.twiddle5.im * x821n.re
                + -self.twiddle2.im * x920n.re
                + self.twiddle1.im * x1019n.re
                + self.twiddle4.im * x1118n.re
                + self.twiddle7.im * x1217n.re
                + self.twiddle10.im * x1316n.re
                + self.twiddle13.im * x1415n.re;
            let b425im_a = u0.im
                + self.twiddle4.re * x128p.im
                + self.twiddle8.re * x227p.im
                + self.twiddle12.re * x326p.im
                + self.twiddle13.re * x425p.im
                + self.twiddle9.re * x524p.im
                + self.twiddle5.re * x623p.im
                + self.twiddle1.re * x722p.im
                + self.twiddle3.re * x821p.im
                + self.twiddle7.re * x920p.im
                + self.twiddle11.re * x1019p.im
                + self.twiddle14.re * x1118p.im
                + self.twiddle10.re * x1217p.im
                + self.twiddle6.re * x1316p.im
                + self.twiddle2.re * x1415p.im;
            let b425im_b = self.twiddle4.im * x128n.re
                + self.twiddle8.im * x227n.re
                + self.twiddle12.im * x326n.re
                + -self.twiddle13.im * x425n.re
                + -self.twiddle9.im * x524n.re
                + -self.twiddle5.im * x623n.re
                + -self.twiddle1.im * x722n.re
                + self.twiddle3.im * x821n.re
                + self.twiddle7.im * x920n.re
                + self.twiddle11.im * x1019n.re
                + -self.twiddle14.im * x1118n.re
                + -self.twiddle10.im * x1217n.re
                + -self.twiddle6.im * x1316n.re
                + -self.twiddle2.im * x1415n.re;
            let b524im_a = u0.im
                + self.twiddle5.re * x128p.im
                + self.twiddle10.re * x227p.im
                + self.twiddle14.re * x326p.im
                + self.twiddle9.re * x425p.im
                + self.twiddle4.re * x524p.im
                + self.twiddle1.re * x623p.im
                + self.twiddle6.re * x722p.im
                + self.twiddle11.re * x821p.im
                + self.twiddle13.re * x920p.im
                + self.twiddle8.re * x1019p.im
                + self.twiddle3.re * x1118p.im
                + self.twiddle2.re * x1217p.im
                + self.twiddle7.re * x1316p.im
                + self.twiddle12.re * x1415p.im;
            let b524im_b = self.twiddle5.im * x128n.re
                + self.twiddle10.im * x227n.re
                + -self.twiddle14.im * x326n.re
                + -self.twiddle9.im * x425n.re
                + -self.twiddle4.im * x524n.re
                + self.twiddle1.im * x623n.re
                + self.twiddle6.im * x722n.re
                + self.twiddle11.im * x821n.re
                + -self.twiddle13.im * x920n.re
                + -self.twiddle8.im * x1019n.re
                + -self.twiddle3.im * x1118n.re
                + self.twiddle2.im * x1217n.re
                + self.twiddle7.im * x1316n.re
                + self.twiddle12.im * x1415n.re;
            let b623im_a = u0.im
                + self.twiddle6.re * x128p.im
                + self.twiddle12.re * x227p.im
                + self.twiddle11.re * x326p.im
                + self.twiddle5.re * x425p.im
                + self.twiddle1.re * x524p.im
                + self.twiddle7.re * x623p.im
                + self.twiddle13.re * x722p.im
                + self.twiddle10.re * x821p.im
                + self.twiddle4.re * x920p.im
                + self.twiddle2.re * x1019p.im
                + self.twiddle8.re * x1118p.im
                + self.twiddle14.re * x1217p.im
                + self.twiddle9.re * x1316p.im
                + self.twiddle3.re * x1415p.im;
            let b623im_b = self.twiddle6.im * x128n.re
                + self.twiddle12.im * x227n.re
                + -self.twiddle11.im * x326n.re
                + -self.twiddle5.im * x425n.re
                + self.twiddle1.im * x524n.re
                + self.twiddle7.im * x623n.re
                + self.twiddle13.im * x722n.re
                + -self.twiddle10.im * x821n.re
                + -self.twiddle4.im * x920n.re
                + self.twiddle2.im * x1019n.re
                + self.twiddle8.im * x1118n.re
                + self.twiddle14.im * x1217n.re
                + -self.twiddle9.im * x1316n.re
                + -self.twiddle3.im * x1415n.re;
            let b722im_a = u0.im
                + self.twiddle7.re * x128p.im
                + self.twiddle14.re * x227p.im
                + self.twiddle8.re * x326p.im
                + self.twiddle1.re * x425p.im
                + self.twiddle6.re * x524p.im
                + self.twiddle13.re * x623p.im
                + self.twiddle9.re * x722p.im
                + self.twiddle2.re * x821p.im
                + self.twiddle5.re * x920p.im
                + self.twiddle12.re * x1019p.im
                + self.twiddle10.re * x1118p.im
                + self.twiddle3.re * x1217p.im
                + self.twiddle4.re * x1316p.im
                + self.twiddle11.re * x1415p.im;
            let b722im_b = self.twiddle7.im * x128n.re
                + self.twiddle14.im * x227n.re
                + -self.twiddle8.im * x326n.re
                + -self.twiddle1.im * x425n.re
                + self.twiddle6.im * x524n.re
                + self.twiddle13.im * x623n.re
                + -self.twiddle9.im * x722n.re
                + -self.twiddle2.im * x821n.re
                + self.twiddle5.im * x920n.re
                + self.twiddle12.im * x1019n.re
                + -self.twiddle10.im * x1118n.re
                + -self.twiddle3.im * x1217n.re
                + self.twiddle4.im * x1316n.re
                + self.twiddle11.im * x1415n.re;
            let b821im_a = u0.im
                + self.twiddle8.re * x128p.im
                + self.twiddle13.re * x227p.im
                + self.twiddle5.re * x326p.im
                + self.twiddle3.re * x425p.im
                + self.twiddle11.re * x524p.im
                + self.twiddle10.re * x623p.im
                + self.twiddle2.re * x722p.im
                + self.twiddle6.re * x821p.im
                + self.twiddle14.re * x920p.im
                + self.twiddle7.re * x1019p.im
                + self.twiddle1.re * x1118p.im
                + self.twiddle9.re * x1217p.im
                + self.twiddle12.re * x1316p.im
                + self.twiddle4.re * x1415p.im;
            let b821im_b = self.twiddle8.im * x128n.re
                + -self.twiddle13.im * x227n.re
                + -self.twiddle5.im * x326n.re
                + self.twiddle3.im * x425n.re
                + self.twiddle11.im * x524n.re
                + -self.twiddle10.im * x623n.re
                + -self.twiddle2.im * x722n.re
                + self.twiddle6.im * x821n.re
                + self.twiddle14.im * x920n.re
                + -self.twiddle7.im * x1019n.re
                + self.twiddle1.im * x1118n.re
                + self.twiddle9.im * x1217n.re
                + -self.twiddle12.im * x1316n.re
                + -self.twiddle4.im * x1415n.re;
            let b920im_a = u0.im
                + self.twiddle9.re * x128p.im
                + self.twiddle11.re * x227p.im
                + self.twiddle2.re * x326p.im
                + self.twiddle7.re * x425p.im
                + self.twiddle13.re * x524p.im
                + self.twiddle4.re * x623p.im
                + self.twiddle5.re * x722p.im
                + self.twiddle14.re * x821p.im
                + self.twiddle6.re * x920p.im
                + self.twiddle3.re * x1019p.im
                + self.twiddle12.re * x1118p.im
                + self.twiddle8.re * x1217p.im
                + self.twiddle1.re * x1316p.im
                + self.twiddle10.re * x1415p.im;
            let b920im_b = self.twiddle9.im * x128n.re
                + -self.twiddle11.im * x227n.re
                + -self.twiddle2.im * x326n.re
                + self.twiddle7.im * x425n.re
                + -self.twiddle13.im * x524n.re
                + -self.twiddle4.im * x623n.re
                + self.twiddle5.im * x722n.re
                + self.twiddle14.im * x821n.re
                + -self.twiddle6.im * x920n.re
                + self.twiddle3.im * x1019n.re
                + self.twiddle12.im * x1118n.re
                + -self.twiddle8.im * x1217n.re
                + self.twiddle1.im * x1316n.re
                + self.twiddle10.im * x1415n.re;
            let b1019im_a = u0.im
                + self.twiddle10.re * x128p.im
                + self.twiddle9.re * x227p.im
                + self.twiddle1.re * x326p.im
                + self.twiddle11.re * x425p.im
                + self.twiddle8.re * x524p.im
                + self.twiddle2.re * x623p.im
                + self.twiddle12.re * x722p.im
                + self.twiddle7.re * x821p.im
                + self.twiddle3.re * x920p.im
                + self.twiddle13.re * x1019p.im
                + self.twiddle6.re * x1118p.im
                + self.twiddle4.re * x1217p.im
                + self.twiddle14.re * x1316p.im
                + self.twiddle5.re * x1415p.im;
            let b1019im_b = self.twiddle10.im * x128n.re
                + -self.twiddle9.im * x227n.re
                + self.twiddle1.im * x326n.re
                + self.twiddle11.im * x425n.re
                + -self.twiddle8.im * x524n.re
                + self.twiddle2.im * x623n.re
                + self.twiddle12.im * x722n.re
                + -self.twiddle7.im * x821n.re
                + self.twiddle3.im * x920n.re
                + self.twiddle13.im * x1019n.re
                + -self.twiddle6.im * x1118n.re
                + self.twiddle4.im * x1217n.re
                + self.twiddle14.im * x1316n.re
                + -self.twiddle5.im * x1415n.re;
            let b1118im_a = u0.im
                + self.twiddle11.re * x128p.im
                + self.twiddle7.re * x227p.im
                + self.twiddle4.re * x326p.im
                + self.twiddle14.re * x425p.im
                + self.twiddle3.re * x524p.im
                + self.twiddle8.re * x623p.im
                + self.twiddle10.re * x722p.im
                + self.twiddle1.re * x821p.im
                + self.twiddle12.re * x920p.im
                + self.twiddle6.re * x1019p.im
                + self.twiddle5.re * x1118p.im
                + self.twiddle13.re * x1217p.im
                + self.twiddle2.re * x1316p.im
                + self.twiddle9.re * x1415p.im;
            let b1118im_b = self.twiddle11.im * x128n.re
                + -self.twiddle7.im * x227n.re
                + self.twiddle4.im * x326n.re
                + -self.twiddle14.im * x425n.re
                + -self.twiddle3.im * x524n.re
                + self.twiddle8.im * x623n.re
                + -self.twiddle10.im * x722n.re
                + self.twiddle1.im * x821n.re
                + self.twiddle12.im * x920n.re
                + -self.twiddle6.im * x1019n.re
                + self.twiddle5.im * x1118n.re
                + -self.twiddle13.im * x1217n.re
                + -self.twiddle2.im * x1316n.re
                + self.twiddle9.im * x1415n.re;
            let b1217im_a = u0.im
                + self.twiddle12.re * x128p.im
                + self.twiddle5.re * x227p.im
                + self.twiddle7.re * x326p.im
                + self.twiddle10.re * x425p.im
                + self.twiddle2.re * x524p.im
                + self.twiddle14.re * x623p.im
                + self.twiddle3.re * x722p.im
                + self.twiddle9.re * x821p.im
                + self.twiddle8.re * x920p.im
                + self.twiddle4.re * x1019p.im
                + self.twiddle13.re * x1118p.im
                + self.twiddle1.re * x1217p.im
                + self.twiddle11.re * x1316p.im
                + self.twiddle6.re * x1415p.im;
            let b1217im_b = self.twiddle12.im * x128n.re
                + -self.twiddle5.im * x227n.re
                + self.twiddle7.im * x326n.re
                + -self.twiddle10.im * x425n.re
                + self.twiddle2.im * x524n.re
                + self.twiddle14.im * x623n.re
                + -self.twiddle3.im * x722n.re
                + self.twiddle9.im * x821n.re
                + -self.twiddle8.im * x920n.re
                + self.twiddle4.im * x1019n.re
                + -self.twiddle13.im * x1118n.re
                + -self.twiddle1.im * x1217n.re
                + self.twiddle11.im * x1316n.re
                + -self.twiddle6.im * x1415n.re;
            let b1316im_a = u0.im
                + self.twiddle13.re * x128p.im
                + self.twiddle3.re * x227p.im
                + self.twiddle10.re * x326p.im
                + self.twiddle6.re * x425p.im
                + self.twiddle7.re * x524p.im
                + self.twiddle9.re * x623p.im
                + self.twiddle4.re * x722p.im
                + self.twiddle12.re * x821p.im
                + self.twiddle1.re * x920p.im
                + self.twiddle14.re * x1019p.im
                + self.twiddle2.re * x1118p.im
                + self.twiddle11.re * x1217p.im
                + self.twiddle5.re * x1316p.im
                + self.twiddle8.re * x1415p.im;
            let b1316im_b = self.twiddle13.im * x128n.re
                + -self.twiddle3.im * x227n.re
                + self.twiddle10.im * x326n.re
                + -self.twiddle6.im * x425n.re
                + self.twiddle7.im * x524n.re
                + -self.twiddle9.im * x623n.re
                + self.twiddle4.im * x722n.re
                + -self.twiddle12.im * x821n.re
                + self.twiddle1.im * x920n.re
                + self.twiddle14.im * x1019n.re
                + -self.twiddle2.im * x1118n.re
                + self.twiddle11.im * x1217n.re
                + -self.twiddle5.im * x1316n.re
                + self.twiddle8.im * x1415n.re;
            let b1415im_a = u0.im
                + self.twiddle14.re * x128p.im
                + self.twiddle1.re * x227p.im
                + self.twiddle13.re * x326p.im
                + self.twiddle2.re * x425p.im
                + self.twiddle12.re * x524p.im
                + self.twiddle3.re * x623p.im
                + self.twiddle11.re * x722p.im
                + self.twiddle4.re * x821p.im
                + self.twiddle10.re * x920p.im
                + self.twiddle5.re * x1019p.im
                + self.twiddle9.re * x1118p.im
                + self.twiddle6.re * x1217p.im
                + self.twiddle8.re * x1316p.im
                + self.twiddle7.re * x1415p.im;
            let b1415im_b = self.twiddle14.im * x128n.re
                + -self.twiddle1.im * x227n.re
                + self.twiddle13.im * x326n.re
                + -self.twiddle2.im * x425n.re
                + self.twiddle12.im * x524n.re
                + -self.twiddle3.im * x623n.re
                + self.twiddle11.im * x722n.re
                + -self.twiddle4.im * x821n.re
                + self.twiddle10.im * x920n.re
                + -self.twiddle5.im * x1019n.re
                + self.twiddle9.im * x1118n.re
                + -self.twiddle6.im * x1217n.re
                + self.twiddle8.im * x1316n.re
                + -self.twiddle7.im * x1415n.re;

            let out1re = b128re_a - b128re_b;
            let out1im = b128im_a + b128im_b;
            let out2re = b227re_a - b227re_b;
            let out2im = b227im_a + b227im_b;
            let out3re = b326re_a - b326re_b;
            let out3im = b326im_a + b326im_b;
            let out4re = b425re_a - b425re_b;
            let out4im = b425im_a + b425im_b;
            let out5re = b524re_a - b524re_b;
            let out5im = b524im_a + b524im_b;
            let out6re = b623re_a - b623re_b;
            let out6im = b623im_a + b623im_b;
            let out7re = b722re_a - b722re_b;
            let out7im = b722im_a + b722im_b;
            let out8re = b821re_a - b821re_b;
            let out8im = b821im_a + b821im_b;
            let out9re = b920re_a - b920re_b;
            let out9im = b920im_a + b920im_b;
            let out10re = b1019re_a - b1019re_b;
            let out10im = b1019im_a + b1019im_b;
            let out11re = b1118re_a - b1118re_b;
            let out11im = b1118im_a + b1118im_b;
            let out12re = b1217re_a - b1217re_b;
            let out12im = b1217im_a + b1217im_b;
            let out13re = b1316re_a - b1316re_b;
            let out13im = b1316im_a + b1316im_b;
            let out14re = b1415re_a - b1415re_b;
            let out14im = b1415im_a + b1415im_b;
            let out15re = b1415re_a + b1415re_b;
            let out15im = b1415im_a - b1415im_b;
            let out16re = b1316re_a + b1316re_b;
            let out16im = b1316im_a - b1316im_b;
            let out17re = b1217re_a + b1217re_b;
            let out17im = b1217im_a - b1217im_b;
            let out18re = b1118re_a + b1118re_b;
            let out18im = b1118im_a - b1118im_b;
            let out19re = b1019re_a + b1019re_b;
            let out19im = b1019im_a - b1019im_b;
            let out20re = b920re_a + b920re_b;
            let out20im = b920im_a - b920im_b;
            let out21re = b821re_a + b821re_b;
            let out21im = b821im_a - b821im_b;
            let out22re = b722re_a + b722re_b;
            let out22im = b722im_a - b722im_b;
            let out23re = b623re_a + b623re_b;
            let out23im = b623im_a - b623im_b;
            let out24re = b524re_a + b524re_b;
            let out24im = b524im_a - b524im_b;
            let out25re = b425re_a + b425re_b;
            let out25im = b425im_a - b425im_b;
            let out26re = b326re_a + b326re_b;
            let out26im = b326im_a - b326im_b;
            let out27re = b227re_a + b227re_b;
            let out27im = b227im_a - b227im_b;
            let out28re = b128re_a + b128re_b;
            let out28im = b128im_a - b128im_b;

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
            chunk[23] = Complex {
                re: out23re,
                im: out23im,
            };
            chunk[24] = Complex {
                re: out24re,
                im: out24im,
            };
            chunk[25] = Complex {
                re: out25re,
                im: out25im,
            };
            chunk[26] = Complex {
                re: out26re,
                im: out26im,
            };
            chunk[27] = Complex {
                re: out27re,
                im: out27im,
            };
            chunk[28] = Complex {
                re: out28re,
                im: out28im,
            };
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        29
    }
}

impl<T: FftSample> R2CFftExecutor<T> for Butterfly29<T> {
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !input.len().is_multiple_of(self.real_length()) {
            return Err(ZaftError::InvalidSizeMultiplier(
                input.len(),
                self.real_length(),
            ));
        }
        if !output.len().is_multiple_of(self.complex_length()) {
            return Err(ZaftError::InvalidSizeMultiplier(
                input.len(),
                self.complex_length(),
            ));
        }

        for (chunk, dst) in input.chunks_exact(29).zip(output.chunks_exact_mut(15)) {
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

            let u25 = chunk[25];
            let u26 = chunk[26];

            let u27 = chunk[27];
            let u28 = chunk[28];

            let x128p = u1 + u28;
            let x128n = u1 - u28;
            let x227p = u2 + u27;
            let x227n = u2 - u27;
            let x326p = u3 + u26;
            let x326n = u3 - u26;
            let x425p = u4 + u25;
            let x425n = u4 - u25;
            let x524p = u5 + u24;
            let x524n = u5 - u24;
            let x623p = u6 + u23;
            let x623n = u6 - u23;
            let x722p = u7 + u22;
            let x722n = u7 - u22;
            let x821p = u8 + u21;
            let x821n = u8 - u21;
            let x920p = u9 + u20;
            let x920n = u9 - u20;
            let x1019p = u10 + u19;
            let x1019n = u10 - u19;
            let x1118p = u11 + u18;
            let x1118n = u11 - u18;
            let x1217p = u12 + u17;
            let x1217n = u12 - u17;
            let x1316p = u13 + u16;
            let x1316n = u13 - u16;
            let x1415p = u14 + u15;
            let x1415n = u14 - u15;
            let sum = u0
                + x128p
                + x227p
                + x326p
                + x425p
                + x524p
                + x623p
                + x722p
                + x821p
                + x920p
                + x1019p
                + x1118p
                + x1217p
                + x1316p
                + x1415p;
            dst[0] = Complex::new(sum, T::zero());

            let b128re_a = fmla(self.twiddle1.re, x128p, u0)
                + self.twiddle2.re * x227p
                + self.twiddle3.re * x326p
                + self.twiddle4.re * x425p
                + self.twiddle5.re * x524p
                + self.twiddle6.re * x623p
                + self.twiddle7.re * x722p
                + self.twiddle8.re * x821p
                + self.twiddle9.re * x920p
                + self.twiddle10.re * x1019p
                + self.twiddle11.re * x1118p
                + self.twiddle12.re * x1217p
                + self.twiddle13.re * x1316p
                + self.twiddle14.re * x1415p;
            let b227re_a = u0
                + self.twiddle2.re * x128p
                + self.twiddle4.re * x227p
                + self.twiddle6.re * x326p
                + self.twiddle8.re * x425p
                + self.twiddle10.re * x524p
                + self.twiddle12.re * x623p
                + self.twiddle14.re * x722p
                + self.twiddle13.re * x821p
                + self.twiddle11.re * x920p
                + self.twiddle9.re * x1019p
                + self.twiddle7.re * x1118p
                + self.twiddle5.re * x1217p
                + self.twiddle3.re * x1316p
                + self.twiddle1.re * x1415p;
            let b326re_a = fmla(self.twiddle3.re, x128p, u0)
                + self.twiddle6.re * x227p
                + self.twiddle9.re * x326p
                + self.twiddle12.re * x425p
                + self.twiddle14.re * x524p
                + self.twiddle11.re * x623p
                + self.twiddle8.re * x722p
                + self.twiddle5.re * x821p
                + self.twiddle2.re * x920p
                + self.twiddle1.re * x1019p
                + self.twiddle4.re * x1118p
                + self.twiddle7.re * x1217p
                + self.twiddle10.re * x1316p
                + self.twiddle13.re * x1415p;
            let b425re_a = fmla(self.twiddle4.re, x128p, u0)
                + self.twiddle8.re * x227p
                + self.twiddle12.re * x326p
                + self.twiddle13.re * x425p
                + self.twiddle9.re * x524p
                + self.twiddle5.re * x623p
                + self.twiddle1.re * x722p
                + self.twiddle3.re * x821p
                + self.twiddle7.re * x920p
                + self.twiddle11.re * x1019p
                + self.twiddle14.re * x1118p
                + self.twiddle10.re * x1217p
                + self.twiddle6.re * x1316p
                + self.twiddle2.re * x1415p;
            let b524re_a = fmla(self.twiddle5.re, x128p, u0)
                + self.twiddle10.re * x227p
                + self.twiddle14.re * x326p
                + self.twiddle9.re * x425p
                + self.twiddle4.re * x524p
                + self.twiddle1.re * x623p
                + self.twiddle6.re * x722p
                + self.twiddle11.re * x821p
                + self.twiddle13.re * x920p
                + self.twiddle8.re * x1019p
                + self.twiddle3.re * x1118p
                + self.twiddle2.re * x1217p
                + self.twiddle7.re * x1316p
                + self.twiddle12.re * x1415p;
            let b623re_a = fmla(self.twiddle6.re, x128p, u0)
                + self.twiddle12.re * x227p
                + self.twiddle11.re * x326p
                + self.twiddle5.re * x425p
                + self.twiddle1.re * x524p
                + self.twiddle7.re * x623p
                + self.twiddle13.re * x722p
                + self.twiddle10.re * x821p
                + self.twiddle4.re * x920p
                + self.twiddle2.re * x1019p
                + self.twiddle8.re * x1118p
                + self.twiddle14.re * x1217p
                + self.twiddle9.re * x1316p
                + self.twiddle3.re * x1415p;
            let b722re_a = fmla(self.twiddle7.re, x128p, u0)
                + self.twiddle14.re * x227p
                + self.twiddle8.re * x326p
                + self.twiddle1.re * x425p
                + self.twiddle6.re * x524p
                + self.twiddle13.re * x623p
                + self.twiddle9.re * x722p
                + self.twiddle2.re * x821p
                + self.twiddle5.re * x920p
                + self.twiddle12.re * x1019p
                + self.twiddle10.re * x1118p
                + self.twiddle3.re * x1217p
                + self.twiddle4.re * x1316p
                + self.twiddle11.re * x1415p;
            let b821re_a = fmla(self.twiddle8.re, x128p, u0)
                + self.twiddle13.re * x227p
                + self.twiddle5.re * x326p
                + self.twiddle3.re * x425p
                + self.twiddle11.re * x524p
                + self.twiddle10.re * x623p
                + self.twiddle2.re * x722p
                + self.twiddle6.re * x821p
                + self.twiddle14.re * x920p
                + self.twiddle7.re * x1019p
                + self.twiddle1.re * x1118p
                + self.twiddle9.re * x1217p
                + self.twiddle12.re * x1316p
                + self.twiddle4.re * x1415p;
            let b920re_a = fmla(self.twiddle9.re, x128p, u0)
                + self.twiddle11.re * x227p
                + self.twiddle2.re * x326p
                + self.twiddle7.re * x425p
                + self.twiddle13.re * x524p
                + self.twiddle4.re * x623p
                + self.twiddle5.re * x722p
                + self.twiddle14.re * x821p
                + self.twiddle6.re * x920p
                + self.twiddle3.re * x1019p
                + self.twiddle12.re * x1118p
                + self.twiddle8.re * x1217p
                + self.twiddle1.re * x1316p
                + self.twiddle10.re * x1415p;
            let b1019re_a = u0
                + self.twiddle10.re * x128p
                + self.twiddle9.re * x227p
                + self.twiddle1.re * x326p
                + self.twiddle11.re * x425p
                + self.twiddle8.re * x524p
                + self.twiddle2.re * x623p
                + self.twiddle12.re * x722p
                + self.twiddle7.re * x821p
                + self.twiddle3.re * x920p
                + self.twiddle13.re * x1019p
                + self.twiddle6.re * x1118p
                + self.twiddle4.re * x1217p
                + self.twiddle14.re * x1316p
                + self.twiddle5.re * x1415p;
            let b1118re_a = u0
                + self.twiddle11.re * x128p
                + self.twiddle7.re * x227p
                + self.twiddle4.re * x326p
                + self.twiddle14.re * x425p
                + self.twiddle3.re * x524p
                + self.twiddle8.re * x623p
                + self.twiddle10.re * x722p
                + self.twiddle1.re * x821p
                + self.twiddle12.re * x920p
                + self.twiddle6.re * x1019p
                + self.twiddle5.re * x1118p
                + self.twiddle13.re * x1217p
                + self.twiddle2.re * x1316p
                + self.twiddle9.re * x1415p;
            let b1217re_a = u0
                + self.twiddle12.re * x128p
                + self.twiddle5.re * x227p
                + self.twiddle7.re * x326p
                + self.twiddle10.re * x425p
                + self.twiddle2.re * x524p
                + self.twiddle14.re * x623p
                + self.twiddle3.re * x722p
                + self.twiddle9.re * x821p
                + self.twiddle8.re * x920p
                + self.twiddle4.re * x1019p
                + self.twiddle13.re * x1118p
                + self.twiddle1.re * x1217p
                + self.twiddle11.re * x1316p
                + self.twiddle6.re * x1415p;
            let b1316re_a = u0
                + self.twiddle13.re * x128p
                + self.twiddle3.re * x227p
                + self.twiddle10.re * x326p
                + self.twiddle6.re * x425p
                + self.twiddle7.re * x524p
                + self.twiddle9.re * x623p
                + self.twiddle4.re * x722p
                + self.twiddle12.re * x821p
                + self.twiddle1.re * x920p
                + self.twiddle14.re * x1019p
                + self.twiddle2.re * x1118p
                + self.twiddle11.re * x1217p
                + self.twiddle5.re * x1316p
                + self.twiddle8.re * x1415p;
            let b1415re_a = u0
                + self.twiddle14.re * x128p
                + self.twiddle1.re * x227p
                + self.twiddle13.re * x326p
                + self.twiddle2.re * x425p
                + self.twiddle12.re * x524p
                + self.twiddle3.re * x623p
                + self.twiddle11.re * x722p
                + self.twiddle4.re * x821p
                + self.twiddle10.re * x920p
                + self.twiddle5.re * x1019p
                + self.twiddle9.re * x1118p
                + self.twiddle6.re * x1217p
                + self.twiddle8.re * x1316p
                + self.twiddle7.re * x1415p;
            let b128im_b = self.twiddle1.im * x128n
                + self.twiddle2.im * x227n
                + self.twiddle3.im * x326n
                + self.twiddle4.im * x425n
                + self.twiddle5.im * x524n
                + self.twiddle6.im * x623n
                + self.twiddle7.im * x722n
                + self.twiddle8.im * x821n
                + self.twiddle9.im * x920n
                + self.twiddle10.im * x1019n
                + self.twiddle11.im * x1118n
                + self.twiddle12.im * x1217n
                + self.twiddle13.im * x1316n
                + self.twiddle14.im * x1415n;
            let b227im_b = self.twiddle2.im * x128n
                + self.twiddle4.im * x227n
                + self.twiddle6.im * x326n
                + self.twiddle8.im * x425n
                + self.twiddle10.im * x524n
                + self.twiddle12.im * x623n
                + self.twiddle14.im * x722n
                + -self.twiddle13.im * x821n
                + -self.twiddle11.im * x920n
                + -self.twiddle9.im * x1019n
                + -self.twiddle7.im * x1118n
                + -self.twiddle5.im * x1217n
                + -self.twiddle3.im * x1316n
                + -self.twiddle1.im * x1415n;
            let b326im_b = self.twiddle3.im * x128n
                + self.twiddle6.im * x227n
                + self.twiddle9.im * x326n
                + self.twiddle12.im * x425n
                + -self.twiddle14.im * x524n
                + -self.twiddle11.im * x623n
                + -self.twiddle8.im * x722n
                + -self.twiddle5.im * x821n
                + -self.twiddle2.im * x920n
                + self.twiddle1.im * x1019n
                + self.twiddle4.im * x1118n
                + self.twiddle7.im * x1217n
                + self.twiddle10.im * x1316n
                + self.twiddle13.im * x1415n;
            let b425im_b = self.twiddle4.im * x128n
                + self.twiddle8.im * x227n
                + self.twiddle12.im * x326n
                + -self.twiddle13.im * x425n
                + -self.twiddle9.im * x524n
                + -self.twiddle5.im * x623n
                + -self.twiddle1.im * x722n
                + self.twiddle3.im * x821n
                + self.twiddle7.im * x920n
                + self.twiddle11.im * x1019n
                + -self.twiddle14.im * x1118n
                + -self.twiddle10.im * x1217n
                + -self.twiddle6.im * x1316n
                + -self.twiddle2.im * x1415n;
            let b524im_b = self.twiddle5.im * x128n
                + self.twiddle10.im * x227n
                + -self.twiddle14.im * x326n
                + -self.twiddle9.im * x425n
                + -self.twiddle4.im * x524n
                + self.twiddle1.im * x623n
                + self.twiddle6.im * x722n
                + self.twiddle11.im * x821n
                + -self.twiddle13.im * x920n
                + -self.twiddle8.im * x1019n
                + -self.twiddle3.im * x1118n
                + self.twiddle2.im * x1217n
                + self.twiddle7.im * x1316n
                + self.twiddle12.im * x1415n;
            let b623im_b = self.twiddle6.im * x128n
                + self.twiddle12.im * x227n
                + -self.twiddle11.im * x326n
                + -self.twiddle5.im * x425n
                + self.twiddle1.im * x524n
                + self.twiddle7.im * x623n
                + self.twiddle13.im * x722n
                + -self.twiddle10.im * x821n
                + -self.twiddle4.im * x920n
                + self.twiddle2.im * x1019n
                + self.twiddle8.im * x1118n
                + self.twiddle14.im * x1217n
                + -self.twiddle9.im * x1316n
                + -self.twiddle3.im * x1415n;
            let b722im_b = self.twiddle7.im * x128n
                + self.twiddle14.im * x227n
                + -self.twiddle8.im * x326n
                + -self.twiddle1.im * x425n
                + self.twiddle6.im * x524n
                + self.twiddle13.im * x623n
                + -self.twiddle9.im * x722n
                + -self.twiddle2.im * x821n
                + self.twiddle5.im * x920n
                + self.twiddle12.im * x1019n
                + -self.twiddle10.im * x1118n
                + -self.twiddle3.im * x1217n
                + self.twiddle4.im * x1316n
                + self.twiddle11.im * x1415n;
            let b821im_b = self.twiddle8.im * x128n
                + -self.twiddle13.im * x227n
                + -self.twiddle5.im * x326n
                + self.twiddle3.im * x425n
                + self.twiddle11.im * x524n
                + -self.twiddle10.im * x623n
                + -self.twiddle2.im * x722n
                + self.twiddle6.im * x821n
                + self.twiddle14.im * x920n
                + -self.twiddle7.im * x1019n
                + self.twiddle1.im * x1118n
                + self.twiddle9.im * x1217n
                + -self.twiddle12.im * x1316n
                + -self.twiddle4.im * x1415n;
            let b920im_b = self.twiddle9.im * x128n
                + -self.twiddle11.im * x227n
                + -self.twiddle2.im * x326n
                + self.twiddle7.im * x425n
                + -self.twiddle13.im * x524n
                + -self.twiddle4.im * x623n
                + self.twiddle5.im * x722n
                + self.twiddle14.im * x821n
                + -self.twiddle6.im * x920n
                + self.twiddle3.im * x1019n
                + self.twiddle12.im * x1118n
                + -self.twiddle8.im * x1217n
                + self.twiddle1.im * x1316n
                + self.twiddle10.im * x1415n;
            let b1019im_b = self.twiddle10.im * x128n
                + -self.twiddle9.im * x227n
                + self.twiddle1.im * x326n
                + self.twiddle11.im * x425n
                + -self.twiddle8.im * x524n
                + self.twiddle2.im * x623n
                + self.twiddle12.im * x722n
                + -self.twiddle7.im * x821n
                + self.twiddle3.im * x920n
                + self.twiddle13.im * x1019n
                + -self.twiddle6.im * x1118n
                + self.twiddle4.im * x1217n
                + self.twiddle14.im * x1316n
                + -self.twiddle5.im * x1415n;
            let b1118im_b = self.twiddle11.im * x128n
                + -self.twiddle7.im * x227n
                + self.twiddle4.im * x326n
                + -self.twiddle14.im * x425n
                + -self.twiddle3.im * x524n
                + self.twiddle8.im * x623n
                + -self.twiddle10.im * x722n
                + self.twiddle1.im * x821n
                + self.twiddle12.im * x920n
                + -self.twiddle6.im * x1019n
                + self.twiddle5.im * x1118n
                + -self.twiddle13.im * x1217n
                + -self.twiddle2.im * x1316n
                + self.twiddle9.im * x1415n;
            let b1217im_b = self.twiddle12.im * x128n
                + -self.twiddle5.im * x227n
                + self.twiddle7.im * x326n
                + -self.twiddle10.im * x425n
                + self.twiddle2.im * x524n
                + self.twiddle14.im * x623n
                + -self.twiddle3.im * x722n
                + self.twiddle9.im * x821n
                + -self.twiddle8.im * x920n
                + self.twiddle4.im * x1019n
                + -self.twiddle13.im * x1118n
                + -self.twiddle1.im * x1217n
                + self.twiddle11.im * x1316n
                + -self.twiddle6.im * x1415n;
            let b1316im_b = self.twiddle13.im * x128n
                + -self.twiddle3.im * x227n
                + self.twiddle10.im * x326n
                + -self.twiddle6.im * x425n
                + self.twiddle7.im * x524n
                + -self.twiddle9.im * x623n
                + self.twiddle4.im * x722n
                + -self.twiddle12.im * x821n
                + self.twiddle1.im * x920n
                + self.twiddle14.im * x1019n
                + -self.twiddle2.im * x1118n
                + self.twiddle11.im * x1217n
                + -self.twiddle5.im * x1316n
                + self.twiddle8.im * x1415n;
            let b1415im_b = self.twiddle14.im * x128n
                + -self.twiddle1.im * x227n
                + self.twiddle13.im * x326n
                + -self.twiddle2.im * x425n
                + self.twiddle12.im * x524n
                + -self.twiddle3.im * x623n
                + self.twiddle11.im * x722n
                + -self.twiddle4.im * x821n
                + self.twiddle10.im * x920n
                + -self.twiddle5.im * x1019n
                + self.twiddle9.im * x1118n
                + -self.twiddle6.im * x1217n
                + self.twiddle8.im * x1316n
                + -self.twiddle7.im * x1415n;

            let out1re = b128re_a;
            let out1im = b128im_b;
            let out2re = b227re_a;
            let out2im = b227im_b;
            let out3re = b326re_a;
            let out3im = b326im_b;
            let out4re = b425re_a;
            let out4im = b425im_b;
            let out5re = b524re_a;
            let out5im = b524im_b;
            let out6re = b623re_a;
            let out6im = b623im_b;
            let out7re = b722re_a;
            let out7im = b722im_b;
            let out8re = b821re_a;
            let out8im = b821im_b;
            let out9re = b920re_a;
            let out9im = b920im_b;
            let out10re = b1019re_a;
            let out10im = b1019im_b;
            let out11re = b1118re_a;
            let out11im = b1118im_b;
            let out12re = b1217re_a;
            let out12im = b1217im_b;
            let out13re = b1316re_a;
            let out13im = b1316im_b;
            let out14re = b1415re_a;
            let out14im = b1415im_b;
            dst[1] = Complex {
                re: out1re,
                im: out1im,
            };
            dst[2] = Complex {
                re: out2re,
                im: out2im,
            };
            dst[3] = Complex {
                re: out3re,
                im: out3im,
            };
            dst[4] = Complex {
                re: out4re,
                im: out4im,
            };
            dst[5] = Complex {
                re: out5re,
                im: out5im,
            };
            dst[6] = Complex {
                re: out6re,
                im: out6im,
            };
            dst[7] = Complex {
                re: out7re,
                im: out7im,
            };
            dst[8] = Complex {
                re: out8re,
                im: out8im,
            };
            dst[9] = Complex {
                re: out9re,
                im: out9im,
            };
            dst[10] = Complex {
                re: out10re,
                im: out10im,
            };
            dst[11] = Complex {
                re: out11re,
                im: out11im,
            };
            dst[12] = Complex {
                re: out12re,
                im: out12im,
            };
            dst[13] = Complex {
                re: out13re,
                im: out13im,
            };
            dst[14] = Complex {
                re: out14re,
                im: out14im,
            };
        }
        Ok(())
    }

    #[inline]
    fn real_length(&self) -> usize {
        29
    }

    #[inline]
    fn complex_length(&self) -> usize {
        15
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    use crate::r2c::test_r2c_butterfly;

    test_r2c_butterfly!(test_r2c_butterfly29, f32, Butterfly29, 29, 1e-5);
    test_butterfly!(test_butterfly29, f32, Butterfly29, 29, 1e-5);
}
