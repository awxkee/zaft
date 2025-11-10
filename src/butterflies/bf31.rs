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

use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num};
use std::ops::{Add, Mul, Neg, Sub};

#[allow(unused)]
pub(crate) struct Butterfly31<T> {
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
    twiddle15: Complex<T>,
}

#[allow(unused)]
impl<T: FftTrigonometry + Float + 'static + Default> Butterfly31<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        let twiddle1: Complex<T> = compute_twiddle(1, 31, fft_direction);
        let twiddle2: Complex<T> = compute_twiddle(2, 31, fft_direction);
        let twiddle3: Complex<T> = compute_twiddle(3, 31, fft_direction);
        let twiddle4: Complex<T> = compute_twiddle(4, 31, fft_direction);
        let twiddle5: Complex<T> = compute_twiddle(5, 31, fft_direction);
        let twiddle6: Complex<T> = compute_twiddle(6, 31, fft_direction);
        let twiddle7: Complex<T> = compute_twiddle(7, 31, fft_direction);
        let twiddle8: Complex<T> = compute_twiddle(8, 31, fft_direction);
        let twiddle9: Complex<T> = compute_twiddle(9, 31, fft_direction);
        let twiddle10: Complex<T> = compute_twiddle(10, 31, fft_direction);
        let twiddle11: Complex<T> = compute_twiddle(11, 31, fft_direction);
        let twiddle12: Complex<T> = compute_twiddle(12, 31, fft_direction);
        let twiddle13: Complex<T> = compute_twiddle(13, 31, fft_direction);
        let twiddle14: Complex<T> = compute_twiddle(14, 31, fft_direction);
        let twiddle15: Complex<T> = compute_twiddle(15, 31, fft_direction);
        Butterfly31 {
            direction: fft_direction,
            twiddle1,
            twiddle2,
            twiddle3,
            twiddle4,
            twiddle5,
            twiddle6,
            twiddle7,
            twiddle8,
            twiddle9,
            twiddle10,
            twiddle11,
            twiddle12,
            twiddle13,
            twiddle14,
            twiddle15,
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
        + Default
        + FftTrigonometry,
> FftExecutor<T> for Butterfly31<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if in_place.len() % self.length() != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(31) {
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

            let u29 = chunk[29];
            let u30 = chunk[30];

            let x130p = u1 + u30;
            let x130n = u1 - u30;
            let x229p = u2 + u29;
            let x229n = u2 - u29;
            let x328p = u3 + u28;
            let x328n = u3 - u28;
            let x427p = u4 + u27;
            let x427n = u4 - u27;
            let x526p = u5 + u26;
            let x526n = u5 - u26;
            let x625p = u6 + u25;
            let x625n = u6 - u25;
            let x724p = u7 + u24;
            let x724n = u7 - u24;
            let x823p = u8 + u23;
            let x823n = u8 - u23;
            let x922p = u9 + u22;
            let x922n = u9 - u22;
            let x1021p = u10 + u21;
            let x1021n = u10 - u21;
            let x1120p = u11 + u20;
            let x1120n = u11 - u20;
            let x1219p = u12 + u19;
            let x1219n = u12 - u19;
            let x1318p = u13 + u18;
            let x1318n = u13 - u18;
            let x1417p = u14 + u17;
            let x1417n = u14 - u17;
            let x1516p = u15 + u16;
            let x1516n = u15 - u16;
            let sum = u0
                + x130p
                + x229p
                + x328p
                + x427p
                + x526p
                + x625p
                + x724p
                + x823p
                + x922p
                + x1021p
                + x1120p
                + x1219p
                + x1318p
                + x1417p
                + x1516p;
            chunk[0] = sum;
            let b130re_a = u0.re
                + self.twiddle1.re * x130p.re
                + self.twiddle2.re * x229p.re
                + self.twiddle3.re * x328p.re
                + self.twiddle4.re * x427p.re
                + self.twiddle5.re * x526p.re
                + self.twiddle6.re * x625p.re
                + self.twiddle7.re * x724p.re
                + self.twiddle8.re * x823p.re
                + self.twiddle9.re * x922p.re
                + self.twiddle10.re * x1021p.re
                + self.twiddle11.re * x1120p.re
                + self.twiddle12.re * x1219p.re
                + self.twiddle13.re * x1318p.re
                + self.twiddle14.re * x1417p.re
                + self.twiddle15.re * x1516p.re;
            let b130re_b = self.twiddle1.im * x130n.im
                + self.twiddle2.im * x229n.im
                + self.twiddle3.im * x328n.im
                + self.twiddle4.im * x427n.im
                + self.twiddle5.im * x526n.im
                + self.twiddle6.im * x625n.im
                + self.twiddle7.im * x724n.im
                + self.twiddle8.im * x823n.im
                + self.twiddle9.im * x922n.im
                + self.twiddle10.im * x1021n.im
                + self.twiddle11.im * x1120n.im
                + self.twiddle12.im * x1219n.im
                + self.twiddle13.im * x1318n.im
                + self.twiddle14.im * x1417n.im
                + self.twiddle15.im * x1516n.im;
            let b229re_a = u0.re
                + self.twiddle2.re * x130p.re
                + self.twiddle4.re * x229p.re
                + self.twiddle6.re * x328p.re
                + self.twiddle8.re * x427p.re
                + self.twiddle10.re * x526p.re
                + self.twiddle12.re * x625p.re
                + self.twiddle14.re * x724p.re
                + self.twiddle15.re * x823p.re
                + self.twiddle13.re * x922p.re
                + self.twiddle11.re * x1021p.re
                + self.twiddle9.re * x1120p.re
                + self.twiddle7.re * x1219p.re
                + self.twiddle5.re * x1318p.re
                + self.twiddle3.re * x1417p.re
                + self.twiddle1.re * x1516p.re;
            let b229re_b = self.twiddle2.im * x130n.im
                + self.twiddle4.im * x229n.im
                + self.twiddle6.im * x328n.im
                + self.twiddle8.im * x427n.im
                + self.twiddle10.im * x526n.im
                + self.twiddle12.im * x625n.im
                + self.twiddle14.im * x724n.im
                + -self.twiddle15.im * x823n.im
                + -self.twiddle13.im * x922n.im
                + -self.twiddle11.im * x1021n.im
                + -self.twiddle9.im * x1120n.im
                + -self.twiddle7.im * x1219n.im
                + -self.twiddle5.im * x1318n.im
                + -self.twiddle3.im * x1417n.im
                + -self.twiddle1.im * x1516n.im;
            let b328re_a = u0.re
                + self.twiddle3.re * x130p.re
                + self.twiddle6.re * x229p.re
                + self.twiddle9.re * x328p.re
                + self.twiddle12.re * x427p.re
                + self.twiddle15.re * x526p.re
                + self.twiddle13.re * x625p.re
                + self.twiddle10.re * x724p.re
                + self.twiddle7.re * x823p.re
                + self.twiddle4.re * x922p.re
                + self.twiddle1.re * x1021p.re
                + self.twiddle2.re * x1120p.re
                + self.twiddle5.re * x1219p.re
                + self.twiddle8.re * x1318p.re
                + self.twiddle11.re * x1417p.re
                + self.twiddle14.re * x1516p.re;
            let b328re_b = self.twiddle3.im * x130n.im
                + self.twiddle6.im * x229n.im
                + self.twiddle9.im * x328n.im
                + self.twiddle12.im * x427n.im
                + self.twiddle15.im * x526n.im
                + -self.twiddle13.im * x625n.im
                + -self.twiddle10.im * x724n.im
                + -self.twiddle7.im * x823n.im
                + -self.twiddle4.im * x922n.im
                + -self.twiddle1.im * x1021n.im
                + self.twiddle2.im * x1120n.im
                + self.twiddle5.im * x1219n.im
                + self.twiddle8.im * x1318n.im
                + self.twiddle11.im * x1417n.im
                + self.twiddle14.im * x1516n.im;
            let b427re_a = u0.re
                + self.twiddle4.re * x130p.re
                + self.twiddle8.re * x229p.re
                + self.twiddle12.re * x328p.re
                + self.twiddle15.re * x427p.re
                + self.twiddle11.re * x526p.re
                + self.twiddle7.re * x625p.re
                + self.twiddle3.re * x724p.re
                + self.twiddle1.re * x823p.re
                + self.twiddle5.re * x922p.re
                + self.twiddle9.re * x1021p.re
                + self.twiddle13.re * x1120p.re
                + self.twiddle14.re * x1219p.re
                + self.twiddle10.re * x1318p.re
                + self.twiddle6.re * x1417p.re
                + self.twiddle2.re * x1516p.re;
            let b427re_b = self.twiddle4.im * x130n.im
                + self.twiddle8.im * x229n.im
                + self.twiddle12.im * x328n.im
                + -self.twiddle15.im * x427n.im
                + -self.twiddle11.im * x526n.im
                + -self.twiddle7.im * x625n.im
                + -self.twiddle3.im * x724n.im
                + self.twiddle1.im * x823n.im
                + self.twiddle5.im * x922n.im
                + self.twiddle9.im * x1021n.im
                + self.twiddle13.im * x1120n.im
                + -self.twiddle14.im * x1219n.im
                + -self.twiddle10.im * x1318n.im
                + -self.twiddle6.im * x1417n.im
                + -self.twiddle2.im * x1516n.im;
            let b526re_a = u0.re
                + self.twiddle5.re * x130p.re
                + self.twiddle10.re * x229p.re
                + self.twiddle15.re * x328p.re
                + self.twiddle11.re * x427p.re
                + self.twiddle6.re * x526p.re
                + self.twiddle1.re * x625p.re
                + self.twiddle4.re * x724p.re
                + self.twiddle9.re * x823p.re
                + self.twiddle14.re * x922p.re
                + self.twiddle12.re * x1021p.re
                + self.twiddle7.re * x1120p.re
                + self.twiddle2.re * x1219p.re
                + self.twiddle3.re * x1318p.re
                + self.twiddle8.re * x1417p.re
                + self.twiddle13.re * x1516p.re;
            let b526re_b = self.twiddle5.im * x130n.im
                + self.twiddle10.im * x229n.im
                + self.twiddle15.im * x328n.im
                + -self.twiddle11.im * x427n.im
                + -self.twiddle6.im * x526n.im
                + -self.twiddle1.im * x625n.im
                + self.twiddle4.im * x724n.im
                + self.twiddle9.im * x823n.im
                + self.twiddle14.im * x922n.im
                + -self.twiddle12.im * x1021n.im
                + -self.twiddle7.im * x1120n.im
                + -self.twiddle2.im * x1219n.im
                + self.twiddle3.im * x1318n.im
                + self.twiddle8.im * x1417n.im
                + self.twiddle13.im * x1516n.im;
            let b625re_a = u0.re
                + self.twiddle6.re * x130p.re
                + self.twiddle12.re * x229p.re
                + self.twiddle13.re * x328p.re
                + self.twiddle7.re * x427p.re
                + self.twiddle1.re * x526p.re
                + self.twiddle5.re * x625p.re
                + self.twiddle11.re * x724p.re
                + self.twiddle14.re * x823p.re
                + self.twiddle8.re * x922p.re
                + self.twiddle2.re * x1021p.re
                + self.twiddle4.re * x1120p.re
                + self.twiddle10.re * x1219p.re
                + self.twiddle15.re * x1318p.re
                + self.twiddle9.re * x1417p.re
                + self.twiddle3.re * x1516p.re;
            let b625re_b = self.twiddle6.im * x130n.im
                + self.twiddle12.im * x229n.im
                + -self.twiddle13.im * x328n.im
                + -self.twiddle7.im * x427n.im
                + -self.twiddle1.im * x526n.im
                + self.twiddle5.im * x625n.im
                + self.twiddle11.im * x724n.im
                + -self.twiddle14.im * x823n.im
                + -self.twiddle8.im * x922n.im
                + -self.twiddle2.im * x1021n.im
                + self.twiddle4.im * x1120n.im
                + self.twiddle10.im * x1219n.im
                + -self.twiddle15.im * x1318n.im
                + -self.twiddle9.im * x1417n.im
                + -self.twiddle3.im * x1516n.im;
            let b724re_a = u0.re
                + self.twiddle7.re * x130p.re
                + self.twiddle14.re * x229p.re
                + self.twiddle10.re * x328p.re
                + self.twiddle3.re * x427p.re
                + self.twiddle4.re * x526p.re
                + self.twiddle11.re * x625p.re
                + self.twiddle13.re * x724p.re
                + self.twiddle6.re * x823p.re
                + self.twiddle1.re * x922p.re
                + self.twiddle8.re * x1021p.re
                + self.twiddle15.re * x1120p.re
                + self.twiddle9.re * x1219p.re
                + self.twiddle2.re * x1318p.re
                + self.twiddle5.re * x1417p.re
                + self.twiddle12.re * x1516p.re;
            let b724re_b = self.twiddle7.im * x130n.im
                + self.twiddle14.im * x229n.im
                + -self.twiddle10.im * x328n.im
                + -self.twiddle3.im * x427n.im
                + self.twiddle4.im * x526n.im
                + self.twiddle11.im * x625n.im
                + -self.twiddle13.im * x724n.im
                + -self.twiddle6.im * x823n.im
                + self.twiddle1.im * x922n.im
                + self.twiddle8.im * x1021n.im
                + self.twiddle15.im * x1120n.im
                + -self.twiddle9.im * x1219n.im
                + -self.twiddle2.im * x1318n.im
                + self.twiddle5.im * x1417n.im
                + self.twiddle12.im * x1516n.im;
            let b823re_a = u0.re
                + self.twiddle8.re * x130p.re
                + self.twiddle15.re * x229p.re
                + self.twiddle7.re * x328p.re
                + self.twiddle1.re * x427p.re
                + self.twiddle9.re * x526p.re
                + self.twiddle14.re * x625p.re
                + self.twiddle6.re * x724p.re
                + self.twiddle2.re * x823p.re
                + self.twiddle10.re * x922p.re
                + self.twiddle13.re * x1021p.re
                + self.twiddle5.re * x1120p.re
                + self.twiddle3.re * x1219p.re
                + self.twiddle11.re * x1318p.re
                + self.twiddle12.re * x1417p.re
                + self.twiddle4.re * x1516p.re;
            let b823re_b = self.twiddle8.im * x130n.im
                + -self.twiddle15.im * x229n.im
                + -self.twiddle7.im * x328n.im
                + self.twiddle1.im * x427n.im
                + self.twiddle9.im * x526n.im
                + -self.twiddle14.im * x625n.im
                + -self.twiddle6.im * x724n.im
                + self.twiddle2.im * x823n.im
                + self.twiddle10.im * x922n.im
                + -self.twiddle13.im * x1021n.im
                + -self.twiddle5.im * x1120n.im
                + self.twiddle3.im * x1219n.im
                + self.twiddle11.im * x1318n.im
                + -self.twiddle12.im * x1417n.im
                + -self.twiddle4.im * x1516n.im;
            let b922re_a = u0.re
                + self.twiddle9.re * x130p.re
                + self.twiddle13.re * x229p.re
                + self.twiddle4.re * x328p.re
                + self.twiddle5.re * x427p.re
                + self.twiddle14.re * x526p.re
                + self.twiddle8.re * x625p.re
                + self.twiddle1.re * x724p.re
                + self.twiddle10.re * x823p.re
                + self.twiddle12.re * x922p.re
                + self.twiddle3.re * x1021p.re
                + self.twiddle6.re * x1120p.re
                + self.twiddle15.re * x1219p.re
                + self.twiddle7.re * x1318p.re
                + self.twiddle2.re * x1417p.re
                + self.twiddle11.re * x1516p.re;
            let b922re_b = self.twiddle9.im * x130n.im
                + -self.twiddle13.im * x229n.im
                + -self.twiddle4.im * x328n.im
                + self.twiddle5.im * x427n.im
                + self.twiddle14.im * x526n.im
                + -self.twiddle8.im * x625n.im
                + self.twiddle1.im * x724n.im
                + self.twiddle10.im * x823n.im
                + -self.twiddle12.im * x922n.im
                + -self.twiddle3.im * x1021n.im
                + self.twiddle6.im * x1120n.im
                + self.twiddle15.im * x1219n.im
                + -self.twiddle7.im * x1318n.im
                + self.twiddle2.im * x1417n.im
                + self.twiddle11.im * x1516n.im;
            let b1021re_a = u0.re
                + self.twiddle10.re * x130p.re
                + self.twiddle11.re * x229p.re
                + self.twiddle1.re * x328p.re
                + self.twiddle9.re * x427p.re
                + self.twiddle12.re * x526p.re
                + self.twiddle2.re * x625p.re
                + self.twiddle8.re * x724p.re
                + self.twiddle13.re * x823p.re
                + self.twiddle3.re * x922p.re
                + self.twiddle7.re * x1021p.re
                + self.twiddle14.re * x1120p.re
                + self.twiddle4.re * x1219p.re
                + self.twiddle6.re * x1318p.re
                + self.twiddle15.re * x1417p.re
                + self.twiddle5.re * x1516p.re;
            let b1021re_b = self.twiddle10.im * x130n.im
                + -self.twiddle11.im * x229n.im
                + -self.twiddle1.im * x328n.im
                + self.twiddle9.im * x427n.im
                + -self.twiddle12.im * x526n.im
                + -self.twiddle2.im * x625n.im
                + self.twiddle8.im * x724n.im
                + -self.twiddle13.im * x823n.im
                + -self.twiddle3.im * x922n.im
                + self.twiddle7.im * x1021n.im
                + -self.twiddle14.im * x1120n.im
                + -self.twiddle4.im * x1219n.im
                + self.twiddle6.im * x1318n.im
                + -self.twiddle15.im * x1417n.im
                + -self.twiddle5.im * x1516n.im;
            let b1120re_a = u0.re
                + self.twiddle11.re * x130p.re
                + self.twiddle9.re * x229p.re
                + self.twiddle2.re * x328p.re
                + self.twiddle13.re * x427p.re
                + self.twiddle7.re * x526p.re
                + self.twiddle4.re * x625p.re
                + self.twiddle15.re * x724p.re
                + self.twiddle5.re * x823p.re
                + self.twiddle6.re * x922p.re
                + self.twiddle14.re * x1021p.re
                + self.twiddle3.re * x1120p.re
                + self.twiddle8.re * x1219p.re
                + self.twiddle12.re * x1318p.re
                + self.twiddle1.re * x1417p.re
                + self.twiddle10.re * x1516p.re;
            let b1120re_b = self.twiddle11.im * x130n.im
                + -self.twiddle9.im * x229n.im
                + self.twiddle2.im * x328n.im
                + self.twiddle13.im * x427n.im
                + -self.twiddle7.im * x526n.im
                + self.twiddle4.im * x625n.im
                + self.twiddle15.im * x724n.im
                + -self.twiddle5.im * x823n.im
                + self.twiddle6.im * x922n.im
                + -self.twiddle14.im * x1021n.im
                + -self.twiddle3.im * x1120n.im
                + self.twiddle8.im * x1219n.im
                + -self.twiddle12.im * x1318n.im
                + -self.twiddle1.im * x1417n.im
                + self.twiddle10.im * x1516n.im;
            let b1219re_a = u0.re
                + self.twiddle12.re * x130p.re
                + self.twiddle7.re * x229p.re
                + self.twiddle5.re * x328p.re
                + self.twiddle14.re * x427p.re
                + self.twiddle2.re * x526p.re
                + self.twiddle10.re * x625p.re
                + self.twiddle9.re * x724p.re
                + self.twiddle3.re * x823p.re
                + self.twiddle15.re * x922p.re
                + self.twiddle4.re * x1021p.re
                + self.twiddle8.re * x1120p.re
                + self.twiddle11.re * x1219p.re
                + self.twiddle1.re * x1318p.re
                + self.twiddle13.re * x1417p.re
                + self.twiddle6.re * x1516p.re;
            let b1219re_b = self.twiddle12.im * x130n.im
                + -self.twiddle7.im * x229n.im
                + self.twiddle5.im * x328n.im
                + -self.twiddle14.im * x427n.im
                + -self.twiddle2.im * x526n.im
                + self.twiddle10.im * x625n.im
                + -self.twiddle9.im * x724n.im
                + self.twiddle3.im * x823n.im
                + self.twiddle15.im * x922n.im
                + -self.twiddle4.im * x1021n.im
                + self.twiddle8.im * x1120n.im
                + -self.twiddle11.im * x1219n.im
                + self.twiddle1.im * x1318n.im
                + self.twiddle13.im * x1417n.im
                + -self.twiddle6.im * x1516n.im;
            let b1318re_a = u0.re
                + self.twiddle13.re * x130p.re
                + self.twiddle5.re * x229p.re
                + self.twiddle8.re * x328p.re
                + self.twiddle10.re * x427p.re
                + self.twiddle3.re * x526p.re
                + self.twiddle15.re * x625p.re
                + self.twiddle2.re * x724p.re
                + self.twiddle11.re * x823p.re
                + self.twiddle7.re * x922p.re
                + self.twiddle6.re * x1021p.re
                + self.twiddle12.re * x1120p.re
                + self.twiddle1.re * x1219p.re
                + self.twiddle14.re * x1318p.re
                + self.twiddle4.re * x1417p.re
                + self.twiddle9.re * x1516p.re;
            let b1318re_b = self.twiddle13.im * x130n.im
                + -self.twiddle5.im * x229n.im
                + self.twiddle8.im * x328n.im
                + -self.twiddle10.im * x427n.im
                + self.twiddle3.im * x526n.im
                + -self.twiddle15.im * x625n.im
                + -self.twiddle2.im * x724n.im
                + self.twiddle11.im * x823n.im
                + -self.twiddle7.im * x922n.im
                + self.twiddle6.im * x1021n.im
                + -self.twiddle12.im * x1120n.im
                + self.twiddle1.im * x1219n.im
                + self.twiddle14.im * x1318n.im
                + -self.twiddle4.im * x1417n.im
                + self.twiddle9.im * x1516n.im;
            let b1417re_a = u0.re
                + self.twiddle14.re * x130p.re
                + self.twiddle3.re * x229p.re
                + self.twiddle11.re * x328p.re
                + self.twiddle6.re * x427p.re
                + self.twiddle8.re * x526p.re
                + self.twiddle9.re * x625p.re
                + self.twiddle5.re * x724p.re
                + self.twiddle12.re * x823p.re
                + self.twiddle2.re * x922p.re
                + self.twiddle15.re * x1021p.re
                + self.twiddle1.re * x1120p.re
                + self.twiddle13.re * x1219p.re
                + self.twiddle4.re * x1318p.re
                + self.twiddle10.re * x1417p.re
                + self.twiddle7.re * x1516p.re;
            let b1417re_b = self.twiddle14.im * x130n.im
                + -self.twiddle3.im * x229n.im
                + self.twiddle11.im * x328n.im
                + -self.twiddle6.im * x427n.im
                + self.twiddle8.im * x526n.im
                + -self.twiddle9.im * x625n.im
                + self.twiddle5.im * x724n.im
                + -self.twiddle12.im * x823n.im
                + self.twiddle2.im * x922n.im
                + -self.twiddle15.im * x1021n.im
                + -self.twiddle1.im * x1120n.im
                + self.twiddle13.im * x1219n.im
                + -self.twiddle4.im * x1318n.im
                + self.twiddle10.im * x1417n.im
                + -self.twiddle7.im * x1516n.im;
            let b1516re_a = u0.re
                + self.twiddle15.re * x130p.re
                + self.twiddle1.re * x229p.re
                + self.twiddle14.re * x328p.re
                + self.twiddle2.re * x427p.re
                + self.twiddle13.re * x526p.re
                + self.twiddle3.re * x625p.re
                + self.twiddle12.re * x724p.re
                + self.twiddle4.re * x823p.re
                + self.twiddle11.re * x922p.re
                + self.twiddle5.re * x1021p.re
                + self.twiddle10.re * x1120p.re
                + self.twiddle6.re * x1219p.re
                + self.twiddle9.re * x1318p.re
                + self.twiddle7.re * x1417p.re
                + self.twiddle8.re * x1516p.re;
            let b1516re_b = self.twiddle15.im * x130n.im
                + -self.twiddle1.im * x229n.im
                + self.twiddle14.im * x328n.im
                + -self.twiddle2.im * x427n.im
                + self.twiddle13.im * x526n.im
                + -self.twiddle3.im * x625n.im
                + self.twiddle12.im * x724n.im
                + -self.twiddle4.im * x823n.im
                + self.twiddle11.im * x922n.im
                + -self.twiddle5.im * x1021n.im
                + self.twiddle10.im * x1120n.im
                + -self.twiddle6.im * x1219n.im
                + self.twiddle9.im * x1318n.im
                + -self.twiddle7.im * x1417n.im
                + self.twiddle8.im * x1516n.im;

            let b130im_a = u0.im
                + self.twiddle1.re * x130p.im
                + self.twiddle2.re * x229p.im
                + self.twiddle3.re * x328p.im
                + self.twiddle4.re * x427p.im
                + self.twiddle5.re * x526p.im
                + self.twiddle6.re * x625p.im
                + self.twiddle7.re * x724p.im
                + self.twiddle8.re * x823p.im
                + self.twiddle9.re * x922p.im
                + self.twiddle10.re * x1021p.im
                + self.twiddle11.re * x1120p.im
                + self.twiddle12.re * x1219p.im
                + self.twiddle13.re * x1318p.im
                + self.twiddle14.re * x1417p.im
                + self.twiddle15.re * x1516p.im;
            let b130im_b = self.twiddle1.im * x130n.re
                + self.twiddle2.im * x229n.re
                + self.twiddle3.im * x328n.re
                + self.twiddle4.im * x427n.re
                + self.twiddle5.im * x526n.re
                + self.twiddle6.im * x625n.re
                + self.twiddle7.im * x724n.re
                + self.twiddle8.im * x823n.re
                + self.twiddle9.im * x922n.re
                + self.twiddle10.im * x1021n.re
                + self.twiddle11.im * x1120n.re
                + self.twiddle12.im * x1219n.re
                + self.twiddle13.im * x1318n.re
                + self.twiddle14.im * x1417n.re
                + self.twiddle15.im * x1516n.re;
            let b229im_a = u0.im
                + self.twiddle2.re * x130p.im
                + self.twiddle4.re * x229p.im
                + self.twiddle6.re * x328p.im
                + self.twiddle8.re * x427p.im
                + self.twiddle10.re * x526p.im
                + self.twiddle12.re * x625p.im
                + self.twiddle14.re * x724p.im
                + self.twiddle15.re * x823p.im
                + self.twiddle13.re * x922p.im
                + self.twiddle11.re * x1021p.im
                + self.twiddle9.re * x1120p.im
                + self.twiddle7.re * x1219p.im
                + self.twiddle5.re * x1318p.im
                + self.twiddle3.re * x1417p.im
                + self.twiddle1.re * x1516p.im;
            let b229im_b = self.twiddle2.im * x130n.re
                + self.twiddle4.im * x229n.re
                + self.twiddle6.im * x328n.re
                + self.twiddle8.im * x427n.re
                + self.twiddle10.im * x526n.re
                + self.twiddle12.im * x625n.re
                + self.twiddle14.im * x724n.re
                + -self.twiddle15.im * x823n.re
                + -self.twiddle13.im * x922n.re
                + -self.twiddle11.im * x1021n.re
                + -self.twiddle9.im * x1120n.re
                + -self.twiddle7.im * x1219n.re
                + -self.twiddle5.im * x1318n.re
                + -self.twiddle3.im * x1417n.re
                + -self.twiddle1.im * x1516n.re;
            let b328im_a = u0.im
                + self.twiddle3.re * x130p.im
                + self.twiddle6.re * x229p.im
                + self.twiddle9.re * x328p.im
                + self.twiddle12.re * x427p.im
                + self.twiddle15.re * x526p.im
                + self.twiddle13.re * x625p.im
                + self.twiddle10.re * x724p.im
                + self.twiddle7.re * x823p.im
                + self.twiddle4.re * x922p.im
                + self.twiddle1.re * x1021p.im
                + self.twiddle2.re * x1120p.im
                + self.twiddle5.re * x1219p.im
                + self.twiddle8.re * x1318p.im
                + self.twiddle11.re * x1417p.im
                + self.twiddle14.re * x1516p.im;
            let b328im_b = self.twiddle3.im * x130n.re
                + self.twiddle6.im * x229n.re
                + self.twiddle9.im * x328n.re
                + self.twiddle12.im * x427n.re
                + self.twiddle15.im * x526n.re
                + -self.twiddle13.im * x625n.re
                + -self.twiddle10.im * x724n.re
                + -self.twiddle7.im * x823n.re
                + -self.twiddle4.im * x922n.re
                + -self.twiddle1.im * x1021n.re
                + self.twiddle2.im * x1120n.re
                + self.twiddle5.im * x1219n.re
                + self.twiddle8.im * x1318n.re
                + self.twiddle11.im * x1417n.re
                + self.twiddle14.im * x1516n.re;
            let b427im_a = u0.im
                + self.twiddle4.re * x130p.im
                + self.twiddle8.re * x229p.im
                + self.twiddle12.re * x328p.im
                + self.twiddle15.re * x427p.im
                + self.twiddle11.re * x526p.im
                + self.twiddle7.re * x625p.im
                + self.twiddle3.re * x724p.im
                + self.twiddle1.re * x823p.im
                + self.twiddle5.re * x922p.im
                + self.twiddle9.re * x1021p.im
                + self.twiddle13.re * x1120p.im
                + self.twiddle14.re * x1219p.im
                + self.twiddle10.re * x1318p.im
                + self.twiddle6.re * x1417p.im
                + self.twiddle2.re * x1516p.im;
            let b427im_b = self.twiddle4.im * x130n.re
                + self.twiddle8.im * x229n.re
                + self.twiddle12.im * x328n.re
                + -self.twiddle15.im * x427n.re
                + -self.twiddle11.im * x526n.re
                + -self.twiddle7.im * x625n.re
                + -self.twiddle3.im * x724n.re
                + self.twiddle1.im * x823n.re
                + self.twiddle5.im * x922n.re
                + self.twiddle9.im * x1021n.re
                + self.twiddle13.im * x1120n.re
                + -self.twiddle14.im * x1219n.re
                + -self.twiddle10.im * x1318n.re
                + -self.twiddle6.im * x1417n.re
                + -self.twiddle2.im * x1516n.re;
            let b526im_a = u0.im
                + self.twiddle5.re * x130p.im
                + self.twiddle10.re * x229p.im
                + self.twiddle15.re * x328p.im
                + self.twiddle11.re * x427p.im
                + self.twiddle6.re * x526p.im
                + self.twiddle1.re * x625p.im
                + self.twiddle4.re * x724p.im
                + self.twiddle9.re * x823p.im
                + self.twiddle14.re * x922p.im
                + self.twiddle12.re * x1021p.im
                + self.twiddle7.re * x1120p.im
                + self.twiddle2.re * x1219p.im
                + self.twiddle3.re * x1318p.im
                + self.twiddle8.re * x1417p.im
                + self.twiddle13.re * x1516p.im;
            let b526im_b = self.twiddle5.im * x130n.re
                + self.twiddle10.im * x229n.re
                + self.twiddle15.im * x328n.re
                + -self.twiddle11.im * x427n.re
                + -self.twiddle6.im * x526n.re
                + -self.twiddle1.im * x625n.re
                + self.twiddle4.im * x724n.re
                + self.twiddle9.im * x823n.re
                + self.twiddle14.im * x922n.re
                + -self.twiddle12.im * x1021n.re
                + -self.twiddle7.im * x1120n.re
                + -self.twiddle2.im * x1219n.re
                + self.twiddle3.im * x1318n.re
                + self.twiddle8.im * x1417n.re
                + self.twiddle13.im * x1516n.re;
            let b625im_a = u0.im
                + self.twiddle6.re * x130p.im
                + self.twiddle12.re * x229p.im
                + self.twiddle13.re * x328p.im
                + self.twiddle7.re * x427p.im
                + self.twiddle1.re * x526p.im
                + self.twiddle5.re * x625p.im
                + self.twiddle11.re * x724p.im
                + self.twiddle14.re * x823p.im
                + self.twiddle8.re * x922p.im
                + self.twiddle2.re * x1021p.im
                + self.twiddle4.re * x1120p.im
                + self.twiddle10.re * x1219p.im
                + self.twiddle15.re * x1318p.im
                + self.twiddle9.re * x1417p.im
                + self.twiddle3.re * x1516p.im;
            let b625im_b = self.twiddle6.im * x130n.re
                + self.twiddle12.im * x229n.re
                + -self.twiddle13.im * x328n.re
                + -self.twiddle7.im * x427n.re
                + -self.twiddle1.im * x526n.re
                + self.twiddle5.im * x625n.re
                + self.twiddle11.im * x724n.re
                + -self.twiddle14.im * x823n.re
                + -self.twiddle8.im * x922n.re
                + -self.twiddle2.im * x1021n.re
                + self.twiddle4.im * x1120n.re
                + self.twiddle10.im * x1219n.re
                + -self.twiddle15.im * x1318n.re
                + -self.twiddle9.im * x1417n.re
                + -self.twiddle3.im * x1516n.re;
            let b724im_a = u0.im
                + self.twiddle7.re * x130p.im
                + self.twiddle14.re * x229p.im
                + self.twiddle10.re * x328p.im
                + self.twiddle3.re * x427p.im
                + self.twiddle4.re * x526p.im
                + self.twiddle11.re * x625p.im
                + self.twiddle13.re * x724p.im
                + self.twiddle6.re * x823p.im
                + self.twiddle1.re * x922p.im
                + self.twiddle8.re * x1021p.im
                + self.twiddle15.re * x1120p.im
                + self.twiddle9.re * x1219p.im
                + self.twiddle2.re * x1318p.im
                + self.twiddle5.re * x1417p.im
                + self.twiddle12.re * x1516p.im;
            let b724im_b = self.twiddle7.im * x130n.re
                + self.twiddle14.im * x229n.re
                + -self.twiddle10.im * x328n.re
                + -self.twiddle3.im * x427n.re
                + self.twiddle4.im * x526n.re
                + self.twiddle11.im * x625n.re
                + -self.twiddle13.im * x724n.re
                + -self.twiddle6.im * x823n.re
                + self.twiddle1.im * x922n.re
                + self.twiddle8.im * x1021n.re
                + self.twiddle15.im * x1120n.re
                + -self.twiddle9.im * x1219n.re
                + -self.twiddle2.im * x1318n.re
                + self.twiddle5.im * x1417n.re
                + self.twiddle12.im * x1516n.re;
            let b823im_a = u0.im
                + self.twiddle8.re * x130p.im
                + self.twiddle15.re * x229p.im
                + self.twiddle7.re * x328p.im
                + self.twiddle1.re * x427p.im
                + self.twiddle9.re * x526p.im
                + self.twiddle14.re * x625p.im
                + self.twiddle6.re * x724p.im
                + self.twiddle2.re * x823p.im
                + self.twiddle10.re * x922p.im
                + self.twiddle13.re * x1021p.im
                + self.twiddle5.re * x1120p.im
                + self.twiddle3.re * x1219p.im
                + self.twiddle11.re * x1318p.im
                + self.twiddle12.re * x1417p.im
                + self.twiddle4.re * x1516p.im;
            let b823im_b = self.twiddle8.im * x130n.re
                + -self.twiddle15.im * x229n.re
                + -self.twiddle7.im * x328n.re
                + self.twiddle1.im * x427n.re
                + self.twiddle9.im * x526n.re
                + -self.twiddle14.im * x625n.re
                + -self.twiddle6.im * x724n.re
                + self.twiddle2.im * x823n.re
                + self.twiddle10.im * x922n.re
                + -self.twiddle13.im * x1021n.re
                + -self.twiddle5.im * x1120n.re
                + self.twiddle3.im * x1219n.re
                + self.twiddle11.im * x1318n.re
                + -self.twiddle12.im * x1417n.re
                + -self.twiddle4.im * x1516n.re;
            let b922im_a = u0.im
                + self.twiddle9.re * x130p.im
                + self.twiddle13.re * x229p.im
                + self.twiddle4.re * x328p.im
                + self.twiddle5.re * x427p.im
                + self.twiddle14.re * x526p.im
                + self.twiddle8.re * x625p.im
                + self.twiddle1.re * x724p.im
                + self.twiddle10.re * x823p.im
                + self.twiddle12.re * x922p.im
                + self.twiddle3.re * x1021p.im
                + self.twiddle6.re * x1120p.im
                + self.twiddle15.re * x1219p.im
                + self.twiddle7.re * x1318p.im
                + self.twiddle2.re * x1417p.im
                + self.twiddle11.re * x1516p.im;
            let b922im_b = self.twiddle9.im * x130n.re
                + -self.twiddle13.im * x229n.re
                + -self.twiddle4.im * x328n.re
                + self.twiddle5.im * x427n.re
                + self.twiddle14.im * x526n.re
                + -self.twiddle8.im * x625n.re
                + self.twiddle1.im * x724n.re
                + self.twiddle10.im * x823n.re
                + -self.twiddle12.im * x922n.re
                + -self.twiddle3.im * x1021n.re
                + self.twiddle6.im * x1120n.re
                + self.twiddle15.im * x1219n.re
                + -self.twiddle7.im * x1318n.re
                + self.twiddle2.im * x1417n.re
                + self.twiddle11.im * x1516n.re;
            let b1021im_a = u0.im
                + self.twiddle10.re * x130p.im
                + self.twiddle11.re * x229p.im
                + self.twiddle1.re * x328p.im
                + self.twiddle9.re * x427p.im
                + self.twiddle12.re * x526p.im
                + self.twiddle2.re * x625p.im
                + self.twiddle8.re * x724p.im
                + self.twiddle13.re * x823p.im
                + self.twiddle3.re * x922p.im
                + self.twiddle7.re * x1021p.im
                + self.twiddle14.re * x1120p.im
                + self.twiddle4.re * x1219p.im
                + self.twiddle6.re * x1318p.im
                + self.twiddle15.re * x1417p.im
                + self.twiddle5.re * x1516p.im;
            let b1021im_b = self.twiddle10.im * x130n.re
                + -self.twiddle11.im * x229n.re
                + -self.twiddle1.im * x328n.re
                + self.twiddle9.im * x427n.re
                + -self.twiddle12.im * x526n.re
                + -self.twiddle2.im * x625n.re
                + self.twiddle8.im * x724n.re
                + -self.twiddle13.im * x823n.re
                + -self.twiddle3.im * x922n.re
                + self.twiddle7.im * x1021n.re
                + -self.twiddle14.im * x1120n.re
                + -self.twiddle4.im * x1219n.re
                + self.twiddle6.im * x1318n.re
                + -self.twiddle15.im * x1417n.re
                + -self.twiddle5.im * x1516n.re;
            let b1120im_a = u0.im
                + self.twiddle11.re * x130p.im
                + self.twiddle9.re * x229p.im
                + self.twiddle2.re * x328p.im
                + self.twiddle13.re * x427p.im
                + self.twiddle7.re * x526p.im
                + self.twiddle4.re * x625p.im
                + self.twiddle15.re * x724p.im
                + self.twiddle5.re * x823p.im
                + self.twiddle6.re * x922p.im
                + self.twiddle14.re * x1021p.im
                + self.twiddle3.re * x1120p.im
                + self.twiddle8.re * x1219p.im
                + self.twiddle12.re * x1318p.im
                + self.twiddle1.re * x1417p.im
                + self.twiddle10.re * x1516p.im;
            let b1120im_b = self.twiddle11.im * x130n.re
                + -self.twiddle9.im * x229n.re
                + self.twiddle2.im * x328n.re
                + self.twiddle13.im * x427n.re
                + -self.twiddle7.im * x526n.re
                + self.twiddle4.im * x625n.re
                + self.twiddle15.im * x724n.re
                + -self.twiddle5.im * x823n.re
                + self.twiddle6.im * x922n.re
                + -self.twiddle14.im * x1021n.re
                + -self.twiddle3.im * x1120n.re
                + self.twiddle8.im * x1219n.re
                + -self.twiddle12.im * x1318n.re
                + -self.twiddle1.im * x1417n.re
                + self.twiddle10.im * x1516n.re;
            let b1219im_a = u0.im
                + self.twiddle12.re * x130p.im
                + self.twiddle7.re * x229p.im
                + self.twiddle5.re * x328p.im
                + self.twiddle14.re * x427p.im
                + self.twiddle2.re * x526p.im
                + self.twiddle10.re * x625p.im
                + self.twiddle9.re * x724p.im
                + self.twiddle3.re * x823p.im
                + self.twiddle15.re * x922p.im
                + self.twiddle4.re * x1021p.im
                + self.twiddle8.re * x1120p.im
                + self.twiddle11.re * x1219p.im
                + self.twiddle1.re * x1318p.im
                + self.twiddle13.re * x1417p.im
                + self.twiddle6.re * x1516p.im;
            let b1219im_b = self.twiddle12.im * x130n.re
                + -self.twiddle7.im * x229n.re
                + self.twiddle5.im * x328n.re
                + -self.twiddle14.im * x427n.re
                + -self.twiddle2.im * x526n.re
                + self.twiddle10.im * x625n.re
                + -self.twiddle9.im * x724n.re
                + self.twiddle3.im * x823n.re
                + self.twiddle15.im * x922n.re
                + -self.twiddle4.im * x1021n.re
                + self.twiddle8.im * x1120n.re
                + -self.twiddle11.im * x1219n.re
                + self.twiddle1.im * x1318n.re
                + self.twiddle13.im * x1417n.re
                + -self.twiddle6.im * x1516n.re;
            let b1318im_a = u0.im
                + self.twiddle13.re * x130p.im
                + self.twiddle5.re * x229p.im
                + self.twiddle8.re * x328p.im
                + self.twiddle10.re * x427p.im
                + self.twiddle3.re * x526p.im
                + self.twiddle15.re * x625p.im
                + self.twiddle2.re * x724p.im
                + self.twiddle11.re * x823p.im
                + self.twiddle7.re * x922p.im
                + self.twiddle6.re * x1021p.im
                + self.twiddle12.re * x1120p.im
                + self.twiddle1.re * x1219p.im
                + self.twiddle14.re * x1318p.im
                + self.twiddle4.re * x1417p.im
                + self.twiddle9.re * x1516p.im;
            let b1318im_b = self.twiddle13.im * x130n.re
                + -self.twiddle5.im * x229n.re
                + self.twiddle8.im * x328n.re
                + -self.twiddle10.im * x427n.re
                + self.twiddle3.im * x526n.re
                + -self.twiddle15.im * x625n.re
                + -self.twiddle2.im * x724n.re
                + self.twiddle11.im * x823n.re
                + -self.twiddle7.im * x922n.re
                + self.twiddle6.im * x1021n.re
                + -self.twiddle12.im * x1120n.re
                + self.twiddle1.im * x1219n.re
                + self.twiddle14.im * x1318n.re
                + -self.twiddle4.im * x1417n.re
                + self.twiddle9.im * x1516n.re;
            let b1417im_a = u0.im
                + self.twiddle14.re * x130p.im
                + self.twiddle3.re * x229p.im
                + self.twiddle11.re * x328p.im
                + self.twiddle6.re * x427p.im
                + self.twiddle8.re * x526p.im
                + self.twiddle9.re * x625p.im
                + self.twiddle5.re * x724p.im
                + self.twiddle12.re * x823p.im
                + self.twiddle2.re * x922p.im
                + self.twiddle15.re * x1021p.im
                + self.twiddle1.re * x1120p.im
                + self.twiddle13.re * x1219p.im
                + self.twiddle4.re * x1318p.im
                + self.twiddle10.re * x1417p.im
                + self.twiddle7.re * x1516p.im;
            let b1417im_b = self.twiddle14.im * x130n.re
                + -self.twiddle3.im * x229n.re
                + self.twiddle11.im * x328n.re
                + -self.twiddle6.im * x427n.re
                + self.twiddle8.im * x526n.re
                + -self.twiddle9.im * x625n.re
                + self.twiddle5.im * x724n.re
                + -self.twiddle12.im * x823n.re
                + self.twiddle2.im * x922n.re
                + -self.twiddle15.im * x1021n.re
                + -self.twiddle1.im * x1120n.re
                + self.twiddle13.im * x1219n.re
                + -self.twiddle4.im * x1318n.re
                + self.twiddle10.im * x1417n.re
                + -self.twiddle7.im * x1516n.re;
            let b1516im_a = u0.im
                + self.twiddle15.re * x130p.im
                + self.twiddle1.re * x229p.im
                + self.twiddle14.re * x328p.im
                + self.twiddle2.re * x427p.im
                + self.twiddle13.re * x526p.im
                + self.twiddle3.re * x625p.im
                + self.twiddle12.re * x724p.im
                + self.twiddle4.re * x823p.im
                + self.twiddle11.re * x922p.im
                + self.twiddle5.re * x1021p.im
                + self.twiddle10.re * x1120p.im
                + self.twiddle6.re * x1219p.im
                + self.twiddle9.re * x1318p.im
                + self.twiddle7.re * x1417p.im
                + self.twiddle8.re * x1516p.im;
            let b1516im_b = self.twiddle15.im * x130n.re
                + -self.twiddle1.im * x229n.re
                + self.twiddle14.im * x328n.re
                + -self.twiddle2.im * x427n.re
                + self.twiddle13.im * x526n.re
                + -self.twiddle3.im * x625n.re
                + self.twiddle12.im * x724n.re
                + -self.twiddle4.im * x823n.re
                + self.twiddle11.im * x922n.re
                + -self.twiddle5.im * x1021n.re
                + self.twiddle10.im * x1120n.re
                + -self.twiddle6.im * x1219n.re
                + self.twiddle9.im * x1318n.re
                + -self.twiddle7.im * x1417n.re
                + self.twiddle8.im * x1516n.re;

            let out1re = b130re_a - b130re_b;
            let out1im = b130im_a + b130im_b;
            let out2re = b229re_a - b229re_b;
            let out2im = b229im_a + b229im_b;
            let out3re = b328re_a - b328re_b;
            let out3im = b328im_a + b328im_b;
            let out4re = b427re_a - b427re_b;
            let out4im = b427im_a + b427im_b;
            let out5re = b526re_a - b526re_b;
            let out5im = b526im_a + b526im_b;
            let out6re = b625re_a - b625re_b;
            let out6im = b625im_a + b625im_b;
            let out7re = b724re_a - b724re_b;
            let out7im = b724im_a + b724im_b;
            let out8re = b823re_a - b823re_b;
            let out8im = b823im_a + b823im_b;
            let out9re = b922re_a - b922re_b;
            let out9im = b922im_a + b922im_b;
            let out10re = b1021re_a - b1021re_b;
            let out10im = b1021im_a + b1021im_b;
            let out11re = b1120re_a - b1120re_b;
            let out11im = b1120im_a + b1120im_b;
            let out12re = b1219re_a - b1219re_b;
            let out12im = b1219im_a + b1219im_b;
            let out13re = b1318re_a - b1318re_b;
            let out13im = b1318im_a + b1318im_b;
            let out14re = b1417re_a - b1417re_b;
            let out14im = b1417im_a + b1417im_b;
            let out15re = b1516re_a - b1516re_b;
            let out15im = b1516im_a + b1516im_b;
            let out16re = b1516re_a + b1516re_b;
            let out16im = b1516im_a - b1516im_b;
            let out17re = b1417re_a + b1417re_b;
            let out17im = b1417im_a - b1417im_b;
            let out18re = b1318re_a + b1318re_b;
            let out18im = b1318im_a - b1318im_b;
            let out19re = b1219re_a + b1219re_b;
            let out19im = b1219im_a - b1219im_b;
            let out20re = b1120re_a + b1120re_b;
            let out20im = b1120im_a - b1120im_b;
            let out21re = b1021re_a + b1021re_b;
            let out21im = b1021im_a - b1021im_b;
            let out22re = b922re_a + b922re_b;
            let out22im = b922im_a - b922im_b;
            let out23re = b823re_a + b823re_b;
            let out23im = b823im_a - b823im_b;
            let out24re = b724re_a + b724re_b;
            let out24im = b724im_a - b724im_b;
            let out25re = b625re_a + b625re_b;
            let out25im = b625im_a - b625im_b;
            let out26re = b526re_a + b526re_b;
            let out26im = b526im_a - b526im_b;
            let out27re = b427re_a + b427re_b;
            let out27im = b427im_a - b427im_b;
            let out28re = b328re_a + b328re_b;
            let out28im = b328im_a - b328im_b;
            let out29re = b229re_a + b229re_b;
            let out29im = b229im_a - b229im_b;
            let out30re = b130re_a + b130re_b;
            let out30im = b130im_a - b130im_b;

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
            chunk[29] = Complex {
                re: out29re,
                im: out29im,
            };
            chunk[30] = Complex {
                re: out30re,
                im: out30im,
            };
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        31
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;

    test_butterfly!(test_butterfly31, f32, Butterfly31, 31, 1e-5);
}
