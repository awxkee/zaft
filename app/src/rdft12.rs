/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

use num_complex::Complex;

/// RDFT for N=12
/// Output: complex[0..7] where complex[k] = X[k] for k=0..6
/// X[0] and X[6] are real (DC and Nyquist)
pub fn rdft12(x: [f64; 12]) -> [Complex<f64>; 7] {
    const SQRT3_2: f64 = f64::from_bits(0x3febb67ae8584caa); // √3/2
    const HALF: f64 = 0.5;

    let s1 = x[1] + x[11];
    let d1 = x[1] - x[11];
    let s2 = x[2] + x[10];
    let d2 = x[2] - x[10];
    let s3 = x[3] + x[9];
    let d3 = x[3] - x[9];
    let s4 = x[4] + x[8];
    let d4 = x[4] - x[8];
    let s5 = x[5] + x[7];
    let d5 = x[5] - x[7];
    let s6 = x[6];

    let sqrt_s1 = SQRT3_2 * s1;
    let sqrt_s5 = SQRT3_2 * s5;
    let hs1 = HALF * s1;
    let hs2 = HALF * s2;
    let hs4 = HALF * s4;
    let hs5 = HALF * s5;
    let p_hs2_hs4 = hs2 - hs4;
    let n_hs2_hs4 = -hs2 - hs4;
    let x0_ns6 = x[0] - s6;
    let x0_ps6 = x[0] + s6;

    let hd1 = HALF * d1;
    let hd5 = HALF * d5;
    let hd1_hd5 = hd1 + hd5;
    let hd1_hd5_d3 = hd1_hd5 + d3;
    let sqrt_d2pd4 = SQRT3_2 * (d2 + d4);
    let sqrt3_d1nd5 = SQRT3_2 * (d1 - d5);
    let sqrt3_d2nd4 = SQRT3_2 * (d2 - d4);

    // DC
    let s2_s6 = s2 + s6;
    let y0 = x[0] + s1 + s2_s6 + s3 + s4 + s5;

    // Real parts

    let j0 = sqrt_s1 - sqrt_s5;
    let j1 = s3 - hs5;
    let j1_m_hs1 = j1 - hs1;

    let p_hs2_x0_ns = p_hs2_hs4 + x0_ns6;
    let x0_ps6_p_n_hs2_hs4 = x0_ps6 + n_hs2_hs4;

    let y1r = j0 + p_hs2_x0_ns;
    let y2r = x0_ps6_p_n_hs2_hs4 - j1_m_hs1;
    let y3r = x[0] + s4 - s2_s6;
    let y4r = x0_ps6_p_n_hs2_hs4 + j1_m_hs1;
    let y5r = p_hs2_x0_ns - j0;
    let y6r = x[0] - s1 + s2_s6 - s3 + s4 - s5;

    // Imaginary parts
    let sqrt3_sum = sqrt3_d1nd5 + sqrt3_d2nd4;
    let sqrt3_dif = sqrt3_d1nd5 - sqrt3_d2nd4;
    let neg_hd1_hd5_d3 = -hd1_hd5_d3;
    let y1i = neg_hd1_hd5_d3 - sqrt_d2pd4;
    let y2i = -sqrt3_sum;
    let y3i = -(d1 - d3 + d5);
    let y4i = -sqrt3_dif;
    let y5i = neg_hd1_hd5_d3 + sqrt_d2pd4;

    [
        Complex::new(y0, 0.0),
        Complex::new(y1r, y1i),
        Complex::new(y2r, y2i),
        Complex::new(y3r, y3i),
        Complex::new(y4r, y4i),
        Complex::new(y5r, y5i),
        Complex::new(y6r, 0.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn naive_dft12(x: [f64; 12]) -> Vec<Complex<f64>> {
        (0..7)
            .map(|k| {
                (0..12usize).fold(Complex::new(0.0, 0.0), |acc, n| {
                    let angle = -2.0 * PI * k as f64 * n as f64 / 12.0;
                    acc + x[n] * Complex::new(angle.cos(), angle.sin())
                })
            })
            .collect()
    }

    #[test]
    fn test_rdft12() {
        let x = [
            1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let got = rdft12(x);
        let ref_ = naive_dft12(x);
        for k in 0..7 {
            assert!(
                (got[k].re - ref_[k].re).abs() < 1e-10,
                "X[{k}].re: {} vs {}",
                got[k].re,
                ref_[k].re
            );
            assert!(
                (got[k].im - ref_[k].im).abs() < 1e-10,
                "X[{k}].im: {} vs {}",
                got[k].im,
                ref_[k].im
            );
        }
    }

    #[test]
    fn test_rdft12_impulse() {
        let mut x = [0.0f64; 12];
        x[0] = 1.0;
        let got = rdft12(x);
        // All bins should be 1+0i
        for k in 0..7 {
            assert!((got[k].re - 1.0).abs() < 1e-10, "X[{k}].re");
            assert!(got[k].im.abs() < 1e-10, "X[{k}].im");
        }
    }

    #[test]
    fn test_rdft12_dc() {
        let x = [1.0f64; 12];
        let got = rdft12(x);
        assert!((got[0].re - 12.0).abs() < 1e-10, "X[0]");
        for k in 1..7 {
            assert!(got[k].re.abs() < 1e-10, "X[{k}].re should be 0");
            assert!(got[k].im.abs() < 1e-10, "X[{k}].im should be 0");
        }
    }
}
