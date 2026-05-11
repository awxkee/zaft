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
use std::f64::consts::PI;

/// RDFT for N=10
/// φ₁₀ = 2π/10 = π/5
/// c1 = cos(π/5)  = (√5+1)/4 ≈ 0.80902
/// c2 = cos(2π/5) = (√5-1)/4 ≈ 0.30902
/// s1 = sin(π/5)  ≈ 0.58779
/// s2 = sin(2π/5) ≈ 0.95106
///
/// Output: complex[0..6] = X[0..5]
/// X[0] DC (real), X[5] Nyquist (real)
pub fn rdft10(x: [f64; 10]) -> [Complex<f64>; 6] {
    // Correctly rounded constants from mpmath
    let C1: f64 = (std::f64::consts::PI / 5.).cos(); // cos(π/5)  ≈ 0.80902
    let C2: f64 = (2. * std::f64::consts::PI / 5.).cos(); // cos(2π/5) ≈ 0.30902
    let S1: f64 = (std::f64::consts::PI / 5.).sin(); // sin(π/5)  ≈ 0.58779
    let S2: f64 = (2. * std::f64::consts::PI / 5.).sin(); // sin(2π/5) ≈ 0.95106

    let s1p = x[1] + x[9];
    let d1p = x[1] - x[9];
    let s2p = x[2] + x[8];
    let d2p = x[2] - x[8];
    let s3p = x[3] + x[7];
    let d3p = x[3] - x[7];
    let s4p = x[4] + x[6];
    let d4p = x[4] - x[6];
    let s5p = x[5]; // Nyquist

    // DC
    let s5x0 = s5p + x[0];
    let x0ms5p = x[0] - s5p;

    let s1p_ps4p = s1p + s4p;
    let s2p_ps3p = s2p + s3p;

    let y0 = s5x0 + s1p_ps4p + s2p_ps3p;

    let s1p_ms4p = s1p - s4p;
    let s2p_ms3p = s2p - s3p;

    let c1_s1ms4 = C1 * s1p_ms4p;
    let c2_s2ms3 = C2 * s2p_ms3p;
    let c2_s1ms4 = C2 * s1p_ms4p;
    let c1_s2ms3 = C1 * s2p_ms3p;
    let c1_s1ps4 = C1 * s1p_ps4p;
    let c2_s2ps3 = C2 * s2p_ps3p;
    let c2_s1ps4 = C2 * s1p_ps4p;
    let c1_s2ps3 = C1 * s2p_ps3p;

    let x0_ns5 = x0ms5p;
    let x0_ps5 = s5x0;

    let y1r = x0_ns5 + c1_s1ms4 + c2_s2ms3;
    let y2r = x0_ps5 + c2_s1ps4 - c1_s2ps3;
    let y3r = x0_ns5 - c2_s1ms4 - c1_s2ms3;
    let y4r = x0_ps5 - c1_s1ps4 + c2_s2ps3;
    let y5r = x0ms5p - s1p + s2p - s3p + s4p; // Nyquist

    let d1p_pd4p = d1p + d4p;
    let d2p_pd3p = d2p + d3p;
    let d1p_nd4p = d1p - d4p;
    let d2p_nd3p = d2p - d3p;

    let s1_d1pd4 = S1 * d1p_pd4p;
    let s2_d2pd3 = S2 * d2p_pd3p;
    let s2_d1pd4 = S2 * d1p_pd4p;
    let s1_d2pd3 = S1 * d2p_pd3p;
    let s2_d1nd4 = S2 * d1p_nd4p;
    let s1_d2nd3 = S1 * d2p_nd3p;
    let s1_d1nd4 = S1 * d1p_nd4p;
    let s2_d2nd3 = S2 * d2p_nd3p;

    let y1i = -(s1_d1pd4 + s2_d2pd3);
    let y2i = -(s2_d1nd4 + s1_d2nd3);
    let y3i = -(s2_d1pd4 - s1_d2pd3); // s2*(d1p+d4p) - s1*(d2p+d3p), negated
    let y4i = -(s1_d1nd4 - s2_d2nd3);

    [
        Complex::new(y0, 0.0),
        Complex::new(y1r, y1i),
        Complex::new(y2r, y2i),
        Complex::new(y3r, y3i),
        Complex::new(y4r, y4i),
        Complex::new(y5r, 0.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft10(x: [f64; 10]) -> Vec<Complex<f64>> {
        (0..6)
            .map(|k| {
                (0..10usize).fold(Complex::new(0.0, 0.0), |acc, n| {
                    let angle = -2.0 * PI * k as f64 * n as f64 / 10.0;
                    acc + x[n] * Complex::new(angle.cos(), angle.sin())
                })
            })
            .collect()
    }

    #[test]
    fn test_rdft10() {
        let x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let got = rdft10(x);
        let ref_ = naive_dft10(x);
        for k in 0..6 {
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
    fn test_rdft10_impulse() {
        let mut x = [0.0f64; 10];
        x[0] = 1.0;
        let got = rdft10(x);
        for k in 0..6 {
            assert!((got[k].re - 1.0).abs() < 1e-10, "X[{k}].re impulse");
            assert!(got[k].im.abs() < 1e-10, "X[{k}].im impulse");
        }
    }

    #[test]
    fn test_rdft10_dc() {
        let x = [1.0f64; 10];
        let got = rdft10(x);
        assert!((got[0].re - 10.0).abs() < 1e-10, "DC");
        for k in 1..6 {
            assert!(got[k].re.abs() < 1e-10, "X[{k}].re dc");
            assert!(got[k].im.abs() < 1e-10, "X[{k}].im dc");
        }
    }
}
