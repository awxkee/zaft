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

pub fn rfft8(x: [f64; 8]) -> [f64; 8] {
    let e0 = x[0] + x[4]; // a
    let e1 = x[0] - x[4]; // b
    let e2 = x[2] + x[6]; // c
    let e3 = x[2] - x[6]; // d  (will be multiplied by -j in next stage)

    let o0 = x[1] + x[5];
    let o1 = x[1] - x[5];
    let o2 = x[3] + x[7];
    let o3 = x[3] - x[7];

    let re_x0 = e0 + e2;
    let re_x4 = e0 - e2;
    let re_x2r = e1;
    let re_x2i = -e3;

    let odd_f0r = o0 + o2;
    let odd_f2r = o0 - o2;
    let odd_f1r = o1;
    let odd_f1i = -o3;

    let sqrt2_inv = std::f64::consts::FRAC_1_SQRT_2; // 1/√2

    let x0 = re_x0 + odd_f0r;

    let x4 = re_x0 - odd_f0r;

    let tw2_f2i = -odd_f2r;

    let tw1_f1r = sqrt2_inv * (odd_f1r + odd_f1i);
    let tw1_f1i = sqrt2_inv * (odd_f1i - odd_f1r);

    let x1r = re_x2r + tw1_f1r;
    let x1i = re_x2i + tw1_f1i;

    let x5r = re_x2r - tw1_f1r;
    let x5i = re_x2i - tw1_f1i;

    let x2r_fixed = re_x4;
    let x2i_fixed = tw2_f2i;

    [x0, x1r, x2r_fixed, x1i, x4, x5r, x2i_fixed, x5i]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_rfft8(x: [f64; 8]) -> [Complex<f64>; 5] {
        std::array::from_fn(|k| {
            (0..8usize).fold(Complex::new(0.0, 0.0), |acc, n| {
                let angle = -2.0 * PI * k as f64 * n as f64 / 8.0;
                acc + x[n] * Complex::new(angle.cos(), angle.sin())
            })
        })
    }

    #[test]
    fn test_rfft8_vs_dft() {
        let x = [1.0, 2.5, 3.1, -4.0, -2.53, 6.0, 7.0, 6.23];
        let out = rfft8(x);
        let ref_ = naive_rfft8(x);

        // X[0]
        assert!((out[0] - ref_[0].re).abs() < 1e-10, "X[0].re");
        // X[1]
        assert!((out[1] - ref_[1].re).abs() < 1e-10, "X[1].re");
        assert!((out[3] - ref_[1].im).abs() < 1e-10, "X[1].im");
        // X[2]
        assert!((out[2] - ref_[2].re).abs() < 1e-10, "X[2].re");
        assert!((out[6] - ref_[2].im).abs() < 1e-10, "X[2].im");
        // X[4]
        assert!((out[4] - ref_[4].re).abs() < 1e-10, "X[4].re");
    }
}
