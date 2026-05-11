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

use std::f64::consts::PI;

/// RDFT for N=5, canonical data-flow diagram (Figure 3).
///
/// Output layout:
///   y[0] = X[0]     (DC, real)
///   y[1] = X[1].re
///   y[2] = X[2].re
///   y[3] = X[1].im
///   y[4] = X[2].im
pub fn rdft5(x: [f64; 5]) -> [f64; 5] {
    // Recompute inline to be safe — replace with constants once verified
    let phi = 2.0 * PI / 5.0;
    let phi2 = 2.0 * phi;

    let d1 = (phi.cos() + phi2.cos()) / 2.0 - 1.0; // -1.25
    let d2 = (phi.cos() - phi2.cos()) / 2.0; //  0.559016...

    let s14 = x[1] + x[4]; // x1 + x4
    let s23 = x[2] + x[3]; // x2 + x3
    let d14 = x[1] - x[4]; // x1 - x4  (feeds imaginary path)
    let d23 = x[2] - x[3]; // x2 - x3  (feeds imaginary path)

    let y0 = x[0] + s14 + s23;

    // Real part intermediates (feed d1 and d2 nodes)
    let r_sum = s14 + s23; // x1+x2+x3+x4
    let r_diff = s14 - s23; // (x1+x4)-(x2+x3)

    // d1 and d2 applied
    let d1_out = d1 * r_sum; // d1*(s14+s23)
    let d2_out = d2 * r_diff; // d2*(s14-s23)

    let real_base = y0 + d1_out; // x0 + s14 + s23 + d1*(s14+s23) = y0 + d1_out

    let y1 = real_base + d2_out; // X[1].re
    let y2 = real_base - d2_out; // X[2].re

    // let y3 = -phi.sin() * d14 - phi2.sin() * d23;   // X[1].im  ✓
    // let y4 = -phi2.sin() * d14 + phi.sin() * d23;   // X[2].im  ✓

    let sin_phi = f64::from_bits(0x3FD73FD61D9DF543) + f64::from_bits(0x3FE2CF2304755A5E); // reconstruct from constants
    let sin_2phi = f64::from_bits(0x3FE2CF2304755A5E);

    let y3 = -(sin_phi * d14 + sin_2phi * d23); // X[1].im
    let y4 = -(sin_2phi * d14 - sin_phi * d23); // X[2].im

    [y0, y1, y2, y3, y4]
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn naive_dft5(x: [f64; 5]) -> Vec<Complex<f64>> {
        (0..3)
            .map(|k| {
                (0..5usize).fold(Complex::new(0.0, 0.0), |acc, n| {
                    let angle = -2.0 * PI * k as f64 * n as f64 / 5.0;
                    acc + x[n] * Complex::new(angle.cos(), angle.sin())
                })
            })
            .collect()
    }

    #[test]
    fn test_rdft5() {
        let x = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let y = rdft5(x);
        let ref_ = naive_dft5(x);

        assert!(
            (y[0] - ref_[0].re).abs() < 1e-10,
            "X[0] {} vs {}",
            y[0],
            ref_[0].re
        );
        assert!(
            (y[1] - ref_[1].re).abs() < 1e-10,
            "X[1].re {} vs {}",
            y[1],
            ref_[1].re
        );
        assert!(
            (y[2] - ref_[2].re).abs() < 1e-10,
            "X[2].re {} vs {}",
            y[2],
            ref_[2].re
        );
        assert!(
            (y[3] - ref_[1].im).abs() < 1e-10,
            "X[1].im {} vs {}",
            y[3],
            ref_[1].im
        );
        assert!(
            (y[4] - ref_[2].im).abs() < 1e-10,
            "X[2].im {} vs {}",
            y[4],
            ref_[2].im
        );
    }
}
