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

pub fn rdft6(x: [f64; 6]) -> [f64; 6] {
    let sqrt3_2 = (PI / 3.0_f64).sin();

    let s03 = x[0] + x[3];
    let s14 = x[1] + x[4];
    let s25 = x[2] + x[5];
    let d03 = x[0] - x[3];
    let d14 = x[1] - x[4];
    let d25 = x[2] - x[5];
    let d15 = x[1] - x[5]; // new
    let d24 = x[2] - x[4]; // new

    let y0 = s03 + s14 + s25; // X[0] DC
    let y3 = d03 - d14 + d25;
    let y1 = d03 + 0.5 * d14 - 0.5 * d25; // X[1].re
    let y2 = s03 - 0.5 * s14 - 0.5 * s25; // X[2].re
    let y4 = -sqrt3_2 * (d14 + d25); // X[1].im
    let y5 = sqrt3_2 * (d24 - d15);

    [y0, y1, y2, y3, y4, y5]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft6(x: [f64; 6]) -> Vec<num_complex::Complex<f64>> {
        (0..4)
            .map(|k| {
                (0..6usize).fold(num_complex::Complex::new(0.0, 0.0), |acc, n| {
                    let angle = -2.0 * PI * k as f64 * n as f64 / 6.0;
                    acc + x[n] * num_complex::Complex::new(angle.cos(), angle.sin())
                })
            })
            .collect()
    }

    #[test]
    fn test_rdft6() {
        let x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = rdft6(x);
        let ref_ = naive_dft6(x);

        assert!((y[0] - ref_[0].re).abs() < 1e-10, "X[0] {}", y[0]);
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
            (y[3] - ref_[3].re).abs() < 1e-10,
            "X[3] {} vs {}",
            y[3],
            ref_[3].re
        );
        assert!(
            (y[4] - ref_[1].im).abs() < 1e-10,
            "X[1].im {} vs {}",
            y[4],
            ref_[1].im
        );
        assert!(
            (y[5] - ref_[2].im).abs() < 1e-10,
            "X[2].im {} vs {}",
            y[5],
            ref_[2].im
        );
    }
}
