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

/// RDFT for N=9, canonical data-flow (Figure 7).
/// φ₉ = 2π/9
///
/// Notations from paper:
///   t1 = x1+x8, t2 = x2+x7, t3 = x3+x6, t4 = x4+x5
///   t5 = x4-x5, t6 = x3-x6, t7 = x2-x7, t8 = x1-x8
///
/// First sub-block (real parts y1,y2,y4):
///   d5 = (cos φ + cos 2φ + cos 4φ)/3
///   d6 = (2cos φ - cos 2φ - cos 4φ)/3
///   d7 = (-cos φ + 2cos 2φ - cos 4φ)/3  ← note: from paper eq(82)
///   d8 = (-cos φ - cos 2φ + 2cos 4φ)/3  ← note: cos 3φ = cos(2π/3) = -0.5 exactly
///
/// Second sub-block (imaginary parts y5,y7,y8):
///   d9  = (sin φ - sin 2φ + sin 4φ)/3
///   d10 = (-sin φ - 2sin 2φ - sin 4φ)/3  ← from paper eq(83)
///   d11 = (2sin φ + sin 2φ - sin 4φ)/3
///   d12 = (-sin φ + sin 2φ + 2sin 4φ)/3
///
/// Special: d13 = d14 = sin 3φ = sin(2π/3) = √3/2
///
/// Output layout:
///   y[0] = X[0]  DC
///   y[1] = X[1].re,  y[2] = X[2].re,  y[3] = X[3].re
///   y[4] = X[4].re
///   y[5] = X[1].im,  y[6] = X[2].im,  y[7] = X[3].im
///   y[8] = X[4].im
pub fn rdft9(x: [f64; 9]) -> [f64; 9] {
    let phi = 2.0 * PI / 9.0;
    let phi2 = 2.0 * phi;
    let phi3 = 3.0 * phi; // = 2π/3
    let phi4 = 4.0 * phi;

    let t1 = x[1] + x[8];
    let t2 = x[2] + x[7];
    let t3 = x[3] + x[6];
    let t4 = x[4] + x[5];
    let t5 = x[4] - x[5];
    let t6 = x[3] - x[6];
    let t7 = x[2] - x[7];
    let t8 = x[1] - x[8];

    let c1 = phi.cos();
    let s1 = phi.sin();
    let c2 = phi2.cos();
    let s2 = phi2.sin();
    let s3 = phi3.sin(); // = √3/2, also d13=d14
    let c4 = phi4.cos();
    let s4 = phi4.sin(); // sin(8π/9)=sin(π/9)

    // DC
    let y0 = x[0] + t1 + t2 + t3 + t4;

    // X[3] sub-block (DFT-3 on stride-3):
    let y3 = x[0] + t3 - 0.5 * (t1 + t2 + t4);
    let y7 = -s3 * (t8 - t7 + t5); // X[3].im

    // First sub-block: X[1].re, X[2].re, X[4].re
    let h0 = c1 + c2;
    let d5 = (h0 + c4) / 3.0;
    let d6 = (2.0 * c1 - c2 - c4) / 3.0;
    let d7 = (-c1 + 2.0 * c2 - c4) / 3.0;
    let d8 = (-h0 + 2.0 * c4) / 3.0;

    let r0 = t1 + t2 + t4;
    let r1 = t1 - t4;
    let r2 = t2 - t4;
    let r3 = -t1 + t2;

    let m1 = d5 * r0;
    let m2 = d6 * r1;
    let m3 = d7 * r2;
    let m4 = d8 * r3;

    let re_base = x[0] - 0.5 * t3 + m1;
    let y1 = re_base + m2 + m3;
    let y2 = re_base - m2 + m4;
    let y4 = re_base - m3 - m4;

    // Imaginary outputs — direct from DFT definition:
    let y5 = -(s1 * t8 + s2 * t7 + s3 * t6 + s4 * t5); // X[1].im
    let y6 = -(s2 * t8 + s4 * t7 - s3 * t6 - s1 * t5); // X[2].im
    // y7 already computed above                    // X[3].im
    let y8 = -(s4 * t8 - s1 * t7 + s3 * t6 - s2 * t5); // X[4].im

    [y0, y1, y2, y3, y4, y5, y6, y7, y8]
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn naive_dft9(x: [f64; 9]) -> Vec<Complex<f64>> {
        (0..5)
            .map(|k| {
                (0..9usize).fold(Complex::new(0.0, 0.0), |acc, n| {
                    let angle = -2.0 * PI * k as f64 * n as f64 / 9.0;
                    acc + x[n] * Complex::new(angle.cos(), angle.sin())
                })
            })
            .collect()
    }

    #[test]
    fn test_rdft9() {
        let x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y = rdft9(x);
        let ref_ = naive_dft9(x);

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
            (y[3] - ref_[3].re).abs() < 1e-10,
            "X[3].re {} vs {}",
            y[3],
            ref_[3].re
        );
        assert!(
            (y[4] - ref_[4].re).abs() < 1e-10,
            "X[4].re {} vs {}",
            y[4],
            ref_[4].re
        );
        assert!(
            (y[5] - ref_[1].im).abs() < 1e-10,
            "X[1].im {} vs {}",
            y[5],
            ref_[1].im
        );
        assert!(
            (y[6] - ref_[2].im).abs() < 1e-10,
            "X[2].im {} vs {}",
            y[6],
            ref_[2].im
        );
        assert!(
            (y[7] - ref_[3].im).abs() < 1e-10,
            "X[3].im {} vs {}",
            y[7],
            ref_[3].im
        );
        assert!(
            (y[8] - ref_[4].im).abs() < 1e-10,
            "X[4].im {} vs {}",
            y[8],
            ref_[4].im
        );
    }
}
