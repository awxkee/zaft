/*
 * // Copyright (c) Radzivon Bartoshyk 1/2026. All rights reserved.
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

fn twiddle(k: usize, n: usize) -> Complex<f64> {
    // Generates the twiddle factor W_N^k = exp(-j * 2*PI * k / N)
    let angle = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
    Complex::new(angle.cos(), angle.sin())
}

pub fn make_twiddles(n: usize) -> Vec<Complex<f64>> {
    let mut w = vec![Complex { re: 0.0, im: 0.0 }; n];
    let step = -2.0 * std::f64::consts::PI / n as f64;
    for k in 0..n {
        let a = step * k as f64;
        w[k] = Complex {
            re: a.cos(),
            im: a.sin(),
        };
    }
    w
}

pub fn split_radix_fft(x: &mut [Complex<f64>], w: &[Complex<f64>]) {
    let n = x.len();
    if n == 1 {
        return;
    }

    if n == 2 {
        let a = x[0];
        let b = x[1];
        x[0] = Complex {
            re: a.re + b.re,
            im: a.im + b.im,
        };
        x[1] = Complex {
            re: a.re - b.re,
            im: a.im - b.im,
        };
        return;
    }

    let n2 = n / 2;
    let n4 = n / 4;

    // Even
    let mut even = Vec::with_capacity(n2);
    for i in (0..n).step_by(2) {
        even.push(x[i]);
    }

    // Odd 1 mod 4
    let mut odd1 = Vec::with_capacity(n4);
    for i in (1..n).step_by(4) {
        odd1.push(x[i]);
    }

    // Odd 3 mod 4
    let mut odd3 = Vec::with_capacity(n4);
    for i in (3..n).step_by(4) {
        odd3.push(x[i]);
    }

    split_radix_fft(&mut even, w);
    split_radix_fft(&mut odd1, w);
    split_radix_fft(&mut odd3, w);

    // Recombine
    {
        let k = 0;
        let e0 = even[k];
        let e1 = even[k + n4];

        let o1 = odd1[k];
        let o3 = odd3[k];

        let t0 = Complex {
            re: o1.re + o3.re,
            im: o1.im + o3.im,
        };
        let t1 = Complex {
            re: o1.im - o3.im,
            im: o3.re - o1.re,
        };

        x[k] = Complex {
            re: e0.re + t0.re,
            im: e0.im + t0.im,
        };
        x[k + n4] = Complex {
            re: e1.re + t1.re,
            im: e1.im + t1.im,
        };
        x[k + n2] = Complex {
            re: e0.re - t0.re,
            im: e0.im - t0.im,
        };
        x[k + n2 + n4] = Complex {
            re: e1.re - t1.re,
            im: e1.im - t1.im,
        };
    }

    for k in 1..n4 {
        let e0 = even[k];
        let e1 = even[k + n4];

        let w1 = w[k * (w.len() / n)];
        let w3 = w[k * 3 * (w.len() / n)];

        let o1 = odd1[k] * w1;
        let o3 = odd3[k] * w3;

        let t0 = Complex {
            re: o1.re + o3.re,
            im: o1.im + o3.im,
        };
        let t1 = Complex {
            re: o1.im - o3.im,
            im: o3.re - o1.re,
        };

        x[k] = Complex {
            re: e0.re + t0.re,
            im: e0.im + t0.im,
        };
        x[k + n4] = Complex {
            re: e1.re + t1.re,
            im: e1.im + t1.im,
        };
        x[k + n2] = Complex {
            re: e0.re - t0.re,
            im: e0.im - t0.im,
        };
        x[k + n2 + n4] = Complex {
            re: e1.re - t1.re,
            im: e1.im - t1.im,
        };
    }
}
