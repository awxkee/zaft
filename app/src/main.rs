/*
 * // Copyright (c) Radzivon Bartoshyk 6/2025. All rights reserved.
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

extern crate core;

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use zaft::Zaft;

fn main() {
    let mut data0 = vec![
        Complex::new(5.0, 3.25),
        Complex::new(5.0, -1.0),
        Complex::new(9.6, -2.0),
        Complex::new(12.6, -3.0),
        Complex::new(14.6, -6.0),
    ];
    let mut data = vec![Complex::<f32>::default(); 1295];
    for (k, z) in data.iter_mut().enumerate() {
        *z = data0[k % data0.len()];
    }

    let o_data = data.clone();

    let mut cvt = data.clone();

    let forward = Zaft::make_forward_fft_f32(cvt.len()).unwrap();
    let inverse = Zaft::make_inverse_fft_f32(cvt.len()).unwrap();

    let mut planner = FftPlanner::<f32>::new();

    let planned_fft = planner.plan_fft_forward(data.len());
    let planned_fft_inv = planner.plan_fft_inverse(data.len());

    forward.execute(&mut data).unwrap();
    planned_fft.process(&mut cvt);

    println!("Rust fft forward -----");

    // for (i, val) in cvt.iter().enumerate() {
    //     println!("X[{}] = {}", i, val);
    // }

    data = data
        .iter()
        .map(|&x| x * (1.0 / f32::sqrt(data.len() as f32)))
        .collect();
    cvt = cvt
        .iter()
        .map(|&x| x * (1.0 / f32::sqrt(cvt.len() as f32)))
        .collect();

    println!("Mine inverse -----");

    // assert_eq!(cvt, data);

    inverse.execute(&mut data).unwrap();

    data = data
        .iter()
        .map(|&x| x * (1.0 / f32::sqrt(data.len() as f32)))
        .collect();

    // for (i, val) in data.iter().enumerate() {
    //     println!("X[{}] = {}", i, val);
    // }

    println!("Rust fft inv -----");

    planned_fft_inv.process(&mut cvt);
    cvt = cvt
        .iter()
        .map(|&x| x * (1.0 / f32::sqrt(cvt.len() as f32)))
        .collect();

    // for (i, val) in cvt.iter().enumerate() {
    //     println!("X[{}] = {}", i, val);
    // }

    data.iter()
        .zip(o_data)
        .enumerate()
        .for_each(|(idx, (a, b))| {
            assert!(
                (a.re - b.re).abs() < 1e-4,
                "a_re {}, b_re {} at {idx}",
                a.re,
                b.re
            );
            assert!(
                (a.im - b.im).abs() < 1e-4,
                "a_im {}, b_im {} at {idx}",
                a.im,
                b.im
            );
        });
}
