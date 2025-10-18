// #![feature(duration_millis_float)]
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
// #![feature(duration_millis_float)]
extern crate core;

use realfft::RealFftPlanner;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use std::time::Instant;
use zaft::Zaft;

fn main() {
    let mut data0 = vec![
        Complex::new(5.0, 3.25),
        Complex::new(5.0, -1.0),
        Complex::new(9.6, -2.0),
        Complex::new(12.6, -3.0),
        Complex::new(14.6, -6.0),
    ];
    let mut data = vec![Complex::new(0.0019528865, 0.); 10000];
    // for (k, z) in data.iter_mut().enumerate() {
    //     *z = data0[k % data0.len()];
    // }
    for (i, chunk) in data.iter_mut().enumerate() {
        *chunk = Complex::new(0.0019528865 + i as f32 * 0.1, 0.);
    }

    // let mut real_data = data.iter().map(|x| x.re).collect::<Vec<_>>();
    // let mut real_data_clone = real_data.to_vec();
    // let real_data_ref = real_data.clone();
    //
    // println!("real data {:?}", real_data);
    //
    // let forward_r2c = Zaft::make_r2c_fft_f32(data.len()).unwrap();
    // let inverse_r2c = Zaft::make_c2r_fft_f32(data.len()).unwrap();
    //
    // let mut complex_data = vec![Complex::<f32>::default(); data.len() / 2 + 1];
    // forward_r2c.execute(&real_data, &mut complex_data).unwrap();
    // // println!("r2c {:?}", complex_data);
    // inverse_r2c.execute(&complex_data, &mut real_data).unwrap();
    //
    // real_data = real_data
    //     .iter()
    //     .map(|&x| x * (1.0 / real_data.len() as f32))
    //     .collect();
    //
    // println!("c2r {:?}", real_data);
    //
    // let r_r2c = RealFftPlanner::new().plan_fft_forward(real_data.len());
    // let r_c2r = RealFftPlanner::new().plan_fft_inverse(real_data.len());
    // r_r2c
    //     .process(&mut real_data_clone, &mut complex_data)
    //     .unwrap();
    // r_c2r.process(&mut complex_data, &mut real_data).unwrap();
    //
    // real_data = real_data
    //     .iter()
    //     .map(|&x| x * (1.0 / real_data.len() as f32))
    //     .collect();
    //
    // real_data
    //     .iter()
    //     .zip(real_data_ref)
    //     .enumerate()
    //     .for_each(|(idx, (a, b))| {
    //         assert!((a - b).abs() < 1e-4, "a_re {}, b_re {} at {idx}", a, b);
    //     });

    let o_data = data.clone();

    let mut cvt = data.clone();

    let mut planner = FftPlanner::<f32>::new();
    //
    // for i in 1..1500 {
    //     let mut data = vec![Complex::<f32>::default(); i];
    //     for (k, z) in data.iter_mut().enumerate() {
    //         *z = data0[k % data0.len()];
    //     }
    //     let forward = Zaft::make_forward_fft_f32(data.len()).unwrap();
    //     let new_plan = planner.plan_fft_forward(data.len());
    //     let s0 = Instant::now();
    //     forward.execute(&mut data).unwrap();
    //     let elapsed1 = s0.elapsed();
    //
    //     let s1 = Instant::now();
    //     new_plan.process(&mut data);
    //     let elapsed2 = s1.elapsed();
    //     let diff = elapsed1.as_millis_f32() / elapsed2.as_millis_f32();
    //     if diff > 1.6 {
    //         println!("Timescale was {diff} on {i}");
    //     }
    // }

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
                (a.re - b.re).abs() < 1e-3,
                "a_re {}, b_re {} at {idx}",
                a.re,
                b.re
            );
            assert!(
                (a.im - b.im).abs() < 1e-3,
                "a_im {}, b_im {} at {idx}",
                a.im,
                b.im
            );
        });
}
