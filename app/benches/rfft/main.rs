/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use criterion::measurement::WallTime;
use criterion::{BatchSize, BenchmarkGroup, Criterion, criterion_group, criterion_main};
use num_complex::Complex;
use rand::Rng;
use rustfft::FftPlanner;
use std::time::Duration;
use zaft::Zaft;

pub fn bench_zaft_average(c: &mut Criterion) {
    c.bench_function("zaft 1..=1500 double", |b| {
        b.iter_batched(
            || {
                // Prepare all inputs and FFT plans
                (1..=1500)
                    .map(|n| {
                        let input: Vec<Complex<f64>> =
                            (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();
                        let fft = Zaft::make_forward_fft_f64(n).unwrap();
                        (input, fft)
                    })
                    .collect::<Vec<_>>()
            },
            |mut plans_and_inputs| {
                // Execute FFTs for all sizes
                for (i, (input, fft)) in plans_and_inputs.iter().enumerate() {
                    let mut c = input.to_vec();
                    match fft.execute(&mut c) {
                        Ok(_) => {}
                        Err(err) => panic!("err: {err} on {i}"),
                    };
                }
            },
            BatchSize::LargeInput,
        );
    });
}

fn check_power_group(c: &mut Criterion, n: usize, group: String) {
    let mut input_power = vec![Complex::<f64>::default(); n];
    for z in input_power.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

    c.bench_function(format!("rustfft {group}").as_str(), |b| {
        let plan = FftPlanner::new().plan_fft_forward(input_power.len());
        let mut working = input_power.to_vec();
        b.iter(|| {
            plan.process(&mut working);
        })
    });

    c.bench_function(format!("zaft {group}").as_str(), |b| {
        let plan = Zaft::make_inverse_fft_f64(input_power.len()).unwrap();
        let mut working = input_power.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });

    c.bench_function(format!("rustfft {group}s").as_str(), |b| {
        let plan = FftPlanner::new().plan_fft_forward(input_power.len());
        let s = input_power
            .iter()
            .map(|&x| Complex::new(x.re as f32, x.im as f32))
            .collect::<Vec<_>>();
        let mut working = s.to_vec();
        b.iter(|| {
            plan.process(&mut working);
        })
    });

    c.bench_function(format!("zaft {group}s").as_str(), |b| {
        let plan = Zaft::make_inverse_fft_f32(input_power.len()).unwrap();
        let s = input_power
            .iter()
            .map(|&x| Complex::new(x.re as f32, x.im as f32))
            .collect::<Vec<_>>();
        let mut working = s.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });
}

fn check_power_groups(c: &mut BenchmarkGroup<WallTime>, n: usize, group: String) {
    let mut input_power = vec![f64::default(); n];
    for z in input_power.iter_mut() {
        *z = rand::rng().random();
    }

    c.bench_function(format!("zaft {group}s").as_str(), |b| {
        let plan = Zaft::make_r2c_fft_f32(input_power.len()).unwrap();
        let s = input_power.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let mut output = vec![Complex::new(0.0, 0.0); n / 2 + 1];
        let working = s.to_vec();
        b.iter(|| {
            plan.execute(&working, &mut output).unwrap();
        })
    });
}

fn check_power_groupd(c: &mut BenchmarkGroup<WallTime>, n: usize, group: String) {
    let mut input_power = vec![f64::default(); n];
    for z in input_power.iter_mut() {
        *z = rand::rng().random();
    }

    c.bench_function(format!("zaft {group}d").as_str(), |b| {
        let plan = Zaft::make_r2c_fft_f64(input_power.len()).unwrap();
        let mut output = vec![Complex::new(0.0, 0.0); n / 2 + 1];
        let working = input_power.to_vec();
        b.iter(|| {
            plan.execute(&working, &mut output).unwrap();
        })
    });
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("group");
    let c = group
        .measurement_time(Duration::from_millis(750))
        .warm_up_time(Duration::from_millis(750));

    check_power_groups(c, 1803, "1803".to_string());
    check_power_groupd(c, 1803, "1803".to_string());

    check_power_groups(c, 2187, "2187".to_string());
    check_power_groupd(c, 2187, "2187".to_string());
    check_power_groups(c, 2565, "2565".to_string());
    check_power_groupd(c, 2565, "2565".to_string());

    // check_power_groups(c, 14, "14".to_string());
    // check_power_groupd(c, 14, "14".to_string());

    check_power_groups(c, 8, "8".to_string());
    check_power_groupd(c, 8, "8".to_string());
    check_power_groups(c, 16, "16".to_string());
    check_power_groupd(c, 16, "16".to_string());
    check_power_groups(c, 32, "32".to_string());
    check_power_groupd(c, 32, "32".to_string());
    check_power_groups(c, 64, "64".to_string());
    check_power_groups(c, 128, "128".to_string());
    check_power_groups(c, 256, "256".to_string());
    check_power_groups(c, 512, "512".to_string());
    check_power_groups(c, 1024, "1024".to_string());
    check_power_groups(c, 2048, "2048".to_string());
    check_power_groups(c, 4096, "4096".to_string());
    check_power_groups(c, 8192, "8192".to_string());
    check_power_groups(c, 16384, "16384".to_string());
    check_power_groups(c, 32768, "32768".to_string());
    check_power_groups(c, 65536, "65536".to_string());
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
