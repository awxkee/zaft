/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use criterion::{Criterion, criterion_group, criterion_main};
use num_complex::Complex;
use rand::Rng;
use rustfft::FftPlanner;
use zaft::Zaft;

pub fn criterion_benchmark(c: &mut Criterion) {
    c.benchmark_group("Fft");
    let mut input_power4 = vec![Complex::<f64>::default(); 1024];
    for z in input_power4.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

    let mut input_power4s = vec![Complex::<f32>::default(); 1024];
    for z in input_power4s.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

    let mut input_power2 = vec![Complex::<f64>::default(); 2048];
    for z in input_power2.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

    let mut input_power2s = vec![Complex::<f32>::default(); 2048];
    for z in input_power2s.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

    let mut input_power3 = vec![Complex::<f64>::default(); 2187];
    for z in input_power3.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

    let mut input_power3s = vec![Complex::<f32>::default(); 2187];
    for z in input_power3s.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

    let mut input_power5 = vec![Complex::<f64>::default(); 3125];
    for z in input_power5.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

    let mut input_power5s = vec![Complex::<f32>::default(); 3125];
    for z in input_power5s.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

    let mut input_power6 = vec![Complex::<f64>::default(); 1296];
    for z in input_power6.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

    let mut input_power6s = vec![Complex::<f32>::default(); 1296];
    for z in input_power6s.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

    c.bench_function("rustfft power2", |b| {
        let plan = FftPlanner::new().plan_fft_forward(input_power2.len());
        let mut working = input_power2.to_vec();
        b.iter(|| {
            plan.process(&mut working);
        })
    });

    c.bench_function("zaft power2", |b| {
        let plan = Zaft::make_forward_fft_f64(input_power2.len()).unwrap();
        let mut working = input_power2.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });

    c.bench_function("rustfft power2s", |b| {
        let plan = FftPlanner::new().plan_fft_forward(input_power2s.len());
        let mut working = input_power2s.to_vec();
        b.iter(|| {
            plan.process(&mut working);
        })
    });

    c.bench_function("zaft power2s", |b| {
        let plan = Zaft::make_forward_fft_f32(input_power2s.len()).unwrap();
        let mut working = input_power2s.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });

    c.bench_function("rustfft power4", |b| {
        let plan = FftPlanner::new().plan_fft_forward(input_power4.len());
        let mut working = input_power4.to_vec();
        b.iter(|| {
            plan.process(&mut working);
        })
    });

    c.bench_function("zaft power4", |b| {
        let plan = Zaft::make_forward_fft_f64(input_power4.len()).unwrap();
        let mut working = input_power4.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });

    c.bench_function("rustfft power4s", |b| {
        let plan = FftPlanner::new().plan_fft_forward(input_power4s.len());
        let mut working = input_power4s.to_vec();
        b.iter(|| {
            plan.process(&mut working);
        })
    });

    c.bench_function("zaft power4s", |b| {
        let plan = Zaft::make_forward_fft_f32(input_power4s.len()).unwrap();
        let mut working = input_power4s.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });

    c.bench_function("rustfft power3", |b| {
        let plan = FftPlanner::new().plan_fft_forward(input_power3.len());
        let mut working = input_power3.to_vec();
        b.iter(|| {
            plan.process(&mut working);
        })
    });

    c.bench_function("zaft power3", |b| {
        let plan = Zaft::make_forward_fft_f64(input_power3.len()).unwrap();
        let mut working = input_power3.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });

    c.bench_function("rustfft power3s", |b| {
        let plan = FftPlanner::new().plan_fft_forward(input_power3s.len());
        let mut working = input_power3s.to_vec();
        b.iter(|| {
            plan.process(&mut working);
        })
    });

    c.bench_function("zaft power3s", |b| {
        let plan = Zaft::make_forward_fft_f32(input_power3s.len()).unwrap();
        let mut working = input_power3s.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });

    c.bench_function("rustfft power5", |b| {
        let plan = FftPlanner::new().plan_fft_forward(input_power5.len());
        let mut working = input_power5.to_vec();
        b.iter(|| {
            plan.process(&mut working);
        })
    });

    c.bench_function("zaft power5", |b| {
        let plan = Zaft::make_forward_fft_f64(input_power5.len()).unwrap();
        let mut working = input_power5.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });

    c.bench_function("rustfft power5s", |b| {
        let plan = FftPlanner::new().plan_fft_forward(input_power5s.len());
        let mut working = input_power5s.to_vec();
        b.iter(|| {
            plan.process(&mut working);
        })
    });

    c.bench_function("zaft power5s", |b| {
        let plan = Zaft::make_forward_fft_f32(input_power5s.len()).unwrap();
        let mut working = input_power5s.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });

    c.bench_function("rustfft power6", |b| {
        let plan = FftPlanner::new().plan_fft_forward(input_power6.len());
        let mut working = input_power6.to_vec();
        b.iter(|| {
            plan.process(&mut working);
        })
    });

    c.bench_function("zaft power6", |b| {
        let plan = Zaft::make_forward_fft_f64(input_power6.len()).unwrap();
        let mut working = input_power6.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });

    c.bench_function("rustfft power6s", |b| {
        let plan = FftPlanner::new().plan_fft_forward(input_power6s.len());
        let mut working = input_power6s.to_vec();
        b.iter(|| {
            plan.process(&mut working);
        })
    });

    c.bench_function("zaft power6s", |b| {
        let plan = Zaft::make_forward_fft_f32(input_power6s.len()).unwrap();
        let mut working = input_power6s.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
