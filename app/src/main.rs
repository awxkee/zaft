mod split_radix;

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

use crate::split_radix::{make_twiddles, split_radix_fft};
use criterion::{BatchSize, Criterion};
use num_traits::Zero;
use primal_check::miller_rabin;
use rand::Rng;
use realfft::RealFftPlanner;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use std::fmt::format;
use std::hint::black_box;
use std::time::{Duration, Instant};
use zaft::{FftDirection, Zaft};

fn check_power_group(c: &mut Criterion, n: usize, group: String) {
    let mut input_power = vec![Complex::<f64>::default(); n];
    for z in input_power.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

    c.bench_function(format!("zaft {group}s").as_str(), |b| {
        let plan = Zaft::make_inverse_fft_f64(input_power.len()).unwrap();
        let s = input_power
            .iter()
            .map(|&x| Complex::new(x.re, x.im))
            .collect::<Vec<_>>();
        let mut working = s.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });
}

fn check_power_groups(c: &mut Criterion, n: usize, group: String) {
    let mut input_power = vec![Complex::<f32>::default(); n];
    for z in input_power.iter_mut() {
        *z = Complex {
            re: rand::rng().random(),
            im: rand::rng().random(),
        };
    }

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

pub fn bench_zaft_averages(c: &mut Criterion) {
    c.bench_function("zaft avg1", |b| {
        b.iter_batched(
            || {
                // Prepare all inputs and FFT plans
                (500..=1500)
                    .map(|n| {
                        let input: Vec<Complex<f32>> =
                            (0..n).map(|i| Complex::new(i as f32, 0.0)).collect();
                        let fft = Zaft::make_forward_fft_f32(n).unwrap();
                        (input, fft)
                    })
                    .collect::<Vec<_>>()
            },
            |mut plans_and_inputs| {
                // Execute FFTs for all sizes
                for (input, fft) in plans_and_inputs.iter() {
                    let mut c = input.to_vec();
                    fft.execute(&mut c)
                        .expect(format!("Failed to execute {}", input.len()).as_str());
                }
            },
            BatchSize::LargeInput,
        );
    });
}

// simple_bencher.rs
// A minimal, fast Rust micro-bencher you can drop into a project.
// Usage: put this file in `src/bin/` or `examples/`, or include as a module.

#[derive(Debug)]
pub struct Stats {
    pub samples: usize,
    pub total: Duration,
    pub mean: Duration,
    pub median: Duration,
    pub stddev: Duration,
    pub min: Duration,
    pub max: Duration,
}

fn duration_mean(ds: &[Duration]) -> Duration {
    let sum_nanos: u128 = ds.iter().map(|d| d.as_nanos()).sum();
    Duration::from_nanos((sum_nanos / (ds.len() as u128)) as u64)
}

fn duration_stddev(ds: &[Duration], mean: Duration) -> Duration {
    if ds.len() < 2 {
        return Duration::ZERO;
    }
    let mean_n = mean.as_secs_f64();
    let mut var = 0f64;
    for d in ds {
        let x = d.as_secs_f64();
        var += (x - mean_n) * (x - mean_n);
    }
    var /= ds.len() as f64 - 1.0;
    Duration::from_secs_f64(var.sqrt())
}

/// Run a closure repeatedly and collect timing statistics.
///
/// `name` is just for printing. `target_sample_time` controls how long the
/// sampler tries to run to collect samples (default ~200ms). `min_iters` is the
/// minimum iterations per timing sample.
pub fn bench<F, R>(name: &str, mut f: F) -> Stats
where
    F: FnMut() -> R,
{
    // Configuration
    let warmup = Duration::from_millis(25);
    let target_sample_time = Duration::from_millis(100);
    let min_iters = 1usize;
    let max_samples = 4usize; // cap number of samples

    // Warmup phase
    let start = Instant::now();
    while Instant::now() - start < warmup {
        black_box(&f());
    }

    // Collect samples adaptively. Each sample measures `iters` invocations.
    let mut samples: Vec<Duration> = Vec::new();
    let mut iters = min_iters;

    while samples.len() < max_samples {
        // Measure a batch of `iters` runs
        let t0 = Instant::now();
        for _ in 0..iters {
            black_box(&f());
        }
        let elapsed = Instant::now() - t0;

        // Per-iteration time
        let per_iter = elapsed / (iters as u32);
        samples.push(per_iter);

        // If our single sample was too fast, increase iterations so next sample takes longer
        if elapsed < target_sample_time {
            // Scale up to try to hit target_sample_time
            let ratio = (target_sample_time.as_secs_f64() / elapsed.as_secs_f64()).max(1.0);
            let scale = (ratio * 1.5) as usize; // a little headroom
            iters = iters.saturating_mul(scale).max(1);
            // cap iters to avoid overflow
            if iters > 1_000_000_000 {
                iters = 1_000_000;
            }
        }

        // Stop when we have a decent number of samples and median is stable-ish
        if samples.len() >= 5 {
            // optional early stop: if last 3 samples are within 2% of median
            let mut window = samples.clone();
            window.sort();
            let median = window[window.len() / 2].as_secs_f64();
            let recent = &samples[samples.len().saturating_sub(3)..];
            if recent
                .iter()
                .all(|d| ((d.as_secs_f64() / median) - 1.0).abs() < 0.02)
            {
                break;
            }
        }

        // Safety cap on total samples
        if samples.len() >= max_samples {
            break;
        }
    }

    // Compute statistics
    samples.sort();
    let total = samples.iter().fold(Duration::ZERO, |a, &b| a + b);
    let mean = duration_mean(&samples);
    let median = samples[samples.len() / 2];
    let stddev = duration_stddev(&samples, mean);
    let min = *samples.first().unwrap_or(&Duration::ZERO);
    let max = *samples.last().unwrap_or(&Duration::ZERO);

    let stats = Stats {
        samples: samples.len(),
        total,
        mean,
        median,
        stddev,
        min,
        max,
    };

    // Print nice summary
    println!(
        "bench '{}': {} samples — median = {:?}, mean = {:?}, stddev = {:?}, min = {:?}, max = {:?}",
        name, stats.samples, stats.median, stats.mean, stats.stddev, stats.min, stats.max
    );

    stats
}

pub(crate) fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut res = Vec::new();
    if n < 2 {
        return res;
    }

    // factor out 2s
    while (n & 1) == 0 {
        res.push(2);
        n >>= 1;
    }

    // factor out 3s
    while n % 3 == 0 {
        res.push(3);
        n /= 3;
    }

    // trial divide by 6k - 1 and 6k + 1
    let mut p: u64 = 5;
    while (p as u128) * (p as u128) <= n as u128 {
        while n % p == 0 {
            res.push(p);
            n /= p;
        }
        let q = p + 2; // p = 6k-1, q = 6k+1
        while n % q == 0 {
            res.push(q);
            n /= q;
        }
        p += 6;
    }

    // if remaining n > 1 it's prime
    if n > 1 {
        res.push(n);
    }
    res
}

// fn bench_prime( n: usize) -> bool {
//     let mut rng = rand::rng();
//
//     let mut data1: Vec<Complex<f32>> = (0..n).map(|_| Complex::new(rng.random_range(1f32..2f32), 0.)).collect::<Vec<_>>();
//     let mut data2 = data1.clone();
//
//     let raders = Zaft::make_raders(n, FftDirection::Forward).unwrap();
//    let stats_rader = bench(format!("rader: {n}" ).as_str(), || {
//         let mut x = data1.clone();
//         raders.execute(&mut x).unwrap();
//     });
//
//     let bluestein = Zaft::make_bluestein(n, FftDirection::Forward).unwrap();
//     let stats_bluestein = bench(format!("bluestein: {n}").as_str(), || {
//         let mut x = data2.clone();
//         bluestein.execute(&mut x).unwrap();
//     });
//     if stats_bluestein.mean < stats_rader.mean {
//         return true;
//     }
//     false
// }

fn main() {
    let mut data = vec![Complex::new(0.0019528865, 0.); 1296];
    let mut c = Criterion::default().sample_size(10);
    for (i, chunk) in data.iter_mut().enumerate() {
        *chunk = Complex::new(-0.19528865 + i as f32 * 0.1, 0.0019528865 - i as f32 * 0.1);
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
