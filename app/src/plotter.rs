/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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
use num_traits::Zero;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::element::PathElement;
use plotters::prelude::{
    BLACK, BLUE, Color, IntoFont, IntoLogRange, LineSeries, MAGENTA, RED, Rectangle,
    SeriesLabelPosition, Text, WHITE,
};
use plotters::style::GREEN;
use pxfm::f_log2;
use rand::RngExt;
use realfft::RealFftPlanner;
use rustfft::FftPlanner;
use std::time::Instant;
use zaft::Zaft;

const PREHEATING: usize = 20;

fn bench_rustfft(size: usize, iterations: usize) -> f64 {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(size);

    let mut rng = rand::rng();
    let mut buffer: Vec<Complex<f32>> = (0..size)
        .map(|_| Complex::new(rng.random_range(1f32..2f32), 0.0))
        .collect();

    let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];

    // preheat
    for _ in 0..PREHEATING {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        fft.process_with_scratch(&mut buffer, &mut scratch);
    }
    start.elapsed().as_secs_f64()
}

fn bench_realfft(size: usize, iterations: usize) -> f64 {
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(size);

    let mut rng = rand::rng();
    let mut input: Vec<f32> = (0..size)
        .map(|_| rng.random_range(1f32..2f32))
        .collect::<Vec<_>>();

    let mut output = r2c.make_output_vec();

    let mut scratch = vec![Complex::zero(); r2c.get_scratch_len()];

    // preheat
    for _ in 0..PREHEATING {
        r2c.process_with_scratch(&mut input, &mut output, &mut scratch)
            .unwrap();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        r2c.process_with_scratch(&mut input, &mut output, &mut scratch)
            .unwrap();
    }
    start.elapsed().as_secs_f64()
}

fn bench_zaft_real(size: usize, iterations: usize) -> f64 {
    let r2c = Zaft::make_r2c_fft_f32(size).unwrap();

    let mut rng = rand::rng();
    let mut input: Vec<f32> = (0..size)
        .map(|_| rng.random_range(1f32..2f32))
        .collect::<Vec<_>>();
    let mut output = vec![Complex::zero(); size / 2 + 1];

    let mut scratch = vec![Complex::zero(); r2c.complex_scratch_length()];

    // preheat
    for _ in 0..PREHEATING {
        r2c.execute_with_scratch(&mut input, &mut output, &mut scratch)
            .unwrap();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        r2c.execute_with_scratch(&mut input, &mut output, &mut scratch)
            .unwrap();
    }
    start.elapsed().as_secs_f64()
}

fn bench_zaft_c2c(size: usize, iterations: usize) -> f64 {
    let c2c = Zaft::make_forward_fft_f32(size).unwrap();

    let mut rng = rand::rng();
    let mut buffer: Vec<Complex<f32>> = (0..size)
        .map(|_| Complex::new(rng.random_range(1f32..2f32), 0.0))
        .collect();

    let mut scratch = vec![Complex::zero(); c2c.scratch_length()];

    // preheat
    for _ in 0..PREHEATING {
        c2c.execute_with_scratch(&mut buffer, &mut scratch).unwrap();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        c2c.execute_with_scratch(&mut buffer, &mut scratch).unwrap();
    }
    start.elapsed().as_secs_f64()
}

fn plot_absolute_log(
    plot_name: &str,
    rustfft: &[(f64, f64)],
    real_fft: &[(f64, f64)],
    zaft_r2c: &[(f64, f64)],
    zaft_c2c: &[(f64, f64)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(plot_name, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("FFT Performance (log-log)", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..24f64, (1e-6f64..1.0f64).log_scale())?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(rustfft.to_vec(), &RED))?
        .label("rustfft (C2C)")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(real_fft.to_vec(), &BLUE))?
        .label("real_fft (R2C)")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(zaft_r2c.to_vec(), &GREEN))?
        .label("zaft (R2C)")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));

    chart
        .draw_series(LineSeries::new(zaft_c2c.to_vec(), &MAGENTA))?
        .label("zaft (C2C)")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &MAGENTA));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn plot_absolute(
    plot_name: &str,
    rustfft: &[(f64, f64)],
    real_fft: &[(f64, f64)],
    zaft_r2c: &[(f64, f64)],
    zaft_c2c: &[(f64, f64)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(plot_name, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let all = rustfft
        .iter()
        .chain(real_fft)
        .chain(zaft_r2c)
        .chain(zaft_c2c);
    let min_y = all.clone().map(|(_, y)| *y).fold(f64::MAX, f64::min);
    let max_y = all.map(|(_, y)| *y).fold(0.0_f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("FFT Performance (abs-log)", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(
            rustfft.first().unwrap().0..rustfft.last().unwrap().0,
            (min_y * 0.9..max_y * 1.1).log_scale(),
        )?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(rustfft.to_vec(), &RED))?
        .label("rustfft (C2C)")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(real_fft.to_vec(), &BLUE))?
        .label("real_fft (R2C)")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(zaft_r2c.to_vec(), &GREEN))?
        .label("zaft (R2C)")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));

    chart
        .draw_series(LineSeries::new(zaft_c2c.to_vec(), &MAGENTA))?
        .label("zaft (C2C)")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &MAGENTA));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn plot_absolute_pair(
    plot_name: &str,
    rust_fft_label: &str,
    rustfft: &[(f64, f64)],
    zaft_label: &str,
    zaft: &[(f64, f64)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(plot_name, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_y = rustfft
        .iter()
        .chain(zaft.iter())
        .map(|(_, y)| *y)
        .fold(f64::MAX, f64::min);
    let max_y = rustfft
        .iter()
        .chain(zaft.iter())
        .map(|(_, y)| *y)
        .fold(0.0_f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(plot_name, ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(
            rustfft.first().unwrap().0..rustfft.last().unwrap().0,
            (min_y * 0.9..max_y * 1.1).log_scale(),
        )?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(rustfft.to_vec(), &RED))?
        .label(rust_fft_label)
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(zaft.to_vec(), &BLUE))?
        .label(zaft_label)
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn plot_ratio(
    title: &str,
    plot_name: &str,
    real_fft: &[(f64, f64)],
    zaft_r2c: &[(f64, f64)],
) -> Result<(), Box<dyn std::error::Error>> {
    let ratio: Vec<f64> = real_fft
        .iter()
        .zip(zaft_r2c.iter())
        .map(|((_, t1), (_, t2))| t1 / t2)
        .collect();

    let mut sizes: Vec<f64> = real_fft.iter().map(|(s, _)| *s).collect();
    sizes.resize(ratio.len(), 0.0);

    let root = BitMapBackend::new(plot_name, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_ratio = ratio.iter().cloned().fold(0.0_f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..ratio.len() as f64, 0f64..(max_ratio * 1.1))?;

    chart
        .configure_mesh()
        .x_labels(ratio.len())
        .x_label_formatter(&|_| String::new())
        .y_desc("Time Ratio")
        .x_desc("FFT Size")
        .draw()?;

    for (i, size) in sizes.iter().enumerate() {
        let x_center = i as f64 + 0.5 - 0.12;

        chart.draw_series(std::iter::once(Text::new(
            format!("{}", size),
            (x_center, 0.0),
            ("sans-serif", 20).into_font(),
        )))?;
    }

    let bar_width = 0.6;
    chart.draw_series(ratio.iter().enumerate().map(|(i, value)| {
        let x0 = i as f64 + (1.0 - bar_width) / 2.0;
        let x1 = x0 + bar_width;

        Rectangle::new([(x0, 0.0), (x1, *value)], GREEN.filled())
    }))?;

    Ok(())
}

pub fn benchmark_compare() {
    benchmark_compare_powers_2();
    benchmark_common_size_not2();
    for i in 0..8 {
        benchmark_compare_any(1 + i * 128, (i + 1) * 128);
    }
}

pub fn benchmark_compare_any(start: usize, end: usize) {
    let mut sizes = vec![];
    for i in start..=end {
        sizes.push(i);
    }
    let iterations = 200;

    let mut rustfft = vec![];
    let mut realfft = vec![];
    let mut zaft_r2c = vec![];
    let mut zaft_c2c = vec![];

    for &size in &sizes {
        let r1 = bench_rustfft(size, iterations);
        let r2 = bench_realfft(size, iterations);
        let r3 = bench_zaft_real(size, iterations);
        let r4 = bench_zaft_c2c(size, iterations);

        rustfft.push((size as f64, r1));
        realfft.push((size as f64, r2));
        zaft_r2c.push((size as f64, r3));
        zaft_c2c.push((size as f64, r4));
        println!("done size, any {}", size);
    }

    let arch = std::env::consts::ARCH;

    plot_absolute(
        &format!("fft_absolute_{start}to{end}_{arch}.png"),
        &rustfft,
        &realfft,
        &zaft_r2c,
        &zaft_c2c,
    )
    .unwrap();

    plot_absolute_pair(
        &format!("fft_absolute_c2c_{start}to{end}_{arch}.png"),
        "RustFFT C2C",
        &rustfft,
        "Zaft C2C",
        &zaft_c2c,
    )
    .unwrap();
    plot_absolute_pair(
        &format!("fft_absolute_r2c_{start}to{end}_{arch}.png"),
        "RealFft R2C",
        &realfft,
        "Zaft R2C",
        &zaft_c2c,
    )
    .unwrap();
    plot_absolute_pair(
        &format!("fft_zaft_c2c_vs_r2c_evens_{start}to{end}_{arch}.png"),
        "Zaft C2C",
        &zaft_c2c
            .iter()
            .filter(|x| (x.0 as usize).is_multiple_of(2))
            .map(|&x| x)
            .collect::<Vec<_>>(),
        "Zaft R2C",
        &zaft_r2c
            .iter()
            .filter(|x| !(x.0 as usize).is_multiple_of(2))
            .map(|&x| x)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    plot_absolute_pair(
        &format!("fft_zaft_c2c_vs_r2c_odd_{start}to{end}_{arch}.png"),
        "Zaft C2C",
        &zaft_c2c
            .into_iter()
            .filter(|x| !(x.0 as usize).is_multiple_of(2))
            .collect::<Vec<_>>(),
        "Zaft R2C",
        &zaft_r2c
            .into_iter()
            .filter(|x| !(x.0 as usize).is_multiple_of(2))
            .collect::<Vec<_>>(),
    )
    .unwrap();
}

pub fn benchmark_compare_powers_2() {
    let mut sizes = vec![];
    for i in 1..20 {
        sizes.push(1 << i);
    }
    let iterations = 200;

    let mut rustfft = vec![];
    let mut realfft = vec![];
    let mut zaft_r2c = vec![];
    let mut zaft_c2c = vec![];

    for &size in &sizes {
        let r1 = bench_rustfft(size, iterations);
        let r2 = bench_realfft(size, iterations);
        let r3 = bench_zaft_real(size, iterations);
        let r4 = bench_zaft_c2c(size, iterations);

        let log_size = f_log2(size as f64);

        rustfft.push((log_size, r1));
        realfft.push((log_size, r2));
        zaft_r2c.push((log_size, r3));
        zaft_c2c.push((log_size, r4));
        println!("done size {}", size);
    }

    let arch = std::env::consts::ARCH;

    plot_absolute_log(
        &format!("fft_absolute_power2_{arch}.png"),
        &rustfft,
        &realfft,
        &zaft_r2c,
        &zaft_c2c,
    )
    .unwrap();
    plot_ratio(
        "RealFFT / ZAFT R2C Time Ratio log2",
        &format!("fft_r2c_power2_ratio_{arch}.png"),
        &realfft,
        &zaft_r2c,
    )
    .unwrap();
    plot_ratio(
        "RustFFT / ZAFT C2C Time Ratio log2",
        &format!("fft_c2c_power2_ratio_{arch}.png"),
        &rustfft,
        &zaft_c2c,
    )
    .unwrap();
}

pub fn benchmark_common_size_not2() {
    let sizes = [180, 192, 300, 600, 900, 1200, 1536, 1920, 192000];
    let iterations = 200;

    let mut rustfft = vec![];
    let mut realfft = vec![];
    let mut zaft_r2c = vec![];
    let mut zaft_c2c = vec![];

    for &size in &sizes {
        let r1 = bench_rustfft(size, iterations);
        let r2 = bench_realfft(size, iterations);
        let r3 = bench_zaft_real(size, iterations);
        let r4 = bench_zaft_c2c(size, iterations);

        rustfft.push((size as f64, r1));
        realfft.push((size as f64, r2));
        zaft_r2c.push((size as f64, r3));
        zaft_c2c.push((size as f64, r4));
        println!("done size {}", size);
    }

    let arch = std::env::consts::ARCH;

    plot_ratio(
        "RealFFT / ZAFT R2C Time Ratio",
        &format!("fft_r2c_common_not2_ratio_{arch}.png"),
        &realfft,
        &zaft_r2c,
    )
    .unwrap();
    plot_ratio(
        "RustFFT / ZAFT C2C Time Ratio",
        &format!("fft_c2c_common_not2_ratio_{arch}.png"),
        &rustfft,
        &zaft_c2c,
    )
    .unwrap();
}
