#![no_main]

use libfuzzer_sys::fuzz_target;
use num_complex::Complex;
use zaft::Zaft;

#[derive(arbitrary::Arbitrary, Debug)]
struct Target {
    size: u16,
    re: f32,
}

fuzz_target!(|data: Target| {
    if data.size == 0 || data.size > 10000 {
        return;
    }
    if !data.re.is_finite() || data.re > 10000. || data.re < 1e-10 {
        return;
    }
    let executor_forward = Zaft::make_r2c_fft_f32(data.size as usize).unwrap();
    let executor_backwards = Zaft::make_c2r_fft_f32(data.size as usize).unwrap();
    let mut chunk = vec![data.re; data.size as usize];
    let mut complex = vec![Complex::new(0.0, 0.0); data.size as usize / 2 + 1];
    for (i, chunk) in chunk.iter_mut().enumerate() {
        *chunk = data.re + i as f32 * 0.1;
    }
    executor_forward.execute(&chunk, &mut complex).unwrap();
    executor_backwards.execute(&complex, &mut chunk).unwrap();
});
