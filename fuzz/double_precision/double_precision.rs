#![no_main]

use libfuzzer_sys::fuzz_target;
use num_complex::Complex;
use zaft::Zaft;

#[derive(arbitrary::Arbitrary, Debug)]
struct Target {
    forward: bool,
    size: u16,
    re: f64,
    im: f64,
}

fuzz_target!(|data: Target| {
    if data.size == 0 || data.size > 15100 {
        return;
    }
    let executor = if data.forward {
        Zaft::make_forward_fft_f64(data.size as usize).unwrap()
    } else {
        Zaft::make_inverse_fft_f64(data.size as usize).unwrap()
    };
    let mut chunk = vec![Complex::new(data.re, data.im); data.size as usize];
    executor.execute(&mut chunk).unwrap();
});
