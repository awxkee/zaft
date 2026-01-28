#![no_main]

use libfuzzer_sys::fuzz_target;
use num_complex::Complex;
use zaft::Zaft;

#[derive(arbitrary::Arbitrary, Debug)]
struct Target {
    size: u16,
    re: f32,
    im: f32,
}

fuzz_target!(|data: Target| {
    if data.size == 0 || data.size > 15100 {
        return;
    }
    let forward = Zaft::make_forward_fft_f32(data.size as usize).unwrap();
    let backward = Zaft::make_inverse_fft_f32(data.size as usize).unwrap();
    let mut chunk = vec![Complex::new(data.re, data.im); data.size as usize];
    forward.execute(&mut chunk).unwrap();
    backward.execute(&mut chunk).unwrap();
    let mut test_target = vec![Complex::new(data.re, data.im); data.size as usize];
    forward
        .execute_out_of_place(&chunk, &mut test_target)
        .unwrap();
    let mut scratch = vec![Complex::default(); forward.destructive_scratch_length()];
    forward
        .execute_destructive_with_scratch(&mut chunk, &mut test_target, &mut scratch)
        .unwrap();
});
