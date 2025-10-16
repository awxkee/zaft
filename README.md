# Zaft FFT Example (Rust)

This example demonstrates how to perform real-to-complex (R2C) and complex-to-real (C2R) FFTs, as well as standard complex-to-complex forward and inverse FFTs using Zaft.

## Real-to-Complex (R2C) and Complex-to-Real (C2R)

```rust
use zaft::Zaft;
use num_complex::Complex;

fn main() -> Result<(), zaft::ZaftError> {
    // Example real input data
    let mut real_data: Vec<f32> = vec![0.0; 1024];

    // Create R2C and C2R FFT executors
    let forward_r2c = Zaft::make_r2c_fft_f32(real_data.len())?;
    let inverse_r2c = Zaft::make_c2r_fft_f32(real_data.len())?;

    // Prepare buffer for complex output
    let mut complex_data = vec![Complex::<f32>::default(); real_data.len() / 2 + 1];

    // Perform forward R2C FFT
    forward_r2c.execute(&real_data, &mut complex_data)?;

    // Perform inverse C2R FFT
    inverse_r2c.execute(&complex_data, &mut real_data)?;

    Ok(())
}
```
Note: After the round-trip, real_data will approximately equal its original values (within floating-point precision).

Complex-to-Complex FFT

```rust
use zaft::Zaft;
use num_complex::Complex;

fn main() -> Result<(), zaft::ZaftError> {
    // Example complex input
    let mut data: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); 1024];

    // Create forward and inverse complex FFT executors
    let forward = Zaft::make_forward_fft_f32(data.len())?;
    let inverse = Zaft::make_inverse_fft_f32(data.len())?;

    // Perform forward FFT
    forward.execute(&mut data)?;

    // Perform inverse FFT
    inverse.execute(&mut data)?;

    Ok(())
}
```

Notes

R2C FFTs only store the non-redundant half of the spectrum: the length of the complex buffer is (real_len / 2 + 1).

C2R FFTs require the same shape of complex input and reconstruct the full real output.

Complex-to-complex FFTs operate in-place and maintain the full length of the data.

----

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.