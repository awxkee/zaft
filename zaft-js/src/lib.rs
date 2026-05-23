/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use zaft::{FftExecutor, Zaft, ZaftError};

fn zaft_err(e: ZaftError) -> JsValue {
    JsValue::from_str(&format!("zaft error: {e:?}"))
}

fn norm_scale(norm: &str, n: usize, inverse: bool) -> Result<f64, JsValue> {
    match norm {
        "backward" => Ok(if inverse { 1.0 / n as f64 } else { 1.0 }),
        "forward" => Ok(if inverse { 1.0 } else { 1.0 / n as f64 }),
        "ortho" => Ok(1.0 / (n as f64).sqrt()),
        other => Err(JsValue::from_str(&format!(
            "Unknown norm '{other}'. Expected 'backward', 'forward', or 'ortho'."
        ))),
    }
}

/// View an interleaved f64 slice as Complex<f64> without copying.
fn as_complex_f64(v: &[f64]) -> &[Complex<f64>] {
    assert!(v.len() % 2 == 0, "interleaved array must have even length");
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const Complex<f64>, v.len() / 2) }
}

fn as_complex_f32(v: &[f32]) -> &[Complex<f32>] {
    assert!(v.len() % 2 == 0, "interleaved array must have even length");
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const Complex<f32>, v.len() / 2) }
}

/// Copy Complex<f64> slice into a new interleaved Vec<f64>.
fn to_interleaved_f64(c: &[Complex<f64>]) -> Vec<f64> {
    let mut out = vec![0.0_f64; c.len() * 2];
    for (i, v) in c.iter().enumerate() {
        out[i * 2] = v.re;
        out[i * 2 + 1] = v.im;
    }
    out
}

fn to_interleaved_f32(c: &[Complex<f32>]) -> Vec<f32> {
    let mut out = vec![0.0_f32; c.len() * 2];
    for (i, v) in c.iter().enumerate() {
        out[i * 2] = v.re;
        out[i * 2 + 1] = v.im;
    }
    out
}

// ─── Plan ────────────────────────────────────────────────────────────────────

/// Pre-planned FFT executor for a fixed transform length.
///
/// Creating a Plan is the recommended approach when the same length is
/// transformed repeatedly — planning cost is paid once at construction.
///
/// ```js
/// const plan = new zaft.Plan(1024);
/// const X    = plan.forward(signal);   // Float64Array (interleaved complex)
/// const x    = plan.inverse(X);        // Float64Array (interleaved complex)
/// plan.free();
/// ```
#[wasm_bindgen]
pub struct Plan {
    n: usize,
    fwd: Arc<dyn FftExecutor<f64> + Send + Sync>,
    inv: Arc<dyn FftExecutor<f64> + Send + Sync>,
}

#[wasm_bindgen]
impl Plan {
    /// Create a plan for transforms of length `n`.
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize) -> Result<Plan, JsValue> {
        if n == 0 {
            return Err(JsValue::from_str("Plan: n must be > 0"));
        }
        Ok(Plan {
            n,
            fwd: Zaft::make_forward_fft_f64(n).map_err(zaft_err)?,
            inv: Zaft::make_inverse_fft_f64(n).map_err(zaft_err)?,
        })
    }

    /// Transform length this plan was built for.
    #[wasm_bindgen(getter)]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Forward FFT.
    ///
    /// `input`  — interleaved complex Float64Array of length `2*n`.
    /// Returns  — interleaved complex Float64Array of length `2*n`.
    /// `norm`   — `"backward"` (default), `"ortho"`, or `"forward"`.
    pub fn forward(&self, input: &[f64], norm: Option<String>) -> Result<Vec<f64>, JsValue> {
        let norm = norm.as_deref().unwrap_or("backward");
        let scale = norm_scale(norm, self.n, false)?;
        let src = as_complex_f64(input);
        let mut buf = vec![Complex::new(0.0_f64, 0.0); self.n];
        let copy = src.len().min(self.n);
        buf[..copy].copy_from_slice(&src[..copy]);
        self.fwd.execute(&mut buf).map_err(zaft_err)?;
        if (scale - 1.0).abs() > f64::EPSILON {
            buf.iter_mut().for_each(|v| *v *= scale);
        }
        Ok(to_interleaved_f64(&buf))
    }

    /// Inverse FFT (normalized by 1/n by default).
    ///
    /// `input`  — interleaved complex Float64Array of length `2*n`.z
    /// Returns  — interleaved complex Float64Array of length `2*n`.
    pub fn inverse(&self, input: &[f64], norm: Option<String>) -> Result<Vec<f64>, JsValue> {
        let norm = norm.as_deref().unwrap_or("backward");
        let scale = norm_scale(norm, self.n, true)?;
        let src = as_complex_f64(input);
        let mut buf = vec![Complex::new(0.0_f64, 0.0); self.n];
        let copy = src.len().min(self.n);
        buf[..copy].copy_from_slice(&src[..copy]);
        self.inv.execute(&mut buf).map_err(zaft_err)?;
        buf.iter_mut().for_each(|v| *v *= scale);
        Ok(to_interleaved_f64(&buf))
    }
}

/// Single-precision (f32) pre-planned FFT executor.
///
/// Useful when memory bandwidth matters more than numerical precision
/// (e.g. audio processing, WebGL texture pipelines).
///
/// ```js
/// const plan = new zaft.Plan32(1024);
/// const X    = plan.forward(signal);   // Float32Array (interleaved complex)
/// ```
#[wasm_bindgen]
pub struct Plan32 {
    n: usize,
    fwd: Arc<dyn FftExecutor<f32> + Send + Sync>,
    inv: Arc<dyn FftExecutor<f32> + Send + Sync>,
}

#[wasm_bindgen]
impl Plan32 {
    /// Create a single-precision plan for transforms of length `n`.
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize) -> Result<Plan32, JsValue> {
        if n == 0 {
            return Err(JsValue::from_str("Plan32: n must be > 0"));
        }
        Ok(Plan32 {
            n,
            fwd: Zaft::make_forward_fft_f32(n).map_err(zaft_err)?,
            inv: Zaft::make_inverse_fft_f32(n).map_err(zaft_err)?,
        })
    }

    #[wasm_bindgen(getter)]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Forward FFT (f32).
    ///
    /// `input`  — interleaved complex Float32Array of length `2*n`.
    /// `norm`   — `"backward"` (default), `"ortho"`, or `"forward"`.
    /// Returns  — interleaved complex Float32Array of length `2*n`.
    pub fn forward(&self, input: &[f32], norm: Option<String>) -> Result<Vec<f32>, JsValue> {
        let norm = norm.as_deref().unwrap_or("backward");
        let scale = norm_scale(norm, self.n, false)? as f32;
        let src = as_complex_f32(input);
        let mut buf = vec![Complex::new(0.0_f32, 0.0); self.n];
        let copy = src.len().min(self.n);
        buf[..copy].copy_from_slice(&src[..copy]);
        self.fwd.execute(&mut buf).map_err(zaft_err)?;
        if (scale - 1.0f32).abs() > f32::EPSILON {
            buf.iter_mut().for_each(|v| *v *= scale);
        }
        Ok(to_interleaved_f32(&buf))
    }

    /// Inverse FFT (f32, normalised by 1/n by default).
    /// `norm`   — `"backward"` (default), `"ortho"`, or `"forward"`.
    /// Returns  — interleaved complex Float32Array of length `2*n`.
    pub fn inverse(&self, input: &[f32], norm: Option<String>) -> Result<Vec<f32>, JsValue> {
        let norm = norm.as_deref().unwrap_or("backward");
        let scale = norm_scale(norm, self.n, true)? as f32;
        let src = as_complex_f32(input);
        let mut buf = vec![Complex::new(0.0_f32, 0.0); self.n];
        let copy = src.len().min(self.n);
        buf[..copy].copy_from_slice(&src[..copy]);
        self.inv.execute(&mut buf).map_err(zaft_err)?;
        buf.iter_mut().for_each(|v| *v *= scale);
        Ok(to_interleaved_f32(&buf))
    }
}

/// Forward FFT (f64, no pre-planning).
///
/// `input`  — interleaved complex Float64Array of length `2*n`.
/// `n`      — transform length; if omitted, `input.length / 2` is used.
/// `norm`   — `"backward"` (default), `"ortho"`, or `"forward"`.
/// Returns  — new interleaved complex Float64Array of length `2*n`.
#[wasm_bindgen]
pub fn fft(input: &[f64], n: Option<usize>, norm: Option<String>) -> Result<Vec<f64>, JsValue> {
    let size = n.unwrap_or(input.len() / 2);
    if size == 0 {
        return Err(JsValue::from_str("fft: n must be > 0"));
    }
    let norm = norm.as_deref().unwrap_or("backward");
    let scale = norm_scale(norm, size, false)?;
    let src = as_complex_f64(input);
    let exec = Zaft::make_forward_fft_f64(size).map_err(zaft_err)?;
    let mut buf = vec![Complex::new(0.0_f64, 0.0); size];
    let copy = src.len().min(size);
    buf[..copy].copy_from_slice(&src[..copy]);
    exec.execute(&mut buf).map_err(zaft_err)?;
    if (scale - 1.0).abs() > f64::EPSILON {
        buf.iter_mut().for_each(|v| *v *= scale);
    }
    Ok(to_interleaved_f64(&buf))
}

/// Inverse FFT (f64).
///
/// `input`  — interleaved complex Float64Array of length `2*n`.
/// Returns  — new interleaved complex Float64Array of length `2*n`.
#[wasm_bindgen]
pub fn ifft(input: &[f64], n: Option<usize>, norm: Option<String>) -> Result<Vec<f64>, JsValue> {
    let size = n.unwrap_or(input.len() / 2);
    if size == 0 {
        return Err(JsValue::from_str("ifft: n must be > 0"));
    }
    let norm = norm.as_deref().unwrap_or("backward");
    let scale = norm_scale(norm, size, true)?;
    let src = as_complex_f64(input);
    let exec = Zaft::make_inverse_fft_f64(size).map_err(zaft_err)?;
    let mut buf = vec![Complex::new(0.0_f64, 0.0); size];
    let copy = src.len().min(size);
    buf[..copy].copy_from_slice(&src[..copy]);
    exec.execute(&mut buf).map_err(zaft_err)?;
    buf.iter_mut().for_each(|v| *v *= scale);
    Ok(to_interleaved_f64(&buf))
}

/// Real-to-complex FFT (f64).
///
/// `input`  — real Float64Array of length `n`.
/// Returns  — interleaved complex Float64Array of length `2*(n/2 + 1)`.
#[wasm_bindgen]
pub fn rfft(input: &[f64], n: Option<usize>, norm: Option<String>) -> Result<Vec<f64>, JsValue> {
    let size = n.unwrap_or(input.len());
    if size == 0 {
        return Err(JsValue::from_str("rfft: n must be > 0"));
    }
    let norm = norm.as_deref().unwrap_or("backward");
    let scale = norm_scale(norm, size, false)?;
    let exec = Zaft::make_r2c_fft_f64(size).map_err(zaft_err)?;
    let mut real_in = vec![0.0_f64; size];
    let copy = input.len().min(size);
    real_in[..copy].copy_from_slice(&input[..copy]);
    let out_len = size / 2 + 1;
    let mut out = vec![Complex::new(0.0_f64, 0.0); out_len];
    exec.execute(&real_in, &mut out).map_err(zaft_err)?;
    if (scale - 1.0).abs() > f64::EPSILON {
        out.iter_mut().for_each(|v| *v *= scale);
    }
    Ok(to_interleaved_f64(&out))
}

/// Complex-to-real inverse FFT (f64).
///
/// `input`  — interleaved complex Float64Array of length `2*(n/2 + 1)`.
/// `n`      — length of the real output (required to disambiguate even/odd).
/// Returns  — real Float64Array of length `n`.
#[wasm_bindgen]
pub fn irfft(input: &[f64], n: Option<usize>, norm: Option<String>) -> Result<Vec<f64>, JsValue> {
    let in_complex_len = input.len() / 2;
    let size = n.unwrap_or(2 * (in_complex_len.saturating_sub(1)));
    if size == 0 {
        return Err(JsValue::from_str("irfft: n must be > 0"));
    }
    let norm = norm.as_deref().unwrap_or("backward");
    let scale = norm_scale(norm, size, true)?;
    let exec = Zaft::make_c2r_fft_f64(size).map_err(zaft_err)?;
    let src = as_complex_f64(input);
    let mut real_out = vec![0.0_f64; size];
    exec.execute(src, &mut real_out).map_err(zaft_err)?;
    if (scale - 1.0).abs() > f64::EPSILON {
        real_out.iter_mut().for_each(|v| *v *= scale);
    }
    Ok(real_out)
}

/// Forward FFT, single-precision (f32).
///
/// `input`  — interleaved complex Float32Array of length `2*n`.
/// Returns  — new interleaved complex Float32Array of length `2*n`.
#[wasm_bindgen]
pub fn fft32(input: &[f32], n: Option<usize>, norm: Option<String>) -> Result<Vec<f32>, JsValue> {
    let size = n.unwrap_or(input.len() / 2);
    if size == 0 {
        return Err(JsValue::from_str("fft32: n must be > 0"));
    }
    let norm = norm.as_deref().unwrap_or("backward");
    let scale = norm_scale(norm, size, false)? as f32;
    let src = as_complex_f32(input);
    let exec = Zaft::make_forward_fft_f32(size).map_err(zaft_err)?;
    let mut buf = vec![Complex::new(0.0_f32, 0.0); size];
    let copy = src.len().min(size);
    buf[..copy].copy_from_slice(&src[..copy]);
    exec.execute(&mut buf).map_err(zaft_err)?;
    if (scale - 1.0f32).abs() > f32::EPSILON {
        buf.iter_mut().for_each(|v| *v *= scale);
    }
    Ok(to_interleaved_f32(&buf))
}

/// Inverse FFT, single-precision (f32).
#[wasm_bindgen]
pub fn ifft32(input: &[f32], n: Option<usize>, norm: Option<String>) -> Result<Vec<f32>, JsValue> {
    let size = n.unwrap_or(input.len() / 2);
    if size == 0 {
        return Err(JsValue::from_str("ifft32: n must be > 0"));
    }
    let norm = norm.as_deref().unwrap_or("backward");
    let scale = norm_scale(norm, size, true)? as f32;
    let src = as_complex_f32(input);
    let exec = Zaft::make_inverse_fft_f32(size).map_err(zaft_err)?;
    let mut buf = vec![Complex::new(0.0_f32, 0.0); size];
    let copy = src.len().min(size);
    buf[..copy].copy_from_slice(&src[..copy]);
    exec.execute(&mut buf).map_err(zaft_err)?;
    buf.iter_mut().for_each(|v| *v *= scale);
    Ok(to_interleaved_f32(&buf))
}

/// Real-to-complex FFT, single-precision (f32).
///
/// `input`  — real Float32Array of length `n`.
/// Returns  — interleaved complex Float32Array of length `2*(n/2 + 1)`.
#[wasm_bindgen]
pub fn rfft32(input: &[f32], n: Option<usize>, norm: Option<String>) -> Result<Vec<f32>, JsValue> {
    let size = n.unwrap_or(input.len());
    if size == 0 {
        return Err(JsValue::from_str("rfft32: n must be > 0"));
    }
    let norm = norm.as_deref().unwrap_or("backward");
    let scale = norm_scale(norm, size, false)? as f32;
    let exec = Zaft::make_r2c_fft_f32(size).map_err(zaft_err)?;
    let mut real_in = vec![0.0_f32; size];
    let copy = input.len().min(size);
    real_in[..copy].copy_from_slice(&input[..copy]);
    let out_len = size / 2 + 1;
    let mut out = vec![Complex::new(0.0_f32, 0.0); out_len];
    exec.execute(&real_in, &mut out).map_err(zaft_err)?;
    if (scale - 1.0f32).abs() > f32::EPSILON {
        out.iter_mut().for_each(|v| *v *= scale);
    }
    Ok(to_interleaved_f32(&out))
}

/// Complex-to-real inverse FFT, single-precision (f32).
#[wasm_bindgen]
pub fn irfft32(input: &[f32], n: Option<usize>, norm: Option<String>) -> Result<Vec<f32>, JsValue> {
    let in_complex_len = input.len() / 2;
    let size = n.unwrap_or(2 * (in_complex_len.saturating_sub(1)));
    if size == 0 {
        return Err(JsValue::from_str("irfft32: n must be > 0"));
    }
    let norm = norm.as_deref().unwrap_or("backward");
    let scale = norm_scale(norm, size, true)? as f32;
    let exec = Zaft::make_c2r_fft_f32(size).map_err(zaft_err)?;
    let src = as_complex_f32(input);
    let mut real_out = vec![0.0_f32; size];
    exec.execute(src, &mut real_out).map_err(zaft_err)?;
    if (scale - 1.0f32).abs() > f32::EPSILON {
        real_out.iter_mut().for_each(|v| *v *= scale);
    }
    Ok(real_out)
}

// ─── frequency helpers ────────────────────────────────────────────────────────

/// DFT sample frequencies — equivalent to `numpy.fft.fftfreq`.
///
/// Returns a Float64Array of length `n`.
#[wasm_bindgen]
pub fn fftfreq(n: usize, d: Option<f64>) -> Vec<f64> {
    let d = d.unwrap_or(1.0);
    let half = (n + 1) / 2;
    let mut out = vec![0.0_f64; n];
    for i in 0..half {
        out[i] = i as f64 / (n as f64 * d);
    }
    for i in half..n {
        out[i] = (i as f64 - n as f64) / (n as f64 * d);
    }
    out
}

/// Sample frequencies for `rfft` — equivalent to `numpy.fft.rfftfreq`.
///
/// Returns a Float64Array of length `n/2 + 1`.
#[wasm_bindgen]
pub fn rfftfreq(n: usize, d: Option<f64>) -> Vec<f64> {
    let d = d.unwrap_or(1.0);
    (0..n / 2 + 1).map(|i| i as f64 / (n as f64 * d)).collect()
}

/// Smallest integer >= `target` that Zaft transforms efficiently (13-smooth).
#[wasm_bindgen]
pub fn next_fast_len(target: usize) -> usize {
    if target <= 1 {
        return target;
    }
    let smooth = |mut n: usize| -> bool {
        for p in [2usize, 3, 5, 7, 11, 13] {
            while n % p == 0 {
                n /= p;
            }
        }
        n == 1
    };
    let mut n = target;
    loop {
        if smooth(n) {
            return n;
        }
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_node_experimental);

    fn make_signal(n: usize) -> Vec<f64> {
        // interleaved complex: im = 0
        let mut v = vec![0.0_f64; n * 2];
        for i in 0..n {
            v[i * 2] = (i as f64 * 0.1).sin();
        }
        v
    }

    #[wasm_bindgen_test]
    fn test_fft_ifft_roundtrip() {
        let n = 256;
        let signal = make_signal(n);
        let spectrum = fft(&signal, None, None).unwrap();
        let recovered = ifft(&spectrum, None, None).unwrap();
        for i in 0..n {
            let orig_re = signal[i * 2];
            let rec_re = recovered[i * 2];
            let rec_im = recovered[i * 2 + 1];
            assert!((orig_re - rec_re).abs() < 1e-9, "re mismatch at {i}");
            assert!(rec_im.abs() < 1e-9, "im should be ~0 at {i}");
        }
    }

    #[wasm_bindgen_test]
    fn test_rfft_irfft_roundtrip() {
        let n = 256;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let spectrum = rfft(&signal, None, None).unwrap();
        let recovered = irfft(&spectrum, Some(n), None).unwrap();
        for i in 0..n {
            assert!((signal[i] - recovered[i]).abs() < 1e-9, "mismatch at {i}");
        }
    }

    #[wasm_bindgen_test]
    fn test_plan_roundtrip() {
        let n = 512;
        let signal = make_signal(n);
        let plan = Plan::new(n).unwrap();
        let spectrum = plan.forward(&signal, None).unwrap();
        let recovered = plan.inverse(&spectrum, None).unwrap();
        for i in 0..n {
            let orig = signal[i * 2];
            let rec = recovered[i * 2];
            assert!((orig - rec).abs() < 1e-9, "mismatch at {i}");
        }
    }

    #[wasm_bindgen_test]
    fn test_fft32_roundtrip() {
        let n = 128;
        let mut signal = vec![0.0_f32; n * 2];
        for i in 0..n {
            signal[i * 2] = (i as f32 * 0.1).sin();
        }
        let spectrum = fft32(&signal, None, None).unwrap();
        let recovered = ifft32(&spectrum, None, None).unwrap();
        for i in 0..n {
            assert!((signal[i * 2] - recovered[i * 2]).abs() < 1e-4);
        }
    }

    #[wasm_bindgen_test]
    fn test_next_fast_len() {
        assert_eq!(next_fast_len(100), 100);
        assert_eq!(next_fast_len(101), 104);
        assert_eq!(next_fast_len(1), 1);
    }

    #[wasm_bindgen_test]
    fn test_fftfreq() {
        let f = fftfreq(4, None);
        // numpy: [0, 0.25, -0.5, -0.25]
        assert!((f[0] - 0.0).abs() < 1e-12);
        assert!((f[1] - 0.25).abs() < 1e-12);
        assert!((f[2] - (-0.5)).abs() < 1e-12);
        assert!((f[3] - (-0.25)).abs() < 1e-12);
    }
}
