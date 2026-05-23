// src/lib.rs  –  PyO3 0.28 bindings for Zaft
use num_complex::Complex;
use numpy::PyUntypedArrayMethods;
use numpy::{
    Complex32,
    Complex64,
    IntoPyArray,
    PyArray1,
    PyArray2,
    PyArrayMethods, // needed for .readonly() on Bound arrays
    PyReadonlyArray1,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use zaft::{FftDirection, FftExecutor, Zaft, ZaftError};

fn zaft_err(e: ZaftError) -> PyErr {
    PyRuntimeError::new_err(format!("zaft error: {e:?}"))
}

fn parse_norm(norm: Option<&str>, n: usize, inverse: bool) -> PyResult<f64> {
    match norm {
        None | Some("backward") => Ok(if inverse { 1.0 / n as f64 } else { 1.0 }),
        Some("forward") => Ok(if inverse { 1.0 } else { 1.0 / n as f64 }),
        Some("ortho") => Ok(1.0 / (n as f64).sqrt()),
        Some(other) => Err(PyValueError::new_err(format!(
            "Unknown norm '{other}'. Expected 'backward', 'forward', or 'ortho'."
        ))),
    }
}

fn c2c_f64(
    input: &[Complex<f64>],
    n: usize,
    direction: FftDirection,
) -> PyResult<Vec<Complex<f64>>> {
    let exec = match direction {
        FftDirection::Forward => Zaft::make_forward_fft_f64(n).map_err(zaft_err)?,
        FftDirection::Inverse => Zaft::make_inverse_fft_f64(n).map_err(zaft_err)?,
    };
    let mut buf = vec![Complex::new(0.0_f64, 0.0); n];
    buf[..input.len().min(n)].copy_from_slice(&input[..input.len().min(n)]);
    exec.execute(&mut buf).map_err(zaft_err)?;
    Ok(buf)
}

fn c2c_f32(
    input: &[Complex<f32>],
    n: usize,
    direction: FftDirection,
) -> PyResult<Vec<Complex<f32>>> {
    let exec = match direction {
        FftDirection::Forward => Zaft::make_forward_fft_f32(n).map_err(zaft_err)?,
        FftDirection::Inverse => Zaft::make_inverse_fft_f32(n).map_err(zaft_err)?,
    };
    let mut buf = vec![Complex::new(0.0_f32, 0.0); n];
    buf[..input.len().min(n)].copy_from_slice(&input[..input.len().min(n)]);
    exec.execute(&mut buf).map_err(zaft_err)?;
    Ok(buf)
}

/// Executors for one precision — only the requested dtype is built.
enum PlanInner {
    F64 {
        fwd: Arc<dyn FftExecutor<f64> + Send + Sync>,
        inv: Arc<dyn FftExecutor<f64> + Send + Sync>,
    },
    F32 {
        fwd: Arc<dyn FftExecutor<f32> + Send + Sync>,
        inv: Arc<dyn FftExecutor<f32> + Send + Sync>,
    },
}

/// Pre-planned FFT executor — planning cost is paid once at construction.
///
/// Parameters
/// ----------
/// n : int
///     Transform length.
/// dtype : str, optional
///     ``'complex128'`` (default) or ``'complex64'``.
///     Determines which precision executor is built and which input arrays
///     ``execute_forward`` / ``execute_inverse`` will accept.
/// workers : int, optional
///     Hint for parallel 2-D transforms (ignored for 1-D).
#[pyclass(name = "Plan")]
struct Plan {
    n: usize,
    workers: usize,
    dtype: String,
    inner: PlanInner,
}

#[pymethods]
impl Plan {
    #[new]
    #[pyo3(signature = (n, dtype = "complex128", workers = 1))]
    fn new(n: usize, dtype: &str, workers: usize) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err("Transform length n must be > 0"));
        }
        let inner = match dtype {
            "complex128" | "float64" => PlanInner::F64 {
                fwd: Zaft::make_forward_fft_f64(n).map_err(zaft_err)?,
                inv: Zaft::make_inverse_fft_f64(n).map_err(zaft_err)?,
            },
            "complex64" | "float32" => PlanInner::F32 {
                fwd: Zaft::make_forward_fft_f32(n).map_err(zaft_err)?,
                inv: Zaft::make_inverse_fft_f32(n).map_err(zaft_err)?,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown dtype '{other}'. Expected 'complex128' or 'complex64'."
                )));
            }
        };
        Ok(Plan {
            n,
            workers: workers.max(1),
            dtype: dtype.to_string(),
            inner,
        })
    }

    /// Execute the forward FFT.
    ///
    /// Accepts ``complex128`` or ``complex64`` depending on the ``dtype``
    /// the plan was constructed with.  Returns the same dtype.
    fn execute_forward<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            PlanInner::F64 { fwd, .. } => {
                let arr = input.cast::<PyArray1<Complex64>>().map_err(|_| {
                    PyValueError::new_err("Plan(dtype='complex128') requires a complex128 array")
                })?;
                let ro = arr.readonly();
                let src = ro.as_slice()?;
                let mut buf = vec![Complex::new(0.0_f64, 0.0); self.n];
                buf[..src.len().min(self.n)].copy_from_slice(&src[..src.len().min(self.n)]);
                fwd.execute(&mut buf).map_err(zaft_err)?;
                Ok(buf.into_pyarray(py).into_any().unbind())
            }
            PlanInner::F32 { fwd, .. } => {
                let arr = input.cast::<PyArray1<Complex32>>().map_err(|_| {
                    PyValueError::new_err("Plan(dtype='complex64') requires a complex64 array")
                })?;
                let ro = arr.readonly();
                let src = ro.as_slice()?;
                let mut buf = vec![Complex::new(0.0_f32, 0.0); self.n];
                buf[..src.len().min(self.n)].copy_from_slice(&src[..src.len().min(self.n)]);
                fwd.execute(&mut buf).map_err(zaft_err)?;
                Ok(buf.into_pyarray(py).into_any().unbind())
            }
        }
    }

    /// Execute the inverse FFT (unnormalised — divide by ``n`` yourself if needed).
    fn execute_inverse<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            PlanInner::F64 { inv, .. } => {
                let arr = input.cast::<PyArray1<Complex64>>().map_err(|_| {
                    PyValueError::new_err("Plan(dtype='complex128') requires a complex128 array")
                })?;
                let ro = arr.readonly();
                let src = ro.as_slice()?;
                let mut buf = vec![Complex::new(0.0_f64, 0.0); self.n];
                buf[..src.len().min(self.n)].copy_from_slice(&src[..src.len().min(self.n)]);
                inv.execute(&mut buf).map_err(zaft_err)?;
                Ok(buf.into_pyarray(py).into_any().unbind())
            }
            PlanInner::F32 { inv, .. } => {
                let arr = input.cast::<PyArray1<Complex32>>().map_err(|_| {
                    PyValueError::new_err("Plan(dtype='complex64') requires a complex64 array")
                })?;
                let ro = arr.readonly();
                let src = ro.as_slice()?;
                let mut buf = vec![Complex::new(0.0_f32, 0.0); self.n];
                buf[..src.len().min(self.n)].copy_from_slice(&src[..src.len().min(self.n)]);
                inv.execute(&mut buf).map_err(zaft_err)?;
                Ok(buf.into_pyarray(py).into_any().unbind())
            }
        }
    }

    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    #[getter]
    fn workers(&self) -> usize {
        self.workers
    }

    #[getter]
    fn dtype(&self) -> &str {
        &self.dtype
    }

    fn __repr__(&self) -> String {
        format!(
            "zaft.Plan(n={}, dtype='{}', workers={})",
            self.n, self.dtype, self.workers
        )
    }
}

/// 1-D complex-to-complex forward FFT.
///
/// Parameters match ``numpy.fft.fft`` / ``scipy.fft.fft``.
#[pyfunction]
#[pyo3(signature = (a, n = None, norm = None))]
fn fft<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: Option<usize>,
    norm: Option<&str>,
) -> PyResult<Py<PyAny>> {
    if let Ok(arr) = a.cast::<PyArray1<Complex64>>() {
        let ro = arr.readonly();
        let src = ro.as_slice()?;
        let size = n.unwrap_or(src.len());
        let scale = parse_norm(norm, size, false)?;
        let mut res = c2c_f64(src, size, FftDirection::Forward)?;
        if (scale - 1.0).abs() > f64::EPSILON {
            res.iter_mut().for_each(|v| *v *= scale);
        }
        return Ok(res.into_pyarray(py).into_any().unbind());
    }
    if let Ok(arr) = a.cast::<PyArray1<Complex32>>() {
        let ro = arr.readonly();
        let src = ro.as_slice()?;
        let size = n.unwrap_or(src.len());
        let scale = parse_norm(norm, size, false)? as f32;
        let mut res = c2c_f32(src, size, FftDirection::Forward)?;
        if (scale - 1.0f32).abs() > f32::EPSILON {
            res.iter_mut().for_each(|v| *v *= scale);
        }
        return Ok(res.into_pyarray(py).into_any().unbind());
    }
    // coerce to complex128 and retry
    let coerced = a.call_method1("astype", ("complex128",))?;
    fft(py, &coerced, n, norm)
}

/// 1-D complex-to-complex inverse FFT.
#[pyfunction]
#[pyo3(signature = (a, n = None, norm = None))]
fn ifft<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: Option<usize>,
    norm: Option<&str>,
) -> PyResult<Py<PyAny>> {
    if let Ok(arr) = a.cast::<PyArray1<Complex64>>() {
        let ro = arr.readonly();
        let src = ro.as_slice()?;
        let size = n.unwrap_or(src.len());
        let scale = parse_norm(norm, size, true)?;
        let mut res = c2c_f64(src, size, FftDirection::Inverse)?;
        res.iter_mut().for_each(|v| *v *= scale);
        return Ok(res.into_pyarray(py).into_any().unbind());
    }
    if let Ok(arr) = a.cast::<PyArray1<Complex32>>() {
        let ro = arr.readonly();
        let src = ro.as_slice()?;
        let size = n.unwrap_or(src.len());
        let scale = parse_norm(norm, size, true)? as f32;
        let mut res = c2c_f32(src, size, FftDirection::Inverse)?;
        res.iter_mut().for_each(|v| *v *= scale);
        return Ok(res.into_pyarray(py).into_any().unbind());
    }
    let coerced = a.call_method1("astype", ("complex128",))?;
    ifft(py, &coerced, n, norm)
}

/// Real-to-complex FFT — returns ``n//2 + 1`` complex values.
#[pyfunction]
#[pyo3(signature = (a, n = None, norm = None))]
fn rfft<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: Option<usize>,
    norm: Option<&str>,
) -> PyResult<Py<PyAny>> {
    if let Ok(arr) = a.cast::<PyArray1<f64>>() {
        let ro = arr.readonly();
        let sl = ro.as_slice()?;
        let size = n.unwrap_or(sl.len());
        let scale = parse_norm(norm, size, false)?;
        let exec = Zaft::make_r2c_fft_f64(size).map_err(zaft_err)?;
        let mut input = vec![0.0_f64; size];
        input[..sl.len().min(size)].copy_from_slice(&sl[..sl.len().min(size)]);
        let mut output = vec![Complex::new(0.0_f64, 0.0); size / 2 + 1];
        exec.execute(&input, &mut output).map_err(zaft_err)?;
        if (scale - 1.0).abs() > f64::EPSILON {
            output.iter_mut().for_each(|v| *v *= scale);
        }
        return Ok(output.into_pyarray(py).into_any().unbind());
    }
    if let Ok(arr) = a.cast::<PyArray1<f32>>() {
        let ro = arr.readonly();
        let sl = ro.as_slice()?;
        let size = n.unwrap_or(sl.len());
        let scale = parse_norm(norm, size, false)? as f32;
        let exec = Zaft::make_r2c_fft_f32(size).map_err(zaft_err)?;
        let mut input = vec![0.0_f32; size];
        input[..sl.len().min(size)].copy_from_slice(&sl[..sl.len().min(size)]);
        let mut output = vec![Complex::new(0.0_f32, 0.0); size / 2 + 1];
        exec.execute(&input, &mut output).map_err(zaft_err)?;
        if (scale - 1.0f32).abs() > f32::EPSILON {
            output.iter_mut().for_each(|v| *v *= scale);
        }
        return Ok(output.into_pyarray(py).into_any().unbind());
    }
    let coerced = a.call_method1("astype", ("float64",))?;
    rfft(py, &coerced, n, norm)
}

// ─── irfft ───────────────────────────────────────────────────────────────────

/// Inverse real FFT — symmetric to ``rfft``.
#[pyfunction]
#[pyo3(signature = (a, n = None, norm = None))]
fn irfft<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: Option<usize>,
    norm: Option<&str>,
) -> PyResult<Py<PyAny>> {
    if let Ok(arr) = a.cast::<PyArray1<Complex64>>() {
        let ro = arr.readonly();
        let src = ro.as_slice()?;
        let in_len = src.len();
        let size = n.unwrap_or(2 * (in_len - 1));
        let scale = parse_norm(norm, size, true)?;
        let exec = Zaft::make_c2r_fft_f64(size).map_err(zaft_err)?;
        let mut output = vec![0.0_f64; size];
        exec.execute(src, &mut output).map_err(zaft_err)?;
        if (scale - 1.0).abs() > f64::EPSILON {
            output.iter_mut().for_each(|v| *v *= scale);
        }
        return Ok(output.into_pyarray(py).into_any().unbind());
    }
    if let Ok(arr) = a.cast::<PyArray1<Complex32>>() {
        let ro = arr.readonly();
        let src = ro.as_slice()?;
        let in_len = src.len();
        let size = n.unwrap_or(2 * (in_len - 1));
        let scale = parse_norm(norm, size, true)? as f32;
        let exec = Zaft::make_c2r_fft_f32(size).map_err(zaft_err)?;
        let mut output = vec![0.0_f32; size];
        exec.execute(src, &mut output).map_err(zaft_err)?;
        if (scale - 1.0f32).abs() > f32::EPSILON {
            output.iter_mut().for_each(|v| *v *= scale);
        }
        return Ok(output.into_pyarray(py).into_any().unbind());
    }
    let coerced = a.call_method1("astype", ("complex128",))?;
    irfft(py, &coerced, n, norm)
}

fn transpose_back<T: Copy + Default>(
    src: &[T],   // transposed: [out_cols × out_rows]
    rows: usize, // original rows    (= out_rows)
    cols: usize, // original columns (= out_cols)
) -> Vec<T> {
    // src layout after execute: element at (r, c) is at src[c * rows + r]
    // dst layout we want:       element at (r, c) is at dst[r * cols + c]
    let mut dst = vec![T::default(); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[r * cols + c] = src[c * rows + r];
        }
    }
    dst
}

/// 2-D complex FFT.
#[pyfunction]
#[pyo3(signature = (a, s = None, norm = None, workers = 1))]
fn fft2<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    s: Option<(usize, usize)>,
    norm: Option<&str>,
    workers: usize,
) -> PyResult<Py<PyAny>> {
    let arr = a
        .cast::<PyArray2<Complex64>>()
        .map_err(|_| PyValueError::new_err("fft2 requires a 2-D complex128 array"))?;
    let ro = arr.readonly();
    let shape = ro.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let (out_rows, out_cols) = s.unwrap_or((rows, cols));
    let scale = parse_norm(norm, out_rows * out_cols, false)?;
    let exec = Zaft::make_2d_c2c_fft_f64(out_cols, out_rows, FftDirection::Forward, workers.max(1))
        .map_err(zaft_err)?;
    let src = ro.as_slice()?;
    if src.len() < rows * cols {
        return Err(PyValueError::new_err("buffer size is too small"));
    }
    let mut buf = src[..rows * cols].to_vec();
    exec.execute(&mut buf).map_err(zaft_err)?;
    let mut buf = transpose_back(&buf, out_rows, out_cols);
    if (scale - 1.0).abs() > f64::EPSILON {
        buf.iter_mut().for_each(|v| *v *= scale);
    }
    let rows2d: Vec<Vec<Complex64>> = buf.chunks(out_cols).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows2d)?.into_any().unbind())
}

/// 2-D inverse complex FFT.
#[pyfunction]
#[pyo3(signature = (a, s = None, norm = None, workers = 1))]
fn ifft2<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    s: Option<(usize, usize)>,
    norm: Option<&str>,
    workers: usize,
) -> PyResult<Py<PyAny>> {
    let arr = a
        .cast::<PyArray2<Complex64>>()
        .map_err(|_| PyValueError::new_err("ifft2 requires a 2-D complex128 array"))?;
    let ro = arr.readonly();
    let shape = ro.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let (out_rows, out_cols) = s.unwrap_or((rows, cols));
    let scale = parse_norm(norm, out_rows * out_cols, true)?;
    let exec = Zaft::make_2d_c2c_fft_f64(out_cols, out_rows, FftDirection::Inverse, workers.max(1))
        .map_err(zaft_err)?;
    let src = ro.as_slice()?;
    if src.len() < rows * cols {
        return Err(PyValueError::new_err("buffer size is too small"));
    }
    let mut buf = transpose_back(&src, out_cols, out_rows);
    exec.execute(&mut buf).map_err(zaft_err)?;
    buf.iter_mut().for_each(|v| *v *= scale);
    // buf after execute is row-major [out_rows x out_cols] — read as such.
    let rows2d: Vec<Vec<Complex64>> = buf.chunks(out_cols).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows2d)?.into_any().unbind())
}

/// N-D complex FFT (delegates to numpy for axis management).
#[pyfunction]
#[pyo3(signature = (a, s = None, axes = None, norm = None, workers = 1))]
fn fftn<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
    norm: Option<&str>,
    workers: usize,
) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    if let Some(v) = s {
        kwargs.set_item("s", v)?;
    }
    if let Some(v) = axes {
        kwargs.set_item("axes", v)?;
    }
    if let Some(v) = norm {
        kwargs.set_item("norm", v)?;
    }
    let _ = workers;
    Ok(np.call_method("fft", (a,), Some(&kwargs))?.unbind())
}

/// N-D inverse complex FFT (delegates to numpy for axis management).
#[pyfunction]
#[pyo3(signature = (a, s = None, axes = None, norm = None, workers = 1))]
fn ifftn<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
    norm: Option<&str>,
    workers: usize,
) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    if let Some(v) = s {
        kwargs.set_item("s", v)?;
    }
    if let Some(v) = axes {
        kwargs.set_item("axes", v)?;
    }
    if let Some(v) = norm {
        kwargs.set_item("norm", v)?;
    }
    let _ = workers;
    Ok(np.call_method("ifft", (a,), Some(&kwargs))?.unbind())
}

/// DFT sample frequencies — matches ``numpy.fft.fftfreq``.
#[pyfunction]
#[pyo3(signature = (n, d = None))]
fn fftfreq<'py>(py: Python<'py>, n: usize, d: Option<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let d = d.unwrap_or(1.0);
    let half = n.div_ceil(2);
    let mut out = vec![0.0_f64; n];
    for (i, dst) in out.iter_mut().enumerate().take(half) {
        *dst = i as f64 / (n as f64 * d);
    }
    for (i, dst) in out.iter_mut().enumerate().take(n).skip(half) {
        *dst = (i as f64 - n as f64) / (n as f64 * d);
    }
    Ok(out.into_pyarray(py))
}

/// Sample frequencies for rfft — matches ``numpy.fft.rfftfreq``.
#[pyfunction]
#[pyo3(signature = (n, d = None))]
fn rfftfreq<'py>(py: Python<'py>, n: usize, d: Option<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let d = d.unwrap_or(1.0);
    let out: Vec<f64> = (0..n / 2 + 1).map(|i| i as f64 / (n as f64 * d)).collect();
    Ok(out.into_pyarray(py))
}

/// Shift zero-frequency component to centre — matches ``numpy.fft.fftshift``.
#[pyfunction]
fn fftshift<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, Complex64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let sl = a.as_slice()?;
    let n = sl.len();
    let half = n / 2;
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    out[..n - half].copy_from_slice(&sl[half..]);
    out[n - half..].copy_from_slice(&sl[..half]);
    Ok(out.into_pyarray(py))
}

/// Inverse fftshift — matches ``numpy.fft.ifftshift``.
#[pyfunction]
fn ifftshift<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, Complex64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let sl = a.as_slice()?;
    let n = sl.len();
    let half = n.div_ceil(2);
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    out[..n - half].copy_from_slice(&sl[half..]);
    out[n - half..].copy_from_slice(&sl[..half]);
    Ok(out.into_pyarray(py))
}

/// Smallest integer >= ``target`` that Zaft transforms efficiently (13-smooth).
///
/// Equivalent to ``scipy.fft.next_fast_len``.
#[pyfunction]
fn next_fast_len(target: usize) -> usize {
    if target <= 1 {
        return target;
    }
    let smooth = |mut n: usize| -> bool {
        for p in [2usize, 3, 5, 7, 11, 13] {
            while n.is_multiple_of(p) {
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

#[pymodule]
fn _zaft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Plan>()?;
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    m.add_function(wrap_pyfunction!(ifft, m)?)?;
    m.add_function(wrap_pyfunction!(rfft, m)?)?;
    m.add_function(wrap_pyfunction!(irfft, m)?)?;
    m.add_function(wrap_pyfunction!(fft2, m)?)?;
    m.add_function(wrap_pyfunction!(ifft2, m)?)?;
    m.add_function(wrap_pyfunction!(fftn, m)?)?;
    m.add_function(wrap_pyfunction!(ifftn, m)?)?;
    m.add_function(wrap_pyfunction!(fftfreq, m)?)?;
    m.add_function(wrap_pyfunction!(rfftfreq, m)?)?;
    m.add_function(wrap_pyfunction!(fftshift, m)?)?;
    m.add_function(wrap_pyfunction!(ifftshift, m)?)?;
    m.add_function(wrap_pyfunction!(next_fast_len, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
