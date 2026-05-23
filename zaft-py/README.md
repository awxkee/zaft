# zaft — Fast Fourier Transform for Python

[![PyPI](https://img.shields.io/pypi/v/zaft)](https://pypi.org/project/zaft/)
[![CI](https://github.com/yourname/zaft-py/actions/workflows/ci.yml/badge.svg)](https://github.com/yourname/zaft-py/actions)

**zaft** is a drop-in replacement for `numpy.fft` and `scipy.fft` backed by a
hand-tuned Rust engine with SIMD acceleration:

| Platform    | Instruction set |
|-------------|-----------------|
| x86-64      | AVX2 + FMA      |
| AArch64     | NEON / SVE      |

## Installation

```bash
pip install zaft
```

Pre-built wheels are available for Linux (x86-64, aarch64), macOS (Intel +
Apple Silicon), and Windows (x86-64).

## Usage

```python
import numpy as np
import zaft

x = np.random.randn(1024).astype(np.complex128)

# 1-D transforms — identical signatures to numpy.fft / scipy.fft
X = zaft.fft(x)
x2 = zaft.ifft(X)

# Real transforms
y = np.random.randn(1024)
Y = zaft.rfft(y)
y2 = zaft.irfft(Y, n=len(y))

# 2-D transforms
img = np.random.randn(512, 512).astype(np.complex128)
IMG = zaft.fft2(img)

# Frequency helpers
freqs = zaft.fftfreq(1024, d=1/44100)
```

### Planner

Create a `Plan` when the same transform length is reused many times — planning
cost is paid once and subsequent calls are essentially free:

```python
plan = zaft.Plan(n=1024, workers=4)

for frame in audio_frames:
    spectrum = plan.execute_forward(frame)
    # … process …
```

### Normalisation

The `norm` keyword matches `scipy.fft` / `numpy.fft`:

| `norm`                 | Forward scale | Inverse scale |
|------------------------|---------------|---------------|
| `"backward"` (default) | 1             | 1/N           |
| `"ortho"`              | 1/√N          | 1/√N          |
| `"forward"`            | 1/N           | 1             |

## API reference

| Function           | Equivalent in NumPy/SciPy          |
|--------------------|------------------------------------|
| `fft(a, n, norm)`  | `numpy.fft.fft`                    |
| `ifft(...)`        | `numpy.fft.ifft`                   |
| `rfft(...)`        | `numpy.fft.rfft`                   |
| `irfft(...)`       | `numpy.fft.irfft`                  |
| `fft2(...)`        | `numpy.fft.fft2`                   |
| `ifft2(...)`       | `numpy.fft.ifft2`                  |
| `fftn(...)`        | `numpy.fft.fftn`                   |
| `ifftn(...)`       | `numpy.fft.ifftn`                  |
| `fftfreq(n, d)`    | `numpy.fft.fftfreq`                |
| `rfftfreq(n, d)`   | `numpy.fft.rfftfreq`               |
| `fftshift(a)`      | `numpy.fft.fftshift`               |
| `ifftshift(a)`     | `numpy.fft.ifftshift`              |
| `next_fast_len(n)` | `scipy.fft.next_fast_len`          |
| `Plan(n, workers)` | `scipy.fft.get_backend` / planning |

## Algorithm selection

The library automatically chooses the fastest algorithm for each transform
length:

- **Powers of 2 / 4 / 8** — split-radix butterfly with AVX2/NEON kernels
- **Radix 3, 5, 6, 7, 10, 11, 13** — dedicated radix implementations
- **Composite lengths** — Good-Thomas or mixed-radix decomposition
- **Primes** — Rader's algorithm (convolution-based) or Bluestein's chirp-Z

## License

BSD 3-Clause — see [LICENSE](LICENSE.md).