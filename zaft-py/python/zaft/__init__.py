"""
zaft — Zero-overhead, Architecture-aware Fast Fourier Transform
===============================================================

A drop-in replacement for ``numpy.fft`` and ``scipy.fft`` backed by a
hand-tuned Rust engine with SIMD acceleration (NEON on AArch64, AVX2+FMA on
x86-64, WASM SIMD on WebAssembly).

Quick start
-----------
>>> import numpy as np
>>> import zaft
>>> x = np.random.randn(1024).astype(np.complex128)
>>> X = zaft.fft(x)
>>> x2 = zaft.ifft(X)

Planner
-------
Create a :class:`Plan` when you need to repeat the same transform many times
so that the planning cost is paid only once:

>>> plan = zaft.Plan(n=1024, workers=4)
>>> X = plan.execute_forward(x)

API compatibility
-----------------
All functions accept the same arguments as their ``numpy.fft`` / ``scipy.fft``
equivalents (``n``, ``norm``, ``workers``, ``axis``, ``axes``, ``s``).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

# Import the compiled Rust extension.
from zaft._zaft import (  # noqa: F401 — re-exported as the public API
    Plan,
    fft,
    fft2,
    fftfreq,
    fftn,
    fftshift,
    ifft,
    ifft2,
    ifftshift,
    ifftn,
    irfft,
    next_fast_len,
    rfft,
    rfftfreq,
)

try:
    from zaft._zaft import __version__
except ImportError:  # built without Cargo metadata
    __version__ = "0.0.0+dev"

__all__ = [
    # 1-D transforms
    "fft",
    "ifft",
    "rfft",
    "irfft",
    # 2-D transforms
    "fft2",
    "ifft2",
    # N-D transforms
    "fftn",
    "ifftn",
    # Frequency helpers
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
    # Utilities
    "next_fast_len",
    # Planner
    "Plan",
]