"""
tests/test_zaft.py
------------------
Validate zaft's output against NumPy's reference implementation.
Run with:  pytest -v
"""
import numpy as np
import pytest

import zaft

RNG = np.random.default_rng(42)
SIZES_1D = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 23, 64,
            100, 127, 128, 256, 512, 1000, 1024, 1800, 2048]
SIZES_2D = [(4, 4), (8, 16), (32, 32), (100, 100)]


# ─── helpers ──────────────────────────────────────────────────────────────────

def cplx(n: int) -> np.ndarray:
    return (RNG.standard_normal(n) + 1j * RNG.standard_normal(n)).astype(np.complex128)


def real(n: int) -> np.ndarray:
    return RNG.standard_normal(n).astype(np.float64)


def cplx2(rows: int, cols: int) -> np.ndarray:
    return (RNG.standard_normal((rows, cols)) + 1j * RNG.standard_normal((rows, cols))).astype(np.complex128)


# ─── fft / ifft ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES_1D)
def test_fft_matches_numpy(n):
    x = cplx(n)
    np.testing.assert_allclose(zaft.fft(x), np.fft.fft(x), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("n", SIZES_1D)
def test_ifft_matches_numpy(n):
    x = cplx(n)
    np.testing.assert_allclose(zaft.ifft(x), np.fft.ifft(x), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("n", SIZES_1D)
def test_fft_roundtrip(n):
    x = cplx(n)
    np.testing.assert_allclose(zaft.ifft(zaft.fft(x)), x, rtol=1e-9, atol=1e-10)


# ─── norm variants ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
def test_fft_norm(norm):
    x = cplx(64)
    np.testing.assert_allclose(
        zaft.fft(x, norm=norm),
        np.fft.fft(x, norm=norm),
        rtol=1e-9, atol=1e-10,
    )


@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
def test_ifft_norm(norm):
    x = cplx(64)
    np.testing.assert_allclose(
        zaft.ifft(x, norm=norm),
        np.fft.ifft(x, norm=norm),
        rtol=1e-9, atol=1e-10,
    )


# ─── zero-pad / truncate ──────────────────────────────────────────────────────

def test_fft_zero_pad():
    x = cplx(64)
    np.testing.assert_allclose(zaft.fft(x, n=128), np.fft.fft(x, n=128), rtol=1e-9, atol=1e-10)


def test_fft_truncate():
    x = cplx(128)
    np.testing.assert_allclose(zaft.fft(x, n=64), np.fft.fft(x, n=64), rtol=1e-9, atol=1e-10)


# ─── rfft / irfft ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES_1D)
def test_rfft_matches_numpy(n):
    x = real(n)
    np.testing.assert_allclose(zaft.rfft(x), np.fft.rfft(x), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("n", SIZES_1D)
def test_irfft_matches_numpy(n):
    x = real(n)
    X = np.fft.rfft(x)
    np.testing.assert_allclose(zaft.irfft(X, n=n), np.fft.irfft(X, n=n), rtol=1e-9, atol=1e-10)


def test_rfft_roundtrip():
    x = real(1024)
    np.testing.assert_allclose(zaft.irfft(zaft.rfft(x), n=len(x)), x, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("shape", SIZES_2D)
def test_fft2_matches_numpy(shape):
    x = cplx2(*shape)
    np.testing.assert_allclose(zaft.fft2(x), np.fft.fft2(x), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("shape", SIZES_2D)
def test_ifft2_matches_numpy(shape):
    x = cplx2(*shape)
    np.testing.assert_allclose(zaft.ifft2(x), np.fft.ifft2(x), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("shape", SIZES_2D)
def test_fft2_roundtrip(shape):
    x = cplx2(*shape)
    np.testing.assert_allclose(zaft.ifft2(zaft.fft2(x)), x, rtol=1e-9, atol=1e-10)


# ─── fftfreq / rfftfreq ───────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [8, 9, 64, 100])
def test_fftfreq(n):
    np.testing.assert_allclose(zaft.fftfreq(n), np.fft.fftfreq(n))
    np.testing.assert_allclose(zaft.fftfreq(n, d=0.01), np.fft.fftfreq(n, d=0.01))


@pytest.mark.parametrize("n", [8, 9, 64, 100])
def test_rfftfreq(n):
    np.testing.assert_allclose(zaft.rfftfreq(n), np.fft.rfftfreq(n))


# ─── fftshift / ifftshift ─────────────────────────────────────────────────────

def test_fftshift():
    x = cplx(64)
    np.testing.assert_allclose(zaft.fftshift(x), np.fft.fftshift(x))


def test_ifftshift():
    x = cplx(64)
    np.testing.assert_allclose(zaft.ifftshift(x), np.fft.ifftshift(x))


def test_shift_roundtrip():
    x = cplx(64)
    np.testing.assert_allclose(zaft.ifftshift(zaft.fftshift(x)), x)


# ─── next_fast_len ────────────────────────────────────────────────────────────

def test_next_fast_len_basic():
    assert zaft.next_fast_len(100) == 100  # 100 = 2^2 * 5^2
    assert zaft.next_fast_len(101) == 104  # 104 = 2^3 * 13
    assert zaft.next_fast_len(1) == 1


def test_next_fast_len_is_fast():
    """Every result should be 13-smooth."""
    for n in range(1, 300):
        m = zaft.next_fast_len(n)
        assert m >= n
        tmp = m
        for p in (2, 3, 5, 7, 11, 13):
            while tmp % p == 0:
                tmp //= p
        assert tmp == 1, f"next_fast_len({n})={m} is not 13-smooth"


# ─── Plan ─────────────────────────────────────────────────────────────────────

def test_plan_forward_f64():
    n = 256
    x = cplx(n)
    plan = zaft.Plan(n, dtype="complex128")
    result = plan.execute_forward(x)
    np.testing.assert_allclose(result, np.fft.fft(x), rtol=1e-9, atol=1e-10)
    assert result.dtype == np.complex128


def test_plan_forward_f32():
    n = 256
    x = cplx(n).astype(np.complex64)
    plan = zaft.Plan(n, dtype="complex64")
    result = plan.execute_forward(x)
    np.testing.assert_allclose(result, np.fft.fft(x.astype(np.complex128)), rtol=1e-4, atol=1e-4)
    assert result.dtype == np.complex64


def test_plan_inverse_f64():
    n = 256
    x = cplx(n)
    plan = zaft.Plan(n, dtype="complex128")
    result = plan.execute_inverse(x)
    # Plan.execute_inverse is unnormalised — divide by n to get numpy's ifft
    np.testing.assert_allclose(result / n, np.fft.ifft(x), rtol=1e-9, atol=1e-10)


def test_plan_roundtrip_f64():
    n = 512
    x = cplx(n)
    plan = zaft.Plan(n, dtype="complex128")
    X = plan.execute_forward(x)
    x2 = plan.execute_inverse(X) / n
    np.testing.assert_allclose(x2, x, rtol=1e-9, atol=1e-10)


def test_plan_roundtrip_f32():
    n = 512
    x = cplx(n).astype(np.complex64)
    plan = zaft.Plan(n, dtype="complex64")
    X = plan.execute_forward(x)
    x2 = (plan.execute_inverse(X) / n).astype(np.complex64)
    np.testing.assert_allclose(x2, x, rtol=1e-4, atol=1e-4)


def test_plan_wrong_dtype_raises():
    n = 64
    plan_f64 = zaft.Plan(n, dtype="complex128")
    x32 = cplx(n).astype(np.complex64)
    with pytest.raises(Exception):
        plan_f64.execute_forward(x32)

    plan_f32 = zaft.Plan(n, dtype="complex64")
    x64 = cplx(n)
    with pytest.raises(Exception):
        plan_f32.execute_forward(x64)


def test_plan_repr():
    plan = zaft.Plan(512, dtype="complex64", workers=2)
    assert "512" in repr(plan)
    assert "complex64" in repr(plan)
    assert "2" in repr(plan)


def test_plan_dtype_getter():
    assert zaft.Plan(16, dtype="complex128").dtype == "complex128"
    assert zaft.Plan(16, dtype="complex64").dtype == "complex64"


def test_plan_zero_size_raises():
    with pytest.raises(Exception):
        zaft.Plan(0)


def test_plan_bad_dtype_raises():
    with pytest.raises(Exception):
        zaft.Plan(16, dtype="float16")


# ─── error handling ───────────────────────────────────────────────────────────

def test_fft_bad_norm():
    x = cplx(16)
    with pytest.raises(ValueError, match="norm"):
        zaft.fft(x, norm="invalid")