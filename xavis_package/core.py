"""
xavis_package.core

Small DSP helper functions for Lab 1.
"""

from __future__ import annotations
import numpy as np


def load_npz_arr0(filename: str) -> np.ndarray:
    """Load a .npz file and return arr_0."""
    try:
        return np.load(filename)["arr_0"]
    except Exception as e:
        raise FileNotFoundError(f"Could not load arr_0 from file: {filename}") from e


def time_axis(N: int, fs: float) -> np.ndarray:
    """Return time axis (seconds) for N samples at sampling rate fs."""
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if N < 0:
        raise ValueError("N must be >= 0")
    return np.arange(N) / fs


def fft_spectrum(x: np.ndarray, fs: float, one_sided: bool = True):
    """Return frequency axis and power spectrum |FFT|^2."""
    if fs <= 0:
        raise ValueError("fs must be > 0")

    x = np.asarray(x).ravel()
    N = x.size
    if N == 0:
        return np.array([]), np.array([])

    X = np.fft.fft(x)
    P = np.abs(X) ** 2
    f = np.fft.fftfreq(N, d=1 / fs)

    if one_sided:
        mask = f >= 0
        return f[mask], P[mask]

    return f, P


def alias_frequency(f0: float, fs: float) -> float:
    """Return aliased frequency for tone f0 sampled at fs."""
    if fs <= 0:
        raise ValueError("fs must be > 0")
    fa = abs(f0 - round(f0 / fs) * fs)
    return fa if fa <= fs / 2 else fs - fa


def dominant_frequency(f: np.ndarray, P: np.ndarray, fmin: float = 0) -> float:
    """Return frequency of dominant spectral peak above fmin."""
    f = np.asarray(f).ravel()
    P = np.asarray(P).ravel()

    if f.size == 0 or P.size == 0:
        raise ValueError("f and P must be non-empty")
    if f.shape != P.shape:
        raise ValueError("f and P must have the same shape")

    mask = f >= fmin
    if not np.any(mask):
        raise ValueError("No frequencies >= fmin found")

    return f[mask][np.argmax(P[mask])]


def rms(x: np.ndarray) -> float:
    """Root-mean-square value."""
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x ** 2)))


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-noise ratio in dB."""
    signal = np.asarray(signal)
    noise = np.asarray(noise)

    if signal.size == 0 or noise.size == 0:
        return float("nan")

    ps = np.mean(signal ** 2)
    pn = np.mean(noise ** 2)
    if pn == 0:
        return float("inf")

    return float(10 * np.log10(ps / pn))

