"""Minimal 3-tap complex FIR equalizer trained on known header symbols."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _as_complex_1d(values, name: str) -> npt.NDArray[np.complex64]:
    array = np.asarray(values, dtype=np.complex64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return array


def train_3tap_equalizer(received_symbols, known_symbols) -> npt.NDArray[np.complex64]:
    """
    Train a 3-tap complex FIR equalizer from known header symbols.
    """
    received = _as_complex_1d(received_symbols, "received_symbols")
    target = _as_complex_1d(known_symbols, "known_symbols")

    if received.size != target.size:
        raise ValueError("received_symbols and known_symbols must have the same length.")

    sample_count = int(received.size)
    X = np.zeros((sample_count, 3), dtype=np.complex64)
    for index in range(sample_count):
        X[index, 0] = received[index - 1] if index - 1 >= 0 else 0.0
        X[index, 1] = received[index]
        X[index, 2] = received[index + 1] if index + 1 < sample_count else 0.0

    taps, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
    return taps.astype(np.complex64, copy=False)


def apply_3tap_equalizer(rx_symbols , taps) -> npt.NDArray[np.complex64]:
    """
    Apply the centered 3-tap complex FIR equalizer.
    """
    received = _as_complex_1d(rx_symbols, "rx_symbols")
    coefficients = _as_complex_1d(taps, "taps")
    if coefficients.size != 3:
        raise ValueError("taps must contain exactly 3 coefficients.")

    sample_count = int(received.size)
    equalized = np.empty(sample_count, dtype=np.complex64)
    for index in range(sample_count):
        left = received[index - 1] if index - 1 >= 0 else 0.0
        center = received[index]
        right = received[index + 1] if index + 1 < sample_count else 0.0
        equalized[index] = (
            left * coefficients[0]
            + center * coefficients[1]
            + right * coefficients[2]
        )
    return equalized


def equalize_from_known_header(
    symbol_stream,
    training_start: int,
    known_symbols,
) -> npt.NDArray[np.complex64]:
    """
    Train the 3-tap equalizer on a known header block and apply it to the full stream.
    """
    received = _as_complex_1d(symbol_stream, "symbol_stream")
    known = _as_complex_1d(known_symbols, "known_symbols")

    start = int(training_start)
    stop = start + int(known.size)
    if start < 0 or stop > received.size or known.size == 0:
        return received

    taps = train_3tap_equalizer(received[start:stop], known)
    return apply_3tap_equalizer(received, taps)
