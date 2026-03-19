"""
Barker and Barker-like preamble generator module.

True Barker codes only exist for a small set of lengths, with 13 being the
longest binary Barker sequence. To support longer preambles in the current RX/TX
pipeline, this module keeps the true Barker sequences where available and uses a
deterministic maximal-length-sequence fallback for longer lengths up to 70 bits.

Exports:
    BARKER_BITS:
        Dict[int, int] of exact-length bit patterns represented as Python ints.
    BARKER_BIT_ARRAYS:
        Dict[int, np.ndarray] of 0/1 bit arrays.
    BARKER_SYMBOLS:
        Dict[str, Dict[int, np.ndarray]] of BPSK and QPSK preamble symbols.
"""

from __future__ import annotations

import numpy as np


MIN_BARKER_LENGTH = 2
MAX_BARKER_LENGTH = 70
SUPPORTED_BARKER_LENGTHS = tuple(range(MIN_BARKER_LENGTH, MAX_BARKER_LENGTH + 1))


# True binary Barker codes mapped to BPSK levels (+1 / -1).
_TRUE_BARKER_BPSK = {
    2: np.array([1, 1], dtype=np.int8),
    3: np.array([1, 1, -1], dtype=np.int8),
    4: np.array([1, 1, -1, 1], dtype=np.int8),
    5: np.array([1, 1, 1, -1, 1], dtype=np.int8),
    7: np.array([1, 1, 1, -1, -1, 1, -1], dtype=np.int8),
    11: np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1], dtype=np.int8),
    13: np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.int8),
}


# Tap positions for a simple Fibonacci LFSR implementation where state[-1] is
# emitted first and the new bit is inserted at state[0].
_MSEQ_TAPS = {
    2: (1, 2),
    3: (1, 2),
    4: (1, 2),
    5: (1, 3),
    6: (1, 2),
    7: (1, 2),
}


def _select_mseq_order(length: int) -> int:
    for order in sorted(_MSEQ_TAPS):
        if (2**order) - 1 >= length:
            return order
    raise ValueError(f"Unsupported Barker-like length: {length}")


def _generate_mseq_bpsk(length: int) -> np.ndarray:
    """Generate a deterministic Barker-like sequence with good autocorrelation."""
    order = _select_mseq_order(length)
    taps = _MSEQ_TAPS[order]
    state = np.ones(order, dtype=np.uint8)
    sequence = np.empty(length, dtype=np.int8)

    for index in range(length):
        sequence[index] = 1 if state[-1] else -1
        feedback = 0
        for tap in taps:
            feedback ^= int(state[-tap])
        state[1:] = state[:-1]
        state[0] = feedback

    return sequence


def _sequence_to_bits(sequence: np.ndarray) -> np.ndarray:
    return (sequence > 0).astype(np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _build_bpsk_sequences() -> dict[int, np.ndarray]:
    sequences: dict[int, np.ndarray] = {}
    for length in SUPPORTED_BARKER_LENGTHS:
        if length in _TRUE_BARKER_BPSK:
            sequences[length] = _TRUE_BARKER_BPSK[length].copy()
        else:
            sequences[length] = _generate_mseq_bpsk(length)
    return sequences


_BPSK_SEQUENCES = _build_bpsk_sequences()

BARKER_BIT_ARRAYS = {
    length: _sequence_to_bits(sequence)
    for length, sequence in _BPSK_SEQUENCES.items()
}

BARKER_BITS = {
    length: _bits_to_int(bits)
    for length, bits in BARKER_BIT_ARRAYS.items()
}

BARKER_SYMBOLS = {
    "BPSK": {
        length: sequence.copy()
        for length, sequence in _BPSK_SEQUENCES.items()
    },
    "QPSK": {
        length: (sequence.astype(np.float32) + 1j * sequence.astype(np.float32)).astype(np.complex64)
        for length, sequence in _BPSK_SEQUENCES.items()
    },
}


if __name__ == "__main__":
    for modulation, sequence_map in BARKER_SYMBOLS.items():
        print(f"{modulation} supports lengths {SUPPORTED_BARKER_LENGTHS[0]}-{SUPPORTED_BARKER_LENGTHS[-1]}")
        for length in (2, 3, 4, 5, 7, 11, 13, 16, 32, 64, 70):
            symbols = sequence_map[length]
            print(f"Length {length}: {symbols}")
        print()
