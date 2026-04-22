"""
Bit interleaver for the current packed-byte runtime path.

This branch moves data between coding stages as 1D ``np.uint8`` arrays, where
each element is a packed byte. The interleaver keeps that external interface and
performs the permutation on the underlying bitstream internally.

The permutation is derived from:
    - a fixed seed from config
    - the actual bitstream length at runtime

This lets TX and RX independently rebuild the same permutation for each frame
without relying on a single global interleaver length.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _unpackbits_little(data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Unpack packed bytes to bits using little-endian bit order (LSB first)."""
    arr = np.asarray(data, dtype=np.uint8).reshape(-1)
    out = np.zeros(arr.size * 8, dtype=np.uint8)
    for i, value in enumerate(arr):
        base = i * 8
        int_value = int(value)
        for bit_index in range(8):
            out[base + bit_index] = np.uint8((int_value >> bit_index) & 0x1)
    return out


def _packbits_little(bits: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Pack bits to bytes using little-endian bit order (LSB first)."""
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n_bytes = (arr.size + 7) // 8
    out = np.zeros(n_bytes, dtype=np.uint8)
    for i, bit in enumerate(arr):
        if bit & 0x1:
            out[i // 8] = np.uint8(out[i // 8] | np.uint8(1 << (i % 8)))
    return out


class Interleaver:
    """
    Length-dependent bit interleaver with packed-byte input/output.

    The public API accepts and returns 1D ``np.uint8`` arrays so it matches the
    current branch's packed byte/bit pipeline:
"""

    def __init__(self, config: dict):
        coding_cfg = config.get("coding", {})
        self.seed = int(coding_cfg.get("interleaver_seed", 42))
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _permutations(self, bit_length: int) -> tuple[np.ndarray, np.ndarray]:
        if bit_length <= 0:
            raise ValueError("Interleaver bit length must be positive.")

        cached = self._cache.get(bit_length)
        if cached is not None:
            return cached

        rng = np.random.default_rng(self.seed)
        permutation = rng.permutation(bit_length)
        inverse = np.empty(bit_length, dtype=np.int64)
        inverse[permutation] = np.arange(bit_length, dtype=np.int64)

        self._cache[bit_length] = (permutation, inverse)
        return permutation, inverse

    def interleave_bits(self, bits: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Interleave an unpacked 1D bit array."""
        bit_array = np.asarray(bits, dtype=np.uint8).reshape(-1)
        permutation, _ = self._permutations(int(bit_array.size))
        return bit_array[permutation]

    def deinterleave_bits(self, bits: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Restore original order for an unpacked 1D bit array."""
        bit_array = np.asarray(bits, dtype=np.uint8).reshape(-1)
        _, inverse = self._permutations(int(bit_array.size))
        return bit_array[inverse]

    def interleave(self, data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Interleave a packed byte stream.

        Args:
            data: 1D ``np.uint8`` array where each element is one packed byte.

        Returns:
            Interleaved packed byte stream with the same length as the input.
        """
        arr = np.asarray(data, dtype=np.uint8)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D uint8 array, got shape={arr.shape}")

        bits = _unpackbits_little(arr)
        interleaved_bits = self.interleave_bits(bits)
        return _packbits_little(interleaved_bits)

    def deinterleave(self, data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Deinterleave a packed byte stream.

        Args:
            data: 1D ``np.uint8`` array where each element is one packed byte.

        Returns:
            Deinterleaved packed byte stream with the same length as the input.
        """
        arr = np.asarray(data, dtype=np.uint8)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D uint8 array, got shape={arr.shape}")

        bits = _unpackbits_little(arr)
        deinterleaved_bits = self.deinterleave_bits(bits)
        return _packbits_little(deinterleaved_bits)


if __name__ == "__main__":
    interleaver = Interleaver(
        config={
            "coding": {
                "interleaver_seed": 42,
            }
        }
    )

    test_bytes = np.array([0x96, 0x3C, 0xA5, 0x5A], dtype=np.uint8)
    print("Original packed bytes:")
    print(test_bytes)

    interleaved = interleaver.interleave(test_bytes)
    print("Interleaved packed bytes:")
    print(interleaved)

    recovered = interleaver.deinterleave(interleaved)
    print("Recovered packed bytes:")
    print(recovered)

    print("Match:", np.array_equal(test_bytes, recovered))
