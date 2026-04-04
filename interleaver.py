"""
Dynamic-length bit interleaver.

The permutation is derived from:
    - a fixed seed from config
    - the actual bitstream length at runtime

This lets TX and RX independently rebuild the same permutation for each frame
without relying on a single global interleaver length.
"""

import numpy as np


class Interleaver:
    def __init__(self, config: dict):
        self.seed = int(config["coding"]["interleaver_seed"])
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _permutations(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        if length <= 0:
            raise ValueError("Interleaver length must be positive.")

        cached = self._cache.get(length)
        if cached is not None:
            return cached

        rng = np.random.default_rng(self.seed)
        permutation = rng.permutation(length)
        inverse = np.empty(length, dtype=np.int64)
        inverse[permutation] = np.arange(length, dtype=np.int64)

        self._cache[length] = (permutation, inverse)
        return permutation, inverse

    def interleave(self, encoded_bits: np.ndarray) -> np.ndarray:
        """Interleave bits using a permutation derived from the input length."""
        permutation, _ = self._permutations(int(encoded_bits.size))
        return encoded_bits[permutation]

    def deinterleave(self, interleaved_bits: np.ndarray) -> np.ndarray:
        """Restore original bit order using the inverse permutation for this length."""
        _, inverse = self._permutations(int(interleaved_bits.size))
        return interleaved_bits[inverse]


if __name__ == "__main__":
    # Example usage
    interleaver = Interleaver(config={
        'coding': {
            'interleaver_seed': 42,
            'interleaver_length': 384 # must be equal to total bit stream length before upsampling.
        }
    })
    
    #test_bits = np.random.randint(0, 2, 256 + 128)  # Generate a random bit stream of length 256
    test_bits = np.repeat(np.array([0, 1], dtype=np.uint8), 192)  # Generate a test bit stream of length 384 with a known pattern
    print("Original bits:")
    print(test_bits)
    print()

    interleaved_bits = interleaver.interleave(test_bits)
    print("Interleaved bits:")
    print(interleaved_bits)
    print()

    deinterleaved_bits = interleaver.deinterleave(interleaved_bits)
    bits_check = np.all(test_bits == deinterleaved_bits)
    print("Deinterleaving successful, bits match!" if bits_check else "Deinterleaving failed, bits do not match.")
    print(deinterleaved_bits)
