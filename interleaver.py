"""
Wrapper class for interleaving and deinterleaving bits using a random interleaver from the commpy library.
"""

from commpy.channelcoding import RandInterlv
import numpy as np



class Interleaver:
    def __init__(self, config: dict):
        seed = int(config['coding']['interleaver_seed'])
        length = int(config['coding']['interleaver_length'])
        self.interleaver = RandInterlv(length, seed)

    def interleave(self, encoded_bits: np.ndarray) -> np.ndarray:
        """Interleave the encoded bits using the random interleaver."""
        return self.interleaver.interlv(encoded_bits)
    
    def deinterleave(self, interleaved_bits: np.ndarray) -> np.ndarray:
        """Deinterleave the received bits to restore original order."""
        return self.interleaver.deinterlv(interleaved_bits)


if __name__ == "__main__":
    # Example usage
    interleaver = Interleaver(config={
        'coding': {
            'interleaver_seed': 42,
            'interleaver_length': 10
        }
    })
    test_bits = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0], dtype=np.uint8)
    
    print("Original bits:")
    print(test_bits)
    print()

    interleaved_bits = interleaver.interleave(test_bits)
    print("Interleaved bits:")
    print(interleaved_bits)
    print()

    deinterleaved_bits = interleaver.deinterleave(interleaved_bits)
    print("Deinterleaved bits:")
    print(deinterleaved_bits)