import numpy as np


class LFSRScrambler:
    """Synchronous additive scrambler based on a linear-feedback shift register.

    The LFSR generates a deterministic pseudo-random bit stream from a shared
    non-zero seed. Each payload bit is XORed with that stream to whiten long runs
    of zeros/ones before modulation. Because the operation is XOR with the same
    generated sequence, the receiver descrambles by running the exact same logic
    again with the same seed at the start of the packet.

    This implementation uses the polynomial x^7 + x^4 + 1:
        - the register has 7 stages
        - feedback is formed from taps 7 and 4
        - the all-zero state is forbidden because it would produce only zeros
    """
    
    def __init__(self, seed: int = 0x5D, register_length: int = 7):
        if seed == 0:
            raise ValueError("Scrambler seed must be non-zero.")
        self.seed = seed
        self.register_length = register_length
        self.mask = (1 << register_length) - 1

    def apply(self, bits: np.ndarray) -> np.ndarray:
        """Scramble or descramble a packet bitstream.

        For a synchronous additive scrambler, TX and RX perform the same XOR
        operation against the same pseudo-random sequence. The caller is expected
        to reset the register to the shared seed once per packet.
        """
        state = self.seed
        out = np.empty(bits.size, dtype=np.uint8)

        for i, bit in enumerate(bits.astype(np.uint8)):
            # Generate the next pseudo-random bit from the LFSR taps. The shifts
            # are zero-based here, so state bit 6 corresponds to x^7 and bit 3
            # corresponds to x^4 in the polynomial description.
            prn = ((state >> 6) ^ (state >> 3)) & 0x1  # x^7 + x^4 + 1
            out[i] = bit ^ prn

            # Shift left and insert the newly generated feedback bit into the
            # register. The mask keeps only the configured register length.
            state = ((state << 1) | prn) & self.mask

        return out



# Example usage
if __name__ == "__main__":
    scrambler = LFSRScrambler(seed=0x5D, register_length=7)
    input_bits = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
    scrambled_bits = scrambler.apply(input_bits)
    print("Input bits:     ", input_bits)
    print("Scrambled bits: ", scrambled_bits)

    
