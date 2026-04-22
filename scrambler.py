import numpy as np
from project_logger import get_logger
import numpy.typing as npt
logger = get_logger(__name__)


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
    
    def __init__(self, config: dict):
        register_length = int(config["coding"]["scrambler_register_length"])

        seed_cfg = config["coding"].get("scrambler_seed", None)
        if seed_cfg is None or str(seed_cfg).lower() == "random":
            self.seed = self.random_seed(self.register_length)
        else:
            self.seed = int(seed_cfg)
        self.register_length = register_length
        self.mask = (1 << register_length) - 1      # 255 for 8 bits (uint8)

        # taps are 0 based from LSB (example: [0, 3] for x^7 + x^4 + 1)
        self.taps = tuple(int(t) for t in config["coding"].get("scrambler_taps", [0, 3]))
        for t in self.taps:
            if t < 0 or t >= self.register_length:
                raise ValueError(f"Invalid tap {t} for register length {self.register_length}.")

        self.mask = (1 << self.register_length) - 1
        self.seed &= self.mask
        if self.seed == 0:
            raise ValueError("Scrambler seed must be non-zero.")

        logger.info("Scrambler init: L=%d seed=0x%X taps=%s", self.register_length, self.seed, self.taps)

    @staticmethod
    def random_seed(register_length: int) -> int:
        rng = np.random.default_rng()
        return int(rng.integers(1, 1 << register_length))  # non-zero



    def apply(self, data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Scramble or descramble a packet bitstream.

        For a synchronous additive scrambler, TX and RX perform the same XOR
        operation against the same pseudo-random sequence. The caller is expected
        to reset the register to the shared seed once per packet.
        """
        arr = np.asarray(data, dtype=np.uint8)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D uint8 array, got shape={arr.shape}")

        out = np.empty_like(arr)
        state = self.seed
        for i, byte in enumerate(arr):
            prn, state = self._next_byte(state)
            out[i] = byte ^ prn
        return out

    def _next_bit(self, state: int) -> tuple[int, int]:
        """Generate the next bit and update the LFSR state."""
        out_bit = state & 0x1

        # feedback = XOR of taps
        fb = 0
        for t in self.taps:
            fb ^= (state >> t) & 0x1

        # shift right, insert feedback at MSB
        new_state = (state >> 1) | (fb << (self.register_length - 1))
        new_state &= self.mask
        if new_state == 0:
            new_state = self.seed  # avoid lock-up
        return out_bit, new_state
    
    def _next_byte(self, state: int) -> tuple[np.uint8, int]:
        b = 0
        for k in range(8):
            bit, state = self._next_bit(state)
            b |= (bit << k)
        return np.uint8(b), state



# Example usage
if __name__ == "__main__":
    from datagram import Datagram

    scrambler = LFSRScrambler(
        config={
            'coding': {
                'scrambler_seed': 0b1010101,  # Example non-zero seed
                'scrambler_register_length': 8
            }
        }
    )
    datagram = Datagram.as_string("Hello, world!")
    print(f"Original datagram: {datagram}")
    input_bits = np.frombuffer(datagram.pack(), dtype=np.uint8)

    scrambled_bits = scrambler.apply(input_bits)

    print("Input bits:     ", input_bits)
    print("Scrambled bits: ", scrambled_bits)

    unscrambled_bits = scrambler.apply(scrambled_bits)
    print("Unscrambled bits:", unscrambled_bits)
    rebuilt_datagram = Datagram.unpack(unscrambled_bits)
    print("Rebuilt datagram:", rebuilt_datagram)
    print("Rebuilt datagram string:\t", rebuilt_datagram.get_payload_as_string)





    
