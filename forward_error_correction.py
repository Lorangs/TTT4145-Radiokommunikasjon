"""
implementation of reed-solomon codes and convolutional codes for forward error correction (FEC)

For 32 ECC symbols, the algorithm can correct up to 128 byte errors in the original message.
For 16 ECC symbols. The algorithm adds 64 bytes of redundancy to the original message.
For 8 ECC symbols, it adds 32 bytes of redundancy
"""

import contextlib
import io

from reedsolo import RSCodec, ReedSolomonError
import numpy as np
import numpy.typing as npt
from project_logger import get_logger

logger = get_logger(__name__)

class FCCodec:
    def __init__(self, config: dict):
        self.num_ecc = int(config['coding']['rs_num_ecc'])
        self.rsc = RSCodec(self.num_ecc)  # Initialize Reed-Solomon codec with enough ECC symbols to correct rs_num_ecc errors

    def encode(self, data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Encode data using Reed-Solomon code."""
        return np.array(self.rsc.encode(data), dtype=np.uint8)

    def decode(self, encoded_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Decode data using Reed-Solomon code, correcting errors if possible."""
        try:
            decoded_msg, decoded_msg_ecc, errata_pos = self.rsc.decode(encoded_data)
            
            return np.array(decoded_msg, dtype=np.uint8)
        except ReedSolomonError as e:
            logger.warning("Reed-Solomon decoding failed: %s", e)
            raise ValueError(f"Reed-Solomon decoding failed: {e}")
        except Exception as e:
            logger.error("Unexpected error during decoding: %s", e)
            raise RuntimeError(f"Unexpected error during decoding: {e}")



MAP_ECC_TO_ADDITIONAL_BYTES = {
    8: 16,   
    32: 128, # 32 ECC symbols can correct up to 128 byte errors
}

if __name__ == "__main__":
    from datagram import Datagram, msgType
    from scrambler import LFSRScrambler
    scrambler = LFSRScrambler(config={
        'coding': {
            'scrambler_seed': 0b1010101,  # Example non-zero seed
            'scrambler_register_length': 8
        }
    })

    fc_codec = FCCodec(config={
        'coding': {
            'rs_num_ecc': 8
        }
    })


    datagram = Datagram.as_string(msg_id=4, msg_type=msgType.DATA, text="Hello, World!")
    print("Original data:")
    print(datagram)

    bytes_arr = datagram.pack()
    print("Packed data:")
    print(bytes_arr)

    encoded_data = fc_codec.encode(bytes_arr)
    print("Encoded data:")
    print(encoded_data)
    print(f"added bytes:\t{len(encoded_data) - len(bytes_arr)}")
    print()

    scrambled_data = scrambler.apply(encoded_data)

    rolled_data = np.roll(scrambled_data, 1)
    descrambled_data = scrambler.apply(rolled_data7ujm0okm)
    print("Descrambled data:")
    print(descrambled_data)

    try:
        decoded_data = fc_codec.decode(descrambled_data)
        print("Decoded data:")
        print(decoded_data)
    except ReedSolomonError as e:
        print(f"Decoding failed with ReedSolomonError: {e}")
    except ValueError as e:
        print(f"Decoding failed with ValueError: {e}")
    except RuntimeError as e:
        print(f"Decoding failed with RuntimeError: {e}")