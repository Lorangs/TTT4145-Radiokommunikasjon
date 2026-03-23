"""
implementation of reed-solomon codes and convolutional codes for forward error correction (FEC)
"""

from reedsolo import RSCodec, ReedSolomonError
import numpy as np

class FCCodec:
    def __init__(self, config: dict):
        self.num_ecc = int(config['coding']['rs_num_ecc'])
        self.rsc = RSCodec(self.num_ecc * 2)  # Initialize Reed-Solomon codec with enough ECC symbols to correct rs_num_ecc errors

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data using Reed-Solomon code."""
        return np.array(self.rsc.encode(data.tobytes()), dtype=np.uint8)

    def rs_decode(self, encoded_data: np.ndarray) -> np.ndarray | None:
        """Decode data using Reed-Solomon code, correcting errors if possible."""
        try:
            decoded_msg, decoded_msg_ecc, errata_pos = self.rsc.decode(encoded_data.tobytes())
            return np.array(decoded_msg, dtype=np.uint8)
        except ReedSolomonError as e:
            print(f"Reed-Solomon decoding failed: {e}")
            return None





if __name__ == "__main__":
    fc_codec = FCCodec(config={
        'coding': {
            'rs_num_ecc': 16
        }
    })


    test_msg = "Hello, World!"
    original_data = np.frombuffer(test_msg.encode('utf-8'), dtype=np.uint8)
    print("Original data:")
    print(original_data)

    encoded_data = fc_codec.encode(original_data)
    print("Encoded data:")
    print(encoded_data)
    print()

    # Introduce some errors for testing
    for i in range(fc_codec.num_ecc + 1):
        print(f"Number of errors: {i+1}")
        encoded_data[i] ^= 0xFF  # Flip bits to simulate errors
        print("Corrupted encoded data:")
        print(encoded_data)


        decoded_data = fc_codec.rs_decode(encoded_data)
        if decoded_data is not None:
            print("Decoded data:")
            print(np.frombuffer(decoded_data, dtype=np.uint8))
            print()
        else :
            print(f"Failed to decode data with {i+1} errors.\n")

    