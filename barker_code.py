"""
Barker code generator module.
Provides functionality to generate Barker codes of specified lengths.
All barker codes are predefined and stored in a dictionary for efficient retrieval. 
The codes are packed into bytes for transmission and can be unpacked back to bits when needed.
All Barker codes are padded to 16 bits (2 bytes) for uniformity in storage and transmission, with unused bits set to zero.
"""

import numpy as np

# Barker codes packed into bytes for transmission (2 bytes = 16 bits)
BARKER_CODES = {
    7: np.packbits(np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)),
    11: np.packbits(np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)),
    13: np.packbits(np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0], dtype=np.uint8)) 
}

def generate_barker_code(length: int) -> np.ndarray:
    """Generate Bipolar Barker code of specified length.
    Args:
        length (int): Length of the desired Barker code. Range 2 - 13.

    Returns:
        array: Barker code as a numpy array of bits , or None if length is invalid.
    """
    
    packed_code = BARKER_CODES.get(length)
    if packed_code is None:
        raise ValueError(f"Invalid Barker code length: {length}")
    
    # Unpack the bytes back to bits, extracting only the relevant bits
    return np.unpackbits(np.frombuffer(packed_code.tobytes(), dtype=np.uint8), bitorder='big')

def get_barker_bitstream(length: int) -> np.uint16:
    """Get the bitstream representation of a Barker code.
    
    Args:
        length: Length of Barker code (7, 11, or 13)
        
    Returns:
        np.ndarray: Bitstream as array of 0s and 1s (only actual code bits)
    """
    packed_code = generate_barker_code(length)
    packed_code = BARKER_CODES.get(length)
    if packed_code is None:
        raise ValueError(f"Invalid Barker code length: {length}")
    
    # Convert 2-byte array to uint16 (big-endian)
    return np.uint16(int.from_bytes(packed_code.tobytes(), byteorder='big'))



if __name__ == "__main__":
    for length in [7, 11, 13]:
        barker_code = generate_barker_code(length)
        bitstream = get_barker_bitstream(length)
        print(f"Barker code (packed) of length {length}:\t{barker_code}")
        print(f"Barker code (bitstream) of length {length}:\t{bitstream}\n")