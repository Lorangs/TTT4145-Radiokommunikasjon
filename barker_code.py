"""
Barker code generator module.
Provides functionality to generate Barker codes of specified lengths.
All barker codes are predefined and stored in a dictionary for efficient retrieval. 
The codes are packed into bytes for transmission and can be unpacked back to bits when needed.
All Barker codes are padded to 16 bits (2 bytes) for uniformity in storage and transmission, with unused bits set to zero.
"""

from numpy import uint16, array, int8

BARKER_BITS = {
    7: uint16(0b1110001000000000),   # Barker code of length 7, padded to 16 bits
    11: uint16(0b1110000100100000),  # Barker code of length 11, padded to 16 bits
    13: uint16(0b1111100110101000)   # Barker code of length 13, padded to 16 bits
}

BARKER_SYMBOLS = {
    7: array([1, 1, 1, -1, -1, 1, -1], dtype=int8),
    11: array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1], dtype=int8),
    13: array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=int8) 
}



if __name__ == "__main__":
    for length in [7, 11, 13]:
        bit_sequence = BARKER_BITS.get(length)
        symbols = BARKER_SYMBOLS.get(length)
        print(f"Barker code length: {length}")
        print(f"Bit sequence (hex): {bit_sequence:04x}")
        print(f"Symbols: {symbols}\n")