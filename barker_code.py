"""
Barker code generator module.
Provides functionality to generate Barker codes of specified lengths.
"""

from numpy import array, int8

# Statically defined Barker codes as numpy arrays for faster runtime
BARKER_CODES = {
    2: array([1, 0], dtype=int8),
    3: array([1, 1, 0, 0], dtype=int8),                                 # Padded to 4 bits
    4: array([1, 1, 0, 1], dtype=int8),     
    5: array([1, 1, 1, 0, 1, 0], dtype=int8),                           # Padded to 6 bits
    7: array([1, 1, 1, 0, 0, 1, 0, 0], dtype=int8),                     # Padded to 8 bits
    11: array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], dtype=int8),        # Padded to 12 bits
    13: array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0], dtype=int8)   # Padded to 14 bits
}

def generate_barker_code(length):
    """Generate Bipolar Barker code of specified length.
    Args:
        length (int): Length of the desired Barker code. Range 2 - 13.

    Returns:
        array: Barker code as a numpy array of bits , or None if length is invalid.
    """
    return BARKER_CODES.get(length, None)

