"""
Convolutional Channel encoder with datarate 1/4

ensure only one bit transition per symbol change

"""

import numpy as np

k = 3   # Constraint length (memory elements in the shift register)
n = 4   # number of output bits per input bit

def generate_generators(k: int, n: int) -> np.ndarray:
    """
    Generate an (n, k) uint8 matrix of Gray-coded rows.
    Each row has k columns (bits). 
    """
    max_rows = (1 << k) - 1
    print(max_rows)
    if n > max_rows:
        raise ValueError(
            f"n={n} is too large for k={k}. Max unique rows is {max_rows}."
        )

    generators = np.zeros((n, k), dtype=np.uint8)

    start = 1 # Start from 1 to avoid the all-zero row
    for i in range(n):
        x = i + start
        gray = x ^ (x >> 1)  # Gray encoding
        # Convert to k bits (MSB -> LSB)
        bits = np.array([(gray >> b) & 1 for b in range(k - 1, -1, -1)], dtype=np.uint8)
        generators[i] = bits

    return generators

def encoder(data: np.ndarray, generator: np.ndarray) -> np.ndarray:
    padded_data = np.concatenate((data, np.zeros(k - 1, dtype=np.uint8)))
    n_bits = len(padded_data) - (k - 1)

    shift_register = np.zeros(k - 1, dtype=np.uint8)

    encoded_bits = np.zeros(len(padded_data) * n, dtype=np.uint8)

    for i, u in enumerate(padded_data):
        reg = np.concatenate(([u], shift_register)) # [current_input, memory ...]
        encoded_bits[i * n : (i + 1) * n] = (generator @ reg) % 2 # Matrix multiplication mod 2
        shift_register = reg[:-1] # Update shift register (drop the last bit)

    return encoded_bits

def viterbi_decoder_soft(received: np.ndarray, generator: np.ndarray) -> np.ndarray:
    # Placeholder for soft-decision Viterbi decoder implementation
    # This function should compute the most likely input sequence given the received bits and the generator matrix
    # For now, it just returns a dummy array of zeros with the same length as the input data
    return np.zeros(len(received) // n, dtype=np.uint8)





if __name__ == "__main__":

    generator = generate_generators(k, n)
    print("Generator polynomials:")
    print(generator)
    print()

    test_string = "H"
    test_bytes = np.frombuffer(test_string.encode('utf-8'),dtype=np.uint8) 
    test_bits = np.unpackbits(test_bytes)
    #test_bits = np.array([1,0,0,0,0,0,0,0], dtype=np.uint8)
    print("Input bits:")
    print(test_bits)
    print()

    print("Encoded bits:")
    code_word = encoder(test_bits, generator)
    print(code_word)

    decoded = viterbi_decode_soft(code_word, generator)

    print("Decoded bits:")
    print(decoded)
