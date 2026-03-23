"""
Convolutional Channel encoder with datarate 1/4

ensure only one bit transition per symbol change

"""

import numpy as np

K = 7  # Constraint length (number of memory elements in the shift register)
DATARATE = "1/4"

# Known good generator polynomials (octal)
GENERATOR_TABLE = {
    3: {
        "1/2": [0o7, 0o5],
        "1/3": [0o7, 0o5, 0o3],
        "1/4": [0o7, 0o5, 0o6, 0o3],
    },
    5: {
        "1/2": [0o23, 0o35],
        "1/3": [0o23, 0o35, 0o33],
        "1/4": [0o23, 0o35, 0o33, 0o37],
    },
    7: {
        "1/2": [0o171, 0o133],  # IEEE 802.11
        "1/3": [0o171, 0o133, 0o165],
        "1/4": [0o171, 0o133, 0o165, 0o117],
    }
}
def octal_to_binary_array(octal_val, K):
    """Convert octal polynomial to binary tap vector"""
    binary_str = bin(octal_val)[2:]  # remove '0b'
    
    # pad to length K
    binary_str = binary_str.zfill(K)
    
    return np.array([int(b) for b in binary_str], dtype=np.uint8)

def get_generator_matrix(K, rate="1/2") -> np.ndarray:
    """Get the generator matrix for the specified constraint length and rate."""
    if K not in GENERATOR_TABLE:
        raise ValueError(f"No generator polynomials defined for K={K}.")
    if rate not in GENERATOR_TABLE[K]:
        raise ValueError(f"No generator polynomials defined for rate {rate} with K={K}.")
    
    octals = GENERATOR_TABLE[K][rate]
    return np.array(
        [octal_to_binary_array(oct, K) for oct in octals], 
        dtype=np.uint8
    )

GENERATOR = get_generator_matrix(K, DATARATE)
n = GENERATOR.shape[0]  # number of output bits per input bit
k = K  # number of bits in the shift register (including the input bit)


def encode(input_bits: np.ndarray, ramp_down: bool = True) -> np.ndarray:
    """Encode input bits using a convolutional code with datarate 1/4."""

    # Pad the input bits with zeros to flush the shift register at the end of the message.
    if ramp_down:
        padded_msg = np.concatenate([np.zeros(k-1, dtype=np.uint8), input_bits, np.zeros(k-1, dtype=np.uint8)])
    else:
        padded_msg = np.concatenate([np.zeros(k-1, dtype=np.uint8), input_bits])

    num_symbols = padded_msg.size - (k-1)
    shift_register = np.zeros(k, dtype=np.uint8)
    output_bits = np.zeros(num_symbols * n, dtype=np.uint8)

    for i in range(num_symbols):
        shift_register = padded_msg[i:i+k]  # Shift in the next bit
        output_bits[i*n : (i+1)*n] = np.mod(np.dot(shift_register, GENERATOR.T), 2)

    return output_bits

def viterbi_decode_hard(
        received_bits: np.ndarray,
        ramp_down: bool = True) -> np.ndarray:
    """Decode received bits using the Viterbi algorithm with hard decision."""
    if received_bits.size % n != 0:
        raise ValueError("Received bits length must be a multiple of n.")

    msg_length = received_bits.size // n
    num_states = 2 ** (k-1)
    path_history = np.zeros((msg_length, num_states), dtype=np.uint8)
    decided_bits = np.zeros((msg_length, num_states), dtype=np.uint8)

    path_metrics = np.full(num_states, np.inf)    
    path_metrics[0] = 0  # Start from the all-zero state

    # process each received symbol
    for i in range(msg_length):
        rx = received_bits[i*n : (i+1)*n]
        new_path_metrics = np.full(num_states, np.inf)

        # for each state, compute the possible transitions
        for state in range(num_states):
            if np.isinf(path_metrics[state]):
                continue  # Skip unreachable states

            # try both possible input bits (0 and 1)
            for input_bit in (0, 1):
                next_state = ((state << 1) | input_bit) & ((1 << (k-1)) - 1)

                shift_register = np.array(
                    [(state >> (k-2-j)) & 1 for j in range(k-1)] + [input_bit], 
                    dtype=np.uint8
                )

                expected_output = np.mod(np.dot(GENERATOR, shift_register), 2)
                hamming_distance = np.sum(expected_output != rx)

                # update path metric if this transition is better
                new_path_metric = path_metrics[state] + hamming_distance
                if new_path_metric < new_path_metrics[next_state]:
                    new_path_metrics[next_state] = new_path_metric
                    path_history[i, next_state] = state 
                    decided_bits[i, next_state] = input_bit

        path_metrics = new_path_metrics
    
    # Traceback to find the most likely input bit sequence.
    # If ram_down, tail padded, the encoder should end in zero state
    best_final_state = 0 if ramp_down else int(np.argmin(path_metrics))

    decoded_bits = np.zeros(msg_length, dtype=np.uint8)
    for i in range(msg_length-1, -1, -1):
        decoded_bits[i] = decided_bits[i, best_final_state]
        best_final_state = path_history[i, best_final_state]

    if ramp_down:
        # Remove the tail bits that were added for ramp down
        return decoded_bits[:msg_length-(k-1)]
    return decoded_bits


if __name__ == "__main__":

    test_string = "H"
    test_bytes = np.frombuffer(test_string.encode('utf-8'),dtype=np.uint8) 
    test_bits = np.unpackbits(test_bytes)
    #test_bits = np.array([1,0,0,0,0,0,0,0], dtype=np.uint8)
    #test_bits = np.array([1, 1, 1], dtype=np.uint8)
    print("Input bits:")
    print(test_bits)
    print()

    print("Generator matrix:")
    print(GENERATOR)
    print()
    
    print("Encoded bits:")
    code_word = encode(test_bits, ramp_down=True)
    print(code_word)

    decoded = viterbi_decode_hard(code_word)

    print("Decoded bits:")
    print(decoded)
