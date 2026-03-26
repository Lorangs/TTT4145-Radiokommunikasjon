"""
Convolutional Channel encoder with datarate 1/4

ensure only one bit transition per symbol change

"""

import numpy as np
import numba 


class ConvolutionalCoder:
    def __init__(self, config: dict, warmup: bool = True, use_numba: bool = True):
        self.K = int(config['coding']['convolutional_K'])
        self.DATARATE = config['coding']['convolutional_datarate']
        self.GENERATOR = get_generator_matrix(self.K, self.DATARATE)
        self.n = self.GENERATOR.shape[0]  # number of output bits per input bit
        self.use_numba = bool(use_numba)

        if warmup and self.use_numba:
            # Run encode and decode once to trigger numba compilation before first real use.
            dummy_input = np.array([0, 1, 0], dtype=np.uint8)
            self.encode(dummy_input)
            self.decode(self.encode(dummy_input))
        
    def encode(self, input_bits: np.ndarray, ramp_down: bool = True) -> np.ndarray:
        """Encode input bits using a convolutional code with datarate 1/4."""
        if not self.use_numba:
            return _encode.py_func(input_bits, self.GENERATOR, self.K, self.n, ramp_down)
        return _encode(input_bits, self.GENERATOR, self.K, self.n, ramp_down)
    
    def decode(self, received_bits: np.ndarray, ramp_down: bool = True) -> np.ndarray:
        """Decode received bits using the Viterbi algorithm with hard decision."""
        if not self.use_numba:
            return _viterbi_decode_hard.py_func(received_bits, self.GENERATOR, self.K, self.n, ramp_down)
        return _viterbi_decode_hard(received_bits, self.GENERATOR, self.K, self.n, ramp_down)
    


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

@numba.njit(fastmath=True, cache=True)
def _encode(
    input_bits: np.ndarray, 
    G: np.ndarray, 
    k: int, 
    n: int, 
    ramp_down: bool = True) -> np.ndarray:
    """Encode input bits using a convolutional code with datarate 1/4."""

    if ramp_down:
        padded_len = (k - 1) + input_bits.size + (k - 1)
    else:
        padded_len = (k - 1) + input_bits.size
    
    padded_msg = np.zeros(padded_len, dtype=np.uint8)
    padded_msg[(k-1):(k-1)+input_bits.size] = input_bits

    num_symbols = padded_msg.size - (k-1)
    shift_register = np.zeros(k, dtype=np.uint8)
    output_bits = np.zeros(num_symbols * n, dtype=np.uint8)

    for i in range(num_symbols):
        shift_register = padded_msg[i:i+k]  # Shift in the next bit

        # Manual dot product to avoid ambiguity
        for j in range(n):
            output_bits[i*n + j] = np.uint8(np.sum(shift_register * G[j, :]) % 2)

    return output_bits

@numba.njit(fastmath=True, cache=True)
def _viterbi_decode_hard(
        received_bits: np.ndarray,
        G: np.ndarray,
        k: int,
        n: int,
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

                expected_output = np.zeros(n, dtype=np.uint8)
                for j in range(n):
                    expected_output[j] = np.uint8(np.sum(shift_register * G[j, :]) % 2)

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
    coder = ConvolutionalCoder(config={
        'coding': {
            'convolutional_K': 7,
            'convolutional_datarate': "1/4"
        }
    })

    test_string = "H"
    test_bytes = np.frombuffer(test_string.encode('utf-8'),dtype=np.uint8) 
    test_bits = np.unpackbits(test_bytes)
    
    print("Input bits:")
    print(test_bits)
    print()

    encoded_bits = coder.encode(test_bits)
    print("Encoded bits:")
    print(encoded_bits)
    print()

    decoded_bits = coder.decode(encoded_bits)
    print("Decoded bits:")
    print(decoded_bits)
    print()
