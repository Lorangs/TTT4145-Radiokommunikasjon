"""
Convolutional Channel encoder with datarate 1/4

ensure only one bit transition per symbol change

"""

import numpy as np
import numpy.typing as npt
from numba import njit
from project_logger import get_logger
logger = get_logger(__name__)

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

class ConvolutionalCoder:
    def __init__(self, config: dict, warmup: bool = True, use_numba: bool = True):
        self.K = int(config['coding']['convolutional_K'])
        self.DATARATE = config['coding']['convolutional_datarate']
        self.GENERATOR = get_generator_matrix(self.K, self.DATARATE)
        self.n = self.GENERATOR.shape[0]  # number of output bits per input bit
        self.use_numba = bool(use_numba)
        self.expected_bit_length = (
            int(config["coding"].get("rs_added_bytes", 16)) +
            int(config["datagram"]["total_size"]) 
        ) * 8
        if warmup:
            # Run encode and decode once to trigger numba compilation before first real use.
            dummy_input = np.zeros((self.expected_bit_length + self.K - 1) // 8, dtype=np.uint8) 
            encoded_dummy = self.encode(dummy_input)
            self.decode(encoded_dummy)
        
    def encode(self, input_bits: np.ndarray) -> np.ndarray:
        """Encode input bits using a convolutional code with datarate 1/4."""
        return _encode_bytes(input_bits, self.GENERATOR, self.K, self.n)
    
    def decode(self, received_bits: np.ndarray) -> np.ndarray:
        """Decode received bits using the Viterbi algorithm with hard decision."""
        return _viterbi_decode_hard(received_bits, self.GENERATOR, self.K, self.n, self.expected_bit_length)

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

@njit(cache=True, fastmath=True)
def _encode_bytes(
    input_bytes: npt.NDArray[np.uint8], 
    G: np.ndarray, 
    k: int, 
    n: int
) -> npt.NDArray[np.uint8]:
    assert n <= 8, "This byte-oriented encoding function only supports up to 8 output bits per input bit (datarate >= 1/8)."
    
    data = np.asarray(input_bytes, dtype=np.uint8).reshape(-1)  # Ensure 1D array
    in_bits = _unpackbits_little(data)  # Unpack to bits, LSB first

    n_input = in_bits.size
    n_steps = n_input + (k-1)  # Total steps including tail bits for ramp down
    out_bits = np.zeros(n_steps * n, dtype=np.uint8)
    shift_register = np.zeros(k, dtype=np.uint8)
  
    for i in range(n_input):
        shift_register[:-1] = shift_register[1:]  # Shift right
        shift_register[-1] = in_bits[i]  # Input bit enters at the end

        base = i * n  
        for j in range(n):
            out_bits[base + j] = np.uint8(np.sum(shift_register * G[j, :]) & 0x1)  # Modulo 2 sum for output bit j

    # Ramp down with tail bits
    for r in range(k-1):
        shift_register[:-1] = shift_register[1:]  # Shift right
        shift_register[-1] = 0  # Tail bits are zero

        base = (n_input + r) * n  
        for j in range(n):
            out_bits[base + j] = np.uint8(np.sum(shift_register * G[j, :]) & 0x1)

    return _packbits_little(out_bits)  # Pack back to bytes, LSB first


@njit(fastmath=True, cache=True)
def _viterbi_decode_hard(
        received_bytes: np.ndarray,
        G: np.ndarray,
        k: int,
        n: int,
        expected_bit_length: int
    ) -> np.ndarray:
    """
    Hard-decision Viterbi decoder for packed-byte input.
    Input: packed encoded bytes (little-endian bit order)
    Output: packed decoded bytes (little-endian bit order)
    """

    data = np.asarray(received_bytes, dtype=np.uint8).reshape(-1)  # Ensure 1D array
    received_bits = _unpackbits_little(data)  # Unpack to bits, LSB first
    expected_coded_bits = (expected_bit_length + (k-1)) * n  # Total bits including tail bits
    
    if received_bits.size < expected_coded_bits:
        raise ValueError(f"Received bits ({received_bits.size}) are fewer than expected coded bits ({expected_coded_bits}).")
    
    received_bits = received_bits[:expected_coded_bits]  # Truncate to expected length if longer than expected

    msg_length = received_bits.size // n
    num_states = 1 << (k-1)

    path_history = np.zeros((msg_length, num_states), dtype=np.uint32)
    decided_bits = np.zeros((msg_length, num_states), dtype=np.uint8)

    path_metrics = np.full(num_states, np.inf)    
    path_metrics[0] = 0  # Start from the all-zero state

    # process each received symbol
    for i in range(msg_length):
        rx = received_bits[i*n : (i+1)*n]
        new_metrics = np.full(num_states, np.inf)

        # for each state, compute the possible transitions
        for state in range(num_states):
            prev_metric = path_metrics[state]
            if np.isinf(path_metrics[state]):
                continue  # Skip unreachable states

            # try both possible input bits (0 and 1)
            for input_bit in (0, 1):
                next_state = ((state << 1) | input_bit) & (num_states - 1)  

                shift_register = np.zeros(k, dtype=np.uint8)
                for j in range(k-1):
                    shift_register[j] = np.uint8((state >> (k-2-j)) & 0x1)
                shift_register[-1] = np.uint8(input_bit)

                expected_output = np.zeros(n, dtype=np.uint8)
                for j in range(n):
                    expected_output[j] = np.uint8(np.sum(shift_register * G[j, :]) & 0x1)

                hamming_distance = np.sum(expected_output != rx)
                metric = prev_metric + hamming_distance
    
                if metric < new_metrics[next_state]:
                    new_metrics[next_state] = metric
                    path_history[i, next_state] = state 
                    decided_bits[i, next_state] = input_bit

        path_metrics = new_metrics
    
    best_final_state = 0 

    decoded_bits = np.zeros(msg_length, dtype=np.uint8)
    for i in range(msg_length-1, -1, -1):
        decoded_bits[i] = decided_bits[i, best_final_state]
        best_final_state = path_history[i, best_final_state]

    # Remove tail bits corresponding to ramp down
    if len(decoded_bits) > expected_bit_length:
        trimmed_output = decoded_bits[:expected_bit_length]
    else:        
        trimmed_output = decoded_bits

    return _packbits_little(trimmed_output)  # Pack back to bytes, LSB first
  

@njit(cache=True)
def _unpackbits_little(data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Helper function to unpack bytes into bits with little-endian bit order (LSB first).
    Can be swapped for np.unpackbits(data, bitorder='little') if not using numba.
    """
    out = np.zeros(data.size * 8, dtype=np.uint8)
    for i in range(data.size):
        v = int(data[i])
        base = i * 8
        for b in range(8):
            out[base + b] = np.uint8((v >> b) & 0x1)
    return out

@njit(cache=True)
def _packbits_little(bits: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Helper function to pack bits into bytes with little-endian bit order (LSB first).
    Can be swapped for np.packbits(bits, bitorder='little') if not using numba.
    """
    n_bytes = (bits.size + 7) // 8
    out = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(bits.size):
        if bits[i] & 0x1:
            out[i // 8] = np.uint8(out[i // 8] | np.uint8(1 << (i % 8)))
    return out


if __name__ == "__main__":

    from datagram import Datagram
    from forward_error_correction import FCCodec
    fcc = FCCodec(config={
        'coding': {
            'rs_num_ecc': 8
        }    })

    coder = ConvolutionalCoder(
        config={
            'coding': {
                'convolutional_K': 7,
                'convolutional_datarate': "1/2",
                'rs_num_ecc': 8,
                'rs_added_bytes': 8
            },
            'datagram': {
                'total_size': 1
            }
        },
        warmup=True,
        use_numba=True
    )
    test_string = "H"
    test_bytes = np.frombuffer(test_string.encode(), dtype=np.uint8)
    #test_string = "Hello World!"
    #dgram = Datagram.as_string(text=test_string, msg_id=1)
    #test_bytes = dgram.pack()

    print(len(test_bytes))
    fcc_bytes = fcc.encode(test_bytes)
    print(len(fcc_bytes))
    
    print("Input bytes:")
    print(fcc_bytes)
    print() 
    print("Fcc encoded bytes:")
    print(fcc_bytes)
    print()

    encoded_bytes = coder.encode(fcc_bytes)
    print("Encoded bits:")
    print(encoded_bytes)
    print()

    # add some noise/errors for testing
    noisy_bytes = np.copy(encoded_bytes)
    noisy_bytes[0] ^= 0xFF  # Flip bits in the first byte
    # add byte in middle
    noisy_bytes = np.insert(noisy_bytes, len(noisy_bytes) // 2, 0xFF)  # Insert a byte of all 1s in the middle
    # add byte at end    noisy_bytes[-1] 

    decoded_bytes = coder.decode(encoded_bytes)
    print("Decoded bytes:")
    print(decoded_bytes)
    print()

    fcc_decoded_bytes = fcc.decode(decoded_bytes)

    print("Decoded bytes:")
    print(fcc_decoded_bytes)
    print() 

    reassembled_dgram = Datagram.unpack(fcc_decoded_bytes)
    print("Decoded string:")
    print(reassembled_dgram.get_payload_as_string)
