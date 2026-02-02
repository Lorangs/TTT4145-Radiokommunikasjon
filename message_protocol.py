"""
Message Protocol for SDR Chat Application
Handles message framing, encoding, decoding, and acknowledgments
"""

import numpy as np
from enum import Enum
import struct
import yaml
from barker_code import generate_barker_code
from scipy import signal


class MessageType(Enum):
    """Message types for the chat protocol"""
    TEXT = 0x01             # Text message
    ACK = 0x02              # Delivered acknowledgment

class MessageProtocol:
    def __init__(self, config_file: str ="config.yaml"):
        """Initialize Message Protocol with given configuration."""
        try: 
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            raise e

        self.modulateion_type = str(config['modulation']['type']).upper().strip()
        self.barker_length = int(config['preamble']['barker_code_length']) # in bits
        self.barker_bits = generate_barker_code(self.barker_length)
        self.payload_size = int(config['framing']['payload_size'])         # in bytes

        # Generate reference Barker sequence (modulated)
        if self.modulateion_type == "BPSK":
            self.barker_symbols = (2 * self.barker_bits - 1).astype(np.int8)

        elif self.modulateion_type == "QPSK":
            barker_b = self.barker_bits.copy()

            if len(barker_b) % 2 != 0:
                barker_b = np.append(barker_b, 0)

            bit_pairs = barker_b.reshape(-1, 2)
            I = (1 - 2 * bit_pairs[:, 0]).astype(np.int8)
            Q = (1 - 2 * bit_pairs[:, 1]).astype(np.int8)
            self.barker_symbols = (I + 1j*Q).astype(np.complex64)

        else:
            raise NotImplementedError(f"Modulation type {self.modulateion_type} not supported")

        # Store expected number of symbols in a message
        expected_num_bits = 8 * (1 + self.payload_size + 2)  # 1 byte type + payload + 2 byte checksum


        if self.modulateion_type == "BPSK":
            self.expected_num_symbols = expected_num_bits * 1  # 1 symbol per bit
        elif self.modulateion_type == "QPSK":
            self.expected_num_symbols = (expected_num_bits + 1) // 2 # 2 bits per symbol

    def create_message(
            self,
            msg_type: MessageType,
            payload: bytes,
        ) -> bytes:
        """Create a message with header and checksum."""

        # Header: [Barker Code (N bytes)] [Message Type (1 byte)] [Payload (M bytes)] [Checksum CRC16 (2 bytes))]
        if len(payload) > self.payload_size:
            raise ValueError(f"Payload size exceeds maximum of {self.payload_size} bytes")
        
        _msg_type = struct.pack("B", msg_type.value)
        padded_payload = payload.ljust(self.payload_size, b'\0')
        checksum = struct.pack("H", self._crc16(_msg_type + padded_payload))
        message = _msg_type + padded_payload + checksum
        return message
    
    def parse_message(self, message: bytes) -> tuple[MessageType, bytes]:
        """Parse a received message and verify checksum.
        
        Args:
            message (bytes): Received message bytes.
        Returns:
            (MessageType, bytes): Tuple of message type and payload bytes.
        """
        if len(message) < 1 + self.payload_size + 2:
            raise ValueError("Message too short to parse")

        msg_type_value = struct.unpack("B", message[0:1])[0]
        payload = message[1:1 + self.payload_size]
        received_checksum = struct.unpack("H", message[1 + self.payload_size:1 + self.payload_size + 2])[0]

        computed_checksum = self._crc16(message[0:1 + self.payload_size])
        if received_checksum != computed_checksum:
            raise ValueError("Checksum mismatch")

        try:
            msg_type = MessageType(msg_type_value)
        except ValueError:
            raise ValueError("Unknown message type")

        return msg_type, payload.rstrip(b'\0')  # Remove padding null bytes


    # ================= Barker detection and message extraction =================
    def detect_barker_sequence(self, received_symbols: np.array) -> int:
        """
        Detect Barker sequence in received bits.
        Args:
            received_symbols (np.array): Array of received symbols.
        Returns:
            int: Index of the start of the Barker sequence, or None if not found.
        """

        correlation = np.abs( signal.correlate( received_symbols, self.barker_symbols, mode='valid', method='fft'))
        peak_index = np.argmax(correlation)
        peak_value = correlation[peak_index]

        # check threshold
        threshold = 0.8 * np.max(correlation)
        if peak_value < threshold:
            return None

        return int(peak_index)
    

    def strip_barker_preamble(self, recieved_symbols: np.array, barker_delay: int) -> np.array:
        """
        Strip Barker preamble from received symbols.
        Args:
            recieved_symbols (np.array): Array of received symbols.
            barker_delay (int): Index where Barker sequence ends.
        Returns:
            np.array: Symbols after stripping Barker preamble.
        """
        start_index = barker_delay + len(self.barker_symbols)
        end_index = start_index + self.expected_num_symbols
        
        # check if we have enough symbols for full message. Could raise error or return None
        if len(recieved_symbols) < end_index:
            raise ValueError("Not enough symbols received for full message")
        
        return recieved_symbols[start_index : end_index]
        



    # ================= CRC16 Checksum =================

    def _crc16(self, data: bytes) -> int:
        """Compute CRC16 checksum."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if (crc & 0x0001):
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc
    
    # ================= Modulation and Demodulation =================
        
    def modulate_message(self, message: bytes) -> np.array:
        """Placeholder for modulation function based on modulation type."""
        
        if self.modulateion_type == "BPSK":
            return self._bpsk_modulate(message)
        elif self.modulateion_type == "QPSK":
            return self._qpsk_modulate(message)
        else:
            raise NotImplementedError(f"Modulation type {self.modulateion_type} not implemented.")
        
    def demodulate_message(self, symbols: np.array) -> bytes:
        """Placeholder for demodulation function based on modulation type."""
        
        if self.modulateion_type == "BPSK":
            return self._bpsk_demodulate(symbols)
        elif self.modulateion_type == "QPSK":
            return self._qpsk_demodulate(symbols)
        else:
            raise NotImplementedError(f"Demodulation type {self.modulateion_type} not implemented.")
        
        
    def _bpsk_modulate(self, message: bytes) -> np.array:
        """BPSK modulation of the message bytes."""

        # Convert bytes to bits
        bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))

        # Prepend barker code
        bits = np.concatenate((self.barker_bits, bits))

        # Map bits to BPSK symbols: 0 -> 1, 1 -> -1
        return (2 * bits - 1).astype(np.int8)
    
    def _bpsk_demodulate(self, symbols: np.array) -> bytes:
        """BPSK demodulation of the symbols to message bytes."""

        # Decision: symbols 
        bits = (symbols > 0).astype(np.int8)  # symbol < 0 -> 1, else 0

        # Pack bits into bytes
        byte_array = np.packbits(bits)

        return byte_array.tobytes()
        
    def _qpsk_modulate(self, message: bytes) -> np.array:
        """QPSK modulation of the message bytes."""

        # Convert bytes to bits
        bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))

        # Prepend barker code
        bits = np.concatenate((self.barker_bits, bits))

        # Reshape into pairs of bits
        bit_pairs = bits.reshape(-1, 2)
        
        # Vectorized mapping: [I, Q] components
        # bit_pairs: [[b0, b1], [b2, b3], ...]
        # Map: 00->1+1j, 01->-1+1j, 11->-1-1j, 10->1-1j
        
        I = (1 - 2 * bit_pairs[:, 0]).astype(np.int8)  # First bit: 0->1, 1->-1
        Q = (1 - 2 * bit_pairs[:, 1]).astype(np.int8)  # Second bit: 0->1, 1->-1
        
        return (I + 1j*Q).astype(np.complex64)
    
    def _qpsk_demodulate(self, symbols: np.array) -> bytes:
        """QPSK demodulation of the symbols to message bytes."""

        # Decision boundaries for I and Q
        I_bits = (symbols.real < 0 ).astype(np.int8)  # I < 0 -> 1, else 0
        Q_bits = (symbols.imag < 0 ).astype(np.int8)  # Q < 0 -> 1, else 0

        # Interleave bits back into original order
        bits = np.empty(I_bits.size + Q_bits.size, dtype=np.int8)
        bits[0::2] = I_bits
        bits[1::2] = Q_bits

        # Pack bits into bytes
        byte_array = np.packbits(bits)

        return byte_array.tobytes()
    


if __name__ == "__main__":
    msg_protocol = MessageProtocol()

    test_payload = b"He"

    message = msg_protocol.create_message(MessageType.TEXT, test_payload)
    print(f"Created Message: {message}")

    modulated_symbols = msg_protocol.modulate_message(message)
    print(f"Modulated Symbols: {modulated_symbols}")


    # shift symbols to simulate delay
    modulated_symbols = np.concatenate((np.zeros(10, dtype=modulated_symbols.dtype), modulated_symbols))

    # add AWGN noise as channel simulator
    noise = np.random.normal(0, 0.1, len(modulated_symbols)).astype(modulated_symbols.dtype)
    modulated_symbols = modulated_symbols + noise
    
    barker_delay = msg_protocol.detect_barker_sequence(modulated_symbols)
    print(f"Barker Delay Index: {barker_delay}")

    stripped_symbols = msg_protocol.strip_barker_preamble(modulated_symbols, barker_delay)
    print(f"Stripped Symbols: {stripped_symbols}")

    demodulated_message = msg_protocol.demodulate_message(stripped_symbols)
    print(f"Demodulated Message: {demodulated_message}")






