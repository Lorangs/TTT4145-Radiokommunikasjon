"""
Message Protocol Module.
Defines message structure, creation, parsing, modulation, and demodulation.

Includes Barker code preamble handling and CRC16 checksum verification.

Message Structure:
    [Barker Code (N bits)] [Message Type (1 byte)] [Payload (M bytes)] [Checksum CRC16 (2 bytes)]

Modulation Types Supported:
    - BPSK
    - QPSK

Generated signal of modulated symbols are recieved / transmitted by SDR transciever module.

"""

import numpy as np
from datagram import msgType, Datagram
import yaml
from scipy import signal


class ModulationProtocol:
    def __init__(self, config_file: str ="config.yaml"):
        """Initialize Message Protocol with given configuration."""
        try: 
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            raise e

        # Pre-computed parameters and values
        self.modulation_type = str(config['modulation']['type']).upper().strip()

        self.correlation_threshold = float(config['receiver']['correlation_threshold'])

        self.barker_energy = np.sqrt(np.sum(np.abs(self.barker_bits)**2))


    # ================= Barker detection and message extraction =================
    def detect_barker_sequence(self, received_symbols: np.array) -> int:
        """
        Detect Barker sequence in received bits.
        Args:
            received_symbols (np.array): Array of received symbols.
        Returns:
            int: Index of the start of the Barker sequence, or None if not found.
        """

        correlation = signal.correlate( received_symbols, self.barker_symbols, mode='valid', method='fft')
        correlation_abs = np.abs(correlation)
        peak_index = np.argmax(correlation_abs)
        peak_value = correlation_abs[peak_index]

        # TODO: Adjust threshold method to be relative to precalculated max correlation value
        threshold = self.correlation_threshold * correlation_abs[peak_index]
        if peak_value < threshold:
            return None

        return int(peak_index)
    
    
    # ================= Modulation and Demodulation =================
        
    def modulate_message(self, message: Datagram) -> np.array:
        """Placeholder for modulation function based on modulation type."""
        
        if self.modulation_type == "BPSK":
            return self._bpsk_modulate(message)
        elif self.modulation_type == "QPSK":
            return self._qpsk_modulate(message)
        else:
            raise NotImplementedError(f"Modulation type {self.modulation_type} not implemented.")
        
    def demodulate_message(self, symbols: np.array) -> Datagram:
        """Placeholder for demodulation function based on modulation type."""
        
        if self.modulation_type == "BPSK":
            return self._bpsk_demodulate(symbols)
        elif self.modulation_type == "QPSK":
            return self._qpsk_demodulate(symbols)
        else:
            raise NotImplementedError(f"Demodulation type {self.modulation_type} not implemented.")
        
        
    def _bpsk_modulate(self, message: Datagram) -> np.array:
        """BPSK modulation of the message bytes."""

        # Convert Datagram to bytes and then to bits
        message_bytes = message.pack()
        bits = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))

        # Map bits to BPSK symbols: 0 -> 1, 1 -> -1
        return (2 * bits - 1).astype(np.int8)
    
    def _bpsk_demodulate(self, symbols: np.array) -> Datagram:
        """BPSK demodulation of the symbols to message bytes."""

        # Decision: symbols 
        bits = (symbols > 0).astype(np.int8)  # symbol < 0 -> 1, else 0

        # Pack bits into bytes
        byte_array = np.packbits(bits)

        return Datagram.unpack(byte_array.tobytes())
    

    def _qpsk_modulate(self, message: Datagram) -> np.array:
        """QPSK modulation of the message bytes."""

        # Convert Datagram to bytes and then to bits
        message_bytes = message.pack()
        bits = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))

        # Reshape into pairs of bits
        bit_pairs = bits.reshape(-1, 2)
        
        # Vectorized mapping: [I, Q] components
        # bit_pairs: [[b0, b1], [b2, b3], ...]
        # Map: 00->1+1j, 01->-1+1j, 11->-1-1j, 10->1-1j
        
        I = (1 - 2 * bit_pairs[:, 0]).astype(np.int8)  # First bit: 0->1, 1->-1
        Q = (1 - 2 * bit_pairs[:, 1]).astype(np.int8)  # Second bit: 0->1, 1->-1
        
        return (I + 1j*Q).astype(np.complex64)
    
    def _qpsk_demodulate(self, symbols: np.array) -> Datagram:
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
        return Datagram.unpack(byte_array.tobytes())
    


if __name__ == "__main__":
    from sdr_plots import StaticSDRPlotter
    from matplotlib.pyplot import show

    modulation_protocol = ModulationProtocol("config.yaml")

    plotter = StaticSDRPlotter()


    test_payload = b"He"

    message = Datagram(payload=np.frombuffer(test_payload, dtype=np.uint8), msg_type=msgType.DATA)
    print(f"Created Message: {message}")


    modulated_symbols = modulation_protocol.modulate_message(message)

    # pad to simulate delay
    delay = np.array([1+1j]*50, dtype=np.complex64)
    modulated_symbols = np.concatenate((delay, modulated_symbols))
    print(f"Modulated Symbols: {message}")


    # add AWGN noise as channel simulator
    noise_I = np.random.normal(0, 0.2, modulated_symbols.shape)
    noise_Q = np.random.normal(0, 0.2, modulated_symbols.shape)
    noise = noise_I + 1j*noise_Q
    modulated_symbols = modulated_symbols + noise


    plotter.plot_constellation(modulated_symbols, title="Received Symbols with Noise - Constellation")

    barker_delay = msg_protocol.detect_barker_sequence(modulated_symbols)
    print(f"Barker Delay Index: {barker_delay}")

    stripped_symbols = msg_protocol.strip_barker_preamble(modulated_symbols, barker_delay)
    print(f"Stripped Symbols: {stripped_symbols}")

    demodulated_message = msg_protocol.demodulate_message(stripped_symbols)
    print(f"Demodulated Message: {demodulated_message}")

    parsed_type, parsed_payload = msg_protocol.parse_message(demodulated_message)
    print(f"Parsed Message Type: {parsed_type}, Payload: {parsed_payload}")

    show()






