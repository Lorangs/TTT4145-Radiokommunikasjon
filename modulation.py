from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit
from commpy import modulation as commpy_modulation
from project_logger import get_logger
logger = get_logger(__name__)


def normalize_modulation_name(modulation_type: str, modulation_order: int) -> str:
    kind = str(modulation_type).upper().strip()
    order = int(modulation_order)

    if kind in {"BPSK", "QPSK"}:
        return kind

    if kind == "PSK":
        if order == 2:
            return "BPSK"
        if order == 4:
            return "QPSK"
        raise ValueError(f"Unsupported PSK order for framing and synchronization: {order}")

    if kind == "QAM":
        return f"{order}QAM"

    raise ValueError(f"Unsupported modulation type: {modulation_type}")


def normalize_config_modulation_name(config: dict) -> str:
    modulation = config["modulation"]
    return normalize_modulation_name(modulation["type"], modulation["order"])



@njit(cache=True, fastmath=True)
def bytes_to_symbols(bytes: npt.NDArray[np.uint8], modulation_name: str) -> npt.NDArray[np.complex64]:
    """Map bytes to complex symbols according to the specified modulation scheme."""
    mod_type = modulation_name.upper().strip()

    if mod_type == "BPSK":
        out = np.empty(bytes.size * 8, dtype=np.complex64)  # 1 bit per symbol
        for i in range(bytes.size):
            val = int(bytes[i])
            base = i * 8
            
            for bit in range(8):
                b = (val >> bit) & 0x1
                out[base + bit] = np.complex64(1.0 - 2.0 * b)  # Map 0 to +1 and 1 to -1
        
    elif mod_type == "QPSK":
        out = np.empty(bytes.size * 4, dtype=np.complex64)  # 2 bits per symbol
        for i in range(bytes.size):
            val = int(bytes[i])
            base = i * 4
            for bit_pair in range(4):
                bits = (val >> (bit_pair * 2)) & 0x3
                i_val = 1.0 - 2.0 * int((bits >> 1) & 0x1)
                q_val = 1.0 - 2.0 * int((bits & 0x1))
                out[base + bit_pair] = np.complex64(i_val + 1j * q_val)

    else:
        raise ValueError(f"Unsupported modulation type: {modulation_name}")
    return out

@njit(cache=True, fastmath=False)
def symbols_to_bytes(symbols: npt.NDArray[np.complex64], modulation_name: str) -> npt.NDArray[np.uint8]:
    mod_type = modulation_name.upper().strip()

    if mod_type == "BPSK":
        length = symbols.size // 8
        remainder = symbols.size % 8
        out = np.empty(length + (1 if remainder > 0 else 0), dtype=np.uint8)

        for i in range(length):
            byte_val = 0
            base = i * 8
            for bit in range(8):
                byte_val |= (0 if symbols[base + bit].real >= 0 else 1) << bit
            out[i] = byte_val

        if remainder > 0:
            byte_val = 0
            for bit in range(remainder):
                byte_val |= (0 if symbols[length * 8 + bit].real >= 0 else 1) << bit
            out[length] = np.uint8(byte_val)

    elif mod_type == "QPSK":
        length = symbols.size // 4
        remainder = symbols.size % 4
        out = np.empty(length + (1 if remainder > 0 else 0), dtype=np.uint8)

        for i in range(length):
            byte_val = 0
            base = i * 4
            for bit_pair in range(4):
                sym = symbols[base + bit_pair]
                bits = ((0 if sym.real >= 0 else 1) << 1) | (0 if sym.imag >= 0 else 1)
                byte_val |= bits << (bit_pair * 2)
            out[i] = byte_val

        if remainder > 0:
            byte_val = 0
            for bit_pair in range(remainder):
                sym = symbols[length * 4 + bit_pair]
                bits = ((0 if sym.real >= 0 else 1) << 1) | (0 if sym.imag >= 0 else 1)
                byte_val |= bits << (bit_pair * 2)
            out[length] = np.uint8(byte_val)

    else:
        raise ValueError(f"Unsupported modulation type: {modulation_name}")
    return out


def nearest_constellation_symbols(symbols: npt.NDArray[np.complex64], modulation_name: str) -> npt.NDArray[np.complex64]:
    mod_type = modulation_name.upper().strip()
    if mod_type == "BPSK":
        return np.where(symbols.real >= 0, 1.0, -1.0).astype(np.complex64)

    if mod_type == "QPSK":
        return (
            np.where(symbols.real >= 0, 1.0, -1.0)
            + 1j * np.where(symbols.imag >= 0, 1.0, -1.0)
        ).astype(np.complex64)

    raise ValueError(f"Unsupported modulation type: {modulation_name}")


def modulation_rotations(modulation_name: str) -> tuple[complex, ...]:
    mod_type = modulation_name.upper().strip()
    if mod_type == "BPSK":
        return (1 + 0j, -1 + 0j)
    elif mod_type == "QPSK":
        return (1 + 0j, -0 + 1j, -1 - 0j, 0 - 1j)
    else:
        raise ValueError(f"Unsupported modulation type: {modulation_name}")
    


class ModulationProtocol:
    def __init__(self, config: dict):
        modulation_config = config["modulation"]
        
        raw_type = str(modulation_config["type"]).upper().strip()
        self.modulation_order = int(modulation_config["order"])
        self.modulation_type = normalize_modulation_name(raw_type, self.modulation_order)
        self.samples_per_symbol = int(config["modulation"]["samples_per_symbol"])
        
        if self.modulation_type in {"BPSK", "QPSK"}:
            self.modulator = None
        elif raw_type == "QAM":
            self.modulator = commpy_modulation.QAMModem(self.modulation_order)
        else:
            raise ValueError(f"Unsupported modulation type: {raw_type}")
        
        # warm up numba by running a dummy modulation and demodulation
        dummy_bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        dummy_symbols = self.modulate_message(dummy_bits)
        self.demodulate_signal(dummy_symbols)


    def modulate_message(self, byte_stream: npt.NDArray[np.uint8]) -> npt.NDArray[np.complex64]:
        """Modulate a bit stream into a complex baseband signal."""

        if self.modulation_type in {"BPSK", "QPSK"}:
            return bytes_to_symbols(byte_stream, self.modulation_type)

        if self.modulator is not None:
            return self.modulator.modulate(byte_stream)
        raise ValueError(f"Unsupported modulation type: {self.modulation_type}")

    def demodulate_signal(self, signal: np.ndarray) -> np.ndarray:
        """Demodulate a complex baseband signal back into a bit stream."""
        signal = np.asarray(signal).astype(np.complex64, copy=False)

        if self.modulation_type in {"BPSK", "QPSK"}:
            return symbols_to_bytes(signal, self.modulation_type)

        if self.modulator is not None:
            return self.modulator.demodulate(signal, demod_type="hard")

        raise ValueError(f"Unsupported modulation type: {self.modulation_type}")



if __name__ == "__main__":
    from sdr_plots import StaticSDRPlotter
    from datagram import Datagram, msgType


    plotter = StaticSDRPlotter()

    modulation_config = {
        "modulation": {
            "type": "BPSK",
            "order": 1,
            "samples_per_symbol": 8
        }
    }

    protocol = ModulationProtocol(modulation_config)
    test_gram = Datagram.as_string(msg_id=1, msg_type=msgType.DATA, text="Hello, World!")
    test_bits = test_gram.pack()
    modulated_signal = protocol.modulate_message(test_bits)

    print("Modulated signal:")
    print(modulated_signal)

    plotter.plot_constellation(modulated_signal, title="Modulated Signal Constellation")

    demodulated_bits = protocol.demodulate_signal(modulated_signal)

    if np.all(test_bits == demodulated_bits):
        print("Demodulation successful, bits match!")
    else:
        print("Demodulation failed, bits do not match.")

    from matplotlib import pyplot as plt

    plt.show()
