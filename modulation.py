from __future__ import annotations

import numpy as np
from commpy import modulation as commpy_modulation


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


def bits_per_symbol(modulation_name: str) -> int:
    normalized = modulation_name.upper().strip()
    if normalized == "BPSK":
        return 1
    if normalized == "QPSK":
        return 2
    raise ValueError(f"Unsupported symbol mapping for modulation {modulation_name}")


def bits_to_symbols(bits: np.ndarray, modulation_name: str) -> np.ndarray:
    normalized = modulation_name.upper().strip()
    if normalized == "BPSK":
        return (1 - 2 * bits.astype(np.int8)).astype(np.complex64)

    if normalized == "QPSK":
        if bits.size % 2 == 1:
            bits = np.concatenate((bits, np.array([0], dtype=np.uint8)))
        pairs = bits.reshape(-1, 2)
        i = (1 - 2 * pairs[:, 0].astype(np.int8)).astype(np.float32)
        q = (1 - 2 * pairs[:, 1].astype(np.int8)).astype(np.float32)
        return (i + 1j * q).astype(np.complex64)

    raise ValueError(f"Unsupported modulation type: {modulation_name}")


def symbols_to_bits(symbols: np.ndarray, modulation_name: str) -> np.ndarray:
    normalized = modulation_name.upper().strip()
    if normalized == "BPSK":
        return np.where(symbols.real < 0, 1, 0).astype(np.uint8)

    if normalized == "QPSK":
        i_bits = np.where(symbols.real < 0, 1, 0).astype(np.uint8)
        q_bits = np.where(symbols.imag < 0, 1, 0).astype(np.uint8)
        bits = np.empty(symbols.size * 2, dtype=np.uint8)
        bits[0::2] = i_bits
        bits[1::2] = q_bits
        return bits

    raise ValueError(f"Unsupported modulation type: {modulation_name}")


def nearest_constellation_symbols(symbols: np.ndarray, modulation_name: str) -> np.ndarray:
    normalized = modulation_name.upper().strip()
    if normalized == "BPSK":
        return np.where(symbols.real >= 0, 1.0, -1.0).astype(np.complex64)

    if normalized == "QPSK":
        return (
            np.where(symbols.real >= 0, 1.0, -1.0)
            + 1j * np.where(symbols.imag >= 0, 1.0, -1.0)
        ).astype(np.complex64)

    raise ValueError(f"Unsupported modulation type: {modulation_name}")


def modulation_rotations(modulation_name: str) -> tuple[complex, ...]:
    normalized = modulation_name.upper().strip()
    if normalized == "BPSK":
        return (1 + 0j, -1 + 0j)
    if normalized == "QPSK":
        return (1 + 0j, -1 + 0j, 1j, -1j)
    raise ValueError(f"Unsupported modulation type: {modulation_name}")


def upsample_symbols(symbols: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    upsampled = np.zeros(symbols.size * samples_per_symbol, dtype=np.complex64)
    upsampled[::samples_per_symbol] = symbols.astype(np.complex64, copy=False)
    return upsampled


class ModulationProtocol:
    def __init__(self, config: dict):
        modulation_config = config["modulation"]
        raw_type = str(modulation_config["type"]).upper().strip()
        self.modulation_order = int(modulation_config["order"])
        self.modulation_type = normalize_modulation_name(raw_type, self.modulation_order)

        if raw_type == "PSK":
            self.modulator = commpy_modulation.PSKModem(self.modulation_order)
        elif raw_type == "QAM":
            self.modulator = commpy_modulation.QAMModem(self.modulation_order)
        else:
            raise ValueError(f"Unsupported modulation type: {raw_type}")

    def modulate_message(self, bit_stream: np.ndarray) -> np.ndarray:
        """Modulate a bit stream into a complex baseband signal."""
        return self.modulator.modulate(bit_stream)

    def demodulate_signal(self, signal: np.ndarray) -> np.ndarray:
        """Demodulate a complex baseband signal back into a bit stream."""
        return self.modulator.demodulate(signal, demod_type="hard")


if __name__ == "__main__":
    from sdr_plots import StaticSDRPlotter

    plotter = StaticSDRPlotter()

    modulation_config = {
        "modulation": {
            "type": "QAM",
            "order": 16,
        }
    }

    protocol = ModulationProtocol(modulation_config)
    test_bits = np.random.randint(0, 2, 256)
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
