"""
Gold-code detector and framing helper.

"""

from __future__ import annotations

import logging

import numpy as np

from gold_code import get_gold_code_symbols


class GoldCodeDetector:
    def __init__(self, config: dict):
        modulation_type = str(config["modulation"]["type"]).upper().strip()
        gold_config = config["gold_sequence"]
        code_length = int(gold_config["code_length"])
        code_index = int(gold_config.get("code_index", 0))

        self.gold_symbols = get_gold_code_symbols(
            modulation_type=modulation_type,
            code_length=code_length,
            code_index=code_index,
        )
        self.code_length = code_length
        self.code_index = code_index

        threshold = gold_config.get(
            "correlation_scale_factor_threshold",
            gold_config.get("correlation_threshold"),
        )
        if threshold is None:
            raise ValueError(
                "Missing Gold correlation threshold. Expected "
                "'correlation_scale_factor_threshold' or 'correlation_threshold'."
            )
        self.correlation_scale_factor_threshold = float(threshold)

    def add_gold_symbols(self, signal: np.ndarray) -> np.ndarray:
        """Add the selected Gold code to the beginning of the symbol stream."""
        return np.concatenate((self.gold_symbols, signal))

    def remove_gold_symbols(self, signal: np.ndarray, start_index: int) -> np.ndarray:
        """Remove the Gold code from the symbol stream starting at start_index."""
        if start_index < 0 or start_index + len(self.gold_symbols) > len(signal):
            logging.warning("Invalid start index for removing Gold code.")
            return signal
        return signal[start_index + len(self.gold_symbols) :]

    def normalized_correlation(self, received_signal: np.ndarray) -> np.ndarray:
        reference = self.gold_symbols.astype(np.complex64, copy=False)
        received = np.asarray(received_signal).astype(np.complex64, copy=False)

        if received.size < reference.size:
            return np.array([], dtype=np.float32)

        raw = np.correlate(received, reference, mode="valid")
        ref_energy = float(np.vdot(reference, reference).real)
        rx_power = np.abs(received) ** 2
        window_energy = np.convolve(
            rx_power,
            np.ones(reference.size, dtype=np.float32),
            mode="valid",
        )
        denom = np.sqrt(np.maximum(ref_energy * window_energy, 1e-12))
        return (np.abs(raw) / denom).astype(np.float32, copy=False)

    def detect(self, received_signal: np.ndarray) -> int | None:
        scores = self.normalized_correlation(received_signal)
        if scores.size == 0:
            return None

        peak_index = int(np.argmax(scores))
        peak_value = float(scores[peak_index])
        if peak_value < self.correlation_scale_factor_threshold:
            return None
        return peak_index

    def detect_with_score(self, received_signal: np.ndarray) -> tuple[int | None, float]:
        scores = self.normalized_correlation(received_signal)
        if scores.size == 0:
            return None, 0.0

        peak_index = int(np.argmax(scores))
        peak_value = float(scores[peak_index])
        if peak_value < self.correlation_scale_factor_threshold:
            return None, peak_value
        return peak_index, peak_value


if __name__ == "__main__":
    from copy import deepcopy
    from yaml import safe_load

    with open("setup/config.yaml", "r", encoding="utf-8") as handle:
        config = safe_load(handle)

    config = deepcopy(config)
    config["gold_sequence"] = {
        "code_length": 31,
        "code_index": 2,
        "correlation_threshold": 0.8,
    }

    detector = GoldCodeDetector(config)
    payload = np.ones(64, dtype=np.complex64)
    tx = detector.add_gold_symbols(payload)

    noise = 0.05 * (
        np.random.randn(tx.size + 100) + 1j * np.random.randn(tx.size + 100)
    ).astype(np.complex64)
    insert_position = 20
    rx = noise.copy()
    rx[insert_position : insert_position + tx.size] += tx

    detected_index, score = detector.detect_with_score(rx)
    print(
        f"Detected Gold code at index={detected_index}, "
        f"score={score:.3f}, expected={insert_position}"
    )
