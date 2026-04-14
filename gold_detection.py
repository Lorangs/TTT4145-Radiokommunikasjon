"""
Gold-code detector and framing helper.

"""

from __future__ import annotations

import logging

import numpy as np

from gold_code import get_gold_code_symbols
from modulation import modulation_rotations, nearest_constellation_symbols
from modulation import normalize_config_modulation_name


class GoldCodeDetector:
    def __init__(self, config: dict):
        modulation_type = normalize_config_modulation_name(config)
        gold_config = config["gold_sequence"]
        code_length = int(gold_config["code_length"])
        code_index = int(gold_config.get("code_index", 0))

        

        gold_symbols = get_gold_code_symbols(
            modulation_type=modulation_type,
            code_length=code_length,
            code_index=code_index,
        ).astype(np.complex64, copy=False)

        rotations = modulation_rotations(modulation_type)
        if modulation_type.upper() == "BPSK":
            self.gold_symbols = {
                0: gold_symbols * rotations[0], 
                180: gold_symbols * rotations[1]
            }
        elif modulation_type.upper() == "QPSK":
            self.gold_symbols = {
                0: gold_symbols * rotations[0],
                90: gold_symbols * rotations[1],
                180: gold_symbols * rotations[2],
                270: gold_symbols * rotations[3],
            }
        else:
            raise ValueError(f"Unsupported modulation type for Gold code: {modulation_type}")

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
        self.ref_energy = float(np.vdot(self.gold_symbols, self.gold_symbols).real)

    def add_gold_symbols(self, signal: np.ndarray) -> np.ndarray:
        """Add the selected Gold code to the beginning and end of the symbol stream."""
        return np.concatenate((self.gold_symbols, signal, self.gold_symbols))

    def remove_gold_symbols(self, signal: np.ndarray, start_index: int) -> np.ndarray:
        """Remove the Gold code from the symbol stream starting at start_index.
            Removes only the leading Gold code, assuming the trailing one is after the payload. 
            Caller should ensure start_index is valid and that the trailing Gold code is not needed before the returned payload.
        """
        if start_index < 0 or start_index + len(self.gold_symbols) > len(signal):
            logging.warning("Invalid start index for removing Gold code.")
            return signal
        return signal[start_index + len(self.gold_symbols) :]

    def normalized_correlation(self, received_signal: np.ndarray) -> np.ndarray:
        """Compute the normalized correlation between the received signal and the Gold code."""
        received = np.asarray(received_signal).astype(np.complex64, copy=False)

        if received.size < self.gold_symbols.size:
            return np.array([], dtype=np.float32)

        raw = np.correlate(received, self.gold_symbols, mode="valid")

        rx_power = np.abs(received) ** 2
        window_energy = np.convolve(
            rx_power,
            np.ones(self.gold_symbols.size, dtype=np.float32),
            mode="valid",
        )
        denom = np.sqrt(np.maximum(self.ref_energy * window_energy, 1e-12))
        return (np.abs(raw) / denom).astype(np.float32, copy=False)

    def detect(self, received_signal: np.ndarray) -> int | None:
        """Detect the presence of the leading and trailing Gold code in the received signal and return the index of the first one, or None if no match exceeds the threshold."""
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

    def detect_with_score_in_window(
        self,
        received_signal: np.ndarray,
        start_index: int,
        stop_index: int,
    ) -> tuple[int | None, float]:
        scores = self.normalized_correlation(received_signal)
        if scores.size == 0:
            return None, 0.0

        start = max(0, int(start_index))
        stop = min(int(stop_index), int(scores.size))
        if stop <= start:
            return None, 0.0

        window_scores = scores[start:stop]
        local_offset = int(np.argmax(window_scores))
        peak_index = start + local_offset
        peak_value = float(window_scores[local_offset])
        if peak_value < self.correlation_scale_factor_threshold:
            return None, peak_value
        return peak_index, peak_value


def detect_gold_with_rotation(
    received_symbols: np.ndarray,
    gold_detector: GoldCodeDetector,
    modulation_type: str,
    expected_index: int | None = None,
    search_radius: int | None = None,
) -> tuple[int | None, float, complex, np.ndarray]:
    best_index = None
    best_peak = -1.0
    best_rotation = 1 + 0j
    best_decisions = np.array([], dtype=np.complex64)

    for rotation in modulation_rotations(modulation_type):
        rotated = received_symbols * rotation
        decisions = nearest_constellation_symbols(rotated, modulation_type)
        if expected_index is not None and search_radius is not None:
            start = max(0, int(expected_index) - int(search_radius))
            stop = int(expected_index) + int(search_radius) + 1
            index, peak = gold_detector.detect_with_score_in_window(decisions, start, stop)
        else:
            index, peak = gold_detector.detect_with_score(decisions)
        if peak > best_peak:
            best_peak = peak
            best_index = index
            best_rotation = rotation
            best_decisions = decisions

    return best_index, best_peak, best_rotation, best_decisions


def rank_gold_candidates(
    symbol_stream: np.ndarray,
    gold_detector: GoldCodeDetector,
    modulation_type: str,
    expected_index: int | None = None,
    search_radius: int | None = None,
    top_k: int = 5,
) -> list[dict]:
    received = np.asarray(symbol_stream).astype(np.complex64, copy=False)
    candidates: list[dict] = []

    for rotation in modulation_rotations(modulation_type):
        rotated = received * rotation
        decisions = nearest_constellation_symbols(rotated, modulation_type)
        scores = gold_detector.normalized_correlation(decisions)
        if scores.size == 0:
            continue

        if expected_index is not None and search_radius is not None:
            start = max(0, int(expected_index) - int(search_radius))
            stop = min(int(scores.size), int(expected_index) + int(search_radius) + 1)
        else:
            start = 0
            stop = int(scores.size)

        if stop <= start:
            continue

        local_scores = scores[start:stop]
        unique_indices: set[int] = set()

        if expected_index is not None and start <= int(expected_index) < stop:
            unique_indices.add(int(expected_index) - start)

        top_count = min(int(max(1, top_k)), int(local_scores.size))
        sorted_local = np.argsort(local_scores)[-top_count:][::-1]
        for local_idx in sorted_local.tolist():
            unique_indices.add(int(local_idx))

        for local_idx in sorted(unique_indices):
            index = start + int(local_idx)
            peak = float(local_scores[local_idx])
            candidates.append(
                {
                    "phase": 0,
                    "index": int(index),
                    "peak": peak,
                    "rotation": rotation,
                    "decisions": decisions,
                }
            )

    if not candidates:
        return [
            {
                "phase": 0,
                "index": None,
                "peak": 0.0,
                "rotation": 1 + 0j,
                "decisions": np.array([], dtype=np.complex64),
            }
        ]

    def sort_key(candidate: dict) -> tuple[float, float, float]:
        if expected_index is None:
            return (-float(candidate["peak"]), 0.0, 0.0)
        distance = abs(int(candidate["index"]) - int(expected_index))
        exact_bias = 0 if distance == 0 else 1
        return (float(exact_bias), float(distance), -float(candidate["peak"]))

    candidates.sort(key=sort_key)
    return candidates[: max(1, int(top_k))]


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
