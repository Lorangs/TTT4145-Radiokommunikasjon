"""
Gold-code detector and framing helper.

"""

from __future__ import annotations

from project_logger import get_logger
logger = get_logger(__name__)
import logging

import numpy as np

from gold_code import get_gold_code_symbols
from modulation import modulation_rotations, nearest_constellation_symbols
from modulation import normalize_config_modulation_name


class GoldCodeDetector:
    def __init__(self, config: dict):
        self.modulation_type = normalize_config_modulation_name(config)
        gold_config = config["gold_sequence"]
        code_length = int(gold_config["code_length"])
        code_index = int(gold_config.get("code_index", 0))

        gold_symbols = get_gold_code_symbols(
            modulation_type=self.modulation_type,
            code_length=code_length,
            code_index=code_index,
        ).astype(np.complex64, copy=False)

        self.rotation_matrix = modulation_rotations(self.modulation_type) / np.sqrt(2)  # Normalize rotation matrix to keep symbol energy consistent
        if self.modulation_type.upper() == "BPSK":
            self.gold_symbols = {
                0: gold_symbols * self.rotation_matrix[0], 
                180: gold_symbols * self.rotation_matrix[1]
            }
        elif self.modulation_type.upper() == "QPSK":
            self.gold_symbols = {
                0: gold_symbols * self.rotation_matrix[0],
                90: gold_symbols * self.rotation_matrix[1],
                180: gold_symbols * self.rotation_matrix[2],
                270: gold_symbols * self.rotation_matrix[3],
            }
        else:
            raise ValueError(f"Unsupported modulation type for Gold code: {self.modulation_type}")

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
        self.ref_energy = float(np.vdot(self.gold_symbols.get(0), self.gold_symbols.get(0)).real)


    def add_gold_symbols(self, signal: np.ndarray) -> np.ndarray:
        """Add the selected Gold code to the beginning and end of the symbol stream."""
        return np.concatenate((self.gold_symbols.get(0), signal, self.gold_symbols.get(0)))

    def remove_gold_symbols(
        self,
        signal: np.ndarray,
        start_index: int,
        payload_symbol_count: int,
    ) -> np.ndarray:
        """
        Remove the leading Gold sequence and return exactly the payload symbols.

        Frame layout:
            [leading Gold][payload][trailing Gold]

        Args:
            signal: Symbol-rate complex stream
            start_index: Detected start index of the leading Gold sequence
            payload_symbol_count: Expected number of payload symbols

        Returns:
            Payload-only symbol stream

        Raises:
            ValueError if the requested payload region does not fit inside the buffer.
        """
        received = np.asarray(signal).astype(np.complex64, copy=False)

        gold_len = int(self.gold_symbols[0].size)
        payload_start = int(start_index + gold_len)
        payload_stop = int(payload_start + payload_symbol_count)

        if start_index < 0 or (start_index + gold_len) > received.size:
            raise ValueError("Invalid leading Gold index.")

        if payload_stop > received.size:
            raise ValueError(
                f"Buffer too short for payload extraction: "
                f"payload_stop={payload_stop}, available={received.size}"
            )
        return received[payload_start:payload_stop]


    def normalized_correlation(self, received_signal: np.ndarray) -> np.ndarray:
        """Compute the normalized correlation between the received signal and the Gold code."""
        received = np.asarray(received_signal).astype(np.complex64, copy=False)

        if received.size < self.gold_symbols.get(0).size:
            return np.array([], dtype=np.float32)

        raw = np.correlate(received, self.gold_symbols.get(0), mode="valid")

        rx_power = np.abs(received) ** 2
        window_energy = np.convolve(
            rx_power,
            np.ones(self.gold_symbols.get(0).size, dtype=np.float32),
            mode="valid",
        )
        denom = np.sqrt(np.maximum(self.ref_energy * window_energy, 1e-12))
        return (np.abs(raw) / denom).astype(np.float32, copy=False)
    
    def _normalized_correlation_with_template(
        self,
        received_signal: np.ndarray,
        template: np.ndarray,
    ) -> np.ndarray:
        received = np.asarray(received_signal).astype(np.complex64, copy=False)
        template = np.asarray(template).astype(np.complex64, copy=False)

        if received.size < template.size:
            return np.array([], dtype=np.float32)

        raw = np.correlate(received, template, mode="valid")
        rx_power = np.abs(received) ** 2
        window_energy = np.convolve(
            rx_power,
            np.ones(template.size, dtype=np.float32),
            mode="valid",
        )
        ref_energy = float(np.vdot(template, template).real)
        denom = np.sqrt(np.maximum(ref_energy * window_energy, 1e-12))
        
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

    def rotate_signal(self, signal: np.ndarray, rotation: int) -> np.ndarray:
        """Rotate the signal by the specified angle (in degrees) if it's a known rotation for the modulation type."""
        return signal * self.rotation_matrix[rotation]

    def detect_with_score(
        self, 
        received_signal: np.ndarray
    ) -> tuple[tuple[int, float] | None, tuple[int, float] | None]:
        """
        Return two strongest peaks (leading, trailing), or (None, None).
        Each peak is (index, score).
        """
        scores = self.normalized_correlation(received_signal)
        p1, p2 = self._top_two_peaks_min_separation(
            scores=scores,
            min_separation=self.code_length,
            threshold=self.correlation_scale_factor_threshold,
        )
        if p1 is None or p2 is None:
            return None, None

        # time order
        if p1[0] <= p2[0]:
            return p1, p2
        return p2, p1

    @staticmethod
    def _top_two_peaks_min_separation(
        scores: np.ndarray,
        min_separation: int,
        threshold: float,
    ) -> tuple[tuple[int, float] | None, tuple[int, float] | None]:
        """Return two strongest peaks >= threshold with |i-j| >= min_separation."""
        if scores.size == 0:
            return None, None

        order = np.argsort(scores)[::-1]  # strongest first
        first: tuple[int, float] | None = None
        second: tuple[int, float] | None = None

        for idx in order:
            i = int(idx)
            v = float(scores[i])
            if v < threshold:
                break

            if first is None:
                first = (i, v)
                continue

            if abs(i - first[0]) >= int(min_separation):
                second = (i, v)
                break

        return first, second


    def detect_with_rotation(
        self,
        received_symbols: np.ndarray,
        payload_symbol_count: int,
    ) -> tuple[int | None, int]:
        """
        Detect the earliest Gold-code match that can fit a full frame.

        Returns:
            (best_index, best_rotation)

        The candidate search scans all allowed rotations, keeps peaks above the
        configured correlation threshold, filters out peaks that cannot be the
        leading Gold sequence of a complete frame, and then prefers the
        earliest valid candidate. This avoids locking onto the trailing Gold
        sequence when both frame edges correlate strongly.

        If no valid correlation peak is found, returns (None, 0).
        """
        received = np.asarray(received_symbols).astype(np.complex64, copy=False)

        candidates: list[tuple[int, int, float]] = []

        for rotation, template in self.gold_symbols.items():
            scores = self._normalized_correlation_with_template(received, template)
            if scores.size == 0:
                continue

            order = np.argsort(scores)[::-1]
            for idx in order:
                peak = float(scores[idx])
                if peak < self.correlation_scale_factor_threshold:
                    break

                start_index = int(idx)
                if not self.candidate_fits_frame(
                    signal_length=received.size,
                    start_index=start_index,
                    payload_symbol_count=payload_symbol_count,
                    require_trailing_gold=True,
                ):
                    continue

                candidates.append((start_index, rotation, peak))

        if not candidates:
            return None, 0

        candidates.sort(key=lambda item: (item[0], -item[2]))
        best_index, best_rotation, _ = candidates[0]
        return best_index, best_rotation


      
    def candidate_fits_frame(
        self,
        signal_length: int,
        start_index: int,
        payload_symbol_count: int,
        require_trailing_gold: bool = False,
    ) -> bool:
        """
        Check whether a detected Gold position can fit a full frame.

        Frame layout:
            [leading Gold][payload][trailing Gold]
        """
        gold_len = int(self.gold_symbols[0].size)

        required_symbols = gold_len + payload_symbol_count
        if require_trailing_gold:
            required_symbols += gold_len
        return 0 <= start_index <= (signal_length - required_symbols)

    

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
