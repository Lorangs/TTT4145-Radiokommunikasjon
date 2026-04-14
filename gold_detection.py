"""
Gold-code detector and framing helper.

"""

from __future__ import annotations

from project_logger import get_logger
logger = get_logger(__name__)

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

    def remove_gold_symbols(self, signal: np.ndarray, start_index: int) -> np.ndarray:
        """Remove the Gold code from the symbol stream starting at start_index.
            Removes the leading Gold code and, when detected, strips the trailing
            Gold code as well so only the payload remains.
        """
        received = np.asarray(signal).astype(np.complex64, copy=False)

        if start_index < 0 or (start_index + self.code_length) > len(received):
            logger.warning("Invalid start index for removing Gold code.")
            return received

        payload_start = int(start_index + self.code_length)
        payload_stop = int(received.size)

        tail = received[payload_start:]
        scores = self.normalized_correlation(tail)
        candidate_indices = np.flatnonzero(scores >= self.correlation_scale_factor_threshold)
        if candidate_indices.size > 0:
            payload_stop = payload_start + int(candidate_indices[-1])

        if payload_stop < payload_start:
            logger.warning("Detected trailing Gold sequence before payload start.")
            payload_stop = payload_start

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
        expected_index: int | None = None,
        search_radius: int | None = None,
    ) -> tuple[int | None, int]:
        """Detect the Gold code in the received symbols, accounting for possible rotations. Returns (index, rotation)."""
        best_index: int | None = None
        best_peak = -1.0
        best_rotation = 0

        for rotation, template in self.gold_symbols.items():
            scores = self._normalized_correlation_with_template(received_symbols, template)
            """ p1, p2 = self._top_two_peaks_min_separation(
                scores=scores,
                min_separation=self.code_length,
                threshold=self.correlation_scale_factor_threshold,
            )
            if p1 is None or p2 is None:
                continue
            index, peak = p1 """
            
            ## Commented out code above to allow detection of single 
            ## peaks when only one Gold code is present (e.g., only leading or only trailing).
            if scores.size == 0:
                continue
            index = int(np.argmax(scores))
            peak = float(scores[index])
            if peak < self.correlation_scale_factor_threshold:
                continue
             ## Remove until here to revert to original behavior. 


            if peak > best_peak:
                best_peak = peak
                best_index = index
                best_rotation = rotation

        return best_index, best_rotation

    def rank_gold_candidates(
        self,
        symbol_stream: np.ndarray,
        expected_index: int | None = None,
        search_radius: int | None = None,
        top_candidates: int = 5,
    ) -> list[dict]:
        received = np.asarray(symbol_stream).astype(np.complex64, copy=False)
        candidates: list[dict] = []

        for rotation, sequence in self.gold_symbols.items():

            decisions = nearest_constellation_symbols(sequence, self.modulation_type)
            scores = self.normalized_correlation(decisions)
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

            top_count = min(int(max(1, top_candidates)), int(local_scores.size))
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
        return candidates[: max(1, int(top_candidates))]


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
