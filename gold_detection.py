"""
Gold-code detector and framing helper.

"""

from __future__ import annotations

from project_logger import get_logger
logger = get_logger(__name__)
import logging

import numpy as np
from numpy import typing as npt
from scipy import signal

from gold_code import get_gold_code_symbols
from modulation import modulation_rotations, normalize_config_modulation_name


class GoldCodeDetector:
    def __init__(self, config: dict, filter_taps: npt.NDArray[np.float64]):
        self.sps = int(config["modulation"]["samples_per_symbol"])
        self.modulation_type = normalize_config_modulation_name(config)
        self.modulation_order = int(config["modulation"]["order"])
        gold_config = config["gold_sequence"]
        code_length = int(gold_config["code_length"])
        code_index = int(gold_config.get("code_index", 0))
        self.estimated_data_length = (
            self.sps * (
                int(config['datagram']['total_size']) +
                int(config['gold_sequence']['code_length']) +
                int(config['coding']['rs_added_bytes'])
            ) // int(config['modulation']['order'])
        )

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

        self.upsampled_and_filtered_gold = signal.convolve(
            self.upsample(self.gold_symbols[0]),
            filter_taps,
            mode='full'
        )      

        self.code_length = code_length
        self.code_index = code_index

        self.correlation_threshold = float(config["gold_sequence"].get("correlation_threshold"))

    def update_threshold_with_noise_floor(self, noise_floor_dB: float):
        """Update the correlation threshold based on the measured noise floor."""
        noise_floor_linear = 10 ** (noise_floor_dB / 10)
        self.correlation_threshold *= noise_floor_linear
        logger.info(f"Updated correlation threshold based on noise floor: {self.correlation_threshold:.3f}")

    def add_gold_symbols(self, signal: np.ndarray) -> npt.NDArray[np.complex64]:
        """Add the selected Gold code to the beginning of the symbol stream."""
        return np.concatenate((self.gold_symbols.get(0), signal))

    def remove_gold_symbols(self, signal: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
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
        payload_start = int(gold_len)
        payload_stop = int(payload_start + self.estimated_data_length)

        if gold_len > received.size:
            raise ValueError("Invalid leading Gold index.")

        if payload_stop > received.size:
            raise ValueError(
                f"Buffer too short for payload extraction: "
                f"payload_stop={payload_stop}, available={received.size}"
            )
        return received[payload_start:payload_stop]


    def timing_estimate(self, samples: npt.NDAarray[np.complex64]) -> tuple[int, int] | None:
        """
        Estimate the timing offset of the leading Gold code in the given sample stream.

        This method assumes that the samples are already aligned to symbol boundaries and that the leading Gold code is present. 
        It returns the estimated timing offset (in samples) and which branch has the maximum energy.
        
        Args:
            samples: Complex baseband samples at symbol rate (after matched filtering)
        Returns:
            - Estimated timing offset in samples (integer)
            - Index of the upsample branch with maximum energy (integer) range: 0 -> sps-1
        """
        correlation = signal.correlate(samples, self.upsampled_and_filtered_gold, mode='valid')
        correlation_magnitude = np.abs(correlation)
        peak_indices = signal.find_peaks(
            x=correlation_magnitude,
            threshold=self.correlation_threshold,
            distance=self.estimated_data_length*2
        )[0]

        if len(peak_indices) == 0:
            raise ValueError("No Gold code peak found above the correlation threshold.")
        else:
            logger.debug(f"Found {len(peak_indices)} peaks above the correlation threshold: {peak_indices}")
            peak_idx = 0
            for idx in peak_indices:
                if correlation_magnitude[idx] > correlation_magnitude[peak_idx]:
                    peak_idx = idx

        # calculate which upsample branch has the maximum energy
        max_energy = 0
        max_energy_branch = 0
        for i in range(self.sps):
            branch_energy = np.sum(np.abs(samples[i:self.estimated_data_length:self.sps])**self.modulation_order)
            if branch_energy > max_energy:
                max_energy = branch_energy
                max_energy_branch = i

        return peak_idx, max_energy_branch, correlation
        
    def upsample(self, symbols: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
        upsampled = np.zeros(symbols.size * self.sps, dtype=np.complex64)
        upsampled[::self.sps] = symbols.astype(np.complex64, copy=False)
        return upsampled

    def downsample_and_crop(self, signal: npt.NDArray[np.complex64], start_inx: int, branch_index: int) -> npt.NDArray[np.complex64]:
        start_idx = int(start_inx + branch_index + self.code_length * self.sps)
        end_idx = int(start_idx + 1280) # 32 bytes + 8 RS bytes = 40 bytes * 8 bits/byte = 320 symbols * sps (4) = 1280 samples
        signal_slice = signal[start_idx:end_idx:self.sps]
        return signal_slice.astype(np.complex64, copy=False)

            
        

    

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
