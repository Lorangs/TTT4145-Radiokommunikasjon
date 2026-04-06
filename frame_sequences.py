from __future__ import annotations

import numpy as np


def _default_block_length(config: dict) -> int:
    gold_cfg = config.get("gold_sequence", {})
    return max(4, int(gold_cfg.get("code_length", 31)))


def build_timing_reference_block(
    config: dict,
    modulation_name: str,
) -> np.ndarray:
    sync_cfg = config.get("synchronization", {})
    block_length = int(sync_cfg.get("timing_preamble_block_symbols", _default_block_length(config)))
    block_length = max(4, block_length)

    modulation = str(modulation_name).upper()
    if modulation == "BPSK":
        return np.where(
            (np.arange(block_length, dtype=np.int32) % 2) == 0,
            1.0 + 0.0j,
            -1.0 + 0.0j,
        ).astype(np.complex64, copy=False)

    if modulation == "QPSK":
        base = np.array(
            [1.0 + 1.0j, -1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j],
            dtype=np.complex64,
        ) / np.sqrt(2.0)
        return np.resize(base, block_length).astype(np.complex64, copy=False)

    raise ValueError(f"Unsupported modulation for timing preamble: {modulation_name}")


def build_sync_preamble_symbols(
    config: dict,
    modulation_name: str,
) -> np.ndarray:
    repeat_count = int(config.get("synchronization", {}).get("preamble_repeat_count", 4))
    repeat_count = max(1, repeat_count)
    block = build_timing_reference_block(config, modulation_name)
    return np.tile(block, repeat_count).astype(np.complex64, copy=False)
