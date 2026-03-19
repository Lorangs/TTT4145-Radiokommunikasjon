"""
Hardware smoke test for the Pluto SDR path.

This script performs two focused checks:
    1. local scrambler round-trip validation through the modem pack/unpack path
    2. over-the-air Gold-code framing, detection, and datagram recovery on Pluto


Arguments:
    --config:
        Path to the YAML configuration file. Default: `setup/config.yaml`
    --payload:
        Payload text to transmit and recover. Default:
        `"Gold code and scrambler hardware test payload"`
    --flush-buffers:
        Number of RX buffers to discard before the measured capture. Default: `10`
    --plots:
        If set, show diagnostic plots at the end of the run.
"""

from __future__ import annotations

import argparse
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from yaml import safe_load

from datagram import Datagram, msgType
from filter import BWLPFilter, RRCFilter
from gold_detection import GoldCodeDetector
from modulation import ModulationProtocol
from sdr_plots import StaticSDRPlotter
from sdr_transciever import SDRTransciever


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return safe_load(handle)


def nearest_constellation_symbols(symbols: np.ndarray, modulation_type: str) -> np.ndarray:
    modulation_type = modulation_type.upper().strip()
    if modulation_type == "BPSK":
        return np.where(symbols.real >= 0, 1.0, -1.0).astype(np.complex64)
    if modulation_type == "QPSK":
        return (
            np.where(symbols.real >= 0, 1.0, -1.0)
            + 1j * np.where(symbols.imag >= 0, 1.0, -1.0)
        ).astype(np.complex64)
    raise ValueError(f"Unsupported modulation type: {modulation_type}")


def modulation_rotations(modulation_type: str) -> tuple[complex, ...]:
    modulation_type = modulation_type.upper().strip()
    if modulation_type == "BPSK":
        return (1 + 0j, -1 + 0j)
    if modulation_type == "QPSK":
        return (1 + 0j, -1 + 0j, 1j, -1j)
    raise ValueError(f"Unsupported modulation type: {modulation_type}")


def bit_balance(bits: np.ndarray) -> tuple[int, int]:
    ones = int(np.count_nonzero(bits))
    zeros = int(bits.size - ones)
    return zeros, ones


def ensure_gold_config(config: dict) -> dict:
    cfg = deepcopy(config)
    gold_cfg = cfg.setdefault("gold_sequence", {})
    gold_cfg.setdefault("code_length", 31)
    gold_cfg.setdefault("code_index", 0)
    if "correlation_scale_factor_threshold" not in gold_cfg:
        gold_cfg["correlation_scale_factor_threshold"] = gold_cfg.get("correlation_threshold", 0.7)
    return cfg


def run_scrambler_test(modem: ModulationProtocol, payload_text: str) -> tuple[Datagram, dict]:
    datagram = Datagram.as_string(payload_text, msg_type=msgType.DATA)
    raw_bits = np.unpackbits(np.frombuffer(datagram.pack(), dtype=np.uint8))
    scrambled_bits = modem.pack_message_bits(datagram)
    zeros, ones = bit_balance(scrambled_bits)
    changed = int(np.count_nonzero(scrambled_bits != raw_bits))

    roundtrip_ok = False
    recovered_text = ""
    error_message = ""
    try:
        recovered = modem.unpack_message_bits(scrambled_bits)
        recovered_text = recovered.get_payload.tobytes().decode("utf-8", errors="replace")
        roundtrip_ok = (
            recovered.get_msg_id == datagram.get_msg_id
            and recovered.get_msg_type == datagram.get_msg_type
            and np.array_equal(recovered.get_payload, datagram.get_payload)
        )
    except Exception as exc:
        error_message = str(exc)

    result = {
        "scrambler_enabled": bool(modem.scrambler_enable),
        "bit_count": int(scrambled_bits.size),
        "zero_bits": zeros,
        "one_bits": ones,
        "changed_bits": changed,
        "roundtrip_ok": roundtrip_ok,
        "recovered_text": recovered_text,
        "error": error_message,
    }
    return datagram, result


def build_tx_burst(
    modem: ModulationProtocol,
    gold_detector: GoldCodeDetector,
    rrc_filter: RRCFilter,
    datagram: Datagram,
    guard_symbols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload_symbols = modem.modulate_message(datagram).astype(np.complex64)
    framed_symbols = gold_detector.add_gold_symbols(payload_symbols).astype(np.complex64)
    guard = np.zeros(guard_symbols, dtype=np.complex64)
    burst_symbols = np.concatenate((guard, framed_symbols, guard))
    upsampled = modem.upsample_symbols(burst_symbols)
    tx_signal = rrc_filter.apply_filter(upsampled).astype(np.complex64)
    peak = max(float(np.max(np.abs(tx_signal))), 1e-12)
    tx_signal = (0.6 * tx_signal / peak * (2**14)).astype(np.complex64)
    return payload_symbols, burst_symbols, tx_signal


def maybe_apply_frontend_filter(config: dict, signal_in: np.ndarray) -> np.ndarray:
    if not bool(config["filter"].get("butterworth_enable", False)):
        return signal_in
    return BWLPFilter(config).apply_filter(signal_in)


def detect_gold_with_rotation(
    received_symbols: np.ndarray,
    gold_detector: GoldCodeDetector,
    modulation_type: str,
) -> tuple[int | None, float, complex, np.ndarray]:
    best_index = None
    best_peak = -1.0
    best_rotation = 1 + 0j
    best_decisions = np.array([], dtype=np.complex64)

    for rotation in modulation_rotations(modulation_type):
        rotated = received_symbols * rotation
        decisions = nearest_constellation_symbols(rotated, modulation_type)
        index, peak = gold_detector.detect_with_score(decisions)
        if peak > best_peak:
            best_peak = peak
            best_index = index
            best_rotation = rotation
            best_decisions = decisions

    return best_index, best_peak, best_rotation, best_decisions


def recover_with_rotation(
    payload_symbols_rx: np.ndarray,
    modem: ModulationProtocol,
    datagram: Datagram,
) -> tuple[bool, str, complex | None, str]:
    expected_text = datagram.get_payload.tobytes().decode("utf-8", errors="replace")
    last_error = ""

    for rotation in modulation_rotations(modem.modulation_type):
        try:
            decisions = nearest_constellation_symbols(payload_symbols_rx * rotation, modem.modulation_type)
            recovered = modem.demodulate_message(decisions)
            recovered_text = recovered.get_payload.tobytes().decode("utf-8", errors="replace")
            if recovered_text == expected_text:
                return True, recovered_text, rotation, ""
        except Exception as exc:
            last_error = str(exc)

    return False, "", None, last_error


def rank_gold_phase_candidates(
    matched_signal: np.ndarray,
    gold_detector: GoldCodeDetector,
    modem: ModulationProtocol,
    sps: int,
) -> list[dict]:
    candidates: list[dict] = []

    for phase in range(sps):
        symbols = matched_signal[phase::sps]
        index, peak, rotation, decisions = detect_gold_with_rotation(
            symbols,
            gold_detector,
            modem.modulation_type,
        )
        candidates.append(
            {
                "phase": phase,
                "index": index,
                "peak": peak,
                "rotation": rotation,
                "decisions": decisions,
            }
        )

    return sorted(candidates, key=lambda item: item["peak"], reverse=True)


def run_gold_hardware_test(
    config: dict,
    modem: ModulationProtocol,
    gold_detector: GoldCodeDetector,
    rrc_filter: RRCFilter,
    sdr: SDRTransciever,
    datagram: Datagram,
    flush_buffers: int,
) -> tuple[dict, dict]:
    payload_symbols, burst_symbols, tx_signal = build_tx_burst(
        modem=modem,
        gold_detector=gold_detector,
        rrc_filter=rrc_filter,
        datagram=datagram,
        guard_symbols=32,
    )

    sps = int(config["modulation"]["samples_per_symbol"])
    sdr.sdr.tx_destroy_buffer()
    sdr.sdr.tx(tx_signal)

    for _ in range(flush_buffers):
        sdr.sdr.rx()

    received_signal = sdr.sdr.rx()

    frontend_filtered_signal = maybe_apply_frontend_filter(config, received_signal)
    matched_signal = rrc_filter.apply_filter(frontend_filtered_signal)

    phase_candidates = rank_gold_phase_candidates(
        matched_signal,
        gold_detector,
        modem,
        sps,
    )

    best_candidate = phase_candidates[0]
    start_index = best_candidate["index"]
    peak = float(best_candidate["peak"])
    best_phase = int(best_candidate["phase"])
    detection_rotation = best_candidate["rotation"]
    decided = best_candidate["decisions"]

    recovered_ok = False
    recovered_text = ""
    decode_rotation = None
    decode_error = ""
    for candidate in phase_candidates:
        candidate_index = candidate["index"]
        if candidate_index is None:
            continue

        payload_rx = gold_detector.remove_gold_symbols(candidate["decisions"], candidate_index)
        payload_rx = payload_rx[: payload_symbols.size]
        recovered_ok, recovered_text, decode_rotation, decode_error = recover_with_rotation(
            payload_rx,
            modem,
            datagram,
        )
        if recovered_ok:
            start_index = int(candidate_index)
            peak = float(candidate["peak"])
            best_phase = int(candidate["phase"])
            detection_rotation = candidate["rotation"]
            decided = candidate["decisions"]
            break

    result = {
        "rx_samples": int(received_signal.size),
        "tx_symbols": int(burst_symbols.size),
        "payload_symbols": int(payload_symbols.size),
        "detected": start_index is not None,
        "detected_index": start_index,
        "peak": peak,
        "best_symbol_phase": best_phase,
        "detection_rotation": str(detection_rotation),
        "recovered_ok": recovered_ok,
        "decode_rotation": str(decode_rotation) if decode_rotation is not None else "",
        "decode_error": decode_error,
        "recovered_text": recovered_text,
    }

    traces = {
        "received_signal": received_signal,
        "frontend_filtered_signal": frontend_filtered_signal,
        "matched_signal": matched_signal,
        "decided_symbols": decided,
    }
    return result, traces


def print_result_block(title: str, result: dict) -> None:
    print(title)
    for key, value in result.items():
        print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scrambler and Gold-code hardware tests on Pluto.")
    parser.add_argument("--config", default="setup/config.yaml", help="Path to configuration file.")
    parser.add_argument("--payload", default="Gold code and scrambler hardware test payload", help="Payload text to transmit.")
    parser.add_argument("--flush-buffers", type=int, default=10, help="Number of RX buffers to discard before capture.")
    parser.add_argument("--plots", action="store_true", help="Show diagnostic plots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ensure_gold_config(load_config(args.config))

    modem = ModulationProtocol(config)
    rrc_filter = RRCFilter(config)
    gold_detector = GoldCodeDetector(config)
    sdr = SDRTransciever(config)

    datagram, scrambler_result = run_scrambler_test(modem, args.payload)
    print_result_block("Scrambler test", scrambler_result)
    if not scrambler_result["roundtrip_ok"]:
        return 1

    if not sdr.connect():
        print("Failed to connect to SDR. Exiting.")
        return 1

    try:
        sdr.measure_noise_floor_dB()

        gold_result, traces = run_gold_hardware_test(
            config=config,
            modem=modem,
            gold_detector=gold_detector,
            rrc_filter=rrc_filter,
            sdr=sdr,
            datagram=datagram,
            flush_buffers=args.flush_buffers,
        )
        print_result_block("Gold code hardware test", gold_result)

        if args.plots:
            plotter = StaticSDRPlotter()
            sample_rate = int(float(config["modulation"]["sample_rate"]))
            sps = int(config["modulation"]["samples_per_symbol"])

            plotter.plot_psd(traces["received_signal"], sample_rate, title="Received Signal PSD")
            plotter.plot_time_domain(traces["matched_signal"], sample_rate, title="Matched Filter Output")
            plotter.plot_constellation(traces["frontend_filtered_signal"], title="Front-End Filter Output")
            plotter.plot_constellation(traces["decided_symbols"], title="Detected Symbol Decisions")
            plotter.plot_eye_diagram(traces["matched_signal"], sps, title="Eye Diagram after Matched Filtering")
            plt.show()

        return 0 if gold_result["detected"] and gold_result["recovered_ok"] else 1
    finally:
        sdr.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
