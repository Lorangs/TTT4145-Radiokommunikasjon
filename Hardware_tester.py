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
    --rx-buffer-size:
        Optional RX buffer size override. Default:
        `receiver.buffer_size` from the YAML config
    --flush-buffers:
        Number of RX buffers to discard before the measured capture. Default: `3`
    --plots:
        If set, show diagnostic plots at the end of the run.
    --runs:
        Number of back-to-back hardware test runs to execute in one invocation. Default: `1
"""

from __future__ import annotations

import argparse
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
from yaml import safe_dump, safe_load

from RX_pipeline import run_rx_pipeline
from TX_pipeline import (
    bit_balance,
    build_frame_layout_summary,
    build_scrambler,
    build_tx_burst,
    decode_payload_symbols,
    encode_datagram,
)
from datagram import Datagram, msgType
from filter import RRCFilter
from interleaver import Interleaver
from gold_detection import GoldCodeDetector, rank_gold_candidates
from modulation import nearest_constellation_symbols, normalize_config_modulation_name
from forward_error_correction import FCCodec
from convolutional_coder import ConvolutionalCoder
from scrambler import LFSRScrambler
from project_logger import configure_project_logging, get_configured_log_level
from project_logger import get_logger
from synchronize import Synchronizer
from sdr_transciever import SDRTransciever

logger = get_logger(__name__)


def compute_nearest_symbol_error(
    signal: np.ndarray,
    modulation_name: str,
) -> float:
    samples = np.asarray(signal).astype(np.complex64, copy=False)
    if samples.size == 0:
        return 0.0
    decisions = nearest_constellation_symbols(samples, modulation_name)
    return float(np.mean(np.abs(samples - decisions)))


def trim_gold_removed_payload(
    payload_suffix: np.ndarray,
    expected_symbol_count: int,
) -> np.ndarray:
    symbols = np.asarray(payload_suffix).astype(np.complex64, copy=False).reshape(-1)
    return symbols[: max(0, int(expected_symbol_count))]


def phase_align_symbols_for_decision_axis(
    signal: np.ndarray,
    modulation_name: str,
) -> dict[str, np.ndarray | float]:
    samples = np.asarray(signal).astype(np.complex64, copy=False).reshape(-1)
    if samples.size == 0:
        empty_complex = np.array([], dtype=np.complex64)
        empty_real = np.array([], dtype=np.float32)
        return {
            "aligned_signal": empty_complex,
            "decisions": empty_complex,
            "projected_signal": empty_real,
            "residual_phase_rad": empty_real,
            "global_phase_rad": 0.0,
        }

    decisions = nearest_constellation_symbols(samples, modulation_name).astype(
        np.complex64,
        copy=False,
    )
    # Rotate the cloud so the nearest ideal symbols lie on the real axis. This
    # makes the saved plots reflect BPSK decision quality instead of raw phase
    # offset, which was easy to misread during tuning.
    correlation = np.vdot(decisions, samples)
    global_phase = float(np.angle(correlation)) if abs(correlation) > 1e-12 else 0.0
    aligned_signal = (samples * np.exp(-1j * global_phase)).astype(
        np.complex64,
        copy=False,
    )
    residual_phase = np.angle(
        aligned_signal * np.conj(decisions).astype(np.complex64, copy=False)
    ).astype(np.float32, copy=False)

    return {
        "aligned_signal": aligned_signal,
        "decisions": decisions,
        "projected_signal": aligned_signal.real.astype(np.float32, copy=False),
        "residual_phase_rad": residual_phase,
        "global_phase_rad": float(global_phase),
    }


def build_decision_diagnostic_traces(
    traces: dict,
    modulation_name: str,
) -> dict[str, np.ndarray | float]:
    # Keep the diagnostics focused on the three places we care about during
    # debugging: before EQ, after EQ, and after final carrier correction.
    pre_equalizer = phase_align_symbols_for_decision_axis(
        traces.get("carrier_aligned_signal", np.array([], dtype=np.complex64)),
        modulation_name,
    )
    post_equalizer = phase_align_symbols_for_decision_axis(
        traces.get("equalized_signal", np.array([], dtype=np.complex64)),
        modulation_name,
    )
    final_output = phase_align_symbols_for_decision_axis(
        traces.get("fine_complex_signal", np.array([], dtype=np.complex64)),
        modulation_name,
    )

    return {
        "pre_equalizer_decision_signal": pre_equalizer["aligned_signal"],
        "pre_equalizer_projected_signal": pre_equalizer["projected_signal"],
        "post_equalizer_decision_signal": post_equalizer["aligned_signal"],
        "post_equalizer_projected_signal": post_equalizer["projected_signal"],
        "final_decision_signal": final_output["aligned_signal"],
        "final_projected_signal": final_output["projected_signal"],
        "final_residual_phase_rad": final_output["residual_phase_rad"],
        "final_decision_symbols": final_output["decisions"],
        "final_global_phase_rad": float(final_output["global_phase_rad"]),
    }


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return safe_load(handle)


def setup_logging(config: dict) -> None:
    configure_project_logging(
        level_name=get_configured_log_level(config),
        session_name="hardware_tester",
        console=True,
        file_output=True,
    )


def print_runtime_summary(config: dict, args: argparse.Namespace) -> None:
    print("Hardware test configuration")
    print(f"  radio_ip: {config['radio']['ip_address']}")
    print(f"  tx_carrier_hz: {int(float(config['transmitter']['tx_carrier']))}")
    print(f"  rx_carrier_hz: {int(float(config['receiver']['rx_carrier']))}")
    print(f"  sample_rate_hz: {int(float(config['modulation']['sample_rate']))}")
    print(f"  modulation: {normalize_config_modulation_name(config)}")
    print(f"  rx_buffer_size: {int(config['receiver']['buffer_size'])}")
    print(f"  flush_buffers: {args.flush_buffers}")
    print(f"  guard_symbols: {args.guard_symbols}")
    print(f"  payload_bytes: {len(args.payload.encode('utf-8'))}")
    print(f"  payload_capacity_bytes: {Datagram.PAYLOAD_SIZE}")
    print(f"  payload: {args.payload!r}")


def ensure_gold_config(config: dict) -> dict:
    cfg = deepcopy(config)
    gold_cfg = cfg.setdefault("gold_sequence", {})
    gold_cfg.setdefault("code_length", 31)
    gold_cfg.setdefault("code_index", 0)
    if "correlation_scale_factor_threshold" not in gold_cfg:
        gold_cfg["correlation_scale_factor_threshold"] = gold_cfg.get("correlation_threshold", 0.7)
    return cfg


def configure_hardware_tester(config: dict, rx_buffer_size: int | None) -> dict:
    cfg = ensure_gold_config(config)
    if rx_buffer_size is not None:
        cfg["receiver"]["buffer_size"] = int(rx_buffer_size)
    return cfg


def get_git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def save_plot_context(
    *,
    output_dir: Path,
    config: dict,
    args: argparse.Namespace,
    batch_runs: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "active_config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        safe_dump(config, handle, sort_keys=False)

    metadata = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "config_source_path": str(Path(args.config).resolve()),
        "active_config_path": str(config_path),
        "git_commit": get_git_commit_hash(),
        "plots_mode": args.plots,
        "batch_runs": int(batch_runs),
        "payload_text": args.payload,
        "payload_bytes": int(len(args.payload.encode("utf-8"))),
        "guard_symbols": int(args.guard_symbols),
        "flush_buffers": int(args.flush_buffers),
        "rx_buffer_size_override": (
            int(args.rx_buffer_size) if args.rx_buffer_size is not None else None
        ),
        "rx_buffer_size_effective": int(config["receiver"]["buffer_size"]),
        "sample_rate_hz": int(float(config["modulation"]["sample_rate"])),
        "samples_per_symbol": int(config["modulation"]["samples_per_symbol"]),
        "modulation": normalize_config_modulation_name(config),
    }
    metadata_path = output_dir / "run_metadata.yaml"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        safe_dump(metadata, handle, sort_keys=False)


def run_pipeline_test(
    modulation_name: str,
    fec: FCCodec,
    interleaver: Interleaver,
    conv_coder: ConvolutionalCoder,
    scrambler: LFSRScrambler,
    payload_text: str,
) -> tuple[Datagram, dict]:
    datagram = Datagram.as_string(payload_text, msg_type=msgType.DATA)
    encoded = encode_datagram(
        datagram=datagram,
        modulation_name=modulation_name,
        fec=fec,
        interleaver=interleaver,
        conv_coder=conv_coder,
        scrambler=scrambler,
    )

    try:
        recovered_datagram = decode_payload_symbols(
            payload_symbols=encoded.payload_symbols,
            modulation_name=modulation_name,
            fec=fec,
            interleaver=interleaver,
            conv_coder=conv_coder,
            scrambler=scrambler,
        )
        recovered_text = recovered_datagram.payload_text(trim_padding=True)
        roundtrip_ok = (
            recovered_datagram.get_msg_id == datagram.get_msg_id
            and recovered_datagram.get_msg_type == datagram.get_msg_type
            and recovered_datagram.get_payload_length == datagram.get_payload_length
            and np.array_equal(recovered_datagram.get_payload, datagram.get_payload)
        )
        error_message = ""
    except Exception as exc:
        recovered_text = ""
        roundtrip_ok = False
        error_message = str(exc)

    zeros, ones = bit_balance(encoded.scrambled_bits)
    changed = int(np.count_nonzero(encoded.scrambled_bits != encoded.fec_bits))

    result = {
        "scrambler_enabled": True,
        "payload_length_bytes": int(datagram.get_payload_length),
        "payload_capacity_bytes": Datagram.PAYLOAD_SIZE,
        "packed_datagram_bytes": Datagram.TOTAL_SIZE,
        "bit_count": int(encoded.fec_bits.size),
        "zero_bits": zeros,
        "one_bits": ones,
        "changed_bits": changed,
        "roundtrip_ok": roundtrip_ok,
        "recovered_text": recovered_text,
        "error": error_message,
    }
    return datagram, result


def run_gold_hardware_test(
    config: dict,
    modulation_name: str,
    gold_detector: GoldCodeDetector,
    rrc_filter: RRCFilter,
    fec: FCCodec,
    interleaver: Interleaver,
    conv_coder: ConvolutionalCoder,
    scrambler: LFSRScrambler,
    synchronizer: Synchronizer,
    sdr: SDRTransciever,
    datagram: Datagram,
    guard_symbols: int,
    flush_buffers: int,
) -> tuple[dict, dict]:
    tx_burst = build_tx_burst(
        config=config,
        datagram=datagram,
        modulation_name=modulation_name,
        samples_per_symbol=int(config["modulation"]["samples_per_symbol"]),
        gold_detector=gold_detector,
        rrc_filter=rrc_filter,
        fec=fec,
        interleaver=interleaver,
        conv_coder=conv_coder,
        scrambler=scrambler,
        guard_symbols=guard_symbols,
    )
    payload_symbols = tx_burst.encoded.payload_symbols
    channel_payload_symbols = tx_burst.channel_payload_symbols
    burst_symbols = tx_burst.burst_symbols
    tx_signal = tx_burst.tx_signal

    for _ in range(flush_buffers):
        sdr.sdr.rx()

    sdr.sdr.tx_destroy_buffer()
    sdr.sdr.tx(tx_signal)

    received_signal = sdr.sdr.rx()
    rx_state = run_rx_pipeline(
        config=config,
        modulation_name=modulation_name,
        received_signal=received_signal,
        payload_symbol_count=int(channel_payload_symbols.size),
        expected_tx_samples=int(tx_signal.size),
        guard_symbols=guard_symbols,
        gold_detector=gold_detector,
        rrc_filter=rrc_filter,
        synchronizer=synchronizer,
    )

    phase_candidates = rank_gold_candidates(
        rx_state.fine_signal,
        gold_detector,
        modulation_name,
        expected_index=rx_state.expected_header_index,
        search_radius=int(max(0, synchronizer.timing_header_search_radius_symbols)),
        top_k=int(max(1, synchronizer.timing_header_candidate_count)),
    )
    logger.debug(
        "Gold candidates ranked: count=%d expected_index=%s peaks=%s",
        int(len(phase_candidates)),
        str(rx_state.expected_header_index),
        [
            {
                "index": (
                    int(candidate["index"])
                    if candidate["index"] is not None
                    else None
                ),
                "peak": round(float(candidate["peak"]), 6),
            }
            for candidate in phase_candidates
        ],
    )

    gold_cfg = config.get("gold_sequence", {})
    decode_candidate_count = int(max(1, gold_cfg.get("decode_candidate_count", 2)))
    decode_candidate_min_peak = float(
        gold_cfg.get(
            "decode_candidate_min_peak",
            gold_detector.correlation_scale_factor_threshold,
        )
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
    decode_candidates_attempted = 0

    expected_header_index = (
        int(rx_state.expected_header_index)
        if rx_state.expected_header_index is not None
        else None
    )
    exact_index_candidates = []
    fallback_candidates = []
    seen_candidates: set[tuple[int, complex]] = set()

    for candidate in phase_candidates:
        candidate_index = candidate["index"]
        if candidate_index is None:
            continue
        if float(candidate["peak"]) < decode_candidate_min_peak:
            continue

        candidate_key = (int(candidate_index), complex(candidate["rotation"]))
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)

        if expected_header_index is not None and int(candidate_index) == expected_header_index:
            exact_index_candidates.append(candidate)
        else:
            fallback_candidates.append(candidate)

    exact_index_candidates.sort(key=lambda candidate: float(candidate["peak"]), reverse=True)
    fallback_candidates.sort(key=lambda candidate: float(candidate["peak"]), reverse=True)

    if exact_index_candidates:
        eligible_candidates = exact_index_candidates[:decode_candidate_count]
    else:
        eligible_candidates = fallback_candidates[:decode_candidate_count]
    logger.debug(
        "Decode candidate selection: min_peak=%.3f eligible=%s",
        float(decode_candidate_min_peak),
        [
            {
                "index": int(candidate["index"]),
                "peak": round(float(candidate["peak"]), 6),
                "rotation": str(candidate["rotation"]),
            }
            for candidate in eligible_candidates
        ],
    )

    if not eligible_candidates:
        decode_error = (
            f"Skipped decode attempts: no header candidates above peak threshold "
            f"{decode_candidate_min_peak:.3f}"
        )
        logger.warning("%s", decode_error)

    for candidate in eligible_candidates:
        candidate_index = candidate["index"]
        if candidate_index is None:
            continue

        required_symbols = (
            int(candidate_index)
            + int(gold_detector.gold_symbols.size)
            + int(channel_payload_symbols.size)
        )
        if required_symbols > int(candidate["decisions"].size):
            decode_error = "Skipped candidate: not enough symbols after detected header"
            logger.warning(
                "%s (candidate_index=%d required_symbols=%d available_symbols=%d)",
                decode_error,
                int(candidate_index),
                int(required_symbols),
                int(candidate["decisions"].size),
            )
            continue

        payload_rx = gold_detector.remove_gold_symbols(candidate["decisions"], candidate_index)
        # Gold removal returns the full suffix after the detected header. Trim it
        # back to the expected coded payload length before demod/decoding.
        payload_rx = trim_gold_removed_payload(payload_rx, payload_symbols.size)
        decode_candidates_attempted += 1
        logger.debug(
            "Attempting payload decode: candidate_index=%d candidate_peak=%.6f payload_symbols=%d attempt=%d",
            int(candidate_index),
            float(candidate["peak"]),
            int(payload_rx.size),
            int(decode_candidates_attempted),
        )

        # Demodulation + decoding pipeline
        try:
            recovered_datagram = decode_payload_symbols(
                payload_symbols=payload_rx,
                modulation_name=modulation_name,
                fec=fec,
                interleaver=interleaver,
                conv_coder=conv_coder,
                scrambler=scrambler,
            )
            recovered_text_candidate = recovered_datagram.payload_text(trim_padding=True)

            if (
                recovered_datagram.get_msg_id == datagram.get_msg_id
                and recovered_datagram.get_msg_type == datagram.get_msg_type
                and recovered_datagram.get_payload_length == datagram.get_payload_length
                and np.array_equal(recovered_datagram.get_payload, datagram.get_payload)
            ):
                recovered_ok = True
                recovered_text = recovered_text_candidate
                decode_rotation = candidate["rotation"]
                decode_error = ""

                start_index = int(candidate_index)
                peak = float(candidate["peak"])
                best_phase = int(candidate["phase"])
                detection_rotation = candidate["rotation"]
                decided = candidate["decisions"]
                logger.info(
                    "Hardware decode succeeded: detected_index=%d peak=%.6f decode_rotation=%s",
                    int(start_index),
                    float(peak),
                    str(decode_rotation),
                )
                break

        except Exception as exc:
            decode_error = str(exc)
            logger.debug(
                "Payload decode failed: candidate_index=%d peak=%.6f error=%s",
                int(candidate_index),
                float(candidate["peak"]),
                decode_error,
            )
            continue

    timing_symbol_error = compute_nearest_symbol_error(rx_state.timing_signal, modulation_name)
    carrier_aligned_symbol_error = compute_nearest_symbol_error(
        rx_state.carrier_aligned_signal,
        modulation_name,
    )
    equalized_symbol_error = compute_nearest_symbol_error(
        rx_state.equalized_signal,
        modulation_name,
    )
    fine_symbol_error = compute_nearest_symbol_error(
        rx_state.fine_signal,
        modulation_name,
    )
    carrier_projected_eye_signal = synchronizer.apply_symbol_rate_phase_correction_to_samples(
        rx_state.preamble_corrected_signal,
        phase_slope_rad_per_symbol=float(
            rx_state.carrier_correction.get("applied_slope_rad_per_symbol", 0.0)
        ),
        reference_symbol_index=int(rx_state.header_phase_symbol_index),
    )
    carrier_projected_eye_signal = (
        carrier_projected_eye_signal
        * np.exp(-1j * float(rx_state.carrier_correction.get("phase_rad", 0.0)))
    ).astype(np.complex64, copy=False)

    result = {
        "payload_length_bytes": int(datagram.get_payload_length),
        "rx_samples": int(received_signal.size),
        "sync_preamble_symbols": int(tx_burst.sync_preamble_symbols.size),
        "channel_payload_symbols": int(channel_payload_symbols.size),
        "sync_input_samples": int(rx_state.sync_input_signal.size),
        "sync_input_offset": int(rx_state.crop_start),
        "burst_crop_window_samples": int(rx_state.crop_window_samples),
        "burst_crop_margin_samples": int(rx_state.crop_margin_samples),
        "burst_crop_full_buffer_fallback": bool(rx_state.crop_full_buffer_fallback),
        "header_alignment_detected": bool(rx_state.header_alignment["detected"]),
        "header_alignment_applied": bool(rx_state.header_alignment.get("align_applied", False)),
        "header_alignment_soft_selected": bool(
            rx_state.header_alignment.get("soft_selected", False)
        ),
        "header_alignment_peak": float(rx_state.header_alignment["peak"]),
        "header_alignment_phase_score": float(
            rx_state.header_alignment.get("phase_score", 0.0)
        ),
        "header_alignment_peak_ratio": float(
            rx_state.header_alignment.get("peak_ratio", 0.0)
        ),
        "header_alignment_start_sample": (
            int(rx_state.header_alignment["header_start_sample"])
            if rx_state.header_alignment["header_start_sample"] is not None
            else None
        ),
        "header_alignment_offset_samples": int(rx_state.header_alignment["alignment_start_sample"]),
        "header_phase": int(rx_state.timing_acquisition["phase"]),
        "header_phase_score": float(rx_state.timing_acquisition["score"]),
        "header_phase_offset_samples": int(rx_state.timing_acquisition["header_offset_samples"]),
        "refined_header_start_sample": int(rx_state.timing_acquisition["refined_header_start_sample"]),
        "matched_rms_before": float(rx_state.matched_normalization["rms_before"]),
        "matched_normalization_scale": float(rx_state.matched_normalization["scale"]),
        "matched_normalization_selected_samples": int(
            rx_state.matched_normalization["selected_samples"]
        ),
        "fractional_timing_enabled": bool(
            rx_state.fractional_timing_alignment.get("enabled", False)
        ),
        "fractional_timing_applied": bool(
            rx_state.fractional_timing_alignment.get("applied", False)
        ),
        "fractional_timing_reason": rx_state.fractional_timing_alignment.get("reason", ""),
        "fractional_timing_sample_offset": float(
            rx_state.fractional_timing_alignment.get("sample_offset", 0.0)
        ),
        "fractional_timing_baseline_header_peak": float(
            rx_state.fractional_timing_alignment.get("baseline_header_peak", 0.0)
        ),
        "fractional_timing_baseline_header_error": float(
            rx_state.fractional_timing_alignment.get("baseline_header_error", 0.0)
        ),
        "fractional_timing_header_peak": float(
            rx_state.fractional_timing_alignment.get("header_peak", 0.0)
        ),
        "fractional_timing_header_error": float(
            rx_state.fractional_timing_alignment.get("header_error", 0.0)
        ),
        "fractional_timing_best_index": (
            int(rx_state.fractional_timing_alignment["best_index"])
            if rx_state.fractional_timing_alignment.get("best_index") is not None
            else None
        ),
        "timing_header_index": (
            int(rx_state.timing_header_index) if rx_state.timing_header_index is not None else None
        ),
        "timing_header_peak": float(rx_state.timing_header_peak),
        "timing_header_rotation": str(rx_state.timing_header_rotation),
        "timing_header_detect_ok": (
            int(rx_state.timing_header_index) == rx_state.expected_header_index
            if rx_state.timing_header_index is not None
            else False
        ),
        "timing_header_best_index": (
            int(rx_state.header_symbol_alignment["best_index"])
            if rx_state.header_symbol_alignment.get("best_index") is not None
            else None
        ),
        "timing_header_best_index_error": (
            int(rx_state.header_symbol_alignment["best_index"]) - int(rx_state.expected_header_index)
            if rx_state.header_symbol_alignment.get("best_index") is not None
            else None
        ),
        "timing_header_slip_symbols": int(
            rx_state.header_symbol_alignment.get("slip_symbols", 0)
        ),
        "timing_header_slip_applied": bool(
            rx_state.header_symbol_alignment.get("apply_slip", False)
        ),
        "timing_header_error": float(
            rx_state.header_symbol_alignment.get("header_error", 0.0)
        ),
        "timing_header_index_error": (
            int(rx_state.timing_header_index) - int(rx_state.expected_header_index)
            if rx_state.timing_header_index is not None
            else None
        ),
        "header_phase_symbol_index": int(rx_state.header_phase_symbol_index),
        "timing_window_start_sample": int(rx_state.timing_acquisition["aligned_window_start_sample"]),
        "timing_window_stop_sample": int(rx_state.timing_acquisition["aligned_window_stop_sample"]),
        "timing_window_samples": int(rx_state.timing_acquisition["window_samples"]),
        "timing_window_requested_symbols": int(rx_state.timing_acquisition["requested_symbols"]),
        "timing_window_available_symbols": int(rx_state.timing_acquisition["available_symbols"]),
        "timing_window_expected_preamble_index": int(
            rx_state.timing_acquisition["expected_preamble_symbol_index"]
        ),
        "timing_window_active_symbols": int(rx_state.timing_acquisition["active_symbol_count"]),
        "timing_window_tracking_payload_symbols": int(
            rx_state.timing_acquisition["tracking_payload_symbols"]
        ),
        "timing_window_tracking_symbols": int(
            rx_state.timing_acquisition["tracking_symbol_count"]
        ),
        "timing_window_post_symbols": int(
            rx_state.timing_acquisition["requested_symbols"]
            - min(int(guard_symbols), 2)
            - int(gold_detector.gold_symbols.size)
            - int(channel_payload_symbols.size)
        ),
        "gardner_update_start_sample": int(rx_state.gardner_traces["update_start_sample"]),
        "gardner_update_stop_sample": int(rx_state.gardner_traces["update_stop_sample"]),
        "gardner_gate_min_energy": float(rx_state.gardner_traces["gate_min_energy"]),
        "gardner_gate_max_energy": float(rx_state.gardner_traces["gate_max_energy"]),
        "gardner_update_rate": float(np.mean(rx_state.gardner_traces["updates"])) if rx_state.gardner_traces["updates"].size else 0.0,
        "gardner_mu_final": float(rx_state.gardner_traces["mu"][-1]) if rx_state.gardner_traces["mu"].size else 0.0,
        "gardner_omega_final": float(rx_state.gardner_traces["omega"][-1]) if rx_state.gardner_traces["omega"].size else 0.0,
        "preamble_cfo_enabled": bool(rx_state.preamble_coarse_estimate["enabled"]),
        "preamble_cfo_repeat_count": int(rx_state.preamble_coarse_estimate["repeat_count"]),
        "preamble_cfo_repeat_length_symbols": int(
            rx_state.preamble_coarse_estimate["repeat_length_symbols"]
        ),
        "preamble_cfo_symbol_index": int(
            rx_state.preamble_coarse_estimate["preamble_symbol_index"]
        ),
        "preamble_cfo_score": float(rx_state.preamble_coarse_estimate["score"]),
        "preamble_cfo_reason": rx_state.preamble_coarse_estimate.get("reason", ""),
        "preamble_cfo_phase_slope_rad_per_symbol": float(
            rx_state.preamble_coarse_estimate["phase_slope_rad_per_symbol"]
        ),
        "preamble_cfo_applied_slope_rad_per_symbol": float(
            rx_state.preamble_coarse_estimate["applied_phase_slope_rad_per_symbol"]
        ),
        "preamble_cfo_frequency_hz": float(rx_state.preamble_coarse_estimate["frequency_hz"]),
        "preamble_cfo_applied_frequency_hz": float(
            rx_state.preamble_coarse_estimate["applied_frequency_hz"]
        ),
        "preamble_symbol_best_index": (
            int(rx_state.preamble_symbol_alignment["best_index"])
            if rx_state.preamble_symbol_alignment.get("best_index") is not None
            else None
        ),
        "preamble_symbol_score": float(rx_state.preamble_symbol_alignment["score"]),
        "preamble_symbol_peak_ratio": float(
            rx_state.preamble_symbol_alignment.get("peak_ratio", 0.0)
        ),
        "preamble_symbol_slip_symbols": int(
            rx_state.preamble_symbol_alignment.get("slip_symbols", 0)
        ),
        "preamble_symbol_alignment_applied": bool(
            rx_state.preamble_symbol_alignment.get("applied", False)
        ),
        "preamble_symbol_alignment_reason": rx_state.preamble_symbol_alignment.get(
            "reason", ""
        ),
        "header_carrier_phase_rad": float(rx_state.header_carrier_estimate["phase_rad"]),
        "header_carrier_phase_slope_rad_per_symbol": float(
            rx_state.header_carrier_estimate["phase_slope_rad_per_symbol"]
        ),
        "header_carrier_frequency_hz": float(rx_state.header_carrier_estimate["frequency_hz"]),
        "header_carrier_score": float(rx_state.header_carrier_estimate["score"]),
        "header_carrier_applied_slope_rad_per_symbol": float(
            rx_state.carrier_correction["applied_slope_rad_per_symbol"]
        ),
        "header_carrier_applied_frequency_hz": float(
            rx_state.carrier_correction["applied_frequency_hz"]
        ),
        "header_carrier_slope_applied": bool(
            rx_state.carrier_correction["slope_applied"]
        ),
        "header_carrier_slope_reason": rx_state.carrier_correction["slope_reason"],
        "header_carrier_branch_selection_enabled": bool(
            rx_state.carrier_branch_evaluation["enabled"]
        ),
        "header_carrier_selected_branch": rx_state.carrier_branch_evaluation["selected_branch"],
        "header_carrier_selection_reason": rx_state.carrier_branch_evaluation["selection_reason"],
        "header_carrier_selected_detect_peak": float(
            rx_state.carrier_branch_evaluation["selected_detect_peak"]
        ),
        "header_carrier_selected_header_score": float(
            rx_state.carrier_branch_evaluation["selected_header_score"]
        ),
        "header_carrier_selected_header_error": float(
            rx_state.carrier_branch_evaluation.get("selected_header_error", 0.0)
        ),
        "header_carrier_phase_only_peak": float(
            rx_state.carrier_branch_evaluation.get("phase_only", {}).get("detect_peak", 0.0)
        ),
        "header_carrier_phase_only_score": float(
            rx_state.carrier_branch_evaluation.get("phase_only", {}).get("header_score", 0.0)
        ),
        "header_carrier_phase_only_error": float(
            rx_state.carrier_branch_evaluation.get("phase_only", {}).get("header_error", 0.0)
        ),
        "header_carrier_phase_slope_peak": float(
            rx_state.carrier_branch_evaluation.get("phase_slope", {}).get("detect_peak", 0.0)
        ),
        "header_carrier_phase_slope_score": float(
            rx_state.carrier_branch_evaluation.get("phase_slope", {}).get("header_score", 0.0)
        ),
        "header_carrier_phase_slope_error": float(
            rx_state.carrier_branch_evaluation.get("phase_slope", {}).get("header_error", 0.0)
        ),
        "short_equalizer_enabled": bool(rx_state.equalizer_state.get("enabled", False)),
        "short_equalizer_applied": bool(rx_state.equalizer_state.get("applied", False)),
        "short_equalizer_reason": rx_state.equalizer_state.get("reason", ""),
        "short_equalizer_tap_count": int(rx_state.equalizer_state.get("tap_count", 0)),
        "short_equalizer_training_symbols": int(
            rx_state.equalizer_state.get("training_symbol_count", 0)
        ),
        "short_equalizer_valid_training_symbols": int(
            rx_state.equalizer_state.get("valid_training_symbols", 0)
        ),
        "short_equalizer_before_error": float(
            rx_state.equalizer_state.get("before_error", 0.0)
        ),
        "short_equalizer_after_error": float(
            rx_state.equalizer_state.get("after_error", 0.0)
        ),
        "short_equalizer_before_mse": float(
            rx_state.equalizer_state.get("before_mse", 0.0)
        ),
        "short_equalizer_after_mse": float(
            rx_state.equalizer_state.get("after_mse", 0.0)
        ),
        "short_equalizer_mse_relative_improvement": float(
            rx_state.equalizer_state.get("mse_relative_improvement", 0.0)
        ),
        "short_equalizer_mse_relative_improvement_min": float(
            rx_state.equalizer_state.get("mse_relative_improvement_min", 0.0)
        ),
        "short_equalizer_peak_tolerance": float(
            rx_state.equalizer_state.get("peak_tolerance", 0.0)
        ),
        "short_equalizer_baseline_header_peak": float(
            rx_state.equalizer_state.get("baseline_header_peak", 0.0)
        ),
        "short_equalizer_baseline_header_error": float(
            rx_state.equalizer_state.get("baseline_header_error", 0.0)
        ),
        "short_equalizer_post_header_peak": float(
            rx_state.equalizer_state.get("post_header_peak", 0.0)
        ),
        "short_equalizer_post_header_error": float(
            rx_state.equalizer_state.get("post_header_error", 0.0)
        ),
        "short_equalizer_post_header_score": float(
            rx_state.equalizer_state.get("post_header_score", 0.0)
        ),
        "short_equalizer_header_peak_improved": bool(
            rx_state.equalizer_state.get("header_peak_improved", False)
        ),
        "short_equalizer_header_peak_within_tol": bool(
            rx_state.equalizer_state.get("header_peak_within_tol", False)
        ),
        "short_equalizer_header_error_improved": bool(
            rx_state.equalizer_state.get("header_error_improved", False)
        ),
        "short_equalizer_header_error_not_worse": bool(
            rx_state.equalizer_state.get("header_error_not_worse", False)
        ),
        "short_equalizer_accepted_via_header_error": bool(
            rx_state.equalizer_state.get("accepted_via_header_error", False)
        ),
        "short_equalizer_accepted_via_training_mse": bool(
            rx_state.equalizer_state.get("accepted_via_training_mse", False)
        ),
        "short_equalizer_header_gate_passed": bool(
            rx_state.equalizer_state.get("header_gate_passed", False)
        ),
        "short_equalizer_acceptance_reason": rx_state.equalizer_state.get(
            "acceptance_reason", ""
        ),
        "short_equalizer_rejection_reason": rx_state.equalizer_state.get(
            "rejection_reason", ""
        ),
        "short_equalizer_post_header_best_index": (
            int(rx_state.equalizer_state["post_header_best_index"])
            if rx_state.equalizer_state.get("post_header_best_index") is not None
            else None
        ),
        "costas_update_start_symbol": int(rx_state.costas_traces["update_start_symbol"]),
        "costas_update_stop_symbol": int(rx_state.costas_traces["update_stop_symbol"]),
        "costas_gate_reference_power": float(rx_state.costas_traces["gate_reference_power"]),
        "costas_gate_min_power": float(rx_state.costas_traces["gate_min_power"]),
        "costas_gate_max_power": float(rx_state.costas_traces["gate_max_power"]),
        "costas_update_rate": float(np.mean(rx_state.costas_traces["updates"])) if rx_state.costas_traces["updates"].size else 0.0,
        "timing_nearest_symbol_error": float(timing_symbol_error),
        "carrier_aligned_nearest_symbol_error": float(carrier_aligned_symbol_error),
        "equalized_nearest_symbol_error": float(equalized_symbol_error),
        "fine_nearest_symbol_error": float(fine_symbol_error),
        "tx_symbols": int(burst_symbols.size),
        "payload_symbols": int(payload_symbols.size),
        "gold_header_symbols": int(gold_detector.gold_symbols.size),
        "guard_symbols": int(guard_symbols),
        "detected": (
            start_index is not None
            and float(peak) >= float(gold_detector.correlation_scale_factor_threshold)
        ),
        "detected_index": start_index,
        "decode_candidate_count": int(decode_candidate_count),
        "decode_candidate_min_peak": float(decode_candidate_min_peak),
        "decode_candidates_ranked": int(len(phase_candidates)),
        "decode_candidates_attempted": int(decode_candidates_attempted),
        "expected_header_index": rx_state.expected_header_index,
        "timing_window_expected_header_index": rx_state.expected_header_index,
        "header_detect_ok": start_index == rx_state.expected_header_index,
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
        "sync_input_signal": rx_state.sync_input_signal,
        "frontend_filtered_signal": rx_state.frontend_filtered_signal,
        "coarse_signal": rx_state.coarse_signal,
        "matched_signal_raw": rx_state.matched_signal_raw,
        "matched_signal": rx_state.matched_signal,
        "header_aligned_signal": rx_state.header_aligned_signal,
        "timing_input_signal": rx_state.timing_input_signal,
        "preamble_corrected_signal": rx_state.preamble_corrected_signal,
        "timing_signal": rx_state.timing_signal,
        "carrier_aligned_signal": rx_state.carrier_aligned_signal,
        "carrier_projected_eye_signal": carrier_projected_eye_signal.real.astype(
            np.float32,
            copy=False,
        ),
        "equalized_signal": rx_state.equalized_signal,
        "gardner_error": rx_state.gardner_traces["error"],
        "gardner_mu": rx_state.gardner_traces["mu"],
        "gardner_updates": rx_state.gardner_traces["updates"],
        "gardner_omega": rx_state.gardner_traces["omega"],
        "fine_signal": rx_state.fine_signal,
        "fine_complex_signal": rx_state.costas_traces["corrected_signal"],
        "costas_error": rx_state.costas_traces["error"],
        "costas_phase_estimate": rx_state.costas_traces["phase_estimate"],
        "costas_updates": rx_state.costas_traces["updates"],
        "decided_symbols": decided,
    }
    logger.info(
        "Hardware test result: detected=%s header_detect_ok=%s recovered_ok=%s detected_index=%s peak=%.6f "
        "timing_error=%.6f carrier_error=%.6f equalized_error=%.6f fine_error=%.6f decode_attempts=%d error=%s",
        bool(result["detected"]),
        bool(result["header_detect_ok"]),
        bool(result["recovered_ok"]),
        str(result["detected_index"]),
        float(result["peak"]),
        float(result["timing_nearest_symbol_error"]),
        float(result["carrier_aligned_nearest_symbol_error"]),
        float(result["equalized_nearest_symbol_error"]),
        float(result["fine_nearest_symbol_error"]),
        int(result["decode_candidates_attempted"]),
        result["decode_error"],
    )
    return result, traces


def print_result_block(title: str, result: dict) -> None:
    print(title)
    for key, value in result.items():
        print(f"  {key}: {value}")


def save_hardware_test_plots(
    *,
    config: dict,
    traces: dict,
    output_dir: Path,
    show_plots: bool,
) -> None:
    import matplotlib.pyplot as plt
    from sdr_plots import StaticSDRPlotter

    plotter = StaticSDRPlotter()
    sample_rate = int(float(config["modulation"]["sample_rate"]))
    sps = int(config["modulation"]["samples_per_symbol"])
    symbol_rate = sample_rate / sps
    modulation_name = normalize_config_modulation_name(config)
    decision_traces = build_decision_diagnostic_traces(
        traces=traces,
        modulation_name=modulation_name,
    )
    # Merge the derived decision-axis views into the normal trace bundle so the
    # plotting code stays simple and every plot is generated from one source.
    traces = {**traces, **decision_traces}

    saved_paths: list[Path] = []
    def save_figure(stem: str, fig: object) -> None:
        if fig is None:
            return
        saved_paths.append(
            plotter.save_named_figure(
                fig=fig,
                output_dir=output_dir,
                stem=stem,
            )
        )
        if not show_plots:
            plt.close(fig)

    with plt.rc_context({"figure.max_open_warning": 0}):
        save_figure(
            "received_signal_psd",
            plotter.plot_psd(
                traces["received_signal"],
                sample_rate,
                title="Received Signal PSD",
            ),
        )
        save_figure(
            "matched_filter_output",
            plotter.plot_time_domain(
                traces["matched_signal"],
                sample_rate,
                title="Matched Filter Output",
            ),
        )
        save_figure(
            "frontend_filtered_signal_constellation",
            plotter.plot_constellation(
                traces["frontend_filtered_signal"],
                title="Front-End Filtered Signal Constellation",
            ),
        )
        save_figure(
            "coarse_frequency_corrected_constellation",
            plotter.plot_constellation(
                traces["coarse_signal"],
                title="Coarse Frequency Corrected Constellation",
            ),
        )
        save_figure(
            "preamble_coarse_corrected_constellation",
            plotter.plot_constellation(
                traces["preamble_corrected_signal"],
                title="Preamble Coarse Corrected Constellation",
            ),
        )
        save_figure(
            "timing_synchronized_constellation",
            plotter.plot_constellation(
                traces["timing_signal"],
                title="Timing Synchronized Constellation",
            ),
        )
        save_figure(
            "pre_costas_carrier_corrected_constellation",
            plotter.plot_constellation(
                traces["carrier_aligned_signal"],
                title="Pre-Costas Carrier Corrected Constellation",
            ),
        )
        save_figure(
            "short_trained_equalized_constellation",
            plotter.plot_constellation(
                traces["equalized_signal"],
                title="Short Trained Equalized Constellation",
            ),
        )
        save_figure(
            "fine_frequency_corrected_constellation",
            plotter.plot_constellation(
                traces["fine_signal"],
                title="Fine Frequency Corrected Constellation",
            ),
        )
        save_figure(
            "fine_frequency_corrected_complex_constellation",
            plotter.plot_constellation(
                traces["fine_complex_signal"],
                title="Fine Frequency Corrected Complex Constellation",
            ),
        )
        save_figure(
            "final_decision_sample_constellation",
            plotter.plot_constellation(
                traces["final_decision_signal"],
                title="Final Decision-Sample Constellation",
            ),
        )
        save_figure(
            "detected_symbol_decisions",
            plotter.plot_constellation(
                traces["decided_symbols"],
                title="Detected Symbol Decisions",
            ),
        )
        save_figure(
            "pre_equalizer_decision_sample_constellation",
            plotter.plot_constellation(
                traces["pre_equalizer_decision_signal"],
                title="Pre-Equalizer Decision-Sample Constellation",
            ),
        )
        save_figure(
            "post_equalizer_decision_sample_constellation",
            plotter.plot_constellation(
                traces["post_equalizer_decision_signal"],
                title="Post-Equalizer Decision-Sample Constellation",
            ),
        )
        save_figure(
            "preamble_coarse_corrected_eye_diagram",
            plotter.plot_eye_diagram(
                traces["preamble_corrected_signal"],
                sps,
                title="Preamble Coarse Corrected Eye Diagram",
            ),
        )
        save_figure(
            "carrier_aligned_projected_eye_diagram",
            plotter.plot_eye_diagram(
                traces["carrier_projected_eye_signal"],
                sps,
                title="Carrier Aligned Projected Eye Diagram",
            ),
        )
        save_figure(
            "pre_equalizer_projected_symbol_eye",
            plotter.plot_symbol_eye(
                traces["pre_equalizer_projected_signal"],
                title="Pre-Equalizer Projected Decision-Axis Eye",
            ),
        )
        save_figure(
            "post_equalizer_projected_symbol_eye",
            plotter.plot_symbol_eye(
                traces["post_equalizer_projected_signal"],
                title="Post-Equalizer Projected Decision-Axis Eye",
            ),
        )
        save_figure(
            "final_projected_symbol_eye",
            plotter.plot_symbol_eye(
                traces["final_projected_signal"],
                title="Final Projected Decision-Axis Eye",
            ),
        )
        save_figure(
            "final_projected_symbol_histogram",
            plotter.plot_histogram(
                traces["final_projected_signal"],
                title="Final Decision-Axis Symbol Histogram",
                xlabel="Projected BPSK Amplitude",
            ),
        )
        save_figure(
            "final_residual_phase_vs_symbol",
            plotter.plot_index_trace(
                traces["final_residual_phase_rad"],
                title="Residual Phase vs Symbol After Final Correction",
                xlabel="Symbol Index",
                ylabel="Residual Phase (rad)",
            ),
        )
        save_figure(
            "preamble_coarse_corrected_output",
            plotter.plot_time_domain(
                traces["preamble_corrected_signal"],
                sample_rate,
                title="Preamble Coarse Corrected Output",
            ),
        )
        save_figure(
            "timing_synchronized_output",
            plotter.plot_time_domain(
                traces["timing_signal"],
                symbol_rate,
                title="Timing Synchronized Output",
            ),
        )
        save_figure(
            "pre_costas_carrier_corrected_output",
            plotter.plot_time_domain(
                traces["carrier_aligned_signal"],
                symbol_rate,
                title="Pre-Costas Carrier Corrected Output",
            ),
        )
        save_figure(
            "short_trained_equalized_output",
            plotter.plot_time_domain(
                traces["equalized_signal"],
                symbol_rate,
                title="Short Trained Equalized Output",
            ),
        )
        save_figure(
            "fine_frequency_corrected_output",
            plotter.plot_time_domain(
                traces["fine_signal"],
                symbol_rate,
                title="Fine Frequency Corrected Output",
            ),
        )
        save_figure(
            "fine_frequency_corrected_complex_output",
            plotter.plot_time_domain(
                traces["fine_complex_signal"],
                symbol_rate,
                title="Fine Frequency Corrected Complex Output",
            ),
        )
        save_figure(
            "gardner_timing_error_trace",
            plotter.plot_scalar_trace(
                traces["gardner_error"],
                symbol_rate,
                title="Gardner Timing Error Trace",
                ylabel="Timing Error",
            ),
        )
        save_figure(
            "gardner_mu_trace",
            plotter.plot_scalar_trace(
                traces["gardner_mu"],
                symbol_rate,
                title="Gardner Mu Trace",
                ylabel="Fractional Timing Phase",
            ),
        )
        save_figure(
            "gardner_update_trace",
            plotter.plot_scalar_trace(
                traces["gardner_updates"],
                symbol_rate,
                title="Gardner Update Trace",
                ylabel="Update Applied",
            ),
        )
        save_figure(
            "gardner_omega_trace",
            plotter.plot_scalar_trace(
                traces["gardner_omega"],
                symbol_rate,
                title="Gardner Omega Trace",
                ylabel="Samples Per Symbol",
            ),
        )
        save_figure(
            "costas_loop_error_trace",
            plotter.plot_scalar_trace(
                traces["costas_error"],
                symbol_rate,
                title="Costas Loop Error Trace",
                ylabel="Carrier Error",
            ),
        )
        save_figure(
            "costas_loop_phase_estimate",
            plotter.plot_scalar_trace(
                traces["costas_phase_estimate"],
                symbol_rate,
                title="Costas Loop Phase Estimate",
                ylabel="Phase Estimate (rad)",
            ),
        )
        save_figure(
            "costas_update_trace",
            plotter.plot_scalar_trace(
                traces["costas_updates"],
                symbol_rate,
                title="Costas Update Trace",
                ylabel="Update Applied",
            ),
        )

    print(f"Saved plot files to: {output_dir}")
    for saved_path in saved_paths:
        print(f"  plot_file: {saved_path}")

    if show_plots:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scrambler and Gold-code hardware tests on Pluto.")
    parser.add_argument("--config", default="setup/config.yaml", help="Path to configuration file.")
    parser.add_argument("--payload", default="hardware test ", help="Payload text to transmit.")
    parser.add_argument(
        "--rx-buffer-size",
        type=int,
        default=None,
        help="Override RX buffer size used by the hardware tester. Defaults to receiver.buffer_size from config.",
    )
    parser.add_argument(
        "--guard-symbols",
        type=int,
        default=32,
        help="Zero guard symbols before and after the framed burst.",
    )
    parser.add_argument(
        "--flush-buffers",
        type=int,
        default=3,
        help="Number of RX buffers to discard before starting the transmission and capture.",
    )
    parser.add_argument(
        "--plots",
        nargs="?",
        const="show",
        choices=("show", "noshow"),
        default=None,
        help="Save diagnostic plots and optionally show them. Use `--plots` or `--plots show` to display, `--plots noshow` to save only.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of back-to-back hardware runs to execute in one invocation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    config = configure_hardware_tester(load_config(args.config), args.rx_buffer_size)
    setup_logging(config)
    print_runtime_summary(config, args)

    modulation_name = normalize_config_modulation_name(config)
    rrc_filter = RRCFilter(config)
    gold_detector = GoldCodeDetector(config)
    fec = FCCodec(config)
    interleaver = Interleaver(config)
    scrambler = build_scrambler(config)
    conv_coder = ConvolutionalCoder(config, warmup=False, use_numba=False)
    synchronizer = Synchronizer(config, warmup=False, use_numba=False)
    sdr = SDRTransciever(config)

    datagram, pipeline_result = run_pipeline_test(
        modulation_name=modulation_name,
        fec=fec,
        interleaver=interleaver,
        conv_coder=conv_coder,
        scrambler=scrambler,
        payload_text=args.payload,
    )
    frame_layout = build_frame_layout_summary(
        config=config,
        datagram=datagram,
        modulation_name=modulation_name,
        gold_detector=gold_detector,
        rrc_filter=rrc_filter,
        fec=fec,
        interleaver=interleaver,
        conv_coder=conv_coder,
        scrambler=scrambler,
        samples_per_symbol=int(config["modulation"]["samples_per_symbol"]),
        guard_symbols=args.guard_symbols,
    )
    print_result_block("Frame layout", frame_layout)
    print_result_block("TX/RX pipeline test", pipeline_result)
    if not pipeline_result["roundtrip_ok"]:
        return 1

    if not sdr.connect():
        print("Failed to connect to SDR. Exiting.")
        return 1

    try:
        noise_floor = sdr.measure_noise_floor_dB()
        if noise_floor is None:
            print("Noise floor measurement failed. Exiting.")
            return 1
        synchronizer.set_noise_floor(noise_floor)
        batch_root: Path | None = None
        if args.plots is not None and args.runs > 1:
            batch_root = (
                Path("hardware_tester_plots")
                / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_batch"
            )
            save_plot_context(
                output_dir=batch_root,
                config=config,
                args=args,
                batch_runs=args.runs,
            )

        overall_ok = True
        for run_index in range(args.runs):
            gold_result, traces = run_gold_hardware_test(
                config=config,
                modulation_name=modulation_name,
                gold_detector=gold_detector,
                rrc_filter=rrc_filter,
                fec=fec,
                interleaver=interleaver,
                conv_coder=conv_coder,
                scrambler=scrambler,
                synchronizer=synchronizer,
                sdr=sdr,
                datagram=datagram,
                guard_symbols=args.guard_symbols,
                flush_buffers=args.flush_buffers,
            )
            run_title = (
                f"Gold code hardware test [{run_index + 1}/{args.runs}]"
                if args.runs > 1
                else "Gold code hardware test"
            )
            print_result_block(run_title, gold_result)
            overall_ok = overall_ok and bool(gold_result["detected"] and gold_result["recovered_ok"])

            if args.plots is not None:
                if batch_root is not None:
                    plots_dir = batch_root / f"run_{run_index + 1:03d}"
                else:
                    plots_dir = (
                        Path("hardware_tester_plots")
                        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    )
                    save_plot_context(
                        output_dir=plots_dir,
                        config=config,
                        args=args,
                        batch_runs=args.runs,
                    )
                save_hardware_test_plots(
                    config=config,
                    traces=traces,
                    output_dir=plots_dir,
                    show_plots=(args.plots == "show"),
                )

        return 0 if overall_ok else 1
    finally:
        sdr.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
