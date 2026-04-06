from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from filter import BWLPFilter, RRCFilter
from gold_detection import GoldCodeDetector, detect_gold_with_rotation
from modulation import modulation_rotations, nearest_constellation_symbols
from project_logger import get_logger
from synchronize import Synchronizer

logger = get_logger(__name__)


@dataclass
class RxPipelineState:
    crop_start: int
    crop_window_samples: int
    crop_margin_samples: int
    crop_full_buffer_fallback: bool
    sync_input_signal: np.ndarray
    frontend_filtered_signal: np.ndarray
    coarse_signal: np.ndarray
    matched_signal_raw: np.ndarray
    matched_signal: np.ndarray
    matched_normalization: dict
    header_aligned_signal: np.ndarray
    header_alignment: dict
    timing_input_signal: np.ndarray
    timing_acquisition: dict
    preamble_coarse_estimate: dict
    preamble_corrected_signal: np.ndarray
    fractional_timing_alignment: dict
    preamble_symbol_alignment: dict
    timing_signal: np.ndarray
    gardner_traces: dict
    expected_header_index: int
    timing_header_index: int | None
    timing_header_peak: float
    timing_header_rotation: complex
    header_symbol_alignment: dict
    header_phase_symbol_index: int
    header_carrier_estimate: dict
    carrier_correction: dict
    carrier_branch_evaluation: dict
    carrier_aligned_signal: np.ndarray
    equalizer_state: dict
    equalized_signal: np.ndarray
    costas_update_start_symbol: int
    costas_update_stop_symbol: int
    fine_signal: np.ndarray
    costas_traces: dict


def maybe_apply_frontend_filter(config: dict, signal_in: np.ndarray) -> np.ndarray:
    if not bool(config["filter"].get("butterworth_enable", False)):
        return signal_in
    return BWLPFilter(config).apply_filter(signal_in)


def crop_signal_to_burst(
    received_signal: np.ndarray,
    expected_samples: int,
    margin_samples: int,
) -> tuple[np.ndarray, int]:
    if received_signal.size == 0:
        return received_signal, 0

    window_length = max(1, min(int(expected_samples), int(received_signal.size)))
    margin = max(0, int(margin_samples))
    if window_length >= received_signal.size:
        return received_signal, 0

    power = np.abs(received_signal).astype(np.float64) ** 2
    window = np.ones(window_length, dtype=np.float64)
    window_energy = np.convolve(power, window, mode="valid")
    start = int(np.argmax(window_energy))

    crop_start = max(0, start - margin)
    crop_end = min(received_signal.size, start + window_length + margin)
    return received_signal[crop_start:crop_end], crop_start


def _burst_crop_settings(
    config: dict,
    *,
    expected_tx_samples: int,
    sps: int,
    filter_span: int,
) -> tuple[bool, int, int, float, bool]:
    sync_cfg = config.get("synchronization", {})
    burst_crop_enable = bool(sync_cfg.get("burst_crop_enable", True))
    burst_crop_margin_symbols = int(
        sync_cfg.get("burst_crop_margin_symbols", max(2, filter_span))
    )
    burst_crop_window_scale = float(sync_cfg.get("burst_crop_window_scale", 1.0))
    burst_crop_min_expected_fraction = float(
        sync_cfg.get("burst_crop_min_expected_fraction", 0.9)
    )
    burst_crop_allow_full_buffer_fallback = bool(
        sync_cfg.get("burst_crop_allow_full_buffer_fallback", True)
    )

    margin_samples = max(0, burst_crop_margin_symbols * sps)
    window_samples = max(1, int(round(expected_tx_samples * max(burst_crop_window_scale, 1e-6))))

    return (
        burst_crop_enable,
        window_samples,
        margin_samples,
        burst_crop_min_expected_fraction,
        burst_crop_allow_full_buffer_fallback,
    )


def _phase_only_carrier_correction(config: dict, header_carrier_estimate: dict) -> dict:
    estimated_phase = float(header_carrier_estimate.get("phase_rad", 0.0))
    estimated_slope = float(
        header_carrier_estimate.get("phase_slope_rad_per_symbol", 0.0)
    )
    estimated_score = float(header_carrier_estimate.get("score", 0.0))
    return {
        "branch_name": "phase_only",
        "phase_rad": estimated_phase,
        "estimated_slope_rad_per_symbol": estimated_slope,
        "applied_slope_rad_per_symbol": 0.0,
        "estimated_frequency_hz": float(header_carrier_estimate.get("frequency_hz", 0.0)),
        "applied_frequency_hz": 0.0,
        "score": estimated_score,
        "slope_applied": False,
        "slope_reason": "phase_only",
        "score_gate_passed": True,
        "candidate_available": True,
    }


def _shift_symbol_stream(symbol_stream: np.ndarray, symbol_slip: int) -> np.ndarray:
    received = np.asarray(symbol_stream).astype(np.complex64, copy=False)
    slip = int(symbol_slip)
    if received.size == 0 or slip == 0:
        return received

    if abs(slip) >= received.size:
        return np.zeros_like(received)

    shifted = np.zeros_like(received)
    if slip > 0:
        shifted[:-slip] = received[slip:]
    else:
        shifted[-slip:] = received[: received.size + slip]
    return shifted


def _header_hypothesis_metrics(
    symbol_stream: np.ndarray,
    *,
    header_index: int,
    rotation: complex,
    reference_symbols: np.ndarray,
    modulation_name: str,
) -> dict | None:
    received = np.asarray(symbol_stream).astype(np.complex64, copy=False)
    reference = np.asarray(reference_symbols).astype(np.complex64, copy=False)
    start = int(header_index)
    stop = start + int(reference.size)
    if start < 0 or stop > received.size:
        return None

    rotated_symbols = received[start:stop] * rotation
    correlation = np.vdot(reference, rotated_symbols)
    ref_energy = float(np.vdot(reference, reference).real)
    rx_energy = float(np.vdot(rotated_symbols, rotated_symbols).real)
    corr_score = float(np.abs(correlation) / np.sqrt(max(ref_energy * rx_energy, 1e-12)))

    fitted_scale = correlation / max(ref_energy, 1e-12)
    phase_only_scale = np.exp(-1j * float(np.angle(fitted_scale)))
    normalized_symbols = rotated_symbols * phase_only_scale
    rms = float(np.sqrt(max(np.mean(np.abs(normalized_symbols) ** 2), 1e-12)))
    normalized_symbols = normalized_symbols / max(rms, 1e-12)

    header_error = float(np.mean(np.abs(normalized_symbols - reference)))
    decisions = nearest_constellation_symbols(normalized_symbols, modulation_name)
    decision_peak = float(np.abs(np.vdot(reference, decisions)) / max(float(reference.size), 1e-12))

    return {
        "index": int(start),
        "rotation": rotation,
        "correlation_score": float(corr_score),
        "header_error": float(header_error),
        "decision_peak": float(decision_peak),
    }


def _scan_header_hypotheses(
    symbol_stream: np.ndarray,
    *,
    reference_symbols: np.ndarray,
    modulation_name: str,
    expected_header_index: int,
    search_radius: int,
) -> dict:
    received = np.asarray(symbol_stream).astype(np.complex64, copy=False)
    reference_symbols = np.asarray(reference_symbols).astype(np.complex64, copy=False)
    expected_index = int(expected_header_index)
    reference_length = int(reference_symbols.size)
    if received.size < reference_length:
        return {
            "best_index": None,
            "detected_index": None,
            "slip_symbols": 0,
            "decision_peak": 0.0,
            "correlation_score": 0.0,
            "header_error": float("inf"),
            "rotation": 1 + 0j,
            "apply_slip": False,
        }

    radius = max(0, int(search_radius))
    max_index = int(received.size - reference_length)
    start = max(0, min(expected_index - radius, max_index))
    stop = max(0, min(expected_index + radius, max_index))

    best: dict | None = None
    for rotation in modulation_rotations(modulation_name):
        for candidate_index in range(start, stop + 1):
            metrics = _header_hypothesis_metrics(
                received,
                header_index=int(candidate_index),
                rotation=rotation,
                reference_symbols=reference_symbols,
                modulation_name=modulation_name,
            )
            if metrics is None:
                continue
            distance = abs(int(candidate_index) - expected_index)
            key = (
                float(metrics["header_error"]),
                -float(metrics["decision_peak"]),
                -float(metrics["correlation_score"]),
                float(distance),
            )
            if best is None or key < best["_sort_key"]:
                metrics["_sort_key"] = key
                best = metrics

    if best is None:
        return {
            "best_index": None,
            "detected_index": None,
            "slip_symbols": 0,
            "decision_peak": 0.0,
            "correlation_score": 0.0,
            "header_error": float("inf"),
            "rotation": 1 + 0j,
            "apply_slip": False,
        }

    best_index = int(best["index"])
    decision_peak = float(best["decision_peak"])
    return {
        "best_index": int(best_index),
        "detected_index": int(best_index),
        "slip_symbols": int(best_index - expected_index),
        "decision_peak": float(decision_peak),
        "correlation_score": float(best["correlation_score"]),
        "header_error": float(best["header_error"]),
        "rotation": best["rotation"],
        "apply_slip": bool(best_index != expected_index),
    }


def _scan_gold_hypotheses(
    symbol_stream: np.ndarray,
    *,
    gold_detector: GoldCodeDetector,
    modulation_name: str,
    expected_header_index: int,
    search_radius: int,
) -> dict:
    metrics = _scan_header_hypotheses(
        symbol_stream,
        reference_symbols=gold_detector.gold_symbols,
        modulation_name=modulation_name,
        expected_header_index=expected_header_index,
        search_radius=search_radius,
    )
    threshold = float(gold_detector.correlation_scale_factor_threshold)
    metrics["detected_index"] = (
        int(metrics["best_index"])
        if metrics["best_index"] is not None and float(metrics["decision_peak"]) >= threshold
        else None
    )
    return metrics


def _resample_symbol_stream(
    samples: np.ndarray,
    *,
    sps: int,
    sample_offset: float = 0.0,
    symbol_count: int | None = None,
) -> np.ndarray:
    received = np.asarray(samples).astype(np.complex64, copy=False)
    if received.size == 0:
        return np.array([], dtype=np.complex64)

    padded = np.pad(received, (2, 2), mode="constant").astype(np.complex64, copy=False)
    if symbol_count is None:
        max_symbol_count = int(np.floor((received.size - 1 - sample_offset) / max(1, int(sps)))) + 1
    else:
        max_symbol_count = int(symbol_count)
    max_symbol_count = max(0, max_symbol_count)
    out = np.empty(max_symbol_count, dtype=np.complex64)

    valid_count = 0
    for idx in range(max_symbol_count):
        position = float(sample_offset) + float(idx * sps)
        shifted_position = position + 2.0
        if shifted_position < 1.0 or shifted_position > float(received.size + 1):
            break
        out[valid_count] = np.complex64(
            synchronizer_interp_cubic(padded, shifted_position)
        )
        valid_count += 1
    return out[:valid_count]


def synchronizer_interp_cubic(samples: np.ndarray, position: float) -> np.complex64:
    i1 = int(np.floor(position))
    mu = float(position - i1)
    y0 = samples[i1 - 1]
    y1 = samples[i1]
    y2 = samples[i1 + 1]
    y3 = samples[i1 + 2]
    a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    a2 = -0.5 * y0 + 0.5 * y2
    a3 = y1
    return np.complex64(((a0 * mu + a1) * mu + a2) * mu + a3)


def _refine_fractional_header_timing(
    sample_window: np.ndarray,
    *,
    sps: int,
    baseline_symbol_stream: np.ndarray,
    gold_detector: GoldCodeDetector,
    modulation_name: str,
    expected_header_index: int,
    search_radius: int,
    max_offset_samples: float,
    step_samples: float,
) -> tuple[np.ndarray, dict]:
    baseline_metrics = _scan_gold_hypotheses(
        baseline_symbol_stream,
        gold_detector=gold_detector,
        modulation_name=modulation_name,
        expected_header_index=expected_header_index,
        search_radius=search_radius,
    )
    baseline_error = float(baseline_metrics["header_error"])
    baseline_peak = float(baseline_metrics["decision_peak"])
    best_signal = baseline_symbol_stream
    best_metrics = baseline_metrics
    best_offset = 0.0

    span = max(0.0, float(max_offset_samples))
    step = max(0.125, float(step_samples))
    if span <= 0.0:
        return best_signal, {
            "enabled": False,
            "applied": False,
            "reason": "disabled",
            "sample_offset": 0.0,
            "baseline_header_error": baseline_error,
            "baseline_header_peak": baseline_peak,
            "header_error": baseline_error,
            "header_peak": baseline_peak,
            "best_index": baseline_metrics.get("best_index"),
        }

    offset = -span
    while offset <= span + 1e-9:
        if abs(offset) <= 1e-9:
            offset += step
            continue
        candidate_signal = _resample_symbol_stream(
            sample_window,
            sps=sps,
            sample_offset=offset,
            symbol_count=int(baseline_symbol_stream.size),
        )
        if candidate_signal.size < gold_detector.gold_symbols.size:
            offset += step
            continue
        candidate_metrics = _scan_gold_hypotheses(
            candidate_signal,
            gold_detector=gold_detector,
            modulation_name=modulation_name,
            expected_header_index=expected_header_index,
            search_radius=search_radius,
        )
        candidate_key = (
            float(candidate_metrics["header_error"]),
            -float(candidate_metrics["decision_peak"]),
        )
        best_key = (
            float(best_metrics["header_error"]),
            -float(best_metrics["decision_peak"]),
        )
        if candidate_key < best_key:
            best_signal = candidate_signal
            best_metrics = candidate_metrics
            best_offset = float(offset)
        offset += step

    applied = bool(abs(best_offset) > 1e-9 and (
        float(best_metrics["header_error"]) < (baseline_error - 1e-6)
        or float(best_metrics["decision_peak"]) > (baseline_peak + 1e-6)
    ))
    if not applied:
        best_signal = baseline_symbol_stream
        best_metrics = baseline_metrics
        best_offset = 0.0

    return best_signal, {
        "enabled": True,
        "applied": bool(applied),
        "reason": "applied" if applied else "not_improved",
        "sample_offset": float(best_offset),
        "baseline_header_error": baseline_error,
        "baseline_header_peak": baseline_peak,
        "header_error": float(best_metrics["header_error"]),
        "header_peak": float(best_metrics["decision_peak"]),
        "best_index": (
            int(best_metrics["best_index"])
            if best_metrics.get("best_index") is not None
            else None
        ),
    }


def _phase_slope_carrier_correction(config: dict, header_carrier_estimate: dict) -> dict:
    sync_cfg = config.get("synchronization", {})
    apply_slope_enable = bool(sync_cfg.get("header_carrier_apply_slope_enable", True))
    slope_min_score = float(sync_cfg.get("header_carrier_slope_min_score", 0.55))
    max_abs_slope = float(sync_cfg.get("header_carrier_max_abs_slope_rad_per_symbol", 0.05))

    estimated_phase = float(header_carrier_estimate.get("phase_rad", 0.0))
    estimated_slope = float(
        header_carrier_estimate.get("phase_slope_rad_per_symbol", 0.0)
    )
    estimated_score = float(header_carrier_estimate.get("score", 0.0))
    applied_slope = 0.0
    candidate_available = False
    slope_reason = "disabled"

    if not apply_slope_enable:
        slope_reason = "disabled"
    elif not np.isfinite(estimated_slope):
        slope_reason = "non_finite"
    else:
        candidate_available = True
        if max_abs_slope > 0.0:
            applied_slope = float(np.clip(estimated_slope, -max_abs_slope, max_abs_slope))
            slope_reason = "clamped" if not np.isclose(applied_slope, estimated_slope) else "applied"
        else:
            applied_slope = float(estimated_slope)
            slope_reason = "applied"

    return {
        "branch_name": "phase_slope",
        "phase_rad": estimated_phase,
        "estimated_slope_rad_per_symbol": estimated_slope,
        "applied_slope_rad_per_symbol": float(applied_slope),
        "estimated_frequency_hz": float(header_carrier_estimate.get("frequency_hz", 0.0)),
        "applied_frequency_hz": float(
            applied_slope
            * float(config["modulation"]["symbol_rate"])
            / (2.0 * np.pi)
        ),
        "score": estimated_score,
        "slope_applied": bool(candidate_available and abs(applied_slope) > 0.0),
        "slope_reason": slope_reason,
        "score_gate_passed": bool(estimated_score >= slope_min_score),
        "candidate_available": bool(candidate_available),
        "slope_min_score": slope_min_score,
        "max_abs_slope_rad_per_symbol": max_abs_slope,
    }


def _evaluate_carrier_branch(
    corrected_signal: np.ndarray,
    *,
    gold_detector: GoldCodeDetector,
    modulation_name: str,
    expected_header_index: int,
) -> dict:
    header_metrics = _scan_gold_hypotheses(
        corrected_signal,
        gold_detector=gold_detector,
        modulation_name=modulation_name,
        expected_header_index=int(expected_header_index),
        search_radius=0,
    )
    return {
        "branch_detect_index": (
            int(header_metrics["detected_index"])
            if header_metrics["detected_index"] is not None
            else None
        ),
        "branch_best_index": (
            int(header_metrics["best_index"])
            if header_metrics["best_index"] is not None
            else None
        ),
        "branch_detect_peak": float(header_metrics["decision_peak"]),
        "branch_detect_rotation": header_metrics["rotation"],
        "branch_header_score": float(header_metrics["correlation_score"]),
        "branch_header_error": float(header_metrics["header_error"]),
    }


def _select_header_carrier_correction(
    *,
    config: dict,
    synchronizer: Synchronizer,
    gold_detector: GoldCodeDetector,
    modulation_name: str,
    timing_signal: np.ndarray,
    header_carrier_estimate: dict,
    expected_header_index: int,
    header_phase_symbol_index: int,
) -> tuple[dict, np.ndarray, dict]:
    sync_cfg = config.get("synchronization", {})
    branch_selection_enable = bool(
        sync_cfg.get("header_carrier_branch_selection_enable", True)
    )

    phase_only = _phase_only_carrier_correction(config, header_carrier_estimate)
    phase_slope = _phase_slope_carrier_correction(config, header_carrier_estimate)
    candidates = [phase_only]
    if phase_slope["candidate_available"]:
        candidates.append(phase_slope)

    branch_signals: dict[str, np.ndarray] = {}
    branch_metrics: dict[str, dict] = {}
    for candidate in candidates:
        corrected_signal = synchronizer.apply_carrier_phase_and_frequency_correction(
            timing_signal,
            phase_rad=float(candidate["phase_rad"]),
            phase_slope_rad_per_symbol=float(candidate["applied_slope_rad_per_symbol"]),
            reference_symbol_index=int(header_phase_symbol_index),
        )
        branch_signals[candidate["branch_name"]] = corrected_signal
        metrics = _evaluate_carrier_branch(
            corrected_signal,
            gold_detector=gold_detector,
            modulation_name=modulation_name,
            expected_header_index=int(expected_header_index),
        )
        candidate.update(metrics)
        branch_metrics[candidate["branch_name"]] = {
            "candidate_available": bool(candidate["candidate_available"]),
            "score_gate_passed": bool(candidate["score_gate_passed"]),
            "applied_slope_rad_per_symbol": float(candidate["applied_slope_rad_per_symbol"]),
            "detect_index": candidate["branch_detect_index"],
            "best_index": candidate["branch_best_index"],
            "detect_peak": float(candidate["branch_detect_peak"]),
            "header_score": float(candidate["branch_header_score"]),
            "header_error": float(candidate["branch_header_error"]),
            "reason": candidate["slope_reason"],
        }

    if branch_selection_enable and len(candidates) > 1:
        selected = min(
            candidates,
            key=lambda candidate: (
                float(candidate["branch_header_error"]),
                -float(candidate["branch_detect_peak"]),
                0 if candidate["branch_name"] == "phase_only" else 1,
            ),
        )
        selection_reason = "best_header_error"
    elif phase_slope["candidate_available"] and phase_slope["score_gate_passed"]:
        selected = phase_slope
        selection_reason = "score_gate"
    else:
        selected = phase_only
        selection_reason = "phase_only"

    selected["selection_reason"] = selection_reason
    evaluation = {
        "enabled": bool(branch_selection_enable),
        "selected_branch": selected["branch_name"],
        "selection_reason": selection_reason,
        "selected_detect_peak": float(selected["branch_detect_peak"]),
        "selected_header_score": float(selected["branch_header_score"]),
        "selected_header_error": float(selected["branch_header_error"]),
        "phase_only": branch_metrics.get("phase_only", {}),
        "phase_slope": branch_metrics.get("phase_slope", {}),
    }
    return selected, branch_signals[selected["branch_name"]], evaluation


def run_rx_pipeline(
    *,
    config: dict,
    modulation_name: str,
    received_signal: np.ndarray,
    expected_tx_samples: int,
    payload_symbol_count: int,
    guard_symbols: int,
    gold_detector: GoldCodeDetector,
    rrc_filter: RRCFilter,
    synchronizer: Synchronizer,
) -> RxPipelineState:
    sps = int(config["modulation"]["samples_per_symbol"])
    filter_span = int(config["filter"].get("rrc_filter_span", 10))
    (
        burst_crop_enable,
        crop_window_samples,
        crop_margin_samples,
        burst_crop_min_expected_fraction,
        burst_crop_allow_full_buffer_fallback,
    ) = _burst_crop_settings(
        config,
        expected_tx_samples=int(expected_tx_samples),
        sps=sps,
        filter_span=filter_span,
    )

    crop_full_buffer_fallback = False
    if burst_crop_enable:
        sync_input_signal, crop_start = crop_signal_to_burst(
            received_signal=received_signal,
            expected_samples=int(crop_window_samples),
            margin_samples=int(crop_margin_samples),
        )
        min_required_samples = int(
            max(1, round(float(expected_tx_samples) * burst_crop_min_expected_fraction))
        )
        if (
            burst_crop_allow_full_buffer_fallback
            and sync_input_signal.size < min_required_samples
        ):
            sync_input_signal = received_signal
            crop_start = 0
            crop_full_buffer_fallback = True
    else:
        sync_input_signal = received_signal
        crop_start = 0

    logger.debug(
        "RX pipeline crop: enabled=%s window_samples=%d margin_samples=%d crop_start=%d "
        "sync_input_samples=%d full_buffer_fallback=%s",
        burst_crop_enable,
        int(crop_window_samples),
        int(crop_margin_samples),
        int(crop_start),
        int(sync_input_signal.size),
        bool(crop_full_buffer_fallback),
    )

    frontend_filtered_signal = maybe_apply_frontend_filter(config, sync_input_signal)
    coarse_signal = synchronizer.coarse_frequenzy_synchronization(frontend_filtered_signal)
    if coarse_signal is None:
        raise RuntimeError("Coarse frequency sync failed (signal too weak)")

    matched_signal_raw = rrc_filter.apply_filter(coarse_signal)
    matched_signal, matched_normalization = synchronizer.normalize_matched_filter_output(
        matched_signal_raw
    )
    logger.debug(
        "Matched filter normalization: rms_before=%.6f scale=%.6e selected_samples=%d",
        float(matched_normalization["rms_before"]),
        float(matched_normalization["scale"]),
        int(matched_normalization["selected_samples"]),
    )

    header_preroll_symbols = min(int(guard_symbols), 2)
    header_aligned_signal, header_alignment = synchronizer.align_to_header(
        matched_signal,
        pre_roll_samples=header_preroll_symbols * sps,
    )
    logger.debug(
        "Header alignment: detected=%s align_applied=%s soft_selected=%s peak=%.6f "
        "phase_score=%.6f peak_ratio=%.6f header_start_sample=%s alignment_start_sample=%d",
        bool(header_alignment["detected"]),
        bool(header_alignment.get("align_applied", False)),
        bool(header_alignment.get("soft_selected", False)),
        float(header_alignment["peak"]),
        float(header_alignment.get("phase_score", 0.0)),
        float(header_alignment.get("peak_ratio", 0.0)),
        (
            str(int(header_alignment["header_start_sample"]))
            if header_alignment["header_start_sample"] is not None
            else "None"
        ),
        int(header_alignment["alignment_start_sample"]),
    )

    timing_pre_symbols = min(int(guard_symbols), 2)
    timing_post_symbols = max(min(int(guard_symbols), 64), 4 * filter_span)
    timing_input_signal, timing_acquisition = synchronizer.acquire_header_timing_window(
        header_aligned_signal,
        header_start_sample=int(header_alignment["header_index_samples"]),
        payload_symbol_count=int(payload_symbol_count),
        pre_symbols=timing_pre_symbols,
        post_symbols=timing_post_symbols,
    )
    logger.debug(
        "Timing acquisition: phase=%d score=%.6f offset_samples=%d refined_start=%d "
        "window_start=%d window_stop=%d available_symbols=%d expected_preamble_index=%d expected_header_index=%d",
        int(timing_acquisition["phase"]),
        float(timing_acquisition["score"]),
        int(timing_acquisition["header_offset_samples"]),
        int(timing_acquisition["refined_header_start_sample"]),
        int(timing_acquisition["aligned_window_start_sample"]),
        int(timing_acquisition["aligned_window_stop_sample"]),
        int(timing_acquisition["available_symbols"]),
        int(timing_acquisition["expected_preamble_symbol_index"]),
        int(timing_acquisition["expected_header_symbol_index"]),
    )

    preamble_coarse_estimate = synchronizer.estimate_repeated_preamble_cfo(
        timing_input_signal,
        preamble_symbol_index=int(timing_acquisition["expected_preamble_symbol_index"]),
    )
    logger.debug(
        "Preamble coarse CFO: enabled=%s score=%.6f slope=%.6f applied_slope=%.6f "
        "freq_hz=%.6f applied_freq_hz=%.6f repeats=%d reason=%s",
        bool(preamble_coarse_estimate["enabled"]),
        float(preamble_coarse_estimate["score"]),
        float(preamble_coarse_estimate["phase_slope_rad_per_symbol"]),
        float(preamble_coarse_estimate["applied_phase_slope_rad_per_symbol"]),
        float(preamble_coarse_estimate["frequency_hz"]),
        float(preamble_coarse_estimate["applied_frequency_hz"]),
        int(preamble_coarse_estimate["repeat_count"]),
        str(preamble_coarse_estimate.get("reason", "")),
    )
    preamble_corrected_signal = synchronizer.apply_symbol_rate_phase_correction_to_samples(
        timing_input_signal,
        phase_slope_rad_per_symbol=float(
            preamble_coarse_estimate["applied_phase_slope_rad_per_symbol"]
        ),
        reference_symbol_index=int(timing_acquisition["expected_preamble_symbol_index"]),
    )

    # From this point on we stay in a narrow, evidence-backed receive path:
    # preamble CFO -> Gardner timing -> fractional timing tweak -> symbol-slip
    # correction -> header carrier correction -> short equalizer -> Costas.
    timing_signal, gardner_traces = synchronizer.gardner_timing_synchronization_with_traces(
        preamble_corrected_signal,
        update_start_sample=int(timing_acquisition["update_start_sample"]),
        update_stop_sample=int(timing_acquisition["update_stop_sample"]),
    )
    logger.debug(
        "Gardner tracking: update_start=%d update_stop=%d update_rate=%.6f mu_final=%.6f omega_final=%.6f",
        int(gardner_traces["update_start_sample"]),
        int(gardner_traces["update_stop_sample"]),
        float(np.mean(gardner_traces["updates"])) if gardner_traces["updates"].size else 0.0,
        float(gardner_traces["mu"][-1]) if gardner_traces["mu"].size else 0.0,
        float(gardner_traces["omega"][-1]) if gardner_traces["omega"].size else 0.0,
    )

    fractional_timing_alignment = {
        "enabled": False,
        "applied": False,
        "reason": "disabled",
        "sample_offset": 0.0,
        "baseline_header_error": 0.0,
        "baseline_header_peak": 0.0,
        "header_error": 0.0,
        "header_peak": 0.0,
        "best_index": None,
    }
    if bool(synchronizer.header_fractional_timing_enable):
        timing_signal, fractional_timing_alignment = _refine_fractional_header_timing(
            preamble_corrected_signal,
            sps=int(sps),
            baseline_symbol_stream=timing_signal,
            gold_detector=gold_detector,
            modulation_name=modulation_name,
            expected_header_index=int(timing_acquisition["expected_header_symbol_index"]),
            search_radius=int(max(0, synchronizer.timing_header_search_radius_symbols)),
            max_offset_samples=float(synchronizer.header_fractional_timing_max_offset_samples),
            step_samples=float(synchronizer.header_fractional_timing_step_samples),
        )
    logger.debug(
        "Fractional header timing: enabled=%s applied=%s reason=%s sample_offset=%.3f "
        "baseline_peak=%.6f baseline_error=%.6f peak=%.6f error=%.6f best_index=%s",
        bool(fractional_timing_alignment.get("enabled", False)),
        bool(fractional_timing_alignment.get("applied", False)),
        str(fractional_timing_alignment.get("reason", "")),
        float(fractional_timing_alignment.get("sample_offset", 0.0)),
        float(fractional_timing_alignment.get("baseline_header_peak", 0.0)),
        float(fractional_timing_alignment.get("baseline_header_error", 0.0)),
        float(fractional_timing_alignment.get("header_peak", 0.0)),
        float(fractional_timing_alignment.get("header_error", 0.0)),
        (
            str(int(fractional_timing_alignment["best_index"]))
            if fractional_timing_alignment.get("best_index") is not None
            else "None"
        ),
    )

    preamble_symbol_alignment = synchronizer.detect_repeated_preamble_symbol_index(
        timing_signal,
        expected_preamble_symbol_index=int(
            timing_acquisition["expected_preamble_symbol_index"]
        ),
    )
    logger.debug(
        "Preamble symbol alignment: expected_index=%d best_index=%s slip=%d score=%.6f "
        "peak_ratio=%.6f applied=%s reason=%s",
        int(preamble_symbol_alignment["expected_index"]),
        (
            str(int(preamble_symbol_alignment["best_index"]))
            if preamble_symbol_alignment["best_index"] is not None
            else "None"
        ),
        int(preamble_symbol_alignment["slip_symbols"]),
        float(preamble_symbol_alignment["score"]),
        float(preamble_symbol_alignment["peak_ratio"]),
        bool(preamble_symbol_alignment["applied"]),
        str(preamble_symbol_alignment["reason"]),
    )
    if bool(preamble_symbol_alignment["applied"]):
        timing_signal = _shift_symbol_stream(
            timing_signal,
            int(preamble_symbol_alignment["slip_symbols"]),
        )

    expected_header_index = int(timing_acquisition["expected_header_symbol_index"])
    header_symbol_alignment = _scan_gold_hypotheses(
        timing_signal,
        gold_detector=gold_detector,
        modulation_name=modulation_name,
        expected_header_index=expected_header_index,
        search_radius=int(max(0, synchronizer.timing_header_search_radius_symbols)),
    )
    timing_header_index = header_symbol_alignment["detected_index"]
    timing_header_peak = float(header_symbol_alignment["decision_peak"])
    timing_header_rotation = header_symbol_alignment["rotation"]
    logger.debug(
        "Timing header detect: expected_index=%d best_index=%s detected_index=%s slip=%d "
        "peak=%.6f header_error=%.6f rotation=%s",
        int(expected_header_index),
        (
            str(int(header_symbol_alignment["best_index"]))
            if header_symbol_alignment["best_index"] is not None
            else "None"
        ),
        str(int(timing_header_index)) if timing_header_index is not None else "None",
        int(header_symbol_alignment["slip_symbols"]),
        float(timing_header_peak),
        float(header_symbol_alignment["header_error"]),
        str(timing_header_rotation),
    )
    if bool(header_symbol_alignment["apply_slip"]):
        timing_signal = _shift_symbol_stream(
            timing_signal,
            int(header_symbol_alignment["slip_symbols"]),
        )
    header_phase_symbol_index = int(expected_header_index)

    header_carrier_estimate = synchronizer.estimate_header_carrier_state(
        timing_signal,
        header_symbol_index=header_phase_symbol_index,
    )
    logger.debug(
        "Header carrier estimate: symbol_index=%d phase_rad=%.6f slope=%.6f freq_hz=%.6f score=%.6f",
        int(header_phase_symbol_index),
        float(header_carrier_estimate["phase_rad"]),
        float(header_carrier_estimate["phase_slope_rad_per_symbol"]),
        float(header_carrier_estimate["frequency_hz"]),
        float(header_carrier_estimate["score"]),
    )
    carrier_correction, carrier_aligned_signal, carrier_branch_evaluation = _select_header_carrier_correction(
        config=config,
        synchronizer=synchronizer,
        gold_detector=gold_detector,
        modulation_name=modulation_name,
        timing_signal=timing_signal,
        header_carrier_estimate=header_carrier_estimate,
        expected_header_index=expected_header_index,
        header_phase_symbol_index=header_phase_symbol_index,
    )
    logger.debug(
        "Carrier correction applied: phase_rad=%.6f estimated_slope=%.6f applied_slope=%.6f "
        "estimated_freq_hz=%.6f applied_freq_hz=%.6f score=%.6f slope_applied=%s reason=%s "
        "selected_branch=%s selection_reason=%s selected_peak=%.6f",
        float(carrier_correction["phase_rad"]),
        float(carrier_correction["estimated_slope_rad_per_symbol"]),
        float(carrier_correction["applied_slope_rad_per_symbol"]),
        float(carrier_correction["estimated_frequency_hz"]),
        float(carrier_correction["applied_frequency_hz"]),
        float(carrier_correction["score"]),
        bool(carrier_correction["slope_applied"]),
        carrier_correction["slope_reason"],
        carrier_branch_evaluation["selected_branch"],
        carrier_branch_evaluation["selection_reason"],
        float(carrier_branch_evaluation["selected_detect_peak"]),
    )

    equalized_signal, equalizer_state = synchronizer.train_and_apply_symbol_equalizer(
        carrier_aligned_signal,
        training_symbol_index=int(timing_acquisition["expected_preamble_symbol_index"]),
    )
    equalizer_header_metrics = _scan_gold_hypotheses(
        equalized_signal,
        gold_detector=gold_detector,
        modulation_name=modulation_name,
        expected_header_index=expected_header_index,
        search_radius=0,
    )
    equalizer_state["post_header_peak"] = float(equalizer_header_metrics["decision_peak"])
    equalizer_state["post_header_error"] = float(equalizer_header_metrics["header_error"])
    equalizer_state["post_header_score"] = float(equalizer_header_metrics["correlation_score"])
    equalizer_state["post_header_best_index"] = (
        int(equalizer_header_metrics["best_index"])
        if equalizer_header_metrics["best_index"] is not None
        else None
    )
    baseline_header_peak = float(carrier_branch_evaluation["selected_detect_peak"])
    baseline_header_error = float(carrier_branch_evaluation["selected_header_error"])
    peak_tol = 1.0e-3
    mse_rel_improve_min = 0.05
    before_mse = float(equalizer_state.get("before_mse", 0.0))
    after_mse = float(equalizer_state.get("after_mse", 0.0))
    mse_relative_improvement = float(
        max(0.0, before_mse - after_mse) / max(before_mse, 1.0e-12)
    )
    header_peak_improved = float(equalizer_state["post_header_peak"]) > (baseline_header_peak + 1e-6)
    header_error_improved = float(equalizer_state["post_header_error"]) < (baseline_header_error - 1e-6)
    header_peak_within_tol = float(equalizer_state["post_header_peak"]) >= (baseline_header_peak - peak_tol)
    header_error_not_worse = float(equalizer_state["post_header_error"]) <= (baseline_header_error + 1e-12)
    accept_via_header_error = bool(header_error_improved and header_peak_within_tol)
    accept_via_training_mse = bool(
        mse_relative_improvement >= mse_rel_improve_min
        and header_error_not_worse
        and header_peak_within_tol
    )
    header_gate_passed = bool(accept_via_header_error or accept_via_training_mse)
    acceptance_reason = ""
    rejection_reason = ""
    if header_gate_passed:
        if accept_via_header_error:
            acceptance_reason = "header_error_improved_peak_within_tol"
        else:
            acceptance_reason = "mse_rel_improved_header_not_worse_peak_within_tol"
    else:
        rejection_parts: list[str] = []
        if not header_peak_within_tol:
            rejection_parts.append("header_peak_dropped")
        if not header_error_not_worse:
            rejection_parts.append("header_error_worsened")
        elif not header_error_improved:
            rejection_parts.append("header_error_not_improved")
        if mse_relative_improvement < mse_rel_improve_min:
            rejection_parts.append("mse_rel_improvement_below_min")
        rejection_reason = ",".join(rejection_parts) if rejection_parts else "gate_not_met"
    equalizer_state["baseline_header_peak"] = baseline_header_peak
    equalizer_state["baseline_header_error"] = baseline_header_error
    equalizer_state["peak_tolerance"] = float(peak_tol)
    equalizer_state["mse_relative_improvement_min"] = float(mse_rel_improve_min)
    equalizer_state["mse_relative_improvement"] = float(mse_relative_improvement)
    equalizer_state["header_peak_improved"] = bool(header_peak_improved)
    equalizer_state["header_peak_within_tol"] = bool(header_peak_within_tol)
    equalizer_state["header_error_improved"] = bool(header_error_improved)
    equalizer_state["header_error_not_worse"] = bool(header_error_not_worse)
    equalizer_state["accepted_via_header_error"] = bool(accept_via_header_error)
    equalizer_state["accepted_via_training_mse"] = bool(accept_via_training_mse)
    equalizer_state["header_gate_passed"] = bool(header_gate_passed)
    equalizer_state["acceptance_reason"] = acceptance_reason
    equalizer_state["rejection_reason"] = rejection_reason
    # The equalizer is only kept if it helps on the known preamble/header region.
    # That keeps it as a stabilizing block instead of letting it win on training
    # MSE while quietly making final header decisions worse.
    if bool(equalizer_state.get("applied", False)) and not header_gate_passed:
        equalizer_state["applied"] = False
        equalizer_state["reason"] = "header_gate_rejected"
        equalized_signal = carrier_aligned_signal
        equalizer_state["post_header_peak"] = baseline_header_peak
        equalizer_state["post_header_error"] = baseline_header_error
        equalizer_state["post_header_score"] = float(
            carrier_branch_evaluation["selected_header_score"]
        )
        equalizer_state["post_header_best_index"] = (
            int(expected_header_index)
            if expected_header_index is not None
            else None
        )
    elif bool(equalizer_state.get("applied", False)):
        equalizer_state["reason"] = acceptance_reason or str(equalizer_state.get("reason", "applied"))
    logger.debug(
        "Short trained equalizer: enabled=%s applied=%s reason=%s taps=%d valid_training_symbols=%d "
        "before_mse=%.6f after_mse=%.6f mse_rel_improvement=%.6f before_error=%.6f after_error=%.6f "
        "baseline_header_peak=%.6f post_header_peak=%.6f peak_tol=%.6f "
        "baseline_header_error=%.6f post_header_error=%.6f "
        "header_peak_improved=%s header_peak_within_tol=%s header_error_improved=%s header_error_not_worse=%s "
        "accepted_via_header_error=%s accepted_via_training_mse=%s header_gate_passed=%s "
        "acceptance_reason=%s rejection_reason=%s",
        bool(equalizer_state.get("enabled", False)),
        bool(equalizer_state.get("applied", False)),
        str(equalizer_state.get("reason", "")),
        int(equalizer_state.get("tap_count", 0)),
        int(equalizer_state.get("valid_training_symbols", 0)),
        float(equalizer_state.get("before_mse", 0.0)),
        float(equalizer_state.get("after_mse", 0.0)),
        float(equalizer_state.get("mse_relative_improvement", 0.0)),
        float(equalizer_state.get("before_error", 0.0)),
        float(equalizer_state.get("after_error", 0.0)),
        float(equalizer_state.get("baseline_header_peak", 0.0)),
        float(equalizer_state.get("post_header_peak", 0.0)),
        float(equalizer_state.get("peak_tolerance", 0.0)),
        float(equalizer_state.get("baseline_header_error", 0.0)),
        float(equalizer_state.get("post_header_error", 0.0)),
        bool(equalizer_state.get("header_peak_improved", False)),
        bool(equalizer_state.get("header_peak_within_tol", False)),
        bool(equalizer_state.get("header_error_improved", False)),
        bool(equalizer_state.get("header_error_not_worse", False)),
        bool(equalizer_state.get("accepted_via_header_error", False)),
        bool(equalizer_state.get("accepted_via_training_mse", False)),
        bool(equalizer_state.get("header_gate_passed", False)),
        str(equalizer_state.get("acceptance_reason", "")),
        str(equalizer_state.get("rejection_reason", "")),
    )

    # Midamble re-anchoring was removed after the stabilization pass. Costas now
    # always runs directly on the accepted post-equalizer symbol stream.
    costas_update_start_symbol = int(header_phase_symbol_index)
    costas_update_stop_symbol = int(
        min(
            equalized_signal.size,
            costas_update_start_symbol
            + int(gold_detector.gold_symbols.size)
            + int(payload_symbol_count),
        )
    )
    fine_signal, costas_traces = synchronizer.fine_frequenzy_synchronization_with_traces(
        equalized_signal,
        update_start_symbol=costas_update_start_symbol,
        update_stop_symbol=costas_update_stop_symbol,
    )
    logger.debug(
        "Costas window: update_start_symbol=%d update_stop_symbol=%d update_rate=%.6f gate_ref_power=%.6e",
        int(costas_update_start_symbol),
        int(costas_update_stop_symbol),
        float(np.mean(costas_traces["updates"])) if costas_traces["updates"].size else 0.0,
        float(costas_traces["gate_reference_power"]),
    )

    return RxPipelineState(
        crop_start=int(crop_start),
        crop_window_samples=int(crop_window_samples),
        crop_margin_samples=int(crop_margin_samples),
        crop_full_buffer_fallback=bool(crop_full_buffer_fallback),
        sync_input_signal=sync_input_signal,
        frontend_filtered_signal=frontend_filtered_signal,
        coarse_signal=coarse_signal,
        matched_signal_raw=matched_signal_raw,
        matched_signal=matched_signal,
        matched_normalization=matched_normalization,
        header_aligned_signal=header_aligned_signal,
        header_alignment=header_alignment,
        timing_input_signal=timing_input_signal,
        timing_acquisition=timing_acquisition,
        preamble_coarse_estimate=preamble_coarse_estimate,
        preamble_corrected_signal=preamble_corrected_signal,
        fractional_timing_alignment=fractional_timing_alignment,
        preamble_symbol_alignment=preamble_symbol_alignment,
        timing_signal=timing_signal,
        gardner_traces=gardner_traces,
        expected_header_index=expected_header_index,
        timing_header_index=timing_header_index,
        timing_header_peak=float(timing_header_peak),
        timing_header_rotation=timing_header_rotation,
        header_symbol_alignment=header_symbol_alignment,
        header_phase_symbol_index=int(header_phase_symbol_index),
        header_carrier_estimate=header_carrier_estimate,
        carrier_correction=carrier_correction,
        carrier_branch_evaluation=carrier_branch_evaluation,
        carrier_aligned_signal=carrier_aligned_signal,
        equalizer_state=equalizer_state,
        equalized_signal=equalized_signal,
        costas_update_start_symbol=int(costas_update_start_symbol),
        costas_update_stop_symbol=int(costas_update_stop_symbol),
        fine_signal=fine_signal,
        costas_traces=costas_traces,
    )
