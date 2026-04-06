from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from yaml import safe_dump

import Hardware_tester as ht
from convolutional_coder import ConvolutionalCoder
from datagram import Datagram
from filter import RRCFilter
from forward_error_correction import FCCodec
from gold_detection import GoldCodeDetector
from interleaver import Interleaver
from modulation import normalize_config_modulation_name
from sdr_transciever import SDRTransciever
from synchronize import Synchronizer
from TX_pipeline import build_scrambler


def _set_nested(config: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _config_diff(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_flat = _flatten_dict(base)
    candidate_flat = _flatten_dict(candidate)
    diff: dict[str, Any] = {}
    for key, value in candidate_flat.items():
        if base_flat.get(key) != value:
            diff[key] = value
    return diff


def _mean(results: list[dict[str, Any]], key: str) -> float:
    values = [float(result.get(key, 0.0)) for result in results]
    return float(np.mean(values)) if values else 0.0


def _rate(results: list[dict[str, Any]], key: str) -> float:
    values = [1.0 if bool(result.get(key, False)) else 0.0 for result in results]
    return float(np.mean(values)) if values else 0.0


def _trace_stats(traces: dict[str, np.ndarray]) -> dict[str, float]:
    costas_error = np.asarray(traces.get("costas_error", []), dtype=np.float32)
    costas_phase = np.asarray(traces.get("costas_phase_estimate", []), dtype=np.float32)
    gardner_error = np.asarray(traces.get("gardner_error", []), dtype=np.float32)

    return {
        "costas_error_mean": float(np.mean(costas_error)) if costas_error.size else 0.0,
        "costas_error_variance": float(np.var(costas_error)) if costas_error.size else 0.0,
        "costas_phase_span": (
            float(np.max(costas_phase) - np.min(costas_phase))
            if costas_phase.size
            else 0.0
        ),
        "gardner_error_mean": float(np.mean(gardner_error)) if gardner_error.size else 0.0,
        "gardner_error_variance": float(np.var(gardner_error)) if gardner_error.size else 0.0,
    }


def _candidate_rank_key(summary: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(summary["payload_success_rate"]),
        float(summary["header_success_rate"]),
        float(summary["detect_success_rate"]),
        float(summary["avg_final_peak"]),
        -float(summary["avg_fine_error"]),
        float(summary["avg_header_alignment_peak"]),
        -float(summary["avg_costas_error_variance"]),
        -float(summary["avg_gardner_error_variance"]),
    )


def _representative_rank_key(result: dict[str, Any]) -> tuple[float, ...]:
    return (
        1.0 if bool(result.get("recovered_ok", False)) else 0.0,
        1.0 if bool(result.get("header_detect_ok", False)) else 0.0,
        1.0 if bool(result.get("detected", False)) else 0.0,
        float(result.get("peak", 0.0)),
        -float(result.get("fine_nearest_symbol_error", 0.0)),
    )


def _summarize_results(
    label: str,
    results: list[dict[str, Any]],
    config_diff: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": label,
        "runs": int(len(results)),
        "config_diff": config_diff,
        "detect_success_rate": _rate(results, "detected"),
        "header_success_rate": _rate(results, "header_detect_ok"),
        "payload_success_rate": _rate(results, "recovered_ok"),
        "equalizer_accept_rate": _rate(results, "short_equalizer_applied"),
        "avg_header_alignment_peak": _mean(results, "header_alignment_peak"),
        "avg_timing_header_peak": _mean(results, "timing_header_peak"),
        "avg_final_peak": _mean(results, "peak"),
        "avg_preamble_cfo_score": _mean(results, "preamble_cfo_score"),
        "avg_preamble_cfo_frequency_hz": _mean(results, "preamble_cfo_frequency_hz"),
        "avg_costas_error_mean": _mean(results, "costas_error_mean"),
        "avg_costas_error_variance": _mean(results, "costas_error_variance"),
        "avg_costas_phase_span": _mean(results, "costas_phase_span"),
        "avg_gardner_error_mean": _mean(results, "gardner_error_mean"),
        "avg_gardner_error_variance": _mean(results, "gardner_error_variance"),
        "avg_timing_error": _mean(results, "timing_nearest_symbol_error"),
        "avg_carrier_error": _mean(results, "carrier_aligned_nearest_symbol_error"),
        "avg_equalized_error": _mean(results, "equalized_nearest_symbol_error"),
        "avg_fine_error": _mean(results, "fine_nearest_symbol_error"),
        "avg_eq_before_mse": _mean(results, "short_equalizer_before_mse"),
        "avg_eq_after_mse": _mean(results, "short_equalizer_after_mse"),
    }


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        safe_dump(data, handle, sort_keys=False)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)


def _build_stage_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "stage0_baseline",
            "description": "Current known-good config, 10-run baseline.",
            "runs": 10,
            "candidates": [
                {"label": "baseline", "overrides": {}},
            ],
        },
        {
            "name": "stage1_gain",
            "description": "Compact local TX/RX gain sweep around the current 0 dB manual point.",
            "runs": 5,
            "candidates": [
                {"label": "current_0_0", "overrides": {}},
                {"label": "tx_m3_rx_0", "overrides": {"transmitter.tx_gain_dB": -3}},
                {"label": "tx_0_rx_6", "overrides": {"receiver.rx_gain_dB": 6}},
                {
                    "label": "tx_m3_rx_6",
                    "overrides": {
                        "transmitter.tx_gain_dB": -3,
                        "receiver.rx_gain_dB": 6,
                    },
                },
                {
                    "label": "tx_m6_rx_6",
                    "overrides": {
                        "transmitter.tx_gain_dB": -6,
                        "receiver.rx_gain_dB": 6,
                    },
                },
            ],
        },
        {
            "name": "stage2_preamble_cfo",
            "description": "Preamble CFO and symbol-alignment sweep.",
            "runs": 5,
            "candidates": [
                {"label": "current", "overrides": {}},
                {
                    "label": "looser_scores",
                    "overrides": {
                        "synchronization.preamble_cfo_min_score": 0.30,
                        "synchronization.preamble_symbol_min_score": 0.35,
                        "synchronization.preamble_symbol_search_radius_symbols": 6,
                    },
                },
                {
                    "label": "stricter_scores",
                    "overrides": {
                        "synchronization.preamble_cfo_min_score": 0.60,
                        "synchronization.preamble_symbol_min_score": 0.65,
                        "synchronization.preamble_symbol_search_radius_symbols": 2,
                    },
                },
                {
                    "label": "clamp_low",
                    "overrides": {
                        "synchronization.preamble_cfo_max_abs_slope_rad_per_symbol": 0.02,
                    },
                },
                {
                    "label": "cfo_disabled",
                    "overrides": {
                        "synchronization.preamble_cfo_enable": False,
                    },
                },
            ],
        },
        {
            "name": "stage3_gardner",
            "description": "Gardner timing-loop sweep.",
            "runs": 5,
            "candidates": [
                {"label": "current", "overrides": {}},
                {
                    "label": "slower_loop",
                    "overrides": {
                        "synchronization.gardner_Kp": 0.001,
                        "synchronization.gardner_Ki": 5.0e-6,
                    },
                },
                {
                    "label": "faster_loop",
                    "overrides": {
                        "synchronization.gardner_Kp": 0.004,
                        "synchronization.gardner_Ki": 2.0e-5,
                    },
                },
                {
                    "label": "tighter_gate_short",
                    "overrides": {
                        "synchronization.gardner_gate_energy_percentile_low": 35.0,
                        "synchronization.gardner_gate_energy_percentile_high": 85.0,
                        "synchronization.gardner_gate_energy_scale_low": 2.0,
                        "synchronization.gardner_gate_energy_scale_high": 2.0,
                        "synchronization.gardner_tracking_payload_symbols": 32,
                    },
                },
                {
                    "label": "looser_gate_long",
                    "overrides": {
                        "synchronization.gardner_gate_energy_percentile_low": 15.0,
                        "synchronization.gardner_gate_energy_percentile_high": 95.0,
                        "synchronization.gardner_gate_energy_scale_low": 4.0,
                        "synchronization.gardner_gate_energy_scale_high": 4.0,
                        "synchronization.gardner_tracking_payload_symbols": 128,
                    },
                },
            ],
        },
        {
            "name": "stage4_carrier",
            "description": "Costas and header-carrier sweep.",
            "runs": 5,
            "candidates": [
                {"label": "current", "overrides": {}},
                {
                    "label": "slower_costas",
                    "overrides": {
                        "synchronization.costas_Kp": 0.0025,
                        "synchronization.costas_Ki": 5.0e-6,
                        "synchronization.costas_gate_power_percentile_low": 20.0,
                        "synchronization.costas_gate_power_percentile_high": 80.0,
                    },
                },
                {
                    "label": "faster_costas",
                    "overrides": {
                        "synchronization.costas_Kp": 0.01,
                        "synchronization.costas_Ki": 2.0e-5,
                        "synchronization.costas_gate_power_percentile_low": 10.0,
                        "synchronization.costas_gate_power_percentile_high": 90.0,
                    },
                },
                {
                    "label": "phase_only_forced",
                    "overrides": {
                        "synchronization.header_carrier_apply_slope_enable": False,
                    },
                },
                {
                    "label": "conservative_slope",
                    "overrides": {
                        "synchronization.header_carrier_apply_slope_enable": True,
                        "synchronization.header_carrier_slope_min_score": 0.70,
                        "synchronization.header_carrier_max_abs_slope_rad_per_symbol": 0.02,
                    },
                },
            ],
        },
        {
            "name": "stage5_equalizer",
            "description": "Short equalizer usefulness and acceptance sweep.",
            "runs": 5,
            "candidates": [
                {"label": "current", "overrides": {}},
                {
                    "label": "equalizer_off",
                    "overrides": {
                        "synchronization.short_equalizer_enable": False,
                    },
                },
                {
                    "label": "eq_3tap_lowreg",
                    "overrides": {
                        "synchronization.short_equalizer_tap_count": 3,
                        "synchronization.short_equalizer_regularization": 1.0e-4,
                    },
                },
                {
                    "label": "eq_5tap_highreg",
                    "overrides": {
                        "synchronization.short_equalizer_tap_count": 5,
                        "synchronization.short_equalizer_regularization": 1.0e-2,
                    },
                },
                {
                    "label": "eq_header_only",
                    "overrides": {
                        "synchronization.short_equalizer_train_on_preamble": False,
                        "synchronization.short_equalizer_train_on_header": True,
                    },
                },
            ],
        },
        {
            "name": "stage6_thresholds",
            "description": "Detection and decode-threshold robustness sweep.",
            "runs": 5,
            "candidates": [
                {"label": "current", "overrides": {}},
                {
                    "label": "lower_decode_gate",
                    "overrides": {
                        "gold_sequence.decode_candidate_min_peak": 0.55,
                    },
                },
                {
                    "label": "lower_corr_decode",
                    "overrides": {
                        "gold_sequence.correlation_threshold": 0.60,
                        "gold_sequence.decode_candidate_min_peak": 0.55,
                    },
                },
                {
                    "label": "stricter_thresholds",
                    "overrides": {
                        "gold_sequence.correlation_threshold": 0.75,
                        "gold_sequence.decode_candidate_min_peak": 0.70,
                    },
                },
                {
                    "label": "softer_alignment",
                    "overrides": {
                        "synchronization.header_alignment_soft_phase_min_score": 0.15,
                        "synchronization.timing_header_candidate_count": 7,
                    },
                },
            ],
        },
    ]


def _run_candidate(
    *,
    config: dict[str, Any],
    config_path: Path,
    runs: int,
    payload: str,
    guard_symbols: int,
    flush_buffers: int,
    candidate_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(config_path, config)

    modulation_name = normalize_config_modulation_name(config)
    rrc_filter = RRCFilter(config)
    gold_detector = GoldCodeDetector(config)
    fec = FCCodec(config)
    interleaver = Interleaver(config)
    scrambler = build_scrambler(config)
    conv_coder = ConvolutionalCoder(config, warmup=False, use_numba=False)
    synchronizer = Synchronizer(config, warmup=False, use_numba=False)
    sdr = SDRTransciever(config)

    datagram, pipeline_result = ht.run_pipeline_test(
        modulation_name=modulation_name,
        fec=fec,
        interleaver=interleaver,
        conv_coder=conv_coder,
        scrambler=scrambler,
        payload_text=payload,
    )
    if not pipeline_result["roundtrip_ok"]:
        raise RuntimeError(f"Local pipeline test failed: {pipeline_result['error']}")

    results: list[dict[str, Any]] = []
    representative_result: dict[str, Any] | None = None
    representative_traces: dict[str, Any] | None = None
    noise_floor: float | None = None

    if not sdr.connect():
        raise RuntimeError("Failed to connect to SDR.")

    try:
        noise_floor = sdr.measure_noise_floor_dB()
        if noise_floor is None:
            raise RuntimeError("Noise floor measurement failed.")
        synchronizer.set_noise_floor(noise_floor)

        for run_index in range(runs):
            result, traces = ht.run_gold_hardware_test(
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
                guard_symbols=guard_symbols,
                flush_buffers=flush_buffers,
            )
            result.update(_trace_stats(traces))
            result["run_index"] = run_index + 1
            results.append(result)

            if (
                representative_result is None
                or _representative_rank_key(result) > _representative_rank_key(representative_result)
            ):
                representative_result = deepcopy(result)
                representative_traces = traces

        if representative_traces is not None:
            plots_dir = candidate_dir / "representative_plots"
            ht.save_hardware_test_plots(
                config=config,
                traces=representative_traces,
                output_dir=plots_dir,
                show_plots=False,
            )

    finally:
        sdr.disconnect()

    _write_json(candidate_dir / "run_results.json", results)
    summary = _summarize_results(
        label=candidate_dir.name,
        results=results,
        config_diff={},
    )
    summary["noise_floor_dB"] = float(noise_floor if noise_floor is not None else 0.0)
    summary["pipeline_roundtrip_ok"] = bool(pipeline_result["roundtrip_ok"])
    summary["representative_run_index"] = (
        int(representative_result["run_index"]) if representative_result is not None else None
    )
    _write_json(candidate_dir / "summary.json", summary)
    return summary, results


def _save_stage_markdown(
    stage: dict[str, Any],
    stage_dir: Path,
    candidate_summaries: list[dict[str, Any]],
    best_summary: dict[str, Any],
    carried_from_label: str,
) -> None:
    lines = [
        f"# {stage['name']}",
        "",
        stage["description"],
        "",
        f"Carried config: `{carried_from_label}`",
        "",
        "| Candidate | Runs | Detect % | Header % | Payload % | Avg header peak | Avg final peak | Avg fine err | Avg CFO score | Eq accept % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in candidate_summaries:
        lines.append(
            "| {label} | {runs} | {detect:.1f} | {header:.1f} | {payload:.1f} | {hpk:.3f} | {fpk:.3f} | {ferr:.3f} | {cfo:.3f} | {eq:.1f} |".format(
                label=summary["label"],
                runs=summary["runs"],
                detect=100.0 * summary["detect_success_rate"],
                header=100.0 * summary["header_success_rate"],
                payload=100.0 * summary["payload_success_rate"],
                hpk=summary["avg_header_alignment_peak"],
                fpk=summary["avg_final_peak"],
                ferr=summary["avg_fine_error"],
                cfo=summary["avg_preamble_cfo_score"],
                eq=100.0 * summary["equalizer_accept_rate"],
            )
        )
    lines.extend(
        [
            "",
            f"Best candidate: `{best_summary['label']}`",
            "",
            "Config diff from carried config:",
            "",
            "```yaml",
            safe_dump(best_summary.get("config_diff", {}), sort_keys=False).rstrip(),
            "```",
            "",
        ]
    )
    (stage_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_campaign(args: argparse.Namespace) -> int:
    base_config = ht.configure_hardware_tester(ht.load_config(args.config), args.rx_buffer_size)
    run_root = Path("tuning_results") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root.mkdir(parents=True, exist_ok=True)
    _write_yaml(run_root / "baseline_config.yaml", base_config)

    carried_config = deepcopy(base_config)
    carried_label = "baseline"
    overall_summary: list[dict[str, Any]] = []

    for stage in _build_stage_definitions():
        stage_dir = run_root / stage["name"]
        stage_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {stage['name']} ===")
        print(stage["description"])

        candidate_summaries: list[dict[str, Any]] = []
        best_summary: dict[str, Any] | None = None
        best_config_for_stage: dict[str, Any] | None = None

        for candidate in stage["candidates"]:
            candidate_config = deepcopy(carried_config)
            for path, value in candidate["overrides"].items():
                _set_nested(candidate_config, path, value)

            candidate_dir = stage_dir / candidate["label"]
            candidate_config_path = candidate_dir / "config.yaml"
            config_diff = _config_diff(carried_config, candidate_config)

            print(f"  -> {candidate['label']} ({stage['runs']} runs)")
            summary, _results = _run_candidate(
                config=candidate_config,
                config_path=candidate_config_path,
                runs=int(stage["runs"]),
                payload=args.payload,
                guard_symbols=int(args.guard_symbols),
                flush_buffers=int(args.flush_buffers),
                candidate_dir=candidate_dir,
            )
            summary["config_diff"] = config_diff
            _write_json(candidate_dir / "summary.json", summary)
            candidate_summaries.append(summary)

            print(
                "     detect={:.0f}% header={:.0f}% payload={:.0f}% peak={:.3f} fine={:.3f}".format(
                    100.0 * summary["detect_success_rate"],
                    100.0 * summary["header_success_rate"],
                    100.0 * summary["payload_success_rate"],
                    summary["avg_final_peak"],
                    summary["avg_fine_error"],
                )
            )

            if best_summary is None or _candidate_rank_key(summary) > _candidate_rank_key(best_summary):
                best_summary = summary
                best_config_for_stage = candidate_config

        assert best_summary is not None
        assert best_config_for_stage is not None
        candidate_summaries.sort(key=_candidate_rank_key, reverse=True)

        _write_json(stage_dir / "candidate_summaries.json", candidate_summaries)
        _save_stage_markdown(
            stage=stage,
            stage_dir=stage_dir,
            candidate_summaries=candidate_summaries,
            best_summary=best_summary,
            carried_from_label=carried_label,
        )
        _write_yaml(stage_dir / "best_config.yaml", best_config_for_stage)

        carried_config = deepcopy(best_config_for_stage)
        carried_label = f"{stage['name']}:{best_summary['label']}"
        overall_summary.append(
            {
                "stage": stage["name"],
                "best_candidate": best_summary["label"],
                "best_summary": best_summary,
            }
        )

    final_diff = _config_diff(base_config, carried_config)
    _write_yaml(run_root / "final_recommended_config.yaml", carried_config)
    _write_yaml(run_root / "final_config_diff.yaml", final_diff)
    _write_json(run_root / "overall_summary.json", overall_summary)

    lines = [
        "# Tuning Campaign",
        "",
        f"Started from `{args.config}`",
        "",
        "| Stage | Best candidate | Detect % | Header % | Payload % | Avg peak | Avg fine err |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in overall_summary:
        summary = entry["best_summary"]
        lines.append(
            "| {stage} | {candidate} | {detect:.1f} | {header:.1f} | {payload:.1f} | {peak:.3f} | {fine:.3f} |".format(
                stage=entry["stage"],
                candidate=entry["best_candidate"],
                detect=100.0 * summary["detect_success_rate"],
                header=100.0 * summary["header_success_rate"],
                payload=100.0 * summary["payload_success_rate"],
                peak=summary["avg_final_peak"],
                fine=summary["avg_fine_error"],
            )
        )
    lines.extend(
        [
            "",
            "Final config diff from baseline:",
            "",
            "```yaml",
            safe_dump(final_diff, sort_keys=False).rstrip(),
            "```",
            "",
        ]
    )
    (run_root / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nCampaign complete. Results saved to {run_root}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged SDR tuning sweeps on hardware.")
    parser.add_argument("--config", default="setup/config.yaml")
    parser.add_argument("--payload", default="hardware test ")
    parser.add_argument("--rx-buffer-size", type=int, default=None)
    parser.add_argument("--guard-symbols", type=int, default=32)
    parser.add_argument("--flush-buffers", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_campaign(parse_args()))
