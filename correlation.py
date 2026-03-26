"""
Analyze correlation between a transmitted text-message burst and one or more received IQ captures.

Technique:
    1. Build the exact transmit reference burst from the local modulation, Gold-code framing, and RRC modules.
    2. Compute normalized complex cross-correlation between the reference burst and each received signal.
    3. Report peak correlation, timing offset, phase offset, and a Gold-only correlation score.

This is appropriate for short framed text-message traffic between two or more radios because the
transmitted burst is known and finite, while normalized cross-correlation remains comparable across
receivers with different gain settings.

Examples:
    .venv/bin/python correlation.py --rx capture.npy
    .venv/bin/python correlation.py --message "Hello" --rx rx_a.npy rx_b.npy
    .venv/bin/python correlation.py --tx tx.npy --rx rx_a.npy rx_b.npy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from yaml import safe_load

from datagram import Datagram, msgType
from filter import RRCFilter
from gold_detection import GoldCodeDetector
from modulation import ModulationProtocol


@dataclass
class CorrelationResult:
    rx_path: str
    sample_offset: int
    time_offset_s: float
    normalized_peak: float
    peak_phase_rad: float
    peak_phase_deg: float
    energy_ratio_db: float
    gold_peak: float
    gold_index: int


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = safe_load(handle)

    gold_cfg = config.setdefault("gold_sequence", {})
    if "correlation_scale_factor_threshold" not in gold_cfg:
        gold_cfg["correlation_scale_factor_threshold"] = gold_cfg.get("correlation_threshold", 0.0)
    return config


def make_datagram(message: str, msg_id: int) -> Datagram:
    payload = np.frombuffer(message.encode("utf-8"), dtype=np.uint8)
    return Datagram(msg_id=np.uint8(msg_id), msg_type=msgType.DATA, payload=payload)


def build_reference_signal(config: dict, message: str, msg_id: int) -> np.ndarray:
    modem = ModulationProtocol(config)
    detector = GoldCodeDetector(config)
    rrc_filter = RRCFilter(config)
    datagram = make_datagram(message, msg_id)

    modulated = modem.modulate_message(datagram)
    framed = detector.add_gold_symbols(modulated)
    upsampled = modem.upsample_symbols(framed)
    return rrc_filter.apply_filter(upsampled).astype(np.complex64)


def load_complex_signal(path: str) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if isinstance(array, np.lib.npyio.NpzFile):
        if "signal" in array:
            data = array["signal"]
        elif len(array.files) == 1:
            data = array[array.files[0]]
        else:
            raise ValueError(f"{path} contains multiple arrays; expected 'signal' or a single array.")
    else:
        data = array

    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError(f"{path} must contain a 1D IQ array.")
    return data.astype(np.complex64, copy=False)


def normalized_cross_correlation(reference: np.ndarray, received: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if received.size < reference.size:
        raise ValueError("Received signal must be at least as long as the reference signal.")

    ref = reference.astype(np.complex64, copy=False)
    rx = received.astype(np.complex64, copy=False)

    correlation = np.correlate(rx, ref, mode="valid").astype(np.complex64)
    ref_energy = float(np.vdot(ref, ref).real)
    if ref_energy <= 0:
        raise ValueError("Reference signal energy must be positive.")

    rx_power = np.abs(rx) ** 2
    window_energy = np.convolve(rx_power, np.ones(ref.size, dtype=np.float32), mode="valid")
    denom = np.sqrt(np.maximum(ref_energy * window_energy, 1e-12))
    normalized = correlation / denom
    return normalized, correlation


def gold_correlation(config: dict, received: np.ndarray) -> tuple[float, int]:
    detector = GoldCodeDetector(config)
    modem = ModulationProtocol(config)
    rrc_filter = RRCFilter(config)

    gold_upsampled = modem.upsample_symbols(detector.gold_symbols.astype(np.complex64))
    gold_reference = rrc_filter.apply_filter(gold_upsampled).astype(np.complex64)

    if received.size < gold_reference.size:
        raise ValueError("Received signal is shorter than the Gold reference.")

    normalized, _ = normalized_cross_correlation(gold_reference, received)
    peak_index = int(np.argmax(np.abs(normalized)))
    peak_value = float(np.abs(normalized[peak_index]))
    return peak_value, peak_index


def analyze_signal(
    config: dict,
    reference: np.ndarray,
    received: np.ndarray,
    sample_rate: float,
    label: str = "signal",
) -> tuple[CorrelationResult, np.ndarray]:
    normalized, raw = normalized_cross_correlation(reference, received)
    abs_normalized = np.abs(normalized)
    peak_index = int(np.argmax(abs_normalized))
    peak_value = float(abs_normalized[peak_index])
    peak_phase = float(np.angle(raw[peak_index]))

    aligned = received[peak_index : peak_index + reference.size]
    ref_rms = np.sqrt(np.mean(np.abs(reference) ** 2))
    rx_rms = np.sqrt(np.mean(np.abs(aligned) ** 2))
    energy_ratio_db = 20 * np.log10(max(rx_rms, 1e-12) / max(ref_rms, 1e-12))

    gold_peak, gold_index = gold_correlation(config, received)

    result = CorrelationResult(
        rx_path=label,
        sample_offset=peak_index,
        time_offset_s=peak_index / sample_rate,
        normalized_peak=peak_value,
        peak_phase_rad=peak_phase,
        peak_phase_deg=np.degrees(peak_phase),
        energy_ratio_db=float(energy_ratio_db),
        gold_peak=gold_peak,
        gold_index=gold_index,
    )
    return result, normalized


def analyze_capture(
    config: dict,
    reference: np.ndarray,
    rx_path: str,
    sample_rate: float,
) -> CorrelationResult:
    received = load_complex_signal(rx_path)
    result, _ = analyze_signal(config, reference, received, sample_rate, label=rx_path)
    return result


def plot_correlation(
    reference: np.ndarray,
    received: np.ndarray,
    normalized: np.ndarray,
    result: CorrelationResult,
    sample_rate: float,
    zoom_radius: int = 256,
) -> None:
    delays = np.arange(normalized.size)
    peak = result.sample_offset
    zoom_start = max(0, peak - zoom_radius)
    zoom_stop = min(normalized.size, peak + zoom_radius + 1)

    aligned = received[peak : peak + reference.size]
    count = min(reference.size, aligned.size)
    time_us = np.arange(count) / sample_rate * 1e6

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(delays, np.abs(normalized), linewidth=1.0)
    axes[0].axvline(peak, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].set_title(f"Normalized Correlation Magnitude - {result.rx_path}")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_xlabel("Sample Offset")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(delays[zoom_start:zoom_stop], np.abs(normalized[zoom_start:zoom_stop]), linewidth=1.2)
    axes[1].axvline(peak, color="tab:red", linestyle="--", linewidth=1.0)
    axes[1].set_title("Peak Neighborhood")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlabel("Sample Offset")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_us, np.abs(reference[:count]), label="|reference|", linewidth=1.1)
    axes[2].plot(time_us, np.abs(aligned[:count]), label="|received aligned|", linewidth=1.1, alpha=0.8)
    axes[2].set_title("Aligned Burst Envelope")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (us)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(
        f"peak={result.normalized_peak:.4f}, phase={result.peak_phase_deg:.2f} deg, "
        f"gold={result.gold_peak:.4f}",
        fontsize=11,
    )
    plt.tight_layout()


def print_result(result: CorrelationResult) -> None:
    print(
        f"[{result.rx_path}] "
        f"peak={result.normalized_peak:.6f} "
        f"offset_samples={result.sample_offset} "
        f"offset_s={result.time_offset_s:.9f} "
        f"phase_deg={result.peak_phase_deg:.2f} "
        f"energy_ratio_db={result.energy_ratio_db:.2f} "
        f"gold_peak={result.gold_peak:.6f} "
        f"gold_index={result.gold_index}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze correlation between a transmitted text burst and received IQ captures."
    )
    parser.add_argument("--config", default="setup/config.yaml", help="Path to config file.")
    parser.add_argument(
        "--message",
        default="Hello, SDR!",
        help="Text payload used when building a reference burst locally.",
    )
    parser.add_argument("--msg-id", type=int, default=101, help="Datagram message ID for the reference burst.")
    parser.add_argument(
        "--tx",
        help="Optional path to a transmitted IQ reference (.npy or .npz). If omitted, build it from the message.",
    )
    parser.add_argument(
        "--rx",
        nargs="+",
        required=True,
        help="One or more received IQ captures (.npy or .npz).",
    )
    parser.add_argument(
        "--sort",
        choices=["path", "peak"],
        default="peak",
        help="How to sort results when analyzing multiple receivers.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show correlation plots for each analyzed receiver.",
    )
    parser.add_argument(
        "--plot-zoom-radius",
        type=int,
        default=256,
        help="Half-width in samples for the peak zoom subplot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sample_rate = float(config["modulation"]["sample_rate"])

    if args.tx:
        reference = load_complex_signal(args.tx)
    else:
        reference = build_reference_signal(config, args.message, args.msg_id)

    results = [analyze_capture(config, reference, rx_path, sample_rate) for rx_path in args.rx]

    if args.sort == "peak":
        results.sort(key=lambda item: item.normalized_peak, reverse=True)
    else:
        results.sort(key=lambda item: Path(item.rx_path).name)

    print(f"reference_samples={reference.size} sample_rate={sample_rate:.0f}")
    for result in results:
        print_result(result)
        if args.plot:
            received = load_complex_signal(result.rx_path)
            _, normalized = analyze_signal(config, reference, received, sample_rate, label=result.rx_path)
            plot_correlation(reference, received, normalized, result, sample_rate, args.plot_zoom_radius)

    if args.plot:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
