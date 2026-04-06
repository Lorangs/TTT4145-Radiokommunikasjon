"""
Simulation harness for exercising the current SDR modules without hardware.

Examples:
    python simulate.py
    python simulate.py --stage modem
    python simulate.py --stage sync --cases clean noisy --noise-std 0.05
    python simulate.py --stage full --message "Hello, SDR!" --freq-offset 800 --timing-offset-samples 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from yaml import safe_load

from correlation import analyze_signal, build_reference_signal
from datagram import Datagram, msgType
from filter import RRCFilter
from gold_detection import GoldCodeDetector
from modulation import ModulationProtocol, nearest_constellation_symbols
from synchronize import Synchronizer


@dataclass
class SimulationCase:
    name: str
    noise_std: float


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = safe_load(handle)

    gold_cfg = config.setdefault("gold_sequence", {})
    if "correlation_scale_factor_threshold" not in gold_cfg:
        gold_cfg["correlation_scale_factor_threshold"] = gold_cfg.get("correlation_threshold", 0.0)
    return config


def build_cases(case_names: Iterable[str], noisy_std: float) -> list[SimulationCase]:
    cases: list[SimulationCase] = []
    for name in case_names:
        if name == "clean":
            cases.append(SimulationCase(name="clean", noise_std=0.0))
        elif name == "noisy":
            cases.append(SimulationCase(name="noisy", noise_std=noisy_std))
        else:
            raise ValueError(f"Unsupported case: {name}")
    return cases


def add_awgn(signal: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    if noise_std <= 0:
        return signal.copy()
    noise = noise_std * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return signal + noise.astype(np.complex64)


def apply_channel(
    signal: np.ndarray,
    sample_rate: float,
    rng: np.random.Generator,
    noise_std: float,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    timing_offset: float = 0.0,
) -> np.ndarray:
    impaired = signal.astype(np.complex64, copy=True)
    if timing_offset:
        impaired = np.roll(impaired, int(round(timing_offset)))
    if freq_offset or phase_offset:
        time = np.arange(impaired.size, dtype=np.float64) / sample_rate
        impaired = impaired * np.exp(1j * (2 * np.pi * freq_offset * time + phase_offset))
    return add_awgn(impaired, noise_std, rng)

def symbol_error_rate(
    received: np.ndarray,
    reference: np.ndarray,
    modulation_type: str,
) -> tuple[float, int]:
    count = min(received.size, reference.size)
    if count == 0:
        return 1.0, 0

    rx = received[:count]
    ref = reference[:count]

    if modulation_type.upper().strip() == "BPSK":
        rotations = np.array([1, -1], dtype=np.complex64)
    else:
        rotations = np.array([1, -1, 1j, -1j], dtype=np.complex64)

    best_errors = count
    for rotation in rotations:
        decisions = nearest_constellation_symbols(rx * rotation, modulation_type)
        errors = int(np.count_nonzero(decisions != ref))
        best_errors = min(best_errors, errors)
    return best_errors / count, count


def best_aligned_symbol_error_rate(
    received: np.ndarray,
    reference: np.ndarray,
    modulation_type: str,
    max_lag: int,
) -> tuple[float, int, int]:
    best_ser = 1.0
    best_count = 0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            rx = received[lag:]
            ref = reference
        else:
            rx = received
            ref = reference[-lag:]
        ser, count = symbol_error_rate(rx, ref, modulation_type)
        if count > 0 and ser < best_ser:
            best_ser = ser
            best_count = count
            best_lag = lag
    return best_ser, best_count, best_lag


def recover_datagram_from_symbols(
    symbols: np.ndarray,
    modem: ModulationProtocol,
) -> Datagram:
    bits = modem.decision_bits_from_symbols(symbols)
    return modem.unpack_message_bits(bits)


def make_datagram(message: str) -> Datagram:
    payload = np.frombuffer(message.encode("utf-8"), dtype=np.uint8)
    return Datagram(msg_id=np.uint8(101), msg_type=msgType.DATA, payload=payload)


def bit_balance(bits: np.ndarray) -> tuple[int, int]:
    ones = int(np.count_nonzero(bits))
    zeros = int(bits.size - ones)
    return zeros, ones


def normalized_symbol_correlation(reference: np.ndarray, received: np.ndarray) -> np.ndarray:
    if received.size < reference.size:
        return np.array([], dtype=np.float32)

    reference = reference.astype(np.complex64, copy=False)
    received = received.astype(np.complex64, copy=False)
    raw = np.correlate(received, reference, mode="valid")
    ref_energy = float(np.vdot(reference, reference).real)
    rx_power = np.abs(received) ** 2
    window_energy = np.convolve(rx_power, np.ones(reference.size, dtype=np.float32), mode="valid")
    denom = np.sqrt(np.maximum(ref_energy * window_energy, 1e-12))
    return np.abs(raw) / denom


def detect_gold_start(
    received_symbols: np.ndarray,
    detector: GoldCodeDetector,
    threshold: float,
) -> tuple[int | None, float]:
    scores = normalized_symbol_correlation(detector.gold_symbols, received_symbols)
    if scores.size == 0:
        return None, 0.0

    peak_index = int(np.argmax(scores))
    peak_value = float(scores[peak_index])
    if peak_value < threshold:
        return None, peak_value
    return peak_index, peak_value


def make_random_symbols(
    rng: np.random.Generator,
    modulation_type: str,
    count: int,
) -> np.ndarray:
    modulation_type = modulation_type.upper().strip()
    if modulation_type == "BPSK":
        return rng.choice(np.array([-1, 1], dtype=np.int8), size=count).astype(np.complex64)

    i = rng.choice(np.array([-1, 1], dtype=np.int8), size=count)
    q = rng.choice(np.array([-1, 1], dtype=np.int8), size=count)
    return (i + 1j * q).astype(np.complex64)


def shape_symbol_stream(
    symbols: np.ndarray,
    modem: ModulationProtocol,
    rrc_filter: RRCFilter,
    guard_symbols: int,
) -> tuple[np.ndarray, np.ndarray]:
    guard = np.zeros(guard_symbols, dtype=np.complex64)
    framed_symbols = np.concatenate((guard, symbols.astype(np.complex64), guard))
    upsampled = modem.upsample_symbols(framed_symbols)
    return rrc_filter.apply_filter(upsampled).astype(np.complex64), framed_symbols


def run_sync_pipeline(
    tx_signal: np.ndarray,
    synchronizer: Synchronizer,
    rrc_filter: RRCFilter,
    rng: np.random.Generator,
    case: SimulationCase,
    args: argparse.Namespace,
) -> np.ndarray:
    rx_signal = apply_channel(
        tx_signal,
        sample_rate=float(synchronizer.sample_rate),
        rng=rng,
        noise_std=case.noise_std,
        freq_offset=args.freq_offset,
        phase_offset=args.phase_offset,
        timing_offset=args.timing_offset_samples,
    )
    coarse = synchronizer.coarse_frequenzy_synchronization(rx_signal)
    matched = rrc_filter.apply_filter(coarse)
    timed = synchronizer.gardner_timing_synchronization(matched)
    return synchronizer.fine_frequenzy_synchronization(timed)


def run_modem_case(
    case: SimulationCase,
    datagram: Datagram,
    modem: ModulationProtocol,
    rng: np.random.Generator,
) -> dict:
    tx_symbols = modem.modulate_message(datagram).astype(np.complex64)
    rx_symbols = add_awgn(tx_symbols, case.noise_std, rng)
    ser, compared = symbol_error_rate(rx_symbols, tx_symbols, modem.modulation_type)

    native_ok = True
    native_error = ""
    try:
        modem.demodulate_message(rx_symbols)
    except Exception as exc:
        native_ok = False
        native_error = str(exc)

    compat_ok = True
    compat_error = ""
    recovered_text = ""
    scrambled_bits = modem.pack_message_bits(datagram)
    zeros, ones = bit_balance(scrambled_bits)
    try:
        recovered = recover_datagram_from_symbols(rx_symbols, modem)
        recovered_text = recovered.payload_text(trim_padding=True)
        compat_ok = recovered_text == datagram.payload_text(trim_padding=True)
    except Exception as exc:
        compat_ok = False
        compat_error = str(exc)

    return {
        "stage": "modem",
        "case": case.name,
        "symbols": int(tx_symbols.size),
        "noise_std": case.noise_std,
        "symbol_error_rate": ser,
        "symbols_compared": compared,
        "scrambled_zero_bits": zeros,
        "scrambled_one_bits": ones,
        "native_roundtrip_ok": native_ok,
        "native_error": native_error,
        "compat_roundtrip_ok": compat_ok,
        "compat_error": compat_error,
        "recovered_text": recovered_text,
    }


def run_gold_case(
    case: SimulationCase,
    datagram: Datagram,
    modem: ModulationProtocol,
    detector: GoldCodeDetector,
    rng: np.random.Generator,
) -> dict:
    payload_symbols = modem.modulate_message(datagram).astype(np.complex64)
    tx = detector.add_gold_symbols(payload_symbols)
    rx = add_awgn(tx, case.noise_std, rng)

    start_index, peak = detect_gold_start(
        nearest_constellation_symbols(rx, modem.modulation_type),
        detector,
        threshold=detector.correlation_scale_factor_threshold,
    )
    remove_ok = False
    if start_index is not None:
        recovered = detector.remove_gold_symbols(rx, start_index)
        trimmed = recovered[: payload_symbols.size]
        decisions = nearest_constellation_symbols(trimmed, modem.modulation_type)
        remove_ok = bool(np.array_equal(decisions, payload_symbols))

    return {
        "stage": "gold",
        "case": case.name,
        "noise_std": case.noise_std,
        "detected": start_index is not None,
        "detected_index": start_index,
        "peak": peak,
        "remove_ok": remove_ok,
        "sequence_length": int(len(detector.gold_symbols)),
    }


def run_sync_case(
    case: SimulationCase,
    modem: ModulationProtocol,
    rrc_filter: RRCFilter,
    synchronizer: Synchronizer,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict:
    source_symbols = make_random_symbols(rng, modem.modulation_type, args.symbol_count)
    tx_signal, framed_symbols = shape_symbol_stream(
        source_symbols,
        modem,
        rrc_filter,
        guard_symbols=args.guard_symbols,
    )
    synced = run_sync_pipeline(tx_signal, synchronizer, rrc_filter, rng, case, args)

    ser, compared, lag = best_aligned_symbol_error_rate(
        synced,
        framed_symbols,
        modem.modulation_type,
        max_lag=args.sync_max_lag,
    )

    return {
        "stage": "sync",
        "case": case.name,
        "noise_std": case.noise_std,
        "symbols_compared": compared,
        "symbol_error_rate": ser,
        "best_lag": lag,
        "timing_output_symbols": int(synced.size),
        "pass": compared > 0 and ser <= args.sync_ser_threshold,
    }


def run_full_case(
    case: SimulationCase,
    datagram: Datagram,
    modem: ModulationProtocol,
    detector: GoldCodeDetector,
    rrc_filter: RRCFilter,
    synchronizer: Synchronizer,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict:
    tx_symbols = modem.modulate_message(datagram).astype(np.complex64)
    framed_symbols = detector.add_gold_symbols(tx_symbols)
    tx_signal, _ = shape_symbol_stream(
        framed_symbols,
        modem,
        rrc_filter,
        guard_symbols=args.guard_symbols,
    )
    synced = run_sync_pipeline(tx_signal, synchronizer, rrc_filter, rng, case, args)
    decided = nearest_constellation_symbols(synced, modem.modulation_type)

    start_index, peak = detect_gold_start(
        decided,
        detector,
        threshold=detector.correlation_scale_factor_threshold,
    )

    recovered_text = ""
    compat_ok = False
    compat_error = ""
    if start_index is not None:
        payload_rx = detector.remove_gold_symbols(decided, start_index)
        payload_rx = payload_rx[: tx_symbols.size]
        try:
            recovered = recover_datagram_from_symbols(payload_rx, modem)
            recovered_text = recovered.payload_text(trim_padding=True)
            compat_ok = recovered_text == datagram.payload_text(trim_padding=True)
        except Exception as exc:
            compat_error = str(exc)

    return {
        "stage": "full",
        "case": case.name,
        "noise_std": case.noise_std,
        "detected": start_index is not None,
        "detected_index": start_index,
        "peak": peak,
        "compat_roundtrip_ok": compat_ok,
        "compat_error": compat_error,
        "recovered_text": recovered_text,
    }


def run_correlation_case(
    case: SimulationCase,
    config: dict,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict:
    reference = build_reference_signal(config, args.message, 101)
    pad = args.correlation_pad_samples
    received = np.zeros(reference.size + 2 * pad, dtype=np.complex64)
    received[pad : pad + reference.size] = reference
    received = apply_channel(
        received,
        sample_rate=float(config["modulation"]["sample_rate"]),
        rng=rng,
        noise_std=case.noise_std,
        freq_offset=args.freq_offset,
        phase_offset=args.phase_offset,
        timing_offset=args.timing_offset_samples,
    )

    result, _ = analyze_signal(
        config=config,
        reference=reference,
        received=received,
        sample_rate=float(config["modulation"]["sample_rate"]),
        label=f"simulated_{case.name}",
    )
    expected_offset = pad + int(round(args.timing_offset_samples))
    offset_error = result.sample_offset - expected_offset

    return {
        "stage": "correlation",
        "case": case.name,
        "noise_std": case.noise_std,
        "expected_offset": expected_offset,
        "detected_offset": result.sample_offset,
        "offset_error": offset_error,
        "normalized_peak": result.normalized_peak,
        "phase_deg": result.peak_phase_deg,
        "gold_peak": result.gold_peak,
        "pass": (
            abs(offset_error) <= args.correlation_offset_tolerance
            and result.normalized_peak >= args.correlation_peak_threshold
        ),
    }


def print_result(result: dict) -> None:
    headline = f"[{result['stage']}/{result['case']}]"
    fields = []
    for key, value in result.items():
        if key in {"stage", "case"}:
            continue
        fields.append(f"{key}={value}")
    print(headline, " ".join(fields))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate coherence between local SDR modules.")
    parser.add_argument(
        "--stage",
        choices=["modem", "gold", "sync", "full", "correlation", "all"],
        default="all",
        help="Which part of the pipeline to test.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["clean", "noisy"],
        default=["clean", "noisy"],
        help="Run clean and/or noisy simulations.",
    )
    parser.add_argument("--config", default="setup/config.yaml", help="Path to config file.")
    parser.add_argument("--message", default="Hello, SDR!", help="Payload used in modem/full tests.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Complex AWGN std for noisy cases.")
    parser.add_argument("--freq-offset", type=float, default=0.0, help="Channel frequency offset in Hz.")
    parser.add_argument("--phase-offset", type=float, default=0.0, help="Constant channel phase offset in radians.")
    parser.add_argument(
        "--timing-offset-samples",
        type=float,
        default=0.0,
        help="Integer-like sample shift applied before synchronization.",
    )
    parser.add_argument("--symbol-count", type=int, default=512, help="Number of random payload symbols for sync tests.")
    parser.add_argument("--guard-symbols", type=int, default=32, help="Zero-valued symbols added before and after the burst.")
    parser.add_argument("--sync-max-lag", type=int, default=64, help="Maximum symbol lag to consider when scoring sync.")
    parser.add_argument("--sync-ser-threshold", type=float, default=0.2, help="Pass threshold for sync symbol error rate.")
    parser.add_argument(
        "--correlation-pad-samples",
        type=int,
        default=128,
        help="Zero padding before and after the reference burst in correlation simulations.",
    )
    parser.add_argument(
        "--correlation-peak-threshold",
        type=float,
        default=0.85,
        help="Minimum normalized peak required for correlation pass.",
    )
    parser.add_argument(
        "--correlation-offset-tolerance",
        type=int,
        default=1,
        help="Maximum offset error in samples allowed for correlation pass.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable simulations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    selected_stages = ["modem", "gold", "sync", "full", "correlation"] if args.stage == "all" else [args.stage]

    cases = build_cases(args.cases, args.noise_std)
    rng = np.random.default_rng(args.seed)
    modem = ModulationProtocol(config)

    detector = GoldCodeDetector(config) if any(stage in {"gold", "full"} for stage in selected_stages) else None
    rrc_filter = RRCFilter(config) if any(stage in {"sync", "full"} for stage in selected_stages) else None
    synchronizer = Synchronizer(config) if any(stage in {"sync", "full"} for stage in selected_stages) else None
    datagram = make_datagram(args.message) if any(stage in {"modem", "gold", "full"} for stage in selected_stages) else None

    results: list[dict] = []

    for case in cases:
        for stage in selected_stages:
            if stage == "modem":
                results.append(run_modem_case(case, datagram, modem, rng))  # type: ignore[arg-type]
            elif stage == "gold":
                results.append(run_gold_case(case, datagram, modem, detector, rng))  # type: ignore[arg-type]
            elif stage == "sync":
                results.append(run_sync_case(case, modem, rrc_filter, synchronizer, rng, args))  # type: ignore[arg-type]
            elif stage == "full":
                results.append(
                    run_full_case(case, datagram, modem, detector, rrc_filter, synchronizer, rng, args)
                    # type: ignore[arg-type]
                )
            elif stage == "correlation":
                results.append(run_correlation_case(case, config, rng, args))

    for result in results:
        print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
