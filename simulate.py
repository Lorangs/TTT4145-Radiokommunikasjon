"""
Standalone simulation harness for testing module coherence without SDR hardware.

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

from barker_detection import BarkerDetector
from datagram import Datagram, msgType
from filter import RRCFilter
from modulation import ModulationProtocol
from synchronize import Synchronizer


@dataclass
class SimulationCase:
    name: str
    noise_std: float


def patch_datagram_pack() -> None:
    def _patched_pack(self: Datagram) -> bytes:
        return (
            int(self._msg_id).to_bytes(2, byteorder="big")
            + bytes([self._msg_type.value])
            + bytes([self._payload_size])
            + self._payload.tobytes()
            + int(self._crc16).to_bytes(2, byteorder="big")
        )

    Datagram.pack = _patched_pack


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return safe_load(handle)


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


def decode_symbols_to_bits(symbols: np.ndarray, modulation_type: str) -> np.ndarray:
    modulation_type = modulation_type.upper().strip()
    if modulation_type == "BPSK":
        return (symbols.real > 0).astype(np.uint8)
    if modulation_type == "QPSK":
        i_bits = (symbols.real < 0).astype(np.uint8)
        q_bits = (symbols.imag < 0).astype(np.uint8)
        bits = np.empty(symbols.size * 2, dtype=np.uint8)
        bits[0::2] = i_bits
        bits[1::2] = q_bits
        return bits
    raise ValueError(f"Unsupported modulation type: {modulation_type}")


def compat_unpack_datagram(raw_bytes: bytes) -> Datagram:
    min_length = 6
    if len(raw_bytes) < min_length:
        raise ValueError("Data length is too short to be a valid packed datagram.")

    msg_id = np.uint8(raw_bytes[1])
    msg_type = msgType(raw_bytes[2])
    payload_size = raw_bytes[3]
    expected_length = min_length + payload_size
    if len(raw_bytes) != expected_length:
        raise ValueError(
            f"Packed datagram length mismatch: got {len(raw_bytes)}, expected {expected_length}."
        )

    payload = np.frombuffer(raw_bytes[4 : 4 + payload_size], dtype=np.uint8).copy()
    received_crc = int.from_bytes(raw_bytes[4 + payload_size : 6 + payload_size], "big")
    datagram = Datagram(msg_id=msg_id, msg_type=msg_type, payload=payload)
    if int(datagram.get_crc16) != received_crc:
        raise ValueError("CRC16 checksum does not match.")
    return datagram


def recover_datagram_from_symbols(
    symbols: np.ndarray,
    reference_datagram: Datagram,
    modulation_type: str,
) -> Datagram:
    bits = decode_symbols_to_bits(symbols, modulation_type)
    reference_bit_count = len(reference_datagram.pack()) * 8
    packed = np.packbits(bits[:reference_bit_count]).tobytes()
    return compat_unpack_datagram(packed)


def make_datagram(message: str) -> Datagram:
    payload = np.frombuffer(message.encode("utf-8"), dtype=np.uint8)
    return Datagram(msg_id=np.uint8(101), msg_type=msgType.DATA, payload=payload)


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
    try:
        recovered = recover_datagram_from_symbols(rx_symbols, datagram, modem.modulation_type)
        recovered_text = recovered.get_payload.tobytes().decode("utf-8", errors="replace")
        compat_ok = recovered_text == datagram.get_payload.tobytes().decode("utf-8", errors="replace")
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
        "native_roundtrip_ok": native_ok,
        "native_error": native_error,
        "compat_roundtrip_ok": compat_ok,
        "compat_error": compat_error,
        "recovered_text": recovered_text,
    }


def run_barker_case(
    case: SimulationCase,
    datagram: Datagram,
    modem: ModulationProtocol,
    detector: BarkerDetector,
    rng: np.random.Generator,
) -> dict:
    payload_symbols = modem.modulate_message(datagram).astype(np.complex64)
    tx = detector.add_barker_symbols(payload_symbols)
    rx = add_awgn(tx, case.noise_std, rng)

    # Disable adaptive thresholding so clean and synthetic noisy cases can be detected directly.
    detector.set_noise_floor_dB(0.0)
    start_index = detector.detect(rx)
    remove_ok = False
    if start_index is not None:
        recovered = detector.remove_barker_symbols(rx, start_index)
        trimmed = recovered[: payload_symbols.size]
        decisions = nearest_constellation_symbols(trimmed, modem.modulation_type)
        remove_ok = bool(np.array_equal(decisions, payload_symbols))

    return {
        "stage": "barker",
        "case": case.name,
        "noise_std": case.noise_std,
        "detected": start_index is not None,
        "detected_index": start_index,
        "remove_ok": remove_ok,
        "sequence_length": int(len(detector.barker_symbols)),
    }


def run_sync_case(
    case: SimulationCase,
    modem: ModulationProtocol,
    rrc_filter: RRCFilter,
    synchronizer: Synchronizer,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict:
    symbol_count = args.symbol_count
    modulation = modem.modulation_type.upper().strip()
    if modulation == "BPSK":
        source_symbols = rng.choice(np.array([-1, 1], dtype=np.int8), size=symbol_count).astype(np.complex64)
    else:
        i = rng.choice(np.array([-1, 1], dtype=np.int8), size=symbol_count)
        q = rng.choice(np.array([-1, 1], dtype=np.int8), size=symbol_count)
        source_symbols = (i + 1j * q).astype(np.complex64)

    upsampled = modem.upsample_symbols(source_symbols)
    tx_signal = rrc_filter.apply_filter(upsampled)

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
    timed = synchronizer.time_synchronization(matched)
    fine = synchronizer.fine_frequenzy_synchronization(timed)

    transient = max(2, rrc_filter.filter_span // 2)
    usable_rx = fine[transient:]
    ser, compared, lag = best_aligned_symbol_error_rate(
        usable_rx,
        source_symbols,
        modem.modulation_type,
        max_lag=max(16, rrc_filter.filter_span * 2),
    )

    return {
        "stage": "sync",
        "case": case.name,
        "noise_std": case.noise_std,
        "symbols_compared": compared,
        "symbol_error_rate": ser,
        "best_lag": lag,
        "timing_output_symbols": int(fine.size),
        "pass": compared > 0 and ser <= args.sync_ser_threshold,
    }


def run_full_case(
    case: SimulationCase,
    datagram: Datagram,
    modem: ModulationProtocol,
    detector: BarkerDetector,
    rrc_filter: RRCFilter,
    synchronizer: Synchronizer,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict:
    tx_symbols = modem.modulate_message(datagram).astype(np.complex64)
    framed_symbols = detector.add_barker_symbols(tx_symbols)
    upsampled = modem.upsample_symbols(framed_symbols)
    tx_signal = rrc_filter.apply_filter(upsampled)

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
    timed = synchronizer.time_synchronization(matched)
    fine = synchronizer.fine_frequenzy_synchronization(timed)

    detector.set_noise_floor_dB(0.0)
    start_index = detector.detect(fine)
    recovered_text = ""
    compat_ok = False
    compat_error = ""
    if start_index is not None:
        payload_rx = detector.remove_barker_symbols(fine, start_index)
        payload_rx = payload_rx[: tx_symbols.size]
        try:
            recovered = recover_datagram_from_symbols(payload_rx, datagram, modem.modulation_type)
            recovered_text = recovered.get_payload.tobytes().decode("utf-8", errors="replace")
            compat_ok = recovered_text == datagram.get_payload.tobytes().decode("utf-8", errors="replace")
        except Exception as exc:
            compat_error = str(exc)

    return {
        "stage": "full",
        "case": case.name,
        "noise_std": case.noise_std,
        "detected": start_index is not None,
        "detected_index": start_index,
        "compat_roundtrip_ok": compat_ok,
        "compat_error": compat_error,
        "recovered_text": recovered_text,
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
        choices=["modem", "barker", "sync", "full", "all"],
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
    parser.add_argument("--sync-ser-threshold", type=float, default=0.2, help="Pass threshold for sync symbol error rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable simulations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patch_datagram_pack()
    config = load_config(args.config)

    modem = ModulationProtocol(config)
    detector = BarkerDetector(config)
    rrc_filter = RRCFilter(config)
    synchronizer = Synchronizer(config)
    datagram = make_datagram(args.message)
    cases = build_cases(args.cases, args.noise_std)
    rng = np.random.default_rng(args.seed)

    selected_stages = ["modem", "barker", "sync", "full"] if args.stage == "all" else [args.stage]

    results: list[dict] = []
    for case in cases:
        for stage in selected_stages:
            if stage == "modem":
                results.append(run_modem_case(case, datagram, modem, rng))
            elif stage == "barker":
                results.append(run_barker_case(case, datagram, modem, detector, rng))
            elif stage == "sync":
                results.append(run_sync_case(case, modem, rrc_filter, synchronizer, rng, args))
            elif stage == "full":
                results.append(
                    run_full_case(case, datagram, modem, detector, rrc_filter, synchronizer, rng, args)
                )

    for result in results:
        print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
