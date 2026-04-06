from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from yaml import safe_load

from convolutional_coder import ConvolutionalCoder
from datagram import Datagram, msgType
from filter import RRCFilter
from TX_pipeline import (
    bit_balance,
    build_frame_layout_summary,
    build_scrambler,
    build_tx_burst,
)
from forward_error_correction import FCCodec
from gold_detection import GoldCodeDetector, detect_gold_with_rotation
from interleaver import Interleaver
from modulation import normalize_config_modulation_name


@dataclass
class FrameLayoutReport:
    payload_bytes_used: int
    datagram_bytes: int
    fec_bytes: int
    fec_bits: int
    conv_tail_input_bits: int
    conv_output_bits: int
    sync_preamble_symbols: int
    payload_symbols: int
    channel_payload_symbols: int
    gold_header_symbols: int
    guard_symbols: int
    burst_symbols: int
    tx_samples: int
    rrc_taps: int
    rc_taps: int
    rc_group_delay_symbols: int
    guard_symbols_sufficient: bool
    detected_phase: int | None
    detected_header_index: int | None
    detected_score: float
    header_detect_ok: bool
    payload_recovery_ok: bool


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return safe_load(handle)

def build_report(
    config: dict,
    message: str,
    msg_id: int,
    guard_symbols: int,
) -> FrameLayoutReport:
    modulation_name = normalize_config_modulation_name(config)
    sps = int(config["modulation"]["samples_per_symbol"])

    datagram = Datagram.as_string(message, msg_id=np.uint8(msg_id), msg_type=msgType.DATA)
    payload_bytes_used = len(message.encode("utf-8"))

    fec = FCCodec(config)
    interleaver = Interleaver(config)
    conv = ConvolutionalCoder(config)
    scrambler = build_scrambler(config)
    detector = GoldCodeDetector(config)
    rrc = RRCFilter(config)
    tx_burst = build_tx_burst(
        config=config,
        datagram=datagram,
        modulation_name=modulation_name,
        samples_per_symbol=sps,
        gold_detector=detector,
        rrc_filter=rrc,
        fec=fec,
        interleaver=interleaver,
        conv_coder=conv,
        scrambler=scrambler,
        guard_symbols=guard_symbols,
        tx_peak_scale=1.0,
    )
    layout_summary = build_frame_layout_summary(
        config=config,
        datagram=datagram,
        modulation_name=modulation_name,
        gold_detector=detector,
        rrc_filter=rrc,
        fec=fec,
        interleaver=interleaver,
        conv_coder=conv,
        scrambler=scrambler,
        samples_per_symbol=sps,
        guard_symbols=guard_symbols,
    )
    packed = tx_burst.encoded.packed_bytes
    payload_symbols = tx_burst.encoded.payload_symbols
    channel_payload_symbols = tx_burst.channel_payload_symbols
    burst_symbols = tx_burst.burst_symbols
    tx_signal = tx_burst.tx_signal
    matched_signal = rrc.apply_filter(tx_signal).astype(np.complex64)
    expected_header_index = guard_symbols + int(tx_burst.sync_preamble_symbols.size)

    best_phase = None
    best_index = None
    best_score = -1.0
    best_rotation = 1 + 0j
    best_decisions = np.array([], dtype=np.complex64)

    for phase in range(sps):
        phase_symbols = matched_signal[phase::sps]
        index, score, rotation, decided = detect_gold_with_rotation(
            phase_symbols,
            detector,
            modulation_name,
            expected_index=expected_header_index,
            search_radius=0,
        )
        score_sort = (0 if index == expected_header_index else 1, -float(score))
        best_sort = (0 if best_index == expected_header_index else 1, -float(best_score))
        if best_phase is None or score_sort < best_sort:
            best_phase = phase
            best_index = index
            best_score = score
            best_rotation = rotation
            best_decisions = decided

    recovered_ok = False
    if best_index is not None:
        recovered_payload = detector.remove_gold_symbols(best_decisions, best_index)
        recovered_payload = recovered_payload[: payload_symbols.size]
        recovered_ok = bool(np.array_equal(recovered_payload, payload_symbols))

    rc_group_delay_samples = (len(rrc.rc_coefficients) - 1) // 2
    rc_group_delay_symbols = int(math.ceil(rc_group_delay_samples / sps))

    return FrameLayoutReport(
        payload_bytes_used=payload_bytes_used,
        datagram_bytes=int(packed.size),
        fec_bytes=int(tx_burst.encoded.fec_encoded_bytes.size),
        fec_bits=int(tx_burst.encoded.fec_bits.size),
        conv_tail_input_bits=int(conv.K - 1),
        conv_output_bits=int(tx_burst.encoded.conv_encoded_bits.size),
        sync_preamble_symbols=int(tx_burst.sync_preamble_symbols.size),
        payload_symbols=int(payload_symbols.size),
        channel_payload_symbols=int(channel_payload_symbols.size),
        gold_header_symbols=int(detector.gold_symbols.size),
        guard_symbols=int(guard_symbols),
        burst_symbols=int(burst_symbols.size),
        tx_samples=int(tx_signal.size),
        rrc_taps=int(len(rrc.coefficients)),
        rc_taps=int(len(rrc.rc_coefficients)),
        rc_group_delay_symbols=rc_group_delay_symbols,
        guard_symbols_sufficient=bool(layout_summary["guard_symbols_sufficient"]),
        detected_phase=best_phase,
        detected_header_index=best_index,
        detected_score=float(best_score),
        header_detect_ok=best_index == expected_header_index,
        payload_recovery_ok=recovered_ok,
    )


def validate_layout(
    config: dict,
    message: str,
    report: FrameLayoutReport,
) -> list[str]:
    failures: list[str] = []
    encoded_message = message.encode("utf-8")
    datagram = Datagram.as_string(message, msg_id=np.uint8(7), msg_type=msgType.DATA)
    packed = np.frombuffer(datagram.pack(), dtype=np.uint8)
    payload_start = Datagram.HEADER_SIZE
    payload_end = payload_start + len(encoded_message)

    if report.datagram_bytes != Datagram.TOTAL_SIZE:
        failures.append(
            f"Datagram size mismatch: expected {Datagram.TOTAL_SIZE}, got {report.datagram_bytes}"
        )

    payload_length_index = (
        Datagram.MSG_ID_SIZE + Datagram.MSG_TYPE_SIZE + Datagram.TIMESTAMP_SIZE
    )
    if int(packed[payload_length_index]) != len(encoded_message):
        failures.append("Payload length field does not match the source message length.")

    if not np.array_equal(packed[payload_start:payload_end], np.frombuffer(encoded_message, dtype=np.uint8)):
        failures.append("Payload bytes in datagram do not match the source message.")

    if not np.all(packed[payload_end:] == Datagram.PAD_BYTE):
        failures.append("Payload padding is not zero-filled after the message bytes.")

    if report.payload_symbols != report.conv_output_bits:
        failures.append(
            f"Payload symbol count mismatch: expected {report.conv_output_bits}, got {report.payload_symbols}"
        )

    if report.channel_payload_symbols != report.payload_symbols:
        failures.append(
            f"Channel payload symbol count mismatch: expected {report.payload_symbols}, got {report.channel_payload_symbols}"
        )

    if not report.guard_symbols_sufficient:
        failures.append(
            "Guard region is shorter than the combined TX/RX pulse-shaping ramp."
        )

    if not report.header_detect_ok:
        failures.append(
            f"Gold header detection failed: expected index {report.guard_symbols + report.sync_preamble_symbols}, "
            f"got {report.detected_header_index}"
        )

    if not report.payload_recovery_ok:
        failures.append("Payload symbols were not recovered cleanly after Gold removal.")

    return failures


def print_report(config: dict, report: FrameLayoutReport) -> None:
    modulation_name = normalize_config_modulation_name(config)
    print("Frame layout check")
    print(f"  modulation: {modulation_name}")
    print(f"  payload_bytes_used: {report.payload_bytes_used}")
    print(f"  datagram_bytes: {report.datagram_bytes}")
    print(f"  fec_bytes: {report.fec_bytes}")
    print(f"  fec_bits: {report.fec_bits}")
    print(f"  convolutional_tail_bits: {report.conv_tail_input_bits}")
    print(f"  convolutional_output_bits: {report.conv_output_bits}")
    print(f"  sync_preamble_symbols: {report.sync_preamble_symbols}")
    print(f"  payload_symbols: {report.payload_symbols}")
    print(f"  channel_payload_symbols: {report.channel_payload_symbols}")
    print(f"  gold_header_symbols: {report.gold_header_symbols}")
    print(f"  guard_symbols: {report.guard_symbols}")
    print(f"  burst_symbols: {report.burst_symbols}")
    print(f"  tx_samples: {report.tx_samples}")
    print(f"  rrc_taps: {report.rrc_taps}")
    print(f"  rc_taps: {report.rc_taps}")
    print(f"  rc_group_delay_symbols: {report.rc_group_delay_symbols}")
    print(f"  guard_symbols_sufficient: {report.guard_symbols_sufficient}")
    print(f"  detected_phase: {report.detected_phase}")
    print(f"  detected_header_index: {report.detected_header_index}")
    print(f"  detected_score: {report.detected_score:.4f}")
    print(f"  header_detect_ok: {report.header_detect_ok}")
    print(f"  payload_recovery_ok: {report.payload_recovery_ok}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the fixed datagram layout and Gold-header framing."
    )
    parser.add_argument("--config", default="setup/config.yaml", help="Path to YAML config.")
    parser.add_argument("--message", default="frame layout check", help="Payload text to pack.")
    parser.add_argument("--msg-id", type=int, default=7, help="Datagram message ID to use.")
    parser.add_argument(
        "--guard-symbols",
        type=int,
        default=32,
        help="Zero guard symbols to place before and after the framed burst.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    report = build_report(
        config=config,
        message=args.message,
        msg_id=args.msg_id,
        guard_symbols=args.guard_symbols,
    )
    print_report(config, report)
    failures = validate_layout(config, args.message, report)
    if failures:
        print("  result: FAIL")
        for failure in failures:
            print(f"  failure: {failure}")
        return 1

    print("  result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
