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
        RX buffer size used by this tester. Default: `65536`
    --flush-buffers:
        Number of RX buffers to discard before the measured capture. Default: `3`
    --plots:
        If set, show diagnostic plots at the end of the run.
"""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy

import numpy as np
from yaml import safe_load

from datagram import Datagram, msgType
from filter import BWLPFilter, RRCFilter
from gold_detection import GoldCodeDetector
from modulation import ModulationProtocol
from forward_error_correction import FCCodec
from convolutional_coder import ConvolutionalCoder
from synchronize import Synchronizer
from sdr_transciever import SDRTransciever


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return safe_load(handle)


def setup_logging(config: dict) -> None:
    level_name = str(config["radio"].get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.core").setLevel(logging.WARNING)
    logging.getLogger("llvmlite").setLevel(logging.WARNING)


def print_runtime_summary(config: dict, args: argparse.Namespace) -> None:
    print("Hardware test configuration")
    print(f"  radio_ip: {config['radio']['ip_address']}")
    print(f"  tx_carrier_hz: {int(float(config['transmitter']['tx_carrier']))}")
    print(f"  rx_carrier_hz: {int(float(config['receiver']['rx_carrier']))}")
    print(f"  sample_rate_hz: {int(float(config['modulation']['sample_rate']))}")
    print(f"  modulation: {str(config['modulation']['type']).upper().strip()}")
    print(f"  rx_buffer_size: {int(config['receiver']['buffer_size'])}")
    print(f"  flush_buffers: {args.flush_buffers}")
    print(f"  payload: {args.payload!r}")


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


def bits_to_symbols(bits: np.ndarray, modulation_type: str) -> np.ndarray:
    modulation_type = modulation_type.upper().strip()
    if modulation_type == "BPSK":
        return (1 - 2 * bits).astype(np.complex64)

    if modulation_type == "QPSK":
        # Pad if odd length
        if bits.size % 2 == 1:
            bits = np.concatenate((bits, np.array([0], dtype=np.uint8)))
        bit_pairs = bits.reshape(-1, 2)
        I = (1 - 2 * bit_pairs[:, 0]).astype(np.float32)
        Q = (1 - 2 * bit_pairs[:, 1]).astype(np.float32)
        return (I + 1j * Q).astype(np.complex64)

    raise ValueError(f"Unsupported modulation type: {modulation_type}")


def ensure_gold_config(config: dict) -> dict:
    cfg = deepcopy(config)
    gold_cfg = cfg.setdefault("gold_sequence", {})
    gold_cfg.setdefault("code_length", 31)
    gold_cfg.setdefault("code_index", 0)
    if "correlation_scale_factor_threshold" not in gold_cfg:
        gold_cfg["correlation_scale_factor_threshold"] = gold_cfg.get("correlation_threshold", 0.7)
    return cfg


def configure_hardware_tester(config: dict, rx_buffer_size: int) -> dict:
    cfg = ensure_gold_config(config)
    cfg["receiver"]["buffer_size"] = int(rx_buffer_size)
    return cfg


def run_pipeline_test(
    modem: ModulationProtocol,
    fec: FCCodec,
    interleaver: Interleaver,
    conv_coder: ConvolutionalCoder,
    payload_text: str,
) -> tuple[Datagram, dict]:
    datagram = Datagram.as_string(payload_text, msg_type=msgType.DATA)

    # TX-side pipeline: DATAGRAM -> FEC -> SCRAMBLER -> INTERLEAVER -> CONVOLUTIONAL ENCODER
    data_bytes = np.frombuffer(datagram.pack(), dtype=np.uint8)
    fec_encoded = fec.encode(data_bytes)
    bits = np.unpackbits(fec_encoded)
    scrambled_bits = modem.scramble_bits(bits)

    interleaved_bits = interleaver.interleave(scrambled_bits)

    conv_encoded_bits = conv_coder.encode(interleaved_bits)

    # RX-side pipeline: CONVOLUTIONAL DECODER -> DEINTERLEAVER -> DESCRAMBLER -> FEC -> DATAGRAM
    conv_decoded_bits = conv_coder.decode(conv_encoded_bits)
    deinterleaved_bits = interleaver.deinterleave(conv_decoded_bits)
    descrambled_bits = modem.descramble_bits(deinterleaved_bits)

    try:
        fec_decoded_bytes = fec.rs_decode(np.packbits(descrambled_bits))
        if fec_decoded_bytes is None:
            raise ValueError("FEC decoding failed")

        recovered_datagram = Datagram.unpack(fec_decoded_bytes.tobytes())
        recovered_text = recovered_datagram.get_payload.tobytes().decode("utf-8", errors="replace")
        roundtrip_ok = (
            recovered_datagram.get_msg_id == datagram.get_msg_id
            and recovered_datagram.get_msg_type == datagram.get_msg_type
            and np.array_equal(recovered_datagram.get_payload, datagram.get_payload)
        )
        error_message = ""
    except Exception as exc:
        recovered_text = ""
        roundtrip_ok = False
        error_message = str(exc)

    zeros, ones = bit_balance(scrambled_bits)
    changed = int(np.count_nonzero(scrambled_bits != bits))

    result = {
        "scrambler_enabled": bool(modem.scrambler_enable),
        "bit_count": int(bits.size),
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
    fec: FCCodec,
    interleaver: Interleaver,
    conv_coder: ConvolutionalCoder,
    datagram: Datagram,
    guard_symbols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # DATAGRAM -> FEC -> SCRAMBLER -> INTERLEAVER -> CONVOLUTIONAL ENCODER -> MODULATOR
    data_bytes = np.frombuffer(datagram.pack(), dtype=np.uint8)
    fec_encoded = fec.encode(data_bytes)
    bits = np.unpackbits(fec_encoded)
    scrambled_bits = modem.scramble_bits(bits)

    interleaved_bits = interleaver.interleave(scrambled_bits)
    conv_encoded_bits = conv_coder.encode(interleaved_bits)

    payload_symbols = bits_to_symbols(conv_encoded_bits, modem.modulation_type)
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
    fec: FCCodec,
    interleaver: Interleaver,
    conv_coder: ConvolutionalCoder,
    synchronizer: Synchronizer,
    sdr: SDRTransciever,
    datagram: Datagram,
    flush_buffers: int,
) -> tuple[dict, dict]:
    payload_symbols, burst_symbols, tx_signal = build_tx_burst(
        modem=modem,
        gold_detector=gold_detector,
        rrc_filter=rrc_filter,
        fec=fec,
        interleaver=interleaver,
        conv_coder=conv_coder,
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
    coarse_signal = synchronizer.coarse_frequenzy_synchronization(frontend_filtered_signal)
    if coarse_signal is None:
        raise RuntimeError("Coarse frequency sync failed (signal too weak)")

    matched_signal = rrc_filter.apply_filter(coarse_signal)
    timing_signal = synchronizer.gardner_timing_synchronization(matched_signal)
    fine_signal = synchronizer.fine_frequenzy_synchronization(timing_signal)

    phase_candidates = rank_gold_phase_candidates(
        fine_signal,
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

        # Demodulation + decoding pipeline
        try:
            # recover symbol decisions, then bitstream
            rx_bits = modem.decision_bits_from_symbols(payload_rx)

            conv_decoded = conv_coder.decode(rx_bits)
            deinterleaved = interleaver.deinterleave(conv_decoded)
            descrambled = modem.descramble_bits(deinterleaved)
            fec_decoded = fec.rs_decode(np.packbits(descrambled))

            if fec_decoded is None:
                raise ValueError("FEC decode failed")

            recovered_datagram = Datagram.unpack(fec_decoded.tobytes())
            recovered_text_candidate = recovered_datagram.get_payload.tobytes().decode("utf-8", errors="replace")

            if (
                recovered_datagram.get_msg_id == datagram.get_msg_id
                and recovered_datagram.get_msg_type == datagram.get_msg_type
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
                break

        except Exception as exc:
            decode_error = str(exc)
            continue

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
        "coarse_signal": coarse_signal,
        "matched_signal": matched_signal,
        "timing_signal": timing_signal,
        "fine_signal": fine_signal,
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
    parser.add_argument("--payload", default="hardware test ", help="Payload text to transmit.")
    parser.add_argument(
        "--rx-buffer-size",
        type=int,
        default=65536,
        help="RX buffer size used by the hardware tester.",
    )
    parser.add_argument("--flush-buffers", type=int, default=3, help="Number of RX buffers to discard before capture.")
    parser.add_argument("--plots", action="store_true", help="Show diagnostic plots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from interleaver import Interleaver

    config = configure_hardware_tester(load_config(args.config), args.rx_buffer_size)
    setup_logging(config)
    print_runtime_summary(config, args)

    modem = ModulationProtocol(config)
    rrc_filter = RRCFilter(config)
    gold_detector = GoldCodeDetector(config)
    fec = FCCodec(config)
    interleaver = Interleaver(config)
    conv_coder = ConvolutionalCoder(config, warmup=False, use_numba=False)
    synchronizer = Synchronizer(config, warmup=False, use_numba=False)
    sdr = SDRTransciever(config)

    datagram, pipeline_result = run_pipeline_test(
        modem=modem,
        fec=fec,
        interleaver=interleaver,
        conv_coder=conv_coder,
        payload_text=args.payload,
    )
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

        gold_result, traces = run_gold_hardware_test(
            config=config,
            modem=modem,
            gold_detector=gold_detector,
            rrc_filter=rrc_filter,
            fec=fec,
            interleaver=interleaver,
            conv_coder=conv_coder,
            synchronizer=synchronizer,
            sdr=sdr,
            datagram=datagram,
            flush_buffers=args.flush_buffers,
        )
        print_result_block("Gold code hardware test", gold_result)

        if args.plots:
            import matplotlib.pyplot as plt
            from sdr_plots import StaticSDRPlotter

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
