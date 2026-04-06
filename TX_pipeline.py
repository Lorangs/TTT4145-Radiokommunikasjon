from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from convolutional_coder import ConvolutionalCoder
from datagram import Datagram
from frame_sequences import build_sync_preamble_symbols
from filter import RRCFilter
from forward_error_correction import FCCodec
from gold_detection import GoldCodeDetector
from interleaver import Interleaver
from modulation import (
    bits_to_symbols,
    modulation_rotations,
    nearest_constellation_symbols,
    symbols_to_bits,
    upsample_symbols,
)
from scrambler import LFSRScrambler


@dataclass
class EncodedFrame:
    packed_bytes: np.ndarray
    fec_encoded_bytes: np.ndarray
    fec_bits: np.ndarray
    scrambled_bits: np.ndarray
    interleaved_bits: np.ndarray
    conv_encoded_bits: np.ndarray
    payload_symbols: np.ndarray


@dataclass
class TxBurst:
    encoded: EncodedFrame
    sync_preamble_symbols: np.ndarray
    channel_payload_symbols: np.ndarray
    framed_symbols: np.ndarray
    burst_symbols: np.ndarray
    upsampled_symbols: np.ndarray
    tx_signal: np.ndarray


def build_scrambler(config: dict) -> LFSRScrambler:
    return LFSRScrambler(
        register_length=int(config["coding"].get("scrambler_register_length", 7))
    )


def bit_balance(bits: np.ndarray) -> tuple[int, int]:
    ones = int(np.count_nonzero(bits))
    zeros = int(bits.size - ones)
    return zeros, ones


def encode_datagram(
    datagram: Datagram,
    modulation_name: str,
    fec: FCCodec,
    interleaver: Interleaver,
    conv_coder: ConvolutionalCoder,
    scrambler: LFSRScrambler,
) -> EncodedFrame:
    packed_bytes = np.frombuffer(datagram.pack(), dtype=np.uint8)
    fec_encoded_bytes = fec.encode(packed_bytes)
    fec_bits = np.unpackbits(fec_encoded_bytes)
    scrambled_bits = scrambler.apply(fec_bits)
    interleaved_bits = interleaver.interleave(scrambled_bits)
    conv_encoded_bits = conv_coder.encode(interleaved_bits)
    payload_symbols = bits_to_symbols(conv_encoded_bits, modulation_name)

    return EncodedFrame(
        packed_bytes=packed_bytes,
        fec_encoded_bytes=fec_encoded_bytes,
        fec_bits=fec_bits,
        scrambled_bits=scrambled_bits,
        interleaved_bits=interleaved_bits,
        conv_encoded_bits=conv_encoded_bits,
        payload_symbols=payload_symbols,
    )


def decode_payload_symbols(
    payload_symbols: np.ndarray,
    modulation_name: str,
    fec: FCCodec,
    interleaver: Interleaver,
    conv_coder: ConvolutionalCoder,
    scrambler: LFSRScrambler,
) -> Datagram:
    rx_bits = symbols_to_bits(payload_symbols, modulation_name)
    conv_decoded = conv_coder.decode(rx_bits)
    deinterleaved = interleaver.deinterleave(conv_decoded)
    descrambled = scrambler.apply(deinterleaved)
    fec_decoded = fec.rs_decode(np.packbits(descrambled))
    if fec_decoded is None:
        detail = getattr(fec, "last_decode_error", "")
        if detail:
            raise ValueError(f"FEC decode failed: {detail}")
        raise ValueError("FEC decode failed")
    return Datagram.unpack(fec_decoded.tobytes())


def build_tx_burst(
    config: dict,
    datagram: Datagram,
    modulation_name: str,
    samples_per_symbol: int,
    gold_detector: GoldCodeDetector,
    rrc_filter: RRCFilter,
    fec: FCCodec,
    interleaver: Interleaver,
    conv_coder: ConvolutionalCoder,
    scrambler: LFSRScrambler,
    guard_symbols: int,
    tx_peak_scale: float = 0.6 * (2**14),
) -> TxBurst:
    encoded = encode_datagram(
        datagram=datagram,
        modulation_name=modulation_name,
        fec=fec,
        interleaver=interleaver,
        conv_coder=conv_coder,
        scrambler=scrambler,
    )

    sync_preamble_symbols = build_sync_preamble_symbols(config, modulation_name)
    # The over-the-air burst is now intentionally simple again:
    # guard | repeated timing preamble | repeated Gold header | coded payload | guard
    channel_payload_symbols = encoded.payload_symbols.astype(np.complex64, copy=False)
    framed_symbols = np.concatenate(
        (
            sync_preamble_symbols,
            gold_detector.add_gold_symbols(channel_payload_symbols).astype(np.complex64),
        )
    ).astype(np.complex64, copy=False)
    guard = np.zeros(guard_symbols, dtype=np.complex64)
    burst_symbols = np.concatenate((guard, framed_symbols, guard))
    upsampled = upsample_symbols(burst_symbols, samples_per_symbol)

    tx_signal = rrc_filter.apply_filter(upsampled).astype(np.complex64)
    peak = max(float(np.max(np.abs(tx_signal))), 1e-12)
    # Normalize once here so the hardware tester and frame-layout checks see the
    # same shaped burst, only scaled to the requested TX amplitude.
    tx_signal = (tx_peak_scale * tx_signal / peak).astype(np.complex64)

    return TxBurst(
        encoded=encoded,
        sync_preamble_symbols=sync_preamble_symbols,
        channel_payload_symbols=channel_payload_symbols,
        framed_symbols=framed_symbols,
        burst_symbols=burst_symbols,
        upsampled_symbols=upsampled,
        tx_signal=tx_signal,
    )


def build_frame_layout_summary(
    config: dict,
    datagram: Datagram,
    modulation_name: str,
    gold_detector: GoldCodeDetector,
    rrc_filter: RRCFilter,
    fec: FCCodec,
    interleaver: Interleaver,
    conv_coder: ConvolutionalCoder,
    scrambler: LFSRScrambler,
    samples_per_symbol: int,
    guard_symbols: int,
) -> dict:
    encoded = encode_datagram(
        datagram=datagram,
        modulation_name=modulation_name,
        fec=fec,
        interleaver=interleaver,
        conv_coder=conv_coder,
        scrambler=scrambler,
    )
    sync_preamble_symbols = build_sync_preamble_symbols(config, modulation_name)
    channel_payload_symbols = encoded.payload_symbols.astype(np.complex64, copy=False)
    preamble_symbols = int(sync_preamble_symbols.size)
    gold_header_symbols = int(gold_detector.gold_symbols.size)
    burst_symbols = int(
        preamble_symbols + channel_payload_symbols.size + gold_header_symbols + 2 * guard_symbols
    )
    rc_group_delay_samples = (len(rrc_filter.rc_coefficients) - 1) // 2
    rc_group_delay_symbols = int(math.ceil(rc_group_delay_samples / samples_per_symbol))

    return {
        "payload_length_bytes": int(datagram.get_payload_length),
        "payload_capacity_bytes": Datagram.PAYLOAD_SIZE,
        "packed_datagram_bytes": int(encoded.packed_bytes.size),
        "fec_bytes": int(encoded.fec_encoded_bytes.size),
        "fec_bits": int(encoded.fec_bits.size),
        "scrambled_bits": int(encoded.scrambled_bits.size),
        "interleaved_bits": int(encoded.interleaved_bits.size),
        "conv_tail_bits": int(conv_coder.K - 1),
        "conv_output_bits": int(encoded.conv_encoded_bits.size),
        "sync_preamble_symbols": preamble_symbols,
        "payload_symbols": int(encoded.payload_symbols.size),
        "channel_payload_symbols": int(channel_payload_symbols.size),
        "gold_header_symbols": gold_header_symbols,
        "guard_symbols": int(guard_symbols),
        "burst_symbols": burst_symbols,
        "rrc_taps": int(len(rrc_filter.coefficients)),
        "rc_taps": int(len(rrc_filter.rc_coefficients)),
        "rc_group_delay_symbols": rc_group_delay_symbols,
        "guard_symbols_sufficient": bool(guard_symbols >= rc_group_delay_symbols),
    }
