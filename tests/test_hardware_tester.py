from types import SimpleNamespace

import numpy as np
import pytest

import Hardware_tester


def test_run_gold_hardware_test_flushes_rx_before_one_shot_tx(monkeypatch):
    call_order: list[str] = []
    flush_capture = np.array([1.0 + 0.0j], dtype=np.complex64)
    burst_capture = np.array([2.0 + 0.0j], dtype=np.complex64)

    class FakePluto:
        def __init__(self) -> None:
            self.tx_started = False

        def rx(self) -> np.ndarray:
            call_order.append("rx")
            return burst_capture if self.tx_started else flush_capture

        def tx_destroy_buffer(self) -> None:
            call_order.append("tx_destroy_buffer")

        def tx(self, samples: np.ndarray) -> None:
            np.testing.assert_array_equal(samples, np.array([0.25 + 0.0j], dtype=np.complex64))
            call_order.append("tx")
            self.tx_started = True

    fake_pluto = FakePluto()
    fake_sdr = SimpleNamespace(sdr=fake_pluto)

    monkeypatch.setattr(
        Hardware_tester,
        "build_tx_burst",
        lambda **kwargs: SimpleNamespace(
            encoded=SimpleNamespace(payload_symbols=np.array([1 + 0j], dtype=np.complex64)),
            channel_payload_symbols=np.array([1 + 0j], dtype=np.complex64),
            burst_symbols=np.array([1 + 0j], dtype=np.complex64),
            tx_signal=np.array([0.25 + 0.0j], dtype=np.complex64),
        ),
    )

    def fake_run_rx_pipeline(*, received_signal, **kwargs):
        np.testing.assert_array_equal(received_signal, burst_capture)
        assert call_order == ["rx", "rx", "tx_destroy_buffer", "tx", "rx"]
        raise RuntimeError("stop after ordering check")

    monkeypatch.setattr(Hardware_tester, "run_rx_pipeline", fake_run_rx_pipeline)

    with pytest.raises(RuntimeError, match="stop after ordering check"):
        Hardware_tester.run_gold_hardware_test(
            config={"modulation": {"samples_per_symbol": 8}},
            modulation_name="BPSK",
            gold_detector=SimpleNamespace(),
            rrc_filter=SimpleNamespace(),
            fec=SimpleNamespace(),
            interleaver=SimpleNamespace(),
            conv_coder=SimpleNamespace(),
            scrambler=SimpleNamespace(),
            synchronizer=SimpleNamespace(),
            sdr=fake_sdr,
            datagram=SimpleNamespace(),
            guard_symbols=32,
            flush_buffers=2,
        )


def test_phase_align_symbols_for_decision_axis_rotates_bpsk_symbols_to_real_axis():
    symbols = np.array([1, -1, 1, -1], dtype=np.complex64) * np.exp(1j * 0.35)

    diagnostics = Hardware_tester.phase_align_symbols_for_decision_axis(
        symbols,
        "BPSK",
    )

    aligned = diagnostics["aligned_signal"]
    projected = diagnostics["projected_signal"]
    residual_phase = diagnostics["residual_phase_rad"]

    assert aligned.shape == symbols.shape
    assert projected.shape == (symbols.size,)
    assert residual_phase.shape == (symbols.size,)
    assert np.max(np.abs(aligned.imag)) < 1e-4
    assert np.all(projected[[0, 2]] > 0.0)
    assert np.all(projected[[1, 3]] < 0.0)


def test_build_decision_diagnostic_traces_returns_pre_post_and_final_views():
    traces = {
        "carrier_aligned_signal": np.array([1.0 + 0.3j, -1.0 - 0.2j], dtype=np.complex64),
        "equalized_signal": np.array([0.9 + 0.1j, -1.1 - 0.1j], dtype=np.complex64),
        "fine_complex_signal": np.array([1.0 + 0.05j, -1.0 - 0.05j], dtype=np.complex64),
    }

    diagnostics = Hardware_tester.build_decision_diagnostic_traces(
        traces,
        "BPSK",
    )

    expected_keys = {
        "pre_equalizer_decision_signal",
        "pre_equalizer_projected_signal",
        "post_equalizer_decision_signal",
        "post_equalizer_projected_signal",
        "final_decision_signal",
        "final_projected_signal",
        "final_residual_phase_rad",
        "final_decision_symbols",
        "final_global_phase_rad",
    }

    assert expected_keys.issubset(diagnostics.keys())
    assert diagnostics["pre_equalizer_decision_signal"].shape == traces["carrier_aligned_signal"].shape
    assert diagnostics["post_equalizer_decision_signal"].shape == traces["equalized_signal"].shape
    assert diagnostics["final_decision_signal"].shape == traces["fine_complex_signal"].shape
    assert diagnostics["final_projected_signal"].shape == (2,)


def test_trim_gold_removed_payload_keeps_exact_coded_payload_length():
    payload_suffix = np.array(
        [1 + 0j, -1 + 0j, 1 + 0j, -1 + 0j, 1 + 0j],
        dtype=np.complex64,
    )

    trimmed = Hardware_tester.trim_gold_removed_payload(
        payload_suffix,
        expected_symbol_count=3,
    )

    np.testing.assert_array_equal(
        trimmed,
        np.array([1 + 0j, -1 + 0j, 1 + 0j], dtype=np.complex64),
    )
    assert trimmed.size == 3
