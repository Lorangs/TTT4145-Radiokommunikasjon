from pathlib import Path

from frame_layout_check import build_report, load_config, validate_layout


def test_frame_layout_and_gold_header_detection_pass():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(str(repo_root / "setup" / "config.yaml"))
    report = build_report(
        config=config,
        message="pilot header check",
        msg_id=11,
        guard_symbols=32,
    )

    failures = validate_layout(config, "pilot header check", report)

    assert failures == []
    assert report.header_detect_ok
    assert report.payload_recovery_ok
    assert report.guard_symbols_sufficient
