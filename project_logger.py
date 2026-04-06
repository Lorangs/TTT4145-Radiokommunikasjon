from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_NOISY_LOGGERS = (
    "matplotlib",
    "matplotlib.font_manager",
    "matplotlib.pyplot",
    "PIL",
    "pyqtgraph",
    "numba",
    "numba.core",
    "llvmlite",
)


class _ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self.COLORS.get(record.levelname, "")
        if not color:
            return message
        return f"{color}{message}{self.RESET}"


def parse_log_level(level_name: str | None) -> int:
    normalized = str(level_name or "INFO").upper().strip()
    return _LEVEL_MAP.get(normalized, logging.INFO)


def get_configured_log_level(config: dict, default: str = "INFO") -> str:
    logging_cfg = config.get("logging", {})
    if "level" in logging_cfg:
        return str(logging_cfg.get("level", default)).upper().strip()
    return str(config.get("radio", {}).get("log_level", default)).upper().strip()


def get_default_log_dir() -> Path:
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def build_log_path(session_name: str, log_dir: str | os.PathLike[str] | None = None) -> Path:
    target_dir = Path(log_dir) if log_dir is not None else get_default_log_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{datetime.now().date()}-{session_name}.log"


def configure_project_logging(
    *,
    level_name: str | None = "INFO",
    session_name: str = "debug",
    log_file: str | os.PathLike[str] | None = None,
    console: bool = True,
    file_output: bool = True,
) -> Path | None:
    log_level = parse_log_level(level_name)
    handlers: list[logging.Handler] = []

    plain_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(threadName)-12s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    color_formatter = _ColorFormatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    resolved_log_path: Path | None = None
    if file_output:
        resolved_log_path = Path(log_file) if log_file is not None else build_log_path(session_name)
        file_handler = logging.FileHandler(resolved_log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(plain_formatter)
        handlers.append(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(color_formatter)
        handlers.append(console_handler)

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for noisy_name in _NOISY_LOGGERS:
        logging.getLogger(noisy_name).setLevel(logging.WARNING)

    logging.info("--- New %s session started ---", session_name)
    if resolved_log_path is not None:
        logging.info(
            "Logging configured: level=%s, file=%s",
            logging.getLevelName(log_level),
            resolved_log_path,
        )
    else:
        logging.info("Logging configured: level=%s", logging.getLevelName(log_level))

    return resolved_log_path


def set_project_log_level(level_name: str | None) -> int:
    log_level = parse_log_level(level_name)
    logging.getLogger().setLevel(log_level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level)
    return log_level


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)
