from __future__ import annotations

import logging
import os
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def configure_logging(
    name: str = "fxalgo",
    level: int | str | None = None,
    log_file: Path | None = None,
) -> logging.Logger:
    resolved_level = level if level is not None else LOG_LEVEL

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=resolved_level,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(name)
