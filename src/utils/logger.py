"""Rich-based logger with consistent format across modules."""
from __future__ import annotations

import logging
from rich.logging import RichHandler

_CONFIGURED = False


def get_logger(name: str = "asr", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)
