"""stdlib logging setup for the ``raman_bayes`` logger: stderr console plus an optional file.

The file handler is what ``run-task`` uses for ``runs/<id>/logs/task-N.log`` (§7.2).
``setup_logging`` is idempotent: it replaces and closes only the handlers it added itself.
"""

import logging
import sys
from pathlib import Path
from typing import TextIO

LOGGER_NAME = "raman_bayes"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _StderrHandler(logging.StreamHandler[TextIO]):
    """Writes to whatever ``sys.stderr`` is at emit time, so a swapped stderr still works."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


class _FileHandler(logging.FileHandler):
    pass


def setup_logging(level: int | str = "WARNING", *, log_file: Path | None = None) -> logging.Logger:
    """Configure the ``raman_bayes`` logger (no propagation to root) and return it."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if isinstance(handler, _StderrHandler | _FileHandler):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_FORMAT)
    handlers: list[logging.Handler] = [_StderrHandler()]
    if log_file is not None:
        handlers.append(_FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger
