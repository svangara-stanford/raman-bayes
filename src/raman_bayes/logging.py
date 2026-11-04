"""stdlib logging setup for the ``raman_bayes`` logger: stderr console plus optional files.

``setup_logging`` is idempotent: it replaces and closes only the handlers it added itself.
``task_log`` adds ``runs/<id>/logs/task-N.log`` (§7.2) for the duration of one task, at INFO,
without making the console any more verbose than it was.
"""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

__all__ = ["LOGGER_NAME", "setup_logging", "task_log"]

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


class _TaskFileHandler(logging.FileHandler):
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


@contextmanager
def task_log(path: Path) -> Iterator[None]:
    """Also write the package's INFO and higher records to ``path`` while the block runs.

    ``path``'s folder is created if needed and the file is appended to. Handlers already
    attached keep receiving only what the logger let through before.
    """
    logger = logging.getLogger(LOGGER_NAME)
    before = logger.level
    effective = logger.getEffectiveLevel()
    unset = [h for h in logger.handlers if h.level == logging.NOTSET]
    for handler in unset:
        handler.setLevel(effective)
    path.parent.mkdir(parents=True, exist_ok=True)
    task_file = _TaskFileHandler(path, encoding="utf-8")
    task_file.setFormatter(logging.Formatter(_FORMAT))
    task_file.setLevel(logging.INFO)
    logger.addHandler(task_file)
    logger.setLevel(min(effective, logging.INFO))
    try:
        yield
    finally:
        logger.removeHandler(task_file)
        task_file.close()
        logger.setLevel(before)
        for handler in unset:
            handler.setLevel(logging.NOTSET)
