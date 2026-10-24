"""Logging setup (DESIGN.md §2 ``logging.py``): console on stderr plus an optional per-task file."""

import io
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from raman_bayes.logging import LOGGER_NAME, setup_logging


@pytest.fixture(autouse=True)
def reset_logger() -> Iterator[None]:
    logger = logging.getLogger(LOGGER_NAME)
    saved = (logger.level, logger.propagate, list(logger.handlers))
    yield
    for handler in list(logger.handlers):
        if handler not in saved[2]:
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(saved[0])
    logger.propagate = saved[1]


def test_logger_name_is_package() -> None:
    assert LOGGER_NAME == "raman_bayes"


def test_output_goes_to_stderr_not_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    logger = setup_logging("INFO")
    logger.info("hello stderr")
    out, err = capsys.readouterr()
    assert out == ""
    assert "hello stderr" in err


def test_stderr_is_looked_up_at_write_time(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = setup_logging("INFO")
    swapped = io.StringIO()
    monkeypatch.setattr("sys.stderr", swapped)
    logger.info("after swap")
    assert "after swap" in swapped.getvalue()


def test_level_and_propagation() -> None:
    logger = setup_logging("ERROR")
    assert logger.name == LOGGER_NAME
    assert logger.level == logging.ERROR
    assert logger.propagate is False
    assert setup_logging().level == logging.WARNING


def test_child_loggers_use_package_handlers(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    logging.getLogger("raman_bayes.storage.shards").info("from a child")
    assert "from a child" in capsys.readouterr().err


def test_file_handler_writes(tmp_path: Path) -> None:
    log_file = tmp_path / "task-0.log"
    logger = setup_logging("INFO", log_file=log_file)
    logger.info("to the file")
    assert "to the file" in log_file.read_text(encoding="utf-8")


def test_second_call_does_not_duplicate_handlers(tmp_path: Path) -> None:
    setup_logging()
    logger = setup_logging()
    assert len(logger.handlers) == 1
    setup_logging(log_file=tmp_path / "a.log")
    logger = setup_logging(log_file=tmp_path / "a.log")
    assert len(logger.handlers) == 2


def test_switching_log_files_closes_the_old_one(tmp_path: Path) -> None:
    logger = setup_logging("INFO", log_file=tmp_path / "a.log")
    (old,) = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    logger = setup_logging("INFO", log_file=tmp_path / "b.log")
    assert old not in logger.handlers
    assert old.stream is None  # FileHandler.close() drops the stream
    logger.info("to b")
    assert "to b" not in (tmp_path / "a.log").read_text(encoding="utf-8")
    assert "to b" in (tmp_path / "b.log").read_text(encoding="utf-8")


def test_foreign_handlers_are_left_alone() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    foreign = logging.NullHandler()
    logger.addHandler(foreign)
    setup_logging()
    setup_logging()
    assert foreign in logger.handlers
