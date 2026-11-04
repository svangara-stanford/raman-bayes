"""Replace a file atomically: write ``<name>.tmp`` beside it, fsync it, then ``os.replace``.

A reader sees the old file or the complete new one, never a partial write. A crash leaves at
most the ``.tmp`` file, which nothing reads and the next write of that file overwrites.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

__all__ = ["STAGING_SUFFIX", "staging_path", "write_atomically", "write_text_atomically"]

STAGING_SUFFIX = ".tmp"


def staging_path(path: Path) -> Path:
    """Return the temporary name ``path`` is written under before it is renamed into place."""
    return path.with_name(path.name + STAGING_SUFFIX)


def write_atomically(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Call ``write`` on the staging file, fsync it, then rename it to ``path``.

    Errors from ``write`` propagate and leave ``path`` as it was.
    """
    staged = staging_path(path)
    with staged.open("wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged, path)


def write_text_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` as UTF-8, atomically."""
    write_atomically(path, lambda stream: stream.write(text.encode("utf-8")))
