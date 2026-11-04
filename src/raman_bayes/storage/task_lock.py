"""The ``_LOCK`` file that keeps two run-task processes off one task (DESIGN.md §7.2).

The lock is created with ``O_EXCL`` and names its holder (host, pid, started_at). While the
holder works, a heartbeat refreshes the file's modification time, so a lock's age is the time
since its holder last showed signs of life, not since it started: nothing assumes how long a
task takes. A lock is stale when that age reaches the stale limit, or when it was taken on this
host by a process that no longer exists; a stale lock is taken over. A holder that stops
before finishing leaves its lock behind, which is how a stopped task is told from one that
never started. Taking over is not exclusive: two processes that find the same stale lock at
once can both proceed, and since a task's parts depend only on the plan they write the same
files.
"""

import os
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from raman_bayes.domain.errors import TaskLockedError
from raman_bayes.domain.markers import LockHolder
from raman_bayes.storage.atomic import write_text_atomically

__all__ = ["LOCK_FILE", "LockState", "acquire_lock", "heartbeat", "read_lock", "release_lock"]

LOCK_FILE = "_LOCK"


@dataclass(frozen=True)
class LockState:
    """A lock file as found: its holder, its age, and whether it still holds the task.

    ``holder`` is None when the file cannot be read as a holder, which is what a lock looks
    like between its creation and its first write; its age alone then decides.
    """

    holder: LockHolder | None
    age: timedelta
    fresh: bool


def read_lock(
    task_dir: Path, stale_after: timedelta, *, now: datetime | None = None
) -> LockState | None:
    """Return the state of ``task_dir/_LOCK`` at ``now`` (default: the current time), or None."""
    path = task_dir / LOCK_FILE
    try:
        modified = path.stat().st_mtime
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    holder = _holder(text)
    age = (now or datetime.now(UTC)) - datetime.fromtimestamp(modified, UTC)
    return LockState(holder, age, age < stale_after and not _gone(holder))


def acquire_lock(
    task_dir: Path, stale_after: timedelta, *, now: datetime | None = None
) -> LockHolder:
    """Take ``task_dir/_LOCK`` for this process and return what it now says.

    ``now`` (default: the current time) is the holder's ``started_at``; whether a lock found
    there is stale is judged by the clock. A stale lock is overwritten. Raises
    TaskLockedError, naming the holder, when the lock is fresh.
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    started = now or datetime.now(UTC)
    mine = LockHolder(host=socket.gethostname(), pid=os.getpid(), started_at=started)
    path = task_dir / LOCK_FILE
    while True:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            state = read_lock(task_dir, stale_after)
            if state is None:
                continue  # released while we looked; try to create it again
            if state.fresh:
                raise TaskLockedError(task_dir, state.holder) from None
            write_text_atomically(path, mine.model_dump_json())
            return mine
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(mine.model_dump_json())
            stream.flush()
            os.fsync(stream.fileno())
        return mine


def release_lock(task_dir: Path) -> None:
    """Remove ``task_dir/_LOCK`` if it is there."""
    (task_dir / LOCK_FILE).unlink(missing_ok=True)


@contextmanager
def heartbeat(task_dir: Path, interval: timedelta) -> Iterator[threading.Thread]:
    """Refresh ``task_dir/_LOCK``'s modification time every ``interval`` while the block runs.

    Yields the heartbeat thread, which ends early if the lock disappears.
    """
    stop = threading.Event()
    path = task_dir / LOCK_FILE

    def beat() -> None:
        while not stop.wait(interval.total_seconds()):
            try:
                os.utime(path)
            except FileNotFoundError:
                return  # released or removed: there is nothing left to keep fresh

    thread = threading.Thread(target=beat, name=f"heartbeat {task_dir.name}", daemon=True)
    thread.start()
    try:
        yield thread
    finally:
        stop.set()
        thread.join()


def _holder(text: str) -> LockHolder | None:
    try:
        return LockHolder.model_validate_json(text)
    except ValidationError:
        return None


def _gone(holder: LockHolder | None) -> bool:
    # Only a process on this host can be checked; elsewhere the heartbeat's age must decide.
    if holder is None or holder.host != socket.gethostname():
        return False
    try:
        os.kill(holder.pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False  # it exists, under another user
    return False
