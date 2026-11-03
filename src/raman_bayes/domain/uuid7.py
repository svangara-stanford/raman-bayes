"""UUIDv7 run ids (RFC 9562, DESIGN.md §7.1), until the minimum Python has ``uuid.uuid7`` (3.14).

Layout: 48-bit Unix time in milliseconds, version 7, then 74 bits used as a counter (RFC 9562
§6.2, method 2), split around the variant bits. A new millisecond starts the counter at a random
73-bit value; a call within the same or an earlier millisecond adds one. Ids from one process
are therefore strictly increasing even when the clock stands still or steps back.
"""

import os
import threading
import time
import uuid

__all__ = ["is_uuid7", "uuid7"]

_VERSION = 7
_COUNTER_BITS = 74  # rand_a (12) and rand_b (62)
_RAND_B_BITS = 62
_START_BITS = _COUNTER_BITS - 1  # a fresh counter leaves headroom to count up


class _Monotonic:
    """The last (millisecond, counter) handed out, behind a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ms = -1
        self._counter = 0

    def next(self) -> uuid.UUID:
        with self._lock:
            now_ms = time.time_ns() // 1_000_000
            if now_ms > self._ms:
                self._ms = now_ms
                self._counter = int.from_bytes(os.urandom(10), "big") >> (80 - _START_BITS)
            else:
                self._counter += 1
                if self._counter >> _COUNTER_BITS:
                    self._ms += 1
                    self._counter = 0
            ms, counter = self._ms, self._counter
        rand_a = counter >> _RAND_B_BITS
        rand_b = counter & ((1 << _RAND_B_BITS) - 1)
        return uuid.UUID(int=(ms << 80) | (_VERSION << 76) | (rand_a << 64) | (0b10 << 62) | rand_b)


_GENERATOR = _Monotonic()


def uuid7() -> uuid.UUID:
    """Return a new UUIDv7, greater than every one this process returned before."""
    return _GENERATOR.next()


def is_uuid7(text: str) -> bool:
    """Return whether ``text`` is a UUIDv7 in canonical form: lowercase, hyphenated, no braces."""
    try:
        value = uuid.UUID(text)
    except ValueError:
        return False
    return value.version == _VERSION and value.variant == uuid.RFC_4122 and str(value) == text
