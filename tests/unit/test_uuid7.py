"""In-package UUIDv7 (RFC 9562) for run ids (DESIGN.md §7.1), until Python 3.14's uuid.uuid7."""

import itertools
import re
import time
import uuid

from raman_bayes.domain.uuid7 import is_uuid7, uuid7

CANONICAL = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")


def test_uuid7_version_and_variant_bits() -> None:
    value = uuid7()
    assert value.version == 7
    assert value.variant == uuid.RFC_4122
    assert (value.int >> 76) & 0xF == 0b0111
    assert (value.int >> 62) & 0b11 == 0b10


def test_uuid7_timestamp_is_unix_ms() -> None:
    before = time.time_ns() // 1_000_000
    value = uuid7()
    after = time.time_ns() // 1_000_000
    assert before <= value.int >> 80 <= after


def test_uuid7_strictly_increasing_over_10000_calls() -> None:
    values = [uuid7() for _ in range(10_000)]
    assert all(a.int < b.int for a, b in itertools.pairwise(values))
    texts = [str(v) for v in values]
    assert texts == sorted(texts)  # the text form sorts the same way


def test_uuid7_unique() -> None:
    assert len({uuid7() for _ in range(10_000)}) == 10_000


def test_uuid7_canonical_lowercase_string() -> None:
    text = str(uuid7())
    assert CANONICAL.fullmatch(text)
    assert is_uuid7(text)


def test_is_uuid7_rejects_other_forms() -> None:
    text = str(uuid7())
    for other in (
        text.upper(),
        text.replace("-", ""),
        f"{{{text}}}",
        str(uuid.uuid4()),
        "0190b3d4-0000-7000-c000-000000000000",  # version 7 but not the RFC variant
        "not a uuid",
    ):
        assert not is_uuid7(other), other
