"""Splitting a dataset's spectra into tasks (DESIGN.md §7.1)."""

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from raman_bayes.domain.errors import TooManyTasksError
from raman_bayes.domain.run import TaskSpec
from raman_bayes.orchestration.chunking import CHUNK_SPECTRA, plan_tasks

sizes = st.integers(min_value=1, max_value=3_000_000)


def _assert_partition(tasks: tuple[TaskSpec, ...], n: int, s: int) -> None:
    assert [t.task_index for t in tasks] == list(range(len(tasks)))
    assert tasks[0].spec_start == 0
    assert tasks[-1].spec_stop == n
    for k, task in enumerate(tasks):
        assert task.spec_start == k * s
        assert task.spec_stop == min((k + 1) * s, n)
    assert sum(t.spec_stop - t.spec_start for t in tasks) == n


@given(n=st.integers(min_value=1, max_value=200_000), s=st.integers(min_value=1, max_value=200_000))
def test_explicit_size_partitions_exactly(n: int, s: int) -> None:
    assume(math.ceil(n / s) <= 5000)  # keeps each example fast; the rule does not depend on it
    tasks, size = plan_tasks(n, explicit=s, workers=8, max_tasks=None)
    assert size == s  # used as given, never rounded
    assert len(tasks) == math.ceil(n / s)
    _assert_partition(tasks, n, s)


@given(n=sizes, workers=st.integers(min_value=1, max_value=128))
def test_derived_size_boundaries_multiples_of_1024(n: int, workers: int) -> None:
    tasks, size = plan_tasks(n, explicit=None, workers=workers, max_tasks=None)
    assert CHUNK_SPECTRA == 1024
    assert size % 1024 == 0
    assert size >= math.ceil(n / workers)
    assert size - math.ceil(n / workers) < 1024  # the smallest such multiple
    assert all(t.spec_start % 1024 == 0 for t in tasks)
    assert all(t.spec_stop % 1024 == 0 for t in tasks[:-1])
    _assert_partition(tasks, n, size)


@given(
    n=sizes,
    workers=st.integers(min_value=1, max_value=512),
    cap=st.integers(min_value=1, max_value=1000),
)
def test_derived_size_raised_to_respect_cap(n: int, workers: int, cap: int) -> None:
    tasks, size = plan_tasks(n, explicit=None, workers=workers, max_tasks=cap)
    assert len(tasks) <= cap
    assert size % 1024 == 0
    _assert_partition(tasks, n, size)
    uncapped, _ = plan_tasks(n, explicit=None, workers=workers, max_tasks=None)
    if len(uncapped) <= cap:
        assert tasks == uncapped  # the cap only ever raises the size


def test_explicit_size_over_cap_refused() -> None:
    with pytest.raises(TooManyTasksError) as exc:
        plan_tasks(10_000, explicit=8, workers=1, max_tasks=500)
    message = str(exc.value)
    assert "1250 tasks" in message
    assert "500" in message
    tasks, _ = plan_tasks(10_000, explicit=20, workers=1, max_tasks=500)
    assert len(tasks) == 500


def test_last_task_may_be_short() -> None:
    tasks, _ = plan_tasks(25, explicit=8, workers=1, max_tasks=None)
    assert [(t.spec_start, t.spec_stop) for t in tasks] == [(0, 8), (8, 16), (16, 24), (24, 25)]


def test_n_below_size_gives_one_task() -> None:
    assert plan_tasks(25, explicit=1024, workers=1, max_tasks=None)[0] == (
        TaskSpec(task_index=0, spec_start=0, spec_stop=25),
    )
    tasks, size = plan_tasks(25, explicit=None, workers=8, max_tasks=None)
    assert size == 1024
    assert tasks == (TaskSpec(task_index=0, spec_start=0, spec_stop=25),)
