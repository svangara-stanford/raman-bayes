"""Split a dataset's spectra into tasks (DESIGN.md §7.1).

An explicit task size is used as given. A derived one is rounded up to whole Zarr chunks of
1024 spectra, so every task boundary but the last falls on a chunk boundary and no two tasks
read the same chunk. A cap on the number of tasks (Slurm's array limit) raises a derived size
and refuses an explicit one.
"""

import math
from typing import NamedTuple

from raman_bayes.domain.errors import TooManyTasksError
from raman_bayes.domain.run import TaskSpec, task_ranges

__all__ = ["CHUNK_SPECTRA", "Chunking", "plan_tasks"]

CHUNK_SPECTRA = 1024  # rows per intensity chunk in dataset.zarr (§4)


class Chunking(NamedTuple):
    """The tasks of a plan and the task size that produced them."""

    tasks: tuple[TaskSpec, ...]
    spectra_per_task: int


def plan_tasks(
    n_spectra: int, *, explicit: int | None, workers: int, max_tasks: int | None
) -> Chunking:
    """Return the tasks for ``n_spectra`` spectra.

    With ``explicit`` set, tasks hold that many spectra. Otherwise the size is
    ``ceil(n_spectra / workers)`` rounded up to a multiple of ``CHUNK_SPECTRA``, raised until
    there are at most ``max_tasks`` tasks. Raises TooManyTasksError when an explicit size gives
    more than ``max_tasks`` tasks; ``max_tasks`` None means no cap.
    """
    if explicit is not None:
        n_tasks = math.ceil(n_spectra / explicit)
        if max_tasks is not None and n_tasks > max_tasks:
            raise TooManyTasksError(n_tasks, explicit, max_tasks)
        return Chunking(task_ranges(n_spectra, explicit), explicit)
    size = _whole_chunks(math.ceil(n_spectra / workers))
    if max_tasks is not None:
        size = max(size, _whole_chunks(math.ceil(n_spectra / max_tasks)))
    return Chunking(task_ranges(n_spectra, size), size)


def _whole_chunks(spectra: int) -> int:
    return math.ceil(spectra / CHUNK_SPECTRA) * CHUNK_SPECTRA
