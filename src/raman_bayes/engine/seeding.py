"""Per-fit random streams (DESIGN.md §6).

Each fit's seed depends only on the run seed, the window's position in the recipe and the
spectrum id, never on which task or sub-batch the spectrum lands in or how often it is retried,
so results do not depend on chunking.
"""

from collections.abc import Iterable

import numpy as np

__all__ = ["spectrum_seed", "spectrum_seeds"]


def spectrum_seed(run_seed: int, window_index: int, spectrum_id: int) -> np.random.SeedSequence:
    """Return ``SeedSequence(run_seed, spawn_key=(window_index, spectrum_id))``."""
    return np.random.SeedSequence(run_seed, spawn_key=(window_index, spectrum_id))


def spectrum_seeds(
    run_seed: int, window_index: int, spectrum_ids: Iterable[int]
) -> list[np.random.SeedSequence]:
    """Return ``spectrum_seed`` for each id, in the order given."""
    return [spectrum_seed(run_seed, window_index, int(s)) for s in spectrum_ids]
