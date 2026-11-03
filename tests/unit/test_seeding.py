"""Per-fit seeds (DESIGN.md §6): ``SeedSequence(run_seed, spawn_key=(window, spectrum_id))``."""

import itertools

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from raman_bayes.engine.seeding import spectrum_seed, spectrum_seeds

RUN_SEED = 20261102


def _states(seeds: list[np.random.SeedSequence]) -> list[tuple[int, ...]]:
    return [tuple(int(v) for v in s.generate_state(4)) for s in seeds]


def test_spawn_key_is_window_index_and_spectrum_id() -> None:
    seed = spectrum_seed(RUN_SEED, 1, 42)
    assert seed.entropy == RUN_SEED
    assert seed.spawn_key == (1, 42)
    reference = np.random.SeedSequence(RUN_SEED, spawn_key=(1, 42))
    assert _states([seed]) == _states([reference])


@given(
    n=st.integers(min_value=1, max_value=400),
    cuts=st.lists(st.integers(min_value=1, max_value=399), max_size=6),
    window_index=st.integers(min_value=0, max_value=3),
    run_seed=st.integers(min_value=0, max_value=2**63 - 1),
)
def test_seeds_independent_of_task_boundaries(
    n: int, cuts: list[int], window_index: int, run_seed: int
) -> None:
    bounds = [0, *sorted({c for c in cuts if c < n}), n]
    per_task = [
        seed
        for start, stop in itertools.pairwise(bounds)
        for seed in spectrum_seeds(run_seed, window_index, range(start, stop))
    ]
    whole = spectrum_seeds(run_seed, window_index, range(n))
    assert _states(per_task) == _states(whole)


def test_seeds_independent_of_sub_batching_and_retry() -> None:
    ids = range(1300)
    whole = _states(spectrum_seeds(RUN_SEED, 0, ids))
    batches = [ids[i : i + 512] for i in range(0, len(ids), 512)]
    batched = [s for batch in batches for s in spectrum_seeds(RUN_SEED, 0, batch)]
    assert _states(batched) == whole
    retried = spectrum_seeds(RUN_SEED, 0, batches[1])  # the sub-batch run again after a crash
    assert _states(retried) == whole[512:1024]


def test_distinct_spectra_get_distinct_streams() -> None:
    seeds = [spectrum_seed(RUN_SEED, w, s) for w in range(3) for s in range(200)]
    assert len(set(_states(seeds))) == 600
    first_draws = {float(np.random.default_rng(s).random()) for s in seeds}
    assert len(first_draws) == 600
    other_run = spectrum_seed(RUN_SEED + 1, 0, 0)
    assert _states([other_run]) != _states([spectrum_seed(RUN_SEED, 0, 0)])
