"""User-facing errors (``domain/errors.py``) survive pickling, as spawn workers need."""

import pickle
from pathlib import Path

import pytest

from raman_bayes.domain import errors


@pytest.mark.parametrize(
    "error",
    [
        errors.SpectrumLineError(Path("a.txt"), 12, "x y z"),
        errors.DatasetStoreError(Path("d.zarr"), "no dataset store here"),
        errors.EngineContractError("mock", "0.1.0", "fit_many returned 3 results for 4 spectra"),
        errors.TaskLockedError(Path("shards/task=00000"), None),
    ],
    ids=lambda e: type(e).__name__,
)
def test_errors_survive_pickling(error: errors.RamanBayesError) -> None:
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is type(error)
    assert str(copy) == str(error)
    assert copy.__dict__ == error.__dict__
