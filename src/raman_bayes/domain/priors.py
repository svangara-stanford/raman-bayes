"""Priors recipe models (DESIGN.md §5). Key names follow ``priors.yaml`` exactly.

Structure only: the §5 validation rules 1-9 arrive in Phase 1. Unknown keys (including the
reserved ``derived:`` section) are rejected by ``extra="forbid"``, and named choices are
``Literal``s holding the v1 vocabulary.
"""

from typing import Literal

from raman_bayes.domain._base import DomainModel

Bounds = tuple[float, float]

# Named hyperparameter rules. ``quarter_range``: sd = (upper - lower) / 4 unless the peak sets sd.
SdRule = Literal["quarter_range"]


class SpecMeta(DomainModel):
    """The ``spec:`` block: identity and approval status of the recipe."""

    name: str
    version: str
    status: Literal["draft", "approved"]
    source: str | None = None


class PresenceDefault(DomainModel):
    prior: Literal["bernoulli"]
    p: float


class CenterDefault(DomainModel):
    prior: Literal["normal_truncated"]
    sd_rule: SdRule


class FwhmDefault(DomainModel):
    """FWHM prior family; it spans each peak's own ``fwhm.bounds``."""

    prior: Literal["uniform"]


class LikelihoodSpec(DomainModel):
    model: Literal["student_t"]
    nu: float


class Defaults(DomainModel):
    """The ``defaults:`` block, applied to every peak."""

    presence: PresenceDefault
    lineshape: Literal["pseudo_voigt"]
    center: CenterDefault
    fwhm: FwhmDefault
    likelihood: LikelihoodSpec


class DespikeSpec(DomainModel):
    method: Literal["median_mad"]
    kernel: int
    thresh: float


class BaselineSpec(DomainModel):
    method: Literal["als"]
    lam: float
    p: float


class SmoothingSpec(DomainModel):
    """Provisional placeholder: §5 shows only ``smoothing: null``; parameters are not specified."""

    method: Literal["savgol"]
    window: int
    polyorder: int


class PreprocessingSpec(DomainModel):
    """The ``preprocessing:`` block, applied per window in the §7.2 order."""

    margin_cm1: float
    despike: DespikeSpec | None
    baseline: BaselineSpec | None
    smoothing: SmoothingSpec | None
    clamp_nonnegative: bool


class EndMember(DomainModel):
    id: str
    label: str | None = None
    peaks: tuple[str, ...]


class CenterPrior(DomainModel):
    mean: float
    bounds: Bounds
    sd: float | None = None


class FwhmPrior(DomainModel):
    bounds: Bounds


class Peak(DomainModel):
    id: str
    label: str | None = None
    center: CenterPrior
    fwhm: FwhmPrior


class Window(DomainModel):
    id: str
    range_cm1: Bounds
    peaks: tuple[Peak, ...]
    allow_overlap: bool = False
    ordered: bool = False


class PriorsSpec(DomainModel):
    """A whole ``priors.yaml`` document."""

    schema_version: str
    spec: SpecMeta
    defaults: Defaults
    preprocessing: PreprocessingSpec
    end_members: tuple[EndMember, ...]
    windows: tuple[Window, ...]
