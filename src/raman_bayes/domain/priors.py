"""Priors recipe models (DESIGN.md §5). Key names follow ``priors.yaml`` exactly.

Structure and the named hyperparameter rules; the §5 validation rules are in
``priors_rules``. Unknown keys (including the reserved ``derived:`` section) are rejected by
``extra="forbid"``, and named choices are ``Literal``s holding the v1 vocabulary.
"""

from collections.abc import Callable
from typing import Literal

from raman_bayes.domain._base import DomainModel

__all__ = [
    "SD_RULES",
    "BaselineSpec",
    "Bounds",
    "CenterDefault",
    "CenterPrior",
    "Defaults",
    "DespikeSpec",
    "EndMember",
    "FwhmDefault",
    "FwhmPrior",
    "LikelihoodSpec",
    "Peak",
    "PreprocessingSpec",
    "PresenceDefault",
    "PriorsSpec",
    "SdRule",
    "SmoothingSpec",
    "SpecMeta",
    "Window",
    "center_sd",
    "quarter_range",
]

Bounds = tuple[float, float]

SdRule = Literal["quarter_range"]


def quarter_range(bounds: Bounds) -> float:
    """Return ``(upper - lower) / 4``: a quarter of the range the bounds span."""
    lower, upper = bounds
    return (upper - lower) / 4


# Every named rule a priors file may cite; the keys are exactly the SdRule vocabulary.
SD_RULES: dict[SdRule, Callable[[Bounds], float]] = {"quarter_range": quarter_range}


class SpecMeta(DomainModel):
    """The ``spec:`` block: identity and approval status of the recipe."""

    name: str
    version: str
    status: Literal["draft", "approved"]
    source: str | None = None


class PresenceDefault(DomainModel):
    """Default presence prior: each peak is present with probability ``p``."""

    prior: Literal["bernoulli"]
    p: float


class CenterDefault(DomainModel):
    """Default center prior family and the named rule that sets its sd."""

    prior: Literal["normal_truncated"]
    sd_rule: SdRule


class FwhmDefault(DomainModel):
    """FWHM prior family; it spans each peak's own ``fwhm.bounds``."""

    prior: Literal["uniform"]


class LikelihoodSpec(DomainModel):
    """Observation model; ``nu`` is the Student-t degrees of freedom."""

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
    """Median/MAD despiking: ``kernel`` channels wide, spikes above ``thresh`` MADs."""

    method: Literal["median_mad"]
    kernel: int
    thresh: float


class BaselineSpec(DomainModel):
    """Asymmetric least-squares baseline with smoothness ``lam`` and asymmetry ``p``."""

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
    """A named group of peaks attributed to one material."""

    id: str
    label: str | None = None
    peaks: tuple[str, ...]


class CenterPrior(DomainModel):
    """A peak's center prior in cm^-1; ``sd`` overrides the default ``sd_rule``."""

    mean: float
    bounds: Bounds
    sd: float | None = None


class FwhmPrior(DomainModel):
    """A peak's FWHM bounds in cm^-1."""

    bounds: Bounds


class Peak(DomainModel):
    """One peak of a window: its id, display label and priors."""

    id: str
    label: str | None = None
    center: CenterPrior
    fwhm: FwhmPrior


class Window(DomainModel):
    """A spectral range fitted independently, with the peaks it contains."""

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


def center_sd(peak: Peak, rule: SdRule) -> float:
    """Return the sd of ``peak``'s center prior: its ``sd`` if set, else ``rule`` of its bounds."""
    if peak.center.sd is not None:
        return peak.center.sd
    return SD_RULES[rule](peak.center.bounds)
