"""Shared base for domain models: immutable and strict about unknown keys."""

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Base for every Pydantic model in ``raman_bayes.domain``: frozen, ``extra="forbid"``."""

    model_config = ConfigDict(frozen=True, extra="forbid")
