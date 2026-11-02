"""Convert one priors window between a recipe and the legacy CSV format (DESIGN.md §5).

Import checks the new window with the single-window rules, so a wrong ``--range`` is reported
the same way ``validate-priors`` would report it. Export validates the whole recipe first.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from raman_bayes.app.validate import validate_priors_file
from raman_bayes.domain.errors import LegacyCsvError
from raman_bayes.domain.priors import Bounds, Window
from raman_bayes.domain.priors_rules import check_window
from raman_bayes.io.legacy_csv import format_legacy_csv, read_legacy_csv

__all__ = ["CsvExport", "CsvImport", "export_window", "import_window"]


@dataclass(frozen=True)
class CsvImport:
    """Outcome of importing a CSV. ``window`` and ``fragment`` are set exactly when no errors.

    ``fragment`` is YAML for one item of a recipe's ``windows:`` list, defaults left out.
    """

    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    window: Window | None = None
    fragment: str | None = None


@dataclass(frozen=True)
class CsvExport:
    """Outcome of exporting a window. ``csv`` is set exactly when there are no errors."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    csv: str | None = None


def import_window(path: Path, window_id: str, range_cm1: Bounds) -> CsvImport:
    """Read the legacy CSV at ``path`` as window ``window_id`` spanning ``range_cm1``."""
    try:
        window = read_legacy_csv(path, window_id, range_cm1)
    except OSError as exc:
        return CsvImport(errors=(f"{path}: cannot read file ({exc.strerror})",))
    except (LegacyCsvError, UnicodeDecodeError) as exc:
        return CsvImport(errors=(str(exc),))
    report = check_window(window)
    if report.errors:
        return CsvImport(errors=report.errors, warnings=report.warnings)
    item = window.model_dump(mode="json", exclude_defaults=True)
    fragment = yaml.safe_dump([item], sort_keys=False, allow_unicode=True, default_flow_style=None)
    return CsvImport(errors=(), warnings=report.warnings, window=window, fragment=fragment)


def export_window(priors_path: Path, window_id: str) -> CsvExport:
    """Return window ``window_id`` of the recipe at ``priors_path`` as legacy CSV text."""
    validation = validate_priors_file(priors_path)
    if validation.spec is None:
        return CsvExport(errors=validation.errors, warnings=validation.warnings)
    windows = {w.id: w for w in validation.spec.windows}
    if window_id not in windows:
        known = ", ".join(windows)
        return CsvExport(errors=(f"no window {window_id!r} in {priors_path}; it has: {known}",))
    return CsvExport(
        errors=(), warnings=validation.warnings, csv=format_legacy_csv(windows[window_id])
    )
