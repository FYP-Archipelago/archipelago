from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..models import Explainability, RunInfo
from .shapley import ShapleyAccumulator


# What every explainability analysis has collected so far, one field per analysis.
@dataclass
class Analyses:
    # The run id comes from run_start, so it is filled in at finish.
    shapley: ShapleyAccumulator = field(default_factory=lambda: ShapleyAccumulator(""))


# A failing analysis must not fail clustering or the other analyses, so its error becomes an unavailable result.
def _guarded(run_id: str, compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return compute()
    except Exception as error:  # noqa: BLE001
        return {"schema_version": 1, "run_id": run_id, "status": "unavailable", "detail": f"Analysis failed: {error}"}


def add(analyses: Analyses, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        analyses.shapley.add(rows)
    except Exception as error:  # noqa: BLE001
        analyses.shapley.unavailable(f"Analysis failed: {error}")


def finish(analyses: Analyses, run: RunInfo) -> Explainability:
    analyses.shapley.run_id = run.run_id
    return Explainability(shapley=_guarded(run.run_id, lambda: analyses.shapley.finish(run)))
