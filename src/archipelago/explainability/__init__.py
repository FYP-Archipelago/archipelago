import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ..models import Explainability, ExplainabilityAnalysisUnavailable, RunInfo
from .lineage import LineageRow, lineage_row
from .shapley import attribute_improvement


@dataclass
class Analyses:
    shapley: list[LineageRow] = field(default_factory=list)


def add(analyses: Analyses, rows: Sequence[Mapping[str, Any]]) -> None:
    analyses.shapley.extend(lineage_row(row) for row in rows)


def finish(analyses: Analyses, run: RunInfo) -> Explainability:
    return Explainability(shapley=_result("Shapley analysis", lambda: attribute_improvement(analyses.shapley, run)))


# A failing analysis must not fail clustering or the other analyses, so we make it unavailable instead.
def _result[T: BaseModel](name: str, compute: Callable[[], T]) -> T | ExplainabilityAnalysisUnavailable:
    try:
        output = compute()
        json.dumps(output.model_dump(), allow_nan=False)
    except Exception as error:  # noqa: BLE001
        return ExplainabilityAnalysisUnavailable(detail=f"{name} unavailable: {error}")
    return output
