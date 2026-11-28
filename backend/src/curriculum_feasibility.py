"""Read timed curriculum cases and report their Rust feasibility assessments."""

import json
import math
from pathlib import Path
from typing import TypedDict

_CASE_FIELDS = {
    "id",
    "start",
    "target",
    "metric",
    "intrinsicTolerance",
    "startSample",
    "deadlineSample",
    "kineticActionBudget",
    "defaultRasterWindow",
}


class CurriculumCaseReport(TypedDict):
    id: str
    feasibility: dict[str, object]


class CurriculumFeasibilityReport(TypedDict):
    cases: list[CurriculumCaseReport]
    eligibleCaseIds: list[str]


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _point(value: object, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must contain two coordinates")
    return (_number(value[0], field), _number(value[1], field))


def _sample_index(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer sample index")
    return value


def check_curriculum_feasibility(manifest: Path) -> CurriculumFeasibilityReport:
    """Keep all candidates except those excluded by an applicable Rust bound."""
    from runtime_core import evaluate_curriculum_feasibility, get_builtin_distance_field_py

    entries: object = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("expected an array of timed curriculum cases")

    cases: list[CurriculumCaseReport] = []
    eligible_case_ids: list[str] = []
    seen_ids: set[str] = set()
    raster_loaded = False
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"case {index} must be an object")
        case_id = entry.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"case {index} requires a nonempty id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id {case_id!r}")
        seen_ids.add(case_id)

        try:
            unknown = entry.keys() - _CASE_FIELDS
            if unknown:
                raise ValueError(f"unexpected fields: {', '.join(sorted(unknown))}")
            metric = entry.get("metric", "runtimeMap")
            if not isinstance(metric, str):
                raise ValueError("metric must be a string")
            budget = entry.get("kineticActionBudget")
            window_raw = entry.get("defaultRasterWindow")
            window = None
            if window_raw is not None:
                if metric != "runtimeMap":
                    raise ValueError("defaultRasterWindow requires runtimeMap")
                if not isinstance(window_raw, list) or len(window_raw) != 2:
                    raise ValueError("defaultRasterWindow must contain minimum and maximum corners")
                window = (_point(window_raw[0], "defaultRasterWindow"), _point(window_raw[1], "defaultRasterWindow"))
                if not all(lo < hi for lo, hi in zip(*window, strict=True)):
                    raise ValueError("defaultRasterWindow minimum must precede maximum on both axes")
                if not raster_loaded:
                    get_builtin_distance_field_py("mandelbrot_default")
                    raster_loaded = True
            report_json = evaluate_curriculum_feasibility(
                _point(entry.get("start"), "start"),
                _point(entry.get("target"), "target"),
                metric,
                _number(entry.get("intrinsicTolerance", 0), "intrinsicTolerance"),
                _sample_index(entry.get("startSample"), "startSample"),
                _sample_index(entry.get("deadlineSample"), "deadlineSample"),
                None if budget is None else _number(budget, "kineticActionBudget"),
                window,
            )
            report: object = json.loads(report_json)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"case {case_id!r}: {error}") from error

        if not isinstance(report, dict):
            raise ValueError(f"case {case_id!r}: invalid Rust feasibility report")
        assessment = report.get("assessment")
        if not isinstance(assessment, dict) or assessment.get("status") not in (
            "ruledOut",
            "notRuledOut",
            "insufficientEvidence",
        ):
            raise ValueError(f"case {case_id!r}: invalid Rust feasibility status")
        cases.append(CurriculumCaseReport(id=case_id, feasibility=report))
        if assessment["status"] != "ruledOut":
            eligible_case_ids.append(case_id)

    return CurriculumFeasibilityReport(cases=cases, eligibleCaseIds=eligible_case_ids)
