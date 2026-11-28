"""Exercise the training CLI on timed point targets and missing evidence."""

import json
import subprocess
import sys
from pathlib import Path

from runtime_core import SAMPLE_RATE

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "train.py"


def _check(
    tmp_path: Path, cases: list[dict[str, object]]
) -> subprocess.CompletedProcess[str]:
    manifest = tmp_path / "curriculum.json"
    manifest.write_text(json.dumps(cases), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--check-curriculum-feasibility",
            str(manifest),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
    )


def test_check_curriculum_feasibility(tmp_path: Path) -> None:
    base: dict[str, object] = {
        "start": [0, 0],
        "target": [2.125, 0],
        "startSample": SAMPLE_RATE,
        "deadlineSample": 2 * SAMPLE_RATE,
        "kineticActionBudget": 2.0,
    }
    ideal = {**base, "metric": "idealDistanceGraph"}
    result = _check(
        tmp_path,
        [
            {**ideal, "id": "short-ideal"},
            {**ideal, "id": "long-ideal", "deadlineSample": 101 * SAMPLE_RATE},
            {**ideal, "id": "tolerant-ideal", "intrinsicTolerance": 100},
            {**base, "id": "runtime-map"},
        ],
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["eligibleCaseIds"] == ["long-ideal", "tolerant-ideal", "runtime-map"]
    cases = {entry["id"]: entry["feasibility"] for entry in report["cases"]}

    short = cases["short-ideal"]
    assert short["referenceStatus"] == "idealDistanceBounds"
    assert short["sampleRate"] == SAMPLE_RATE
    assert short["startSample"] == SAMPLE_RATE
    assert short["deadlineSample"] == 2 * SAMPLE_RATE
    assert short["proof"] is not None
    assert short["assessment"]["durationSamples"] == SAMPLE_RATE
    assert short["assessment"]["status"] == "ruledOut"
    assert short["assessment"]["kineticActionStatus"] == "ruledOut"
    assert short["assessment"]["minimumKineticAction"] > 2.0
    assert short["assessment"]["minimumSeconds"] is None
    assert short["assessment"]["deadlineStatus"] == "insufficientEvidence"

    for case_id in ("long-ideal", "tolerant-ideal"):
        assessed = cases[case_id]["assessment"]
        assert assessed["status"] == "insufficientEvidence"
        assert assessed["kineticActionStatus"] == "notRuledOut"
    assert cases["tolerant-ideal"]["assessment"]["minimumKineticAction"] == 0

    actual = cases["runtime-map"]
    assert actual["metric"] == "runtimeMap"
    assert actual["referenceStatus"] == "runtimeMetricCorrespondenceUncertified"
    assert actual["runtimeGeometryVersion"]
    assert actual.get("proof") is None
    assert actual["assessment"]["status"] == "insufficientEvidence"
    assert actual["assessment"]["minimumKineticAction"] is None
    assert not (tmp_path / "logs").exists()


def test_check_rejects_manifest_speed_cap(tmp_path: Path) -> None:
    result = _check(tmp_path, [{"id": "uncertified-cap", "speedCap": 1}])
    assert result.returncode == 2
    assert result.stdout == ""
    assert "case 'uncertified-cap': unexpected fields: speedCap" in result.stderr


def test_default_raster_diagnostic_does_not_filter_curriculum(tmp_path: Path) -> None:
    result = _check(
        tmp_path,
        [{
            "id": "raster-window",
            "start": [-1.0, 0.0],
            "target": [-0.7, 0.1],
            "startSample": 0,
            "deadlineSample": SAMPLE_RATE,
            "kineticActionBudget": 0.02,
            "defaultRasterWindow": [[-1.5, -0.45], [-0.4, 0.45]],
        }],
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["eligibleCaseIds"] == ["raster-window"]
    task = report["cases"][0]["feasibility"]
    assert task["assessment"]["status"] == "insufficientEvidence"
    diagnostic = task["defaultRasterDiagnostic"]
    assert diagnostic["rasterGeneration"] > 0
    continuous = diagnostic["continuousAssessment"]
    assert continuous["kineticActionStatus"] == "ruledOut"
    assert continuous["deadlineStatus"] == "insufficientEvidence"
