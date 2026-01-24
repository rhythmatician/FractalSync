from pathlib import Path
import json
import subprocess


def test_tune_contour_quick(tmp_path):
    out = tmp_path / "tune.json"
    # Run the script with a tiny sweep to verify output format
    import sys

    cmd = [
        sys.executable,
        "backend/scripts/tune_contour.py",
        "--d-star",
        "0.3",
        "--max-step",
        "0.01",
        "--trials",
        "1",
        "--steps",
        "10",
        "--out",
        str(out),
    ]
    subprocess.check_call(cmd)
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert data[0]["d_star"] == 0.3
    assert data[0]["max_step"] == 0.01
    assert "mean_deltaV" in data[0]
